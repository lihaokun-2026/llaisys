"""
LLAISYS Chat Server — OpenAI-compatible /v1/chat/completions 接口

用法:
    python chat_server.py --model /path/to/model [--device cpu|nvidia] [--port 8000]

流式调用示例:
    curl http://localhost:8000/v1/chat/completions \\
         -H "Content-Type: application/json" \\
         -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
"""

import argparse
import asyncio
import json
import re
import threading
import time
import uuid
import sys
import os
from typing import AsyncIterator, Dict, List, Optional, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 模型 (OpenAI schema 子集)
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llaisys"
    messages: List[Message]
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    # 扩展字段：会话 ID，用于 KV-Cache 前缀复用
    session_id: Optional[str] = "default"



# ─────────────────────────────────────────────────────────────────────────────
# 用户会话：独立 KV-Cache + 前缀复用逻辑
# ─────────────────────────────────────────────────────────────────────────────

class UserSession:
    """
    一个用户的完整会话状态。

    - model_session : Qwen2Session，拥有独立 KV-Cache（隔离于其他用户）
    - lock          : asyncio.Lock，同一会话同一时刻只允许一个推理请求
    - cached_tokens : 当前 KV-Cache 对应的完整 token 序列，用于前缀复用
    """

    def __init__(self, model_session, tokenizer):
        self.model_session = model_session  # Qwen2Session
        self.tokenizer = tokenizer
        self.lock = asyncio.Lock()
        self.cached_tokens: List[int] = []

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _tokenize(self, messages: List[Message]) -> List[int]:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        try:
            # Qwen3 思考模型：禁用思考模式，直接生成回答
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False
            )
        except TypeError:
            # 不支持 enable_thinking 参数的 tokenizer（Qwen2/2.5 等）
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        return self.tokenizer.encode(prompt)

    # ── 核心生成（同步，在推理线程中运行）────────────────────────────────────

    def _stream_tokens_sync(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Iterator[tuple]:
        """
        同步生成器，yield (token_id, text_delta)。
        使用"全量 decode 做差"方法避免 BPE 边界乱码。
        在独立 OS 线程中调用；thread_local CUDA Context 确保并发隔离。
        """
        prompt_tokens = self._tokenize(messages)
        prefix_len = self._common_prefix_len(self.cached_tokens, prompt_tokens)

        # KV-Cache 前缀复用：回退到公共前缀末尾
        if prefix_len < len(self.cached_tokens):
            self.model_session.cache_pos = prefix_len

        new_tokens = prompt_tokens[prefix_len:]
        if not new_tokens:
            return

        # 简洁实现：skip_special_tokens=True 直接输出干净文本。
        # 思考模式已在 _tokenize 中通过 enable_thinking=False 禁用（Qwen3），
        # 无需在此进行额外的 in_think 过滤。
        accumulated_ids: List[int] = []
        text_so_far: str = ""
        generated_ids: List[int] = []

        for token_id in self.model_session.stream_generate(
            new_tokens,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ):
            generated_ids.append(token_id)
            accumulated_ids.append(token_id)

            new_text = self.tokenizer.decode(
                accumulated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            delta = new_text[len(text_so_far):]
            text_so_far = new_text
            if delta:
                yield token_id, delta

        # 推理完成后更新前缀缓存
        self.cached_tokens = prompt_tokens + generated_ids

    def clear(self):
        """清空 KV-Cache，重置前缀缓存。"""
        self.model_session.reset_cache()
        self.cached_tokens = []


# ─────────────────────────────────────────────────────────────────────────────
# ModelServer：多用户 Session 池
# ─────────────────────────────────────────────────────────────────────────────

class ModelServer:
    """
    多用户推理服务核心。

    - 每个 session_id 对应一个独立 UserSession（独立 KV-Cache）
    - 不同 session 的推理并发在不同线程执行（每线程独立 CUDA stream）
    - 同一 session 内请求通过 asyncio.Lock 串行化，避免 KV-Cache 竞争
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._sessions: Dict[str, UserSession] = {}
        self._sessions_mu = threading.Lock()  # 保护 _sessions dict

    def get_or_create_session(self, session_id: str) -> UserSession:
        with self._sessions_mu:
            if session_id not in self._sessions:
                model_session = self.model.create_session()
                self._sessions[session_id] = UserSession(model_session, self.tokenizer)
            return self._sessions[session_id]

    def delete_session(self, session_id: str):
        with self._sessions_mu:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                del self._sessions[session_id]

    def clear_session(self, session_id: str):
        with self._sessions_mu:
            if session_id in self._sessions:
                self._sessions[session_id].clear()

    def session_count(self) -> int:
        with self._sessions_mu:
            return len(self._sessions)


# ─────────────────────────────────────────────────────────────────────────────
# Web UI（内嵌 HTML）
# ─────────────────────────────────────────────────────────────────────────────

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLAISYS Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; display: flex; height: 100vh; overflow: hidden; }

  /* 左侧会话栏 */
  #sidebar {
    width: 220px; background: #202123; color: #fff; display: flex;
    flex-direction: column; padding: 12px; flex-shrink: 0;
  }
  #sidebar h2 { font-size: 14px; margin-bottom: 12px; color: #aaa; }
  #new-session-btn {
    background: #343541; border: 1px solid #555; color: #fff;
    border-radius: 6px; padding: 8px 12px; cursor: pointer; margin-bottom: 12px;
    font-size: 13px; width: 100%;
  }
  #new-session-btn:hover { background: #444654; }
  #session-list { flex: 1; overflow-y: auto; }
  .session-item {
    padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px; user-select: none;
  }
  .session-item:hover { background: #343541; }
  .session-item.active { background: #343541; border-left: 3px solid #10a37f; }
  .session-item .del-btn {
    opacity: 0; font-size: 16px; cursor: pointer; color: #888;
    background: none; border: none; line-height: 1;
  }
  .session-item:hover .del-btn { opacity: 1; }

  /* 右侧主体 */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* 标题栏 */
  #header {
    padding: 14px 20px; background: #fff; border-bottom: 1px solid #e5e7eb;
    font-size: 15px; font-weight: 600; color: #1a1a1a;
    display: flex; align-items: center; justify-content: space-between;
  }
  #header span { font-size: 12px; font-weight: 400; color: #888; }

  /* 消息区 */
  #messages { flex: 1; overflow-y: auto; padding: 20px; }
  .msg { display: flex; margin-bottom: 20px; animation: fadeIn .2s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
  .msg.user { justify-content: flex-end; }
  .msg.assistant { justify-content: flex-start; }
  .bubble {
    max-width: 75%; padding: 12px 16px; border-radius: 16px;
    line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-break: break-word;
  }
  .msg.user .bubble { background: #10a37f; color: #fff; border-bottom-right-radius: 4px; }
  .msg.assistant .bubble {
    background: #fff; color: #1a1a1a; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  .msg.user .bubble:hover { cursor: pointer; opacity: .9; }
  .edit-hint { font-size: 11px; color: rgba(255,255,255,.6); margin-top: 4px; text-align: right; }

  /* 输入区 */
  #input-area {
    padding: 16px 20px; background: #fff; border-top: 1px solid #e5e7eb;
    display: flex; gap: 10px; align-items: flex-end;
  }
  #user-input {
    flex: 1; border: 1px solid #d1d5db; border-radius: 10px; padding: 10px 14px;
    font-size: 14px; resize: none; outline: none; max-height: 150px; overflow-y: auto;
    line-height: 1.5; font-family: inherit;
  }
  #user-input:focus { border-color: #10a37f; }
  #send-btn {
    background: #10a37f; color: #fff; border: none; border-radius: 10px;
    padding: 10px 18px; cursor: pointer; font-size: 14px; font-weight: 600;
    transition: background .15s; white-space: nowrap;
  }
  #send-btn:hover:not(:disabled) { background: #0d8a6b; }
  #send-btn:disabled { background: #ccc; cursor: not-allowed; }

  #params { font-size: 12px; color: #888; padding: 6px 20px 0;
            display: flex; gap: 16px; align-items: center; }
  #params label { display: flex; align-items: center; gap: 4px; }
  #params input[type=range] { width: 80px; }
  #params input[type=number] { width: 50px; border: 1px solid #ddd; border-radius: 4px;
                                 padding: 2px 4px; font-size: 12px; }
  .cursor { display: inline-block; width: 2px; height: 1em; background: #555;
            animation: blink .7s step-end infinite; vertical-align: text-bottom; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>

<div id="sidebar">
  <h2>LLAISYS Chat</h2>
  <button id="new-session-btn" onclick="newSession()">+ 新建对话</button>
  <div id="session-list"></div>
</div>

<div id="main">
  <div id="header">
    <span id="session-title">对话</span>
    <div style="display:flex;gap:8px;align-items:center">
      <button onclick="clearCurrentSession()" id="clear-btn"
        style="font-size:12px;padding:4px 10px;border:1px solid #ddd;border-radius:6px;
               background:#fff;color:#888;cursor:pointer" title="清空当前会话历史">清空</button>
      <span id="model-info">LLAISYS</span>
    </div>
  </div>

  <div id="params">
    <label>温度
      <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.8"
             oninput="document.getElementById('temp-val').textContent=this.value">
      <span id="temp-val">0.8</span>
    </label>
    <label>Top-K <input type="number" id="top_k" value="50" min="1" max="500"></label>
    <label>Top-P <input type="number" id="top_p" value="0.9" min="0" max="1" step="0.05"></label>
    <label>最大 Token <input type="number" id="max_tokens" value="512" min="1" max="4096"></label>
  </div>

  <div id="messages"></div>

  <div id="input-area">
    <textarea id="user-input" rows="1" placeholder="输入消息，Enter 发送，Shift+Enter 换行…"
              onkeydown="handleKey(event)"></textarea>
    <button id="send-btn" onclick="sendMessage()">发送</button>
  </div>
</div>

<script>
const SERVER = "";
// session_id  { title, messages[], isGenerating, pendingReply, _bubble }
let sessions = {};
let currentSessionId = null;

//  localStorage 
function saveSessions() {
  try {
    const data = {};
    for (const [id, s] of Object.entries(sessions))
      data[id] = { title: s.title, messages: s.messages, isGenerating: false, pendingReply: "" };
    localStorage.setItem("llaisys_sessions", JSON.stringify(data));
  } catch {}
}

//  会话管理 
function genId() { return "s_" + Math.random().toString(36).slice(2, 10); }

function newSession() {
  const id = genId();
  sessions[id] = { title: "新对话", messages: [], isGenerating: false, pendingReply: "", _bubble: null };
  currentSessionId = id;
  renderSidebar();
  renderMessages();
  document.getElementById("send-btn").disabled = false;
  saveSessions();
}

function switchSession(id) {
  if (id === currentSessionId) return;
  currentSessionId = id;
  renderSidebar();
  renderMessages();  // 如该会话正在生成，renderMessages 会重建 _bubble 并显示已生成内容
  document.getElementById("send-btn").disabled = sessions[id]?.isGenerating || false;
  // 不再调用 /clear：每个 session_id 在服务器端有独立 KV-Cache，切换不应重置
}

function deleteSession(id, e) {
  e.stopPropagation();
  fetch(`${SERVER}/v1/sessions/${id}`, { method: "DELETE" }).catch(() => {});
  delete sessions[id];
  if (currentSessionId === id) {
    const ids = Object.keys(sessions);
    if (ids.length) {
      currentSessionId = ids[ids.length - 1];
      renderSidebar();
      renderMessages();
      document.getElementById("send-btn").disabled = sessions[currentSessionId]?.isGenerating || false;
    } else {
      newSession();
      return;
    }
  } else {
    renderSidebar();
  }
  saveSessions();
}

function clearCurrentSession() {
  const sess = sessions[currentSessionId];
  if (!sess || sess.isGenerating) return;
  sess.messages = [];
  sess.title = "新对话";
  sess.pendingReply = "";
  fetch(`${SERVER}/v1/sessions/${currentSessionId}/clear`, { method: "POST" }).catch(() => {});
  renderSidebar();
  renderMessages();
  saveSessions();
}

function renderSidebar() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  for (const [id, s] of Object.entries(sessions).reverse()) {
    const el = document.createElement("div");
    el.className = "session-item" + (id === currentSessionId ? " active" : "");
    el.onclick = () => switchSession(id);
    el.innerHTML = `<span>${s.title.slice(0, 20)}</span>
      <button class="del-btn" onclick="deleteSession('${id}', event)"></button>`;
    list.appendChild(el);
  }
  document.getElementById("session-title").textContent =
    sessions[currentSessionId]?.title || "对话";
}

//  消息渲染 
function renderMessages() {
  const container = document.getElementById("messages");
  container.innerHTML = "";
  const sess = sessions[currentSessionId];
  if (!sess) return;
  sess.messages.forEach((m, i) => appendBubble(m.role, m.content, i));
  // 如该会话正在生成，重建流式气泡并更新 _bubble 引用
  // （切换回此会话时可看到已生成的内容）
  if (sess.isGenerating) {
    const b = appendBubble("assistant", sess.pendingReply || "");
    if (!sess.pendingReply) b.innerHTML = '<span class="cursor"></span>';
    sess._bubble = b;
  }
  container.scrollTop = container.scrollHeight;
}

function appendBubble(role, content, idx) {
  const container = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.id = `msg-${idx ?? Date.now()}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  if (role === "user") {
    bubble.title = "点击编辑并重新生成";
    bubble.onclick = () => editMessage(idx);
    const hint = document.createElement("div");
    hint.className = "edit-hint";
    hint.textContent = "点击可编辑";
    div.appendChild(bubble);
    div.appendChild(hint);
  } else {
    div.appendChild(bubble);
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

//  编辑历史消息（重新生成） 
function editMessage(idx) {
  if (sessions[currentSessionId]?.isGenerating) return;
  const session = sessions[currentSessionId];
  const editedContent = prompt("编辑消息：", session.messages[idx].content);
  if (editedContent === null) return;
  session.messages[idx].content = editedContent;
  session.messages = session.messages.slice(0, idx + 1);
  renderMessages();
  doGenerate();
}

//  发送消息 
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendMessage() {
  const sess = sessions[currentSessionId];
  if (!sess || sess.isGenerating) return;
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  sess.messages.push({ role: "user", content: text });
  if (sess.messages.length === 1) { sess.title = text.slice(0, 24); renderSidebar(); }
  appendBubble("user", text, sess.messages.length - 1);
  saveSessions();
  doGenerate();
}

async function doGenerate() {
  const genSessionId = currentSessionId;
  const session = sessions[genSessionId];
  if (!session || session.isGenerating) return;

  session.isGenerating = true;
  session.pendingReply = "";
  // 仅对当前可见会话禁用发送按钮
  if (currentSessionId === genSessionId)
    document.getElementById("send-btn").disabled = true;

  const temperature = parseFloat(document.getElementById("temperature").value);
  const top_k = parseInt(document.getElementById("top_k").value);
  const top_p = parseFloat(document.getElementById("top_p").value);
  const max_tokens = parseInt(document.getElementById("max_tokens").value);

  // 创建初始气泡并将引用存入 session
  // （renderMessages 切换会话时会更新 session._bubble 指向新 DOM 元素）
  const bubble = appendBubble("assistant", "");
  bubble.innerHTML = '<span class="cursor"></span>';
  session._bubble = bubble;
  let reply = "";

  try {
    const resp = await fetch(`${SERVER}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llaisys",
        messages: session.messages,
        temperature, top_k, top_p, max_tokens,
        stream: true,
        session_id: genSessionId,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", streamDone = false;

    while (!streamDone) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") { streamDone = true; break; }
        try {
          const delta = JSON.parse(payload).choices?.[0]?.delta?.content || "";
          if (!delta) continue;
          reply += delta;
          session.pendingReply = reply;
          // session._bubble 可能被 renderMessages 更新为新 DOM 元素，始终用最新引用
          if (session._bubble && document.body.contains(session._bubble)) {
            session._bubble.textContent = reply;
            document.getElementById("messages").scrollTop = 99999;
          }
        } catch {}
      }
    }
    reader.cancel().catch(() => {});
  } catch (err) {
    reply = reply || `[错误: ${err.message}]`;
    if (session._bubble && document.body.contains(session._bubble))
      session._bubble.textContent = reply;
  } finally {
    session.pendingReply = "";
    session._bubble = null;
    session.isGenerating = false;
    session.messages.push({ role: "assistant", content: reply || "[无回复]" });
    saveSessions();
    // 如果用户当前在此会话，重新渲染消息并恢复按钮
    if (currentSessionId === genSessionId) {
      renderMessages();
      document.getElementById("send-btn").disabled = false;
    }
  }
}

//  初始化 
(function() {
  try {
    const saved = localStorage.getItem("llaisys_sessions");
    if (saved) {
      const data = JSON.parse(saved);
      const ids = Object.keys(data || {});
      if (ids.length > 0) {
        sessions = data;
        for (const s of Object.values(sessions)) {
          s.isGenerating = false;
          s.pendingReply = "";
          s._bubble = null;
        }
        currentSessionId = ids[ids.length - 1];
        renderSidebar();
        renderMessages();
        document.getElementById("send-btn").disabled = false;
        return;
      }
    }
  } catch {}
  newSession();
})();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLAISYS Chat Server", version="0.2.0")
_server: Optional[ModelServer] = None


def _make_sse_chunk(content: str, chat_id: str, model: str) -> str:
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _make_sse_done(chat_id: str, model: str) -> str:
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


async def _sse_stream(
    user_session: UserSession,
    request: "ChatCompletionRequest",
    chat_id: str,
) -> AsyncIterator[str]:
    """
    异步 SSE 生成器。

    将同步推理放入单独 OS 线程，通过 asyncio.Queue 将
    token 流传回事件循环。每个线程拥有独立的
    thread_local Context（独立 CUDA stream），不同 session 完全并发。
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def worker():
        try:
            for _, delta in user_session._stream_tokens_sync(
                request.messages,
                request.max_tokens,
                request.temperature,
                request.top_k,
                request.top_p,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(("delta", delta)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        kind, data = await queue.get()
        if kind == "done":
            yield _make_sse_done(chat_id, request.model)
            break
        if kind == "error":
            yield _make_sse_chunk(f"[Error: {data}]", chat_id, request.model)
            yield _make_sse_done(chat_id, request.model)
            break
        if kind == "delta" and data:
            yield _make_sse_chunk(data, chat_id, request.model)

    thread.join(timeout=5.0)


@app.get("/", response_class=HTMLResponse)
def index():
    return WEB_UI_HTML


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    sid = request.session_id or "default"
    user_session = _server.get_or_create_session(sid)

    if request.stream:
        async def locked_stream():
            async with user_session.lock:
                async for chunk in _sse_stream(user_session, request, chat_id):
                    yield chunk

        return StreamingResponse(
            locked_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        def blocking_generate():
            parts = []
            for _, delta in user_session._stream_tokens_sync(
                request.messages, request.max_tokens,
                request.temperature, request.top_k, request.top_p,
            ):
                parts.append(delta)
            return "".join(parts)

        async with user_session.lock:
            reply = await asyncio.to_thread(blocking_generate)

        return JSONResponse({
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "llaisys", "object": "model",
         "active_sessions": _server.session_count() if _server else 0}
    ]}


@app.post("/v1/sessions/{session_id}/clear")
def clear_session(session_id: str):
    """清除指定会话的 KV-Cache（会话切换时调用）。"""
    if _server is not None:
        _server.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    """彻底删除会话并释放其 KV-Cache 内存。"""
    if _server is not None:
        _server.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/sessions")
def list_sessions():
    """查看当前活跃会话数量。"""
    count = _server.session_count() if _server else 0
    return {"active_sessions": count}


# ─────────────────────────────────────────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _server

    parser = argparse.ArgumentParser(
        description="LLAISYS Chat Server (OpenAI-compatible, multi-user)"
    )
    parser.add_argument("--model", required=True, help="模型目录路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import llaisys
    from llaisys import DeviceType
    from transformers import AutoTokenizer

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU

    print(f"[server] 加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[server] 加载模型 (device={args.device}): {args.model}")
    model = llaisys.models.Qwen2(args.model, device)
    print("[server] 模型加载完成。")

    _server = ModelServer(model, tokenizer)

    print(f"[server] 多用户模式已启动: 每个 session_id 拥有独立 KV-Cache")
    print(f"[server] 监听 http://{args.host}:{args.port}")
    print(f"[server] Web UI: http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
