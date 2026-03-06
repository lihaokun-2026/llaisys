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
import json
import time
import uuid
import sys
import os
from typing import List, Optional, Iterator

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
# KV-Cache 前缀匹配引擎
# ─────────────────────────────────────────────────────────────────────────────

class ChatEngine:
    """
    单模型聊天引擎，支持跨请求 KV-Cache 前缀复用。

    设计：
    - 维护 _cached_tokens：当前模型 KV-Cache 对应的 token 序列
    - 每次请求时，计算新 prompt 与 _cached_tokens 的最长公共前缀
    - 只处理差量 token，复用已缓存的 KV 状态
    - 单用户假设：无并发锁，每请求串行执行
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # 当前 KV-Cache 对应的完整 token 序列（含 prompt 和已生成部分）
        self._cached_tokens: List[int] = []

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _common_prefix_len(self, a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _tokenize(self, messages: List[Message]) -> List[int]:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer.encode(prompt)

    # ── 核心生成 ─────────────────────────────────────────────────────────────

    def _prepare_and_stream(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Iterator[int]:
        """
        执行带前缀匹配的增量推理，yield 生成的 token IDs。
        调用者负责检测 EOS 并停止迭代。
        """
        prompt_tokens = self._tokenize(messages)

        # 前缀匹配：找出与当前 KV-Cache 的公共前缀长度
        prefix_len = self._common_prefix_len(self._cached_tokens, prompt_tokens)

        # 将 cache_pos 回退到公共前缀末尾
        if prefix_len < len(self._cached_tokens):
            self.model.cache_pos = prefix_len

        # 只处理新增 token
        new_tokens = prompt_tokens[prefix_len:]
        if not new_tokens:
            # 极少数情况：prompt 完全已在缓存中，不应发生
            return

        generated_ids: List[int] = []
        for token_id in self.model.stream_generate(
            new_tokens,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ):
            generated_ids.append(token_id)
            yield token_id
            if token_id == self.tokenizer.eos_token_id:
                break

        # 更新全局缓存 token 序列
        self._cached_tokens = prompt_tokens + generated_ids

    def stream_text(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Iterator[str]:
        """
        yield 增量文本片段（正确处理多 token Unicode 字符）。
        使用"全量 decode 做差"方法，避免 BPE 边界产生乱码。
        """
        accumulated_ids: List[int] = []
        text_so_far = ""

        for token_id in self._prepare_and_stream(
            messages, max_tokens, temperature, top_k, top_p
        ):
            accumulated_ids.append(token_id)
            # 全量 decode，与上次相比取差量
            new_text = self.tokenizer.decode(
                accumulated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            delta = new_text[len(text_so_far):]
            text_so_far = new_text
            if delta:
                yield delta

    def generate_text(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        """阻塞式生成，返回完整回复文本。"""
        return "".join(
            self.stream_text(messages, max_tokens, temperature, top_k, top_p)
        )

    def clear_cache(self):
        """清空 KV-Cache（强制下次请求从头重算）。"""
        self.model.reset_cache()
        self._cached_tokens = []


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
    <span id="model-info">LLAISYS</span>
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
let sessions = {};          // session_id -> { title, messages[] }
let currentSessionId = null;
let isGenerating = false;

// ── 会话管理 ──────────────────────────────────────────────────────────────────
function genId() { return "s_" + Math.random().toString(36).slice(2, 10); }

function newSession() {
  const id = genId();
  sessions[id] = { title: "新对话", messages: [] };
  switchSession(id);
  renderSidebar();
}

function switchSession(id) {
  currentSessionId = id;
  renderSidebar();
  renderMessages();
  // 通知服务器清空 KV-Cache（不同会话无法共享缓存）
  fetch(`${SERVER}/v1/sessions/${id}/clear`, { method: "POST" }).catch(() => {});
}

function deleteSession(id, e) {
  e.stopPropagation();
  delete sessions[id];
  if (currentSessionId === id) {
    const ids = Object.keys(sessions);
    if (ids.length) switchSession(ids[ids.length - 1]);
    else newSession();
  }
  renderSidebar();
}

function renderSidebar() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  for (const [id, s] of Object.entries(sessions).reverse()) {
    const el = document.createElement("div");
    el.className = "session-item" + (id === currentSessionId ? " active" : "");
    el.onclick = () => switchSession(id);
    el.innerHTML = `<span>${s.title.slice(0, 20)}</span>
      <button class="del-btn" onclick="deleteSession('${id}', event)">×</button>`;
    list.appendChild(el);
  }
  document.getElementById("session-title").textContent =
    sessions[currentSessionId]?.title || "对话";
}

// ── 消息渲染 ─────────────────────────────────────────────────────────────────
function renderMessages() {
  const container = document.getElementById("messages");
  container.innerHTML = "";
  const msgs = sessions[currentSessionId]?.messages || [];
  msgs.forEach((m, i) => appendBubble(m.role, m.content, i));
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

// ── 编辑历史消息（重新生成） ──────────────────────────────────────────────────
function editMessage(idx) {
  if (isGenerating) return;
  const session = sessions[currentSessionId];
  // 截断到这条 user 消息
  const editedContent = prompt("编辑消息：", session.messages[idx].content);
  if (editedContent === null) return;
  session.messages[idx].content = editedContent;
  session.messages = session.messages.slice(0, idx + 1); // 删除后续
  renderMessages();
  // 重新生成 assistant 回复
  doGenerate();
}

// ── 发送消息 ─────────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendMessage() {
  if (isGenerating) return;
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";

  const session = sessions[currentSessionId];
  session.messages.push({ role: "user", content: text });
  if (session.messages.length === 1) {
    session.title = text.slice(0, 24);
    renderSidebar();
  }
  appendBubble("user", text, session.messages.length - 1);
  doGenerate();
}

async function doGenerate() {
  if (isGenerating) return;
  isGenerating = true;
  document.getElementById("send-btn").disabled = true;

  const session = sessions[currentSessionId];
  const temperature = parseFloat(document.getElementById("temperature").value);
  const top_k = parseInt(document.getElementById("top_k").value);
  const top_p = parseFloat(document.getElementById("top_p").value);
  const max_tokens = parseInt(document.getElementById("max_tokens").value);

  // 添加 assistant 气泡（流式填充）
  const bubble = appendBubble("assistant", "");
  bubble.innerHTML = '<span class="cursor"></span>';
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
        session_id: currentSessionId,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;
        try {
          const chunk = JSON.parse(payload);
          const delta = chunk.choices?.[0]?.delta?.content || "";
          reply += delta;
          bubble.textContent = reply;
          document.getElementById("messages").scrollTop = 99999;
        } catch {}
      }
    }
  } catch (err) {
    bubble.textContent = `[错误: ${err.message}]`;
  }

  session.messages.push({ role: "assistant", content: reply });
  isGenerating = false;
  document.getElementById("send-btn").disabled = false;
}

// ── 初始化 ────────────────────────────────────────────────────────────────────
newSession();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")
_engine: Optional["ChatEngine"] = None


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


@app.get("/", response_class=HTMLResponse)
def index():
    return WEB_UI_HTML


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        def token_stream():
            try:
                for text in _engine.stream_text(
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.top_k,
                    request.top_p,
                ):
                    yield _make_sse_chunk(text, chat_id, request.model)
            except Exception as e:
                err_chunk = _make_sse_chunk(f"[Server Error: {e}]", chat_id, request.model)
                yield err_chunk
            yield _make_sse_done(chat_id, request.model)

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        reply = _engine.generate_text(
            request.messages,
            request.max_tokens,
            request.temperature,
            request.top_k,
            request.top_p,
        )
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
    return {"object": "list", "data": [{"id": "llaisys", "object": "model"}]}


@app.post("/v1/sessions/{session_id}/clear")
def clear_session(session_id: str):
    """清除 KV-Cache（用于会话切换时强制重算）。"""
    if _engine is not None:
        _engine.clear_cache()
    return {"status": "ok", "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _engine

    parser = argparse.ArgumentParser(
        description="LLAISYS Chat Server (OpenAI-compatible)"
    )
    parser.add_argument("--model", required=True, help="模型目录路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # 延迟导入以避免在 import 时就加载模型
    import llaisys
    from llaisys import DeviceType
    from transformers import AutoTokenizer

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU

    print(f"[server] 加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[server] 加载模型 (device={args.device}): {args.model}")
    model = llaisys.models.Qwen2(args.model, device)
    print("[server] 模型加载完成。")

    _engine = ChatEngine(model, tokenizer)

    print(f"[server] 监听 http://{args.host}:{args.port}")
    print(f"[server] Web UI: http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
