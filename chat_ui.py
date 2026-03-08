#!/usr/bin/env python3
"""
LLAISYS Chat UI — Gradio 6.x  豆包风格（全屏 + 历史侧边栏）

用法:
    python chat_ui.py --server http://localhost:8000 [--port 7860]
"""

import html as _html
import json
import re
import uuid
import argparse

import gradio as gr
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

# 匹配所有 <|token|> / <｜token｜> 变体（含全角竖线 U+FF5C、▁ U+2581、空格）
_SPECIAL_RE = re.compile(r"<[\s|｜]*[^<>]*[|｜][^<>]*>", re.IGNORECASE | re.UNICODE)

def _clean(text: str) -> str:
    return _SPECIAL_RE.sub("", text).strip()

def _normalize_think(text: str) -> str:
    if "</think>" in text and "<think>" not in text:
        return "<think>" + text
    return text

def _strip_think(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = re.sub(r"<think>.*$",         "", t,    flags=re.DOTALL)
    return _clean(t)

def _build_api_messages(history: list) -> list:
    msgs = []
    for item in history:
        role    = item["role"]
        content = item["content"] if isinstance(item["content"], str) else ""
        if role == "assistant":
            content = _strip_think(content)
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 历史对话管理
# ─────────────────────────────────────────────────────────────────────────────

def _conv_title(history: list) -> str:
    for m in history:
        if m.get("role") == "user":
            text = re.sub(r"<[^>]+>", "", m.get("content", "")).strip()
            return (text[:22] + "…") if len(text) > 22 else text
    return "新对话"

def _get_title_by_id(conversations: list, sid: str) -> str:
    for c in conversations:
        if c["id"] == sid:
            return c.get("title") or "新对话"
    return "新对话"

def _update_conversations(conversations: list, sid: str, history: list) -> list:
    convs = [dict(c) for c in conversations]
    for c in convs:
        if c["id"] == sid:
            c["messages"] = history
            if not c.get("title"):
                c["title"] = _conv_title(history)
            return convs
    convs.append({"id": sid, "title": _conv_title(history), "messages": history})
    return convs

def render_sidebar(conversations: list, current_id: str) -> str:
    items = ""
    for conv in reversed(conversations):
        cls   = "hi active" if conv["id"] == current_id else "hi"
        title = _html.escape((conv.get("title") or "新对话")[:26])
        cid   = conv["id"]
        items += (
            f'<div class="{cls}" onclick="__llaSel(\'{cid}\')">'
            f'<svg class="hico" viewBox="0 0 16 16"><path d="M8 1.5c-3.59 0-6.5 '
            f'2.69-6.5 6s2.91 6 6.5 6a6.4 6.4 0 002.8-.64l2.7.82-.83-2.56A5.84 '
            f'5.84 0 0014.5 7.5c0-3.31-2.91-6-6.5-6z" stroke="currentColor" '
            f'stroke-width="1.2" fill="none"/></svg>'
            f'<span class="hti">{title}</span>'
            f'</div>\n'
        )
    if not items:
        items = '<div class="hempty">暂无历史对话</div>'
    js = ("<script>function __llaSel(id){"
          "var e=document.querySelector('#hcb textarea')||document.querySelector('#hcb input');"
          "if(e){e.value=id;e.dispatchEvent(new Event('input',{bubbles:true}));}}"
          "</script>")
    return f'<div id="hsc">{items}</div>{js}'


# ─────────────────────────────────────────────────────────────────────────────
# CSS — 豆包全屏布局
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ══ RESET & FULL PAGE ═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
html, body {
    height: 100% !important; margin: 0 !important; padding: 0 !important;
    overflow: hidden !important;
    background: #f7f8fa !important;
    font-family: -apple-system,'PingFang SC','Microsoft YaHei','Segoe UI',sans-serif !important;
}
.gradio-container {
    height: 100vh !important; max-width: 100% !important; width: 100% !important;
    padding: 0 !important; margin: 0 !important;
    background: transparent !important; overflow: hidden !important;
}
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
    height: 100vh !important; max-height: 100vh !important;
    padding: 0 !important; margin: 0 !important;
    overflow: hidden !important; max-width: 100% !important;
}
footer { display: none !important; }

/* ══ OUTER ROW ═══════════════════════════════════════════════════════════════ */
#outer-row {
    display: flex !important; flex-direction: row !important;
    flex-wrap: nowrap !important; align-items: stretch !important;
    height: 100vh !important; width: 100% !important;
    gap: 0 !important; overflow: hidden !important;
}
#outer-row > * { min-width: 0 !important; }

/* ══ SIDEBAR ═════════════════════════════════════════════════════════════════ */
#sidebar {
    flex: 0 0 220px !important; width: 220px !important;
    min-width: 220px !important; max-width: 220px !important;
    height: 100vh !important;
    background: #f7f8fa !important;
    border-right: 1px solid #e8e8e8 !important;
    display: flex !important; flex-direction: column !important;
    overflow: hidden !important; padding: 0 !important; gap: 0 !important;
}
/* Sidebar header */
#sb-hdr { padding: 18px 16px 12px !important; border-bottom: 1px solid #efefef !important;
    flex-shrink: 0 !important; gap: 0 !important; }
#sb-hdr .gr-markdown p,
#sb-hdr p { font-size: 16px !important; font-weight: 700 !important;
    color: #111 !important; margin: 0 !important; }

/* Sidebar new-chat button */
#sb-new { padding: 8px 10px !important; flex-shrink: 0 !important; gap: 0 !important; }
#sb-new button {
    width: 100% !important; background: #e8faf3 !important; color: #059669 !important;
    border: 1px solid #bbf7d0 !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 600 !important; height: 36px !important;
    padding: 0 14px !important; text-align: left !important;
    transition: all 0.15s !important;
}
#sb-new button:hover { background: #d1fae5 !important; border-color: #6ee7b7 !important; }

/* History label */
#sb-lbl { padding: 10px 16px 2px !important; flex-shrink: 0 !important; gap: 0 !important; }
#sb-lbl .gr-markdown p,
#sb-lbl p { font-size: 11px !important; color: #9ca3af !important;
    font-weight: 600 !important; letter-spacing: 0.7px !important;
    text-transform: uppercase !important; margin: 0 !important; }

/* History HTML scroll area */
#sb-hist {
    flex: 1 !important; min-height: 0 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    padding: 0 8px !important;
}
#sb-hist::-webkit-scrollbar { width: 4px; }
#sb-hist::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
#sb-hist > div { display: block !important; }
#hsc { padding: 4px 0 8px !important; }

/* History items */
.hi {
    display: flex !important; align-items: center !important; gap: 7px !important;
    padding: 7px 9px !important; border-radius: 7px !important;
    cursor: pointer !important; color: #374151 !important;
    font-size: 13px !important; user-select: none !important;
    margin: 1px 0 !important; transition: background 0.12s !important;
}
.hi:hover  { background: #eef0f3 !important; }
.hi.active { background: #e8faf3 !important; color: #065f46 !important; }
.hico { width: 14px !important; height: 14px !important; flex-shrink: 0 !important;
    color: #9ca3af !important; }
.hi.active .hico { color: #09b37b !important; }
.hti  { flex: 1 !important; overflow: hidden !important;
    text-overflow: ellipsis !important; white-space: nowrap !important; }
.hempty { color: #9ca3af !important; font-size: 12px !important;
    text-align: center !important; padding: 24px 0 !important; }

/* Hidden JS click receiver */
#hcb { display: none !important; }

/* Sidebar bottom */
#sb-bot { border-top: 1px solid #efefef !important; padding: 12px 16px !important;
    flex-shrink: 0 !important; gap: 0 !important; }
#sb-bot .gr-markdown p,
#sb-bot p { font-size: 12px !important; color: #6b7280 !important; margin: 0 !important; }

/* ══ MAIN CHAT COLUMN ════════════════════════════════════════════════════════ */
#main-col {
    flex: 1 1 0 !important; height: 100vh !important; min-width: 0 !important;
    display: flex !important; flex-direction: column !important;
    background: #ffffff !important; padding: 0 !important; gap: 0 !important;
    overflow: hidden !important;
}
/* Chat topbar */
#ct-bar { height: 52px !important; min-height: 52px !important; flex-shrink: 0 !important;
    border-bottom: 1px solid #f0f0f0 !important; padding: 0 24px !important;
    display: flex !important; align-items: center !important;
    background: #fff !important; gap: 0 !important; }
#ct-bar .gr-markdown p,
#ct-bar p { font-size: 15px !important; font-weight: 600 !important;
    color: #111 !important; margin: 0 !important; }

/* Chatbot fills remaining height */
#chatbot-box {
    flex: 1 !important; min-height: 0 !important;
    border: none !important; box-shadow: none !important;
    background: #fafbfc !important;
}
/* Override Gradio's inline height */
#chatbot-box > div { height: 100% !important; }
#chatbot-box .bubble-wrap { max-height: none !important; }

/* Message bubbles */
#chatbot-box .message-wrap { padding: 6px 24px !important; }
#chatbot-box .message.user .bubble-wrap,
#chatbot-box .user .bubble-wrap {
    background: #f0fdf7 !important; border: 1px solid #c6f0dc !important;
    border-radius: 14px 14px 4px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
#chatbot-box .message.bot .bubble-wrap,
#chatbot-box .bot .bubble-wrap,
#chatbot-box .message.assistant .bubble-wrap {
    background: #ffffff !important; border: 1px solid #eaecef !important;
    border-radius: 14px 14px 14px 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ══ INPUT ZONE ══════════════════════════════════════════════════════════════ */
#inp-zone {
    border-top: 1px solid #f0f0f0 !important; flex-shrink: 0 !important;
    padding: 12px 24px 8px !important; background: #fff !important; gap: 10px !important;
}
#msg-input { margin: 0 !important; }
#msg-input textarea {
    border: 1.5px solid #dde1e7 !important; border-radius: 12px !important;
    background: #f9fafb !important; font-size: 14px !important;
    line-height: 1.6 !important; padding: 10px 14px !important; color: #111 !important;
    transition: border-color .18s, box-shadow .18s !important; resize: none !important;
    font-family: inherit !important;
}
#msg-input textarea:focus {
    border-color: #09b37b !important; background: #fff !important;
    box-shadow: 0 0 0 3px rgba(9,179,123,.10) !important; outline: none !important;
}
#msg-input textarea::placeholder { color: #b0b7c0 !important; }
#send-btn { margin: 0 !important; }
#send-btn button {
    background: #09b37b !important; border: none !important;
    border-radius: 11px !important; color: #fff !important;
    font-size: 14px !important; font-weight: 600 !important; min-height: 50px !important;
    box-shadow: 0 2px 8px rgba(9,179,123,.28) !important;
    transition: background .15s, box-shadow .15s, transform .12s !important;
    font-family: inherit !important;
}
#send-btn button:hover {
    background: #07a06e !important; box-shadow: 0 5px 18px rgba(9,179,123,.38) !important;
    transform: translateY(-1px) !important;
}
#send-btn button:active { transform: translateY(0) !important; }

/* Action row */
#act-row { padding: 2px 24px 12px !important; background: #fff !important;
    gap: 10px !important; flex-shrink: 0 !important; }
#act-row button {
    background: transparent !important; border: 1px solid #e5e7eb !important;
    border-radius: 8px !important; color: #6b7280 !important;
    font-size: 12.5px !important; height: 32px !important; transition: all .15s !important;
}
#clear-btn button:hover {
    border-color: #fca5a5 !important; color: #dc2626 !important;
    background: #fff8f8 !important;
}

/* ══ RIGHT PANEL ═════════════════════════════════════════════════════════════ */
#rp-col {
    flex: 0 0 260px !important; width: 260px !important;
    min-width: 260px !important; max-width: 260px !important;
    height: 100vh !important;
    border-left: 1px solid #f0f0f0 !important; background: #fafbfc !important;
    overflow-y: auto !important; padding: 20px 16px !important;
    flex-shrink: 0 !important; gap: 10px !important;
}
#rp-col::-webkit-scrollbar { width: 4px; }
#rp-col::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
#rp-col .gr-markdown h3 { font-size: 13px !important; font-weight: 600 !important;
    color: #374151 !important; margin: 0 0 12px !important; }
#rp-col .gr-markdown p { font-size: 13px !important; color: #4b5563 !important;
    line-height: 1.6 !important; }
#rp-col label span { font-size: 13px !important; }

/* ══ REASONING (think) BLOCK ═════════════════════════════════════════════════ */
.thinking {
    background: linear-gradient(135deg,#f0fdf8,#ecfdf5) !important;
    border: 1px solid #a7f3d0 !important; border-left: 3px solid #09b37b !important;
    border-radius: 10px !important; margin: 6px 0 10px !important;
    font-size: 13px !important; overflow: hidden !important;
}
.thinking > summary {
    padding: 9px 13px !important; color: #065f46 !important; font-weight: 500 !important;
    cursor: pointer !important; list-style: none !important; user-select: none !important;
    background: rgba(9,179,123,.05) !important;
    display: flex !important; align-items: center !important; gap: 6px !important;
}
.thinking > summary::marker,
.thinking > summary::-webkit-details-marker { display: none !important; }
.thinking > summary::before {
    content: "▶" !important; font-size: 10px !important; color: #09b37b !important;
    transition: transform .2s !important; display: inline-block !important;
}
.thinking[open] > summary::before { transform: rotate(90deg) !important; }
.thinking > summary:hover { background: rgba(9,179,123,.10) !important; }
.thinking > div, .thinking > p {
    padding: 8px 14px 11px !important; color: #1a3a2e !important;
    line-height: 1.7 !important; border-top: 1px solid #bbf7d0 !important;
}

/* ══ CODE BLOCKS ═════════════════════════════════════════════════════════════ */
code, pre { font-family: 'JetBrains Mono','Fira Code',Consolas,monospace !important; }
pre { background: #1e2433 !important; border-radius: 8px !important;
    padding: 14px 16px !important; overflow-x: auto !important; border: none !important; }
pre code { color: #e2e8f0 !important; font-size: 13px !important; line-height: 1.6 !important; }
:not(pre) > code { background: #f1f4f8 !important; color: #d63384 !important;
    border-radius: 4px !important; padding: 2px 5px !important; font-size: 13px !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# 流式生成
# ─────────────────────────────────────────────────────────────────────────────

def respond(user_msg, history, session_id, server_url,
            temperature, top_k, top_p, max_tokens, conversations):
    if not user_msg.strip():
        yield history, "", conversations, render_sidebar(conversations, session_id), \
              _get_title_by_id(conversations, session_id)
        return

    api_msgs = _build_api_messages(history)
    api_msgs.append({"role": "user", "content": user_msg})

    history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": "▌"},
    ]
    # Pre-compute sidebar (won't change during streaming)
    pre_sidebar = render_sidebar(conversations, session_id)
    pre_title   = _get_title_by_id(conversations, session_id) or "新对话"
    yield history, "", conversations, pre_sidebar, pre_title

    full_text = ""
    try:
        with requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages":    api_msgs,
                "temperature": float(temperature),
                "top_k":       int(top_k),
                "top_p":       float(top_p),
                "max_tokens":  int(max_tokens),
                "stream":      True,
                "session_id":  session_id,
            },
            stream=True, timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode() if isinstance(line, bytes) else line
                if not s.startswith("data: "):
                    continue
                payload = s[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text += delta
                        history[-1]["content"] = _normalize_think(_clean(full_text))
                        yield history, "", conversations, pre_sidebar, pre_title
                except Exception:
                    pass
    except Exception as exc:
        history[-1]["content"] = f"❌ 连接错误：{exc}"
        yield history, "", conversations, pre_sidebar, pre_title
        return

    history[-1]["content"] = _normalize_think(_clean(full_text))
    # 生成完毕：更新历史记录和侧边栏
    updated_convs = _update_conversations(conversations, session_id, history)
    new_title     = _conv_title(history)
    new_sidebar   = render_sidebar(updated_convs, session_id)
    yield history, "", updated_convs, new_sidebar, new_title


def do_clear(session_id, server_url, conversations):
    try:
        requests.post(f"{server_url}/v1/sessions/{session_id}/clear", timeout=5)
    except Exception:
        pass
    # 清空消息，保留此会话 id
    sidebar = render_sidebar(conversations, session_id)
    title   = _get_title_by_id(conversations, session_id) or "新对话"
    return [], "", sidebar, title


def do_new_session(conversations):
    new_id  = str(uuid.uuid4())
    sidebar = render_sidebar(conversations, new_id)
    return [], "", new_id, sidebar, "新对话"


def on_history_click(conv_id, conversations, server_url):
    if not conv_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""
    for conv in conversations:
        if conv["id"] == conv_id:
            msgs    = conv.get("messages", [])
            title   = conv.get("title") or "新对话"
            sidebar = render_sidebar(conversations, conv_id)
            return msgs, conv_id, "", sidebar, title, ""
    return [], conv_id, "", render_sidebar(conversations, conv_id), "新对话", ""


# ─────────────────────────────────────────────────────────────────────────────
# UI 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_ui(server_url: str) -> gr.Blocks:
    with gr.Blocks(title="LLAISYS Chat") as demo:

        # ── State ──────────────────────────────────────────────────────────
        session_id_st    = gr.State(str(uuid.uuid4()))
        server_url_st    = gr.State(server_url)
        conversations_st = gr.State([])

        with gr.Row(elem_id="outer-row"):

            # ══ LEFT SIDEBAR ═══════════════════════════════════════════════
            with gr.Column(scale=0, min_width=220, elem_id="sidebar"):

                with gr.Row(elem_id="sb-hdr"):
                    gr.Markdown("🤖 **LLAISYS Chat**")

                sb_new_btn = gr.Button("✦  新对话", elem_id="sb-new")

                with gr.Row(elem_id="sb-lbl"):
                    gr.Markdown("历史对话")

                history_html = gr.HTML(
                    render_sidebar([], ""),
                    elem_id="sb-hist",
                )

                # 隐藏的 textbox，接收侧边栏 JS 点击事件
                hist_click = gr.Textbox(value="", visible=False, elem_id="hcb")

                with gr.Row(elem_id="sb-bot"):
                    gr.Markdown("👤 &nbsp;LLAISYS User")

            # ══ MAIN CHAT COLUMN ═══════════════════════════════════════════
            with gr.Column(scale=5, elem_id="main-col"):

                # 顶部标题栏
                with gr.Row(elem_id="ct-bar"):
                    conv_title_md = gr.Markdown("新对话")

                # 聊天区
                chatbot = gr.Chatbot(
                    elem_id="chatbot-box",
                    height=600,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                    reasoning_tags=[("<think>", "</think>")],
                    placeholder=(
                        "<div style='text-align:center;padding:100px 20px 40px'>"
                        "<div style='font-size:52px;margin-bottom:16px;opacity:.35'>✦</div>"
                        "<p style='color:#4b5563;font-size:16px;font-weight:500;margin:0 0 8px'>"
                        "有什么可以帮助你？</p>"
                        "<p style='color:#9ca3af;font-size:13px;margin:0'>"
                        "推理过程将以可折叠的思考块展示</p>"
                        "</div>"
                    ),
                )

                # 输入行
                with gr.Row(elem_id="inp-zone"):
                    msg_box = gr.Textbox(
                        placeholder="发消息给 LLAISYS… （Enter 发送，Shift+Enter 换行）",
                        show_label=False, lines=2, max_lines=8,
                        scale=9, container=False, autofocus=True,
                        elem_id="msg-input",
                    )
                    send_btn = gr.Button(
                        "发 送", variant="primary", scale=1, min_width=90,
                        elem_id="send-btn",
                    )

                # 操作行
                with gr.Row(elem_id="act-row"):
                    clear_btn = gr.Button("🗑  清空对话", size="sm", elem_id="clear-btn")

            # ══ RIGHT SETTINGS PANEL ═══════════════════════════════════════
            with gr.Column(scale=0, min_width=260, elem_id="rp-col"):
                gr.Markdown("### ⚙️  生成参数")
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    label="温度 Temperature", info="↑ 更随机 · ↓ 更保守",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=200, value=50, step=1, label="Top-K",
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P  核采样",
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=4096, value=1024, step=64, label="最大 Token 数",
                )
                gr.Markdown(
                    "<hr style='border-color:#ebebeb;margin:14px 0'/>\n\n"
                    "### 💭  思考块\n\n"
                    "点击 **Reasoning** 折叠块\n"
                    "展开 / 收起推理过程。\n\n"
                    "适用于 **DeepSeek-R1**、\n"
                    "**Qwen3** 等推理模型。"
                )

        # ── 事件绑定 ────────────────────────────────────────────────────────
        # gen_outputs: [chatbot, msg_box, conversations_st, history_html, conv_title_md]
        gen_inputs  = [msg_box, chatbot, session_id_st, server_url_st,
                       temperature, top_k, top_p, max_tokens, conversations_st]
        gen_outputs = [chatbot, msg_box, conversations_st, history_html, conv_title_md]

        send_btn.click(respond, inputs=gen_inputs, outputs=gen_outputs)
        msg_box.submit(respond, inputs=gen_inputs, outputs=gen_outputs)

        clear_btn.click(
            do_clear,
            inputs=[session_id_st, server_url_st, conversations_st],
            outputs=[chatbot, msg_box, history_html, conv_title_md],
        )

        sb_new_btn.click(
            do_new_session,
            inputs=[conversations_st],
            outputs=[chatbot, msg_box, session_id_st, history_html, conv_title_md],
        )

        hist_click.change(
            on_history_click,
            inputs=[hist_click, conversations_st, server_url_st],
            outputs=[chatbot, session_id_st, msg_box, history_html, conv_title_md, hist_click],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chat UI (Gradio 6)")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args()

    print(f"[ui] 后端: {args.server}")
    print(f"[ui] 界面: http://localhost:{args.port}")
    demo = build_ui(args.server)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.teal,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
