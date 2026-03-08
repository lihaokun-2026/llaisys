#!/usr/bin/env python3
"""
LLAISYS Chat UI — Gradio 前端 (Gradio 6.x)

用法:
    python chat_ui.py --server http://localhost:8000 [--port 7860]
"""

import json
import re
import uuid
import argparse

import gradio as gr
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

# 匹配 <|token|> / <｜token｜>（含全角竖线 U+FF5C、SentencePiece ▁ U+2581、空格变体）
_SPECIAL_RE = re.compile(r"<[\s|｜]*[^<>]*[|｜][^<>]*>", re.IGNORECASE | re.UNICODE)

def _clean(text: str) -> str:
    return _SPECIAL_RE.sub("", text).strip()

def _normalize_think(text: str) -> str:
    """若仅有 </think> 无 <think>，补上开标签，供 reasoning_tags 渲染。"""
    if "</think>" in text and "<think>" not in text:
        return "<think>" + text
    return text

def _strip_think(text: str) -> str:
    """多轮对话时，assistant 消息去掉思考块，只保留最终回答。"""
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
# CSS — 豆包风格：白底卡片、极简线条、绿色点缀
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ===== 全局 ===== */
html, body { background: #f2f3f5 !important; }
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: #f2f3f5 !important;
    font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', sans-serif !important;
}
.main { padding: 0 !important; }
footer { display: none !important; }

/* ===== 顶部导航栏 ===== */
#top-bar {
    background: #ffffff;
    border-bottom: 1px solid #e8e8e8;
    padding: 0 28px;
    height: 58px;
    display: flex !important;
    align-items: center;
    gap: 0;
    margin-bottom: 0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
#top-bar .gr-markdown, #top-bar > div { margin: 0 !important; padding: 0 !important; }
#top-bar p { margin: 0 !important; font-size: 16px; }

/* ===== 内容布局 ===== */
#content-wrap {
    max-width: 1380px;
    margin: 16px auto !important;
    padding: 0 16px;
    gap: 14px !important;
    align-items: flex-start !important;
}

/* ===== 聊天卡片 ===== */
#chat-card {
    background: #ffffff !important;
    border: 1px solid #e8eaed !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07) !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* Chatbot 本体 */
#chatbot-box {
    border: none !important;
    box-shadow: none !important;
    background: #fafbfc !important;
}

/* 消息区域内边距 */
#chatbot-box .message-wrap { padding: 4px 20px !important; }

/* 用户气泡 */
#chatbot-box .message.user .bubble-wrap,
#chatbot-box .user .bubble-wrap {
    background: #edfbf4 !important;
    border: 1px solid #c6f0dc !important;
    border-radius: 14px 14px 4px 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
/* AI 气泡 */
#chatbot-box .message.bot .bubble-wrap,
#chatbot-box .message.assistant .bubble-wrap,
#chatbot-box .bot .bubble-wrap {
    background: #ffffff !important;
    border: 1px solid #ebebeb !important;
    border-radius: 14px 14px 14px 4px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ===== 输入区 ===== */
#input-area {
    border-top: 1px solid #f0f0f0 !important;
    background: #ffffff !important;
    padding: 12px 16px 8px !important;
    margin: 0 !important;
    gap: 10px !important;
}

/* 文本框 */
#msg-input { margin: 0 !important; }
#msg-input textarea {
    border: 1.5px solid #dde1e7 !important;
    border-radius: 12px !important;
    background: #fafafa !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 10px 14px !important;
    color: #1a1a1a !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: #09b37b !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(9,179,123,0.10) !important;
    outline: none !important;
}
#msg-input textarea::placeholder { color: #b0b7c0 !important; }

/* 发送按钮 */
#send-btn { margin: 0 !important; }
#send-btn button {
    background: #09b37b !important;
    border: none !important;
    border-radius: 11px !important;
    color: #fff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    min-height: 50px !important;
    box-shadow: 0 2px 8px rgba(9,179,123,0.30) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.12s !important;
}
#send-btn button:hover {
    background: #07a06e !important;
    box-shadow: 0 5px 18px rgba(9,179,123,0.40) !important;
    transform: translateY(-1px) !important;
}
#send-btn button:active { transform: translateY(0) !important; }

/* 底部按钮行 */
#action-row {
    background: #ffffff !important;
    padding: 4px 16px 14px !important;
    margin: 0 !important;
    gap: 10px !important;
}
#clear-btn button, #new-btn button {
    background: #fff !important;
    border: 1px solid #e2e6ea !important;
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    height: 34px !important;
    transition: all 0.15s !important;
}
#clear-btn button:hover {
    border-color: #fca5a5 !important;
    color: #dc2626 !important;
    background: #fff8f8 !important;
}
#new-btn button:hover {
    border-color: #6ee7b7 !important;
    color: #059669 !important;
    background: #f0fdf7 !important;
}

/* ===== 参数面板 ===== */
#param-card {
    background: #ffffff !important;
    border: 1px solid #e8eaed !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07) !important;
    gap: 10px !important;
}
#param-card label span {
    font-size: 13px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}

/* ===== Reasoning 思考块 ===== */
.thinking {
    background: linear-gradient(135deg, #f0fdf8 0%, #ecfdf5 100%) !important;
    border: 1px solid #a7f3d0 !important;
    border-left: 3px solid #09b37b !important;
    border-radius: 10px !important;
    margin: 6px 0 10px !important;
    font-size: 13px !important;
    overflow: hidden !important;
}
.thinking > summary {
    padding: 9px 13px !important;
    color: #065f46 !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    list-style: none !important;
    user-select: none !important;
    background: rgba(9,179,123,0.05) !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.thinking > summary::marker,
.thinking > summary::-webkit-details-marker { display: none !important; }
.thinking > summary:hover { background: rgba(9,179,123,0.10) !important; }
.thinking > summary::before {
    content: "▶" !important;
    font-size: 10px !important;
    color: #09b37b !important;
    transition: transform 0.2s !important;
    display: inline-block !important;
}
.thinking[open] > summary::before { transform: rotate(90deg) !important; }
.thinking > div, .thinking > p {
    padding: 8px 14px 11px !important;
    color: #1a3a2e !important;
    line-height: 1.7 !important;
    border-top: 1px solid #bbf7d0 !important;
}

/* ===== 代码块美化 ===== */
code, pre {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, 'Courier New', monospace !important;
}
pre {
    background: #1e2433 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    overflow-x: auto !important;
    border: none !important;
}
pre code { color: #e2e8f0 !important; font-size: 13px !important; line-height: 1.6 !important; }
:not(pre) > code {
    background: #f1f4f8 !important;
    color: #d63384 !important;
    border-radius: 4px !important;
    padding: 2px 5px !important;
    font-size: 13px !important;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 流式生成
# ─────────────────────────────────────────────────────────────────────────────

def respond(user_msg, history, session_id, server_url,
            temperature, top_k, top_p, max_tokens):
    if not user_msg.strip():
        yield history, ""
        return

    api_msgs = _build_api_messages(history)
    api_msgs.append({"role": "user", "content": user_msg})

    history = history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": "▌"},
    ]
    yield history, ""

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
                        yield history, ""
                except Exception:
                    pass
    except Exception as exc:
        history[-1]["content"] = f"❌ 连接错误：{exc}"
        yield history, ""
        return

    history[-1]["content"] = _normalize_think(_clean(full_text))
    yield history, ""


def do_clear(session_id, server_url):
    try:
        requests.post(f"{server_url}/v1/sessions/{session_id}/clear", timeout=5)
    except Exception:
        pass
    return [], ""


def do_new_session():
    return [], "", str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# UI 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_ui(server_url: str) -> gr.Blocks:
    with gr.Blocks(title="LLAISYS Chat") as demo:

        session_id_st = gr.State(str(uuid.uuid4()))
        server_url_st = gr.State(server_url)

        # ── 顶部导航栏 ──────────────────────────────────────────────────────
        with gr.Row(elem_id="top-bar"):
            gr.Markdown(
                "🤖 &nbsp;**LLAISYS Chat**"
                "&ensp;<span style='color:#9ca3af;font-weight:400;font-size:13px'>"
                "高性能 LLM 推理框架</span>"
            )

        # ── 主体内容 ────────────────────────────────────────────────────────
        with gr.Row(equal_height=False, elem_id="content-wrap"):

            # ── 聊天列 ──────────────────────────────────────────────────────
            with gr.Column(scale=5, elem_id="chat-card"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot-box",
                    height=545,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                    reasoning_tags=[("<think>", "</think>")],
                    placeholder=(
                        "<div style='text-align:center;padding:90px 20px 40px;'>"
                        "<div style='font-size:52px;margin-bottom:16px;opacity:.5'>✦</div>"
                        "<p style='color:#4b5563;font-size:16px;font-weight:500;margin:0 0 8px'>"
                        "有什么可以帮助你？</p>"
                        "<p style='color:#9ca3af;font-size:13px;margin:0'>"
                        "推理过程将以可折叠的思考块展示</p>"
                        "</div>"
                    ),
                )

                # 输入行
                with gr.Row(elem_id="input-area"):
                    msg_box = gr.Textbox(
                        placeholder="发消息给 LLAISYS… （Enter 发送，Shift+Enter 换行）",
                        show_label=False,
                        lines=2,
                        max_lines=8,
                        scale=9,
                        container=False,
                        autofocus=True,
                        elem_id="msg-input",
                    )
                    send_btn = gr.Button(
                        "发 送",
                        variant="primary",
                        scale=1,
                        min_width=88,
                        elem_id="send-btn",
                    )

                # 操作按钮行
                with gr.Row(elem_id="action-row"):
                    clear_btn = gr.Button("🗑  清空对话", size="sm", elem_id="clear-btn")
                    new_btn   = gr.Button("✦  新对话",   size="sm", elem_id="new-btn")

            # ── 参数面板 ────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=248, elem_id="param-card"):
                gr.Markdown("### ⚙️  生成参数")
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    label="温度 Temperature",
                    info="↑ 更随机 · ↓ 更保守",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=200, value=50, step=1,
                    label="Top-K",
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P  核采样",
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=4096, value=1024, step=64,
                    label="最大 Token 数",
                )
                gr.Markdown(
                    "<hr style='border-color:#f0f0f0;margin:14px 0'/>\n\n"
                    "### 💭  思考块\n\n"
                    "点击 **Reasoning** 折叠块\n"
                    "展开 / 收起推理过程。\n\n"
                    "适用于 DeepSeek-R1、\n"
                    "Qwen3 等推理模型。"
                )

        # ── 事件绑定 ────────────────────────────────────────────────────────
        gen_inputs  = [msg_box, chatbot, session_id_st, server_url_st,
                       temperature, top_k, top_p, max_tokens]
        gen_outputs = [chatbot, msg_box]

        send_btn.click(respond, inputs=gen_inputs, outputs=gen_outputs)
        msg_box.submit(respond, inputs=gen_inputs, outputs=gen_outputs)
        clear_btn.click(
            do_clear,
            inputs=[session_id_st, server_url_st],
            outputs=[chatbot, msg_box],
        )
        new_btn.click(
            do_new_session,
            inputs=[],
            outputs=[chatbot, msg_box, session_id_st],
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
