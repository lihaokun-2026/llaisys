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
# 特殊 token 过滤（Qwen / DeepSeek / ChatML 等模型）
# ─────────────────────────────────────────────────────────────────────────────

# 匹配 <|end_of_sentence|>  <|im_end|>  <|endoftext|> 等各类特殊 token
_SPECIAL_RE = re.compile(r"<\|[^|>]*\|>", re.IGNORECASE)

def _clean(text: str) -> str:
    """移除特殊 token，去掉首尾空白。"""
    return _SPECIAL_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# think 标签规范化
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_think(text: str) -> str:
    """
    若仅有 </think> 而无 <think>（<think> 被 tokenizer 作为 special token 跳过），
    在最前面补上 <think>，使 Gradio reasoning_tags 能正确渲染。
    """
    if "</think>" in text and "<think>" not in text:
        return "<think>" + text
    return text


def _strip_think(text: str) -> str:
    """去掉思考块，仅保留最终回答（构建多轮 API 上下文时使用）。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = re.sub(r"<think>.*$",         "", t,    flags=re.DOTALL)
    return _clean(t)


# ─────────────────────────────────────────────────────────────────────────────
# API 消息构建
# ─────────────────────────────────────────────────────────────────────────────

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
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* === 全局 ================================================================ */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
}

/* === 标题渐变卡片 ========================================================= */
#llaisys-header {
    background: linear-gradient(120deg, #0d9488 0%, #0e7490 55%, #1e40af 100%);
    border-radius: 16px !important;
    padding: 22px 28px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 6px 30px rgba(13, 148, 136, 0.28) !important;
}
#llaisys-header h1 {
    color: #ffffff !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    margin: 0 0 5px 0 !important;
    letter-spacing: -0.4px !important;
    line-height: 1.2 !important;
}
#llaisys-header p {
    color: rgba(255, 255, 255, 0.75) !important;
    font-size: 0.84rem !important;
    margin: 0 !important;
}

/* === 聊天列容器 =========================================================== */
#chat-col {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 18px rgba(0, 0, 0, 0.07) !important;
    padding: 0 !important;
}

/* === 气泡圆角 ============================================================= */
.message-bubble-border { border-radius: 14px !important; }

/* === 输入区底部背景 ======================================================= */
#input-area {
    background: #f8fafc !important;
    border-top: 1px solid #e8edf4 !important;
    padding: 10px 14px 6px 14px !important;
}
#action-row {
    background: #f8fafc !important;
    padding: 4px 14px 12px 14px !important;
}

/* === 文本输入框 =========================================================== */
#msg-input textarea {
    border-radius: 10px !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #ffffff !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.13) !important;
    outline: none !important;
}

/* === 发送按钮 ============================================================= */
#send-btn button {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 3px 10px rgba(13, 148, 136, 0.30) !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease !important;
}
#send-btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(13, 148, 136, 0.42) !important;
}
#send-btn button:active { transform: translateY(0) !important; }

/* === 清空 / 新对话按钮 ==================================================== */
#clear-btn button {
    border-radius: 8px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fff !important;
    color: #64748b !important;
    font-size: 0.84rem !important;
    transition: border-color 0.14s, color 0.14s, background 0.14s !important;
}
#clear-btn button:hover {
    border-color: #fca5a5 !important;
    color: #dc2626 !important;
    background: #fff5f5 !important;
}
#new-btn button {
    border-radius: 8px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fff !important;
    color: #64748b !important;
    font-size: 0.84rem !important;
    transition: border-color 0.14s, color 0.14s, background 0.14s !important;
}
#new-btn button:hover {
    border-color: #5eead4 !important;
    color: #0d9488 !important;
    background: #f0fdfa !important;
}

/* === 参数面板 ============================================================= */
#param-panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 18px 16px !important;
    box-shadow: 0 2px 18px rgba(0, 0, 0, 0.07) !important;
}

/* === Gradio 6 内置 Reasoning 块（思考块）==================================== */
/* 正在思考中（展开）*/
.thinking[open],
.thinking {
    background: linear-gradient(135deg, #f0fdf9 0%, #ecfeff 100%) !important;
    border: 1px solid #a7f3d0 !important;
    border-left: 3px solid #0d9488 !important;
    border-radius: 10px !important;
    margin: 4px 0 8px 0 !important;
    overflow: hidden !important;
    font-size: 13px !important;
}
/* summary 行 */
.thinking > summary {
    padding: 8px 12px !important;
    color: #0f766e !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    user-select: none !important;
    list-style: none !important;
    background: rgba(13, 148, 136, 0.06) !important;
}
.thinking > summary:hover { background: rgba(13, 148, 136, 0.10) !important; }
/* 内容区 */
.thinking > div,
.thinking > p {
    padding: 8px 12px 10px !important;
    color: #134e4a !important;
    line-height: 1.65 !important;
    border-top: 1px solid #ccfbf1 !important;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 流式生成
# ─────────────────────────────────────────────────────────────────────────────

def respond(
    user_msg: str,
    history: list,
    session_id: str,
    server_url: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
):
    """流式生成器：每收到增量 token 就 yield 更新后的 history。"""
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
            stream=True,
            timeout=300,
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
                        # 过滤特殊 token，规范 think 标签
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


def do_clear(session_id: str, server_url: str):
    try:
        requests.post(f"{server_url}/v1/sessions/{session_id}/clear", timeout=5)
    except Exception:
        pass
    return [], ""


def do_new_session():
    return [], "", str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui(server_url: str) -> gr.Blocks:
    with gr.Blocks(title="LLAISYS Chat") as demo:

        session_id_st = gr.State(str(uuid.uuid4()))
        server_url_st = gr.State(server_url)

        # ── 标题卡片 ────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🤖 LLAISYS Chat\n\n"
            "高性能 LLM 推理框架 · 支持实时推理过程展示",
            elem_id="llaisys-header",
        )

        with gr.Row(equal_height=False):

            # ── 聊天列 ──────────────────────────────────────────────────────
            with gr.Column(scale=5, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=530,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                    # Gradio 6 内置：将 <think>…</think> 渲染为可折叠 Reasoning 块
                    reasoning_tags=[("<think>", "</think>")],
                    placeholder=(
                        "<div style='text-align:center;padding:70px 0 40px;'>"
                        "<div style='font-size:3rem;margin-bottom:12px'>💬</div>"
                        "<h4 style='color:#475569;font-weight:500;margin:0 0 8px'>"
                        "向 AI 提问吧</h4>"
                        "<p style='color:#94a3b8;font-size:13px;margin:0'>"
                        "模型的推理过程将以可折叠的思考块呈现</p>"
                        "</div>"
                    ),
                )

                # 输入区
                with gr.Row(elem_id="input-area"):
                    msg_box = gr.Textbox(
                        placeholder="输入消息… Enter 发送 / Shift+Enter 换行",
                        show_label=False,
                        lines=2,
                        max_lines=8,
                        scale=9,
                        container=False,
                        autofocus=True,
                        elem_id="msg-input",
                    )
                    send_btn = gr.Button(
                        "发送 ▶",
                        variant="primary",
                        scale=1,
                        min_width=90,
                        elem_id="send-btn",
                    )

                # 操作按钮行
                with gr.Row(elem_id="action-row"):
                    clear_btn = gr.Button(
                        "🗑  清空对话",
                        size="sm",
                        variant="secondary",
                        elem_id="clear-btn",
                    )
                    new_btn = gr.Button(
                        "➕  新对话",
                        size="sm",
                        variant="secondary",
                        elem_id="new-btn",
                    )

            # ── 参数面板 ────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=230, elem_id="param-panel"):
                gr.Markdown("### ⚙️  生成参数")

                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    label="温度 (Temperature)",
                    info="越高越随机，越低越保守",
                )
                top_k = gr.Slider(
                    minimum=1, maximum=200, value=50, step=1,
                    label="Top-K",
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P (核采样)",
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=4096, value=1024, step=64,
                    label="最大 Token 数",
                )

                gr.Markdown(
                    "<hr style='border-color:#e2e8f0;margin:14px 0'/>\n\n"
                    "### 💭  思考块\n\n"
                    "模型的推理过程自动渲染为\n"
                    "**可折叠的 Reasoning 块**，\n"
                    "点击即可展开 / 收起。\n\n"
                    "适用于 **DeepSeek-R1**、\n"
                    "**Qwen3** 等推理模型。"
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
    parser.add_argument("--server", default="http://localhost:8000",
                        help="FastAPI 后端地址（chat_server.py）")
    parser.add_argument("--host",   default="0.0.0.0",  help="监听地址")
    parser.add_argument("--port",   type=int, default=7860, help="监听端口")
    parser.add_argument("--share",  action="store_true",  help="生成 Gradio 公共链接")
    args = parser.parse_args()

    print(f"[ui] 连接后端: {args.server}")
    print(f"[ui] Gradio UI: http://localhost:{args.port}")
    demo = build_ui(args.server)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
