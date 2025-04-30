import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv

import patch_aiohttp  # noqa: F401
from constants import SCORE_TO_ISSUE

USE_SOLUTION = os.environ.get("USE_SOLUTION")
if USE_SOLUTION is not None:
    if USE_SOLUTION in {"1", "2", "3", "4"}:
        import importlib
        solution = importlib.import_module(f"solutions.part{USE_SOLUTION}")
        RAG = solution.RAG
    else:
        msg = f"Invalid USE_SOLUTION value: {USE_SOLUTION}. Expected '1', '2', '3', or '4'."
        raise ValueError(msg)
else:
    from rag import RAG


def main() -> None:
    load_dotenv()

    rag = RAG()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RAG Chat Interface")
        gr.Markdown("This application uses RAG (Retrieval-Augmented Generation) to answer your questions.")

        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Ask a question...", show_label=False, submit_btn=True)

        def user_input(message: str, history: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
            return "", [{"role": "user", "content": message}]

        def bot_response(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
            message = history[-1]["content"]
            assert isinstance(message, str)
            response_data = rag.query(message)

            bot_message = response_data["response"]
            history.append({"role": "assistant", "content": bot_message})

            if response_data.get("is_expert_answer"):
                history.append(
                    {
                        "role": "assistant",
                        "content": "(expert answers have no evals)",
                        "metadata": {"title": "\u2705 Expert answer"},
                    }
                )
            else:
                if response_data.get("is_bad_response"):
                    issue_names = [
                        SCORE_TO_ISSUE.get(eval["name"], eval["name"])
                        for eval in response_data.get("evals", [])
                        if eval["is_bad"]
                    ]
                    title = f"\u2757 Issues detected: {', '.join(issue_names)}"
                else:
                    title = "\u2705 No issues detected"
                evals = [f"{eval['name']}: {eval['score']:.3f}" for eval in response_data.get("evals", [])]
                content = f"Evals:\n\n{'\n'.join(evals)}"
                history.append({"role": "assistant", "content": content, "metadata": {"title": title}})

            return history

        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(bot_response, chatbot, chatbot)

    demo.launch(show_api=False, server_port=8080)


if __name__ == "__main__":
    main()
