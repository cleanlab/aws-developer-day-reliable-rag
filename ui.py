import os
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from constants import SCORE_TO_ISSUE
from knowledge_base import KnowledgeBase

if os.environ.get("USE_SOLUTION") == "baseline":
    from solutions.baseline_rag_solution import RAG
elif os.environ.get("USE_SOLUTION") == "custom_evals":
    from solutions.custom_evals_example import RAG  # type: ignore
elif os.environ.get("USE_SOLUTION"):
    from solutions.cleanlab_rag_solution import RAG  # type: ignore
else:
    from rag import RAG  # type: ignore


def main() -> None:
    load_dotenv()

    print("Loading knowledge base...")
    knowledge_base = KnowledgeBase.from_persisted(str(Path(__file__).parent / "vector_store/"))
    print("Knowledge base loaded")
    rag = RAG(knowledge_base)

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

    demo.launch(show_api=False)


if __name__ == "__main__":
    main()
