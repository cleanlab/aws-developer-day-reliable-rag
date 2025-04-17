import os
import pprint
from pathlib import Path

from dotenv import load_dotenv

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
    print()
    try:
        while True:
            message = input("Query: ")
            if not message:
                break
            response = rag.query(message)
            print()
            pprint.pp(response)
            print(f"\n{'-' * 40}", end="\n\n")
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()
