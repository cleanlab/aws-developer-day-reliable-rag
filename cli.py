import os
import pprint

from dotenv import load_dotenv

import patch_aiohttp  # noqa: F401

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
