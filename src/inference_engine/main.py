import argparse

from .engine import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Give me a short introduction to large language models.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    llm = LLM(args.model, device=args.device)
    result = llm.generate([{"role": "user", "content": args.prompt}], args.max_new_tokens)
    print(result["text"])


if __name__ == "__main__":
    main()
