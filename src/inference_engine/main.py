import argparse

from .utils.weights import load_model
from .utils.tokenizer import Tokenizer
from .generate import greedy_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Give me a short introduction to large language models.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    model, cfg, path = load_model(args.model, device=args.device)
    tok = Tokenizer(path)
    ids = tok.encode_chat(args.prompt, enable_thinking=not args.no_thinking).to(args.device)
    out = greedy_generate(model, ids, args.max_new_tokens, cfg.eos_token_id)
    print(tok.decode(out[0, ids.shape[1]:]))


if __name__ == "__main__":
    main()
