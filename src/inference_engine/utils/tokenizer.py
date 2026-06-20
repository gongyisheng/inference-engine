from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path)

    def apply_chat(self, messages, enable_thinking: bool = True):
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return self.tok(text, return_tensors="pt").input_ids

    def encode_chat(self, prompt: str, enable_thinking: bool = True):
        return self.apply_chat([{"role": "user", "content": prompt}], enable_thinking)

    def encode(self, text: str):
        return self.tok(text, return_tensors="pt").input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)
