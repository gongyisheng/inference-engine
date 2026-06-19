import torch

from .cache import KVCache


@torch.inference_mode()
def greedy_generate(model, input_ids, max_new_tokens: int, eos_token_id):
    eos = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    cache = KVCache(model.cfg.num_hidden_layers)
    generated = input_ids
    cur = input_ids  # prefill the whole prompt first, then one token at a time
    for _ in range(max_new_tokens):
        logits = model(cur, cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() in eos:
            break
        cur = next_token
    return generated
