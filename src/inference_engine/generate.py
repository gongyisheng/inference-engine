import torch


@torch.inference_mode()
def greedy_generate(model, input_ids, max_new_tokens: int, eos_token_id):
    eos = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    generated = input_ids
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() in eos:
            break
    return generated
