import torch
import torch.nn.functional as F

@torch.no_grad()
def logprob_of_response(
    model, tokenizer,
    instruction: str,           # prompt x
    generated_text: str,        # list of model response y
    include_eos: bool = False
):

    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant"},
        {"role": "user", "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    for i in range(len(generated_text)):
        full_text = prompt_text + generated_text + (tokenizer.eos_token if include_eos and tokenizer.eos_token else "")

    enc_prompt = tokenizer(prompt_text, return_tensors="pt")
    enc_full   = tokenizer(full_text,  return_tensors="pt")

    enc_prompt = {k: v.to(model.device) for k, v in enc_prompt.items()}
    enc_full   = {k: v.to(model.device) for k, v in enc_full.items()}

    prompt_len = enc_prompt["input_ids"].shape[1]            # 프롬프트 토큰 수
    input_ids  = enc_full["input_ids"]                       # [1, L]
    attn_mask  = enc_full.get("attention_mask", None)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits                                   # [1, L, V]

    # 4) 다음 토큰 예측 정렬(shifting)
    # logits[:, t]가 예측하는 것은 input_ids[:, t+1]
    logits_next = logits[:, :-1, :]                           # [1, L-1, V]
    target_next = input_ids[:, 1:]                            # [1, L-1]

    # 5) y 영역만 슬라이스해서 로그확률 계산
    # 첫 y 토큰은 위치 prompt_len 에 있으므로, 그것을 예측한 로짓 인덱스는 prompt_len-1
    y_logits_slice  = logits_next[:, prompt_len-1:, :]        # [1, |y|, V]
    y_targets_slice = target_next[:, prompt_len-1:]           # [1, |y|]

    logprobs = F.log_softmax(y_logits_slice, dim=-1)          # [1, |y|, V]
    y_token_logprobs = logprobs.gather(2, y_targets_slice.unsqueeze(-1)).squeeze(-1)  # [1, |y|]

    # 6) 집계
    seq_logprob = y_token_logprobs.sum().item()               # log p(y|x)
    avg_logprob = y_token_logprobs.mean().item()              # 평균 토큰 로그확률
    nll = -avg_logprob                                        # 토큰당 NLL
    ppl = float(torch.exp(-y_token_logprobs.mean()))          # 퍼플렉서티


    return {
        "seq_logprob": seq_logprob,
        "avg_logprob": avg_logprob,
        "nll": nll,
        "ppl": ppl,
        "num_y_tokens": int(y_token_logprobs.shape[1]),
        # "y_token_logprobs": y_token_logprobs.squeeze(0).tolist(),  # 필요시 주석 해제
    }
