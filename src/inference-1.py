# run_infer_hf_llama3_jsonl.py
import os
import json
import time
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======  config  ======
INPUT_PATH = "./results/instruction-ultrafeedback10.jsonl"            # "instruction" jsonl
OUTPUT_PATH = "./results/inference-ultrafeedback10-1.jsonl"             # instruction + 4 responses jsonl
MODEL_ID = "princeton-nlp/Llama-3-Base-8B-SFT"                      # model id

N_RESPONSES = 2                        # response count per instruction
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42                        # random seed
USE_FP16_BF16 = True                   # fp16 or bfloat16 if possible

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def append_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")

def load_done_instructions(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                instr = rec.get("instruction")
                if instr is not None:
                    done.add(instr)
            except Exception:
                continue
    return done

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = None

def load_model():
    global _tokenizer, _model, _dtype
    if _tokenizer is not None and _model is not None:
        return
    if USE_FP16_BF16 and torch.cuda.is_available():
        # Llama3는 bfloat16 권장, 미지원 GPU면 float16
        if torch.cuda.is_bf16_supported():
            _dtype = torch.bfloat16
        else:
            _dtype = torch.float16
    else:
        _dtype = torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        _model.to(_device)

def hf_infer(
    instruction: str,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    seed: Optional[int] = None,
) -> str:
    load_model()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant"},
        {"role": "user", "content": instruction},
    ]
    prompt_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()

def main():
    done = load_done_instructions(OUTPUT_PATH)

    # 총 개수 파악은 선택
    try:
        total = sum(1 for _ in read_jsonl(INPUT_PATH))
    except Exception:
        total = 0

    total1 = int(total / 3)
    total2 = int((total / 3) * 2)
    total3 = total

    read_idx = 0
    saved = 0

    for rec in read_jsonl(INPUT_PATH):
        if read_idx == total1:
            break

        instr = rec.get("instruction")
        if not instr:
            continue
        if instr in done:
            continue

        responses: List[str] = []
        for k in range(N_RESPONSES):
            seed = SEEDS_BASE + k
            try:
                resp = hf_infer(instruction=instr, seed=seed)
            except Exception as e:
                resp = f"[ERROR] {type(e).__name__}: {e}"
            responses.append(resp)

        out_obj = {
            "instruction": instr,
            "responses": responses,
        }
        append_jsonl(OUTPUT_PATH, out_obj)
        saved += 1

        if total:
            print(f"{saved} saved  |  {read_idx}/{total} read")
        else:
            print(f"{saved} saved  |  {read_idx} read")
        read_idx += 1

    print("done")

if __name__ == "__main__":
    main()
