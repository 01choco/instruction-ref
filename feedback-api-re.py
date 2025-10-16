import json
from typing import Dict, List, Iterable, Optional, Tuple
import os
import asyncio
from asyncio import Semaphore
from collections import deque

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datasets import load_dataset

from openai import AsyncOpenAI

# =========================
# Config-like constants
# =========================
MODEL_ID = "princeton-nlp/Llama-3-Base-8B-SFT"    # local HF generation model
N_RESPONSES = 2
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42
USE_FP16_BF16 = True

# =========================
# Globals for HF model
# =========================
_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = None

# =========================
# OpenAI client (async)
# =========================
api_key = os.getenv("OPENAI_API_KEY")
aclient = AsyncOpenAI(api_key=api_key)
# 동시성(레이트리밋/인프라 상황에 맞게 조절)
OPENAI_CONCURRENCY = 5
_openai_sem = Semaphore(OPENAI_CONCURRENCY)

# =========================
# 1) HF Local Generation Acceleration
#    - flash_attention_2
#    - bf16/fp16
#    - TF32 허용
#    - inference_mode
# =========================
def load_model():
    global _tokenizer, _model, _dtype
    if _tokenizer is not None and _model is not None:
        return

    # Allow fast math where safe
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if USE_FP16_BF16 and torch.cuda.is_available():
        _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        _dtype = torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",  # 핵심 가속
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
    stop: Optional[List[str]] = None,
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
    prompt_text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)

    gen_kwargs = dict(
        **inputs,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
    )
    if temperature <= 0 or top_p <= 0:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    # 간단한 stop 지원 (토크나이저 레벨로 충분치 않을 수 있음)
    # 필요 시 커스텀 StoppingCriteria 구현 가능
    with torch.inference_mode():
        output_ids = _model.generate(**gen_kwargs)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()

# =========================
# 2) RM(ArmoRM) 배치화
# =========================
class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16,
                 truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def _encode_pair(self, user: str, assistant: str) -> torch.Tensor:
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        ids = self.tokenizer.apply_chat_template(
            msgs,
            return_tensors="pt",
            add_generation_prompt=False,
            truncation=self.truncation,
            max_length=self.max_length,
        )[0]
        return ids

    def score_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        # 길이에 따라 정렬 → 패딩 손실 최소화
        encs = [(i, self._encode_pair(u, a)) for i, (u, a) in enumerate(pairs)]
        encs.sort(key=lambda x: x[1].shape[0])
        scores = [0.0] * len(pairs)

        with torch.inference_mode():
            for i in range(0, len(encs), batch_size):
                chunk = encs[i:i+batch_size]
                ids_list = [t[1] for t in chunk]
                maxlen = min(self.max_length, max(t.shape[0] for t in ids_list))
                padded = torch.nn.utils.rnn.pad_sequence(
                    [t[-maxlen:] for t in ids_list],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ).to(self.device)
                out = self.model(padded)
                # 모델이 .score (shape: [B])를 제공한다고 가정
                batch_scores = out.score.float().tolist()
                for (orig_idx, _), s in zip(chunk, batch_scores):
                    scores[orig_idx] = float(s)
        return scores


from openai import AsyncOpenAI
aclient = AsyncOpenAI(api_key=api_key)

COMBINED_PROMPT = """You are an evaluator and refiner.
Given an Original Instruction, first produce a 7-line Evaluation block in the exact format,
then produce a single-line refined instruction that is directly executable.

STRICT:
1) Output JSON with keys: "evaluation_lines" (array of 7 strings), "refined" (string)
2) Keep the evaluation criteria order exactly:
   Clarity, Specificity, Completeness, Safety, Answerability, Conciseness, FormatConsistency
3) The refined must be a single instruction, preserving the intent but resolving ambiguity.

Original Instruction:
{prompt}
"""

# --- generate_both_openai_async 간단 보강 ---
async def generate_both_openai_async(cfg, prompt: str):
    async with _openai_sem:
        resp = await aclient.chat.completions.create(
            model=cfg.gpt_model,
            messages=[
                {"role": "system", "content": "Return only valid JSON with keys: evaluation_lines (7 strings), refined (string). No code fences."},
                {"role": "user", "content": COMBINED_PROMPT.format(prompt=prompt)},
            ],
            # 구버전 SDK면 아래 줄 없어도 됨
            response_format={"type": "json_object"},
            max_tokens=cfg.gpt_max_tokens,
            temperature=0,
        )
        raw = resp.choices[0].message.content
        data = parse_json_relaxed(raw)

        # 마지막 1회만 간단 재시도
        if data is None:
            retry_msg = f"""Your previous output was not valid JSON.
Return ONLY JSON with:
- evaluation_lines: array of exactly 7 strings
- refined: string

Do not include any extra text:
{raw}"""
            resp2 = await aclient.chat.completions.create(
                model=cfg.gpt_model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON. No code fences, no explanations."},
                    {"role": "user", "content": retry_msg},
                ],
                # 구버전 SDK면 이 줄 제거 가능
                response_format={"type": "json_object"},
                max_tokens=cfg.gpt_max_tokens,
                temperature=0,
            )
            raw2 = resp2.choices[0].message.content
            data = parse_json_relaxed(raw2)

        if data is None:
            raise ValueError("LLM output is not valid JSON")

        evaluation = "Evaluation:\n" + "\n".join(data["evaluation_lines"])
        refined = data["refined"].strip()
        return evaluation, refined

# =========================
# 4) 파일 I/O 버퍼링 유틸
# =========================
class JsonlBufferedWriter:
    def __init__(self, path: str, buffer_size: int = 1000):
        self.path = path
        self.buffer = deque()
        self.buffer_size = buffer_size

    def write(self, obj: dict):
        self.buffer.append(obj)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            while self.buffer:
                x = self.buffer.popleft()
                json.dump(x, f, ensure_ascii=False)
                f.write("\n")

# =========================
# Pipeline functions
# =========================
def refine_with_hf_and_openai(cfg, rm: ArmoRMPipeline, item: dict,
                              loop_cnt: int) -> dict:
    """
    동기 함수: 내부에서 OpenAI 부분만 필요시 asyncio.run으로 호출
    (대규모에서는 상위 루프 자체를 async로 구성하는 게 더 좋지만,
     기존 hydra main과의 호환을 위해 여기서 감쌈)
    """
    # OpenAI 1회 호출로 평가+리파인 획득
    evaluation, refined_instruction = asyncio.run(generate_both_openai_async(cfg, item["instruction"]))

    # 로컬 HF로 N_RESPONSES 생성
    responses: List[str] = []
    for k in range(N_RESPONSES):
        seed = SEEDS_BASE + k
        try:
            resp = hf_infer(instruction=refined_instruction, seed=seed)
        except Exception as e:
            resp = f"[ERROR] {type(e).__name__}: {e}"
        responses.append(resp)

    out_obj = {
        "original_instruction": item["instruction"] if loop_cnt == 0 else item["original_instruction"],
        "original_responses": item["responses"] if loop_cnt == 0 else item["original_responses"],
        "instruction": refined_instruction,
        "responses": responses,
    }

    refine_obj = {
        "instruction": item["instruction"],
        "responses": item["responses"],
        "feedback": evaluation,
        "original_instruction": item["instruction"] if loop_cnt == 0 else item["original_instruction"],
        "original_responses": item["responses"] if loop_cnt == 0 else item["original_responses"],
        "refined_instruction": refined_instruction,
        "refined_responses": responses,
        "loop_cnt": loop_cnt,
    }

    return out_obj, refine_obj

def loop(cfg, rm: ArmoRMPipeline, dataset, loop_cnt: int):
    cnt = 0
    new_dataset = []

    # 파일 버퍼 준비
    ensure_dir(cfg.feedback_path)
    ultra_path = f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl"
    refine_path = f"{cfg.feedback_path}/refine.jsonl"

    ultra_writer = JsonlBufferedWriter(ultra_path, buffer_size=1000)
    refine_writer = JsonlBufferedWriter(refine_path, buffer_size=1000)

    for item in tqdm(dataset, desc=f"Processing dataset (loop {loop_cnt})"):
        prompt = item["instruction"]

        # responses 수 제한
        if len(item["responses"]) > N_RESPONSES:
            item["responses"] = item["responses"][:N_RESPONSES]

        # ---- 2) RM 배치화 활용: 한 아이템 내 응답들을 한 번에 ----
        pairs = [(prompt, r) for r in item["responses"]]
        similarity_scores = rm.score_batch(pairs, batch_size=16)
        item["scores"] = similarity_scores

        # ---- 임계 방식별 판단 ----
        trigger = False
        if cfg.threshold == "max":
            trigger = max(similarity_scores) < cfg.gamma
        elif cfg.threshold == "mean":
            trigger = (sum(similarity_scores) / len(similarity_scores)) < cfg.gamma
        elif cfg.threshold == "min":
            trigger = min(similarity_scores) < cfg.gamma

        if trigger:
            cnt += 1
            out_obj, refine_obj = refine_with_hf_and_openai(cfg, rm, item, loop_cnt)
            new_dataset.append(out_obj)
            refine_writer.write(refine_obj)
        else:
            ultra_writer.write(item)

    # 남은 버퍼 flush
    ultra_writer.flush()
    refine_writer.flush()

    return cnt, new_dataset

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_armo(cfg, dataset):
    rm = ArmoRMPipeline(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map={"": "cuda:0"} if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )

    print(f"Processing {len(dataset)} items...")
    loop_cnt = 0
    while True:
        cnt, dataset = loop(cfg, rm, dataset, loop_cnt)
        print(f"Low scores detected: {cnt} items need feedback.")
        loop_cnt += 1
        if cnt == 0 or loop_cnt >= cfg.max_loop:
            # 최종 병합/정리 단계 - 기존 로직 유지, 단지 버퍼링 없이 한 번에 처리
            final_left_path = f"{cfg.feedback_path}/final_left.jsonl"
            with open(final_left_path, 'w', encoding='utf-8') as f:
                for data in dataset:
                    json.dump(data, f, ensure_ascii=False); f.write('\n')

            original_path = f"{cfg.feedback_path}/final_original.jsonl"
            refined_path = f"{cfg.feedback_path}/final_refined.jsonl"
            # 초기화
            open(original_path, 'w', encoding='utf-8').close()
            open(refined_path, 'w', encoding='utf-8').close()

            cnt_total = 0
            for i in range(loop_cnt):
                path = f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl"
                if not os.path.exists(path):
                    continue
                with open(path, 'r', encoding='utf-8') as fin, \
                     open(original_path, 'a', encoding='utf-8') as of, \
                     open(refined_path, 'a', encoding='utf-8') as rf:
                    for line in fin:
                        item = json.loads(line.strip())
                        original_instruction = item.get('original_instruction', item['instruction'])
                        refined_instruction = item['instruction']
                        original_responses = item.get('original_responses', item['responses'])
                        refined_responses = item['responses']

                        json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False); of.write('\n')
                        json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False); rf.write('\n')
                        cnt_total += 1
                print(f"{i} step data processing complete.")

            print(f"Total {cnt_total} refined items processed.")

            original_left_path = f"{cfg.feedback_path}/final_original_left.jsonl"
            refined_left_path = f"{cfg.feedback_path}/final_refined_left.jsonl"
            # 초기화
            open(original_left_path, 'w', encoding='utf-8').close()
            open(refined_left_path, 'w', encoding='utf-8').close()

            cnt_left = 0
            with open(final_left_path, 'r', encoding='utf-8') as fin, \
                 open(original_left_path, 'a', encoding='utf-8') as of, \
                 open(refined_left_path, 'a', encoding='utf-8') as rf:
                for line in fin:
                    item = json.loads(line.strip())
                    original_instruction = item.get('original_instruction', item['instruction'])
                    refined_instruction = item['instruction']
                    original_responses = item.get('original_responses', item['responses'])
                    refined_responses = item['responses']

                    json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False); of.write('\n')
                    json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False); rf.write('\n')
                    cnt_left += 1

            print(f"Total {cnt_left} left items processed.")
            break

# =========================
# Hydra entry
# =========================
@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")
    ds = load_dataset("json", data_files=cfg.inference_path, split="train")
    load_model()
    get_armo(cfg, ds)
    print("Data Processing completed.")

if __name__ == "__main__":
    main()
