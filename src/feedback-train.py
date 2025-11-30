import json
from typing import Dict, List, Iterable, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import yaml
from pairwise import convert
from train import create_yaml, export_model, train_model

N_RESPONSES = 4                        # response count per instruction
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42                        # random seed
USE_FP16_BF16 = True                   # fp16 or bfloat16 if possible

_tokenizer = None
_model = None
rm = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = None

def load_model(model_path):
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
    
    _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        _model.to(_device)

from accelerate.hooks import remove_hook_from_module
import gc, torch

def _remove_accelerate_hooks(model):
    # 각 서브모듈에 붙은 _hf_hook 제거
    for m in model.modules():
        if getattr(m, "_hf_hook", None) is not None:
            remove_hook_from_module(m)

def unload_model():
    global _model, _tokenizer, rm

    # 생성 모델
    if _model is not None:
        # device_map='auto'일 때는 .to(...) 금지 → 훅 제거 후 삭제
        if any(getattr(m, "_hf_hook", None) is not None for m in _model.modules()):
            _remove_accelerate_hooks(_model)
        del _model
        _model = None

    # 토크나이저
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    # RM 모델
    if rm is not None:
        if any(getattr(m, "_hf_hook", None) is not None for m in rm.model.modules()):
            _remove_accelerate_hooks(rm.model)
        del rm.model
        del rm.tokenizer
        rm = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def hf_infer(
    instruction: str,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    seed: Optional[int] = None,
) -> str:
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

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score

def resumed_dataset(cfg, dataset, loop_cnt):
    processed_instructions = set()
    for i in range(loop_cnt):
        feedback_file = f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl"
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    processed_instructions.add(item["instruction"])
        except FileNotFoundError:
            continue

    new_dataset = []
    for item in dataset:
        if item["instruction"] not in processed_instructions:
            new_dataset.append(item)

    return new_dataset

def get_armo(cfg, model_name, dataset):
    global rm

    print(f"Processing {len(dataset)} items...")
    if cfg.resume == False:
        loop_cnt = 0
        while True:
            # RM 로딩 시
            rm = ArmoRMPipeline(
                "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                device_map={"": "cuda:0"},  # 단일 GPU 고정
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

            cnt, dataset= loop(cfg, rm, dataset, loop_cnt)
            print(f"Low scores detected: {cnt} items need feedback.")

            train(model_name, cfg, loop_cnt)
            loop_cnt += 1
            if cnt == 0:
                break
    if cfg.resume == True:
        loop_cnt = cfg.resume_step
        dataset = resumed_dataset(cfg, dataset, loop_cnt)
        if cfg.resume_point == "armorm":
            # RM 로딩 시
            rm = ArmoRMPipeline(
                "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                device_map={"": "cuda:0"},  # 단일 GPU 고정
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

            cnt, dataset= loop(cfg, rm, dataset, loop_cnt)
            print(f"Low scores detected: {cnt} items need feedback.")
            train(model_name, cfg, loop_cnt)
            loop_cnt += 1
        if cfg.resume_point == "train":
            train(model_name, cfg, loop_cnt)
            loop_cnt += 1
            
        while True:
            # RM 로딩 시
            rm = ArmoRMPipeline(
                "RLHFlow/ArmoRM-Llama3-8B-v0.1",
                device_map={"": "cuda:0"},  # 단일 GPU 고정
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

            cnt, dataset= loop(cfg, rm, dataset, loop_cnt)
            print(f"Low scores detected: {cnt} items need feedback.")
            train(model_name, cfg, loop_cnt)
            loop_cnt += 1
            if cnt == 0:
                break

def train(model_name, cfg, loop_cnt):
    input_path = f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl"
    cnt, file_path = convert(cfg, input_path)
    data_name = f"ultrafeedback-{loop_cnt}-{cfg.exp_num}"
    # Update dataset_info.json
    dataset_info_path = "./LLaMA-Factory/data/dataset_info.json"
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    dataset_info[data_name] = {
        "file_name": file_path,
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    global template
    with open(f"./LLaMA-Factory/{cfg.yaml_path}/template.yaml") as f:
        template = yaml.safe_load(f)

    global exp_template
    with open(f"./LLaMA-Factory/{cfg.export_yaml_path}/template.yaml") as f:
        exp_template = yaml.safe_load(f)
    
    unload_model()

    create_yaml(cfg, template, data_name, loop_cnt)
    if loop_cnt != 0:
        train_model(cfg, data_name, loop_cnt)
    model_name = export_model(cfg, exp_template, data_name, loop_cnt)
    
    print(f"Training completed. Model loading: {model_name}")
    load_model(model_name)


def refine(item, loop_cnt, cfg):
    feedback, refined_instruction = refine_instruction(item)
    refine_obj = {
        "instruction": item["instruction"],
        "feedback": feedback,
        "refined_instruction": refined_instruction,
        "loop_cnt": loop_cnt,
    }
    with open(f"{cfg.feedback_path}/refine.jsonl", 'a', encoding='utf-8') as f:
                json.dump(refine_obj, f, ensure_ascii=False)
                f.write('\n')
                
    item['instruction'] = refined_instruction
    responses: List[str] = []
    for k in range(N_RESPONSES):
        seed = SEEDS_BASE + k
        try:
            resp = hf_infer(instruction=refined_instruction, seed=seed)
        except Exception as e:
            resp = f"[ERROR] {type(e).__name__}: {e}"
        responses.append(resp)
    out_obj = {
        "instruction": refined_instruction,
        "responses": responses,
    }
    return out_obj

def refine_instruction(item):
    prompt = item["instruction"]

    evaluation_template = f"""You are an evaluator. Given an Original Instruction,
evaluate the instruction using the criteria below. 
Follow these STRICT rules:
1. Output must start with exactly 'Evaluation:' on its own line.
2. You must include ALL 7 criteria in the following order: 
   Clarity, Specificity, Completeness, Safety, Answerability, Conciseness, FormatConsistency.
3. Each line must follow the format:
   * <Criterion>: <digit 1-5>/5 - <one concise note>
4. Do NOT add any text before or after the evaluation block.

Output format:
Evaluation:
* Clarity: <1-5>/5 - <one-line note>
* Specificity: <1-5>/5 - <one-line note>
* Completeness: <1-5>/5 - <one-line note>
* Safety: <1-5>/5 - <one-line note>
* Answerability: <1-5>/5 - <one-line note>
* Conciseness: <1-5>/5 - <one-line note>
* FormatConsistency: <1-5>/5 - <one-line note>

---

### Few-shot Examples

Original Instruction:
Write something about animals

Evaluation:
* Clarity: 2/5 - vague request
* Specificity: 2/5 - no length, no scope
* Completeness: 2/5 - missing output format
* Safety: 5/5 - safe
* Answerability: 4/5 - feasible but broad
* Conciseness: 3/5 - some redundancy
* FormatConsistency: 3/5 - unspecified output format

###

Original Instruction:
Make a paragraph using that language

Evaluation:
* Clarity: 2/5 - unclear what "that language" refers to
* Specificity: 2/5 - format is defined (paragraph) but content is vague
* Completeness: 2/5 - missing target language or subject
* Safety: 5/5 - safe request
* Answerability: 2/5 - partially answerable but underspecified
* Conciseness: 4/5 - concise but incomplete
* FormatConsistency: 3/5 - loosely consistent but ambiguous wording

###

Original Instruction:
{prompt}
"""
    feedback = hf_infer(evaluation_template)

    refinement = f"""You are an instruction refiner. 
Given an Original Instruction and its Evaluation, rewrite the instruction so it is clear, specific, complete, safe, feasible, concise, and consistent.

STRICT RULES:
1. Output ONLY the refined instruction text — no labels, no commentary, no examples.
2. The refined instruction must be executable directly by a model (not guidelines).
3. Preserve the intent of the original, but resolve ambiguity using the feedback.
4. Do not include the words "Refined Instruction" in the output.
5. Write a single instruction only.

Original Instruction:
{prompt}

Evaluation:
{feedback}
"""
    return feedback, hf_infer(refinement)

def loop(cfg, rm, dataset, loop_cnt):
    cnt = 0
    new_dataset = []
    for item in tqdm(dataset, desc="Processing dataset"):
        # set response data
        similarity_scores = []

        prompt = item["instruction"]
        for j in range(len(item["responses"])):
            generated_response = item["responses"][j]
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_response}])
            similarity_scores.append(score)

        item['scores'] = similarity_scores

        if max(similarity_scores) < cfg.gamma:  # threshold
            print(f"Low scores detected: {similarity_scores}")
            cnt += 1
            item = refine(item, loop_cnt, cfg)
            new_dataset.append(item)

        else:
            with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    return cnt, new_dataset

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.inference_path, split="train")

    if cfg.model == "llama3-8b":
        model_name = "princeton-nlp/Llama-3-Base-8B-SFT"
    
    load_model(model_name)
    get_armo(cfg, model_name, ds)
    print(f"Data Processing completed.")

if __name__ == "__main__":
    main()