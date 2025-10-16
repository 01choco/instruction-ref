import json
from typing import Dict, List, Iterable, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from pairwise import convert
import os
from openai import OpenAI
from typing import Optional

# Custom variables 
STOPS = [
    "\n<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "\nuser", "User:", "<|user|>", "user\n\n"
]
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"                      # model id
N_RESPONSES = 2                        # response count per instruction
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42                        # random seed
USE_FP16_BF16 = True                   # fp16 or bfloat16 if possible

# OpenAI Config 
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = None
stop_token_ids_cpu: List[torch.Tensor] = []   # 토크나이즈 캐싱
stop_criteria: Optional[StoppingCriteriaList] = None

class StopOnIds(StoppingCriteria):
    def __init__(self, stop_ids_list: List[torch.Tensor]):
        self.stop_ids_list = stop_ids_list
        self.max_len = max((t.numel() for t in stop_ids_list), default=0)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs):
        cur = input_ids[0]
        tail = cur[-self.max_len:] if self.max_len > 0 else cur
        for s in self.stop_ids_list:
            s = s.to(tail.device)
            L = s.size(0)
            if L and tail.size(0) >= L and torch.equal(tail[-L:], s):
                return True
        return False

def load_model():
    global _tokenizer, _model, _dtype, stop_token_ids_cpu, stop_criteria
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
    
    stop_token_ids_cpu = []
    for s in STOPS:
        ids = _tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            stop_token_ids_cpu.append(torch.tensor(ids, dtype=torch.long))

    # 미리 criteria 구성(재사용)
    stop_criteria = StoppingCriteriaList([StopOnIds(stop_token_ids_cpu)])

# def hf_infer_conv(
#     instruction: str,
#     model_id: str,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_new_tokens: int = 512,
#     repetition_penalty: float = 1.0,
#     seed: Optional[int] = None,
# ) -> str:
#     # 모델과 tokenizer 로드
#     model, tokenizer = load_model(model_id)

#     # 시드 설정
#     if seed is not None:
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     # Conversation template 가져오기
#     conv = get_conversation_template(model_id)
#     conv.append_message(conv.roles[0], instruction)  # user message
#     conv.append_message(conv.roles[1], None)         # assistant placeholder

#     # prompt 생성
#     prompt_text = conv.get_prompt()
#     inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

#     # 생성
#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             max_new_tokens=max_new_tokens,
#             repetition_penalty=repetition_penalty,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     # user 입력 길이 이후부터 decoding
#     generated = output_ids[0][inputs["input_ids"].shape[1]:]
#     text = tokenizer.decode(generated, skip_special_tokens=True)

#     # template에서 제공하는 stop string 사용
#     if conv.stop_str:
#         if isinstance(conv.stop_str, list):
#             stop_indices = [text.find(s) for s in conv.stop_str if text.find(s) > 0]
#             if stop_indices:
#                 text = text[:min(stop_indices)]
#         elif text.find(conv.stop_str) > 0:
#             text = text[:text.find(conv.stop_str)]

#     return text.strip()

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

    out_ids = _model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
        stopping_criteria=stop_criteria,
    )
    # 사후 트림: stop 토큰 포함 전까지만 디코드
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    stops_dev = [t.to(gen_ids.device) for t in stop_token_ids_cpu]

    def first_stop_index(seq, stops):
        earliest = len(seq)
        for st in stops:
            L = len(st)
            # 슬라이딩 윈도우 검색
            for i in range(0, len(seq) - L + 1):
                if torch.equal(seq[i:i+L], st):
                    earliest = min(earliest, i)
                    break
        return earliest

    cut_idx = first_stop_index(gen_ids, stops_dev)
    if cut_idx < len(gen_ids):
        gen_ids = gen_ids[:cut_idx]

    text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text
    # inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)
    # with torch.no_grad():
    #     output_ids = _model.generate(
    #         **inputs,
    #         do_sample=True,
    #         temperature=temperature,
    #         top_p=top_p,
    #         max_new_tokens=max_new_tokens,
    #         repetition_penalty=repetition_penalty,
    #         eos_token_id=_tokenizer.eos_token_id,
    #         pad_token_id=_tokenizer.eos_token_id,
    #     )

    # generated = output_ids[0][inputs["input_ids"].shape[1]:]
    # text = _tokenizer.decode(generated, skip_special_tokens=True)
    # stop_str = "user\n\n"  # 정확히 template에서 나오는 구분자
    # stop_index = text.find(stop_str)
    # if stop_index >= 0:
    #     text = text[:stop_index].strip()
    # return text.strip()

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

def get_armo(cfg, dataset):
    rm = ArmoRMPipeline(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map={"": "cuda:0"},  # 단일 GPU 고정
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )    
    
    print(f"Processing {len(dataset)} items...")
    loop_cnt = 0
    while True:
        cnt, dataset= loop(cfg, rm, dataset, loop_cnt)
        print(f"Low scores detected: {cnt} items need feedback.")
        loop_cnt += 1
        if cnt == 0 or loop_cnt >= cfg.max_loop:
            with open(f"{cfg.feedback_path}/final_left.jsonl", 'w', encoding='utf-8') as f:
                for data in dataset:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')

            original_path = f"{cfg.feedback_path}/final_original.jsonl"
            refined_path = f"{cfg.feedback_path}/final_refined.jsonl"

            cnt = 0
            for i in range(loop_cnt):
                with open(f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        original_instruction = item['original_instruction'] if 'original_instruction' in item else item['instruction']
                        refined_instruction = item['instruction']
                        original_responses = item['original_responses'] if 'original_responses' in item else item['responses']
                        refined_responses = item['responses']

                        with open(original_path, 'a', encoding='utf-8') as of:
                            json.dump({
                                "instruction": original_instruction,
                                "responses": original_responses
                            }, of, ensure_ascii=False)
                            of.write('\n')
                        with open(refined_path, 'a', encoding='utf-8') as rf:
                            json.dump({
                                "instruction": refined_instruction,
                                "responses": refined_responses
                            }, rf, ensure_ascii=False)
                            rf.write('\n')
                        cnt += 1
                print(f"{i} step data processing complete.")

            print(f"Total {cnt} refined items processed.")

            original_left_path = f"{cfg.feedback_path}/final_original_left.jsonl"
            refined_left_path = f"{cfg.feedback_path}/final_refined_left.jsonl"

            cnt_left = 0
            with open(f"{cfg.feedback_path}/final_left.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    original_instruction = item['original_instruction'] if 'original_instruction' in item else item['instruction']
                    refined_instruction = item['instruction']
                    original_responses = item['original_responses'] if 'original_responses' in item else item['responses']
                    refined_responses = item['responses']

                    with open(original_left_path, 'a', encoding='utf-8') as of:
                        json.dump({
                            "instruction": original_instruction,
                            "responses": original_responses
                        }, of, ensure_ascii=False)
                        of.write('\n')
                    with open(refined_left_path, 'a', encoding='utf-8') as rf:
                        json.dump({
                            "instruction": refined_instruction,
                            "responses": refined_responses
                        }, rf, ensure_ascii=False)
                        rf.write('\n')
                    cnt_left += 1

            print(f"Total {cnt_left} left items processed.")
            break

def refine(item, loop_cnt, cfg):
    feedback, refined_instruction = refine_instruction(cfg, item)
                
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
        "feedback": feedback,
        "original_instruction": item["instruction"] if loop_cnt == 0 else item["original_instruction"],
        "original_responses": item["responses"] if loop_cnt == 0 else item["original_responses"],
        "refined_instruction": refined_instruction,
        "refined_responses": responses,
        "loop_cnt": loop_cnt,
    }
    
    with open(f"{cfg.feedback_path}/refine.jsonl", 'a', encoding='utf-8') as f:
                json.dump(refine_obj, f, ensure_ascii=False)
                f.write('\n')
    
    return out_obj

def generate_response_openai(cfg, prompt):
    resp = client.responses.create(
        model=cfg.gpt_model,  # "gpt-5" 등
        input=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": prompt}
        ],
        max_output_tokens=cfg.gpt_max_tokens,
    )
    return resp.output_text

def refine_instruction(cfg, item):
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
    feedback = generate_response_openai(cfg, evaluation_template)
    print(feedback)

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
    refined_instruction = generate_response_openai(cfg, refinement)
    print(refined_instruction)
    return feedback, refined_instruction


# def refine_instruction(item):
#     prompt = item["instruction"] #response도 주어질수도 
#     evaluation_templat
# e = f"""You are an evaluator. Given an Original Instruction,
# evaluate the instruction using the criteria below. 
# Follow these STRICT rules:
# 1. Output must start with exactly 'Evaluation:' on its own line.
# 2. You must include ALL 7 criteria in the following order: 
#    Clarity, Specificity, Completeness, Safety, Answerability, Conciseness, FormatConsistency.
# 3. Each line must follow the format:
#    * <Criterion>: <digit 1-5>/5 - <one concise note>
# 4. Do NOT add any text before or after the evaluation block.

# Output format:
# Evaluation:
# * Clarity: <1-5>/5 - <one-line note>
# * Specificity: <1-5>/5 - <one-line note>
# * Completeness: <1-5>/5 - <one-line note>
# * Safety: <1-5>/5 - <one-line note>
# * Answerability: <1-5>/5 - <one-line note>
# * Conciseness: <1-5>/5 - <one-line note>
# * FormatConsistency: <1-5>/5 - <one-line note>

# ---

# ### Few-shot Examples

# Original Instruction:
# Write something about animals

# Evaluation:
# * Clarity: 2/5 - vague request
# * Specificity: 2/5 - no length, no scope
# * Completeness: 2/5 - missing output format
# * Safety: 5/5 - safe
# * Answerability: 4/5 - feasible but broad
# * Conciseness: 3/5 - some redundancy
# * FormatConsistency: 3/5 - unspecified output format

# ###

# Original Instruction:
# Make a paragraph using that language

# Evaluation:
# * Clarity: 2/5 - unclear what "that language" refers to
# * Specificity: 2/5 - format is defined (paragraph) but content is vague
# * Completeness: 2/5 - missing target language or subject
# * Safety: 5/5 - safe request
# * Answerability: 2/5 - partially answerable but underspecified
# * Conciseness: 4/5 - concise but incomplete
# * FormatConsistency: 3/5 - loosely consistent but ambiguous wording

# ###

# Original Instruction:
# {prompt}

# """
#     feedback = hf_infer(evaluation_template)
#     refinement = f"""You are an instruction refiner. Given an Original Instruction and Feedback about instruction. 
#     Produce a Refined Instruction that is clear, specific, complete, safe, feasible, concise, 
#     and follows a consistent template. The output must contain only the refined instruction itself. 
#     Output must contain only the refined instruction, nothing else.

# Output format:
# Refined Instruction:
# <improved instruction the model can follow directly>

# ---

# ### Few-shot Examples

# Original Instruction:
# Write something about animals

# Evaluation:
# * Clarity: 2/5 - vague request
# * Specificity: 2/5 - no length, no scope
# * Completeness: 2/5 - missing output format
# * Safety: 5/5 - safe
# * Answerability: 4/5 - feasible but broad
# * Conciseness: 3/5 - some redundancy
# * FormatConsistency: 3/5 - unspecified output format

# Refined Instruction:
# Write a 150-word informative paragraph about the habitat and diet of horses in a clear academic style

# ###

# Original Instruction:
# Summarize the article

# Evaluation:
# * Clarity: 3/5 - generic but understandable
# * Specificity: 2/5 - no target length or audience
# * Completeness: 3/5 - lacks citation or section guidance
# * Safety: 5/5 - safe
# * Answerability: 5/5 - fully feasible
# * Conciseness: 4/5 - short but fine
# * FormatConsistency: 3/5 - no output structure

# Refined Instruction:
# Summarize the article in 5 bullet points for a general audience, each ≤20 words, preserving key facts and names

# ###

# Original Instruction:
# {prompt}

# Evaluation:
# {feedback}

# """
#     return feedback, hf_infer(refinement)


def loop(cfg, rm, dataset, loop_cnt):
    cnt = 0
    new_dataset = []
    for item in tqdm(dataset, desc="Processing dataset"):
        
        # set response data
        similarity_scores = []

        prompt = item["instruction"]
        if len(item["responses"]) > N_RESPONSES:
            item["responses"] = item["responses"][:N_RESPONSES]
        
        for j in range(len(item["responses"])):
            generated_response = item["responses"][j]
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_response}])
            similarity_scores.append(score)

        item['scores'] = similarity_scores

        if cfg.threshold == "max":
            if max(similarity_scores) < cfg.gamma:  # threshold
                print(f"Low scores detected: {similarity_scores}")
                cnt += 1
                item = refine(item, loop_cnt, cfg)
                new_dataset.append(item)
            else:
                with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        elif cfg.threshold == "mean":
            if sum(similarity_scores) / len(similarity_scores) < cfg.gamma:  # threshold
                print(f"Low scores detected: {similarity_scores}")
                cnt += 1
                item = refine(item, loop_cnt, cfg)
                new_dataset.append(item)
            else:
                with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        elif cfg.threshold == "min":
            if min(similarity_scores) < cfg.gamma:  # threshold
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

    os.makedirs(cfg.feedback_path, exist_ok=True)
    ds = load_dataset("json", data_files=cfg.inference_path, split="train")
    load_model()
    get_armo(cfg, ds)
    print(f"Data Processing completed.")

if __name__ == "__main__":
    main()