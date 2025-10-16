import json
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import sys
from datasets import load_dataset

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
    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    print(f"Processing {len(dataset)} items...")
    for item in tqdm(dataset, desc="Processing dataset"):
        # set response data
        similarity_scores = []

        prompt = item["instruction"]
        for j in range(len(item["responses"])):
            generated_response = item["responses"][j]
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_response}])
            similarity_scores.append(score)

        item['scores'] = similarity_scores

        with open(f"{cfg.armorm_path}", 'a', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.inference_path, split="train")
    cnt = get_armo(cfg, ds)
    print(f"Data Processing completed. {cnt} data converted and appended.")

if __name__ == "__main__":
    main()