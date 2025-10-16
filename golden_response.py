import json
import re
from typing import Dict
from datasets import load_dataset
from omegaconf import DictConfig
import hydra
from openai import OpenAI
import os
import csv
from tqdm import tqdm

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def find_golden_response (item):
    full_dataset 

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.refined_path, split="train")
    ultrafeedback = load_dataset("json", data_files=cfg.ultrafeedback_path, split="train")
    find_golden_response(ds)

if __name__ == "__main__":
    main()
