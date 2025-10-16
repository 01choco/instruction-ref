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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def generate_response(cfg, prompt):  
    response = client.chat.completions.create(
        model=cfg.gpt_model,
        messages=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content

def find_golden_response (cfg, model, item):
    gr = generate_response(cfg, item["instruction"])
    rf_gr = generate_response(cfg, item["refined_instruction"])

    new_item = item.copy()
    new_item["golden_response"] = gr
    new_item["refined_golden_response"] = rf_gr

    similarity_scores = []
    for j in range(len(item["responses"])):
        embeddings = model.encode([gr, item["responses"][j]])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_scores.append(similarity)
    
    similarity_scores_list = [float(arr[0][0]) for arr in similarity_scores]
    new_item['similarity_scores'] = similarity_scores_list
    new_item['avg_similarity_score'] = sum(similarity_scores_list) / len(similarity_scores_list) if similarity_scores_list else 0

    refined_similarity_scores = []
    for j in range(len(item["responses"])):
        embeddings = model.encode([rf_gr, item["responses"][j]])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
        refined_similarity_scores.append(similarity)

    refined_similarity_scores_list = [float(arr[0][0]) for arr in refined_similarity_scores]
    new_item['refined_similarity_scores'] = refined_similarity_scores_list
    new_item['avg_refined_similarity_score'] = sum(refined_similarity_scores_list) / len(refined_similarity_scores_list) if refined_similarity_scores_list else 0

    return new_item


def yeval (cfg, dataset):
    output_file = cfg.output_dir
    model = SentenceTransformer('all-mpnet-base-v2')
    
    avg_similarity_scores = 0
    avg_refined_similarity_scores = 0

    for item in tqdm(dataset, desc="Processing dataset"):
        new_item = find_golden_response(cfg, model, item)

        avg_similarity_scores += sum(new_item['similarity_scores'])
        avg_refined_similarity_scores += sum(new_item['refined_similarity_scores'])

        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(new_item, f, ensure_ascii=False)
            f.write('\n')
    
    avg_similarity_scores /= (len(dataset)*4) if dataset else 1
    avg_refined_similarity_scores /= (len(dataset)*4) if dataset else 1

    print(f"Average Similarity Score: {avg_similarity_scores}")
    print(f"Average Refined Similarity Score: {avg_refined_similarity_scores}")

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.refined_path, split="train")

    if cfg.sampling and cfg.size > 0:
        sample_size = cfg.size
        ds = ds.shuffle(seed=cfg.seed).select(range(sample_size))
        print(f"Randomly sampled {sample_size} items from the dataset.")
    yeval(cfg, ds)

if __name__ == "__main__":
    main()
