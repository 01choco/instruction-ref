import json
import hydra
from datasets import load_dataset

def convert_data(input_file_name, train):
    file_path = f"/data/seohyeong/instruction-refinement/data/{input_file_name}-pairwise.jsonl"
    cnt = 0
    for i in range(len(train)):
        item = train[i]
        
        # set response data
        data_input = item["instruction"]
        generated_responses = item["responses"]
        scores = item["scores"]
        
        new_datas = []

        for j in range(len(generated_responses)):
            for k in range(j + 1, len(generated_responses)):
                sentence1 = generated_responses[j]
                sentence2 = generated_responses[k]
                score1 = scores[j]
                score2 = scores[k]

                if sentence1 == sentence2:
                    continue
                else:
                    if score1 > score2:
                        chosen = sentence1
                        rejected = sentence2
                    elif score1 < score2:
                        chosen = sentence2
                        rejected = sentence1
                    elif score1 == score2:
                        continue
                    new_data = {
                        "instruction": data_input,
                        "input": "",
                        "chosen": chosen,
                        "rejected": rejected
                    }
                    new_datas.append(new_data)
        
        for new_data in new_datas:
            with open(file_path, 'a') as file:
                json.dump(new_data, file)
                file.write('\n')
            print(f"{len(new_datas)} data appended. {i+1} / {len(train)} Data converted.")
        cnt += len(new_datas)
    return cnt, file_path

def convert(cfg, input_path):
    print(cfg)

    data = load_dataset("json", data_files=input_path)
    print("-----------------------------")
    print("Data Conversion started.")

    input_file_name = input_path.split("/")[-1]
    print(f"Processing file: {input_file_name}")
    input_file_name = input_file_name.replace(".jsonl", "")

    cnt, file_path = convert_data(input_file_name, data["train"])
    print(f"Data Processing completed. {cnt} data converted and appended to {file_path}.")

    return cnt, file_path