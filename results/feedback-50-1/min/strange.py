import json

input_files = ['./ultrafeedback-0.jsonl', './ultrafeedback-1.jsonl', './ultrafeedback-2.jsonl']

def main():
    evil_instructions = [item["instruction"] for item in json.load(open('/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-1/min/inference-ultrafeedback50-error.jsonl', 'r', encoding='utf-8'))]
    i = 0
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line)
                if input_file == './ultrafeedback-0.jsonl' and data["instruction"] in evil_instructions:
                    i += 1
                    print(f"error detected! {i}th error prompt")
                elif input_file != './ultrafeedback-0.jsonl' and data["original_instruction"] in evil_instructions:
                    i += 1
                    print(f"error detected! {i}th error prompt")
                