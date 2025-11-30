import json

# Input and output file paths
input_files = [
    "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/data/50-13-min/original_left.jsonl",
    "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/data/50-13-min/original.jsonl"
    # "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-2/min/yeval/min_original.jsonl",
    # "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-3/min/yeval/min_original.jsonl",
    # "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-4/min/yeval/min_original.jsonl",
    # "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-error/min/yeval/min_original.jsonl",
]
output_file = "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/data/50-13-min/original_entire.jsonl"

# Combine JSONL files
combined_data = []
for file in input_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_data.append(json.loads(line))

# Write combined data to output file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in combined_data:
        f.write(json.dumps(item) + '\n')

print(f"Combined {len(input_files)} files into {output_file}")
