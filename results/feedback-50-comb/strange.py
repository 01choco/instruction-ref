import json
import pandas as pd

# 입력 파일들
input_files = [
    './min_original.jsonl',
    './min_refined.jsonl',
    './min_original_left.jsonl',
    './min_refined_left.jsonl'
]

# refine 매핑 파일 (original ↔ refined 쌍이 들어있는 jsonl)
refine_path = '/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/feedback-50-comb/refine.jsonl'

def load_refine_pairs():
    """refine.jsonl에서 (instruction, original_instruction) 페어를 읽어서 DataFrame과 dict를 만든다."""
    data = {
        "instruction": [],
        "original_instruction": []
    }
    with open(refine_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data["instruction"].append(item["refined_instruction"])
            data["original_instruction"].append(item["original_instruction"])
    df = pd.DataFrame(data)
    # refined instruction -> original_instruction 매핑 dict
    inst2orig = dict(zip(df["instruction"], df["original_instruction"]))
    return df, inst2orig

def main():
    # 1) refine 매핑 로드
    refine_df, inst2orig = load_refine_pairs()
    print("refine pair 개수:", len(inst2orig))

    # 2) original 파일들에서 [ERROR] 포함된 샘플 찾기 + 필터링
    bad_original_insts = set()

    for file_name in input_files:
        if "original" not in file_name:
            continue

        filtered = []
        total = 0
        removed = 0

        with open(file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                total += 1
                data = json.loads(line)
                inst = data["instruction"]
                resp = data.get("responses")

                # response가 리스트이고, 그 안 문자열 중 하나라도 "[ERROR]"가 들어 있으면 제거 대상
                has_error = False
                if isinstance(resp, list):
                    for x in resp:
                        if isinstance(x, str) and "[ERROR]" in x:
                            has_error = True
                            break

                if has_error:
                    bad_original_insts.add(inst)
                    removed += 1
                    # 이 줄은 남기지 않음
                else:
                    filtered.append(data)

        print(f"{file_name} (original) 필터링 전 개수: {total}")
        print(f"{file_name} (original) 제거된 개수: {removed}")
        print(f"{file_name} (original) 필터링 후 개수: {len(filtered)}")

        out_name = f"./filtered_{file_name.split('/')[-1]}"
        with open(out_name, 'w', encoding='utf-8') as outfile:
            for item in filtered:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("나쁜 original instruction 개수:", len(bad_original_insts))

    # 3) refined 파일들에서
    #    - 자기 original_instruction이 bad_original_insts에 속하면 제거
    for file_name in input_files:
        if "refined" not in file_name:
            continue

        filtered = []
        total = 0
        removed = 0

        with open(file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                total += 1
                data = json.loads(line)
                inst = data["instruction"]

                # refined 라인이 original_instruction 필드를 이미 갖고 있을 수도 있고,
                # 없으면 refine.jsonl에서 찾아온다.
                orig_inst = data.get("original_instruction")
                if orig_inst is None:
                    orig_inst = inst2orig.get(inst)

                # 매핑이 안 잡히면 정책 선택:
                # - 여기서는 "나쁜 original인지 알 수 없으니 살린다"로 가정
                if orig_inst is None:
                    filtered.append(data)
                    continue

                if orig_inst in bad_original_insts:
                    removed += 1
                    # 버림
                else:
                    filtered.append(data)

        print(f"{file_name} (refined) 필터링 전 개수: {total}")
        print(f"{file_name} (refined) 제거된 개수: {removed}")
        print(f"{file_name} (refined) 필터링 후 개수: {len(filtered)}")

        out_name = f"./filtered_{file_name.split('/')[-1]}"
        with open(out_name, 'w', encoding='utf-8') as outfile:
            for item in filtered:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
