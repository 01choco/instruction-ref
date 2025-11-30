# trim_inference_jsonl.py
import json, os

IN_PATH  = "./results/inference-ultrafeedback10.jsonl"
OUT_PATH = "./results/inference-ultrafeedback10.trimmed.jsonl"

# 자를 기준 패턴들 (가장 먼저 등장하는 위치로 자름)
STOP_STRINGS = [
    "user\n\n",
    "\nuser\n",
    "User:",
]


def cut_at_first_stop(s: str) -> str:
    # 가장 이른(작은) 인덱스를 찾아 자르기
    earliest = len(s)
    for stop in STOP_STRINGS:
        i = s.find(stop)
        if i != -1 and i < earliest:
            earliest = i
    return s[:earliest].rstrip()

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    trimmed_cnt = 0
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for rec in iter_jsonl(IN_PATH):
            total += 1
            instr = rec.get("instruction", "")
            resps = rec.get("responses", [])
            new_resps = []
            for r in resps:
                new_r = cut_at_first_stop(r)
                if new_r != r:
                    trimmed_cnt += 1
                new_resps.append(new_r)
            json.dump({"instruction": instr, "responses": new_resps}, out, ensure_ascii=False)
            out.write("\n")

    print(f"done. {trimmed_cnt} responses trimmed out of {total} records.")

if __name__ == "__main__":
    main()
