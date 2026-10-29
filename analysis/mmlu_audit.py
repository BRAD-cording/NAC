"""MMLU 存档复核：抽取规则敏感性 + 无生成率 + 答案分布。

背景：GSM8K 已证明"末数字抽取器"会把截断/续写误判为错误；MMLU 是 32-token 单字母
任务，理论上不受截断影响，但 extract_choice 有两条规则（answer-is 优先、前 8 字符
首个 A-D），仍需核验不同模型是否被同一条规则差异化对待。

用法：python analysis/mmlu_audit.py
"""
import glob
import io
import json
import os
import re

ROOT = "/media/pi/DATA/rknn-llm/nac_paper"
LETTERS = "ABCD"


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def rule_code(text):
    """现行 extract_choice 的等价实现。"""
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text.strip()[:8])
    return m.group(1).upper() if m else None


def rule_first_letter(text):
    """更宽松的规则：全文里第一个独立的大写字母 A-D。"""
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1).upper() if m else None


def rule_answer_only(text):
    """只认 'answer is X' / 'Answer: X'。"""
    m = re.search(r"answer\s*(?:is|:|=)?\s*\(?([ABCD])\)?\b", text, re.I)
    return m.group(1).upper() if m else None


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "results", "acc_mmlu_*.records*.json")))
    print(f"{'config':44s} {'n':>4s} {'stored':>7s} {'code':>6s} {'1st':>6s} {'ansOnly':>7s} {'empty':>6s} {'agree':>6s}")
    for f in files:
        recs = load(f)
        if not isinstance(recs, list) or not recs:
            continue
        n = len(recs)
        stored = sum(1 for r in recs if r.get("ok"))
        ok_code = ok_1st = ok_ans = 0
        empty = 0
        for r in recs:
            g = (r.get("gen") or "").strip()
            gold = str(r.get("gold"))
            if not g:
                empty += 1
            if rule_code(g) == gold:
                ok_code += 1
            if rule_first_letter(g) == gold:
                ok_1st += 1
            if rule_answer_only(g) == gold:
                ok_ans += 1
        name = os.path.basename(f).replace("acc_mmlu_", "").replace(".records.json", "")
        print(f"{name:44s} {n:4d} {stored/n:7.4f} {ok_code/n:6.4f} {ok_1st/n:6.4f} "
              f"{ok_ans/n:7.4f} {empty/n:6.3f} {stored/max(1,ok_code):6.3f}")


if __name__ == "__main__":
    main()
