"""Check the LLM judge against labelled samples before trusting any judged score.

    python scripts/calibrate_judge.py

Uses `human_label` where filled in, else `author_label` (AI-authored, so treat the result as a
smoke test only). Reports overall agreement and every disagreement. Judge errors are excluded,
not counted as disagreements.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

import json  # noqa: E402

from evals.evaluators import JUDGE_MODEL, JudgeError, judge  # noqa: E402
from seed_dataset import load  # noqa: E402


def main() -> int:
    items = {i["id"]: i for i in load()["items"]}
    samples = json.loads((ROOT / "dataset" / "judge_calibration.json").read_text())["samples"]
    human = sum(1 for s in samples if s["human_label"])
    print(f"judge={JUDGE_MODEL}  samples={len(samples)}  human-labelled={human}"
          + ("  (falling back to AI author labels: smoke test only)" if human < len(samples) else "") + "\n")

    agree = errors = 0
    for s in samples:
        item = items[s["case"]]
        exp = item["expected_output"]
        label = s["human_label"] or s["author_label"]
        try:
            ok, why = judge(exp["judge"], item["input"]["message"], s["reply"], exp)
        except JudgeError as e:
            errors += 1
            print(f"  {s['id']}  ERROR  {e}")
            continue
        got = "pass" if ok else "fail"
        agree += got == label
        if got != label:
            print(f"  {s['id']}  DISAGREE  label={label} judge={got}  ({exp['judge']})\n      {why}")
    scored = len(samples) - errors
    print(f"\nagreement {agree}/{scored}" + (f"  ({errors} judge errors excluded)" if errors else ""))
    return 0 if scored and agree >= scored - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
