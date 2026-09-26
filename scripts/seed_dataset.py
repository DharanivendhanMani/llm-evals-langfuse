"""Upsert dataset/arusuvai_v0.json into Langfuse. Dry-run by default.

    python scripts/seed_dataset.py            # validate and preview, touches nothing
    python scripts/seed_dataset.py --apply    # create/update the dataset and its items

Items carry stable ids, so re-running updates them in place (Langfuse versions the dataset)
instead of creating duplicates. The dataset carries JSON Schemas, so Langfuse rejects a
malformed item at write time.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = ROOT / "dataset" / "arusuvai_v0.json"


def load(path: Path = DATASET_FILE) -> dict:
    return json.loads(path.read_text())


def validate(spec: dict) -> list[str]:
    errors, seen = [], set()
    for it in spec["items"]:
        iid = it.get("id", "<missing id>")
        if iid in seen:
            errors.append(f"{iid}: duplicate id")
        seen.add(iid)
        exp, md = it.get("expected_output", {}), it.get("metadata", {})
        for key in ("behavior", "reference"):
            if key not in exp:
                errors.append(f"{iid}: expected_output.{key} missing")
        for key in ("case", "category", "severity", "reason"):
            if key not in md:
                errors.append(f"{iid}: metadata.{key} missing")
        if not (exp.get("must_include") or exp.get("must_not_include") or exp.get("tools_required") or exp.get("judge")):
            errors.append(f"{iid}: no check would run (need must_include/must_not_include/tools_required/judge)")
        if exp.get("judge") in ("honest_handoff", "injection_resistance") and not exp.get("must_not_claim") and not exp.get("must_not_include"):
            errors.append(f"{iid}: judged unanswerable/adversarial case without must_not_claim")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write to Langfuse (default: dry run)")
    args = parser.parse_args()

    spec = load()
    errors = validate(spec)
    counts = Counter(i["metadata"]["category"] for i in spec["items"])
    print(f"{spec['dataset']['name']}: {len(spec['items'])} items {dict(counts)}")
    for it in spec["items"]:
        md = it["metadata"]
        print(f"  {md['case']:7} {md['category']:12} {md['severity']:9} {it['input']['message'][:70]}")
    if errors:
        print("\nINVALID:\n  " + "\n  ".join(errors))
        return 1
    if not args.apply:
        print("\nDry run OK. Nothing written. Re-run with --apply after reviewing expected outputs.")
        return 0

    load_dotenv(ROOT / ".env")
    from langfuse import get_client

    lf = get_client()
    ds = spec["dataset"]
    lf.create_dataset(
        name=ds["name"],
        description=ds["description"],
        metadata=ds["metadata"],
        input_schema=ds["input_schema"],
        expected_output_schema=ds["expected_output_schema"],
    )
    for it in spec["items"]:
        lf.create_dataset_item(
            dataset_name=ds["name"],
            id=it["id"],
            input=it["input"],
            expected_output=it["expected_output"],
            metadata=it["metadata"],
        )
    lf.flush()
    print(f"\nUpserted {len(spec['items'])} items into '{ds['name']}'.")
    print("REMINDER: expected outputs are AI-authored. Review them in the Langfuse UI before trusting any score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
