#!/usr/bin/env python3
"""
sample_datasets.py — Reproducible random sampling from DROP, SNIPS, and SkillsBench
for the pilot annotation / first-pass translation run.

Samples drawn:
  DROP        : 100 records  — stratified by answer type (number / span / date)
  SNIPS       : 100 utterances — stratified by intent class (~14 per class)
  SkillsBench : 50 tasks  — random from tasks/ + tasks-extra/ combined (101 total)

Outputs land in:
  IRTIQA/Benchmarks/samples/drop/
      drop_sample100.jsonl       same JSONL format as source
      sample_meta.json           query_ids + answer_type for each record

  IRTIQA/Benchmarks/samples/snips/
      seq.in / seq.out / label   same 3-file format as SNIPS splits
      sample_meta.json           original split + line index per sample

  IRTIQA/Benchmarks/samples/skillsbench/
      <task-name>/instruction.md   copied from source
      <task-name>/task.toml        copied from source
      sample_meta.json             list of selected task names + source paths

Usage:
  python sample_datasets.py [--seed 42] [--drop N] [--snips N] [--skills N]
  python sample_datasets.py --seed 123 --drop 100 --snips 100 --skills 50
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IRTIQA = Path(os.environ.get("IRTIQA_ROOT", Path(__file__).resolve().parents[2]))
BENCHMARKS = IRTIQA / "Benchmarks"

DROP_SRC   = BENCHMARKS / "drop" / "drop_10k.jsonl"
SNIPS_SRC  = BENCHMARKS / "SNIPS"
SKILLS_SRC = BENCHMARKS / "skillsbench"
SAMPLES    = BENCHMARKS / "samples"

SNIPS_SPLITS = ("train", "dev", "test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def write_lines(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved {len(lines)} lines → {path}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stratified_sample(groups: dict[str, list], total: int, rng: random.Random) -> list:
    """
    Draw `total` items from `groups` (dict of key → list of items).
    Allocates proportionally, then fills remainder from largest groups.
    """
    n_groups = len(groups)
    base = total // n_groups
    remainder = total % n_groups

    sampled = []
    sorted_keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)

    for i, key in enumerate(sorted_keys):
        n = base + (1 if i < remainder else 0)
        pool = groups[key]
        n = min(n, len(pool))
        sampled.extend(rng.sample(pool, n))

    return sampled


# ---------------------------------------------------------------------------
# DROP sampling
# ---------------------------------------------------------------------------

def sample_drop(n: int, rng: random.Random, out_dir: Path) -> None:
    print(f"\n[DROP] Sampling {n} from {DROP_SRC.name}…")
    records = load_jsonl(DROP_SRC)

    # Stratify by answer type (number / span / date / other)
    def answer_type(rec: dict) -> str:
        types = rec.get("answers_spans", {}).get("types", [])
        if not types:
            return "other"
        t = types[0].lower()
        if t in ("number",):
            return "number"
        if t in ("date",):
            return "date"
        return "span"

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[answer_type(r)].append(r)

    print(f"  Answer-type distribution in source:")
    for k, v in sorted(groups.items()):
        print(f"    {k:10s}: {len(v)}")

    sampled = stratified_sample(groups, n, rng)
    rng.shuffle(sampled)

    save_jsonl(sampled, out_dir / "drop_sample100.jsonl")

    meta = {
        "source": str(DROP_SRC),
        "total_sampled": len(sampled),
        "seed": rng.getstate(),
        "answer_type_counts": {k: sum(1 for r in sampled if answer_type(r) == k) for k in groups},
        "query_ids": [r["query_id"] for r in sampled],
    }
    # seed state is large; just store the summary
    meta["seed"] = "(see --seed arg)"
    save_json(meta, out_dir / "sample_meta.json")
    print(f"  Sampled type breakdown: { {k: v for k,v in meta['answer_type_counts'].items()} }")


# ---------------------------------------------------------------------------
# SNIPS sampling
# ---------------------------------------------------------------------------

def sample_snips(n: int, rng: random.Random, out_dir: Path) -> None:
    print(f"\n[SNIPS] Sampling {n} utterances from all splits (stratified by intent)…")

    # Load all splits together, track source for metadata
    all_utterances: list[str] = []
    all_slots:      list[str] = []
    all_intents:    list[str] = []
    all_meta:       list[dict] = []

    for split in SNIPS_SPLITS:
        split_dir = SNIPS_SRC / split
        if not split_dir.exists():
            continue
        utts   = (split_dir / "seq.in").read_text(encoding="utf-8").splitlines()
        slots  = (split_dir / "seq.out").read_text(encoding="utf-8").splitlines()
        labels = (split_dir / "label").read_text(encoding="utf-8").splitlines()

        # Align lengths (guard against trailing newlines)
        length = min(len(utts), len(slots), len(labels))
        for i in range(length):
            all_utterances.append(utts[i])
            all_slots.append(slots[i])
            all_intents.append(labels[i])
            all_meta.append({"split": split, "line_idx": i})

    # Stratify by intent
    intent_groups: dict[str, list[int]] = defaultdict(list)
    for i, intent in enumerate(all_intents):
        intent_groups[intent].append(i)

    print(f"  Intent distribution in source:")
    for k, v in sorted(intent_groups.items()):
        print(f"    {k:30s}: {len(v)}")

    sampled_indices = stratified_sample(intent_groups, n, rng)
    rng.shuffle(sampled_indices)

    sampled_utts    = [all_utterances[i] for i in sampled_indices]
    sampled_slots   = [all_slots[i]      for i in sampled_indices]
    sampled_intents = [all_intents[i]    for i in sampled_indices]
    sampled_meta    = [all_meta[i]       for i in sampled_indices]

    write_lines(sampled_utts,    out_dir / "seq.in")
    write_lines(sampled_slots,   out_dir / "seq.out")
    write_lines(sampled_intents, out_dir / "label")

    # Copy label-list files
    for fname in ("intent_label.txt", "slot_label.txt"):
        src = SNIPS_SRC / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    intent_counts = defaultdict(int)
    for i in sampled_indices:
        intent_counts[all_intents[i]] += 1

    meta = {
        "source": str(SNIPS_SRC),
        "total_sampled": len(sampled_indices),
        "intent_counts": dict(sorted(intent_counts.items())),
        "samples": [
            {"utterance": all_utterances[i], "intent": all_intents[i], **all_meta[i]}
            for i in sampled_indices
        ],
    }
    save_json(meta, out_dir / "sample_meta.json")
    print(f"  Sampled intent breakdown: {dict(intent_counts)}")


# ---------------------------------------------------------------------------
# SkillsBench sampling
# ---------------------------------------------------------------------------

def sample_skillsbench(n: int, rng: random.Random, out_dir: Path) -> None:
    print(f"\n[SkillsBench] Sampling {n} tasks from tasks/ + tasks-extra/…")

    all_tasks: list[Path] = []
    for sub in ("tasks", "tasks-extra"):
        td = SKILLS_SRC / sub
        if td.exists():
            all_tasks.extend(sorted(p for p in td.iterdir() if p.is_dir()))

    print(f"  Total available tasks: {len(all_tasks)}")

    sampled_tasks = rng.sample(all_tasks, min(n, len(all_tasks)))
    sampled_tasks.sort(key=lambda p: p.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_entries = []

    for task_dir in sampled_tasks:
        task_out = out_dir / task_dir.name
        task_out.mkdir(exist_ok=True)

        for fname in ("instruction.md", "task.toml"):
            src = task_dir / fname
            if src.exists():
                shutil.copy2(src, task_out / fname)

        meta_entries.append({
            "task": task_dir.name,
            "source": str(task_dir),
            "sub_collection": task_dir.parent.name,
        })
        print(f"    Copied: {task_dir.name}")

    meta = {
        "source": str(SKILLS_SRC),
        "total_sampled": len(sampled_tasks),
        "tasks": meta_entries,
    }
    save_json(meta, out_dir / "sample_meta.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random sampling from DROP, SNIPS, SkillsBench.")
    p.add_argument("--seed",   type=int, default=42,  help="Random seed (default: 42)")
    p.add_argument("--drop",   type=int, default=100, help="DROP sample size (default: 100)")
    p.add_argument("--snips",  type=int, default=100, help="SNIPS sample size (default: 100)")
    p.add_argument("--skills", type=int, default=50,  help="SkillsBench sample size (default: 50)")
    p.add_argument("--out",    type=str, default=str(SAMPLES),
                   help=f"Output root directory (default: {SAMPLES})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)

    print(f"Random seed : {args.seed}")
    print(f"Output root : {out}")

    sample_drop(args.drop, rng, out / "drop")

    # Re-seed per dataset so each sample is independently reproducible
    rng = random.Random(args.seed + 1)
    sample_snips(args.snips, rng, out / "snips")

    rng = random.Random(args.seed + 2)
    sample_skillsbench(args.skills, rng, out / "skillsbench")

    print("\nDone. Samples written to:", out)
    print("  drop/drop_sample100.jsonl")
    print("  snips/seq.in|seq.out|label")
    print("  skillsbench/<task-name>/instruction.md")


if __name__ == "__main__":
    main()
