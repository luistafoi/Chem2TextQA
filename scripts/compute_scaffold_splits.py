"""Scaffold-based train/val/test split for Chem2TextQA.

Follows the MoleculeNet convention: compounds sharing the same Murcko
scaffold (core ring system) stay in the same split. This prevents
train/test leakage via scaffold similarity — a standard requirement for
chemistry ML evaluation.

Two modes:
  strict     — sort scaffold groups by size DESC, greedy-pack into
               train/val/test. Deterministic (seed only affects tie-breaks).
               This is what most papers report. Default.
  randomized — shuffle scaffolds before packing. Seed matters; use for
               multi-seed robustness / variance analysis.

Canary compounds (split=="canary" or is_canary==true) are preserved —
never reassigned by scaffold.

Usage:

  # 1. Compute the CID → split mapping from a dataset_final.jsonl:
  python3 scripts/compute_scaffold_splits.py compute \\
      --dataset data/qa_pipeline/full_premium_kimi/dataset_final.jsonl \\
      --canary  data/qa_pipeline/full_premium_kimi/canary/dataset_final.jsonl \\
      --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \\
      --output data/qa_pipeline/full_premium_kimi/cid_to_split.json \\
      --report data/qa_pipeline/full_premium_kimi/scaffold_split_report.json

  # 2. Apply the mapping in place, rewriting the `split` field:
  python3 scripts/compute_scaffold_splits.py apply \\
      --src data/qa_pipeline/full_premium_kimi/dataset_final.jsonl \\
      --mapping data/qa_pipeline/full_premium_kimi/cid_to_split.json

Requires: rdkit (pip install rdkit)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")  # silence parser noise
        return Chem, MurckoScaffold
    except ImportError:
        print("ERROR: RDKit not installed. Install with: pip install rdkit", file=sys.stderr)
        sys.exit(1)


def murcko_scaffold(smiles: str):
    """Return the Murcko scaffold SMILES for a compound, or None if unparseable.
    Acyclic compounds return an empty string (their scaffold is empty)."""
    Chem, MurckoScaffold = _load_rdkit()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def compute_splits(records: list[dict],
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   mode: str = "strict",
                   seed: int = 42) -> tuple[dict[int, str], dict]:
    """Return (cid_to_split, stats).

    `records` is the list of top-level dataset records (one per compound,
    with cid + smiles fields). Canary compounds are excluded from splitting
    and returned with split=="canary".
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1: {train_ratio}+{val_ratio}+{test_ratio}"

    rng = random.Random(seed)

    # Separate canary; preserve its assignment
    cid_to_split: dict[int, str] = {}
    non_canary: list[dict] = []
    for r in records:
        cid = int(r["cid"])
        if r.get("is_canary") or r.get("split") == "canary":
            cid_to_split[cid] = "canary"
        else:
            non_canary.append(r)

    # Group by scaffold
    scaffold_to_cids: dict[str, list[int]] = defaultdict(list)
    unparseable: list[int] = []
    acyclic: list[int] = []

    for r in non_canary:
        sm = r.get("smiles") or ""
        sc = murcko_scaffold(sm)
        if sc is None:
            unparseable.append(int(r["cid"]))
            continue
        if sc == "":
            acyclic.append(int(r["cid"]))
            continue
        scaffold_to_cids[sc].append(int(r["cid"]))

    # Order scaffolds
    if mode == "strict":
        # MoleculeNet: sort by group size DESC, then by scaffold SMILES
        # (stable tiebreak for reproducibility).
        ordered = sorted(scaffold_to_cids.items(),
                         key=lambda kv: (-len(kv[1]), kv[0]))
    elif mode == "randomized":
        # Shuffle scaffolds within equal-size tiers, preserving size DESC order.
        items = list(scaffold_to_cids.items())
        rng.shuffle(items)
        ordered = sorted(items, key=lambda kv: -len(kv[1]))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Targets apply to the scaffold-packed pool only. Acyclic compounds are
    # distributed separately by CID hash below, contributing their own
    # proportional split. Mixing them into these targets makes train
    # overshoot when acyclic compounds are a non-trivial share of the data.
    scaffold_pool = sum(len(cids) for cids in scaffold_to_cids.values())
    train_target = train_ratio * scaffold_pool
    val_target = val_ratio * scaffold_pool
    # test gets the rest

    train_cids: list[int] = []
    val_cids: list[int] = []
    test_cids: list[int] = []

    # Greedy pack: each scaffold group goes entirely into one split.
    for scaffold, cids in ordered:
        if len(train_cids) + len(cids) <= train_target:
            train_cids.extend(cids)
        elif len(val_cids) + len(cids) <= val_target:
            val_cids.extend(cids)
        else:
            test_cids.extend(cids)

    # Acyclic compounds have no scaffold → distribute by CID hash to keep
    # proportions roughly right. They cannot leak via scaffold similarity
    # (no shared scaffold), so random assignment is safe here.
    import hashlib
    for cid in acyclic:
        h = int(hashlib.sha256(str(cid).encode()).hexdigest()[:8], 16) % 100
        if h < train_ratio * 100:
            train_cids.append(cid)
        elif h < (train_ratio + val_ratio) * 100:
            val_cids.append(cid)
        else:
            test_cids.append(cid)

    # Unparseable SMILES go to train (safest default — don't evaluate on them)
    train_cids.extend(unparseable)

    for cid in train_cids:
        cid_to_split[cid] = "train"
    for cid in val_cids:
        cid_to_split[cid] = "val"
    for cid in test_cids:
        cid_to_split[cid] = "test"

    stats = {
        "mode": mode,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_non_canary_compounds": len(non_canary),
        "n_canary_compounds": sum(1 for v in cid_to_split.values() if v == "canary"),
        "n_unique_scaffolds": len(scaffold_to_cids),
        "n_acyclic": len(acyclic),
        "n_unparseable_smiles": len(unparseable),
        "largest_scaffold_group_size": max((len(v) for v in scaffold_to_cids.values()), default=0),
        "median_scaffold_group_size": (
            sorted(len(v) for v in scaffold_to_cids.values())[len(scaffold_to_cids) // 2]
            if scaffold_to_cids else 0
        ),
        "final_split_sizes": {
            "train": len(train_cids),
            "val": len(val_cids),
            "test": len(test_cids),
            "canary": sum(1 for v in cid_to_split.values() if v == "canary"),
        },
        "final_split_percentages": {
            "train": 100 * len(train_cids) / len(non_canary) if non_canary else 0.0,
            "val": 100 * len(val_cids) / len(non_canary) if non_canary else 0.0,
            "test": 100 * len(test_cids) / len(non_canary) if non_canary else 0.0,
        },
    }
    return cid_to_split, stats


def cmd_compute(args):
    recs = []
    for path in [args.dataset] + ([args.canary] if args.canary else []):
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {p} missing, skipping", file=sys.stderr)
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    print(f"  Loaded {len(recs):,} compound records")

    cid_to_split, stats = compute_splits(
        recs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        mode=args.mode,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON with string keys (CIDs) for portability
    out.write_text(json.dumps({str(k): v for k, v in cid_to_split.items()}, indent=2))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(stats, indent=2))
        print(f"  Wrote report: {args.report}")

    print(f"  Wrote CID→split mapping: {out}")
    print()
    print(f"  Mode:              {stats['mode']}")
    print(f"  Unique scaffolds:  {stats['n_unique_scaffolds']:,}")
    print(f"  Acyclic compounds: {stats['n_acyclic']:,}")
    print(f"  Unparseable SMILES: {stats['n_unparseable_smiles']:,}")
    print(f"  Largest scaffold group: {stats['largest_scaffold_group_size']:,} compounds")
    print(f"  Final split sizes:")
    for k, v in stats["final_split_sizes"].items():
        print(f"    {k:<8} {v:>6,}")
    print(f"  Final split %:")
    for k, v in stats["final_split_percentages"].items():
        print(f"    {k:<8} {v:>5.2f}%")


def cmd_apply(args):
    mapping = json.loads(Path(args.mapping).read_text())
    # Keys may be str; normalize to int
    cid_to_split = {int(k): v for k, v in mapping.items()}

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src
    if dst == src:
        tmp = src.with_suffix(src.suffix + ".tmp")
        out = tmp
    else:
        out = dst
    out.parent.mkdir(parents=True, exist_ok=True)

    updated = unchanged = unknown = 0
    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = int(rec["cid"])
            new_split = cid_to_split.get(cid)
            if new_split is None:
                unknown += 1
            elif rec.get("split") == new_split:
                unchanged += 1
            else:
                rec["split"] = new_split
                updated += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if dst == src:
        out.replace(src)

    print(f"  Source:         {src}")
    print(f"  Destination:    {dst}")
    print(f"  Updated:        {updated:,}")
    print(f"  Unchanged:      {unchanged:,}")
    print(f"  Unknown CIDs:   {unknown:,}  (left with original split field)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="Compute scaffold-based CID→split mapping")
    c.add_argument("--dataset", required=True, help="Main dataset_final.jsonl")
    c.add_argument("--canary", default=None, help="Canary dataset_final.jsonl (optional)")
    c.add_argument("--train-ratio", type=float, default=0.70)
    c.add_argument("--val-ratio",   type=float, default=0.15)
    c.add_argument("--test-ratio",  type=float, default=0.15)
    c.add_argument("--mode", choices=["strict", "randomized"], default="strict")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--output", required=True, help="Output cid_to_split.json")
    c.add_argument("--report", default=None, help="Optional JSON stats report")
    c.set_defaults(func=cmd_compute)

    a = sub.add_parser("apply", help="Apply CID→split mapping to a dataset JSONL")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", default=None, help="Output JSONL; default = overwrite src in place")
    a.add_argument("--mapping", required=True, help="cid_to_split.json from compute step")
    a.set_defaults(func=cmd_apply)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
