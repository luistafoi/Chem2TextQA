"""Convert unpacked SMolInstruct `molecule_captioning` test split to our eval
schema (`prompt`, `response`). Uses the same SMILES+question prompt format we
trained on so the finetuned models see a familiar prefix.
"""
import json
from pathlib import Path

SRC = Path("/mnt/data_lab/ChemQA/eval/data/smolinstruct/raw/test/molecule_captioning.jsonl")
OUT = Path("/mnt/data_lab/ChemQA/eval/data/smolinstruct_mcap_test.jsonl")


def main():
    records = [json.loads(l) for l in SRC.open()]
    print(f"source: {len(records)} molecule_captioning test samples")
    with OUT.open("w") as f:
        for i, r in enumerate(records):
            smiles = r["input"]
            caption = r["output"]
            prompt = f"{smiles}\nDescribe this molecule in detail."
            f.write(
                json.dumps(
                    {
                        "sample_id": f"molecule_captioning.test.{i}",
                        "task": "molecule_captioning",
                        "prompt": prompt,
                        "response": caption,
                        "raw_input": smiles,
                        "raw_output": caption,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {OUT}  n={len(records)}")


if __name__ == "__main__":
    main()
