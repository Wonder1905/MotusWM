"""
Offline T5 embedding cache builder for a local LeRobot dataset.

Skips all huggingface_hub round-trips (no auth needed). Takes a local dataset
root, reads `meta/episodes.jsonl` (the user is responsible for creating this if
the dataset uses v3 parquet format) and `meta/tasks.jsonl` (for task_index ->
task_string mapping), encodes each unique task string once with WAN's T5
encoder, and writes ONE .pt per unique task under
`{root}/t5_embedding/task_<task_index>.pt`.

The Motus LeRobot loader (patched in `lerobot_dataset.py`) looks up
`task_<task_index>.pt` first, then falls back to the legacy
`episode_<episode_index>.pt` naming. The episodes.jsonl is also patched in-place
with `t5_embedding_path` set to the task-level file, so dataloaders that read
the pointer directly still work.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


def _bak_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "bak").resolve()


def _import_t5_encoder():
    bak = str(_bak_root())
    if bak not in sys.path:
        sys.path.insert(0, bak)
    from wan.modules.t5 import T5EncoderModel  # type: ignore
    return T5EncoderModel


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl_atomic(p: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _instruction_from_ep(ep: Dict[str, Any]) -> str:
    tasks = ep.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return str(ep.get("task", ""))


def _load_tasks_map(tasks_jsonl: Path) -> Dict[str, int]:
    """task_string -> task_index from meta/tasks.jsonl"""
    mapping: Dict[str, int] = {}
    if not tasks_jsonl.exists():
        return mapping
    with open(tasks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_str = row.get("task")
            ti = row.get("task_index")
            if isinstance(task_str, str) and ti is not None:
                mapping[task_str] = int(ti)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local LeRobot dataset root (contains meta/episodes.jsonl)")
    ap.add_argument("--wan_path", required=True, help="path containing Wan2.2-TI2V-5B/ with the T5 .pth + tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--folder", default="t5_embedding")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    episodes_jsonl = root / "meta" / "episodes.jsonl"
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"missing {episodes_jsonl} — generate it from meta/episodes/*.parquet first")

    out_dir = root / args.folder
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_jsonl(episodes_jsonl)
    print(f"loaded {len(episodes)} episode records from {episodes_jsonl}")

    tasks_map = _load_tasks_map(tasks_jsonl)  # task_string -> task_index
    if tasks_map:
        print(f"loaded {len(tasks_map)} task-index mappings from {tasks_jsonl}")
    else:
        print(f"no tasks.jsonl found at {tasks_jsonl}; will assign task indices in encounter order")

    # Build T5 once.
    T5EncoderModel = _import_t5_encoder()
    ckpt = os.path.join(args.wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(args.wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    assert os.path.exists(ckpt), f"missing T5 ckpt: {ckpt}"
    assert os.path.exists(tok), f"missing T5 tokenizer dir: {tok}"

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"loading T5 from {ckpt} on {args.device} ({dtype}) ...")
    encoder = T5EncoderModel(
        text_len=args.text_len, dtype=dtype, device=args.device,
        checkpoint_path=ckpt, tokenizer_path=tok,
    )

    # Encode each unique instruction once, then write ONE .pt per unique task_index.
    unique_instructions: List[str] = []
    seen = set()
    for ep in episodes:
        instr = _instruction_from_ep(ep)
        if instr not in seen:
            seen.add(instr)
            unique_instructions.append(instr)
    print(f"unique instructions: {len(unique_instructions)}")

    instr_to_task_idx: Dict[str, int] = {}
    for instr in unique_instructions:
        if instr in tasks_map:
            instr_to_task_idx[instr] = tasks_map[instr]
        else:
            # Fallback: assign indices in encounter order (works even without tasks.jsonl).
            instr_to_task_idx[instr] = len(instr_to_task_idx)

    written = 0
    instr_to_rel: Dict[str, str] = {}
    with torch.no_grad():
        for instr in unique_instructions:
            ti = instr_to_task_idx[instr]
            rel = f"{args.folder}/task_{ti:06d}.pt"
            abs_pt = root / rel
            instr_to_rel[instr] = rel
            if abs_pt.exists() and not args.overwrite:
                print(f"  skip existing  task_{ti:06d}.pt  '{instr[:60]}...'")
                continue
            out = encoder([instr], args.device)
            emb = out[0] if isinstance(out, list) else out
            if emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            torch.save(emb.detach().cpu(), abs_pt)
            print(f"  wrote task_{ti:06d}.pt  shape={tuple(emb.shape)}  '{instr[:60]}...'")
            written += 1

    # Patch episodes.jsonl pointers to point at the shared task file.
    for ep in episodes:
        ep["t5_embedding_path"] = instr_to_rel[_instruction_from_ep(ep)]

    _write_jsonl_atomic(episodes_jsonl, episodes)
    print(f"done. wrote {written} new task-level embeddings under {out_dir}")
    print(f"      patched {len(episodes)} episode pointers in {episodes_jsonl}")


if __name__ == "__main__":
    main()
