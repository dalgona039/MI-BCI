"""Five-fold, subject-disjoint paired contamination audit.

The same fixed outer folds and clean test trials are used for all models,
training conditions, and seeds. One clean validation subject is selected from
the training side of each fold. Each test subject receives one prediction per
seed and condition, after model selection on validation data.

This is a separate protocol from rigorous_loso.py. Its results must not be
pooled with LOSO results. HDF5 epochs are streamed to bound host RAM use.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader

from ablation_study_v5 import (CFG, MODEL_CLASSES, _metrics, evaluate,
                               pick_device, set_seed, train_epoch)
from rigorous_loso import (DEFAULT_DATA, DEFAULT_EXCLUSIONS, FIELDS,
                           load_manifest, make_dataset)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "BCI_Research/results/rigorous_group5"
FOLD_SEED = 20261011
GROUP_FIELDS = ["outer_fold", *FIELDS, "fold_elapsed_seconds"]


def outer_folds(eligible):
    """Fixed, reproducible five-way partition of the 49 test subjects."""
    shuffled = np.random.default_rng(FOLD_SEED).permutation(sorted(eligible))
    return [sorted(int(s) for s in part) for part in np.array_split(shuffled, 5)]


def split_group(fold_idx, folds, eligible, pool, seed):
    test_sids = folds[fold_idx]
    candidates = sorted(set(eligible) - set(test_sids))
    validation_sid = candidates[(seed + fold_idx) % len(candidates)]
    train_sids = sorted(set(pool) - set(test_sids) - {validation_sid})
    assert not (set(train_sids) & set(test_sids))
    assert validation_sid not in test_sids and validation_sid not in train_sids
    return train_sids, validation_sid, test_sids


def run_group(fold_idx, folds, eligible, pool, exclusions, data_dir,
              model_type, condition, seed, max_epochs, patience, device):
    train_sids, validation_sid, test_sids = split_group(
        fold_idx, folds, eligible, pool, seed)
    set_seed(seed * 100000 + fold_idx)
    training = ConcatDataset([
        make_dataset(s, data_dir, exclusions, condition == "clean_train", model_type)
        for s in train_sids
    ])
    validation = make_dataset(validation_sid, data_dir, exclusions, True, model_type)
    train_loader = DataLoader(training, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=64, shuffle=False)

    model = MODEL_CLASSES[model_type](dict(CFG)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, train_loader, optimizer, scaler, device)
        pred, true = evaluate(model, val_loader, device)
        score = f1_score(true, pred, average="macro", zero_division=0)
        epochs_run = epoch
        if score > best_f1:
            best_f1 = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= patience:
            print(f"    fold {fold_idx + 1}/5 epoch {epoch}/{max_epochs} "
                  f"validation_F1={score:.4f} best_epoch={best_epoch} "
                  f"patience={stale}/{patience}", flush=True)
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    rows = []
    for sid in test_sids:
        testing = make_dataset(sid, data_dir, exclusions, True, model_type)
        test_loader = DataLoader(testing, batch_size=64, shuffle=False)
        pred, true = evaluate(model, test_loader, device)
        row = _metrics(sid, pred, true)
        row.pop("confusion", None)
        row.update(outer_fold=fold_idx + 1, validation_sid=validation_sid,
                   seed=seed, model_type=model_type, condition=condition,
                   n_train_trials=len(training), n_val_trials=len(validation),
                   best_epoch=best_epoch, epochs_run=epochs_run)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["dirty_train", "clean_train"], required=True)
    parser.add_argument("--model_type", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--exclude_json", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("max_epochs and patience must be positive")
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(1, 6)):
        parser.error("--folds must contain distinct integers from 1 to 5")

    exclusions, pool, eligible = load_manifest(args.exclude_json, args.data_dir)
    folds = outer_folds(eligible)
    if args.dry_run:
        for fold in args.folds:
            train, val, test = split_group(fold - 1, folds, eligible, pool, args.seed)
            print(f"fold={fold}/5 test={test} validation=s{val:02d} "
                  f"train_subjects={len(train)}")
        return 0

    device = pick_device(args.model_type, args.device)
    if device.type == "mps" and args.model_type != "emg_only":
        raise RuntimeError("MPS is disallowed for EEGNet conditions")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.condition}_{args.model_type}_seed{args.seed}.csv"
    config_path = output.with_suffix(".config.json")
    config = {
        "protocol": "fixed_subject_group5", "fold_seed": FOLD_SEED,
        "folds": folds, "condition": args.condition, "model_type": args.model_type,
        "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
        "pool": pool, "eligible": eligible, "clean_eeg_renormalization": True,
        "exclusions_sha256": hashlib.sha256(args.exclude_json.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(Path(sys.modules["rigorous_loso"].__file__).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(sys.modules["ablation_study_v5"].__file__).read_bytes()).hexdigest(),
        "data_dir": str(args.data_dir.resolve()),
    }
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"Refusing to resume with different settings: {output}")
    else:
        if output.exists():
            raise RuntimeError(f"Results exist without a config manifest: {output}")
        config_path.write_text(json.dumps(config, indent=2))

    completed = set()
    if output.exists():
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for fold in range(1, 6):
            actual = {int(row["sid"]) for row in rows if int(row["outer_fold"]) == fold}
            expected = set(folds[fold - 1])
            if actual and actual != expected:
                raise RuntimeError(f"Partial or inconsistent fold {fold}: {output}")
            if actual:
                completed.add(fold)
        if len(rows) != sum(len(folds[fold - 1]) for fold in completed):
            raise RuntimeError(f"Duplicate or unexpected result rows: {output}")

    fold_times = {}
    if output.exists():
        for row in rows:
            if row.get("fold_elapsed_seconds"):
                fold_times[int(row["outer_fold"])] = float(row["fold_elapsed_seconds"])

    print(f"{args.condition} / {args.model_type} / seed {args.seed}: "
          f"완료 {len(completed)}/5 folds, 남음 {5 - len(completed)}", flush=True)
    for fold in args.folds:
        if fold in completed:
            continue
        _, validation_sid, test_sids = split_group(
            fold - 1, folds, eligible, pool, args.seed)
        print(f"[{len(completed)}/5] fold {fold} 시작 "
              f"(test={test_sids}, validation=s{validation_sid:02d})", flush=True)
        started = time.perf_counter()
        rows = run_group(fold - 1, folds, eligible, pool, exclusions, args.data_dir,
                         args.model_type, args.condition, args.seed,
                         args.max_epochs, args.patience, device)
        elapsed = time.perf_counter() - started
        for row in rows:
            row["fold_elapsed_seconds"] = round(elapsed, 1)
        new_file = not output.exists()
        with output.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=GROUP_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerows({key: row[key] for key in GROUP_FIELDS} for row in rows)
        completed.add(fold)
        fold_times[fold] = elapsed
        remaining = len(set(range(1, 6)) - completed)
        estimated_minutes = remaining * np.mean(list(fold_times.values())) / 60
        print(f"[{len(completed)}/5] fold {fold} 완료 "
              f"subjects={len(rows)}, best_epoch={rows[0]['best_epoch']}, "
              f"elapsed={elapsed:.0f}s; 남음 {remaining}; "
              f"이 조합 예상 잔여 {estimated_minutes:.0f}분", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
