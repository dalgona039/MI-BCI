"""Paired, subject-disjoint LOSO audit for the movement-flag question.

For each outer test subject, a different eligible subject supplies early-stopping
validation. Both training conditions are scored on the same clean test trials:

* dirty_train: retain movement/voltage-flagged training trials;
* clean_train: remove them before constructing the training DataLoader.

Validation is clean in both conditions. The test subject is never inspected for
model selection. This is a new protocol: do not compare its absolute scores to
the saved v4/v5 scores as though only the exclusion policy had changed.

The clean condition recalculates each subject's EEG channel mean/std using
only retained epochs, reversing the effect of flagged epochs on the stored
z-score. ICA and continuous filtering were fitted before this step, so this
is not a fully reprocessed raw-signal pipeline. See RESEARCH_DIAGNOSIS.md.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader

from ablation_study_v5 import (
    BCIDataset, CFG, MODEL_CLASSES, _metrics, evaluate, pick_device,
    set_seed, train_epoch,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "BCI_Research/preprocessed/member_A"
DEFAULT_EXCLUSIONS = ROOT / "BCI_Research/results/ablation_v5_clean/excluded_epochs.json"
DEFAULT_OUTPUT = ROOT / "BCI_Research/results/rigorous_loso"
FIELDS = ["sid", "validation_sid", "seed", "model_type", "condition",
          "accuracy", "kappa", "itr", "left_recall", "right_recall",
          "n_trials", "n_train_trials", "n_val_trials", "best_epoch",
          "epochs_run"]


def split_subjects(test_sid, eligible_sids, pool_sids, seed):
    """Select a deterministic, eligible validation subject inside the outer fold."""
    eligible = sorted(set(eligible_sids))
    pool = sorted(set(pool_sids))
    if test_sid not in eligible or len(eligible) < 3:
        raise ValueError("Need an eligible test subject and at least 3 eligible subjects")
    offset = 1 + (seed % (len(eligible) - 1))
    validation_sid = eligible[(eligible.index(test_sid) + offset) % len(eligible)]
    train_sids = [s for s in pool if s not in (test_sid, validation_sid)]
    assert train_sids and test_sid != validation_sid
    assert test_sid not in train_sids and validation_sid not in train_sids
    return train_sids, validation_sid


def load_manifest(exclusions_path, data_dir):
    with open(exclusions_path) as stream:
        exclusions = json.load(stream)
    pool = sorted(int(s) for s in exclusions)
    eligible = [s for s in pool if exclusions[str(s)]["eligible"]]
    missing = [s for s in pool if not (data_dir / f"sub-{s:02d}_member_A.h5").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed subjects: {missing}")
    if len(pool) != 51 or len(eligible) != 49:
        raise ValueError(f"Unexpected cohort: {len(pool)} in pool, {len(eligible)} eligible")
    if sum(len(v["excl_idx"]) for v in exclusions.values()) != 753:
        raise ValueError("Exclusion count differs from verified 753-trial union")
    return exclusions, pool, eligible


def make_dataset(sid, data_dir, exclusions, clean):
    ds = BCIDataset(str(data_dir / f"sub-{sid:02d}_member_A.h5"),
                    CFG["emg_ds_factor"],
                    exclude_idx=exclusions[str(sid)]["excl_idx"] if clean else None)
    if clean:
        # HDF5 eeg/epochs was z-scored over all subject epochs. An affine
        # re-z-score of retained epochs is mathematically the same as
        # reconstructing pre-normalized epochs with saved norm_mean/std,
        # then fitting z-score statistics on retained epochs alone.
        mean = ds.eeg.mean(dim=(0, 2), keepdim=True)
        std = ds.eeg.std(dim=(0, 2), keepdim=True, correction=0)
        ds.eeg = (ds.eeg - mean) / (std + 1e-10)
    return ds


def run_fold(test_sid, eligible, pool, exclusions, data_dir, model_type,
             condition, seed, max_epochs, patience, device):
    train_sids, validation_sid = split_subjects(test_sid, eligible, pool, seed)
    set_seed(seed * 100000 + test_sid)
    train_clean = condition == "clean_train"
    training = ConcatDataset([make_dataset(s, data_dir, exclusions, train_clean)
                              for s in train_sids])
    validation = make_dataset(validation_sid, data_dir, exclusions, True)
    testing = make_dataset(test_sid, data_dir, exclusions, True)
    train_loader = DataLoader(training, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=64, shuffle=False)
    test_loader = DataLoader(testing, batch_size=64, shuffle=False)

    model = MODEL_CLASSES[model_type](dict(CFG)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = (torch.amp.GradScaler("cuda") if device.type == "cuda" else None)
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, train_loader, optimizer, scaler, device)
        pred, true = evaluate(model, val_loader, device)
        score = f1_score(true, pred, average="macro", zero_division=0)
        epochs_run = epoch
        if score > best_f1:
            best_f1 = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    # The test set is evaluated exactly once, after model selection is over.
    pred, true = evaluate(model, test_loader, device)
    row = _metrics(test_sid, pred, true)
    row.pop("confusion", None)
    row.update(validation_sid=validation_sid, seed=seed, model_type=model_type,
               condition=condition, n_train_trials=len(training),
               n_val_trials=len(validation), best_epoch=best_epoch,
               epochs_run=epochs_run)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["dirty_train", "clean_train"], required=True)
    parser.add_argument("--model_type", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sids", type=int, nargs="+")
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

    exclusions, pool, eligible = load_manifest(args.exclude_json, args.data_dir)
    targets = args.sids or eligible
    if len(set(targets)) != len(targets) or not set(targets).issubset(eligible):
        parser.error("--sids must be unique eligible subject IDs")
    if args.dry_run:
        for sid in targets:
            training, validation = split_subjects(sid, eligible, pool, args.seed)
            count = sum(exclusions[str(s)]["n_total"] -
                        (exclusions[str(s)]["n_excl_union"]
                         if args.condition == "clean_train" else 0)
                        for s in training)
            print(f"test=s{sid:02d} val=s{validation:02d} "
                  f"train_subjects={len(training)} train_trials={count} "
                  f"clean_test_trials={exclusions[str(sid)]['n_total'] - exclusions[str(sid)]['n_excl_union']}")
        return 0

    device = pick_device(args.model_type, args.device)
    if device.type == "mps" and args.model_type != "emg_only":
        raise RuntimeError("MPS is disallowed for EEGNet conditions due to grouped convolution errors")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.condition}_{args.model_type}_seed{args.seed}.csv"
    config = {"condition": args.condition, "model_type": args.model_type,
              "seed": args.seed, "max_epochs": args.max_epochs,
              "patience": args.patience, "pool": pool, "eligible": eligible,
              "clean_eeg_renormalization": True,
              "exclusions_sha256": hashlib.sha256(args.exclude_json.read_bytes()).hexdigest(),
              "data_dir": str(args.data_dir.resolve())}
    config_path = output.with_suffix(".config.json")
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
            completed = {int(row["sid"]) for row in csv.DictReader(stream)}
    for sid in targets:
        if sid in completed:
            print(f"s{sid:02d}: already complete")
            continue
        row = run_fold(sid, eligible, pool, exclusions, args.data_dir,
                       args.model_type, args.condition, args.seed,
                       args.max_epochs, args.patience, device)
        new_file = not output.exists()
        with output.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({key: row[key] for key in FIELDS})
        print(f"s{sid:02d} {args.condition} {args.model_type}: "
              f"accuracy={row['accuracy']:.4f}, best_epoch={row['best_epoch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
