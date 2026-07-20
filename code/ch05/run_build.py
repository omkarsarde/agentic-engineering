"""Run the deterministic Chapter 5 scaling-and-data build."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch

from ch05_render import save_pipeline_figure, save_scaling_figure, save_training_figure
from experiment_fixture import CLOZE_TASKS, FERTILITY_SAMPLES, MIXTURE_WEIGHTS
from mixture import measure_fertility
from pipeline_build import prepare_pipeline
from provenance import document_ledger, sha256_file, sha256_text, write_csv
from scaling_fit import extrapolation_interval, fit_scaling_law, load_observations
from synthetic_corpus import write_synthetic_wet
from training_probe import compare_toy_pretraining


HERE = Path(__file__).resolve().parent


def run_build(
    out_dir: Path = HERE / "generated",
    *,
    raw_target_bytes: int = 50 * 1024 * 1024,
    train_steps: int = 80,
    bootstrap_samples: int = 120,
    replicate_seeds: tuple[int, ...] = (17, 29, 43),
) -> dict[str, object]:
    """Run the local lab and write tables, metrics, prompts, and SVGs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    ladder_path = HERE / "fixtures" / "proxy_ladder.csv"
    observations = load_observations(ladder_path)
    law = fit_scaling_law(observations)
    extrapolation = extrapolation_interval(
        observations, bootstrap_samples=bootstrap_samples, seed=5
    )
    fit_rows = [
        {
            "parameters": int(row.parameters),
            "tokens": int(row.tokens),
            "compute_flops": row.compute_flops,
            "observed_loss": row.loss,
            "fitted_loss": float(law.loss(row.parameters, row.tokens)),
            "residual": row.loss - float(law.loss(row.parameters, row.tokens)),
        }
        for row in observations
    ]

    pipeline = prepare_pipeline(out_dir, raw_target_bytes)
    training_summary, curves, samples, tokenizer = compare_toy_pretraining(
        pipeline["raw_text"],
        pipeline["clean_text"],
        pipeline["holdout_text"],
        steps=train_steps,
        seeds=replicate_seeds,
    )
    fertility = measure_fertility(FERTILITY_SAMPLES, tokenizer.encode)
    ledger = document_ledger(
        pipeline["extracted"],
        pipeline["filter_rejections"],
        pipeline["cluster_rows"],
        pipeline["contamination_rejections"],
        pipeline["mixed"],
    )
    wet_path = pipeline["wet_path"]
    tokenizer_hash = sha256_text(
        json.dumps(
            {
                "vocab_size": tokenizer.vocab_size,
                "merges": [
                    [merge.left, merge.right, merge.new_id] for merge in tokenizer.merges
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    ledger_hash = sha256_text(json.dumps(ledger, sort_keys=True, separators=(",", ":")))

    metrics: dict[str, object] = {
        "experiment_contracts": {
            "scaling": {
                "claim": "synthetic proxy ladder demonstrates fitting mechanics, not frontier prediction",
                "observations": len(observations),
                "compute_multiplier": 10,
                "bootstrap_seed": 5,
            },
            "pipeline": {
                "claim": "synthetic WET-style fixture exercises policy gates; it is not scraped web data",
                "target_bytes": raw_target_bytes,
                "actual_bytes": wet_path.stat().st_size,
            },
            "training": {
                "claim": "local CPU toy paired-seed comparison with independent tokenizer and disjoint holdout",
                "paired_seeds": list(replicate_seeds),
                "steps": train_steps,
            },
        },
        "scaling_law": law.as_dict(),
        "extrapolation": extrapolation,
        "pipeline": {
            "stages": pipeline["stage_rows"],
            "filter_rejections": dict(
                Counter(row["reason"] for row in pipeline["filter_rejections"])
            ),
            "contamination_rejections": len(pipeline["contamination_rejections"]),
            "mixture_target_weights": MIXTURE_WEIGHTS,
            "mixture_sources": dict(
                Counter(document.source for document in pipeline["mixed"])
            ),
            "document_ledger_sha256": ledger_hash,
        },
        "toy_pretraining": training_summary,
        "fertility": fertility,
        "reproducibility": {
            "input_hashes": {
                "proxy_ladder_sha256": sha256_file(ladder_path),
                "synthetic_wet_sha256": sha256_file(wet_path),
                "raw_training_text_sha256": sha256_text(pipeline["raw_text"]),
                "clean_training_text_sha256": sha256_text(pipeline["clean_text"]),
                "disjoint_holdout_text_sha256": sha256_text(pipeline["holdout_text"]),
                "tokenizer_sha256": tokenizer_hash,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "hardware": {
                "device": "CPU",
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or "not reported",
                "torch_threads": torch.get_num_threads(),
            },
        },
    }
    write_csv(out_dir / "scaling-fit.csv", fit_rows)
    write_csv(out_dir / "pipeline-stages.csv", pipeline["stage_rows"])
    write_csv(out_dir / "dedup-clusters.csv", pipeline["cluster_rows"])
    write_csv(out_dir / "document-ledger.csv", ledger)
    write_csv(out_dir / "training-curves.csv", curves)
    write_csv(out_dir / "fertility.csv", fertility)
    (out_dir / "samples.txt").write_text(samples, encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_scaling_figure(observations, law, extrapolation, out_dir)
    save_pipeline_figure(pipeline["stage_rows"], pipeline["cluster_rows"], out_dir)
    save_training_figure(curves, fertility, training_summary, out_dir)
    wet_path.unlink()
    return metrics


if __name__ == "__main__":
    run_build()
