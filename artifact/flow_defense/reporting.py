import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config, ExperimentResult, TriggerSpec


def _to_serializable(value):
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def print_trigger_summary(trigger_spec: TriggerSpec) -> None:
    print("Trigger feature idx:", trigger_spec.feature_indices)
    print("Trigger feature names:", trigger_spec.feature_names)
    print("[Trigger Summary]")
    for feature_name, detail in zip(trigger_spec.feature_names, trigger_spec.trigger_details):
        valid_mark = "[OK]" if detail.get("trigger_valid", 1) else "[CLIPPED]"
        type_mark = detail.get("feature_type", "unknown")
        compliant_mark = "yes" if detail.get("protocol_compliant", 0) else "no"
        reason = detail.get("validation_reason", "ok")
        print(
            f"  {valid_mark} {feature_name} ({type_mark}): "
            f"raw_range=[{detail['train_min_raw']:.4f}, {detail['train_max_raw']:.4f}]  "
            f"attack_mean={detail['attack_mean_raw']:.4f}  benign_mean={detail['benign_mean_raw']:.4f}  "
            f"q={detail['quantile']:.3f}  raw_q={detail['raw_quantile']:.4f}  "
            f"trigger_z={detail['trigger_value']:.4f}  valid={reason}"
        )
        if type_mark != "numeric":
            print(
                f"    feature_min={detail.get('feature_min_raw', float('nan')):.4f}  "
                f"feature_max={detail.get('feature_max_raw', float('nan')):.4f}  "
                f"protocol_compliant={compliant_mark}"
            )
    if trigger_spec.feature_scores:
        print("[Flow-Aware Candidate Ranking]")
        for feature_name, score in trigger_spec.feature_scores[:8]:
            print(f"  {feature_name}: score={score:.4f}")


def print_stage_summary(results: Sequence[ExperimentResult], cfg: Config) -> None:
    print("\n===== Summary =====")
    for result in results:
        verdict = "PASS" if result.asr_is_strong else "FAIL"
        overhead_str = ""
        if result.round_rows:
            runtimes = [r["round_runtime_sec"] for r in result.round_rows if "round_runtime_sec" in r]
            if runtimes:
                overhead_str = f"  avg_round={np.mean(runtimes):.2f}s  total={sum(runtimes):.1f}s"
        print(
            f"{result.stage_name}: benign_acc={result.final_benign_acc:.4f}  "
            f"ASR={result.final_asr:.4f}  backdoor_baseline={verdict}{overhead_str}"
        )
    print(f"ASR decision threshold: {cfg.attack_success_threshold:.2f}")


def make_output_dir(cfg: Config) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cfg.export_root_dir / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def write_dataframe_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(Path(path), index=False)


def export_experiment_artifacts(cfg: Config, trigger_spec: TriggerSpec, results: Sequence[ExperimentResult], output_dir: str) -> None:
    trigger_rows = []
    for feature_name, detail in zip(trigger_spec.feature_names, trigger_spec.trigger_details):
        row = {"feature_name": feature_name}
        row.update(detail)
        trigger_rows.append(row)
    output_path = Path(output_dir)
    write_dataframe_csv(trigger_rows, str(output_path / "trigger_summary.csv"))

    if trigger_spec.feature_scores:
        write_dataframe_csv(
            [{"feature_name": name, "score": score} for name, score in trigger_spec.feature_scores],
            str(output_path / "flow_aware_ranking.csv"),
        )

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "stage_name": result.stage_name,
                "final_benign_acc": result.final_benign_acc,
                "final_asr": result.final_asr,
                "asr_is_strong": result.asr_is_strong,
                "poisoned_samples": result.poisoned_samples,
            }
        )
        stem = result.stage_name.replace(":", "").replace(" ", "_")
        if result.round_rows:
            write_dataframe_csv([{**row, "stage_name": result.stage_name} for row in result.round_rows], str(output_path / f"{stem}_rounds.csv"))
        if result.pruning_rows:
            write_dataframe_csv([{**row, "stage_name": result.stage_name} for row in result.pruning_rows], str(output_path / f"{stem}_pruning.csv"))
        if result.overlap_stats:
            write_dataframe_csv([{"stage_name": result.stage_name, **result.overlap_stats}], str(output_path / f"{stem}_overlap.csv"))

    write_dataframe_csv(summary_rows, str(output_path / "summary.csv"))

    overhead_rows: List[Dict[str, float]] = []
    for result in results:
        if result.round_rows:
            runtimes = [r["round_runtime_sec"] for r in result.round_rows if "round_runtime_sec" in r]
            if runtimes:
                overhead_rows.append({
                    "stage_name": result.stage_name,
                    "avg_round_sec": float(np.mean(runtimes)),
                    "total_sec": float(sum(runtimes)),
                    "rounds": float(len(runtimes)),
                })
    if overhead_rows:
        write_dataframe_csv(overhead_rows, str(output_path / "overhead_summary.csv"))

    payload = {
        "config": {key: _to_serializable(value) for key, value in vars(cfg).items()},
        "trigger": {
            "feature_indices": trigger_spec.feature_indices,
            "feature_names": trigger_spec.feature_names,
            "trigger_map": _to_serializable(trigger_spec.trigger_map),
            "trigger_details": _to_serializable(trigger_spec.trigger_details),
            "feature_scores": _to_serializable(trigger_spec.feature_scores),
        },
        "results": [
            {
                "stage_name": result.stage_name,
                "final_benign_acc": result.final_benign_acc,
                "final_asr": result.final_asr,
                "asr_is_strong": result.asr_is_strong,
                "poisoned_samples": result.poisoned_samples,
                "round_rows": _to_serializable(result.round_rows),
                "pruning_rows": _to_serializable(result.pruning_rows),
                "overlap_stats": _to_serializable(result.overlap_stats),
            }
            for result in results
        ],
    }
    with open(output_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] Saved plotting artifacts to: {output_dir}")


def aggregate_seed_results(seed_to_results: Dict[int, Sequence[ExperimentResult]]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    stage_names = []
    for results in seed_to_results.values():
        for result in results:
            if result.stage_name not in stage_names:
                stage_names.append(result.stage_name)

    summary_rows: List[Dict[str, float]] = []
    raw_rows: List[Dict[str, float]] = []
    for stage_name in stage_names:
        benign_values = []
        asr_values = []
        poisoned_values = []
        for seed, results in seed_to_results.items():
            matched = next((result for result in results if result.stage_name == stage_name), None)
            if matched is None:
                continue
            benign_values.append(matched.final_benign_acc)
            asr_values.append(matched.final_asr)
            poisoned_values.append(float(matched.poisoned_samples))
            raw_rows.append(
                {
                    "seed": float(seed),
                    "stage_name": stage_name,
                    "final_benign_acc": matched.final_benign_acc,
                    "final_asr": matched.final_asr,
                    "poisoned_samples": float(matched.poisoned_samples),
                }
            )

        if benign_values:
            summary_rows.append(
                {
                    "stage_name": stage_name,
                    "num_seeds": float(len(benign_values)),
                    "benign_acc_mean": float(np.mean(benign_values)),
                    "benign_acc_std": float(np.std(benign_values, ddof=0)),
                    "asr_mean": float(np.mean(asr_values)),
                    "asr_std": float(np.std(asr_values, ddof=0)),
                    "poisoned_samples_mean": float(np.mean(poisoned_values)),
                    "poisoned_samples_std": float(np.std(poisoned_values, ddof=0)),
                }
            )
    return summary_rows, raw_rows


def export_multi_seed_artifacts(seed_to_results: Dict[int, Sequence[ExperimentResult]], suite_output_dir: str) -> None:
    summary_rows, raw_rows = aggregate_seed_results(seed_to_results)
    suite_path = Path(suite_output_dir)
    write_dataframe_csv(raw_rows, str(suite_path / "multi_seed_raw.csv"))
    write_dataframe_csv(summary_rows, str(suite_path / "multi_seed_summary.csv"))
    payload = {"seed_list": list(seed_to_results.keys()), "summary": summary_rows, "raw": raw_rows}
    with open(suite_path / "multi_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n===== Multi-Seed Summary =====")
    for row in summary_rows:
        print(
            f"{row['stage_name']}: "
            f"benign_acc={row['benign_acc_mean']:.4f}+-{row['benign_acc_std']:.4f}  "
            f"ASR={row['asr_mean']:.4f}+-{row['asr_std']:.4f}"
        )
    print(f"[Export] Saved multi-seed summary to: {suite_output_dir}")
