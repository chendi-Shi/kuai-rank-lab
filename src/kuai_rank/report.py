"""Aggregate compatible runs; write a self-contained experiment report."""
import html
import json
from pathlib import Path

import numpy as np

from .data import save_json


def compare_runs(root, out):
    files = sorted(Path(root).rglob("metrics.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not runs:
        raise ValueError(f"No completed runs under {root}")
    for field in ["training_event_ids_sha256", "data_manifest_sha256"]:
        if len({r[field] for r in runs}) != 1:
            raise ValueError(f"Incompatible runs: {field} differs")
    if len({(r['model'], r['seed']) for r in runs}) != len(runs):
        raise ValueError("Duplicate model/seed runs would bias aggregate statistics")
    rows = []
    for model in sorted({r["model"] for r in runs}):
        selected = [r for r in runs if r["model"] == model]
        for split in ["valid", "standard_aligned_valid", "random_valid", "test", "random_test"]:
            available = [r for r in selected if split in r["metrics"]]
            if not available:
                continue
            row = {"model": model, "split": split, "seeds": [r["seed"] for r in available],
                   "training_rows": available[0]["training_rows"],
                   "parameters": available[0]["parameters"], "tasks": available[0]["tasks"]}
            for metric in ["auc", "user_gauc", "logloss", "ece", "logged_user_day_ndcg@10"]:
                values = [r["metrics"][split][r["tasks"][0]][metric] for r in available]
                values = [v for v in values if v is not None]
                row[metric] = {"mean": float(np.mean(values)) if values else None,
                               "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            rows.append(row)
    result = {"primary_task": runs[0]["tasks"][0], "rows": rows,
              "run_files": [str(p) for p in files],
              "limitations": ["KuaiRand-Pure contains candidate-pool histories, not complete user sequences.",
                              "Compare standard_aligned_valid with random_valid for matched dates; item/user supports can still differ.",
                              "User-day NDCG scores logged exposure sets, not arbitrary or full-catalog candidates.",
                              "Random exposure supports policy-shift diagnosis, not an online A/B lift claim.",
                              "Histories replay earlier standard logs with previous-day feedback assumed finalized.",
                              "MMoE and shared-bottom can have different parameter counts; report them explicitly.",
                              "One seed is preliminary; between-seed std is not a confidence interval."]}
    save_json(out, result)
    headings = ["model", "split", "seeds", "training_rows", "parameters", "auc", "user_gauc", "logloss", "ece", "logged_user_day_ndcg@10"]
    body = []
    for row in rows:
        cells = []
        for key in headings:
            value = row[key]
            if isinstance(value, dict):
                value = f"{value['mean']:.5f}" if value["mean"] is not None else "N/A"
                if row[key]["std"] is not None:
                    value += f" ± {row[key]['std']:.5f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>KuaiRank 实验报告</title>
<style>body{font-family:system-ui;margin:40px;background:#f5f7fb;color:#142033}h1{font-size:28px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #dde3ed;text-align:left;font-size:13px}th{background:#182b4d;color:white}.note{padding:20px;background:white;border-left:4px solid #5278d9;line-height:1.8}li{margin:8px 0}</style>
<h1>KuaiRank · 真实曝光日志精排实验</h1><p>主任务：长播 long_view。以下指标由本机训练与评估产物自动生成。</p>
<div class="note">普通曝光与随机曝光分别报告。NDCG 是已曝光视频集合内的 user-day 排序指标，不能解释为全库推荐效果。± 为不同 seed 的样本标准差；单 seed 不显示波动。</div>
<h2>模型对照</h2><div style="overflow:auto"><table><thead><tr>"""
    document += "".join(f"<th>{html.escape(h)}</th>" for h in headings)
    document += "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div><h2>实验边界</h2><ul>"
    document += "".join(f"<li>{html.escape(text)}</li>" for text in result["limitations"])
    document += "</ul></html>"
    html_path = Path(out).with_suffix(".html")
    html_path.write_text(document, encoding="utf-8")
    print(f"Saved {out} and {html_path}", flush=True)
    return result
