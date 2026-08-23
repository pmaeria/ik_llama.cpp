#!/usr/bin/env python3
"""Compare two EXP-0001 summary.json files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def index(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["config_id"]), str(item["test"])): item
        for item in summary.get("groups", [])
        if "median_avg_ts" in item
    }


def percent_change(before: float, after: float) -> float | None:
    if before == 0 or not math.isfinite(before) or not math.isfinite(after):
        return None
    return 100.0 * (after / before - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before_path = Path(args.before).expanduser().resolve()
    after_path = Path(args.after).expanduser().resolve()
    before = load_summary(before_path)
    after = load_summary(after_path)
    before_index = index(before)
    after_index = index(after)

    rows: list[dict[str, Any]] = []
    for config_id, test in sorted(set(before_index) | set(after_index)):
        left = before_index.get((config_id, test))
        right = after_index.get((config_id, test))
        row: dict[str, Any] = {"config_id": config_id, "test": test}
        if left:
            row["before_median_ts"] = float(left["median_avg_ts"])
        if right:
            row["after_median_ts"] = float(right["median_avg_ts"])
        if left and right:
            row["change_percent"] = percent_change(
                float(left["median_avg_ts"]),
                float(right["median_avg_ts"]),
            )
        rows.append(row)

    payload = {
        "before": str(before_path),
        "after": str(after_path),
        "rows": rows,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("| Config | Test | Before t/s | After t/s | Change |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        left = row.get("before_median_ts")
        right = row.get("after_median_ts")
        change = row.get("change_percent")
        left_text = f"{left:.3f}" if isinstance(left, (int, float)) else "-"
        right_text = f"{right:.3f}" if isinstance(right, (int, float)) else "-"
        change_text = f"{change:+.2f}%" if isinstance(change, (int, float)) else "-"
        print(f"| `{row['config_id']}` | `{row['test']}` | {left_text} | {right_text} | {change_text} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
