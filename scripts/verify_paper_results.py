"""Check trajectory records against manuscript Tables 5 and 6, without ML imports."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
SLUG = {"PPO+GRU": "ppo_gru", "PPO": "ppo", "SAC+GRU": "sac_gru", "SAC": "sac",
        "DQN": "dqn", "Astar_shadow": "astar_shadow", "Astar": "astar"}


def pair(text):
    values = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(values) != 2:
        raise ValueError(f"Expected two coordinates/limits, found {text!r}")
    return tuple(float(x) for x in values)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=0.0051)


def inspect_trajectory(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trajectory: {path}")
    actions = [int(r["action"]) for r in rows]
    if any(a not in range(9) for a in actions):
        raise ValueError(f"Unknown action in {path}")
    positions = [tuple(int(v) for v in pair(r["Position"])) for r in rows]
    first_delta = DIRECTIONS[actions[0]]
    initial = tuple(p - d for p, d in zip(positions[0], first_delta))
    current = initial
    for index, (row, action, position) in enumerate(zip(rows, actions, positions), start=1):
        current = tuple(p + d for p, d in zip(current, DIRECTIONS[action]))
        if position != current or int(row["Step"]) != index:
            raise ValueError(f"Position/step inconsistency in {path.name}, row {index}")
        if any(not 1 <= v <= 200 for v in position):
            raise ValueError("Position outside interior map")
    goal = tuple(int(v) for v in pair(rows[-1]["Goal Position"]))
    if current != goal or any(tuple(int(v) for v in pair(r["Goal Position"])) != goal for r in rows):
        raise ValueError(f"Goal mismatch: {path}")
    numeric = {key: [float(r[key]) for r in rows] for key in ("Reward", "Slope", "Thermal", "Battery")}
    if any(not math.isfinite(v) for series in numeric.values() for v in series):
        raise ValueError(f"Nonfinite trajectory value: {path}")
    return {"actions": len(actions), "points_including_start": len(actions) + 1,
            "total_reward": round(math.fsum(numeric["Reward"]), 2), "stay_actions": actions.count(0),
            "initial": initial, "goal": goal,
            **{name.lower(): {"range": [min(numeric[key]), max(numeric[key])],
                              "mean": math.fsum(numeric[key]) / len(rows)}
               for name, key in (("battery", "Battery"), ("temperature", "Thermal"), ("slope", "Slope"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report; defaults to console only")
    args = parser.parse_args()
    references = json.loads((ROOT / "results/paper_reference.json").read_text(encoding="utf-8"))
    cache, checks, failures = {}, [], []
    for reference in references:
        scenario, method = reference["scenario"], reference["method"]
        key = (scenario, method)
        if key not in cache:
            cache[key] = inspect_trajectory(ROOT / "results/raw" / SLUG[method] / scenario / "result_info.csv")
        actual = cache[key]
        errors = []
        for actual_key, ref_key in (("points_including_start", "total_steps_including_start"),
                                     ("total_reward", "reward"), ("stay_actions", "stay_actions")):
            if not close(actual[actual_key], reference[ref_key]):
                errors.append(f"{actual_key}: {actual[actual_key]} != {reference[ref_key]}")
        for name in ("battery", "temperature"):
            expected_range = pair(reference[name + "_range"])
            if any(not close(a, b) for a, b in zip(actual[name]["range"], expected_range)):
                errors.append(f"{name} range")
            if not close(actual[name]["mean"], reference[name + "_mean"]):
                errors.append(f"{name} mean: {actual[name]['mean']} != {reference[name + '_mean']}")
        checks.append({"table": reference["table"], "scenario": scenario, "method": method,
                       "matched": not errors, "errors": errors})
        failures.extend(f"Table{reference['table']} {scenario} {method}: {error}" for error in errors)
    # Check the archived summary tables against the same manuscript references.
    for scenario in sorted({r["scenario"] for r in references}):
        with (ROOT / "results/tables" / scenario / "result_info.csv").open(newline="") as stream:
            table_rows = {r[0]: r for r in list(csv.reader(stream))[1:]}
        for reference in (r for r in references if r["scenario"] == scenario):
            row = table_rows[reference["method"]]
            expected = [reference["total_steps_including_start"], reference["reward"],
                        reference["total_steps_including_start"], reference["stay_actions"],
                        reference["battery_mean"], reference["temperature_mean"], reference["violation_rate"]]
            found = [row[i] for i in (1, 2, 3, 4, 8, 10, 11)]
            if any(not close(a, b) for a, b in zip(expected, found)):
                failures.append(f"Summary table mismatch: {scenario} {reference['method']}")
            for idx, field in ((7, "battery_range"), (9, "temperature_range")):
                if any(not close(a, b) for a, b in zip(pair(row[idx]), pair(reference[field]))):
                    failures.append(f"Summary range mismatch: {scenario} {reference['method']} {field}")
    report = {"unique_trajectories": len(cache), "manuscript_rows_checked": len(checks),
              "failures": failures, "checks": checks,
              "trajectories": [{"scenario": scenario, "method": method, **actual}
                               for (scenario, method), actual in cache.items()],
              "scope": "Saved records and table numbers; not a new training/evaluation reproduction"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(cache)} trajectories and {len(checks)} manuscript table rows; {len(failures)} mismatches.")
    for message in failures:
        print(message)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
