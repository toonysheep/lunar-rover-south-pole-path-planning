"""Data access and checks for manuscript figures; no training dependencies."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = ROOT / "tmp_ppo+GRU/sensitive experiment"
_RECORDS = json.loads((ROOT / "manifests/source_files.json").read_text(encoding="utf-8"))
_SOURCE_INDEX = {
    r["original_path"].casefold(): r["path"]
    for r in _RECORDS
    if "original_path" in r
}


def source(path):
    """Resolve original experiment names through the portable, case-stable manifest."""
    relative = path.relative_to(ROOT).as_posix()
    resolved = ROOT / _SOURCE_INDEX.get(relative.casefold(), relative)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing figure input: {resolved}. See data/README.md.")
    key = resolved.relative_to(ROOT).as_posix()
    if key not in INPUTS:
        h = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        INPUTS[key] = {"bytes": resolved.stat().st_size, "sha256": h.hexdigest()}
    return resolved

METHODS = ["PPO+GRU", "PPO", "SAC+GRU", "SAC", "DQN", "Astar_shadow", "Astar"]

PARAMETERS = [
    ("a", "lr", "Learning rate", ["0.001", "0.0001", "0.00001"],
     [r"$\eta=10^{-3}$", r"$\eta=10^{-4}$", r"$\eta=10^{-5}$"]),
    ("b", "gamma", "Discount factor", ["0.95", "0.99", "0.995"],
     [r"$\gamma=0.95$", r"$\gamma=0.99$", r"$\gamma=0.995$"]),
    ("c", "lambda", "GAE weight", ["0.9", "0.95", "0.98"],
     [r"$\lambda=0.9$", r"$\lambda=0.95$", r"$\lambda=0.98$"]),
    ("d", "clip", "Clipping ratio", ["0.1", "0.2", "0.3"],
     [r"$\epsilon=0.1$", r"$\epsilon=0.2$", r"$\epsilon=0.3$"]),
    ("e", "entropy coeff", "Entropy coefficient", ["0.1", "0.01", "0.001"],
     [r"$c_H=0.1$", r"$c_H=0.01$", r"$c_H=0.001$"]),
]

DIRECTIONS = np.array([(0,0),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)])

INPUTS: dict[str, dict] = {}

AUDIT: dict[str, object] = {}

STD_BAND_MULTIPLIER = .95

def load_reward(path: Path) -> np.ndarray:
    values = np.loadtxt(source(path), delimiter=",")
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected a finite one-dimensional reward series: {path}")
    return values

def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    # Preserve pandas defaults (adjust=True) used by the original script.
    return pd.Series(values).ewm(alpha=alpha).mean().to_numpy()

def stability_data(site: str):
    runs = [ema(load_reward(SENSITIVE/f"Standard_{site}"/f"reward-{seed}.csv"),.1)
            for seed in range(1,6)]
    n = min(map(len,runs))
    runs = np.array([r[:n] for r in runs])
    mean, sd = runs.mean(axis=0), runs.std(axis=0,ddof=0)
    AUDIT[f"fig08/{site}"] = {"seeds": [1,2,3,4,5], "shared_episodes": n,
                               "ema_alpha": .1, "std_ddof": 0,
                               "band": "mean +/- 0.95 SD of individually smoothed runs",
                               "band_multiplier": STD_BAND_MULTIPLIER}
    return mean,sd

def site01_data():
    summary={}
    with source(ROOT/"results/tables/site01/result_info.csv").open(newline="") as f:
        for row in csv.reader(f):
            if row and row[0]!="Algorithm":
                summary[row[0]]=(int(row[1]),float(row[2]))
    data={}
    for method in METHODS:
        path=source(ROOT/f"tmp_{method}"/"site01/result_info.csv")
        with path.open(newline="") as f:
            rows=list(csv.DictReader(f))
        actions=np.array([int(r["action"]) for r in rows])
        expected=np.array((175,25))+np.cumsum(DIRECTIONS[actions],axis=0)
        positions=np.array([[int(x) for x in re.findall(r"-?\d+",r["Position"])] for r in rows])
        if not np.array_equal(expected,positions):
            raise ValueError(f"Recorded positions and reconstructed path disagree: {method}")
        values={key:np.array([float(r[key]) for r in rows])
                for key in ("Reward","Slope","Thermal","Battery")}
        fullpath=np.vstack(((175,25),expected))
        if len(fullpath)!=summary[method][0] or round(values["Reward"].sum(),2)!=summary[method][1]:
            raise ValueError(f"Data do not match manuscript Site01 summary: {method}")
        if positions[-1].tolist()!=[25,175]:
            raise ValueError(f"Unexpected endpoint: {method}")
        if any(not np.all(np.isfinite(v)) for v in values.values()):
            raise ValueError(f"Nonfinite trajectory data: {method}")
        data[method]={**values,"actions":actions,"path":fullpath}
        AUDIT[f"fig09/{method}"]={"path_points_including_start":len(fullpath),
             "recorded_actions":len(actions),"total_reward":round(values["Reward"].sum(),2),
             "slope_min":float(values["Slope"].min()),"slope_max":float(values["Slope"].max()),
             "matched_manuscript_summary":True}
    # CSV layout is [row, column * 1440 + time]. Read only queried rows;
    # retain the original plotting convention [path_row-1, path_col-1, time].
    queries={}
    for method,item in data.items():
        item["Illumination"]=np.full(len(item["path"]),np.nan)
        for t,(r,c) in enumerate(item["path"]):
            if not (1<=r<=200 and 1<=c<=200 and t<1440):
                raise ValueError(f"Path outside illumination data: {method}, {t}")
            queries.setdefault(int(r-1),[]).append((method,t,int((c-1)*1440+t)))
    light_path=source(ROOT/"site01/lunarSolarLight.csv")
    with light_path.open() as f:
        for r,line in enumerate(f):
            if r in queries:
                numbers=np.fromstring(line,sep=",")
                if len(numbers)!=200*1440:
                    raise ValueError("Unexpected illumination CSV layout")
                for method,t,index in queries[r]:
                    data[method]["Illumination"][t]=numbers[index]
    for method,item in data.items():
        if not np.all(np.isfinite(item["Illumination"])):
            raise ValueError(f"Missing illumination samples: {method}")
    return data
