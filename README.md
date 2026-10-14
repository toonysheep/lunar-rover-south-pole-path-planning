# Lunar Rover Path Planning in the South Polar Region

This repository provides the lunar rover environment, the experimental data used in the paper, and the code used to generate the numerical figures. The current release is intended for environment inspection, result verification, and figure reproduction. Training algorithms and trained checkpoints will be added separately.

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/lunar_rover_env/` | Portable entry point for the lunar rover environment |
| `vendor/gym_minigrid/` | Environment dynamics, observations, reward, thermal model, and battery model |
| `vendor/gym/` | Gym 0.18.0 runtime used by the environment |
| `data/environments/` | Slope, solar-direction, illumination, and surface-temperature inputs |
| `results/raw/` | Recorded trajectories and episode rewards |
| `results/sensitivity/` | PPO hyperparameter-sensitivity records |
| `results/repeated_runs/` | Five PPO+GRU reward runs for each site |
| `results/tables/` | Summary data corresponding to the paper tables |
| `plotting/` | Figure-generation code that reads the archived data directly |
| `manifests/` | File sizes, SHA256 checksums, and Release-asset names |

## Paper figures and data

| Paper figure | Data used by the plotting code |
| --- | --- |
| Fig. 7 | `results/sensitivity/` |
| Fig. 8 | `results/repeated_runs/site01`, `site06`, and `site07` |
| Fig. 9 | All methods in `results/raw/*/site01/`, plus the Site01 slope and illumination maps |
| Fig. 10 | All methods in `results/raw/*/site06/`, plus the Site06 slope and illumination maps |
| Fig. 11 | All methods in `results/raw/*/site07/`, plus the Site07 slope and illumination maps |
| Fig. 12 | Site01 trajectories from Fig. 9, with Site01 illumination and temperature snapshots |
| Fig. 14 | PPO, PPO+GRU, and A*-shadow in `results/raw/*/site01_t660/`, using Site01 maps with a 660-step temporal offset |
| Fig. 15 | The corresponding T=0 and T=660 trajectories from Figs. 9 and 14 |
| Fig. 16 | The PPO+GRU trajectories used in Figs. 9, 14, and 15, shown on Site01 temperature maps |

`plotting/paper_figures.py` reads these saved CSV files and does not run a learning algorithm. The plotting parameters, data conventions, and more detailed mapping are documented in [`docs/FIGURE_DATA_MAP.md`](docs/FIGURE_DATA_MAP.md) and [`docs/DATA_CONVENTIONS.md`](docs/DATA_CONVENTIONS.md).

## Environment-map data

Each site contains four inputs:

| File | Purpose | Distribution |
| --- | --- | --- |
| `slope.csv` | 200 × 200 terrain-slope map used for traversability, slope observations, rewards, and trajectory backgrounds | GitHub Release asset |
| `Vsun.csv` | 1440 solar-direction samples used by the thermal and power model | GitHub Release asset |
| `lunarSolarLight.csv` | Time-varying illumination over the 200 × 200 map; used by the environment and illumination panels | GitHub Release asset |
| `lunarSurfaceTemp.csv` | Time-varying surface temperature over the 200 × 200 map; used by the environment, thermal calculation, Fig. 12, and Fig. 16 | GitHub Release asset |

Download the Release assets and verify them against the SHA-256 checksums in `manifests/release_assets.json`:

```bash
python scripts/download_data.py --repository OWNER/REPOSITORY
python scripts/check_data.py --sha256
```

The downloader decompresses each Release asset, places the CSV in the path expected by the environment and plotting code, and verifies the original file. Use `--site site01`, `--site site06`, or `--site site07` to download and verify one site only. See [`data/README.md`](data/README.md) for shapes, units, and indexing.

## Usage

Python 3.10 is recommended.

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt

# Check the experiment records against the paper tables.
python scripts/verify_paper_results.py

# Figures 7 and 8 use only data stored in Git.
python plotting/paper_figures.py --figures 7 8

# After downloading the environment-map assets:
python scripts/smoke_env.py --site site01 --steps 2
python plotting/paper_figures.py --figures 9 10 11 12 14 15 16
```

The figure script writes high-resolution PNG, SVG, PDF, preview, and individual-panel files to `outputs/paper_figures/`. Final manuscript assembly may apply additional panel spacing and annotations in layout software.

Third-party licenses and provenance are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
