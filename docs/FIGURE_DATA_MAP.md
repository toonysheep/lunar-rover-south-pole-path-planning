# Figure and data mapping

| Figure | Archived input |
| --- | --- |
| 7 | `results/sensitivity/{lr,gamma,lambda,clip,entropy_coeff}/.../reward.csv` |
| 8 | `results/repeated_runs/{site01,site06,site07}/reward-{1..5}.csv` |
| 9 | `results/raw/*/site01/`, Site01 slope and illumination |
| 10 | `results/raw/*/site06/`, Site06 slope and illumination |
| 11 | `results/raw/*/site07/`, Site07 slope and illumination |
| 12 | Same Site01 trajectories as Fig.9, Site01 temperature/illumination snapshots |
| 14 | PPO+GRU, PPO and A*-shadow under `site01_t660/`, temporally shifted Site01 inputs |
| 15 | The corresponding T=0 and T=660 records from Figs.9 and 14 / Table6 |
| 16 | PPO+GRU from those same records, Site01 surface-temperature snapshots |

Figures 15 and 16 use the same records as Figures 9 and 14 and Table 6. For T=0, the PPO+GRU record has 497 points including the start; the Step=500 panel therefore reaches the goal.

The plotting style uses reward smoothing, sparse markers, differentiated line patterns and colors, local reward insets, and foreground ordering. The state panels (d–g) display their complete sequences without zoom insets. Figure 12 paths use 2.2 pt for PPO+GRU and 1.8 pt for the other methods; Figure 16 uses 2.2 pt. Figure 15 legends have an opaque white background.

The final typeset layout was assembled in Visio. The code produces component panels and reference assemblies; identical final spacing, vector annotations and manually placed icons are not claimed.
