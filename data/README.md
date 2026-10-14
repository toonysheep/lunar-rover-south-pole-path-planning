# Environment inputs

Each of `site01`, `site06`, `site07` has four original, unmodified CSV files:

| File | Stored layout | Meaning |
| --- | --- | --- |
| `slope.csv` | 200 rows × 200 columns | Terrain slope in degrees |
| `Vsun.csv` | 1440 rows × 3 columns | Solar direction samples used by the original model |
| `lunarSolarLight.csv` | 200 rows × (200 × 1440) columns | Illumination, reshaped as [row, column, time] |
| `lunarSurfaceTemp.csv` | Same flattened layout | Surface temperature in degrees Celsius |

The map resolution is 5 m per cell; the time interval is 30 min. Start time 660 uses the same Site01 inputs with a temporal offset, not a duplicated dataset. The original environment repeats the temporal inputs and shifts them by `start_time`.

The environment files are excluded from normal Git and distributed as gzip-compressed Release assets. Their checksums, download names, and target paths are in `manifests/release_assets.json`.

The downloader decompresses each asset, verifies the restored CSV, and places it at the path specified in the manifest:

```bash
python scripts/download_data.py --repository OWNER/REPOSITORY --site site01
python scripts/check_data.py --sha256
```

Replace `OWNER/REPOSITORY` with the published repository name. The downloader requires the `environment-data-v0.1.0` Release and its 12 data assets.

Model checkpoints, raw DEM/TIFF stacks and data-preparation dependencies are outside this first release.
