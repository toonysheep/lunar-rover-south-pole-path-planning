# Data conventions and scope

- `result_info.csv` has one row per executed action; positions describe the state after that action. Reconstruct the start as `first_position - direction[first_action]`.
- The paper's “Total Step” values count the initial path point: `number_of_actions + 1`. Keep this convention when comparing figures and tables; do not append a fabricated action or reward for the initial point.
- Action indices: 0 stay, 1 up, 2 up-right, 3 right, 4 down-right, 5 down, 6 down-left, 7 left, 8 up-left. Coordinates in the recorded paths are row/column pairs using the original 1-based interior grid convention.
- `Reward` is the per-action reward. Its sum, rounded to two decimals, gives the reported cumulative reward. `reward.csv` instead stores one scalar episode reward per line, without a header.
- Recorded `Thermal` and the surface-temperature CSVs are in Celsius. The environment also uses Kelvin internally. Battery values in the records are percentages; slope values are degrees.
- The trajectory and reward CSVs are preserved without numerical modification. The summary CSVs correspond to the paper tables. Their header repeats “Average slope”; the second occurrence is the average thermal value, so the validation script reads columns by position.
- Five reward records (`reward-1.csv` through `reward-5.csv`) per site are included for Figure 8. Filenames are run identifiers; the release does not independently establish the original random-seed configuration.
- Temporal map indexing for a trajectory sample is `(start_time + step) % 1440`, with flattened column index `(column - 1) * 1440 + time`.

The environment retains the current research-source equations and parameter snapshot. In particular, `simulation_param.py` contains a shared topocentric-frame setting of latitude -85°, longitude 35°; the adapter does not replace it with per-site values. Exact regeneration of all experiment records by new episodes/training has not been established. The independently supported task in this release is checking and plotting the archived paper records.
