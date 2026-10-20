# Capacity Drop Modeling via Stochastic Disturbance Propagation

Data and Python code accompanying **“Capacity Drop as a Phase Transition: An Analytical Model via Stochastic Disturbance Propagation”**, submitted to *Communications in Transportation Research* (COMMTR).

The study connects microscopic stochastic vehicle interactions with macroscopic capacity drop through speed-disturbance propagation and correlation length. The repository contains analysis scripts for the I-24 MOTION and Zen traffic datasets, together with two processed leader–follower datasets in Apache Parquet format.

**Documentation status — 20 October 2026:** this README describes the public code at commit [`be47d33318549287baae5a12146d52de9bec785c`](https://github.com/tjzxh/Capacity-Drop-Modeling-via-Stochastic-Disturbance-Propagation/tree/be47d33318549287baae5a12146d52de9bec785c). Both supplied Parquet files are readable, and both theoretical capacity-drop scripts execute using stored parameters. Complete reproduction from trajectories is pending: the macroscopic trajectory inputs, several provenance details and the validated full-analysis environment are not yet included. The specific gaps and required release actions are documented below.

## 1. Study and replication-package metadata

| Field | Description |
|---|---|
| Proposed ETS-DATA package title | **Traffic Data and Code for Stochastic Capacity Drop Modeling** |
| Journal | Communications in Transportation Research |
| Manuscript title | Capacity Drop as a Phase Transition: An Analytical Model via Stochastic Disturbance Propagation |
| Manuscript ID | COMMTR-2026-0158.R2; R3 preparation. Confirm the revision suffix assigned by ScholarOne before deposit. |
| Proposed manuscript authors | Xiaohui Zhang, Jie Sun, Jian Sun and Yunpeng Wang |
| Corresponding author | Jian Sun; sunjian@tongji.edu.cn; Tongji University |
| Keywords | capacity drop; phase transition; stochastic traffic flow; disturbance propagation; correlation length; car following; freeway bottleneck; I-24 MOTION; Zen traffic data |
| Public code and processed data | [GitHub repository](https://github.com/tjzxh/Capacity-Drop-Modeling-via-Stochastic-Disturbance-Propagation) |
| Documented code version | `be47d33318549287baae5a12146d52de9bec785c` |
| Archived package DOI | **Pending ETS-DATA deposit.** |
| Article DOI | Not yet supplied. |
| Full replication validation | Pending; the checks in Section 7 have a narrower scope. |

## 2. Data description and provenance

### Included data

Each row in the supplied homogeneous datasets represents a processed leader–follower observation, rather than a complete vehicle trajectory. These files support inspection of microscopic traffic interactions. The I-24 microscopic calibration script reads its corresponding homogeneous file directly. The inspected repository does not include the Zen homogeneous-data generator or a Zen microscopic calibration workflow.

| File | Format | Rows × columns | Size (bytes) |
|---|---|---:|---:|
| `I-24Motion/data/I24_lane2_homo_data.parquet` | Apache Parquet | 170,660 × 11 | 7,775,733 |
| `Zen/data/Zen_homo_data.parquet` | Apache Parquet | 294,512 × 11 | 3,773,281 |

The inspected files have no null values in their 11 columns. This does not establish the absence of other missing-value codes or non-finite numeric values; those conventions require confirmation.

SHA-256 checksums:

```text
3a204087603b5d675d2679b5ae4aa603a3f6f8df1270d3c97cb8aa35a1dda5ba  I-24Motion/data/I24_lane2_homo_data.parquet
23fb860867455b6f56261e88dee561b62144788b467e532c8c16fdcb9953f4bc  Zen/data/Zen_homo_data.parquet
```

### Source observations and selection

| Dataset | Study coverage reported in the manuscript | Provenance still to supply |
|---|---|---|
| I-24 MOTION | 22 November 2022, westbound, 06:00–10:00, mile markers 58.8–59.2, lane 2 | Original provider record/DOI, release version, exact source-file identifiers and redistribution terms |
| Zen | Hanshin Expressway Route 11 Ikeda Line, Japan, KP 4.20–3.65, right passing lane, 10:00–11:00 | Collection date, provider record/DOI, release version, exact source-file identifiers and redistribution terms |

The I-24 preprocessing script documents vehicle trajectories with timestamps, positions, lengths and vehicle classes. Its active stage reads an intermediate file, `I-24Motion/data/I24_data.parquet`, and writes lane-specific trajectory files. The raw-JSON conversion stage is commented out and refers to a local source path.

The Zen preprocessing script reads `L001_F005_trajectory.csv` from a hard-coded local path, converts velocity from km/h to m/s and writes `Zen/data/Zen_data.parquet`. Its active filter includes positions 3000–4200 m, passing lane 2 and times at or after 10:00; the subsequent correlation and disturbance analyses restrict positions to 3650–4200 m. Do not infer a collection date from the parsed time-of-day values.

The I-24 homogeneous-data generator actively selects 07:00–08:00, retains vehicles observed for more than 10 s, constructs adjacent pairs and retains sedan pairs with positive net spacing. These active settings differ from some comments in the script. Its final comment records 912,917 homogeneous observations, whereas the deposited subset contains 170,660. The definitive filtering history must be reconciled before claiming that the uploaded subset has been regenerated.

### Required inputs not included at the documented commit

| Missing item | Role |
|---|---|
| `I-24Motion/data/I24_data_lane2.parquet` | Lane trajectories required by the I-24 macroscopic analysis and homogeneous-data generator |
| `Zen/data/Zen_data.parquet` | Trajectories required by the Zen macroscopic analysis |
| I-24 upstream intermediate/raw files and Zen source CSV | Inputs for reconstructing the missing trajectory files |
| Zen homogeneous-data generation and microscopic calibration code | Reconstructs and calibrates the supplied Zen leader–follower subset |
| Zen-compatible `funcs_for_correlation` module | Imported by Zen correlation and disturbance scripts; only an I-24 copy is present |

**Do not rename a homogeneous Parquet file to replace a missing trajectory file.** Their schemas and meanings differ. If original data cannot be redistributed, include the exact lawful acquisition route and executable transformations so a researcher with the same source data can reconstruct the analysis inputs. A specific restriction has not yet been documented for this package.

## 3. Usage notes and data dictionary

Column names are case-sensitive. I-24 units below are established by the included preprocessing and calibration code. The Zen macro preprocessing uses m and m/s, but the generator of the **Zen homogeneous file** is absent; its units and pairing conventions must be confirmed independently rather than assumed from the macro script.

| Field | I-24 type / meaning / stored unit | Zen type / interpretation requiring confirmation |
|---|---|---|
| `frame` / `FrameID` | `Int32` `frame`: relative frame index constructed using a 0.04 s interval | `Int64` `FrameID`: temporal identifier; exact encoding and origin to confirm |
| `subjectSpeed` | `Float64`; subject speed, ft/s | `Float64`; subject speed; unit to confirm |
| `leaderSpeed` | `Float64`; adjacent leader speed, ft/s | `Float64`; leader speed; unit and pairing convention to confirm |
| `speedDiff` | `Float64`; subject minus leader speed, ft/s | `Float64`; speed difference; sign convention and unit to confirm |
| `spacing` | `Float64`; difference between adjacent longitudinal positions, ft | `Float64`; spacing; reference points and unit to confirm |
| `subjectClass` | `Int64`; subject vehicle class; homogeneous filter selects `0` (sedan) | `Int64`; class code; homogeneous-file codebook to confirm |
| `relativeClass` | `Int64`; subject minus leader class code; homogeneous filter selects `0` | `Int64`; relative class indicator; definition to confirm |
| `Vehlength` | `Float64`; subject vehicle length, ft | `Float64`; vehicle length; vehicle identity and unit to confirm |
| `meanSpeed` / `laneSpeed` | `Float64` `meanSpeed`: mean speed of selected vehicles in the frame, ft/s | `Float64` `laneSpeed`: aggregate speed; population/window and unit to confirm |
| `vID` | `String`; subject vehicle identifier | `Int64`; vehicle identifier; scope to confirm |
| `net_spacing` | `Float64`; `spacing − Vehlength`, ft, following the implemented definition | `Float64`; net spacing; calculation and unit to confirm |

In I-24 calibration, multiply stored lengths by **0.3048** to convert ft to m. Multiply stored speeds by **0.3048** for m/s, and additionally by **3.6** for km/h. Preserve the selected unit system when interpreting the estimated model coefficients.

Expected macro input columns include `VehID`, `FrameID`, `VehClass`, `MeanSpeed`, `LaneID`, `Vehlen` and `LocalY`. Zen preprocessing also writes `Epoch` and `time`. I-24 macro scripts use longitudinal position in ft and speed in ft/s before conversion; Zen macro scripts use m and m/s. Exact source timestamp conventions and class codebooks must accompany the final release.

## 4. Code organization

The documented commit contains **19 Python files**:

```text
I-24Motion/
  filter_data_polar.py
  data/
    I24_lane2_homo_data.parquet
  Micro_calib/
    filter_homo_data.py
    mle_for_lambda.py
    lambda_spacing_regression.py
  Macro_calib/
    breakdown_idenfication.py
    empirical_capacity_drop.py
    fd_data.py
    fd_fit.py
    correlation_length.py
    disturb_propagation.py
    theory_capacity_drop.py
    funcs_for_correlation.py
Zen/
  filter_data_polar.py
  data/
    Zen_homo_data.parquet
  Macro_calib/
    breakdown_idenfication.py
    empirical_capacity_drop.py
    fd_data.py
    fd_fit.py
    correlation_length.py
    disturb_propagation.py
    theory_capacity_drop.py
README.md
```

The misspelling `breakdown_idenfication.py` is the actual filename and is retained in the commands below. The repository also contains an Excel temporary lock file, `I-24Motion/Macro_calib/~$avg_speed_I24.xlsx`; it is not an analysis workbook and should be excluded from the final deposit.

## 5. Code description: software and environment

The code uses Python. Imports identify NumPy, pandas, Polars, SciPy, Matplotlib, scikit-learn, lmfit and statsmodels. Excel operations additionally need compatible read/write engines, such as openpyxl, XlsxWriter and the engine selected by the installed Polars version. 

## 6. Replication instructions

### 6.1 Obtain the documented version

```bash
git clone https://github.com/tjzxh/Capacity-Drop-Modeling-via-Stochastic-Disturbance-Propagation.git
cd Capacity-Drop-Modeling-via-Stochastic-Disturbance-Propagation
git checkout be47d33318549287baae5a12146d52de9bec785c
```

Alternatively, obtain the archive for that commit through GitHub. The final ETS-DATA deposit should contain the frozen code and data themselves, with this README and its explanatory-file companion.

### 6.2 Run the stored-parameter theoretical calculations

From the repository root, in the environment described above:

```bash
python I-24Motion/Macro_calib/theory_capacity_drop.py
python Zen/Macro_calib/theory_capacity_drop.py
```

These calculations print results to the terminal and do not require the missing macro inputs. Expected outputs are in Section 7. Their success checks the analytical calculation with stored constants; it does not re-estimate those constants from observations.

### 6.3 Inspect the supplied Parquet data

With Polars installed, run this Python code from the repository root:

```python
from pathlib import Path
import hashlib
import polars as pl

for name in [
    "I-24Motion/data/I24_lane2_homo_data.parquet",
    "Zen/data/Zen_homo_data.parquet",
]:
    path = Path(name)
    data = pl.read_parquet(path)
    print(name, data.shape)
    print(data.schema)
    print(data.null_count())
    print("SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())
```

Compare the dimensions and hashes with Section 2.

### 6.4 I-24 microscopic calibration

The supplied I-24 homogeneous file is the direct input to `mle_for_lambda.py`. To produce the input expected by the next script, set **`unit_flag = 1`** in `mle_for_lambda.py` before running it. The checked-in default is `2`, which writes a different filename and unit system.

Then run from `I-24Motion/Micro_calib`:

```bash
python mle_for_lambda.py
python lambda_spacing_regression.py
```

With `unit_flag = 1`, the first script writes `../data/I24_lamda_lane2_m.txt`; the second reads that file, prints regression results and displays plots. The spelling `lamda` is part of the filename. This calibration sequence has been inspected but has not been rerun for this README. Record any configuration edit in the release. Rebuilding the homogeneous file itself also needs the missing `I24_data_lane2.parquet`.

### 6.5 Full macroscopic workflow — pending required inputs and environment

Run each case in its own `Macro_calib` directory because relative paths are resolved against the working directory. The intended dependency order, inferred from file reads and writes, is:

| Stage | Script | Required input | Result |
|---|---|---|---|
| Trajectory preparation | `filter_data_polar.py`, run from the case root | Original/intermediate source data | Lane/segment trajectory Parquet files |
| Traffic-state identification | `breakdown_idenfication.py` | Case trajectory file | `../data/avg_speed_I24.xlsx` or `avg_speed_Zen.xlsx`; displayed speed/wavelet plots |
| Empirical capacity drop | `empirical_capacity_drop.py` | Case trajectory file | Printed capacity, QDF and drop; displayed time series |
| Fundamental diagram | `fd_data.py` | Case trajectory file | `../data/fd_data.xlsx`; displayed plots |
| Fundamental-diagram fit | `fd_fit.py` | `../data/fd_data.xlsx` | Printed fitted parameters and capacity; displayed fitted curve |
| Correlation-length calculation | `correlation_length.py` | Case trajectory file; helper module; `fd_data.xlsx` for the later analysis | `all_corr_coef.xlsx`, `all_corr_dis.xlsx`, `correlation_length.xlsx` under the case’s `data/`, when the generation call is enabled |
| Disturbance propagation | `disturb_propagation.py` | Case trajectory file; helper; `fd_data.xlsx`; `correlation_length.xlsx` | Displayed disturbance plots and printed regression coefficients |
| Analytical prediction | `theory_capacity_drop.py` | Constants stored in the script | Printed theoretical capacity drop and QDF |

Before the first correlation run, enable the existing generation call in the appropriate `correlation_length.py`: `correlation_length_for_polars(data, 35)` for I-24 or `correlation_length_for_polars(data, 40)` for Zen. Both calls are commented out in the documented commit; otherwise the script tries to read a workbook that has not been generated. Confirm these settings against the definitive manuscript analysis before freezing the release.

After supplying the missing inputs, resolving the Zen helper import and validating dependencies, the commands within each `Macro_calib` folder are:

```bash
python breakdown_idenfication.py
python empirical_capacity_drop.py
python fd_data.py
python fd_fit.py
python correlation_length.py
python disturb_propagation.py
python theory_capacity_drop.py
```

This sequence is **not currently an end-to-end executable reproduction** of the public package. The theoretical scripts contain stored constants rather than loading newly generated calibration results. Transfer and validate the intended fitted parameters explicitly. In particular, the Zen FD-fit output comments and the constants stored in its theoretical script describe different parameter sets.

Generated Parquet files, text calibration files and spreadsheets can overwrite files with the same names. Run reconstruction in a separate working copy and retain the original supplied inputs and their checksums.

## 7. Expected results and verification scope

### Executed checks

Both theoretical scripts completed successfully on 20 October 2026 using the versions in Section 5:

| Quantity | I-24 MOTION | Zen |
|---|---:|---:|
| Theoretical capacity drop (%) | 15.255217481982896 | 21.002650176257355 |
| Theoretical QDF (veh/h) | 1467.1130420548352 | 1514.5482258105776 |
| Difference from the empirical drop constant (percentage points) | 2.2560437792783645 | 0.4973498237426419 |
| Relative QDF error against the empirical constant (%) | 0.14423495254847815 | 0.48754152140243895 |

The two local execution times were approximately 4.9 s and 1.7 s respectively, including interpreter startup. These are single smoke checks on the inspection machine, not runtime benchmarks or estimates for calibration.

The Zen result agrees with the current manuscript’s rounded **21.0%** and approximately **1514 veh/h**. The older comments in the Zen script instead report 20.827108% and 1508.493371 veh/h and should be corrected after the definitive parameter version is confirmed. The Zen script prints an estimated correlated-vehicle count of 13 but actually uses the stored value 12; retain and explain the intended distinction.

The empirical scripts contain recorded capacities/QDFs of **1776/1465 veh/h** for I-24 and **1920/1507.2 veh/h** for Zen. These are recorded reference values, **not newly recomputed empirical results**, because the trajectory inputs are absent. Absolute differences between two percentage-valued drop magnitudes are measured in **percentage points**.
