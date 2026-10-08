# Raw data

The five raw CSV files are not stored in this repository (the telemetry file alone is 79 MB).

**Source:** Microsoft Azure AI predictive maintenance sample, mirrored on Kaggle as
*"Microsoft Azure Predictive Maintenance"* (uploaded by arnabbiswas1).

Download it and place these files in this folder before running the notebook:

| File | Rows | Contents |
|---|---|---|
| `PdM_telemetry.csv` | 876,100 | Hourly volt, rotate, pressure, vibration for 100 machines in 2015 |
| `PdM_errors.csv` | 3,919 | Non-fatal error events |
| `PdM_maint.csv` | 3,286 | Component replacements (scheduled and after failure) |
| `PdM_failures.csv` | 761 | Component failures (comp1–comp4) |
| `PdM_machines.csv` | 100 | Model and age of each machine |

The data is **synthetic**: Microsoft generated it to demonstrate predictive maintenance.
