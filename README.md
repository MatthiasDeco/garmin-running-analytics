# garmin-running-analytics
Personal pipeline to extract, process, and store running data from Garmin Connect. Downloads `.fit` activity files, computes training metrics, and persists everything to a local SQLite database with CSV exports for analysis.

---

## What it does

1. **Downloads** all Garmin Connect activities as `.fit` files (skips already-downloaded ones)
2. **Processes** each `.fit` file into second-by-second time series (pace, BPM, altitude, cadence, stride, energy...)
3. **Computes** per-activity summary stats: VDOT, TSS, training load, training zones, best marks, running economy
4. **Stores** processed and summary data in a local SQLite database (incremental — only new activities)
5. **Exports** SQLite tables to CSV for use in Excel, Power BI, or any analysis tool

---

## Project structure

```
garmin-running-analytics/
├── main_data_processor.py
├── .env.dev
├── .env.prod
├── .env.example
├── run_dev.bat
├── run_prod.bat
├── requirements.txt
├── README.md
├── src/
    ├── fgarmin.py
    ├── garmin_transformation.py
    └── fsql.py

```

---

## Setup

**1. Clone and create virtual environment**
```bash
git clone https://github.com/YOUR_USERNAME/garmin-running-analytics.git
cd garmin-running-analytics
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. Configure `.env`**

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

```env
# Garmin Connect credentials
garmin_mail=your@email.com
garmin_password=yourpassword

# Paths
fit_path=data/fit_files
ddbb_sqlite=data/garmin.db
output_summary_csv_path=output/summary.csv
output_processed_csv_path=output/processed.csv

# Personal FTP values (used for zones and training load)
FTP_bpm=172
FTP_pace=4.5
FTP_rap=270

# Body weight in kg
weight=70

# SQLite table names
summary_table_sqlite=summary_data
processed_table_sqlite=processed_data
```

**3. Run**
```bash
python main_data_processor.py
```

---

## Computed metrics

### Processed data (second-by-second)
| Column | Description |
|--------|-------------|
| `instant_pace` | Running pace (min/km), zeroed when stopped |
| `instant_bpm` | Heart rate |
| `instant_cadence` | Steps per minute (both feet) |
| `instant_stride` | Stride length in meters |
| `instant_level` | Instantaneous gradient (m/s) |
| `accumulated_positive_level` | Cumulative elevation gain |
| `instant_MET` | Metabolic equivalent (polynomial model) |
| `instant_consum_per_kg` | Caloric consumption per kg bodyweight |

### Summary data (per activity)
| Column | Description |
|--------|-------------|
| `vdot` | VO2max proxy via Jack Daniels VDOT table |
| `training_load` | Average of TSS by pace and TSS by BPM |
| `TSS_pace` / `TSS_bpm` | Training Stress Score by pace / heart rate |
| `intensity_factor` | Average of IF pace and IF BPM |
| `PI` | Performance Index vs personal FTP reference |
| `average_rap` | Pace adjusted for elevation (RAP) |
| `equivalence_distance` | Flat-equivalent distance accounting for climb |
| `standardized_power` | Power normalized to 170 BPM |
| `time_zN_bpm/pace` | Time (seconds) spent in each of 5 training zones |

### Activity classification
Activities are auto-classified based on speed, altitude, pace, and elevation ratio:
`Road` · `Trail` · `Walk` · `Snow` · `Climb` · `Gym` · `Other`

---

## Incremental processing

The pipeline checks existing `file_id` values in the SQLite summary table before processing. Only new `.fit` files are processed — re-running is safe and fast.

`file_id` format: `YYYYMMDDHHmm` (timestamp of first GPS record)

---

## Dependencies

```
fitparse==1.2.0
garminconnect==0.3.3
numpy==2.4.4
pandas==3.0.2
python-dotenv==1.2.2
pytz==2026.2
```

---

## Notes

- Garmin enforces rate limits on their API.
- The processed table can grow large (one row per second per activity). The CSV export supports a `limit` parameter for partial exports.
- FTP values and activity classification thresholds are personal — review `garmin_transformation.py` sections marked `VERY PERSONAL CRITERIA` before first use.