"""
ICMECAT Time Series Builder
============================
Reads the HELIO4CAST ICMECAT v2.3 catalog, fetches hourly Wind solar wind
speed data from NASA OMNIWeb for each event window, and saves one CSV per
event with a binary label (1 = ICME, 0 = quiet solar wind).

Output structure
----------------
output_dir/
    icme/
        ICME_Wind_1997-01-10_12-00.csv   # label = 1
        ICME_Wind_1997-02-09_15-00.csv
        ...
    quiet/
        QUIET_Wind_1997-01-08_12-00.csv  # label = 0
        ...
    labels.csv                           # master label file

Each per-event CSV has columns:
    timestamp          : ISO 8601 datetime
    icme_speed_mean    : solar wind proton speed (km/s)
    label              : 1 (ICME) or 0 (quiet solar wind)

Usage
-----
    pip install pandas numpy requests tqdm
    python icmecat_timeseries_builder.py

Notes
-----
- Only Wind spacecraft events are used (most complete hourly coverage 1995-2023).
- Quiet windows are taken as the 24-hour period immediately BEFORE each ICME
  start time (same length as the event, capped at 48 h, floored at 6 h).
- OMNIWeb hourly data (OMNI2) is used for the raw speed time series.
- Events with >30% missing speed data are skipped and logged.
- Rate-limit: 1-second sleep between OMNIWeb requests to be a good citizen.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

ICMECAT_URL   = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "ICMECAT"
OMNI2_LOCAL   = PROJECT_ROOT / "data" / "ICMECAT" / "omni2_all_years.dat"
SPACECRAFT    = "Wind"           # filter to Wind only for clean hourly coverage
MAX_EVENTS    = None             # set to e.g. 50 for a quick test run
MISSING_FRAC  = 0.30             # skip event if >30% of timesteps are NaN speed
QUIET_PAD_H   = 6               # minimum quiet window size in hours
QUIET_CAP_H   = 48              # maximum quiet window size in hours

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "icmecat_builder.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_omni2(path: str) -> pd.DataFrame:
    """
    Load the full OMNI2 hourly flat file (omni2_all_years.dat) into memory once.
    Download from: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat

    Columns used: 0=Year, 1=DOY, 2=Hour, 24=Flow Speed (km/s, fill=9999.9).
    """
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 25:
                continue
            try:
                year  = int(parts[0])
                doy   = int(parts[1])
                hour  = int(parts[2])
                speed = float(parts[24])
            except ValueError:
                continue
            dt = pd.Timestamp(year=year, month=1, day=1) + timedelta(days=doy - 1, hours=hour)
            speed = np.nan if speed >= 9999.0 else speed
            rows.append((dt, speed))

    df = pd.DataFrame(rows, columns=["timestamp", "icme_speed_mean"])
    df.set_index("timestamp", inplace=True)
    return df


def slice_omni2(omni2: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Slice the in-memory OMNI2 DataFrame to [start_dt, end_dt]."""
    mask = (omni2.index >= start_dt) & (omni2.index <= end_dt)
    return omni2.loc[mask].reset_index()


def window_duration_hours(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / 3600.0


def save_event_csv(df: pd.DataFrame, label: int, name: str, subdir: str):
    """Save a per-event CSV with label column appended."""
    df = df.copy()
    df["label"] = label
    path = os.path.join(OUTPUT_DIR, subdir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def build_event_name(prefix: str, sc: str, dt: pd.Timestamp) -> str:
    return f"{prefix}_{sc}_{dt.strftime('%Y-%m-%d_%H-%M')}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Create output directories
    for sub in ("icme", "quiet"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # 2. Load OMNI2 flat file into memory
    log.info(f"Loading OMNI2 from {OMNI2_LOCAL} …")
    if not os.path.exists(OMNI2_LOCAL):
        log.error(f"OMNI2 file not found: {OMNI2_LOCAL}")
        log.error("Download with: wget -O data/ICMECAT/omni2_all_years.dat https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat")
        raise FileNotFoundError(OMNI2_LOCAL)
    omni2 = load_omni2(OMNI2_LOCAL)
    log.info(f"OMNI2 loaded: {len(omni2):,} hourly rows")

    # 3. Load ICMECAT
    log.info("Fetching ICMECAT catalog …")
    try:
        ic = pd.read_csv(ICMECAT_URL)
    except Exception as e:
        log.error(f"Could not fetch ICMECAT: {e}")
        log.error("Download it manually from https://helioforecast.space/icmecat")
        log.error("and replace ICMECAT_URL with the local path, e.g. 'HELIO4CAST_ICMECAT_v23.csv'")
        raise

    log.info(f"Catalog loaded: {len(ic)} total events, columns: {list(ic.columns)}")

    # 4. Filter to Wind spacecraft
    wind = ic[ic["sc_insitu"].str.strip().str.lower() == SPACECRAFT.lower()].copy()
    log.info(f"Wind events: {len(wind)}")

    # 5. Parse timestamps
    for col in ("icme_start_time", "mo_start_time", "mo_end_time"):
        wind[col] = pd.to_datetime(wind[col], errors="coerce", utc=True).dt.tz_localize(None)

    wind = wind.dropna(subset=["icme_start_time", "mo_end_time"])
    log.info(f"Wind events with valid timestamps: {len(wind)}")

    if MAX_EVENTS:
        wind = wind.head(MAX_EVENTS)
        log.info(f"Capped to {MAX_EVENTS} events for test run")

    # 6. Process each event
    master_records = []
    skipped = 0

    for _, row in tqdm(wind.iterrows(), total=len(wind), desc="Events"):
        icme_start = row["icme_start_time"]
        icme_end   = row["mo_end_time"]
        event_h    = window_duration_hours(icme_start, icme_end)

        if event_h < 1:
            log.warning(f"Skipping {icme_start}: duration < 1h")
            skipped += 1
            continue

        # ── ICME window ──────────────────────────────────────────────────────
        icme_df = slice_omni2(omni2, icme_start, icme_end)

        if icme_df.empty:
            skipped += 1
            continue

        miss_frac = icme_df["icme_speed_mean"].isna().mean()
        if miss_frac > MISSING_FRAC:
            log.warning(f"Skipping ICME {icme_start}: {miss_frac:.0%} missing speed")
            skipped += 1
            continue

        # Forward-fill then back-fill residual NaNs
        icme_df["icme_speed_mean"] = (
            icme_df["icme_speed_mean"].ffill().bfill()
        )

        name_icme = build_event_name("ICME", SPACECRAFT, icme_start)
        save_event_csv(icme_df, label=1, name=name_icme, subdir="icme")
        master_records.append({
            "file": f"icme/{name_icme}.csv",
            "label": 1,
            "icme_start_time": icme_start,
            "icme_end_time": icme_end,
            "duration_hours": round(event_h, 2),
            "n_timesteps": len(icme_df),
            "speed_mean_kms": round(icme_df["icme_speed_mean"].mean(), 1),
        })

        # ── Quiet window (same duration, immediately before ICME start) ──────
        quiet_h     = max(QUIET_PAD_H, min(event_h, QUIET_CAP_H))
        quiet_end   = icme_start - timedelta(hours=1)
        quiet_start = quiet_end - timedelta(hours=quiet_h)

        quiet_df = slice_omni2(omni2, quiet_start, quiet_end)

        if quiet_df.empty:
            continue

        miss_frac_q = quiet_df["icme_speed_mean"].isna().mean()
        if miss_frac_q > MISSING_FRAC:
            log.warning(f"Skipping quiet window {quiet_start}: {miss_frac_q:.0%} missing")
            continue

        quiet_df["icme_speed_mean"] = (
            quiet_df["icme_speed_mean"].ffill().bfill()
        )

        name_quiet = build_event_name("QUIET", SPACECRAFT, quiet_start)
        save_event_csv(quiet_df, label=0, name=name_quiet, subdir="quiet")
        master_records.append({
            "file": f"quiet/{name_quiet}.csv",
            "label": 0,
            "icme_start_time": quiet_start,
            "icme_end_time": quiet_end,
            "duration_hours": round(quiet_h, 2),
            "n_timesteps": len(quiet_df),
            "speed_mean_kms": round(quiet_df["icme_speed_mean"].mean(), 1),
        })

    # 7. Save master label file
    labels_df = pd.DataFrame(master_records)
    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    labels_df.to_csv(labels_path, index=False)

    # 8. Summary
    if labels_df.empty:
        log.error(f"No events saved — all {skipped} events were skipped. Check OMNIWeb connectivity or response format.")
        return
    n_icme  = (labels_df["label"] == 1).sum()
    n_quiet = (labels_df["label"] == 0).sum()
    log.info("=" * 60)
    log.info(f"Done.  ICME files : {n_icme}")
    log.info(f"       Quiet files: {n_quiet}")
    log.info(f"       Skipped    : {skipped}")
    log.info(f"       Labels CSV : {labels_path}")
    log.info("=" * 60)

    print("\nSample from labels.csv:")
    print(labels_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()