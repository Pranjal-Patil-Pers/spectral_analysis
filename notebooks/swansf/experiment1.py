"""
Experiment 1: SWAN-SF partition4 preprocessing for three SHARP features.

This script selects all FL samples and stride-samples NF to match the FL count,
preserving chronological order across the full NF time range. Only TOTUSJH,
TOTPOT, and TOTUSJZ are retained. Each column is min-max normalized to [0, 1]
using statistics computed over all selected samples.

Outputs are written under data/swansf by default:
- selected_files_manifest.csv: selected source files and parsed metadata
- normalization_metadata.json: per-column min/max used for scaling
- sample_csv_files/{FL,NF}: normalized per-sample CSVs with only the 3 columns
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("TOTUSJH", "TOTPOT", "TOTUSJZ")
CLASS_NAMES = ("FL", "NF")
DEFAULT_PARTITION4_URL = (
    "https://www.dropbox.com/scl/fi/kmbx0skl79yk3ywk3jfkq/"
    "partition4_instances.tar.gz?rlkey=sx3lijmh7foyzy736hfym87cs&st=1gtxbj90&dl=0"
)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    class_label: str
    source_path: str
    filename: str
    category: str
    sub_category: str
    start_time: str
    end_time: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_start_end_time(filename: str) -> tuple[str, str]:
    match = re.search(r"_s(?P<start>[^_]+)_e(?P<end>[^.]+)", filename)
    if not match:
        return "", ""
    return match.group("start"), match.group("end")


def parse_category(filename: str) -> tuple[str, str]:
    if filename.startswith("FQ"):
        return "FQ", "NA"

    category = filename[:1]
    try:
        sub_category = filename.split("@", maxsplit=1)[0][1:]
    except IndexError:
        sub_category = ""
    return category, sub_category


def default_input_candidates(root: Path) -> list[Path]:
    return [
        root / "data" / "swansf" / "partition4",
        root / "data" / "raw" / "swansf" / "partition4",
        root / "partition4",
        Path("/content/partition4"),
    ]


def dropbox_direct_download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["dl"] = ["1"]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        dropbox_direct_download_url(url),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response, open(output_path, "wb") as f:
        shutil.copyfileobj(response, f)


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract linked tar member: {member.name}")
            member_path = (output_dir / member.name).resolve()
            if output_root != member_path and output_root not in member_path.parents:
                raise ValueError(f"Refusing to extract path outside output directory: {member.name}")
        tar.extractall(output_dir)


def download_partition4(download_url: str, root: Path) -> Path:
    swansf_dir = root / "data" / "swansf"
    partition_dir = swansf_dir / "partition4"
    if partition_dir.exists():
        return partition_dir.resolve()

    archive_path = swansf_dir / "downloads" / "partition4_instances.tar.gz"
    if not archive_path.exists():
        print(f"Downloading partition4 archive from: {download_url}")
        print(f"Saving archive to: {archive_path}")
        download_file(download_url, archive_path)

    print(f"Extracting partition4 archive into: {swansf_dir}")
    safe_extract_tar(archive_path, swansf_dir)

    if partition_dir.exists():
        return partition_dir.resolve()

    candidates = sorted(swansf_dir.rglob("partition4"))
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(f"Archive extracted, but no partition4 directory was found under {swansf_dir}.")


def resolve_input_dir(
    input_dir: str | None,
    root: Path,
    download_url: str,
    allow_download: bool,
) -> Path:
    if input_dir:
        path = Path(input_dir).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Input partition directory does not exist: {path}")

    for candidate in default_input_candidates(root):
        if candidate.exists():
            return candidate.resolve()

    if allow_download:
        return download_partition4(download_url, root)

    candidates = "\n".join(f"  - {path}" for path in default_input_candidates(root))
    raise FileNotFoundError(
        "Could not find partition4 data. Pass --input-dir explicitly or place it at one of:\n"
        f"{candidates}"
    )


def read_feature_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", usecols=list(FEATURE_COLUMNS))
    except ValueError as exc:
        raise ValueError(f"{path} is missing one or more required columns: {FEATURE_COLUMNS}") from exc


def collect_partition_files(input_dir: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []

    for class_label in CLASS_NAMES:
        class_dir = input_dir / class_label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing expected class directory: {class_dir}")

        for path in sorted(class_dir.glob("*.csv")):
            filename = path.stem
            start_time, end_time = parse_start_end_time(filename)
            category, sub_category = parse_category(filename)
            records.append(
                SampleRecord(
                    sample_id=f"{class_label}_{filename}",
                    class_label=class_label,
                    source_path=str(path),
                    filename=filename,
                    category=category,
                    sub_category=sub_category,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    return records


def select_stride_balanced(records: list[SampleRecord]) -> list[SampleRecord]:
    grouped = {
        class_label: sorted(
            [r for r in records if r.class_label == class_label],
            key=lambda r: (r.start_time, r.end_time, r.filename),
        )
        for class_label in CLASS_NAMES
    }

    fl_records = grouped["FL"]
    nf_records = grouped["NF"]
    target = len(fl_records)

    if len(nf_records) <= target:
        selected_nf = nf_records
    else:
        stride = len(nf_records) // target
        selected_nf = nf_records[::stride][:target]

    selected = fl_records + selected_nf
    return sorted(selected, key=lambda r: (r.start_time, r.end_time, r.class_label, r.filename))


def compute_normalization_stats(records: list[SampleRecord]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {
        col: {"min": np.inf, "max": -np.inf} for col in FEATURE_COLUMNS
    }
    for record in records:
        df = read_feature_frame(Path(record.source_path))
        for col in FEATURE_COLUMNS:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.empty:
                continue
            stats[col]["min"] = min(stats[col]["min"], float(values.min()))
            stats[col]["max"] = max(stats[col]["max"], float(values.max()))
    for col, s in stats.items():
        if not (np.isfinite(s["min"]) and np.isfinite(s["max"])):
            raise ValueError(f"No finite values found for required column {col}.")
    return stats


def apply_normalization(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        denom = float(stats[col]["max"] - stats[col]["min"])
        df[col] = 0.0 if denom == 0.0 else (values - float(stats[col]["min"])) / denom
    return df


def copy_normalized_sample_csv(
    record: SampleRecord,
    sample_output_dir: Path,
    stats: dict[str, dict[str, float]],
) -> None:
    df = read_feature_frame(Path(record.source_path))
    df = apply_normalization(df, stats)
    target_dir = sample_output_dir / record.class_label
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{record.sample_id}.csv"
    df.to_csv(target_path, index=False)


def write_outputs(
    selected_records: list[SampleRecord],
    output_dir: Path,
    normalization_stats: dict[str, dict[str, float]],
    copy_sample_csvs: bool,
) -> None:
    manifest_df = pd.DataFrame([asdict(record) for record in selected_records])
    manifest_df.to_csv(output_dir / "selected_files_manifest.csv", index=False)

    with open(output_dir / "normalization_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": list(FEATURE_COLUMNS),
                "normalization": "min_max_on_selected_stride_balanced_subset",
                "stats": normalization_stats,
            },
            f,
            indent=2,
        )

    if copy_sample_csvs:
        sample_output_dir = output_dir / "sample_csv_files"
        for record in selected_records:
            copy_normalized_sample_csv(record, sample_output_dir, normalization_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select all FL samples and stride-sample NF to match, preserving "
            "chronological spread across the full NF time range."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Path to extracted partition4 directory containing FL and NF subdirectories.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_PARTITION4_URL,
        help="URL for the partition4 tar.gz archive used when --input-dir is omitted and no local copy is found.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading partition4 when no local copy is found.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <project_root>/data/swansf/partition4_stride_<selected_count>.",
    )
    parser.add_argument(
        "--no-copy-sample-csvs",
        action="store_true",
        help="Only write the manifest; do not copy per-sample CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before writing new outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    input_dir = resolve_input_dir(
        input_dir=args.input_dir,
        root=root,
        download_url=str(args.download_url),
        allow_download=not bool(args.no_download),
    )

    records = collect_partition_files(input_dir)
    selected_records = select_stride_balanced(records)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "data" / "swansf" / f"partition4_stride_{len(selected_records)}"
    )

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Output directory already exists and is not empty — skipping: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization_stats = compute_normalization_stats(selected_records)
    write_outputs(
        selected_records=selected_records,
        output_dir=output_dir,
        normalization_stats=normalization_stats,
        copy_sample_csvs=not bool(args.no_copy_sample_csvs),
    )

    class_counts = pd.Series([r.class_label for r in selected_records]).value_counts().to_dict()
    print(f"Input partition: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Selected files: {len(selected_records)}")
    print(f"Class counts: {class_counts}")
    print(f"NF stride: {50096 // class_counts.get('NF', 1)}")
    for col, s in normalization_stats.items():
        print(f"  {col}: min={s['min']:.4g}, max={s['max']:.4g}")


if __name__ == "__main__":
    main()
