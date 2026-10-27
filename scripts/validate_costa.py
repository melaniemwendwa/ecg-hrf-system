"""
Check src.hrf_analysis.costa_hrf against Costa's
reference software, HRF.awk (PhysioNet:
heart-rate-fragmentation-code 1.0.0).

Both are run on Costa's three test files and on
random 2-minute windows from the local MIT-BIH and
Long-Term AF annotations, and every count behind
PIP, PNNSS and PNNLS must be identical.

Requires awk. Run from the project root:

    .venv/bin/python -m scripts.validate_costa
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests

from src.dataset import RAW_DIR, database_records, load_record_annotations
from src.hrf_analysis import costa_beat_codes, costa_hrf


COSTA_URL = "https://physionet.org/files/heart-rate-fragmentation-code/1.0.0/"
COSTA_DIR = RAW_DIR / "costa"
COSTA_FILES = [
    "HRF.awk",
    "timeseries_1PVC.txt",
    "timeseries_1PVC_1outlier.txt",
    "timeseries_1PVC_2outliers.txt"
]


def download_reference():
    """Fetch the reference software once."""

    COSTA_DIR.mkdir(parents=True, exist_ok=True)

    for name in COSTA_FILES:
        path = COSTA_DIR / name

        if not path.exists():
            response = requests.get(COSTA_URL + name, timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)


def run_reference(times, codes, fs):
    """Counts reported by HRF.awk, as strings."""

    script = (COSTA_DIR / "HRF.awk").read_text()
    script = script.replace("SF = 250;", f"SF = {fs:g};")

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "hrf.awk"
        beats = Path(tmp) / "beats.txt"

        program.write_text(script)
        beats.write_text("".join(
            f"{float(t)!r}\t{int(c)}\n" for t, c in zip(times, codes)
        ))

        output = subprocess.run(
            ["awk", "-f", str(program), str(beats)],
            capture_output=True,
            text=True,
            check=True
        ).stdout

    # "PIP = 33/53 = 62.264;   PNNSS = 24/43*100 = ...; PNNLS = ..."
    (ip, nn_total), (short, n_acc_dec), (long, n_dnn) = re.findall(
        r"= (\d*)/(\d*)", output
    )

    return [ip, nn_total, short, n_acc_dec, long, n_dnn]


def run_python(times, codes, fs):
    """The same counts from costa_hrf, formatted like AWK."""

    r = costa_hrf(times, codes, fs)

    # AWK prints counters that were never set as
    # empty strings. N_acc_dec is assigned once per
    # beat, so it is only empty with no beats at all.
    def as_awk(value):
        return str(value) if value else ""

    return [
        as_awk(r["inflection_count"]),
        as_awk(r["nn_total"]),
        as_awk(r["short"]),
        str(r["short"] + r["long"]) if len(times) else "",
        as_awk(r["long"]),
        as_awk(r["n_dnn"])
    ]


def random_windows(n_windows, seconds=120, seed=0):
    """Yield (label, times, codes, fs) from local annotations."""

    rng = np.random.default_rng(seed)

    variants = [("mitdb", 128), ("mitdb", None), ("ltafdb", 128)]

    for i in range(n_windows):
        database, common_fs = variants[i % len(variants)]
        records = database_records(database)
        record = records[rng.integers(len(records))]

        ann = load_record_annotations(record, database, common_fs)
        fs = ann["fs"]
        window = int(seconds * fs)
        start = rng.integers(0, ann["signal_length"] - window)

        first, last = np.searchsorted(
            ann["beat_samples"],
            [start, start + window]
        )

        times = (ann["beat_samples"][first:last] - start) / fs
        codes = costa_beat_codes(
            np.asarray(ann["beat_types"])[first:last]
        )

        yield f"{database} {record} at {fs:g} Hz", times, codes, fs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=495)
    args = parser.parse_args()

    download_reference()

    cases = []

    for name in COSTA_FILES[1:]:
        times, codes = np.loadtxt(COSTA_DIR / name, unpack=True)
        cases.append((name, times, codes.astype(int), 250))

    cases.extend(random_windows(args.windows))

    mismatches = 0

    for label, times, codes, fs in cases:
        reference = run_reference(times, codes, fs)
        ours = run_python(times, codes, fs)

        if reference != ours:
            mismatches += 1
            print(f"MISMATCH {label}: reference {reference}, ours {ours}")

    print(f"Checked {len(cases)} beat sequences: {mismatches} mismatches")

    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
