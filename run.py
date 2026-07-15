#!/usr/bin/env python3
"""
run.py — Minimal MLOps-style batch job (MetaStackerBandit T0 assessment).

Loads a YAML config, reads an OHLCV CSV dataset, computes a rolling mean on
`close`, derives a binary trading signal, and writes structured metrics +
logs. Designed to be deterministic, observable, and Docker-deployable.

Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log

No paths are hard-coded — everything required is supplied via CLI args.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_KEYS = {"seed": int, "window": int, "version": str}
REQUIRED_COLUMN = "close"


class PipelineError(Exception):
    """Raised for any validation/processing error that should abort the run
    (but still result in a written metrics.json with status='error')."""


# --------------------------------------------------------------------------- #
# CLI + logging
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MLOps batch job: rolling-mean signal pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to input OHLCV CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write run log")
    return parser.parse_args(argv)


def setup_logging(log_file: str) -> logging.Logger:
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise PipelineError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PipelineError(f"Config file is not valid YAML: {exc}") from exc

    if not isinstance(config, dict):
        raise PipelineError("Config file must contain a top-level mapping (dict)")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise PipelineError(f"Config missing required field(s): {missing}")

    for key, expected_type in REQUIRED_CONFIG_KEYS.items():
        value = config[key]
        if expected_type is int and isinstance(value, bool):
            raise PipelineError(f"Config field '{key}' must be an int, got bool")
        if not isinstance(value, expected_type):
            raise PipelineError(
                f"Config field '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    if config["window"] < 1:
        raise PipelineError("Config field 'window' must be a positive integer")

    logger.info(
        "Config loaded + validated | seed=%s window=%s version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


def _read_csv_robust(input_path: str) -> pd.DataFrame:
    """Read a CSV that may have each row wrapped in outer double-quotes.

    The provided ``data.csv`` ships with every line (including the header)
    enclosed in quotes, e.g.::

        "timestamp,open,high,low,close,volume_btc,volume_usd"
        "2024-01-01 00:00:00,44910.83,45085.78,..."

    Standard ``pd.read_csv`` treats the whole quoted line as a single
    column.  We detect this case and strip the wrapping quotes before
    parsing so the pipeline works with *both* normal and quoted CSVs.
    """
    import io

    with open(input_path, "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    if not raw_lines:
        raise pd.errors.EmptyDataError("No data in CSV file")

    # Detect the "every-row-is-quoted" pattern by checking the header line.
    first = raw_lines[0].strip()
    if first.startswith('"') and first.endswith('"') and first.count('"') == 2:
        cleaned = []
        for line in raw_lines:
            s = line.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            cleaned.append(s)
        text = "\n".join(cleaned)
        return pd.read_csv(io.StringIO(text))

    # Fall through to the standard path for normal CSVs.
    return pd.read_csv(input_path)


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.isfile(input_path):
        raise PipelineError(f"Input data file not found: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise PipelineError(f"Input data file is empty: {input_path}")

    try:
        df = _read_csv_robust(input_path)
    except pd.errors.EmptyDataError as exc:
        raise PipelineError(f"Input CSV has no data/columns: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise PipelineError(f"Input CSV is malformed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface as a clean pipeline error
        raise PipelineError(f"Failed to read input CSV: {exc}") from exc

    if df.empty:
        raise PipelineError("Input CSV contains no rows")

    if REQUIRED_COLUMN not in df.columns:
        raise PipelineError(
            f"Input CSV missing required column '{REQUIRED_COLUMN}'. "
            f"Found columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df[REQUIRED_COLUMN]):
        df[REQUIRED_COLUMN] = pd.to_numeric(df[REQUIRED_COLUMN], errors="coerce")
        if df[REQUIRED_COLUMN].isna().all():
            raise PipelineError(f"Column '{REQUIRED_COLUMN}' is non-numeric")

    logger.info("Dataset loaded | rows=%d columns=%s", len(df), list(df.columns))
    return df


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    df["rolling_mean"] = df[REQUIRED_COLUMN].rolling(window=window, min_periods=window).mean()
    logger.info("Rolling mean computed | window=%d", window)

    # The first (window - 1) rows have no defined rolling mean (NaN).
    # Policy (documented + deterministic): those warm-up rows are KEPT in the
    # output (rows_processed == input row count) and their signal is
    # defaulted to 0 (no long signal) rather than dropped or forward-filled.
    df["signal"] = np.where(df[REQUIRED_COLUMN] > df["rolling_mean"], 1, 0)
    df.loc[df["rolling_mean"].isna(), "signal"] = 0

    warmup_rows = int(df["rolling_mean"].isna().sum())
    logger.info(
        "Signal generated | warmup_rows_defaulted_to_0=%d signal_rate=%.4f",
        warmup_rows, df["signal"].mean(),
    )
    return df


def write_metrics(output_path: str, metrics: Dict[str, Any], logger: logging.Logger) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", output_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_file)
    start_time = time.perf_counter()
    logger.info(
        "Job start | input=%s config=%s output=%s log_file=%s",
        args.input, args.config, args.output, args.log_file,
    )

    version = "unknown"
    try:
        config = load_config(args.config, logger)
        version = config["version"]

        np.random.seed(config["seed"])
        logger.info("Random seed set | seed=%d", config["seed"])

        df = load_dataset(args.input, logger)
        df = compute_signal(df, config["window"], logger)

        rows_processed = int(len(df))
        signal_rate = float(round(df["signal"].mean(), 4))
        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": elapsed_ms,
            "seed": config["seed"],
            "status": "success",
        }

        logger.info(
            "Metrics summary | rows_processed=%d signal_rate=%.4f latency_ms=%d",
            rows_processed, signal_rate, elapsed_ms,
        )

        write_metrics(args.output, metrics, logger)
        logger.info("Job end | status=success")
        print(json.dumps(metrics, indent=2))
        return 0

    except PipelineError as exc:
        logger.error("Validation/processing error: %s", exc)
        error_metrics = {"version": version, "status": "error", "error_message": str(exc)}
        write_metrics(args.output, error_metrics, logger)
        logger.info("Job end | status=error")
        print(json.dumps(error_metrics, indent=2))
        return 1

    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        logger.exception("Unexpected error: %s", exc)
        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": f"Unexpected error: {exc}",
        }
        write_metrics(args.output, error_metrics, logger)
        logger.info("Job end | status=error")
        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
