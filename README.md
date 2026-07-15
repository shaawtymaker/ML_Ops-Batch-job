# MLOps Batch Signal Pipeline — T0 Technical Assessment

A minimal, reproducible MLOps-style batch job that reads OHLCV data, computes
a rolling-mean based binary trading signal, and emits structured metrics +
logs. Built to mirror the signal-pipeline style work used in
**MetaStackerBandit**.

## What it does

1. Loads and validates `config.yaml` (`seed`, `window`, `version`)
2. Sets `numpy.random.seed(seed)` for deterministic behaviour
3. Loads and validates `data.csv` (must contain a `close` column)
4. Computes a rolling mean of `close` over `window` rows
5. Derives a binary `signal`: `1` if `close > rolling_mean`, else `0`
   - The first `window - 1` "warm-up" rows have no defined rolling mean;
     they are **kept** in the output and their signal is deterministically
     defaulted to `0` (documented in code + logs) so `rows_processed`
     always equals the input row count.
6. Writes `metrics.json` (always — success **or** error) and a detailed
   `run.log`
7. Prints the final metrics JSON to stdout and exits `0` on success / `1`
   on failure

## Repository layout

```
├── run.py              # entry point
├── config.yaml         # seed / window / version
├── data.csv            # 10,000-row OHLCV dataset (provided)
├── requirements.txt    # Python dependencies
├── Dockerfile          # one-command Docker build & run
├── .dockerignore       # keeps image lean
├── .gitignore          # standard git patterns
├── metrics.json        # sample output from a successful run
├── run.log             # sample log from a successful run
└── README.md
```

## Local run instructions

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

The command is fully parameterized via CLI flags — there are **no
hard-coded paths** anywhere in `run.py`, so it can be pointed at any input,
config, or output location.

## Docker build & run

Build:

```bash
docker build -t mlops-task .
```

Run:

```bash
docker run --rm mlops-task
```

This will:
- Use the `data.csv` and `config.yaml` baked into the image
- Produce `/app/metrics.json` and `/app/run.log` inside the container
- Print the final metrics JSON to stdout
- Exit `0` on success, non-zero on any failure

To pull the generated files back out of the container onto the host:

```bash
docker run --name mlops-task-run mlops-task
docker cp mlops-task-run:/app/metrics.json ./metrics.json
docker cp mlops-task-run:/app/run.log ./run.log
docker rm mlops-task-run
```

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 29,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input CSV missing required column 'close'. Found columns: ['timestamp', 'open']"
}
```

`metrics.json` is written in **both** the success and error paths, and the
process exit code reflects the outcome (`0` / non-zero).

## Error handling covered

- Missing config file / missing input file
- Invalid YAML structure or missing required config fields
  (`seed`, `window`, `version`) / wrong types
- `window < 1`
- Empty input file
- Malformed / unparseable CSV
- Quoted-row CSV format (auto-detected and handled)
- Missing or non-numeric `close` column
- Any unexpected exception (caught, logged with stack trace, still writes
  `metrics.json` with `status: "error"`)

## Reproducibility

- `numpy.random.seed(config["seed"])` is set immediately after config
  validation
- Rolling mean / signal logic is purely deterministic pandas/numpy math —
  no randomness is introduced downstream
- Given the same `data.csv` + `config.yaml`, `signal_rate` and
  `rows_processed` will be identical across runs; only `latency_ms` is
  expected to vary run-to-run

## Observability

`run.log` records (with timestamps):
- Job start (with all CLI args)
- Config loaded + validated (seed/window/version)
- Rows loaded + columns detected
- Rolling mean + signal generation steps (including warm-up row handling)
- Metrics summary
- Job end + final status
- Any validation errors or unexpected exceptions (with stack trace via
  `logger.exception`)
