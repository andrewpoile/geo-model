# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository is a companion simulation to a paper that extends the school choice model by including transport routes into preference lists.
The simulation uses real data found in "C:\Users\aejp1u19\geo-model\data."

More information on each script can be found in the agent_docs folder.

Coding language is python.
Package manager is uv.
Testing tool is pytest.
Linting and formatting tool is ruff.
Type checker is pyrefly.

NEVER GUESS, NEVER INVENT DATA, NEVER SUPPRESS ERRORS. Do not deviate from instructions.
If anything breaks or doesn't make sense, DO NOT IGNORE OR SUPPRESS IT; report it.

All written code should prioritise accuracy and correctness above all else.
All written code should be as lean and human-readable as possible, no bloat.
Test suite should be as lean as possible.
CLAUDE.md should stay concise.

## Commands

CI (`.github/workflows/ci.yml`) runs exactly these; the prek hooks (`prek.toml`) run ruff and pyrefly on commit.

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check
uv run pytest --cov
uv run pytest tests/model/test_matching.py::test_name
```

Entry points are the modules of `src/geo_model/scripts/`, run as `uv run python -m geo_model.scripts.<name>`, e.g. `uv run python -m geo_model.scripts.run --min-distance 4828`; `uv run python -m geo_model` (also `uv run geo-model`) runs the parameter sweep. Each takes `--la NAME [NAME ...]` (school register spelling, Southampton by default), `--merge` and `--priority {distance,bracket}`, the rule school priorities follow, distance by default; run, the scenarios, sweep and optimise also take `--anon`, which numbers every school, district and authority and draws no map, writing to `output/<region>/anonymised/`. The options several scripts share are added by the `add_*` functions of cli.py.

## Data, outputs and tests

`data/`, `output/`, `temp/` and `agent_docs/` are gitignored, so they exist only locally. Tests that need `data/` take the session fixtures in `tests/conftest.py`, which load it once and skip where it is absent, as in CI, so a green CI run does not cover them: run the suite locally. Tests mirror the package and run in pytest's importlib mode, so a test file cannot import another; builders several folders share live in `tests/shared/frames.py`, on pytest's `pythonpath`. Each region writes to `output/<LA names joined by "+", alphabetical>/` (`region_dir` in output.py), a run under `--priority bracket` to its `bracket/` folder. `temp/population_lsoa.pkl` caches the ONS population estimates (`POPULATION_CACHE` in data/sources.py); delete it if those inputs change.

## Architecture

The package is layered, each layer importing only from those above it, and no library module imports a script:

1. `data`: reads `data/` into GeoDataFrames of LSOAs (districts) and schools for the chosen LAs. `sources.py` holds every path, `areas.py`, `schools.py` and `travel.py` the readers, `regions.py` the grouping of LAs into regions, `anonymise.py` the relabelling `--anon` applies right after loading.
2. `model`: simulates one instance. `settings.py` holds the seed, the default sample count, the route and preference defaults and `Settings`/`DEFAULTS`, `DISADVANTAGE` (`--disadvantage`), who is disadvantaged and who may ride, and `PRIORITY` (`--priority`), the rule school priorities follow. `students.py` samples students per seed, `preferences.py` ranks school-route bundles for students and students for schools, `routes.py` builds the route network from a `Settings`, and `matching.py` runs Deferred Acceptance with Transportation (numba `@njit`, defined formally in agent_docs/matching.md).
3. `measures`: `segregation.py` (dissimilarity index, Gini), `travel.py` (expected travel modes), `statistics.py` (t-tests with Holm correction) and `summaries.py` (tables of one setting's rows).
4. `analysis`: `scoring.py` matches and scores a sample with and without routes, `sweep.py` holds `GRID`, `search.py` the NSGA-II problem, `workers.py` the process pools the sweep and search score in.
5. `plots`: every figure, each written through `save` in output.py.
6. `scripts`: the entry points. `run.py` is the atom: it scores one setting (`score_settings`) and writes, plots and reports it (`run`), every `Settings` field an option. The rest build on it and import no other script:
   - `scenario_1.py` to `scenario_3.py`: presets of `run.main`, fixed hand-built route networks.
   - `sweep.py`: one-at-a-time sweep of each parameter in `GRID` around `DEFAULTS`, every cell scored with `score_settings`; `__main__.py` only runs it.
   - `optimise.py`: NSGA-II search over route parameters, trading segregation off against modal shift, every candidate scored with `score_settings` and the compromise run with `run`.
   - `build_prefs.py`: writes one instance of both phases.

`cli.py` adds the options several scripts share; `output.py` holds the output folders and writers.

Every comparison between "with routes" and "without routes" uses the same student samples and seeds, so differences come from the routes alone. Keep this when changing sampling or scoring.
