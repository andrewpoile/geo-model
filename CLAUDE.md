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
uv run pytest tests/test_matching.py::test_name
```

Entry points are run as modules, e.g. `uv run python -m geo_model` (the parameter sweep in sweep.py, also `uv run geo-model`), `uv run python -m geo_model.optimise`, `uv run python -m geo_model.scenario_1`. Each takes `--la NAME [NAME ...]` (school register spelling, Southampton by default) and `--merge` (see `add_region_arguments` in utils.py).

## Data, outputs and tests

`data/`, `output/`, `temp/` and `agent_docs/` are gitignored, so they exist only locally. Tests that need `data/` take the session fixtures in `tests/conftest.py`, which load it once and skip where it is absent, as in CI, so a green CI run does not cover them: run the suite locally. Each region writes to `output/<LA names joined by "+", alphabetical>/` (`region_dir` in utils.py). `temp/population_lsoa.pkl` caches the ONS population estimates (`load_data.POPULATION_CACHE`); delete it if those inputs change.

## Architecture

The pipeline, in dependency order:

1. `load_data.py` reads `data/` into GeoDataFrames of LSOAs (districts) and schools for the chosen LAs.
2. `build_routes.py` builds the route network (which districts get routes to which schools, and the seats on each) from a `Settings` of route parameters. `DISADVANTAGE` here (`--disadvantage`) sets who is disadvantaged and who may ride.
3. `build_prefs.py` samples students per seed and builds their ranked preferences over schools and school-route bundles, and the schools' priorities.
4. `matching.py` runs Deferred Acceptance with Transportation (numba `@njit`), the mechanism defined formally in agent_docs/matching.md.
5. `dissimilarity.py` holds `Settings`/`DEFAULTS`, draws the samples, and matches and scores each one with and without routes over the same students. `utils.py` holds the shared measures (dissimilarity index, Gini, expected travel modes, t-tests with Holm correction) and region handling.

Analyses built on top of that pipeline:
- `sweep.py`: one-at-a-time sweep of each parameter in `GRID` around `DEFAULTS`, with plots and maps; `__main__.py` only runs it.
- `car_displacement.py`: the change in travel mode routes induce (National Travel Survey mode shares).
- `optimise.py`: NSGA-II (pymoo) search over route parameters, trading off segregation against modal shift.
- `scenario_1.py`, `scenario_2.py` and `scenario_3.py`: fixed hand-built route networks, scored and plotted with optimise's plots. Scenarios 2 and 3 reuse scenario 1's runner.

Every comparison between "with routes" and "without routes" uses the same student samples and seeds, so differences come from the routes alone. Keep this when changing sampling or scoring.
