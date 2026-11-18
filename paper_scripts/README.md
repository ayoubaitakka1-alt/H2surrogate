# Analysis scripts used for the manuscript (archive)

These are the scripts that produced the tables and figures of the paper, in the state they were run. They contain absolute paths
(`/home/claude/work/...`) for the data, results and figure folders; adjust `DATA`, the `results_*.pkl` paths and the figure folder before running them.
The maintained, path-independent code is in `h2surrogate/`, and `tests/test_paper_numbers.py` checks that it reproduces the main numbers.

| Script | Content |
|---|---|
| `common.py`, `loader.py` | model definitions, metrics, parsing of the H2FillS result files |
| `run_protocols.py` | random 5-fold CV, leave-one-level-out, off-grid, edge-withholding, confirmation |
| `run_extras.py`, `run_application.py` | learning curves, permutation importance, set-point optimisation (Section 4) |
| `run_gp.py`, `run_tuned.py` | Gaussian-process baseline and tuned hyperparameters (Section 3.11) |
| `run_mena2.py`, `run_econ.py`, `run_scen.py`, `gen_mena_tables.py` | MENA station application, economics and scenarios (Section 5) |
| `figs_*.py`, `fig_common.py` | figures |
