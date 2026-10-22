# Controlled experiment protocol

Start with the exact same preprocessed artifacts, held-out project and seed for
ERM and CC-DRO. `run_focused_ablations.py` prints bounded sequential commands for
lambda_inv = 0, 0.01, 0.1, 1 and max_length = 256, 512, including class-balanced
and ordinary shuffled environment sampling. ERM always pools all source rows.

Inspect validation diagnostics before enabling execution. There is no automatic
selection of an ablation winner and no test-based hyperparameter search. Full
sweeps are deliberately not launched by the audit. Different effective configs
remain separate aggregation cohorts. The local 4 GiB GPU has not been benchmarked
for these commands; do not infer feasibility from CPU fixture tests.
