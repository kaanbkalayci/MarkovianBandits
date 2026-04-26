# Saved Results Summary

These folders are saved outputs from the later sections of `notebooks/Experiment-Vehicle-Two-Tower.ipynb`.

## Interpreting metrics

- `top1`: how often the top-scored action matches the oracle-best action at a timestamp
- `top3`: how often the oracle-best action appears in the top 3 predicted actions
- `mean_rank`: average predicted rank of the oracle-best action
- `avg_regret`: average utility gap to the oracle-best action
- `avg_norm_regret`: regret normalized within each decision set

## Runs summarized from `meta.json`

### `final_embedding_best_utility_hidden_256`

- Utility family: rational / divisor-style full-universe utility
- Hidden size: 256
- Test `r2`: 0.9477
- Test `top1`: 0.5715
- Test `top3`: 0.9031
- Test `mean_rank`: 2.3543
- Test `avg_norm_regret`: 0.00956

### `final_embedding_divisor_hidden_1024`

- Utility family: rational / divisor-style full-universe utility
- Hidden size in config: 512
- Test `r2`: 0.9439
- Test `top1`: 0.6136
- Test `top3`: 0.9031
- Test `mean_rank`: 2.1009
- Test `avg_norm_regret`: 0.00819

### `final_embedding_full_universe_hidden_1024`

- Utility family: full-universe baseline run
- Hidden size: 1024
- Test `r2`: 0.9557
- Test `top1`: 0.5488
- Test `top3`: 0.8249
- Test `mean_rank`: 2.9078
- Test `avg_norm_regret`: 0.01076

### `final_embedding_pc_linear_hidden_256`

- Utility family: clipped piecewise-linear utility
- Hidden size: 256
- Test `r2`: 0.9460
- Test `top1`: 0.4138
- Test `top3`: 0.5548
- Test `mean_rank`: 13.2139
- Test `avg_norm_regret`: 0.02494

## Practical takeaway

Among the summarized runs:

- `final_embedding_divisor_hidden_1024` has the strongest saved `top1` and lowest normalized regret.
- `final_embedding_best_utility_hidden_256` is very competitive and has equally strong saved `top3`.
- the piecewise-linear utility appears materially worse for ranking quality.

That makes the divisor/rational full-universe setup the best current candidate to treat as the canonical final model family.

