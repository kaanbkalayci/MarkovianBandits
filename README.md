# real-data-tests

This subproject is the real-data bandit pipeline for the vehicle/sensor experiments.

At this point the repo has four layers:

1. raw data loading and synchronization
2. bandit dataset construction
3. learned-embedding run loading and bandit instance construction
4. offline notebook experiments and saved result artifacts

## Repo map

```text
real-data-tests/
|- notebooks/
|  |- Experiment-Vehicle-Two-Tower.ipynb
|  |- Vehicle-Bandit_instance.ipynb
|  `- Experiment.ipynb
|- results/
|  |- README.md
|  `- final_embedding_*/
|- scripts/
|  |- __init__.py
|  |- dataloader.py
|  |- bandit_dataset.py
|  |- bandit_instance.py
|  |- bandit_eval.py
|  |- bandit_dataloader.py
|  |- bandit_dataloader_simple.py
|  |- bandit_dataloader_utility.py
|  `- bandit_dataloader_utility_pc_linear.py
`- requirements.txt
```

## Canonical code paths

### `scripts/dataloader.py`

Raw-data entrypoint for the vehicle pipeline.

Main job:

- read vehicle GPS
- load FLAC recordings for each node
- aggregate chunked audio power
- synchronize sensor and GPS streams
- compute distance-to-node features
- provide plotting helpers for sanity checks

Main function:

- `load_data(...)`

### `scripts/bandit_dataset.py`

Canonical bandit dataset entrypoint. This is the organized wrapper over the older dataset-builder files.

Main functions:

- `build_bandit_dataset_variant(...)`
- `build_triggered_bandit_dataset(...)`
- `build_full_universe_gaussian_dataset(...)`
- `build_full_universe_rational_dataset(...)`
- `build_full_universe_piecewise_linear_dataset(...)`

Supported variants:

- `triggered`
- `gaussian`
- `rational`
- `piecewise_linear` or `pc_linear`

### `scripts/bandit_instance.py`

Builds the linear bandit instance from a saved embedding run.

Main jobs:

- load `examples_with_embed.pkl`, `ridge_model.pkl`, and `meta.json`
- recover `theta_star` from the ridge head
- construct per-time action feature matrices
- expose oracle rewards, linearized rewards, and misspecification

Main functions:

- `load_embedding_run(...)`
- `build_bandit_instance(...)`
- `build_bandit_instance_from_run(...)`
- `summarize_instance(...)`

### `scripts/bandit_eval.py`

Offline evaluation helpers for a constructed bandit instance.

Main functions:

- `oracle_action_indices(...)`
- `random_action_indices(...)`
- `greedy_action_indices_from_reward_source(...)`
- `evaluate_action_indices(...)`
- `evaluate_score_lists(...)`

## Legacy dataset-builder modules

These are still useful and are kept for direct notebook compatibility:

- `scripts/bandit_dataloader.py`
  Triggered-action builder. Only subsets from the triggered node set are available at a given time.
- `scripts/bandit_dataloader_simple.py`
  Full action universe with Gaussian-style utility.
- `scripts/bandit_dataloader_utility.py`
  Full action universe with rational or divisor-style utility.
- `scripts/bandit_dataloader_utility_pc_linear.py`
  Full action universe with clipped piecewise-linear utility.

## Notebook roles

### `notebooks/Experiment-Vehicle-Two-Tower.ipynb`

Main training notebook for the vehicle bandit workflow. It contains:

- raw data loading
- the early triggered-action setup
- later full 129-arm formulations
- multiple two-tower runs
- export of trained artifacts into `results/final_embedding_*`

This is still the source of truth for how the two-tower embeddings were trained.

### `notebooks/Vehicle-Bandit_instance.ipynb`

Focused notebook for loading a saved embedding run and turning it into a bandit instance for downstream experiments.

This is the notebook that corresponds most directly to:

- `scripts/bandit_instance.py`
- `scripts/bandit_eval.py`

### `notebooks/Experiment.ipynb`

Separate experiment notebook for an older or adjacent embedding/history pipeline. It is not the main current vehicle-bandit path.

## Recommended workflow

For the current vehicle bandit setup, the cleanest path is:

1. Use `scripts/dataloader.py` to build the synchronized vehicle/sensor dataframe.
2. Use `scripts/bandit_dataset.py` to build the dataset variant you want.
3. Use `notebooks/Experiment-Vehicle-Two-Tower.ipynb` to train and save a learned embedding run.
4. Use `scripts/bandit_instance.py` to convert that saved run into a linear bandit instance.
5. Use `scripts/bandit_eval.py` to evaluate policies or offline score vectors on that instance.

## Outputs in `results/`

A saved run directory typically contains:

- `two_tower_model.pt`
- `ridge_model.pkl`
- `standardizers.pkl`
- `learned_embeddings.npy`
- `examples_with_embed.pkl`
- `meta.json`

See [results/README.md](./results/README.md) for a quick summary of the saved runs.

## Current naming note

The piecewise-linear dataset builder on disk is:

- `scripts/bandit_dataloader_utility_pc_linear.py`

If you have older notes or tabs referring to `bandit_dataloader_pc_linear.py`, that appears to be the intended name, but it is not the current file path in the repo.

## Next cleanup worth doing

The highest-value next step would be extracting the final two-tower training block from `Experiment-Vehicle-Two-Tower.ipynb` into a normal script so the full pipeline becomes:

- data loader
- dataset builder
- two-tower trainer
- bandit instance builder
- evaluator
