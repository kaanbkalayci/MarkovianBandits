# real-data-tests

This subproject contains the real-data pipeline for the vehicle/sensor bandit experiments.

The repo currently has three layers:

1. Data loading and synchronization from raw GPS + FLAC sensor recordings.
2. Bandit dataset construction from synchronized vehicle/sensor traces.
3. Notebook-driven two-tower training runs and saved result artifacts.

## Current repo map

```text
real-data-tests/
├── notebooks/
│   ├── Experiment-Vehicle-Two-Tower.ipynb
│   ├── Experiment.ipynb
│   └── Vehicle-Bandit_instance.ipynb
├── results/
│   └── final_embedding_*/
├── scripts/
│   ├── dataloader.py
│   ├── bandit_dataset.py
│   ├── bandit_dataloader.py
│   ├── bandit_dataloader_simple.py
│   ├── bandit_dataloader_utility.py
│   └── bandit_dataloader_utility_pc_linear.py
└── requirements.txt
```

## What each script does

### `scripts/dataloader.py`

Canonical raw-data loader. It:

- loads vehicle GPS
- loads FLAC audio per node
- aggregates chunked audio power
- synchronizes GPS and sensor data
- computes distance-to-node features
- provides plotting helpers for continuity, map layout, and RSSI/distance sanity checks

Main entrypoint:

- `load_data(...)`

### `scripts/bandit_dataset.py`

Canonical bandit dataset entrypoint. Use this first going forward.

Main entrypoints:

- `build_bandit_dataset_variant(...)`
- `build_triggered_bandit_dataset(...)`
- `build_full_universe_gaussian_dataset(...)`
- `build_full_universe_rational_dataset(...)`
- `build_full_universe_piecewise_linear_dataset(...)`

### Legacy builder modules in `scripts/`

These are still kept for notebook compatibility:

- `bandit_dataloader.py`
  Triggered-action builder. Only admissible subsets of triggered nodes are generated at each time.
- `bandit_dataloader_simple.py`
  Full action universe with Gaussian-style utility.
- `bandit_dataloader_utility.py`
  Full action universe with rational utility.
- `bandit_dataloader_utility_pc_linear.py`
  Full action universe with clipped piecewise-linear utility.

## Notebook status

### `notebooks/Experiment-Vehicle-Two-Tower.ipynb`

This is the main notebook for the vehicle bandit work. It contains:

- the data loading pass
- the first triggered-action bandit formulation
- later full 129-arm formulations
- multiple two-tower experiments
- final artifact export into `results/final_embedding_*`

Important note:

- this notebook contains several successive experiment blocks, including repeated training code with different utility definitions and hidden sizes
- the end of the notebook appears to be the most relevant "final" workflow

### `notebooks/Experiment.ipynb`

Separate experiment notebook for a different embedding/history pipeline. It does not look like the main vehicle two-tower production path.

### `notebooks/Vehicle-Bandit_instance.ipynb`

This file currently appears malformed or empty rather than a valid notebook JSON document. Treat it as non-canonical until repaired.

## Recommended canonical workflow

For the vehicle bandit setup, the cleanest current path is:

1. Use `scripts/dataloader.py` to produce `gdf_cleaned`, `normalized_cleaned`, `valid_indices`, and `gdf_nodes`.
2. Use `scripts/bandit_dataset.py` to build a bandit dataset variant.
3. Use the final section of `notebooks/Experiment-Vehicle-Two-Tower.ipynb` as the reference training workflow.
4. Save trained artifacts under a dedicated `results/<run_name>/` directory.

## Utility variants used so far

There are two main modeling decisions in the notebook:

1. Action space:
   - triggered subsets only
   - full 129-action universe over node subsets of size 1 to 3
2. Oracle utility:
   - Gaussian distance decay
   - rational distance decay
   - clipped piecewise-linear decay

The full-universe formulation is the one associated with the final saved embedding runs in `results/`.

## Saved result folders

Each result directory typically stores:

- `two_tower_model.pt`
- `ridge_model.pkl`
- `standardizers.pkl`
- `learned_embeddings.npy`
- `examples_with_embed.pkl`
- `meta.json`

See [results/README.md](./results/README.md) for a quick summary of the saved runs.

## Suggested next cleanup

The repo is now documented around one canonical loader and one canonical bandit entrypoint. The next high-value cleanup would be:

1. extract the final two-tower training code from `Experiment-Vehicle-Two-Tower.ipynb` into a script or module
2. reduce repeated notebook cells
3. move invalid or abandoned notebooks into an `archive/` folder once you are sure they are not needed

