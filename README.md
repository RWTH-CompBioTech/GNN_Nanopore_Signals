# Ion Channel Event Detection with Graph Neural Networks

Automated detection and classification of single-channel ion channel events from high-frequency electrophysiology recordings.

---

## Data Requirements

The pipeline expects **Axon Binary Format (`.abf`) files** recorded at **1 MHz** sampling rate. These are typically single-channel patch-clamp or nanopore recordings in voltage-clamp configuration, where events appear as transient current deflections from a stable baseline.

- Format: `.abf` (Axon Binary Format, compatible with pyABF)
- Signal: ionic current in pA
- Recording type: single-channel, voltage-clamp
- Multi-sweep recordings are supported (all sweeps concatenated automatically)

A labeled event table in **Parquet format** is required for training (columns: event waveform arrays, duration, start/end indices).

---

## Installation

```bash
pip install pyabf torch torch_geometric pandas numpy scipy scikit-learn matplotlib
```

GPU (CUDA) is strongly recommended for training and inference at scale.

---

## Usage

### 1. Training

Open and run `training_GNN.ipynb`.

Set the paths at the top of the notebook:

```python
ABF_FILE   = "path/to/your_recording.abf"
EVENTS_FILE = "path/to/your_events.parquet"
```

The notebook will:
- Load and preprocess labeled events and baseline samples
- Train a fast CNN screener
- Train a Graph Autoencoder (GAE) for event validation and clustering
- Save model weights to `fast_screener.pth` and `gae_model.pth`

### 2. Inference

Open and run `inference_GNN.ipynb`.

Set the paths at the top of the notebook:

```python
ABF_FILE          = "path/to/your_recording.abf"
SCREENER_WEIGHTS  = "fast_screener.pth"
GAE_WEIGHTS       = "gae_model.pth"
```

The notebook will scan the full recording, validate and cluster detected events, and produce a results table with per-event metadata (timing, duration, cluster label, reconstruction quality).

---

## Output

Each detected event is returned with:

| Field | Description |
|---|---|
| `start_sample` / `end_sample` | Sample indices in the raw signal |
| `start_time_ms` | Event onset in milliseconds |
| `duration_ms` | Event duration in milliseconds |
| `cluster` | Cluster label (0-indexed) |
| `reconstruction_mse` | Waveform reconstruction quality |
| `reconstruction_corr` | Correlation with reconstructed waveform |
| `screener_confidence` | CNN screener probability score |

---


