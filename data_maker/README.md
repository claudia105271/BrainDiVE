# Data Preparation

This stage preprocesses Natural Scenes Dataset (NSD) data for downstream modeling.

## Overview

1. **Normalize neural responses:** Z-score voxel responses per session and average repeated presentations of the same image.
2. **Extract cortical responses:** Remove non-cortical voxels and associate responses with COCO image IDs.
3. **Prepare images:** Retrieve, crop, and resize the corresponding COCO images.

## Scripts and Outputs

### `Z_score.ipynb`

Normalizes neural responses and averages repeated presentations.

**Outputs:**
- `image_data_partitions.npy` — Training, holdout, and shared validation image partitions.
- `pre_processing_results/NSD_zscored/subj_{}.npy` — Z-scored neural responses.
- `pre_processing_results/NSD_zscored/subj_{}_order.npy` — Image ordering information.

### `data_reorder.py`

Extracts cortical voxels and associates neural responses with COCO image IDs.

**Output:**
- `pre_processing_results/NSD_zscored/cortex_subj_{}.npy` — Cortical neural responses for each subject.

### `retrieve_NSD_images.py`

Retrieves and preprocesses COCO images corresponding to NSD stimuli.

**Output:**
- `pre_processing_results/image_data.h5py` — Processed images indexed by COCO image ID.

## Changes to the Upstream Code

The code was adapted to run on a machine with 32 GB of RAM:

- **Reduced memory usage:** Used `float32` instead of `float64`.
- **Processed data session by session:** Normalized and averaged responses incrementally to avoid loading the full dataset, which would require approximately 76 GB of memory.
- **Separated image preparation:** Moved image retrieval and preprocessing into a dedicated script.
