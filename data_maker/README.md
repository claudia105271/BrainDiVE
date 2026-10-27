# Data Preparation

This stage preprocesses Natural Scenes Dataset (NSD) data for downstream modeling.

## Overview

1. **Normalize neural responses:** Z-score voxel responses per session and average repeated presentations of the same image.
2. **Extract cortical responses:** Remove non-cortical voxels and associate responses with COCO image IDs.
3. **Prepare images:** Retrieve, crop, and resize the corresponding COCO images.
4. **Rank HCP ROIs:** Rank HCP-MMP1 cortical parcels by mean ncsnr across NSD subjects to establish a shared ROI ordering.
5. **Generate HCP masks:** Create subject-specific functional volume masks and 1D masks selecting voxels belonging to the top 45 HCP ROIs.

## Scripts and Outputs

| Script | Description | Outputs |
|---|---|---|
| `Z_score.ipynb` | Z-scores neural responses and averages repeated image presentations. | `image_data_partitions.npy`<br>`pre_processing_results/NSD_zscored/subj_{}.npy`<br>`pre_processing_results/NSD_zscored/subj_{}_order.npy` |
| `data_reorder.py` | Extracts cortical responses and associates them with COCO image IDs. | `pre_processing_results/NSD_zscored/cortex_subj_{}.npy` |
| `retrieve_NSD_images.py` | Retrieves, crops, and resizes COCO images. | `pre_processing_results/image_data.h5py` |
| `rank_by_HCP.py` | Ranks HCP-MMP1 ROIs by mean ncsnr across NSD subjects. | `pre_processing_results/best_HCP.npy` |
| `create_HCP_masks.py` | Creates subject-specific functional volume masks and 1D masks selecting voxels from the top 45 HCP ROIs. | `pre_processing_results/volume_to_functional.pkl`<br>``pre_processing_results/double_mask_HCP.pkl` |

## Changes to the Upstream Code

- **Code cleanup:** Remove deadcode, add top level paths, convert notebooks to py.
- **Added HCP mask generation:**  Created subject-specific functional volume masks and 1D masks selecting voxels from the top 45 HCP ROIs.
- **Update Z_score to decrease RAM usage:** Use `float32` instead of `float64`. Normalize and averaged responses incrementally to avoid loading the full dataset.
- **Separate image preparation:** Moved image retrieval and preprocessing into a dedicated script.
