import pickle
import os
import h5py
import numpy as np
import nibabel as nib

nsd_root = '/mnt/lexar/NSD'
roi_root = os.path.join(nsd_root, 'nsddata', 'ppdata')
data_root = '/mnt/lexar/BrainDiVE/data_maker'
output_dir = os.path.join(data_root, 'pre_processing_results')
neural_root = os.path.join(output_dir, 'NSD_zscored')

EXPECTED_EARLY = 4657
EXPECTED_HIGHER = 22961
EXPECTED_TOTAL = EXPECTED_EARLY + EXPECTED_HIGHER


def load_from_nii(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'NIfTI file not found: {path}')
    return nib.load(path).get_fdata(dtype=np.float32)


def make_food_volume(food_data, nsdgeneral):
    food_data = np.asarray(food_data).squeeze()

    if food_data.shape == nsdgeneral.shape:
        if not np.all(np.isfinite(food_data)):
            raise ValueError('Food mask contains NaN or infinite values')
        return food_data >= 0.5

    vector_mask = nsdgeneral > -0.5
    vector_size = int(np.count_nonzero(vector_mask))

    if (
        food_data.size == vector_size
        and np.all(np.isin(food_data, [0, 1]))
    ):
        food_vector = food_data.reshape(-1).astype(bool)

    elif np.issubdtype(food_data.dtype, np.integer):
        indices = food_data.reshape(-1).astype(np.int64)

        if indices.size and (
            indices.min() < 0 or indices.max() >= vector_size
        ):
            raise ValueError(
                f'Food-mask indices must be between 0 and '
                f'{vector_size - 1}'
            )

        food_vector = np.zeros(vector_size, dtype=bool)
        food_vector[indices] = True

    else:
        raise ValueError(
            f'Cannot map food mask shape {food_data.shape} '
            f'to nsdgeneral shape {nsdgeneral.shape}, '
            f'or vector length {vector_size}'
        )

    food_volume = np.zeros(nsdgeneral.shape, dtype=bool)
    food_volume[vector_mask] = food_vector
    return food_volume


best_hcp_path = os.path.join(output_dir, 'best_HCP.npy')
food_path = os.path.join(output_dir, 'food_s1.npy')

if not os.path.isfile(best_hcp_path):
    raise FileNotFoundError(f'HCP ranking not found: {best_hcp_path}')

if not os.path.isfile(food_path):
    raise FileNotFoundError(f'Food mask not found: {food_path}')

best_hcp = np.load(best_hcp_path).astype(np.int32)
top_hcp_rois = best_hcp[:45]

if len(top_hcp_rois) != 45:
    raise ValueError(
        f'Expected 45 HCP labels, found {len(top_hcp_rois)}'
    )

for subject in [1]:
    print(f'\nProcessing subject {subject}')

    roi_dir = os.path.join(roi_root, f'subj0{subject}', 'func1pt8mm', 'roi')
    neural_path = os.path.join(neural_root, f'cortex_subj_{subject}.npy')
    roi_masks = [
        (
            'prf-visualrois', load_from_nii(os.path.join(roi_dir, 'prf-visualrois.nii.gz'))>= 0.5,
        ),
        (
            'floc-bodies', load_from_nii(os.path.join(roi_dir, 'floc-bodies.nii.gz'))>= 0.5,
        ),
        (
            'floc-faces', load_from_nii(os.path.join(roi_dir, 'floc-faces.nii.gz'))>= 0.5,
        ),
        (
            'floc-places', load_from_nii(os.path.join(roi_dir, 'floc-places.nii.gz'))>= 0.5,
        ),
        (
            'floc-words', load_from_nii(os.path.join(roi_dir, 'floc-words.nii.gz'))>= 0.5,
        ),
    ]

    hcp = load_from_nii(os.path.join(roi_dir, 'HCP_MMP1.nii.gz')).astype(np.int32)
    hcp_mask = np.isin(hcp, top_hcp_rois)
    roi_masks.append(('HCP top 45', hcp_mask))

    nsdgeneral = load_from_nii(os.path.join(roi_dir, 'nsdgeneral.nii.gz'))

    food_data = np.load(food_path, allow_pickle=False)
    food_volume = make_food_volume(food_data, nsdgeneral)
    roi_masks.append(('food', food_volume))

    with h5py.File(neural_path, 'r') as f:
        if 'mask' not in f:
            raise KeyError(f'No mask dataset in {neural_path}')
        neural_vector_mask = f['mask'][:].astype(bool)

    if neural_vector_mask.shape != hcp.shape:
        raise ValueError(
            f'Neural mask shape {neural_vector_mask.shape} '
            f'does not match HCP shape {hcp.shape}'
        )

    for name, mask in roi_masks:
        if mask.shape != neural_vector_mask.shape:
            raise ValueError(
                f'{name} shape {mask.shape} does not match '
                f'neural mask shape {neural_vector_mask.shape}'
            )

    print(f'Neural-vector mask voxels: {np.count_nonzero(neural_vector_mask)}')
    print(f'Food volume voxels: {np.count_nonzero(food_volume)}')
    print('\nIndividual and cumulative contributions')

    combined_mask = np.zeros(neural_vector_mask.shape, dtype=bool)
    for name, mask in roi_masks:
        mask_in_neural = mask & neural_vector_mask
        added_mask = mask_in_neural & ~combined_mask
        added_count = int(np.count_nonzero(added_mask))

        combined_mask |= mask_in_neural

        print(
            f'{name}: '
            f'individual={np.count_nonzero(mask_in_neural)}, '
            f'new_voxels={added_count}, '
            f'cumulative_total={np.count_nonzero(combined_mask)}'
        )

    actual_total = int(np.count_nonzero(combined_mask))
    print('\nCheckpoint comparison')
    print(f'Expected total:  {EXPECTED_TOTAL}')
    print(f'Actual total:    {actual_total}')

volume_functional_mask = combined_mask.astype(bool)
double_mask = combined_mask[neural_vector_mask.astype(bool)].astype(bool)

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "volume_functional_mask_s1.pkl"), "wb") as f:
    pickle.dump(volume_functional_mask[np.newaxis, ...], f,
                protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_dir, "double_mask_s1.pkl"), "wb") as f:
    pickle.dump(double_mask[np.newaxis, ...], f,
                protocol=pickle.HIGHEST_PROTOCOL)

print("Volume mask:", volume_functional_mask.shape,
      int(volume_functional_mask.sum()))
print("Double mask:", double_mask.shape,
      int(double_mask.sum()))
