import h5py
import numpy as np
import nibabel as nib
import os
import pickle
from scipy.io import loadmat

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata(dtype=np.float32)

nsd_root = '/mnt/lexar/NSD/';
roi_root = '{}/nsddata/ppdata/subj0{}/func1pt8mm/roi/{}';
output_dir = "pre_processing_results/NSD_zscored"
subjects = [1] # [1,2,3,4,5,6,7,8]

for s in subjects:
    selected = []

    # Get the voxels that belong to cortex. Basically anywhere the mask is 0 or greater.
    for roi_strings in ["prf-visualrois.nii.gz","floc-bodies.nii.gz", "floc-faces.nii.gz", "floc-places.nii.gz", "floc-words.nii.gz", "HCP_MMP1.nii.gz", "Kastner2015.nii.gz", "nsdgeneral.nii.gz"]:
        full_path = roi_root.format(nsd_root, s, roi_strings)
        all_roi = load_from_nii(full_path)
        selected.append(all_roi>=-0.5)
    all_good = np.logical_or.reduce(selected)
    print(np.sum(all_good))
    print(s, " SUBJECT")


    nsd_meta_file = os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    with open(nsd_meta_file, 'rb') as f:
        stim_info = pickle.load(f, encoding="latin1")
    exp_design_file = os.path.join(nsd_root, "nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx_MATRIX = exp_design['subjectim']

    subject_df = stim_info.loc[subject_idx_MATRIX[s - 1, :] - 1]
    COCO_ids_unorder = np.array(subject_df["cocoId"].tolist()).astype(np.int64)

    # Data loaded in here
    # Change the path to where you set it in Z_score.ipynb

    full_data = np.load("{}/subj_{}.npy".format(output_dir, s))
    order = np.load("{}/subj_{}_order.npy".format(output_dir, s))

    COCO_ids_ordered = COCO_ids_unorder[order]
    COCO_ids_ordered = COCO_ids_ordered.tolist()

    # Change the location to where you want to save it
    file = h5py.File("{}/cortex_subj_{}.npy".format(output_dir, s), 'w')

    # The mask is an object that goes from 3D full grid to just the cortical voxels
    file.create_dataset("mask", data=all_good)

    for index in range(len(COCO_ids_ordered)):

        # However remember that Maggie's script gives us data in a flat 1D array, so you want to flatten the mask to use it.
        dset = file.create_dataset(str(COCO_ids_ordered[index]), data=full_data[index][all_good.flatten()].astype(np.single))
    file.close()
