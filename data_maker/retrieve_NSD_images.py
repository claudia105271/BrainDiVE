import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import h5py

from skimage.transform import rescale, resize
from PIL import Image

nsd_root = '/mnt/lexar/NSD/';
coco_path = '/mnt/lexar/COCO';
output_dir = '/mnt/lexar/BrainDiVE/data_maker/pre_processing_results';

csv_obj = pd.read_csv(
    os.path.join(nsd_root, "nsddata", "experiments", "nsd", "nsd_stim_info_merged.csv")
)

def coco_crop(img,cropbox_in):
    if type(cropbox_in) is str:
        cropbox = eval(cropbox_in)
    else:
        cropbox = cropbox_in
    top = int(img.shape[0]*cropbox[0])
    bottom = int(img.shape[0]*(1-cropbox[1]))
    left = int(img.shape[1]*cropbox[2])
    right = int(img.shape[1]*(1-cropbox[3])
    return img[top:bottom, left:right]

def get_cocopath(coco_id_):
    if not isinstance(coco_id_, int):
        mycoco_id = str(int(coco_id_))
    else:
        mycoco_id = str(coco_id_)
    k = mycoco_id.zfill(12)+".jpg"
    f1 = os.path.join(coco_path, "train2017", k) 
    f2 = os.path.join(coco_path, "test2017", k) 
    f3 = os.path.join(coco_path, "val2017", k) 
    if os.path.isfile(f1):
        ff = f1

    if os.path.isfile(f2):
        ff = f2

    if os.path.isfile(f3):
        ff = f3
    return ff

all_coco = list(csv_obj["cocoId"])
file = h5py.File(os.path.join(output_dir, "image_data.h5py"), "w")
zz = 0
for i in all_coco:
    zz += 1
    if zz %100 == 0:
        print(zz)
    current_box = list(csv_obj[csv_obj["cocoId"]==int(i)]["cropBox"])[0]
    example_img = imageio.v2.imread(get_cocopath(i))
    cropped_image = np.array(coco_crop(example_img, current_box))
    cropped_image = Image.fromarray(cropped_image)
    output = np.array(cropped_image.resize((256, 256), resample=Image.Resampling.BICUBIC))
    file.create_dataset(str(i).zfill(12), data=output[:])
file.close()
