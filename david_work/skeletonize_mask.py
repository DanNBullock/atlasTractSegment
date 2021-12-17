"""
Created on Fri Dec 17 11:19:16 2021

Script to skeletonize bundle mask

@author: David Romero-Bascones
"""

import os
import pandas as pd
from skimage.morphology import skeletonize, skeletonize_3d

from dipy.io.image import load_nifti, save_nifti

atlas_file = 'data/Recobundles_atlas/recobundles_atlas_prob.nii.gz'
atlas_info = 'data/Recobundles_atlas/tract_list.csv'
out_dir = 'data/Recobundles_atlas/skeleton'

th = 0.2  # threshold to binarize the mask (might not be necessary for skeletonization)

" Load atlas "
[atlas, aff] = load_nifti(atlas_file)
[ni, nj, nk, n_tract] = atlas.shape
tract_info = pd.read_csv(atlas_info, delimiter='\t')

tract_id = 0 # select tract
tract_name = tract_info['File Name'][tract_id]

mask = atlas[:,:,:,tract_id]
mask[mask < th] = 0
mask[mask >= th] = 1

skeleton = skeletonize(mask)

fname = f'skeleton_tract_{tract_name}.nii.gz'
fout = os.path.join(out_dir, fname)
save_nifti(fout, skeleton, aff)

# skeletonize_3d --> same result
# skeleton_3d = skeletonize_3d(mask)
# fname = f'skeleton_3d_tract_{tract_name}.nii.gz'
# fout = os.path.join(out_dir, fname)
# save_nifti(fout, skeleton_3d, aff)