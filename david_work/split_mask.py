"""
Created on Thu Dec 16 20:28:46 2021

Script to split bundle masks into several clusters based on K-MEANS

@author: David Romero-Bascones
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dipy.io.image import load_nifti, save_nifti

atlas_file = 'data/Recobundles_atlas/recobundles_atlas_prob.nii.gz'
atlas_info = 'data/Recobundles_atlas/tract_list.csv'
out_dir = 'data/Recobundles_atlas/kmeans'
n_cluster_list = [2,3,4,5]
th = 0.2  # threshold

" Load atlas "
[atlas, aff] = load_nifti(atlas_file)
[ni, nj, nk, n_tract] = atlas.shape
tract_info = pd.read_csv(atlas_info, delimiter='\t')

ind = np.indices((ni,nj,nk))
I, J, K = ind[0], ind[1], ind[2]

" Loop through tracts "
for tract_id in range(n_tract):
    tract_name = tract_info['File Name'][tract_id]
    print(f'Processing {tract_name}')
    
    mask = atlas[:, :, :, tract_id]

    " Get coordinates of each voxel in mask"
    X = I[mask > th]
    Y = J[mask > th]
    Z = K[mask > th]
    
    XYZ = np.column_stack((X,Y,Z)) # stack them as features for K-means

    " KMeans for every cluster size"
    for n_cluster in n_cluster_list:
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(XYZ)
        labels = kmeans.labels_

        " Create new mask with kmeans labels"
        new_mask= np.zeros(mask.shape)
        new_mask = np.reshape(new_mask,-1)
        new_mask[mask.reshape(-1) >th] = labels + 1
        new_mask = np.reshape(new_mask,(ni,nj,nk))
    
        fname = f'mask_tract_{tract_name}_n_cluster_{n_cluster}.nii.gz'
        fout = os.path.join(out_dir, fname)
        save_nifti(fout, new_mask, aff)

        print(f'Finished {tract_name}: Kmeans with {n_cluster} clusters')
