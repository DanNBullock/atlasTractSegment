"""
Created on Fri Dec 17 12:09:02 2021

Script to segment bundles based on masks.

Segmentation methods:
    - points_in_mask: only keep streamlines with a x% of points inside the mask
    - prob_coverage: compute the mean probabilistic value of each streamline 
    and keep those streamlines bellow a threshold
    - multi_mask: keep streamlines that go through all the submasks previously
    calculated. Results look a bit weird, check carefully the code. The multi-
    mask needs to be computed before using split_mask.py but could in principle
    be computed on the fly.

@author: David Romero-Bascones
"""

import numpy as np
import matplotlib.pyplot as plt

from fury.colormap import colormap_lookup_table

from dipy.io.streamline import load_trk
from dipy.io.image import load_nifti
from dipy.tracking.streamline import relist_streamlines, unlist_streamlines, transform_streamlines
from dipy.viz import actor, window

def values_from_mask(streamlines, mask, affine):
    
    affine_inv = np.linalg.inv(affine)    
    streamlines_vox = transform_streamlines(streamlines, affine_inv)
        
    vals = []
    for stream in streamlines_vox:
        coords = np.floor(stream).astype('int')        
        vals.append(mask[coords[:,0], coords[:,1], coords[:,2]])
        
    return vals

def segment_bundle(tractogram, mask, method, affine, mask_th=None, cov_th=None):

    n_stream = len(tractogram)
    
    if method == 'points_in_mask':
        mask[mask >= mask_th] = 1
        mask[mask < mask_th] = 0

        cov_all = values_from_mask(tractogram, mask, affine)
        
        points_in_mask = [np.sum(cov) for cov in cov_all]
        n_point = np.array([len(stream) for stream in tractogram])
        
        percentage_in_mask = points_in_mask/n_point
        selected_ind = np.where(percentage_in_mask >= cov_th)[0]
        
    elif method == 'prob_coverage':
        cov_all = values_from_mask(tractogram, mask, affine)
        
        cov_stream = np.array([np.mean(cov) for cov in cov_all])

        selected_ind = np.where(cov_stream >= cov_th)[0]
        
    elif method == 'multi_mask':
        mask_id_list = np.unique(mask)
        mask_id_list = mask_id_list[mask_id_list != 0]
        
        n_mask = len(mask_id_list)
        in_mask = np.zeros((n_mask, n_stream))
        
        for i, mask_id in enumerate(mask_id_list):
            mask_i = np.zeros(mask.shape)
            mask_i[mask == mask_id] = 1
            
            cov_all = values_from_mask(tractogram, mask_i, affine)
            points_in_mask = np.array([np.sum(cov) for cov in cov_all])
            
            in_mask[i,:] = points_in_mask > 1
            
        selected_ind = np.where(np.sum(in_mask,0) == n_mask)[0]

    if len(selected_ind) == 0:
        print("No streamline meets the criteria");
        return 
    bundle = [tractogram[ind] for ind in selected_ind]        
    return bundle
    
            
atlas_file = 'data/Recobundles_atlas/recobundles_atlas_prob.nii.gz'
atlas_multi_mask = 'data/Recobundles_atlas/kmeans/mask_tract_AF_L_n_cluster_4.nii.gz'
tractogram_file = 'data/HCP-842/whole_brain_MNI.trk'

" Load probabilistic atlas"
[img, affine] = load_nifti(atlas_file)
mask = img[:,:,:,41]

" Load multi-mask atlas"
[multi_mask, _] = load_nifti(atlas_multi_mask)

affine_inv = np.linalg.inv(affine)

" Load tractogram "
bundle_obj = load_trk(tractogram_file, 'same', bbox_valid_check=False)
[points, offsets] =  unlist_streamlines(bundle_obj.streamlines)
tractogram = relist_streamlines(points, offsets)

" Segment bundle"
bundle = segment_bundle(tractogram, mask, 'points_in_mask', affine, mask_th=0.2, cov_th=0.5)

# bundle = segment_bundle(tractogram, mask, 'prob_coverage', affine, cov_th=0.4)
# bundle = segment_bundle(tractogram, multi_mask, 'multi_mask', affine, cov_th=0)


scene = window.Scene()
scene.clear()
scene.SetBackground(1, 1, 1)
aux = transform_streamlines(bundle, affine_inv)
stream_actor = actor.line(aux, linewidth=0.3, opacity=0.2)
scene.add(stream_actor)
window.show(scene)
