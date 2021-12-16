#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:30:13 2021

@author: dan
"""

import nibabel as nib
import os
#some how set a path to wma pyTools repo directory
wmaToolsDir='/media/dan/storage/gitDir/wma_pyTools'
import os
os.chdir(wmaToolsDir)
import wmaPyTools.roiTools
import wmaPyTools.segmentationTools
import wmaPyTools.streamlineTools
import wmaPyTools.visTools
import numpy as np
from dipy.workflows.align import ImageRegistrationFlow
from dipy.workflows.align import SynRegistrationFlow
from dipy.workflows.align import ApplyTransformFlow
from dipy.io.image import save_nifti
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

#load an example atlas 
atlasPath='/media/dan/storage/gitDir/Pandora-WhiteMatterAtlas/Recobundles/Recobundles.nii.gz'
testAtlas=nib.load(atlasPath)
testAtlasT1Path='/media/dan/storage/gitDir/Pandora-WhiteMatterAtlas/T1/T1.nii.gz'
testAtlasT1=nib.load(testAtlasT1Path)

#load subject T1 anatomy (to warp to)
testAnatomyPath='/media/dan/storage/data/proj-5ffc884d2ba0fba7a7e89132/sub-100206/dt-neuro-anat-t1w.id-6099aad9ddc68808d13c689b/t1.nii.gz'
testAnatomy=nib.load(testAnatomyPath)
#load test streamlines (hopefully this will be small)
testStreamlines=nib.streamlines.load('/media/dan/storage/data/proj-5c3caea0a6747b0036dcbf9a/sub-100206/dt-neuro-track-tck.tag-ensemble.id-5c3caee7a6747b0036dcbf9d/track.tck')

#lets arbitrarily select the first atlas mask in the 4d atlas
firstMask=testAtlas.get_fdata()[:,:,:,0]
#turn that into a nifti
#firstMaskNifti=nib.Nifti1Image(firstMask, affine=testAtlas.affine, header=testAtlas.header)
save_nifti('firstMask.nii.gz',firstMask,affine=testAtlas.affine,hdr=testAtlas.header)

#ok, but here I need to warp it to subject space
affineReg=ImageRegistrationFlow()
affineReg.run(static_image_files=testAnatomyPath, moving_image_files=testAtlasT1Path)


#now perform nonlinear

#synReg=SynRegistrationFlow()
#synReg.run(static_image_files=testAnatomyPath, moving_image_files='moved.nii.gz', prealign_file='affine.txt')

#out_warped='warped_moved.nii.gz',
#out_inv_static='inc_static.nii.gz',
#out_field='displacement_field.nii.gz'
applyTransform=ApplyTransformFlow()
applyTransform.run(static_image_files=testAnatomyPath, moving_image_files='firstMask.nii.gz', transform_type='affine', transform_map_file='affine.txt', )

applyTransform=ApplyTransformFlow()
applyTransform.run(static_image_files=testAnatomyPath, moving_image_files='transformed.nii.gz', transform_type='affine', transform_map_file='displacement_field.nii.gz' )

data, affine =load_nifti('transformed.nii.gz')
data[data<.2]=0
firstMaskNifti=nib.Nifti1Image((data>0).astype(np.uint8), affine=affine)

firstMaskNifti
comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(testStreamlines.streamlines, [firstMaskNifti], [True], ['any'])
testStreamlines.streamlines

segmentedTract = StatefulTractogram(testStreamlines.streamlines[comboROIBool], testAnatomy, Space.RASMM)
save_tractogram( segmentedTract,'testSegmentedTract.trk')

# #use tractProbabilityMap2SegCriteria to get a dictionary of segmentation criteria
# segCriteriaDict=wmaPyTools.segmentationTools.tractProbabilityMap2SegCriteria(warpedFirstMaskNifti)
# #create a vector with the string "any" to serve as the segmentation criteria specification
# anyCritVec=['any' for iCriteria in range(len(segCriteriaDict['any']['include'])+len(len(segCriteriaDict['any']['exclude'])))]
# #do the same for the inclusion and exclusion specification
# includeVec=[True for iCriteria in range(len(segCriteriaDict['any']['include']))]
# excludeVec=[False for iCriteria in range(len(segCriteriaDict['any']['exclude']))]

#apply these criteria
comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(testStreamlines.streamlines, [segCriteriaDict['any']['include'] + segCriteriaDict['any']['exclude']], [includeVec + excludeVec], [anyCritVec])

