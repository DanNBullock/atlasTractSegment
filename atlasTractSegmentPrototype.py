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

#load an example atlas 
testAtlas=nib.load('/media/dan/storage/gitDir/Pandora-WhiteMatterAtlas/Recobundles/Recobundles.nii.gz')
#load subject T1 anatomy (to warp to)
testAnatomy=nib.load('/media/dan/storage/data/proj-5ffc884d2ba0fba7a7e89132/sub-100206/dt-neuro-anat-t1w.id-6099aad9ddc68808d13c689b/t1.nii.gz')
#load test streamlines (hopefully this will be small)
testStreamlines=nib.streamlines.load('/media/dan/storage/data/proj-5c3caea0a6747b0036dcbf9a/sub-100206/dt-neuro-track-tck.tag-ensemble.id-5c3caee7a6747b0036dcbf9d/track.tck')

#lets arbitrarily select the first atlas mask in the 4d atlas
firstMask=testAtlas.get_fdata()
#turn that into a nifti
firstMaskNifti=nib.Nifti1Image(firstMask, affine=testAtlas.affine, header=testAtlas.header)

#ok, but here I need to warp it to subject space


#use tractProbabilityMap2SegCriteria to get a dictionary of segmentation criteria
segCriteriaDict=wmaPyTools.segmentationTools.tractProbabilityMap2SegCriteria(warpedFirstMaskNifti)
#create a vector with the string "any" to serve as the segmentation criteria specification
anyCritVec=['any' for iCriteria in range(len(segCriteriaDict['any']['include'])+len(len(segCriteriaDict['any']['exclude'])))]
#do the same for the inclusion and exclusion specification
includeVec=[True for iCriteria in range(len(segCriteriaDict['any']['include']))]
excludeVec=[False for iCriteria in range(len(segCriteriaDict['any']['exclude']))]

#apply these criteria
comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(testStreamlines.streamlines, [segCriteriaDict['any']['include'] + segCriteriaDict['any']['exclude']], [includeVec + excludeVec], [anyCritVec])

