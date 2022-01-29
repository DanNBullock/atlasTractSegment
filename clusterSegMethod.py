#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:07:06 2022

@author: dan
"""
import nibabel as nib

#atlas
atlasParcellationNiftiPath='/media/dan/storage/gitDir/Pandora-WhiteMatterAtlas/Recobundles/Recobundles.nii.gz'
atlasParcellationNifti=nib.load(atlasParcellationNiftiPath)
#atlas anatomy
atlasAnatomyNiftiPath='/media/dan/storage/gitDir/Pandora-WhiteMatterAtlas/T1/T1.nii.gz'
atlasAnatomyNifti=nib.load(atlasAnatomyNiftiPath)
#load subject T1 anatomy (to warp to)
targetAnatomyPath='/media/dan/storage/data/proj-5ffc884d2ba0fba7a7e89132/sub-100206/dt-neuro-anat-t1w.id-6099aad9ddc68808d13c689b/t1.nii.gz'
targetAnatomyNifti=nib.load(targetAnatomyPath)
#load test streamlines (hopefully this will be small)
testStreamlines=nib.streamlines.load('/media/dan/storage/data/proj-5c3caea0a6747b0036dcbf9a/sub-100206/dt-neuro-track-tck.tag-ensemble.id-5c3caee7a6747b0036dcbf9d/track.tck')
streamlines=testStreamlines.streamlines


def warpVolumetricAtlasToSubject(targetAnatomyNifti,atlasAnatomyNifti,atlasParcellationNifti,threshold=0):
    """
    Performs a linear affine warp of a input atlas parcellation to a specified
    anatomy T1.  Also requires a anatomy T1 for the input atlas as well.
    Works on 4D volumetric atlases as well, and outputs a list of warped 3D
    atlases in such a case.

    Parameters
    ----------
    targetAnatomyNifti : string path or nibabel.nifti1.Nifti1Image
        The target anatomy that you would like the parcellation(s) warped to.
    atlasAnatomyNifti : string path or nibabel.nifti1.Nifti1Image
        The atlas anatomy T1 that will serve as the basis for the transform/
        warp.
    atlasParcellationNifti : string path or nibabel.nifti1.Nifti1Image
        The volumetric atlas that will be transformed/warped to the subject's
        space. 4D atlases are accepted
    threshold : float, between 0 and max density value for atlasParcellationNifti(s),optional
        The threshold to be applied to the data in the nifti. All values below 
        this value are set to zero.  The default is 0, and so effectively none
        is applied by default

    Returns
    -------
    transformedAtlasParcellationNiftiHolder : nibabel.nifti1.Nifti1Image or list thereof
        Either a single 3D atlas parcellation that has been transformed to subject
        space, or a list thereof.

    """
    import nibabel as nib
    from dipy.workflows.align import ImageRegistrationFlow
    from dipy.workflows.align import SynRegistrationFlow
    from dipy.workflows.align import ApplyTransformFlow
    import os

    #load things if you need to
    if isinstance(targetAnatomyNifti,str):
        targetAnatomyNifti=nib.load(targetAnatomyNifti)
    if isinstance(atlasAnatomyNifti,str):
        atlasAnatomyNifti=nib.load(atlasAnatomyNifti)
    if isinstance(atlasParcellationNifti,str):
        atlasParcellationNifti=nib.load(atlasParcellationNifti)
        
    #determine if the input parcellation is actually 4d
    if len(atlasParcellationNifti.shape)==4:
        print('4 dimensional atlas detected.  Spliting 4th dimension into separate nifti masks')
        holdSeparateAtlasNiftis=[ nib.Nifti1Image(atlasParcellationNifti.get_data()[:,:,:,iSlices], affine=atlasParcellationNifti.affine) for iSlices in range(atlasParcellationNifti.shape[3])]
        atlasParcellationNifti=holdSeparateAtlasNiftis
    else:
        #just throw it in a one item list so you can do a single iteration loop later
        atlasParcellationNifti=[atlasParcellationNifti]
        
    #now, because dipy doesn't handle nifti objects for some reason with it's workflows...
    #we have to re save everything
    #save the target anatomy
    nib.save(nib.Nifti1Image(targetAnatomyNifti.get_data(),affine=targetAnatomyNifti.affine,header=targetAnatomyNifti.header), 'targetAnatomy.nii.gz' )
    #save the atlas anatomy
    nib.save(nib.Nifti1Image(atlasAnatomyNifti.get_data(),affine=atlasAnatomyNifti.affine,header=atlasAnatomyNifti.header), 'atlasAnatomy.nii.gz' )
    #iteratively save the atlas slices, if necessary
    atlasNames=['atlasParcellation_'+str(iAtlasSlices)+'.nii.gz' for iAtlasSlices in range(len(atlasParcellationNifti))]
    #now loop and save
    for iAtlasSlices in range(len(atlasNames)):
        nib.save(nib.Nifti1Image(atlasParcellationNifti[iAtlasSlices].get_data(),affine=atlasParcellationNifti[iAtlasSlices].affine,header=atlasParcellationNifti[iAtlasSlices].header), atlasNames[iAtlasSlices] )
    
    #now we can begin to use the workflows
    affineReg=ImageRegistrationFlow()
    affineReg.run(static_image_files='targetAnatomy.nii.gz', moving_image_files='atlasAnatomy.nii.gz')
    
    #if you want to do a nonlinear warp
    #synReg=SynRegistrationFlow()
    #synReg.run(static_image_files=testAnatomyPath, moving_image_files='moved.nii.gz', prealign_file='affine.txt')

    applyTransform=ApplyTransformFlow()
    #establblish the holder for the niftis
    transformedAtlasParcellationNiftiHolder=[[] for iAtlases in atlasNames]
    for iAtlases in range(len(atlasNames)):
        #apply the linear transform
        applyTransform.run(static_image_files='targetAnatomy.nii.gz', moving_image_files=atlasNames[iAtlases], transform_type='affine', transform_map_file='affine.txt', )
        
        #here's where you'd apply the nonlinear warp
        #applyTransform=ApplyTransformFlow()
        #applyTransform.run(static_image_files=testAnatomyPath, moving_image_files='transformed.nii.gz', transform_type='affine', transform_map_file='displacement_field.nii.gz' )
        
        #seems we have to load it
        currentAtlasNifti=nib.load('transformed.nii.gz')
        #if the input data is a float, then you can apply the theshold operation
        if currentAtlasNifti.get_data_dtype()==float:
            #get and then thesholdd the data
            currentDataHold=currentAtlasNifti.get_data()
            #set anything less than the threshold to zero
            currentDataHold[currentDataHold<threshold]=0
            #form back into a nifti and pass to the holder
            transformedAtlasParcellationNiftiHolder[iAtlases]=nib.Nifti1Image(currentDataHold, affine=currentAtlasNifti.affine, header=currentAtlasNifti.header)
        #else, it's probably boolean or integer, and you shouldn't mess with it
        else:
            transformedAtlasParcellationNiftiHolder[iAtlases]=currentAtlasNifti
        
    #if it's just a singleton atlas, only output the atlas, and not a list
    if len(atlasNames)==1:
        transformedAtlasParcellationNiftiHolder=transformedAtlasParcellationNiftiHolder[0]

    #maybe you want to clean up here and remove all of your mess?  
    os.remove('targetAnatomy.nii.gz') 
    os.remove('atlasAnatomy.nii.gz') 
    os.remove('transformed.nii.gz')
    os.remove('affine.txt')
    [ os.remove(iAtlasNames) for iAtlasNames in atlasNames]
    
    return transformedAtlasParcellationNiftiHolder

def tractQBCentroidsAsStreamlines(streamlines, **kwargs):
    """
    (quickly?, via qbx_and_merge?) perform a quick-bundling of an input
    collection of streamlines, and then extract centroids of the resultant
    clusters as streamlines.

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        The input streamlines 
    **kwargs : keyword arguments for the qbx_and_merge
        Currently only supports [thresholds] and [nb_pts]
        See dipy.segment.bundles.qbx_and_merge for more details

    Returns
    -------
    centroidsAsStreamlines : nibabel.streamlines.array_sequence.ArraySequence
        A streamline object containing the centroids of the clusters resulting
        quickBundle-ification of the input streamlines
    clusters : dipy.segment.clustering.ClusterMapCentroid
        The clusters resulting from the quickBundle-ification of the input 
        streamlines

    """
    from dipy.segment.bundles import qbx_and_merge
    from dipy.tracking.streamline import Streamlines
    
    #fill in parameters if they are there.
    if not 'thresholds' in kwargs.keys():
        thresholds = [30,20,10,5]
    if not 'nb_points' in kwargs.keys():
        nb_pts=20
    #perform the quick, iterave bundling
    clusters=qbx_and_merge(streamlines,thresholds , nb_pts, select_randomly=None, rng=None, verbose=False)
    
    #iterate through the clusters to obtain the centroids
    #not necessary
    #clusterCentroids=[iClusters.centroids for iClusters in clusters] 
    
    #convert the centroids to streamlines
    centroidsAsStreamlines=Streamlines(clusters.centroids)
    
    return centroidsAsStreamlines, clusters

def segmentViaCentroidClusters(streamlines, maskNiftis, multipleMode='distinct',returnAs='bool', **kwargs):
    """
    Segmentation function which clusters streamlines (using quickBundles) and
    performs the desired segmentation operations on their centroids, and then
    returns the streamlines (or a boolean vector representing valid streamlines 
    in the source streamlines object) associated with the centroids which
    survive the segmentation operation.

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence 
                                     OR
                  dipy.segment.clustering.ClusterMapCentroid
        The streamlines which are to be segmented from.  Alternatively, if 
        a quiickbunles segmentation is precomputed, detects this, and skips
        recomputing these.
    maskNiftis : nib.nifti1.Nifti1Image or list thereof
        The Nifti masks which are to serve as the masks for the requested
        segmentation opperations
    multipleMode : string, either 'distinct' or 'recursive', optional
        This input determines whether the multiple masks and segmentation
        operation arguments should be thought of as being additive criteria 
        (i.e. increasingly refined/harsh criteria or . The default is 'distinct'.
    returnAs : string, either 'bool' or 'streams', optional
        This input determines which format the output will be, either as a
        boolean vector (or list thereof) or as a collection of streamlines (or 
        list thereof). The default is 'bool'.
    **kwargs : Keyword arguments for dipy.tracking.utils.near_roi
        Paramters for the specification of segmentation operations:
            include : array or list of bool
                A list or 1D array of boolean values marking inclusion or exclusion
                criteria. If a streamline is near any of the inclusion ROIs, it
                should evaluate to True, unless it is also near any of the exclusion
                ROIs.
            mode :  string or list thereof, optional
                One of {"any", "all", "either_end", "both_end"}, where return True
                if:
                "any" : any point is within tol from ROI. Default.
                "all" : all points are within tol from ROI.
                "either_end" : either of the end-points is within tol from ROI
                "both_end" : both end points are within tol from ROI.
            tol : float or list thereof
                Distance (in the units of the streamlines, usually mm). If any
                coordinate in the streamline is within this distance from the center
                of any voxel in the ROI, the filtering criterion is set to True for
                this streamline, otherwise False. Defaults to the distance between
                the center of each voxel and the corner of the voxel.
            

    Raises
    ------
    TypeError
        Raised in cases of passing ndarray instead of Nifti
    ValueError
        Raised when input string arguments not understood

    Returns
    -------
    finalOutput : boolean vector or streamlines, or (in the case of multi input), lists thereof
        Either a boolean vector (or list thereof) indicating streamlines from
        the original input streamlines which have survived the segmentation
        process, or the streamines themselves which have suvived (or lists
        thereof).

    """
    import nibabel as nib
    import dipy
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import near_roi
    import numpy as np
    import re
    import itertools
    
    #determine if the input is a single nifti or a list of niftis
    if isinstance(maskNiftis, list):
        multiSeg=True
        #check to make sure the input is correct
        if isinstance(maskNiftis[0], np.ndarray):
            raise TypeError ('Input mask format expected to be Nifti, not ndarray')
    elif isinstance(maskNiftis, nib.nifti1.Nifti1Image):
        multiSeg=False
    elif isinstance(maskNiftis, np.ndarray):
        raise TypeError ('Input mask format expected to be Nifti, not ndarray')
    
    #interpret kwargs, autopopulate in accordance with input mask number
    if not 'include' in kwargs.keys():
        if isinstance(maskNiftis, list):
            include = [True for iMasks in maskNiftis]
        else:
            include = True
    else:
        include=kwargs['include']
    if not 'mode' in kwargs.keys():
        if isinstance(maskNiftis, list):
            mode = ['any' for iMasks in maskNiftis]
        else:
            mode = 'any'
    else:
        mode=kwargs['mode']
    if not 'tol' in kwargs.keys():
        if isinstance(maskNiftis, list):
            tol = [None for iMasks in maskNiftis]
        else:
            tol = None
    else:
        tol=kwargs['tol']
    
    #determine if the input was streamlines or qbcluster
    if isinstance(streamlines,nib.streamlines.array_sequence.ArraySequence):
        print('Streamline input detected, performing qbx_and_merge')
        #extract the centroids and their clusters
        centroidsAsStreamlines, clusters=tractQBCentroidsAsStreamlines(streamlines)
    elif isinstance(streamlines,dipy.segment.clustering.ClusterMapCentroid):
        #isn't there another qb cluster class/ type?  If so, add as needed.
        print('precomputed quickbundle input detected, extracting centroids and streamlines')
        #set the streamline variable to the appropriate label (clusters)
        clusters=streamlines
        streamlines=clusters.refdata
        centroidsAsStreamlines=Streamlines(clusters.centroids)
    
    #convert everything to a list of one so that you can just have one version of this
    if not multiSeg:
        maskNiftis=[maskNiftis]
        include=[include]
        mode=[mode]
        tol=[tol]
    
    #create output list for the segmentations
    segOutputs=[[] for iMasks in range(len(maskNiftis))]
    #perform iterative segmentation
    for iMasks in range(len(maskNiftis)):
        #if this is an inclusion operation
        if include[iMasks]:
            segOutputs[iMasks]=near_roi(centroidsAsStreamlines, maskNiftis[iMasks].affine, maskNiftis[iMasks].get_data(), tol[iMasks], mode[iMasks])
        else:
            #otherwise, if it's exclusion, do so
            segOutputs[iMasks]=np.logical_not(near_roi(centroidsAsStreamlines, maskNiftis[iMasks].affine, maskNiftis[iMasks].get_data(), tol[iMasks], mode[iMasks]))

    #create an output list for the outputs themselves
    outputs=[[] for iMasks in range(len(maskNiftis))]
    #unpack the outputs for each mask applied
    for iOutputs in range(len(segOutputs)):
        currentOutput=segOutputs[iOutputs]
        #unpack the indexes of each cluster, if it met the criterion
        for iClusters in range(len(clusters)):
            listsOfIndexes=[ clusters.clusters[iClusters].indices for iClusters in np.where(currentOutput)[0] ]
        outputs[iOutputs] = list(itertools.chain(*listsOfIndexes))
    
    #setup the output in accordance with the returnAs
    r = re.compile('^bool',)
    boolFlag=bool(re.search(r, returnAs.lower()))
    
    r = re.compile('^stream')
    streamFlag=bool(re.search(r, returnAs.lower()))
    
    #generate some empty boolean vector outputs
    finalOutput=[np.zeros(len(streamlines),dtype=bool) for iOutputs in outputs]
    #iterate across them and set True in the right places
    #this will get destroyed if streams have been requested
    for iOutputs in range(len(outputs)):
        finalOutput[iOutputs][outputs[iOutputs]]=True
        
    #set the output data in accordance with the multiplemode setting    
    if multipleMode=='recursive':
        collectiveOutputs =np.all(outputs,axis=0)
        finalOutput=[collectiveOutputs]
    
    #if streamlines have been requested
    if streamFlag:
        #set up a holder
        finalOutput=[[] for iOutputs in outputs]
        for iOutputs in range(len(outputs)):
            #set streamlines in the relevant output hoders
            finalOutput[iOutputs]=streamlines[outputs[iOutputs]]
    elif not boolFlag:
        raise ValueError('Requested output type ' + returnAs + ' not understood, returning boolean vector(s)')
        
    #just return the output object, streamlines or boolean vector, if there's only one item
    if len(finalOutput)==1:
        finalOutput=finalOutput[0]
        
    return finalOutput


warpedMasks=warpVolumetricAtlasToSubject(targetAnatomyNifti,atlasAnatomyNifti,atlasParcellationNifti,threshold=.2)
mode=['all' for warpedMasks in warpedMasks]
include=[True for warpedMasks in warpedMasks]
tol=[.5 for warpedMasks in warpedMasks]
testOutput=segmentViaCentroidClusters(streamlines, warpedMasks, multipleMode='distinct',returnAs='bool',  mode=mode, include=include, tol=tol)

import wmaPyTools.visTools

for iTestOutput in len(range(testOutput)): 
    #using a homebrew wma_pytools function wmaPyTools.visTools.dipyPlotTract()
    #https://github.com/DanNBullock/wma_pyTools/blob/da8634b66d55d90b44abe122c2af1f2e130b0512/wmaPyTools/visTools.py#L1002-L1227
    wmaPyTools.visTools.dipyPlotTract(streamlines[testOutput[iTestOutput]],refAnatT1=None, tractName=atlasNames[iTestOutput])