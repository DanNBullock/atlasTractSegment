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
from dipy.tracking.utils import target  
from dipy.tracking.streamline import Streamlines

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

applyTransform=ApplyTransformFlow()
applyTransform.run(static_image_files=testAnatomyPath, moving_image_files='firstMask.nii.gz', transform_type='affine', transform_map_file='affine.txt', )

#applyTransform=ApplyTransformFlow()
#applyTransform.run(static_image_files=testAnatomyPath, moving_image_files='transformed.nii.gz', transform_type='affine', transform_map_file='displacement_field.nii.gz' )

data, affine =load_nifti('transformed.nii.gz')
#no need to threshold at the moment
#data[data<.2]=0
firstMaskNifti=nib.Nifti1Image((data>0).astype(np.uint8), affine=affine)

firstMaskNifti
comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(testStreamlines.streamlines, [firstMaskNifti], [True], ['any'])
outStreamsGenerator=target(testStreamlines.streamlines, firstMaskNifti.affine, firstMaskNifti.get_fdata(), include=True)
outStreams=Streamlines(outStreamsGenerator)

#perform quickbundles on outstreams
#bundlesOut=qwBundles(outStreams)

#exclude all streamllines not returned by these centroids
#=nearROI(bundlesOubundlesOut.centroids,True,'all')



boundaryRois=wmaPyTools.roiTools.boundaryROIPlanesFromMask(firstMaskNifti)
comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(outStreams, [firstMaskNifti], [True], ['any'])


#threshold on cluster centroid nodes in mask


segmentedTract = StatefulTractogram(outStreams, testAnatomy, Space.RASMM)
save_tractogram( segmentedTract,'testSegmentedTract.trk',bbox_valid_check=False)

# #use tractProbabilityMap2SegCriteria to get a dictionary of segmentation criteria
# segCriteriaDict=wmaPyTools.segmentationTools.tractProbabilityMap2SegCriteria(warpedFirstMaskNifti)
# #create a vector with the string "any" to serve as the segmentation criteria specification
# anyCritVec=['any' for iCriteria in range(len(segCriteriaDict['any']['include'])+len(len(segCriteriaDict['any']['exclude'])))]
# #do the same for the inclusion and exclusion specification
# includeVec=[True for iCriteria in range(len(segCriteriaDict['any']['include']))]
# excludeVec=[False for iCriteria in range(len(segCriteriaDict['any']['exclude']))]

#apply these criteria
#comboROIBool=wmaPyTools.segmentationTools.segmentTractMultiROI(testStreamlines.streamlines, [segCriteriaDict['any']['include'] + segCriteriaDict['any']['exclude']], [includeVec + excludeVec], [anyCritVec])

def skeletonizeMask(maskNifti):
    """
    Created on Fri Dec 17 11:19:16 2021
    
    Script to skeletonize bundle mask
    
    @author: David Romero-Bascones
    
    Issues?
    https://forum.image.sc/t/3d-skeletonization-problem-of-skeletonize-2d-3d/10828
    #thresholding seems to fix it
    
    
    
    """
    
    import os
    import pandas as pd
    from skimage.morphology import skeletonize, skeletonize_3d
    import nibabel as nib
    
    from dipy.io.image import load_nifti, save_nifti
    
    
    skeleton = skeletonize(maskNifti.get_data())
    
    skeletonMaskNifti=nib.Nifti1Image(skeleton.astype(np.uint8), affine=maskNifti.affine)
    
    fname = 'skeleton_tract_test.nii.gz'
    outDir='/media/dan/storage/gitDir/atlasTractSegment2/atlasTractSegment'
    fout = os.path.join(outDir, fname)
    nib.save(skeletonMaskNifti,fout)
    
def decomposeTractSkeleton(tractSkeletonNifti):
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    
    import numpy as np
    #total neighborhood (including self = 3 layers of 9 voxels = 27 voxels)
    #self = dist 0; max items = 1; 26 remain
    #ortho adjacent = dist 1; max items = 6 (each face of center voxel); 20 remain
    #radial ortho adjacent= dist 2; max items = 12; (diagonal adjacents for each ortho adjacent; some sort of combinatorial thing here ); 8 remain
    #diagonal adjacent = dist 3; max items = 8 (the verticies of the 3x3 cube); 0 remain
    #1 + 6 + 12 + 8 = 27 -> OK
    acceptableNeighborLimit=3
    skeletonImgIndexes=np.asarray(np.where(tractSkeletonNifti.get_data())).T
    orthoAdjacent_IndexList=[]
    radialOrthoAdjacent_IndexList=[]
    diagonalAdjacent_IndexList=[]
    anyConnectivity_IndexList=[]
    for iSkeletonVoxels in skeletonImgIndexes:
        voxelDistances=np.subtract(skeletonImgIndexes,iSkeletonVoxels)
        distSums=np.sum(np.abs(voxelDistances),axis=1)
        orthoAdjacent_IndexList=orthoAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==1],axis=0))[0])]
        radialOrthoAdjacent_IndexList=radialOrthoAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==2],axis=0))[0])]
        diagonalAdjacent_IndexList=diagonalAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==3],axis=0))[0])]
        anyConnectivity_IndexList=anyConnectivity_IndexList+[list(np.where(np.all([0<distSums, distSums<=3],axis=0))[0])]
    
    minimallyAdjacentNeighborsList=[[] for iVoxels in range(len(skeletonImgIndexes))]
    for iSkeletonVoxels in range(len(skeletonImgIndexes)):
        if len(orthoAdjacent_IndexList[iSkeletonVoxels])>0:
            minimallyAdjacentNeighborsList[iSkeletonVoxels]=orthoAdjacent_IndexList[iSkeletonVoxels]
        elif len(radialOrthoAdjacent_IndexList[iSkeletonVoxels])>0:
            minimallyAdjacentNeighborsList[iSkeletonVoxels]=radialOrthoAdjacent_IndexList[iSkeletonVoxels]
        elif len(diagonalAdjacent_IndexList[iSkeletonVoxels])>0:
            minimallyAdjacentNeighborsList[iSkeletonVoxels]=diagonalAdjacent_IndexList[iSkeletonVoxels]
        else:
            minimallyAdjacentNeighborsList[iSkeletonVoxels]=[]
    
    minimallyAdjacentNeighborsCounts=[len(iVoxels) for iVoxels in minimallyAdjacentNeighborsList]

    #create a connectivity matrix using these adjacency values
    voxelAdjacencyMatrix=np.zeros([len(minimallyAdjacentNeighborsList),len(minimallyAdjacentNeighborsList)],dtype=bool)
    for iVoxels in range(len(minimallyAdjacentNeighborsList)):
        currentAdjacents=minimallyAdjacentNeighborsList[iVoxels]
        for iAdjacentItems in range(len(minimallyAdjacentNeighborsList[iVoxels])):
          
            voxelAdjacencyMatrix[iVoxels,currentAdjacents[iAdjacentItems]]=True
    convertedGraph=nx.convert_matrix.from_numpy_matrix(voxelAdjacencyMatrix)
    bet_centrality = nx.betweenness_centrality(convertedGraph, normalized = True, 
                                              endpoints = True)
    plt.hist(bet_centrality.values())
    
    #generous connectivity matrix
    voxelAdjacencyGenerousMatrix=np.zeros([len(anyConnectivity_IndexList),len(anyConnectivity_IndexList)],dtype=bool)
    for iVoxels in range(len(anyConnectivity_IndexList)):
        currentAdjacents=anyConnectivity_IndexList[iVoxels]
        for iAdjacentItems in range(len(anyConnectivity_IndexList[iVoxels])):
          
            voxelAdjacencyGenerousMatrix[iVoxels,currentAdjacents[iAdjacentItems]]=True
    
    
    #networkx.algorithms.bridges.bridges?
    
    fig=matplotlib.pyplot.figure(figsize=(20, 20))
    #betweenness_centrality
    #communicability_betweenness_centrality
    #generousCoveredGraph=nx.convert_matrix.from_numpy_matrix(voxelAdjacencyGenerousMatrix)
    #generousBet_centrality = nx.betweenness_centrality(generousCoveredGraph, normalized = True, 
    #                                          endpoints = True)
    #generousBet_centrality = nx.subgraph_centrality_exp(generousCoveredGraph)
    pos=nx.drawing.layout.spring_layout(covertedGraph)
    colorMap=matplotlib.cm.get_cmap('jet')
    nodeColors=[colorMap(iValue/np.max(list(generousBet_centrality.values()))) for iValue in list(generousBet_centrality.values())]
    nx.draw(generousCoveredGraph,node_color=nodeColors, pos=pos, with_labels=False,node_shape='.',node_size=300)
    matplotlib.pyplot.savefig('subgraph_centrality_exp.svg',dpi=300)
    
    chainOut=nx.algorithms.chains.chain_decomposition(generousCoveredGraph)
    chainOutHolder=list(chainOut)
    
    highlyConnectedVoxels=np.asarray(list(bet_centrality.values()))>.005
    numberOfComponents=nx.number_connected_components(generousCoveredGraph)
    
    
    #begin here
    anyConnectivity_IndexList=[]
    for iSkeletonVoxels in skeletonImgIndexes:
        voxelDistances=np.subtract(skeletonImgIndexes,iSkeletonVoxels)
        distSums=np.sum(np.abs(voxelDistances),axis=1)
        #orthoAdjacent_IndexList=orthoAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==1],axis=0))[0])]
        #radialOrthoAdjacent_IndexList=radialOrthoAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==2],axis=0))[0])]
        #diagonalAdjacent_IndexList=diagonalAdjacent_IndexList+[list(np.where(np.all([0<distSums, distSums==3],axis=0))[0])]
        anyConnectivity_IndexList=anyConnectivity_IndexList+[list(np.where(np.all([0<distSums, distSums<=3],axis=0))[0])]

    #generous connectivity matrix
    voxelAdjacencyGenerousMatrix=np.zeros([len(anyConnectivity_IndexList),len(anyConnectivity_IndexList)],dtype=bool)
    for iVoxels in range(len(anyConnectivity_IndexList)):
        currentAdjacents=anyConnectivity_IndexList[iVoxels]
        for iAdjacentItems in range(len(anyConnectivity_IndexList[iVoxels])):
          
            voxelAdjacencyGenerousMatrix[iVoxels,currentAdjacents[iAdjacentItems]]=True

    dontDeleteThese=[]
    runs=0
    while len(dontDeleteThese) > len(voxelAdjacencyGenerousMatrix):
    #convert to graph
        covertedGraph=nx.convert_matrix.from_numpy_matrix(voxelAdjacencyGenerousMatrix)
        subgraph_centrality_measures = nx.subgraph_centrality(covertedGraph)
        #chainOut=nx.algorithms.chains.chain_decomposition(covertedGraph)        
        #chainOutHolder=list(chainOut)
        graphBridges=nx.bridges(covertedGraph)
        graphBridgesHolder=list(graphBridges)
        dontDeleteThese=np.unique(graphBridgesHolder)
        
        roundedMeasures=np.round(list(subgraph_centrality_measures.values()),decimals=2)
        sortedUniqueMeasures=np.unique(roundedMeasures)
        threshVal=sortedUniqueMeasures[3]
        diffVec=roundedMeasures-threshVal
        tooLow=np.where(diffVec<0)[0]
        forbidRemove=dontDeleteThese+list(tooLow)
        
        candidates=np.where(roundedMeasures>=threshVal)
        
        
        
       
        
            
        
        # runs = runs+1
        # if len(chainOutHolder)>0:
        #     toDeleteNodes=[]
        #     for iChains in range(len(chainOutHolder)):
        #         currentNodes=list(np.unique(chainOutHolder[iChains]))
        #         [ currentNodes.remove(iNodesRemoved) for iNodesRemoved in toDeleteNodes if iNodesRemoved in currentNodes]
        #         [ currentNodes.remove(iNodesRemoved) for iNodesRemoved in dontDeleteThese if iNodesRemoved in currentNodes]
        #         if len(currentNodes)>=2:
        #             currentCentralityMeasures=[subgraph_centrality_measures[iNodeMeasures] for iNodeMeasures in currentNodes]
        #             toDeleteNode=currentNodes[np.where(currentCentralityMeasures==np.min(currentCentralityMeasures))[0][0]]
        #             toDeleteNodes=toDeleteNodes+[toDeleteNode]
        #     toDeleteNodes=sorted(toDeleteNodes, reverse=True)
        #     for iToRemove in range(len(toDeleteNodes)):
        #         voxelAdjacencyGenerousMatrix=np.delete(voxelAdjacencyGenerousMatrix, toDeleteNodes[iToRemove], 0)
        #         voxelAdjacencyGenerousMatrix=np.delete(voxelAdjacencyGenerousMatrix, toDeleteNodes[iToRemove], 1)
        # print('Run ' +str(runs)+ 'complete')
    
    fig=matplotlib.pyplot.figure(figsize=(20, 20))
    pos=nx.drawing.layout.spring_layout(covertedGraph)
    colorMap=matplotlib.cm.get_cmap('jet')
    nodeColors=[colorMap(iValue/np.max(list(subgraph_centrality_measures.values()))) for iValue in list(subgraph_centrality_measures.values())]
    nx.draw(covertedGraph,node_color=nodeColors, pos=pos, with_labels=False,node_shape='.',node_size=300)
    matplotlib.pyplot.savefig('subgraph_centrality_postClean.svg',dpi=300)
    
        
