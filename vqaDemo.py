# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

dataDir='D:\Thesis\VizWiz\data'
split = 'train'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Images/' %dataDir

# initialize VQA api for QA annotations
vqa=VQA(annFile)

# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns();   
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgFilename = randomAnn['image']
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given images
#imgs = vqa.getImgs()
#anns = vqa.getAnns(imgs=imgs)
#randomAnn = random.choice(anns)
#vqa.showQA([randomAnn])  
#imgFilename = randomAnn['image']
#if os.path.isfile(imgDir + imgFilename):
#	I = io.imread(imgDir + imgFilename)
#	plt.imshow(I)
#	plt.axis('off')
#	plt.show()