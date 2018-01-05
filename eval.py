
from numpy import *
import numpy as np
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color
from skimage import morphology
import timeit
import pdb

np.set_printoptions(threshold='nan')

def readGrayImage(path):
	i1 = img_as_float(io.imread(path));
	i1  = color.rgb2gray(i1);
	i1 = (i1>0.9).astype(float);
	return i1;
	
def readColorImage(path):
	i1 = img_as_float(io.imread(path));
	return i1;

def showGrayImage(i1):
	plt.figure();
	plt.imshow(i1,cmap='gray');
	

def showColorImage(i1):
	plt.figure();
	plt.imshow(i1);
	
def getBbox(image):
	cols = np.any(image, axis=0)
	rows = np.any(image, axis=1)
	cmin, cmax = np.where(cols)[0][[0, -1]]
	rmin, rmax = np.where(rows)[0][[0, -1]]
	return [rmin, cmin, rmax, cmax];
	
def eval_edge_recall(i1,i2,thresh):
	TP=1; FN=1;
	
	i1d = morphology.binary_dilation(i1.astype(bool), np.ones((thresh,thresh)));
	i2d = morphology.binary_dilation(i2.astype(bool), np.ones((thresh,thresh)));
	
	i2Correct = ((i1d.astype(bool)) & (i2.astype(bool))).astype(float)
	TP = sum(i2Correct);
	i1incorrect = i1-(i1.astype(bool) & i2d).astype(float);
	FN = sum(i1incorrect);
	
	#plt.show();
	return (TP/(TP+FN));

def eval_undersegmetation_error(TS,MS):	
	#TS = true superpixels, MS = my superpixels
	
	TSunique = np.unique(TS);
	TSSuperpixelCount = TSunique.shape[0];
	
	MSPixelCount = np.bincount(MS.ravel());
	
	usErrorSum = 0;
	
	for i in xrange(0,TSSuperpixelCount):
		binarySpx = (TS==TSunique[i]);
		bb = getBbox(binarySpx.astype(int));
		
		TSroi = binarySpx[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ];
		MSroi = MS[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ]
		
		myspx = np.unique(MSroi);
		
		for j in xrange(0,myspx.shape[0]):
			spx = (MSroi == myspx[j]);
			intersectArea = sum( ( spx & TSroi ).astype(float) );
			outsideArea = MSPixelCount[ myspx[j] ] - intersectArea;
			
			#print TSunique[i], ' & ', myspx[j], " : ", intersectArea,',\t' , outsideArea;
			usErrorSum+= min(intersectArea, outsideArea)
			
	
	usError = usErrorSum/(TS.shape[0]*TS.shape[1]);
	#print (TS.shape[0]*TS.shape[1])
	return usError;
	
def eval_unexplained_variation(img, MS):
	
	MS3d = np.repeat(MS[:, :, np.newaxis], 3, axis=2)
	
	MSunique = np.unique(MS);
	MSSuperpixelCount = MSunique.shape[0];
	
	MSPixelCount = np.bincount(MS.ravel());
	
	# Image mean
	iMean = [ mean(img[:,:,0]) , mean(img[:,:,1]), mean(img[:,:,2]) ];
	
	# print 'Image mean color:' ,iMean
	
	# Superpixel means
	spMean = [None] * (max(MSunique)+1);#(MSSuperpixelCount+1);
	
	for i in xrange(0,MSSuperpixelCount):
		spId = MSunique[i];
		spImg = np.multiply(img, (MS3d==spId).astype(float));
		
		#showColorImage(spImg);
		#plt.show();
		
		C = MSPixelCount[ spId ];

		spMean[ spId ] = [ sum(spImg[:,:,0])/C , sum(spImg[:,:,1])/C, sum(spImg[:,:,2])/C ];
		
	
	meanSpImg = zeros(img.shape);
	
	for i in xrange(0,MSSuperpixelCount):
		spId = MSunique[i];
		spImg = zeros(img.shape);
		spImg[:,:] = spMean[ spId ];
		#print spMean[ spId ];
		#showColorImage(spImg);
		#plt.show();
		spImg = np.multiply(spImg,(MS3d==spId).astype(float));
		
		
		meanSpImg +=spImg;
		#showColorImage(meanSpImg);
		#plt.show();
	
	#print 'Showing mean Image'
	#showColorImage(meanSpImg);
	#plt.show();
	
	meanImg = zeros(img.shape);
	meanImg[:,:] = iMean;
	
	Numer = 0.0000000001 + mean( pow( (meanSpImg - meanImg),2));
	Denom = 0.0000000001 + mean( pow( (img - meanImg),2));
	# print Numer, Denom
	UnexplainedVariation = 1- (Numer/Denom);
	
	return UnexplainedVariation;

def eval_chamfer(truth, predict):
	truthPoints = []
	predictPoints = []
	netScore = 0
	row, col = truth.shape

	for i in range(row):
		for j in range(col):
			if truth[i][j]==1:
				truthPoints.append([i, j])
			if predict[i][j]==1:
				predictPoints.append([i, j])

	for i in predictPoints:
		chamfer = 999999999
		for j in truthPoints:
			temp = abs(i[0]-j[0]) + abs(i[1]-j[1])
			chamfer = min(chamfer, temp)
		netScore+= chamfer

	return netScore