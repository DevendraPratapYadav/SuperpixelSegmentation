
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.segmentation import *
from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

from sknn.mlp import Classifier, Layer
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import sys
import os
import subprocess
import numpy as np
import disjointset
import matplotlib.pyplot as plt
import scipy.stats.mstats as stat
import pdb
 
def showColorImage(i1):
	plt.figure();
	plt.imshow(i1);
	
serial = (sys.argv[1]);#'0010830'


image = img_as_float(io.imread('./release/images/'+serial+'.jpg'))
region = np.loadtxt('./release/labels/'+serial+'.regions.txt')
row, col, dim = np.shape(image)
region = abs(region);
#showColorImage(image);
#showColorImage(region);
#plt.show();
 
# superpixels = slic(image, n_segments=600, max_iter=10, sigma=4, slic_zero=True)
# superpixels = felzenszwalb(image, scale=3.0, sigma=0.95, min_size=5)
superpixels = quickshift(image, ratio=1, kernel_size=5, max_dist=10, convert2lab=True, random_seed=42)
mergedSuperpixels = np.zeros([row, col])
numSegments = np.amax(superpixels)

red = np.zeros([numSegments+1, 256], dtype=float)
blue = np.zeros([numSegments+1, 256], dtype=float)
green = np.zeros([numSegments+1, 256], dtype=float)

for i in range(row):
	for j in range(col):
		temp = int(image[i][j][0] * 255);
		red[superpixels[i][j]][temp]+= 2

		temp = int(image[i][j][1] * 255);
		green[superpixels[i][j]][temp]+= 2

		temp = int(image[i][j][2] * 255);
		blue[superpixels[i][j]][temp]+= 2

"""
weights = [0.2, 0.6, 0.2]
weights = np.array(weights)
for i in range(numSegments+1):
	red[i] = np.convolve(red[i], weights[::-1], 'same')
	blue[i] = np.convolve(blue[i], weights[::-1], 'same')
	green[i] = np.convolve(green[i], weights[::-1], 'same')
"""

for i in range(numSegments+1):
	red[i]/= np.sum(red[i])
	green[i]/= np.sum(green[i])
	blue[i]/= np.sum(blue[i])	

dist = np.zeros([numSegments+1, numSegments+1])
cdist = np.zeros([numSegments+1, numSegments+1])


nn = joblib.load('ANN.pkl') 

for i in range(numSegments+1):
	for j in range(numSegments+1):
		diff = 0.0	
		"""
		method = cv2.HISTCMP_INTERSECT

		hist1 = red[i].ravel().astype('float32')
		hist2 = red[j].ravel().astype('float32')
		diff+= abs(cv2.compareHist(hist1, hist2, method))

		hist1 = green[i].ravel().astype('float32')
		hist2 = green[j].ravel().astype('float32')
		diff+= abs(cv2.compareHist(hist1, hist2, method))

		hist1 = blue[i].ravel().astype('float32')
		hist2 = blue[j].ravel().astype('float32')
		diff+= abs(cv2.compareHist(hist1, hist2, method))
		"""

		# ann CLASSIFIER
		
		HIST1 = np.hstack((red[i],green[i],blue[i])).reshape(1,-1);
		HIST2 = np.hstack((red[j],green[j],blue[j])).reshape(1,-1);
		
		INP = np.hstack((HIST1, HIST2)).reshape(1,-1);
		#print INP.shape;
		diff = nn.predict_proba(INP);
		diff = diff[0][1];
		#print i,',',j,' :',diff;
		
		dist[i][j] = diff

ds = disjointset.DisjointSet(numSegments+1)
print 'Original superpixels:', ds.get_num_sets()

for i in range(row):
	for j in range(col):
		a = i
		b = j-1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i
		b = j+1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i-1
		b = j
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i+1
		b = j
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i-1
		b = j-1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i-1
		b = j+1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i+1
		b = j-1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

		a = i+1
		b = j+1
		if(a>=0 and a<row and b>=0 and b<col):
			if(not superpixels[i][j]==superpixels[a][b]):
				cdist[superpixels[i][j]][superpixels[a][b]] = dist[superpixels[i][j]][superpixels[a][b]]

threshold = 0.8 #input('Enter the threshold: ')
threshold*= np.amax(cdist)


for i in range(numSegments+1):
	for j in range(numSegments+1):
		if cdist[i][j] > threshold:
			if not ds.are_in_same_set(i, j):
				ds.merge_sets(i, j)

for i in range(row):
	for j in range(col):
		mergedSuperpixels[i][j] = ds._get_repr(superpixels[i][j])

# Statistics
print 'Reduced superpixels:', ds.get_num_sets()

#region = region.astype(float);


showColorImage(region);
showColorImage(superpixels);
showColorImage(mergedSuperpixels);
plt.show();


"""
print min(image.ravel()),', ',max(image.ravel())
print min(region.ravel()),', ',max(region.ravel())

print type(region), type(image)
print image.shape, region.shape

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(mark_boundaries(image, region))

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(mark_boundaries(image, superpixels))

ax2 = fig.add_subplot(2,2,3)
ax2.imshow(mark_boundaries(image, mergedSuperpixels))

plt.show()
"""

