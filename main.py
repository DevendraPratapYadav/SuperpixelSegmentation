
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

import cv2
import os
import subprocess
import numpy as np
import disjointset
import matplotlib.pyplot as plt
import scipy.stats.mstats as stat
import pdb
 
# Input and load image
serial = raw_input('Enter the image serial number: ')
serial = str(serial)
image = img_as_float(io.imread('./release/images/'+serial+'.jpg'))
region = np.loadtxt('./release/labels/'+serial+'.regions.txt')
row, col, dim = np.shape(image)
 
# Initial superpixel segmentation
# superpixels = slic(image, n_segments=600, max_iter=10, sigma=4, slic_zero=True)
# superpixels = felzenszwalb(image, scale=3.0, sigma=0.95, min_size=5)
superpixels = quickshift(image, ratio=1, kernel_size=5, max_dist=10, convert2lab=True, random_seed=42)
mergedSuperpixels = np.zeros([row, col])
numSegments = np.amax(superpixels)

# Compute color histograms 
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

# Histogram smoothing 
weights = [0.1, 0.2, 0.4, 0.2, 0.1]
weights = np.array(weights)
for i in range(numSegments+1):
	red[i] = np.convolve(red[i], weights[::-1], 'same')
	blue[i] = np.convolve(blue[i], weights[::-1], 'same')
	green[i] = np.convolve(green[i], weights[::-1], 'same')

# Histogram normalization 
for i in range(numSegments+1):
	red[i]/= np.sum(red[i])
	green[i]/= np.sum(green[i])
	blue[i]/= np.sum(blue[i])	

# Histogram similarities - pre processing 
dist = np.zeros([numSegments+1, numSegments+1])
cdist = np.zeros([numSegments+1, numSegments+1])

for i in range(numSegments+1):
	for j in range(numSegments+1):
		diff = 0.0		
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

		dist[i][j] = diff

# Find comparable superpixels 
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

# Thresholding and merging 
threshold = input('Enter the threshold: ')
threshold*= np.amax(cdist)

for i in range(numSegments+1):
	for j in range(numSegments+1):
		if cdist[i][j] > threshold:
			if not ds.are_in_same_set(i, j):
				ds.merge_sets(i, j) 


# YOLO 
def getBbox(image):
	cols = np.any(image, axis=0)
	rows = np.any(image, axis=1)
	cmin, cmax = np.where(cols)[0][[0, -1]]
	rmin, rmax = np.where(rows)[0][[0, -1]]
	return [rmin, cmin, rmax, cmax];

os.chdir('darknet')
open('../bbox.txt', 'w').close()
filepath = '../release/images/'+serial+'.jpg'
subprocess.call("./darknet detect cfg/yolo-voc.cfg yolo-voc.weights "+filepath+" >> ../bbox.txt", shell=True)

fdesc = open('../bbox.txt', 'r')
lines = [line.rstrip('\n') for line in fdesc]
lines.pop(0)

bbox = []
numBoxes = 0

for i in lines:
	box = i.split(' ')
	box = map(int, box)
	bbox.append(box)
	numBoxes+= 1

membership = np.zeros([numSegments+1, numBoxes])

for i in range(numSegments+1):
	temp = np.array(superpixels==i)
	temp = temp.astype('int')
	superbox = getBbox(temp)


	for j in range(numBoxes):
		if superbox[0]>=bbox[j][0] and superbox[1]>=bbox[j][1] and superbox[2]<=bbox[j][2] and superbox[3]<=bbox[j][3]:
			membership[i][j] = 1

for i in range(numBoxes):
	temp = []
	for j in range(numSegments+1):
		if membership[j][i]==1:
			temp.append(j)

	for j in range(len(temp)):
		for k in range(len(temp)):
			if k<=j:
				continue
			else:
				if not ds.are_in_same_set(temp[j], temp[k]) and cdist[temp[j]][temp[k]] > threshold/3:
					ds.merge_sets(temp[j], temp[k])
					

# Merging superpixels 
for i in range(row):
	for j in range(col):
		mergedSuperpixels[i][j] = ds._get_repr(superpixels[i][j])

# Statistics
print 'Reduced superpixels:', ds.get_num_sets()

# Show output 
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(mark_boundaries(image, region))

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(mark_boundaries(image, superpixels))

ax2 = fig.add_subplot(2,2,3)
ax2.imshow(mark_boundaries(image, mergedSuperpixels))

plt.show()

io.imsave('result.jpg', mark_boundaries(image, mergedSuperpixels))
