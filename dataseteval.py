
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from os import listdir

import cv2
import eval
import imghdr
import numpy as np
import disjointset
import matplotlib.pyplot as plt
import scipy.stats.mstats as stat
import pdb

file1 = open('proposed_algorithm.txt', 'w')
file2 = open('quickshift.txt', 'w')
file3 = open('slic.txt', 'w')

for filename in listdir('./release/images/'):
	try:
		filetype = imghdr.what('./release/images/'+filename)
		if not filetype=='jpeg':
			continue

		print filename
		filename = filename.split('.')
		
		serial = filename[0]

		image = img_as_float(io.imread('./release/images/'+serial+'.jpg'))
		region = np.loadtxt('./release/labels/'+serial+'.regions.txt')
		row, col, dim = np.shape(image)
		 
		# superpixels = slic(image, n_segments=600, max_iter=10, sigma=4, slic_zero=True)
		# superpixels = felzenszwalb(image, scale=3.0, sigma=0.95, min_size=5)
		superpixels = quickshift(image, ratio=1, kernel_size=5, max_dist=10, convert2lab=True, random_seed=42)
		mergedSuperpixels = np.zeros([row, col])
		numSegments = np.amax(superpixels)

		slics = slic(image, n_segments=20, max_iter=10, sigma=4, slic_zero=False)
		quicks = superpixels

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

		weights = [0.1, 0.2, 0.4, 0.2, 0.1]
		weights = np.array(weights)
		for i in range(numSegments+1):
			red[i] = np.convolve(red[i], weights[::-1], 'same')
			blue[i] = np.convolve(blue[i], weights[::-1], 'same')
			green[i] = np.convolve(green[i], weights[::-1], 'same')

		for i in range(numSegments+1):
			red[i]/= np.sum(red[i])
			green[i]/= np.sum(green[i])
			blue[i]/= np.sum(blue[i])	

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
				hist2 = green[j ].ravel().astype('float32')
				diff+= abs(cv2.compareHist(hist1, hist2, method))

				hist1 = blue[i].ravel().astype('float32')
				hist2 = blue[j].ravel().astype('float32')
				diff+= abs(cv2.compareHist(hist1, hist2, method))

				dist[i][j] = diff

		ds = disjointset.DisjointSet(numSegments+1)

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

		threshold = 0.5
		threshold*= np.amax(cdist)

		for i in range(numSegments+1):
			for j in range(numSegments+1):
				if cdist[i][j] > threshold:
					if not ds.are_in_same_set(i, j):
						ds.merge_sets(i, j)

		for i in range(row):
			for j in range(col):
				mergedSuperpixels[i][j] = ds._get_repr(superpixels[i][j])

		# Proposed algorithm 
		truth = find_boundaries(region).astype(np.uint8)
		predict = find_boundaries(mergedSuperpixels).astype(np.uint8)
		mergedSuperpixels = mergedSuperpixels.astype(np.uint8)

		recall = eval.eval_edge_recall(truth, predict, 5)
		underseg = eval.eval_undersegmetation_error(region, mergedSuperpixels)
		unex = eval.eval_unexplained_variation(image, mergedSuperpixels)

		print '\nProposed algorithm'
		print 'Edge recall:', eval.eval_edge_recall(truth, predict, 5)
		print 'Undersegmentation error:', eval.eval_undersegmetation_error(region, mergedSuperpixels)
		print 'Unexplained variation:', eval.eval_unexplained_variation(image, mergedSuperpixels)	

		file1.write(serial+' '+str(recall)+' '+str(underseg)+' '+str(unex)+'\n') 

		# SLIC 
		truth = find_boundaries(region).astype(np.uint8)
		predict = find_boundaries(slics).astype(np.uint8)
		slics = slics.astype(np.uint8)

		recall = eval.eval_edge_recall(truth, predict, 5)
		underseg = eval.eval_undersegmetation_error(region, slics)
		unex = eval.eval_unexplained_variation(image, slics)

		print '\nSLIC'
		print 'Edge recall:', eval.eval_edge_recall(truth, predict, 5)
		print 'Undersegmentation error:', eval.eval_undersegmetation_error(region, mergedSuperpixels)
		print 'Unexplained variation:', eval.eval_unexplained_variation(image, mergedSuperpixels)

		file3.write(serial+' '+str(recall)+' '+str(underseg)+' '+str(unex)+'\n')

		# Quickshift 
		truth = find_boundaries(region).astype(np.uint8)
		predict = find_boundaries(quicks).astype(np.uint8)
		quicks = quicks.astype(np.uint8)

		recall = eval.eval_edge_recall(truth, predict, 5)
		underseg = eval.eval_undersegmetation_error(region, quicks)
		unex = eval.eval_unexplained_variation(image, quicks)

		print '\nQuickshift'
		print 'Edge recall:', eval.eval_edge_recall(truth, predict, 5)
		print 'Undersegmentation error:', eval.eval_undersegmetation_error(region, mergedSuperpixels)
		print 'Unexplained variation:', eval.eval_unexplained_variation(image, mergedSuperpixels)
		print 

		file2.write(serial+' '+str(recall)+' '+str(underseg)+' '+str(unex)+'\n') 
	except KeyboardInterrupt:
		break
	except:
		pass

file1.close()
file2.close()
file3.close()
