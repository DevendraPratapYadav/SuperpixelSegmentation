
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
import matplotlib.pyplot as plt
import pdb

def eval(segment, region, image):
	truth = mark_boundaries(image, region) - image
	pre = mark_boundaries(image, segment) - image
	shp = truth.shape
	
	#truth = np.sum(truth,axis=2)
	#pre = np.sum(pre,axis=2)
	
	truth = np.absolute(truth)
	pre = np.absolute(pre)

	truth[truth>0.0001] = 1
	pre[pre>0.0001] = 1
	
	""""
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(2,2,1)
	ax1.imshow(pre)
	plt.show();
	"""
	denom = np.sum(np.sum(np.sum(truth)))
	
	truth = truth - pre
	truth[truth>0] = 1
	truth[truth<0] = 0
	

	numer = np.sum(np.sum(np.sum(truth)))

	ratio = (numer)/denom
	print ratio
 
serial = '0000051'
image = img_as_float(io.imread('./release/images/'+serial+'.jpg'))
region = np.loadtxt('./release/labels/'+serial+'.regions.txt')
 
slics = slic(image, n_segments = 10, sigma = 5)

felzens = felzenszwalb(image, scale=3.0, sigma=0.95, min_size=50)

quicks = quickshift(image, ratio=1.0, kernel_size=10)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(mark_boundaries(image, region))

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(mark_boundaries(image, slics))
print 'SLIC'
eval(slics, region, image) 


plt.show()



