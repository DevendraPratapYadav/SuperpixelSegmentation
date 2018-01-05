from numpy import *
import numpy as np
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color
from skimage import morphology
import timeit
from scipy import ndimage


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
	
def getPatches(img, MS):
	
	MS3d = np.repeat(MS[:, :, np.newaxis], 3, axis=2)
	
	MSunique = np.unique(MS);
	MSSuperpixelCount = MSunique.shape[0];
	
	MSPixelCount = np.bincount(MS.ravel());
	
	# Superpixel patches
	spMean = [None] * (max(MSunique)+1);
	
	for i in xrange(0,MSSuperpixelCount):
	
		spId = MSunique[i];
		mask = (MS3d==spId).astype(float);
		spImg = np.multiply(img, mask);
		mask2d = mask[:,:,0];
		
		bb = getBbox(mask2d.astype(int));
		mask2d = mask2d[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ];
		spImg = spImg[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ];
		
		# if mask is very small; less area than 16x16, convert image to square and use it as patch
		
		
		
		com = ndimage.measurements.center_of_mass(mask2d)
		psize = int(min(mask2d.shape[0], mask2d.shape[1])/2);
		CBox = [ int(com[0]-psize/2), int(com[1]-psize/2) ];
		
		patches = [];
		maxDev = [(mask2d.shape[0]-psize)/2, (mask2d.shape[1]-psize)/2];
		for j in xrange(0,10):
			Dev = np.random.uniform(-1,1,2)*maxDev;
			Rbox = CBox + Dev;
			Rbox = Rbox.astype(int);		
			Box = [Rbox[0], Rbox[1], Rbox[0]+psize , Rbox[1]+psize ]
			myPatch = spImg[ Box[0]:Box[2]+1 , Box[1]:Box[3]+1] ;
			
			
			#ti = (color.rgb2gray(spImg) > 0.1).astype(int)
			
			myMask = mask2d[ Box[0]:Box[2]+1 , Box[1]:Box[3]+1] ;
			print mean(myMask);
			
			#showGrayImage(myPatch);
			#showGrayImage(myMask);
			#plt.show();
			
		
		showColorImage(spImg);
		plt.show();
		


def getHistogram(img,mask):
	img = (img*255).astype(int);
	
	rr = np.bincount(img[:,:,0].ravel());
	gg = np.bincount(img[:,:,1].ravel());
	bb = np.bincount(img[:,:,2].ravel());
	
	r = zeros(256); g = zeros(256); b = zeros(256);
	
	r[0:rr.shape[0]] = rr; 
	g[0:gg.shape[0]] = gg;
	b[0:bb.shape[0]] = bb;
	
	nz = (mask.shape[0]*mask.shape[1])-sum(mask);
	
	r[0]-=nz; g[0]-=nz; b[0]-=nz;
	r= r.astype(float); g= g.astype(float); b= b.astype(float);
	
	r/=sum(r)+1; g/=sum(g)+1; b/=sum(b)+1;
	
	"""
	# blur the histograms
	weights = [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05]
	weights = np.array(weights)
	
	r = np.convolve(r, weights[::-1], 'same')
	b= np.convolve(b, weights[::-1], 'same')
	g= np.convolve(g, weights[::-1], 'same')
	"""
	
	HIST = array([r,g,b]);
	
	#HIST = mean(HIST,0);
	#print r.shape, ', ',g.shape, ', ',b.shape;
	
	return HIST
	
def getHistogramPairs(img, MS):
	
	MS = abs(MS);
	MS3d = np.repeat(MS[:, :, np.newaxis], 3, axis=2)
	
	MSunique = np.unique(MS);
	MSSuperpixelCount = MSunique.shape[0];
	
	#print MS.ravel();
	#print MSunique;
	
	MSPixelCount = np.bincount(MS.ravel());
	
	# Superpixel patches
	spPatches = [None] * (max(MSunique)+1);
	
	for i in xrange(0,MSSuperpixelCount):
	
		spId = MSunique[i];
		mask = (MS3d==spId).astype(float);
		
		spImg = np.multiply(img, mask);
		mask2d = mask[:,:,0];
		
		bb = getBbox(mask2d.astype(int));
		mask2d = mask2d[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ];
		spImg = spImg[ bb[0]:bb[2]+1 , bb[1]:bb[3]+1 ];
		
		
		
		HIST = getHistogram(spImg, mask2d);
		
		HIST = HIST.ravel();
		#print HIST;
		spPatches[ spId ] = [HIST]; 
		
		#showColorImage(spImg);
		#plt.show();
		
		
		com = ndimage.measurements.center_of_mass(mask2d)
		psize = int(min(mask2d.shape[0], mask2d.shape[1])/2);
		CBox = [ int(com[0]-psize/2), int(com[1]-psize/2) ];
		
		maxDev = [(mask2d.shape[0]-psize)/2, (mask2d.shape[1]-psize)/2];
		for j in xrange(0,10):
			Dev = np.random.uniform(-1,1,2)*maxDev;
			Rbox = CBox + Dev;
			Rbox = Rbox.astype(int);		
			Box = [Rbox[0], Rbox[1], Rbox[0]+psize , Rbox[1]+psize ]
			myPatch = spImg[ Box[0]:Box[2]+1 , Box[1]:Box[3]+1] ;
			
			
			#ti = (color.rgb2gray(spImg) > 0.1).astype(int)
			
			myMask = mask2d[ Box[0]:Box[2]+1 , Box[1]:Box[3]+1] ;
			fillPercentage = mean(myMask);
			
			if (fillPercentage < 0.7):
				continue;
			
			HIST = getHistogram(myPatch, myMask);
			HIST = HIST.ravel();
			
			if (spPatches[ spId ] == None):
				spPatches[ spId ] = [HIST];
			else:
				spPatches[ spId ].append(HIST);
			
			#showGrayImage(myPatch);
			#showGrayImage(myMask);
			#plt.show();
			
		
		#print 'Total HISTs :', len(spPatches[ spId ]);
		
	return spPatches;	

def processImage(image,regions,numSamples):
	
	Pos = []; Neg = [];
	
	patches = getHistogramPairs(image,regions);
	print 'Superpixels: ',size(patches);

	PAT = [];
	for x in patches:
		if (x != None):
			PAT.append(x);

	if (len(PAT)<2):
		return Pos, Neg;
	
	pe =0 ; ne =0 ;
	
	for x in xrange(0,numSamples):
		a = np.random.randint(0,len(PAT));
		b = np.random.randint(0,len(PAT));
		if (b==a):
			b=(b+1)%(len(PAT));
		
		h1 = PAT[a][ np.random.randint(0,len(PAT[a])) ];
		h2 = PAT[a][ np.random.randint(0,len(PAT[a])) ];
		h3 = PAT[b][ np.random.randint(0,len(PAT[b])) ];
		
		#print 'Pair: ',a,', ',b;
		pe+= sum( abs(h1-h2) );
		ne+= sum( abs(h1-h3) );
		
		
		Pos.append( h1.tolist() + h2.tolist() );
		Neg.append( h1.tolist() + h3.tolist() );
		
	print 'Errors: ',pe/numSamples,', ',ne/numSamples;
	return Pos, Neg;
	
"""
img = readColorImage('win.png');
print img.shape

ssize = 64;
s1 = ones((ssize,ssize));
s2 = 2*ones((ssize,ssize));
s3 = 3*ones((ssize,ssize));
s4 = 4*ones((ssize,ssize));
TS = vstack (( hstack((s1,s2)) , hstack((s3,s4)) )).astype(int);

patches = getHistogramPairs(img,TS);
print size(patches);

PAT = [];
for x in patches:
	if (x != None):
		PAT.append(x);

#if (len(PAT)<2):
#	return;
	
numSamples = 100;

Pos = [];
Neg = [];

for x in xrange(0,numSamples):
	a = np.random.randint(0,len(PAT));
	b = np.random.randint(0,len(PAT));
	if (b==a):
		b=(b+1)%(len(PAT));
	
	h1 = PAT[a][ np.random.randint(0,len(PAT[a])) ];
	h2 = PAT[a][ np.random.randint(0,len(PAT[a])) ];
	h3 = PAT[b][ np.random.randint(0,len(PAT[b])) ];
	
	#print 'Pair: ',a,', ',b;
	
	Pos.append( h1.tolist() + h2.tolist() );
	Neg.append( h1.tolist() + h3.tolist() );
	
	
#print getHistogram(img,(TS>0).astype(int));
	
print len(Pos[0]);
	
"""


#image = img_as_float(io.imread('./release/images/'+serial+'.jpg'))
#region = np.loadtxt('./release/labels/'+serial+'.regions.txt')

FILE = open('./release/names.txt');
lines = FILE.readlines();
lines = [x.strip() for x in lines];
#print lines;
Fout = open('output.txt','w');
for x in lines:
	image = img_as_float(io.imread('./release/images/'+x+'.jpg'))
	#image = color.rgb2hsv(image);
	region = np.loadtxt('./release/labels/'+x+'.regions.txt')
	region = region.astype(int)
	
	#print image.shape,',',region.shape
	if ( (image.shape[0]!=region.shape[0] ) or (image.shape[1]!=region.shape[1] ) ):
		continue;
	
	print x
	#print region
	#showColorImage(image);
	#showGrayImage(region);
	#plt.show();
	Pos,Neg = processImage(image, region,10);
	
	print shape(Pos), ', ',shape(Neg)
	
	
	for p in Pos:
		Fout.write( '1,'+(',').join(map(str,p)) +'\n');
	for p in Neg:
		Fout.write( '0,'+(',').join(map(str,p)) +'\n');
	
	
	

	
	


