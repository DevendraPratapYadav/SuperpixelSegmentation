1. getPatches.py - place 'release' folder (containing SBD dataset) in same directory as this file and run this.
This will extract patches, create color histograms and write labelled training data to 'output.txt' file

2. NNR_CV.py - place 'output.txt' along with this file and run it to train a classifier and save it as 'ANN.pkl'

3. annSuperpixel.py - place 'ANN.pkl' and 'release' folder(containing SBD dataset) along with this file.
Give image index as command line argument and superpixel segmentation is shown as image output.
