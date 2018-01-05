# SuperpixelSegmentation

## Visit : [Project report and results](https://github.com/DevendraPratapYadav/SuperpixelSegmentation/blob/master/Report.pdf)          

Project : Computer Vision [CSL462] - IIT Ropar

### Dataset used: [Stanford Background Dataset(SBD)](http://dags.stanford.edu/projects/scenedataset.html)

### HOW TO RUN : 
Prerequisites: Python 2.7, OpenCV 3.1, Darknet and YOLO

### Setup YOLO pretrained model: 
* Get cfg and model for "YOLOv2 544x544" [HERE](https://pjreddie.com/darknet/yolo/)
* Create a folder "darknet" in project directory. Place "cfg/yolo-voc.cfg" and yolo-voc.weights in the "darknet" folder. This pretrained YOLO model will be used to get object detection results. 

### Execute programs: 
1) Place 'release' folder (containing SBD dataset) in project directory

2) To test on a single image, run 'main.py'. Enter the exact image serial number
   as present in the SBD dataset, and the threshold value (between 0 to 1), when prompted.
   
3) To test on the entire dataset, run 'dataseteval.py'. It will generate three files,
   proposed_algorithm.txt, slic.txt and quickshift.txt, containing evaluated measures
   for each image.
   
   
 Evaluation measures used:
 * Boundary Recall
 * Undersegmentation Error  
 * Unexplained Variation
 
 The measures are explained in the Project Report linked at the top. 
 
 
 Currently, we get results competitive with SLIC superpixel algorithm, while obtaining much lesser number of superpixels which encapsulate semantic entities. Sample results are shown in the report.
 
 Work is still being done on the project to improve results and speed up the algorithm.
