**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/YCrCbHOG.png
[image3]: ./output_images/HLS_sameConfiguration.png
[image4]: ./output_images/YUV_sameConfiguration.png
[image5]: ./output_images/YCrCb_sameConfiguration.png
[image6]: ./output_images/slidingWindowSearch.png
[image7]: ./output_images/prove.png
[image8]: ./output_images/prove_.png
[image9]: ./output_images/heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

All the code are in "./Notebook_Final.ipynb"

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of 10 random images of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and `hog_channel=ALL`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters in order to find most efficient and fastest in order to get a good classifier. 

First, I tried a combination of parameters with "HLS", but the result was good. I had too group of boxes, I didn not how to change de threshold and other images where there was just one group instead of two. 
![alt text][image3]


Second, I tried a combination of parameters with "YUV". It work similar to the last one, but in the video it works better.
![alt text][image4]


Finally, I tried with "YCrCb" and I got excellent results. I had the quantity of group of boxes equal to the quantity of cars. The threshold always was 1, instead of the other ones.
![alt text][image5]

Now I had to check the parameters of the Sliding Window Search to verify that everything went well.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the feature vector that was obtained from the extract features and the y_train from 0's and 1's.
At the extract features, the parameters of the HOG were passed through. colorspac = 'YCrCb', orient = 9, pix_per_cell = 8, cell_per_block = 2 and hog_channel = "ALL".

Each feature vector have a lenght of 5292 and map to 1 or 0 depend if it is a car or non-car.

The classifier took 7.68 to train and got an accuracy of 98.51%. I think It is acceptable. When I calculated the accuracy of HLS and YUV, they were under that number.

Another thing, It takes 0.00313 seconds to predict 10 labels. It is relatively faster compared to other algoriths of classification.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The algorithm is restringed between 400 and 560 height. It is waste of time If I search in another place where the car is never going to be. 

![alt text][image6]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to optimize the performance I change to YCrCb 3-channel HOG features and using heat map to avoid false positives.

In this image we can see a detector that does not work well.
![alt text][image8]

In this image, the detector inside of the pipeline was improved avoiding false positives and combining boxes.
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 4 frames and their corresponding heatmaps:

#### Later, the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the results bounding boxes are drawn onto the last frame in the series:


![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

#### Use of Scaler
I am not using scaler because it is not necessary If I am not combining  features. HOG + color_hist/bin_spatial, for example.

I got good results, but adding color_hist/bin_spatial I would get even better.

#### SVM Classifier
I think If I would use gridsearch for hyper parameter optimizacion, I will get better performance.

Maybe, If I used Neural Networks or Decision Tree, I would get 100% accuracy in the part of classification. 

#### Fails of Pipeline video
Sometime when two cars are overlapped, the pipeline identify just one car. There are some techniques like RNN where you can predict where is going to be the car just looking last images. It works even If the cars are overlapped.

It would fails If I find one car very different to the dataset.

Next improvement, try YOLO algorithm.