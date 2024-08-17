# MV-Feature-Detection-and-Matching
CV courses of Cornell-CS5670.

## Introduction

The goal of **feature detection** and **matching** is to identify a pairing between a point in one image and a corresponding point in another image. These correspondences can then be used to stitch multiple images together into a panorama.
Click [here](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa2/index.html) to view projects introduction. 

## Features

* Feature detection: Identify points of interest in the image using the Harris corner detection method
* Feature description: Come up with a *descriptor* for the feature centered at each interest point. Implement a simplified version of the MOPS descriptor.  Compute an 8x8 oriented patch sub-sampled from a 40x40 pixel region around the feature. Come up with a transformation matrix which transforms the 40x40 rotated window around the feature to an 8x8 patch rotated so that its keypoint orientation points to the right. Normalize the patch to have zero mean and unit variance. If the variance is very close to zero (less than 10-5 in magnitude) then just return an all-zeros vector to avoid a divide by zero error.
* Feature Matching: Have detected and described your features, the next step is to write code to match them.
  * Sum of squared differences (SSD): This is the the squared Euclidean distance between the two feature vectors.
  * The ratio test: Find the closest and second closest features by SSD distance. The ratio test distance is their ratio (i.e., SSD distance of the closest feature match divided by SSD distance of the second closest feature match).
* Complete features descriptor that has attribute Scale Invariant Feature Transform (SIFT) 

## Implement
[View Implement Here](Feature-Detection-and-Matching.pdf)
