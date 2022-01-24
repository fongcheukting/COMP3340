# Lecture 3: Detection

## 1. Introduction

In this section we will introduce the Object Detection problem, which is the task of localizing a set of objects and recognizing their categories in an image. This is one of the fundamental problems in Computer Vision. Its applications are everywhere, for example, face detection, autonomous driving, unmanned supermarket, etc.

![demo](https://peizesun.github.io/lecture_notes/demo.jpg)
(Figure is from SSD: Single Shot MultiBox Detector)

## 2. From Image Classification to Object Detection
### Single object detection
Supoose there is only one object in the image, then we can use image classification we learned from Lecture 2 to recognize the category of the object, and treat localization as a regresssion problem.

### Multiple object detection
An image always contains multiple objects, and different image may contain different number of objects. The most intuitive method is to crop many image patches and treat each as single object detection. However, an exhaustive list of potential locations, scales and aspect ratios of patches are needed to obtain a good result. Runing such a huge times of single object detection is extremely time-consuming. 

### Selective Search
To speed up, Selective Search is proposed to find the image patches that are likely to contain objects. It gives 2000 region proposals in a few seconds on CPU. Modern object detection seldom uses selective search to provide region proposals. More details about [selective search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) are for reading after class.

### NMS
NMS (Non-maximum Suppression) is designed to filter the proposals based on some criteria. At the most basic level, most object detectors do some form of windowing. Thousands of windows (anchors) of various sizes and shapes are generated. Once the detector outputs a large number of bounding boxes, it is necessary to filter out the best ones. NMS is the most commonly used algorithm for this task. Typically, NMS filters proposals according to the IOU between the bounding boxes. 

![NMS](https://miro.medium.com/max/1400/1*6d_D0ySg-kOvfrzIRwHIiA.png)

## 3. Representative Object Detection Methods
### R-CNN
R-CNN(Region-Convolutional Neural Network) is the first deep-learning based object detector. The idea of R-CNN is very simple and intuitive but its accuracy is surprising at that time!

The pipeline of R-CNN:
(1) Selective Search gives 2000 region proposals from the input image.
(2) Crop the image patches of region proposals and resize them to a fixed size.
(3) Forward each resized image patch through ConvNet to get feature vector
(4) Classify the category and regress the box.
(5) NMS
![rcnn](https://peizesun.github.io/lecture_notes/rcnn.jpg)
(Figure is from Rich feature hierarchies for accurate object detection and semantic segmentation)

The problem of R-CNN is obvious. It forwars ConvNet for each image patch, totally about 2000 times for a single image. To speed up R-CNN, Fast R-CNN proposes to forward ConvNet once for a single time. 

### Fast R-CNN
The pipeline of Fast R-CNN:
(1) Selective Search gives 2000 region proposals from the input image.
(2) Forward the whole image through ConvNet to get feature map
(3) Crop the feature of each region proposal from the feature map by RoI-Pooling
(4) Classify the category and regress the box.
(5) NMS
![fastrcnn](https://peizesun.github.io/lecture_notes/fastrcnn.jpg)
(Figure is from Fast R-CNN)

RoI-Pooling produces the fixed-size feature maps from non-uniform region proposals. The details of RoI-Pooling and its improved version of RoI-Align are in Quiz.

### Faster R-CNN
Since Fast R-CNN, the speed bottleneck lies in selective search. To further speed up, Faster R-CNN proposes Region Proposal Network to predict region proposals. Others are the same as Fast R-CNN.

![fasterrcnn](https://peizesun.github.io/lecture_notes/fasterrcnn.jpg)
(Figure is from Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks)

#### Anchor box
Anchor boxes are a set of default boxes over different aspect ratios and scales per feature map location. Region Proposal Network pre-defines anchor boxes and predict whether each anchor box contains an object, if any, predict the corrections from the anchor box to the ground-truth box.

![anchor](https://peizesun.github.io/lecture_notes/anchor.jpg)
(Figure is from SSD: Single Shot MultiBox Detector)

#### Region Proposal Network
The Region Proposal Network(RPN) is actually two convolutional layer after ConvNet, one is for classification, other is for regression.
The pipeline of RPN:
(1) Classify the foreground and regress the box.
(2) Select 2000 top-scoring boxes 
(3) NMS
(4) Output regions proposals

#### Dissussion
**Question**: How can RPN run faster than selective search when both enumerate all potential locations, scales and aspect ratios of object boxes ?
**Answer**: Forwards ConvNet once !

In general, R-CNN family is a two-stage object detector, the first stage provides regions proposals, and the second stage predict the category and box location for each proposal. On the other hand, one-stage detectors apply a single ConvNet to predicts bounding boxes and class probabilities directly from full images in one shot.

### YOLO
YOLO is a widely-used one-stage detetor. The pipeline of YOLO:
(1) Forward the whole image through ConvNet to get feature map
(2) Classify the category and regress the box.
(3) NMS
![yolo](https://peizesun.github.io/lecture_notes/yolo.jpg)
(Figure is from YOLO: Unified, Real-Time Object Detection)

#### Dissussion
**Question**: What is relationship of YOLO and RPN ?
**Answer**: YOLO can be seen as multi-category RPN !

### SSD
SSD borrows the concept of anchor box and multi-level feature maps to one-stage detector.
![ssd](https://peizesun.github.io/lecture_notes/comparison2.jpg)
(Figure is from SSD: Single Shot MultiBox Detector)

## 4. Accuracy vs Speed
For a long time, the researchers in object detection agree that two-stage is slower but more accurate, and one-stage is much faster but not as accurate. However...

### RetinaNet
RetinaNet is the first one-stage detector surpassing the accuracy of two-stage. Its paper proposes that one-stage detector is not born with lower accuracy than two-stage detectors. The extreme foreground-background class imbalance encountered during training is the central cause. More details about [RetinaNet](https://arxiv.org/abs/1708.02002) are for reading after class.
![retinanet](https://peizesun.github.io/lecture_notes/retinanet.jpg)
(Figure is from Focal Loss for Dense Object Detection)

### Dissussion
If we review the development of one-stage detectors, we could see:
(1) SSD applies multi-level feature maps than single-level feature map of YOLO.
(2) RetinaNet applies heavier classification head and regression head (4 consecutive convolutional layer) than SSD and YOLO.

This demonstrates the rule of deep learning: Bigger and deeper models usually work better.

## 5.Recommended reading
The methods introduced above are published before 2017. Since 2018, object detection methods develop towards to anchor-free and NMS-free.
#### Anchor-free:
CornerNet: Detecting Objects as Paired Keypoints [[paper link]](https://arxiv.org/abs/1808.01244)
FCOS: Fully Convolutional One-Stage Object Detection [[paper link]](https://arxiv.org/abs/1904.01355)
Objects as Points [[paper link]](https://arxiv.org/abs/1904.07850)
YOLOX: Exceeding YOLO Series in 2021 [[paper link]](https://arxiv.org/abs/2107.08430)

#### NMS-free:
End-to-End Object Detection with Transformers [[paper link]](https://arxiv.org/abs/2005.12872)
Deformable Transformers for End-to-End Object Detection [[paper link]](https://arxiv.org/abs/2010.04159)
Sparse R-CNN: End-to-End Object Detection with Learnable Proposals [[paper link]](https://arxiv.org/abs/2011.12450)
What Makes for End-to-End Object Detection? [[paper link]](https://arxiv.org/abs/2012.05780)

# Quiz
### 1. RoI Pooling and RoI Align
#### RoI Pooling
ROI pooling layer produces the fixed-size feature maps from non-uniform inputs by doing max-pooling / avg-pooling on the input. Like most 2D pooling operaters, the number of output channels is equal to the number of input channels. ROI pooling applies quantization twice. First time in the mapping process and the second time during the pooling process. For the mapping process, the proposal coordinate is rescaled to match the feature map size. For the pooling process, the roi bin is calculated to determine which feature points are included in for pooling operation.

**Sample code.** ROI Pooling
```python
# Mapping
# rois is a list containing the start point and the end point
# spatial_scale_ is the ratio of the proposal to feature map size
int roi_start_w = round(rois[1] * spatial_scale_);    
int roi_start_h = round(rois[2] * spatial_scale_);    
int roi_end_w = round(rois[3] * spatial_scale_);    
int roi_end_h = round(rois[4] * spatial_scale_);

float bin_size_h = float(roi_end_h - roi_start_h + 1) / pooled_height
float bin_size_w = float(roi_end_w - roi_start_w + 1) / pooled_width

# Pooling
# the start point and end point of ph-th row, pw-th col bin
# Each bin contains feature points in [ hstart, hend ), [ wstart, wend )
int hstart = floor(ph * bin_size_h) + roi_start_h
int wstart = floor(pw * bin_size_w) + roi_start_w
int hend = ceil((ph + 1) * bin_size_h) + roi_start_h
int wend = ceil((pw + 1) * bin_size_w) + + roi_start_w
```
![ROIPOOLING](https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif)


#### RoI Align
RoI Align is not using quantization for data pooling. Suppose that the feature map is 16x16 (for simplicity we ignore the channel dim), a proposal is mapped to a roi region: (9.25,6) — top left corner, 6.25 — width, 4.53 — height. For 3x3 roi pooled output, we divide original RoI into 9 equal size boxes and apply bilinear interpolation inside every one of them. To sample data we have to create 4 sampling points inside that box by dividing the height and width of the box by 3. (Four sampling points is the best choice according to Mask RCNN.)


![ROIALIGN](https://miro.medium.com/max/1400/0*7WFmQBxoOCPu2BDJ.gif)



### 2. NMS
NMS (Non-maximum Suppression) is designed to filter the proposals based on some criteria. At the most basic level, most object detectors do some form of windowing. Thousands of windows (anchors) of various sizes and shapes are generated. Once the detector outputs a large number of bounding boxes, it is necessary to filter out the best ones. NMS is the most commonly used algorithm for this task. Typically, NMS filters proposals according to the IOU between the bounding boxes. 

\begin{eqnarray}
IOU(Box1, Box2)  = Intersection\_Size(Box1, Box2) / Union\_Size(Box1, Box2) 
\end{eqnarray}


![NMS](https://miro.medium.com/max/1400/1*6d_D0ySg-kOvfrzIRwHIiA.png) 

**Procedure.** NMS
```python
Input: A list of Proposal boxes B, 
corresponding confidence scores S and overlap threshold N.

Output: A list of filtered proposals D.

1.Select the proposal with highest confidence score, remove it from B 
and add it to the final proposal list D. (Initially D is empty).

2.Compare this proposal with all the proposals: calculate the IOU 
(Intersection over Union) of this proposal with every other proposal. 
If the IOU is greater than the threshold N, remove that proposal from B.

3.This process is repeated until there are no more proposals left in B.

```

### 3. Mean Average Precision
The average precision (AP) is a way to summarize the precision-recall curve into a single value representing the average of all precisions. To calculate the mAP, start by calculating the AP for each class. The mean of the APs for all classes is the mAP. 
Average precision computes the average precision value for recall value over 0 to 1. Precision measures how accurate is your predictions. i.e. the percentage of your predictions are correct.
Recall measures how good you find all the positives. For example, we can find 80% of the possible positive cases in our top K predictions. 


\begin{eqnarray}
Precision = True\ Positives / Total\ Positives \\
Recall = True\ Positives / Total\ Ground\ Truth 
\end{eqnarray}

Suppose that there are ten predictions and the prediction is correct if IoU between prediction and ground truth ≥ 0.5. 
![AP1](https://miro.medium.com/max/1400/1*9ordwhXD68cKCGzuJaH2Rg.png)
Then plot the precision against the recall value to see this zig-zag pattern. Before calculating AP for the object detection, we often smooth out the zigzag pattern first. Mathematically, we replace the precision value for recall r with the maximum precision for any recall ≥ r.
![AP2](https://miro.medium.com/max/1400/1*TAuQ3UOA8xh_5wI5hwLHcg.jpeg)

Actually, AP is the Area under the Precision-Recall Curve. 
\begin{eqnarray}
& AP = \sum(r_{n+1} - r_{n})p_{interp}(r_{n+1}) \\
& p_{interp}(r_{n+1}) =  \max_{r > r_{n+1}}p(r)
\end{eqnarray}

