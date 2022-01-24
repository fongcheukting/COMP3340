# Lecture 4: Semantic Segmentation

## 1. Introduction

In this section we will introduce the task of semantic segmentation, which is the task of classifying each pixel in an image from a predefined set of classes. In the following example, different entities are classified. In the below example, the pixels belonging to the bed are classified in the class “bed”, the pixels corresponding to the walls are labeled as “wall”, etc. Semantic segmentation has a wide range of applications, such as human parsing, matting and autonomous driving.



![demo](https://divamgupta.com/assets/images/posts/imgseg/image15.png?style=centerme)
(Figure 1: Semantic segmentation of a bedroom image)

## 2. From Image Classification to Semantic Segmentation

For image classification, the goal is to input an image with shape HxWx3, and output one predicted class ID. 
However, the goal of semantic segmentation is to take an image of size W x H x 3 and generate a W x H matrix containing the predicted class ID’s corresponding to all the pixels, as shown below:
![task](https://divamgupta.com/assets/images/posts/imgseg/image14.png?style=centerme)


### Difference of semantic segmentation and object detection
Semantic segmentation is different from object detection as it does not predict any bounding boxes around the objects. We do not distinguish between different instances of the same object. For example, there could be multiple cars in the scene and all of them would have the same label.

![diff](https://divamgupta.com/assets/images/posts/imgseg/image7.png?style=centerme)
(An example where there are multiple instances of the same object class)

### Semantic Segmentation using Convolutional Neural Networks
The Convolutional Neural Network (CNN) often contains several convolutional layers, non-linear activations, batch normalization, and pooling layers. The shallow layers tend to learn the low-level concepts such as edges and colors and the deep layers learn the higher level concepts such as different objects.

Different from image classification, semantic segmentation need to retain the spatial information, so **fully convolutional networks** are adopted for this task. The convolutional layers coupled with downsampling layers produce a low-resolution tensor containing the high-level information. A brief pipeline is shown as below:

![piepline](https://divamgupta.com/assets/images/posts/imgseg/image5.png?style=centerme)

## 3. Representative Methods

### FCN: 
FCN is the first end-to-end semantic segmentation model. Here standard image classification models such as VGG and AlexNet are converted to fully convolutional by making FC layers 1x1 convolutions. At FCN, transposed convolutions are used to upsample.

![FCN](https://divamgupta.com/assets/images/posts/imgseg/image2.png?style=centerme)


**Sample code.** pytorch code of FCN
```python
class _FCNHead():
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)
        
class FCN():
    def __init__(self):
        self.decoder = _FCNHead(C, self.nclass) # init FCN head

    def forward(self, img):
        # get multi-level features from backbone encoder
        size = img.size()[2:] 
        _, _, c3, c4 = self.forward_backbone(img)
        
        # go through decoder
        x = self.decoder(c4)
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return out
```


### PSPNet
The Pyramid Scene Parsing Network is optimized to learn better global context representation of a scene. The key module is Pyramid Pooling module. The feature map from encoder is downsampled to different scales. Convolution is applied to the pooled feature maps. After that, all the feature maps are upsampled to a common scale and concatenated together. Finally a another convolution layer is used to produce the final segmentation outputs. Here, the smaller objects are captured well by the features pooled to a high resolution, whereas the large objects are captured by the features pooled to a smaller size.

![PSPNet](https://divamgupta.com/assets/images/posts/imgseg/image11.png?style=centerme)

**Sample code.** pytorch code of Pyramid Pooling
```python
class PyramidPooling():
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, **kwargs):
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpools.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer, **kwargs))

    def forward(self, x):
        # x is feature map output from encoder
        size = x.size()[2:]
        feats = [x]
        # do pyramid pooling operation
        for (avgpool, conv) in zip(self.avgpools, self.convs):
            _x = conv(avgpool(x))
            feats.append(F.interpolate(_x, size, mode='bilinear'))
        return torch.cat(feats, dim=1)
```



### Deeplab

Deeplab uses an ImageNet pre-trained ResNet as its main feature extractor network. However, it proposes a new Residual block for multi-scale feature learning. Instead of regular convolutions, the last ResNet block uses atrous convolutions. Also, each convolution (within this new block) uses different dilation rates to capture multi-scale context.

Additionally, on top of this new block, it uses Atrous Spatial Pyramid Pooling (ASPP). ASPP uses dilated convolutions with different rates as an attempt of classifying regions of an arbitrary scale.

![deeplab](https://sthalles.github.io/assets/deep_segmentation_network/network_architecture.png)


#### Atrous Convolutions
Atrous (or dilated) convolutions are regular convolutions with a factor that allows us to expand the filter’s field of view.

Consider a 3x3 convolution filter for instance. When the dilation rate is equal to 1, it behaves like a standard convolution. But, if we set the dilation factor to 2, it has the effect of enlarging the convolution kernel.

In theory, it works like that. First, it expands (dilates) the convolution filter according to the dilation rate. Second, it fills the empty spaces with zeros - creating a sparse like filter. Finally, it performs regular convolution using the dilated filter.
![aspp](https://sthalles.github.io/assets/deep_segmentation_network/atrous_conv.png)
(Atrous convolutions with different rates.)

As a consequence, a convolution with a dilated 2, 3x3 filter would make it able to cover an area equivalent to a 5x5. Yet, because it acts like a sparse filter, only the original 3x3 cells will do computation and produce results. I said “act” because most frameworks don’t implement atrous convolutions using sparse filters - because of memory concerns.

In a similar way, setting the atrous factor to 3 allows a regular 3x3 convolution to get signals from a 7x7 corresponding area.

This effect allows us to control the resolution at which we compute feature responses. Also, atrous convolution adds larger context without increasing the number of parameters or the amount of computation.


**Sample code.** pytorch code of ASPP

```python
class ASPPModule():
    """Atrous Spatial Pyramid Pooling (ASPP) Module.
    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs
```


## 4. Metrics for Semantic Segmentation
Intuitively, a successful prediction is one which maximizes the overlap between the predicted and true objects. Two related but different metrics for this goal are the Dice and Jaccard coefficients (or indices):

![](https://i.imgur.com/lSxN30m.png)

Here, $A$ and $B$ are two segmentation masks for a given class, $||A||$ is the norm of $A$ (for images, the area in pixels), and $∩$, $∪$ are the intersection and union operators.

Both the Dice and Jaccard indices are bounded between 0 (when there is no overlap) and 1 (when A and B match perfectly). The Jaccard index is also known as Intersection over Union (IoU) and because of its simple and intuitive expression is widely used in computer vision applications.

In terms of the confusion matrix, the metrics can be rephrased in terms of true/false positives/negatives:

![](https://i.imgur.com/AzTl0AG.png)

Here is an illustration of the Dice and IoU metrics given two circles representing the ground truth and the predicted masks for an arbitrary object class:

![](https://ilmonteux.github.io/assets/images/segmentation/metrics_iou_dice.png)


## 5. Recommended Reading
Above representative methods are from 2015-2017. In this section, we will provide some new papers, these papers are published from 2018 to now.

**Real-time semantic segmentation**
ICNet for Real-Time Semantic Segmentation on High-Resolution Images [[paper link]](https://arxiv.org/abs/1704.08545)
Fast-SCNN: Fast Semantic Segmentation Network [[paper link]](https://arxiv.org/abs/1902.04502)
BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation [[paper link]](https://arxiv.org/abs/1808.00897)

**Transformer based semantic segmentation**
Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers [[paper link]](https://arxiv.org/abs/2012.15840)
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers [[paper link]](https://arxiv.org/abs/2105.15203)

## 6. Quiz

### Q1: Implement the forward function of DeeplabV3+ 

```python
class ASPPHead():
    """Rethinking Atrous Convolution for Semantic Image Segmentation.
    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        # TODO: Please complete this function,start
        # aspp_outs = [
        #     resize(
        #         self.image_pool(x),
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        # aspp_outs.extend(self.aspp_modules(x))
        # aspp_outs = torch.cat(aspp_outs, dim=1)
        # output = self.bottleneck(aspp_outs)
        # TODO: Please complete this function,end
        output = self.cls_seg(output)
        return output
```

### Q2: Write the code to calculate IoU of two generated binary masks.

Question: 

```python=
import numpy as np

a = np.random.randint(2, size=(10,10))
b = np.random.randint(2, size=(10,10))

IoU = ?
```

Answer:

```python=
import numpy as np

a = np.random.randint(2, size=(10,10))
b = np.random.randint(2, size=(10,10))
intersect = (a & b).sum()
union = (a | b).sum()
IoU = intersect / union 
```





