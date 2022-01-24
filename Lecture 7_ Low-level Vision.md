# Lecture 7: Low-level Vision

## Introdution

In this section, an overview of low-level vision is provided, including background information for understanding and a few key terms.


### David Marr’s representational framework [[10]](#cv)




In 1982, David Marr, a British neuroscientist, published another influential paper — “Vision: A computational investigation into the human representation and processing of visual information”[[]](#). Building on the ideas of Hubel and Wiesel (who discovered that vision processing always starts with simple structures such as oriented edges.), David gave us the next important insight: He established that vision is hierarchical. The vision system’s main function, he argued, is to create 3D representations of the environment so we can interact with it.


<center>
    <img src="https://i.imgur.com/I3e0szJ.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Stages of Visual Representation, David Marr, 1970s.[1] </div>
</center>

He introduced a framework for vision where low-level algorithms that detect edges, curves, corners, etc., are used as stepping stones towards a high-level understanding of visual data.

David Marr’s representational framework for vision includes:
* A Primal Sketch of an image, where edges, bars, boundaries etc., are represented (this is clearly inspired by Hubel and Wiesel’s research).
* A 2½D sketch representation where surfaces, information about depth and discontinuities on an image are pieced together.
* A 3D model that is hierarchically organized in terms of surface and volumetric primitives.




### Low-, Mid-, High-level Vision

Visual processing is typically considered in a hierarchical framework, comprising a series of discrete stages that successively produce increasingly abstract representations []. These different stages are often considered in terms of low-, mid- and high-level representations (figure shown following). Low-level vision is thought to involve the representation of elementary features, such as local colour, luminance or contrast.[[2]](#low-mid-high)

<center>
    <img src="https://i.imgur.com/Y5kFt16.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Illustration of Marr’s paradigm [Marr 1982] for a vision system.</div>
</center>



Digital image processing is the use of computer algorithms to perform image processing on digital images. Image processing is a method to perform some operations on an image, in order to get an enhanced image or to extract some useful information from it. It is a type of signal processing in which input is an image and output may be image or characteristics/features associated with that image. The continuum from image processing to computer vision can be broken up into low, mid- and high-level processes.[[11]](#thesis_classification)


![]()


<center>
    <img src="https://i.imgur.com/9undcoY.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">The image processing to computer vision low, mid, and high-level processes.</div>
</center>

### More Applications in Low-Level Vision


#### Style transfer [[13]](#style_transfer)
Style transfer is a computer vision technique that takes two images—a content image and a style reference image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be "painted" in the style of the style reference image.

A selfie, for example, would supply the image content, and the van Gogh painting would be the style reference image. The output result would be a self-portrait that looks like a van Gogh original!


<center>
    <img src="https://i.imgur.com/tn9228n.gif">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Style transfer using Instance Normalization.[12]</div>
</center>




#### Image inpainting[[14]](#image_inpainting_overview)


Image Inpainting is a task of reconstructing missing regions in an image. It could be considered as a conservation process where damaged, deteriorating, or missing parts of the image/video are filled in to present a complete one. It is an important problem in computer vision and an essential functionality in many imaging and graphics applications, e.g. object removal, image restoration, manipulation, re-targeting, compositing, and image-based rendering[[15]](#image_inpainting).

<center>
    <img src="https://i.imgur.com/8KR6MM1.gif">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;"> An illustration of image inpainting. It allows you to edit images with a smart retouching brush and then reconstruct the "damaged" area by using algorithms[15].
    </div>
    
</center>
https://www.nvidia.com/research/inpainting/index.html
https://paperswithcode.com/task/image-inpainting




#### Sketch to Photo [[16]](#sketch)

The sketch to photo aims to convert the rapidly executed freehand drawings to a realistic photo. Formally，it is very similar to the image style transfer task, but in practice it is more difficult to imagine a object photo given a sketch that is spatially imprecise and missing colorful details.


<center>
    <img src="https://i.imgur.com/k9zccz5.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">An illustration of the sketch to photo[16]. The left one is the input sketch, while the other two are the corresponding generated object photos.</div>
</center>












#### Image/video de-raining, de-hazing, de-snowing, etc.

In many applications such as drone-based surveillance and self-driving cars, one has to process images and videos containing undesirable artifacts such as rain, snow, and fog. Thus the relevant algorithms (e.g. de-raining[[17]](#image-deraining), de-hazing[[18]](#image-dehazing), and de-snowing) are developed. To solve these problems, the main challenges are two-fold. (1) For most cases, it is difficult or expensive to obtain the ground truth images/videos for model training (2) The algorithms need to deal with multiple types of degraded images. For example, the de-raining methods should effectively consider various shapes, scales, and densities of raindrops into their algorithms.


<center>
    <img src="https://i.imgur.com/i5vrlDr.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">An illustration of de-hazing, left is the original image and right is the de-hazed part.</div>
</center>

<center>
    <img src="https://i.imgur.com/UL90UDH.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">An illustration of de-raining, left is the input image and right is the de-rained one.</div>
</center>



<center>
    <img src="https://i.imgur.com/vhEURCY.gif">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
                padding: 2px;">An illustration of video de-hazing, left is the input image and right is the de-hazed one. 
        
<b>Here it is worth mentioning that, for the low-level video processing, we also need to take the smoothness of the temporal domain into account when designing the algorithms to avoid the jitter and strobe in the output videos.</b> </div>
</center>




## Image Quality Metrics



The quality factors of an image are: Contrast, brightness, spatial resolution, noise.[[4]](#IQA)

Objective Fidelity Criterias are based on mathematical formulations. MSE, PSNR are examples of objective fidelity criteria.

Subjective Fidelity Criterias are based upon the perception of an individual rather on any mathematical formulations. HVS is such a model. The quality is rated as very poor, poor, good, very good, excellent. UQI, SSIM, FSIM (FSIMc), GSM are examples of subjective fidelity criteria.



### Mean Squared Error (MSE)

The error between two images f(x, y) and g(x, y) might be negative. To avoid negative numbers, MSE is used.

$$
MSE=\frac{1}{MN}\sum^{M-1}_{i=0} \sum^{N-1}_{j=0} {[f(x,y)-g(x,y)]^2}
$$

MSE is simple to use and does not involve costly computations [[]](). It satisfies the interpretations of similarity, i.e., non-negativity, identity, symmetry, and triangular inequality [[]]()[[]](). Lower value of MSE means a good value, i.e., higher similarity of the reference image and distorted image. MSE works satisfactorily when distortion is mainly caused by contamination of additive noise [[]]().


### Peak Signal to Noise Ratio (PSNR)
PSNR is the ratio between the maximum power of a signal to the maximum power of noise signal. PSNR is measured with respect to peak signal power. Its unit is decibels. If f(x, y) is the original reference image and g(x, y) is the distorted image, then,

$$
PSNR=20\log_{10} \frac{L^2MN}{\sum^{M-1}_{i=0} \sum^{N-1}_{j=0} {[f(x,y)-g(x,y)]^2}}=20\log_{10}\frac{L^2}{MSE}
$$

where, M and N are dimensions of the image. L is the dynamic range of the image pixels. PSNR is useful if images with different dynamic ranges are compared, otherwise, it doesn’t contain any new information other than MSE [[]](). Higher value of PSNR means a good value. PSNR is an excellent measure of quality for white noise distortion[[]](). PSNR involves simple calculations, has clear physical meaning and is convenient in context of optimization but PSNR is not according to the characteristics of human visual system (HVS) [[]]().


### Structural Similarity Index (SSIM)
SSIM is a human visual system (HVS) feature based metric proposed by Wang et al. in [[]](). The HVS performs many image processing tasks which are superior than other models [[]](). SSIM measures the similarity between two images. It is an improvement over methods like MSE and PSNR [[]](). It is calculated over several windows of an image as,

$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}
$$
where, $\mu_x$ and $\mu_y$ are the average of $x$ and $y$ respectively. $\sigma_x^2$ and \sigma_y^2 are the variance of $x$ and $y$ respectively. $\sigma_{xy}$ is the covariance of $x$ and $y$.
Constraints $c_1 = (k_1L)^2$ and $c_2 = (k_2L)^2$. $L =$ dynamic range of pixel values $= 255$ (default). $k_1 = 0.01$ and $k_2 = 0.03$ (default)[[]]().



### Python Implementation 

#### skimage

##### Installation via pip [[20]](#skimage)
To install the latest scikit-image you’ll need at least Python 3.6. If your Python is older, pip will find the most recent compatible version.



```shell
# Update pip
python -m pip install -U pip
# Install scikit-image
python -m pip install -U scikit-image
```


##### Example of API Usage

```python=
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

MSE = mean_squared_error(img1, img2)
PSNR = peak_signal_noise_ratio(img1, img2)
SSIM = structural_similarity(img1, img2)

print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM: ', SSIM)
```



#### Numpy Implemention
An example of numpy implementation[[19]](#psnr_ssim_numpy) is shown for more more math details.


```python=
import numpy as np
import math
import cv2

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

```


## Percetual Losses

> Perceptual loss is a term in the loss function that encourages natural and perceptually pleasing results.[[5]](#perceptual_losses)



Perceptual loss functions are used when comparing two different images that look similar, like the same photo but shifted by one pixel. The function is used to compare high level differences, like content and style discrepancies, between images. A perceptual loss function is very similar to the per-pixel loss function, as both are used for training feed-forward neural networks for image transformation tasks. The perceptual loss function is a more commonly used component as it often provides more accurate results regarding style transfer.[[6]](#perceptual_loss_function)




### Why do we need a perceptual loss function? [[5]](#perceptual_losses)

One of the components influencing the performance of image restoration methods is a loss function, defining the optimization objective.

In the case of image restoration, the goal is to recover the impaired image to visually match the pristine undistorted counterpart. Thus we need to design the loss that would adhere to that goal.

While solving this problem we accept that developing the method that would perfectly recover the target image might be impossible, since the reconstruction problem is inherently ill-posed, i.e. for any distorted image there could be multiple plausible solutions that would be perceptually pleasing.




<center>
    <img src="https://i.imgur.com/1hzz2wm.jpg">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Results of training a super-resolution method (EDSR) with L2 and L1 losses. Image from BSD dataset.</div>
</center>


### How does a Perceptual Loss Function work?

In short, the perceptual loss function works by summing all the squared errors between all the pixels and taking the mean. This is in contrast to a per-pixel loss function which sums all the absolute errors between pixels. Johnson et al.[[7]](#perceptual_loss_for_ir) argues that perceptual loss functions are not only more accurate in generating high quality images, but also do so as much as three times faster, when optimized. The neural network model is trained on images where the perceptual loss function is optimized based upon high level features extracted from already trained networks.




<center>
    <img src="https://i.imgur.com/rss9bxZ.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">A loss network pretrained for image classification to define perceptual loss functions that measure perceptual differences in content and style between images. </div>
</center>


The image above represents the neural network that is trained to transform input images into output images. A pre-trained loss network used for image classification helps inform the loss functions. The pre-trained network helps to define the perceptual loss functions needed to measure the perceptual differences of the content and style between the images. 

<!--
### PyTorch Implementation

```python=
import torch
import torchvision

```
-->

## Haar Wavelet

***Wavelets*** are a mathematical tool for hierarchically decomposing functions. Wavelets allow any function to be described in terms of a coarse overall shape, plus details that range from broad to narrow. Regardless of whether the function of interest is an image, a curve, or a surface, wavelets provide an elegant technique for representing the levels of detail present. [[9]](#wavelets)

A ***Haar wavelet*** is the simplest type of wavelet. In discrete form, Haar wavelets are related to a mathematical operation called the Haar transform. The Haar transform serves as a prototype for all other wavelet transforms. [[3]](#haar_wiki)[[8]](#wavelets_book)


<!--https://www.cs.toronto.edu/~mangas/teaching/320/slides/CSC320L11.pdf -->

### 1D Haar wavelet transform

To get a sense for how haar wavelets work, let's start out with a simple example. Suppose we are given a one-dimensional "image" with a resolution of 4 pixels, having the following pixel values:

$$
\begin{bmatrix}
8 & 4 & 1 & 3
\end{bmatrix}
$$


This image can be represented in the Haar basis, the simplest wavelet basis, as
follows. Start by averaging the pixels together, pairwise, to get the new lower resolution image with pixel values:

$$
\begin{bmatrix}
6 & 2
\end{bmatrix}
$$


Clearly, some information has been lost in this averaging and downsampling process. In order to be able to recover the original four pixel values from the two averaged pixels, we need to store some detail coefficients, which capture that missing information. In our example, we will choose 2 for the first *detail coefficient*, since the average we computed is 2 less than 8 and 2 more than 4. This single number allows us to recover the first two pixels of our original 4-pixel image. Similarly, the second detail coefficient is $-1$, since $2 + (-1) = 1$ and $2 - (-1) = 3$. 


Summarizing, we have so far decomposed the original image into a lower-resolution 2-pixel image version and detail coefficients as follows:


| Resolution | Averages | Detail coefficients |
|:----------:|:--------:|:-------------------:|
|    $4$    |   $[8 ~4 ~1 ~3]$  |             |
|    $2$    |   $[6 ~2]$  |    $[2 ~-1]$   |



Repeating this process recursively on the averages gives the full decomposition:


| Resolution | Averages | Detail coefficients |
|:----------:|:--------:|:-------------------:|
|    $4$    |   $[8 ~4 ~1 ~3]$  |             |
|    $2$    |   $[6 ~2]$  |    $[2 ~-1]$   |
|    $1$    |   $[4]$  |    $[2]$   |


Finally, we will define the wavelet transform of the original 4-pixel image to be the single coefficient representing the overall average of the original image, followed by the detail coefficients in order of increasing resolution. Thus, for the one-dimensional Haar basis, the wavelet transform of our original 4-pixel image is given by

$$
\begin{bmatrix}
4 & 2 & 2 & -1
\end{bmatrix}
$$


Note that no information has been gained or lost by this process: The original image had 4 coefficients, and so does the transform. Also note that, given the transform, we can reconstruct the image to any resolution by recursively adding and subtracting the detail coefficients from the lower-resolution versions.




### Reconstructing a 1D image from its wavelet coefs

In the 1D Haar wavelet transform section, we obtain two types of coefficients from Haar wavelet transform:
* Coarse approximation (calculated by averaging two adjacient samples)
* Fine details (calculated by substracting two adjacient samples and dividing by 2)

For there are no information lost in the forward transform of Haar wavelet transform, the original image can be recovered completedly using simply addition and substration. Consider two adjacient samples $x$ and $y$, forward transform can be achieved by

$$
\begin{align}
\text{Average}, a = (x+y)/2, \\
\text{Difference}, d=(x-y)/2.
\end{align}
$$

And the inverse transform is applied to get the original smaple values:

$$
x = a+d, \\
y = a-d.
$$

Let us recover the wavelet transform obtained above $[4 ~2 ~2 ~-1]$. $a=[4]$ and $d=[2]$ are the average result and detail coefficients at Resolution 1. We can reconstruct the image at Resolution 2 using the above fomulation:

$$
x = a+d = 4+2 = 6, \\
y = a-d = 4-2 = 2.
$$

Then, the image at Resolution 2 is recovered as $[6 ~2]$. By applying the reconstruction operation recusively using the detail coefficients $[2, ~-1]$ at Resolution 2, the original image values are obtained. 





### The 2D Haar wavelet transform [[21]](#2d_haar)

To perform 2D Haar wavelet transform, we can simply do full 1D transform along one dimension and then do another full 1D transform along the other dimension. This is called standard decomposition.

<center>
    <img src="https://i.imgur.com/oXaAY8Y.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Standard decomposition of an image.[9] </div>
</center>


Or, we can perform 1D Haar transform alternatively along different dimensions, in each level. This is called nonstandard decomposition.

<center>
    <img src="https://i.imgur.com/vqsuNnD.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Non-standard decomposition of an image.[9] </div>
</center>

Here we focus on the non-standard decomposition. The non-standard decomposition gives coefficients for the non-standard construction of basis functions. 


The following figure illustrates a processing sample of two dimensional Haar wavelet transform performed on an example set with 16 values. 2D wavelet transform decomposes input data into a lower frequency component that represents the average of the input and three high frequency components of horizontal, vertical or diagonal values that represent the differences from the average in each direction. The overall average of the input 16 values, 50 in this case, is obtained by repeating transforms with upper-left values at each level. Higher frequency components at each level are also calculated simultaneously. 


<center>
    <img src="https://i.imgur.com/VgAC2O0.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Example of 2D Haar wavelet transform on 16 sample values </div>
</center>

The total amount of the data is not reduced at all by these transforms because both the original and the wavelet-decomposed data at each level has 16 values. That is, the original values can be completely restored by reconstructing these decomposed values at each level like the 1D Haar wavelet inverse transform has done.



## References

<text id="cs231n"> [1]: http://cs231n.stanford.edu/
<text id="low-mid-high"> [2]: Groen, Iris IA, Edward H. Silson, and Chris I. Baker. "Contributions of low-and high-level properties to neural processing of visual scenes in the human brain." Philosophical Transactions of the Royal Society B: Biological Sciences 372.1714 (2017): 20160102.
<text id="haar_wiki"> [3]: https://en.wikipedia.org/wiki/Haar_wavelet 
<text id="IQA"> [4]:Samajdar, Tina, and Md Iqbal Quraishi. "Analysis and evaluation of image quality metrics." Information Systems Design and Intelligent Applications. Springer, New Delhi, 2015. 369-378.
<text id="perceptual_losses"> [5]: https://towardsdatascience.com/perceptual-losses-for-image-restoration-dd3c9de4113
<text id="perceptual_loss_function"> [6]: https://deepai.org/machine-learning-glossary-and-terms/perceptual-loss-function
<text id="perceptual_loss_for_ir"> [7]:    Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European conference on computer vision. Springer, Cham, 2016.
<text id="wavelets_book"> [8]:  Walker, James S. A primer on wavelets and their scientific applications. CRC press, 2008.
<text id="wavelets"> [9]:  Stollnitz, Eric J., A. D. DeRose, and David H. Salesin. "Wavelets for computer graphics: a primer. 1." IEEE computer graphics and applications 15.3 (1995): 76-84.
<text id="cv"> [10]: https://medium.com/codex/an-intuitive-journey-to-object-detection-through-human-vision-computer-vision-and-cnns-58d15ac6578c
<text id="thesis_classification"> [11]: Image Classification Using Non Negative Matrix Factorization and Ensemble Methods for Classification Information Technology
<text id="instance_norm"> [12]:    Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
<text id="style_transfer"> [13]: https://www.fritz.ai/style-transfer/ 
<text id="image_inpainting_overview"> [14]:     https://paperswithcode.com/task/image-inpainting
<text id="image_inpainting"> [15]: https://www.nvidia.com/research/inpainting/index.html 
<text id="sketch"> [16]: Liu R, Yu Q, Yu SX. Unsupervised sketch to photo synthesis. In 16th European Conference  of Computer Vision, Glasgow, UK, August 23–28, 2020
<text id="image-deraining"> [17]: https://paperswithcode.com/task/single-image-deraining
<text id="image-dehazing"> [18]:https://paperswithcode.com/task/single-image-dehazing
<text id="psnr_ssim_numpy"> [19]:   https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
<text id="skimage"> [20]: https://scikit-image.org/docs/stable/install.html
<text id="2d_haar"> [21]: https://chengtsolin.wordpress.com/2015/04/15/real-time-2d-discrete-wavelet-transform-using-opengl-compute-shader/
    
    
    
    
## Quiz
    
### 1. PSNR and SSIM
<!--
(a) Compute the PSNR and SSIM of image $I$ given $I'$ as a reference image. 
    $$
    I=
    \begin{bmatrix}
    100 & 101 & 100 & 101 & 50 \\ 
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    \end{bmatrix}
    $$
    $$
    I'=
    \begin{bmatrix}
    100 & 101 & 100 & 101 & 50 \\ 
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    100 & 101 & 100 & 101 & 50 \\
    \end{bmatrix}
    $$
(b) Analyze the quality of image $I$ based on the computed results.
-->


(a) If two pictures have the same PSNR value, can we say that these two pictures are the same?
Reference solution:
No. PSNR compares the “true” pixel values of the original image to the degraded image. Two degraded images with the smae PSNR value may be different. 
    
(b)Explain the difference of PSNR and SSIM based on the following results.
![](https://i.imgur.com/j87DeDc.png)
    
Reference solution:
The main limitation of PSNR is that it relies strictly on numeric comparison and does not actually take into account any level of biological factors of the human vision system.  Made up of three terms, the structural similarity (SSIM) index estimates the visual impact of shifts in image luminance, changes in photograph contrast, as well as any other remaining errors, collectively identified as structural changes. 
    
    
### 2. Haar Wavelet

(a) Write down 4 coefficients ($f_{LL}, f_{LH}, f_{HL}, f_{HH}$) of 2D Haar wavelet. Given an image $I$, apply these 4 Haar Wavelet transforms on $I$ at 1 level.
    $$
    I=
    \begin{bmatrix}
    90 & 58 & 34 & 166 \\ 
    76 & 44 & 20 & 120 \\
    86 & 10 & 178 & 190 \\
    86 & 66 & 58 & 210 \\
    \end{bmatrix}
    $$
    
Reference solution:
    
4 coefficients ($f_{LL}, f_{LH}, f_{HL}, f_{HH}$) of 2D Haar wavelet:
$$
    f_{LL}=\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix},
    f_{LH}=\begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix},
    f_{HL}=\begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix},
    f_{HH}=\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix},
$$
    
Level 1:
    $$
    I'=
    \begin{bmatrix}
    67 & 85 & 40 & -58 \\ 
    62 & 159 & 24 & -41 \\
    7 & 12 & 0 & -8 \\
    -14 & 25 & 14 & 35 \\
    \end{bmatrix}
    $$  
  
<!--
(b) Explain the property or function of 4 bands of Haar Wavelet. In order to extract the edge structure infomation, which bands should be applied on the image?
  -->  
    
    
### 3. Blur Mask Generation
    
Complete the following program to blur an image with a specific kernel.
    kernel size: random selected from [3,9] * 2 + 1
    sigma: random selected from [0.1,10]
    
```python=
import numpy as np
import cv2

img = np.random.randint(0, 256, size=(256,256), dtype=np.uint8)
kernel_size =        # Fill in the statements
sigma =              # Fill in the statements
blur_img =           # Fill in the statements
```

Answer:
    
```python=
import numpy as np
import cv2

img = np.random.randint(0, 256, size=(256,256), dtype=np.uint8)
kernel_size = np.random.randint(3,9) * 2 + 1
sigma = np.random.uniform(0.1, 10)
blur_img =cv2.GuassianBlur(img, (kernel_size, kernel_size), sigma)
```
  