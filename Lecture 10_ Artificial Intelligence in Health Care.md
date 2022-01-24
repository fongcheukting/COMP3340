# Lecture 10: Artificial Intelligence in Health Care

## Introduction

**Artificial intelligence in health care** is an overarching term used to describe the use of machine-learning algorithms and software, or artificial intelligence (AI), to mimic human cognition in the analysis, presentation, and comprehension of complex medical and health care data. 

The primary aim of health-related AI applications is to analyze relationships between prevention or treatment techniques and patient outcomes. AI programs are applied to practices such as diagnosis processes, treatment protocol development, drug development, personalized medicine, and patient monitoring and care. 

![avatar](https://360digit.b-cdn.net/assets/admin/ckfinder/userfiles/images/blog/ai-blog/Applications-of-AI-in-Life-Science-and-Health-Care-Industry%20(2).png "")

>[[1]](#x) Reference: https://360digitmg.com/

As shown above, AI can be used in many aspects of health care, from small molecule drug development and drug toxicity prediction to anatomical medical imaging and disease detection, etc. In this note, we will detail two specific applications: 

- Medical Image Registration
- Protein Structure Prediction

## Medical Image Registration

### Introduction

Within the current clinical setting, medical imaging is a vital component of a large number of applications. Such applications occur throughout the clinical track of events; Since information gained from two images acquired in the clinical track of events is usually of a complementary nature, proper integration of useful data obtained from the separate images is often desired. A first step in this integration process is to bring the modalities involved into spatial alignment, a procedure referred to as registration. After registration, a fusion step is required for the integrated display of the data involved. As to now, there has already many image registration application:

- Multi-modal registration for image-guided surgery
- Atlas-based image segmentation
- Longitudinal comparison of images for a given patient with the same imaging modality
- ...

Let's use a specific example to illustrate the medical image registration. Registering different modalities can be found in radiotherapy treatment planning, where currently CT is used almost exclusively. However, the use of MR and CT combined would be beneficial, as the former is better suited for delineation of tumor tissue (and has in general better soft tissue contrast), while the latter is needed for accurate computation of the radiation dose.  As shown in the figure, aligning the CT image (moving image) on the left with the MRI image (fixed image) in the middle spatially, we can obtain the registered image (result) on the right. 

![avatar](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-14442-5_2/MediaObjects/448565_1_En_2_Fig2_HTML.jpg)

[[2]](#mir) Reference: https://360digitmg.com/https://link.springer.com/chapter/10.1007/978-3-030-14442-5_2

### Definition

Given **moving image** and the **fixed image**,  the goal of medical image registration is to find the correspondence that aligns the moving image to the fixed image, the correspondence specifies the mapping between all pixel from one image to those from another image. The correspondence can be represented by a **dense displacement field (DDF)** , defined as a set of displacement vectors for all pixels (2D images) or voxels (3D images) from one image to another. By using these displacement vectors, the moving image can be ”warped” to become more ”similar” to fixed image.

As illustrated in the figure below,   A displacement field gives for every pixel position in the moving image the direction and distance how it has to move in order to match the fixed image. Therefore, we can use DDF to transform the moving image so that it is spatially aligned with the fixed image for the purpose of alignment. 

![image-20211024173217064](https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024173217064.png?raw=True)

### Method

Image registration has been an active area of research for decades. Historically, image registration algorithms posed registration as an optimization problem between a given pair of moving and fixed images. In this tutorial, we refer these algorithms as to the *classical methods* - if they only use a pair of images, as opposed to the *learning-based* algorithms, which require a seperate training step with many more pairs of training images (just like all other machine learning problems). 

And these method can be divided as:

- Classical Methods
  - Feature-based registration
  - Intensity-based registration
- Deep learning-based Methods
  - Supervised Learning
  - Unsupervised Learning

#### Classical Registration Methods

In the classical methods, a pre-defined *transformation model*, rigid or nonrigid, is iteratively *optimised* to minimize a *similarty measure* - a metric that quantifies how "similar" the warped moving image and the fixed image are.

Similarity measures can be designed to consider only important image features (extracted from a pre-processing step) or directly sample all intensity values from both images. As such, we can subdivide algorithms into two sub-types:

- **Feature-based registration**: Important features in images are used to calculate transformations between the dataset pairs. For example, point set registration - a type of features widely used in many applications - finds a transformation between point clouds. These types of transformations can be estimated using Iterative Closest Point (ICP) [7] or coherent point drift [8] (CPD), for rigid or nonrigid transformation, respectively.

  For example, the basis of ICP is to iteratively minimise the distance between the two point clouds by matching the points from one set to the closest points in the other set. The transformation can then be estimated from the found set of corresponding point pairs and repeating the process many times to update the correspondence and the transformation in an alternate fashion.

- **Intensity-based registration**: Typically, medical imaging data does not come in point cloud format, but rather, 2D, 3D, and 4D matrices with a range of intensity values at each pixel or voxel. As such, different measures can be used directly on the intensity distributions of the data to measure the similarity between the moving and fixed images. Examples of measures are cross-correlation, mutual information, and simple sum-square-difference - these intensity-based algorithms can optimize a transformation model directly using images without the feature extraction step

##### Iterative Closest Point (ICP)

Having two scans $P =\left\{{p_i}\right\}$ and $Q =\left\{{q_i}\right\}$, we want to find a transformation (rotation $R$ and translation $t$) to apply to $P$ to match $Q$ as good as possible. The total process can be divided as : 1) Data association;, 2) Transformation;

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024201720395.png?raw=True" >
</div>


###### Data association

We compute correspondences from $P$ to $Q$, i.e. for every $p_i$ we search the closest $q_j$ to it.

```python
def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences
```

 We can draw the computed correspondences， which is shown below.
 
<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024200730012.png?raw=True" >
</div>


###### Transformation

If the sets would match exactly, their cross-covariance would be identity. Therefore, we can iteratively optimize their cross-covariance to be as close as possible to an identity matrix by applying transformations to $P$.  Let's dive into details.

 **Single iteration**: In a single iteration we assume that the correspondences are known. We can compute the cross-covariance between the corresponding points. Let $C=\left\{\{i, j\}: p_{i} \leftrightarrow q_{j}\right\}$ be a set of all correspondences, also $|C|=N$. Then, the cross-covariance $K$ is computed as:
$$
\begin{aligned}
K &=E\left[\left(q_{i}-\mu_{Q}\right)\left(p_{i}-\mu_{P}\right)^{T}\right] \\
&=\frac{1}{N} \sum_{\{i, j\} \in C}\left(q_{i}-\mu_{Q}\right)\left(p_{i}-\mu_{P}\right)^{T} \\
& \sim \sum_{\{i, j\} \in C}\left(q_{i}-\mu_{Q}\right)\left(p_{i}-\mu_{P}\right)^{T}
\end{aligned}
$$
Each point has two dimentions, that is $p_{i}, q_{j} \in \mathbb{R}^{2}$, thus cross-covariance has the form of (we drop indices $i$ and $j$ for notation simplicity);
$$
K=\left[\begin{array}{ll}
\operatorname{cov}\left(p_{x}, q_{x}\right) & \operatorname{cov}\left(p_{x}, q_{y}\right) \\
\operatorname{cov}\left(p_{y}, q_{x}\right) & \operatorname{cov}\left(p_{y}, q_{y}\right)
\end{array}\right]
$$

---

**Intuition**: Intuitively, cross-covariance tells us how a coordinate of point $q$ changes with the change of $p$ coordinate, i.e. $\operatorname{cov}\left(p_{x}, q_{x}\right)$ tells us how the $x$ coordinate of $q$ will change with the change in $x$ coordinate of $p$ given that the points are corresponding. Ideal cross-covariance matrix is an identity matrix, i.e., we want the $x$ coordinates to be ideally correlated between the scans $P$ and $Q$, while there should be no correlation between the $x$ coordinate of points from $P$ to the $y$ coordinate of points in $Q$. In our case, however, the position of $P$ is derived from the position of $Q$ through some rotation $R$ and translation $t$. Therefore, whenever we would move the scan $Q$, scan $P$ would move in a related way, but perturbed through the rotation and translation applied, making the cross-covariance matrix non-identity.

---

Knowing the cross-covariance we can compute its SVD decomposition:
$$
\operatorname{SVD}(K)=U S V^{T}
$$
The SVD decomposition gives us how to rotate our data to align it with its prominent direction with $U V^{T}$ and how to scale it with its singular values $S$. Therefore:
$$
\begin{aligned}
R &=U V^{T} \\
t &=\mu_{Q}-R \mu_{P}
\end{aligned}
$$
We apply the computed $t$ to $P$, and plot the result.

 
<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024201659756.png?raw=True" >
</div>


**Make it iterative** If we would know the correct correspondences from the start, we would be able to get the optimal solution in a single iteration. This is rarely the case and we need to iterate. That consists of the following steps:

1. Make data centered by subtracting the mean
2. Find correspondences for each point in $P$
3. Perform a single iteration by computing the cross-covariance matrix and performing the SVD
4. Apply the found rotation to $P$
5. Repeat until correspondences don't change
6. Apply the found rotation to the mean vector of $P$ and uncenter $P$ with it.

The final result is shown below:

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024201946197.png?raw=True" >
</div>
#### Why DL?

Usually, it is challenging for classical methods to handle real-time registration of large feature sets or high dimensional image volumes owing to their computationally intense nature, especially in the case of 3D or high dimensional nonrigid registration. State-of-the-art classical methods that are implemented on GPU still struggle for real-time performance for many time-critical clinical applications.

Secondly, classical algorithms are inherently pairwise approaches that can not directly take into account population data statistics and relying on well-designed transformation models and valid similarity being available and robust, challenging for many real-world tasks.

In contrast, the computationally efficient inference and the ability to model complex, non-linear transformations of learning-based methods have motivated the development of neural networks that infer the optimal transformation from unseen data [1].

However, it is important to point out that

- Many deep-learning-based methods are still subject to the limitations discussed with classical methods, especially those that borrow transformation models and similarity measures directly from the classical algorithms;
- Deep learning models are limited at inference time by how the model was trained - it is well known that deep learning models can overfit to the training data;
- Deep learning models can be more computationally intensive to train than classical methods at inference;
- Classical algorithms have been refined for many clinical applications and still work well.

#### Deep Learning based method

In recent years, learning-based image registration has been reformulated as a machine learning problem, in which, many pairs of moving and fixed images are passed to a machine learning model (usually a neural network nowadays) to predict a transformation between a new pair of images.

In this note, we investigate three factors that determine a deep learning approach for image registration:

1. What type of network output is one trying to predict?
2. What type of image data is being registered? Are there any other data, such as segmentations, to support the registration?
3. Are the data paired? Are they labeled?

**Types of network output**:  We need to choose what type of network output we want to predict.

- **Predicting a dense displacement field**

  Given a pair of moving and fixed images, a registration network can be trained to output dense displacement field (DDF) [9] of the same shape as the moving image. Each value in the DDF can be considered as the placement of the corresponding pixel / voxel of the moving image. Therefore, the DDF defines a mapping from the moving image's coordinates to the fixed image.

  *In this note, we mainly focus on DDF-based methods.*

- **Predict a static velocity field**

  Another option is to predict a static dense velocity field (SVF or DVF) between a pair of images, such that a diffeomorphic DDF can be numerically integrated. We refer you to [9] and [15] for more details.

- **Predict an affine transformation**

  A more constrained option is to predict an affine transformation and parameterize the affine transformation matrix to 12 degrees of freedom. The DDF can then be computed to resample the moving images in fixed image space.

- **Predict a region of interest**

  Instead of outputting the transformation between coordinates, given moving image, fixed image, and a region of interest (ROI) in the moving image, the network can predict the ROI in fixed image directly. Interested readers are referred to the MICCAI 2019 paper [10].

**Data availability, level of supervision and network training strategies**:  Depending on the availability of the data labels, registration networks can be trained with different approaches. These will influence our loss choice. 

Therefore, the deep learning base method can be divided as :

##### Unsupervised 

In practice, it is almost impossible to correlate pixel-by-pixel annotation of two images that need to be aligned, therefore, the unsupervised training strategy is more common used ,which the algorithm is not provided with any pre-assigned labels or scores for the training data.  The optimization loss  often consists of the intensity-based loss and deformation loss.

The overall unsupervised registration framework are shown below:



![Unsupervised DDF-based registration network](https://camo.githubusercontent.com/0ce8ee68a55ff316c0946cd6e3ef9c82b28d6a3c/68747470733a2f2f6769746875622e636f6d2f446565705265674e65742f446565705265672f626c6f622f6d61696e2f646f63732f736f757263652f5f696d616765732f726567697374726174696f6e2d6464662d6e6e2d756e737570657276697365642e7376673f7261773d74727565)

In details, we take the mnist dataset as illustrated example, we want to register any pair (moving and fixed images)of number 5 from the below data.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211024210101462.png?raw=True">
</div>

###### Input

 Given two images ($P$, $Q$), out goal is to find the deformation between them we use a network that takes in two images (moving $P$ and fixed images $Q$) , and outputs a dense deformation $ϕ$ (e.g. size 32x32x2, because at each pixel we want a vector telling us where to go)

###### Network

Since the input images and the DDF should be same spatial size, encoder-decoder architecture is usually adopted.

```
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

###### Loss

Given that the displacement $\phi$ is output from the network, we need to figure out a loss to tell if it makes sense
In a supervised setting we would have ground truth deformations $\phi_{g t}$ and we could use a supervised loss like $M S E=\left\|\phi-\phi_{g t}\right\|$
The main idea in unsupervised registration is to use loss inspired by classical registration.Without supervision, how do we know this deformation is good?

- make sure that $m \circ \phi(m$ warped by $\phi$ ) is close to $f$
- regularize $\phi$ (often meaning make sure it's smooth)

To achieve first item, we need to warp input image $Q$. To do this, we use a spatial transformation network layer, which essentially does linear interpolation.

###### Visualization.

Let's visualize the results, we input the moving image and fixed image to the network, and obtain the predicted DDF after training. We warp the input moving image using the generated DDF, and get the warped image. We also provide the DDF in color and direction.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025095028241.png?raw=True">
</div>


##### Weakly supervised 

When an intensity-based loss is not appropriate for the image pair one would like to register, the training can take a pair of corresponding **moving and fixed labels** (in addition to the image pair), represented by binary masks, to compute a label dissimilarity (label based loss) to drive the registration. We provide a simple example of moving and fixed labels in the figure below.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025101319775.png?raw=True">
</div>

Combined with the regularisation on the predicted displacement field, this forms a weakly-supervised training. An illustration of a weakly-supervised DDF-based registration network is provided below.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025100714274.png?raw=True">
</div>

As shown in the figure, the input and network of Weakly supervised method is not different with the unsueprvised based model.

The only difference is the learned DDF is used to warp the moving label.  

For the loss item, provided labels for the input images, a label based loss may be used to measure the (dis)similarity of warped regions of interest. Having computed a transformation between images using the net, one of the labels is warped and compared to the ground truth image label. Labels are typically manually contoured organs.

The common loss function is Dice loss, Jacard and average cross-entropy over all voxels, which are measures of the overlap of the ROIs. For example, the Dice score between two sets, $X$ and $Y$, is defined like:
$$
\text { Dice }=\frac{2(X \cap Y)}{|X|+|Y|}
$$

##### Combined[¶](https://render.githubusercontent.com/view/ipynb?color_mode=auto&commit=9d98198f90b7011acdd3f80e96ab4b8c5ac8c9e8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f446565705265674e65742f446565705265672f396439383139386639306237303131616364643366383065393661623462386335616338633965382f646f63732f496e74726f5f746f5f4d65646963616c5f496d6167655f526567697374726174696f6e2e6970796e62&nwo=DeepRegNet%2FDeepReg&path=docs%2FIntro_to_Medical_Image_Registration.ipynb&repository_id=269365590&repository_type=Repository#Combined)

When the data label is available, combining intensity-based, label-based, and deformation based losses together has shown superior registration accuracy, compared to unsupervised and weakly supervised methods. Following is an illustration of a combined DDF-based registration network.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025102115632.png?raw=True">
</div>

## Protein Structure Prediction

### Introduction

Proteins are essential to life, supporting practically all its functions. They are large complex molecules, made up of chains of amino acids, and what a protein does largely depends on its unique 3D structure. First, let us understand how make genes to proteins to traits.

The DNA is a type of molecule that carries the inherited information passed from parents to offspring, and a gene is a segment (or part) of DNA that contains the instructions for building a product, usually a protein. The journey from gene to protein is complex and tightly controlled within each cell. It consists of two major steps: transcription and translation. Together, transcription and translation are known as gene expression.

During the process of transcription, the information stored in a gene's DNA is passed to a similar molecule called RNA (ribonucleic acid) in the cell nucleus. Both RNA and DNA are made up of a chain of building blocks called nucleotides, but they have slightly different chemical properties. The type of RNA that contains the information for making a protein is called messenger RNA (mRNA) because it carries the information, or message, from the DNA out of the nucleus into the cytoplasm.

Translation, the second step in getting from a gene to a protein, takes place in the cytoplasm. The mRNA interacts with a specialized complex called a ribosome, which "reads" the sequence of mRNA nucleotides. Each sequence of three nucleotides, called a codon, usually codes for one particular amino acid. (Amino acids are the building blocks of proteins.) A type of RNA called transfer RNA (tRNA) assembles the protein, one amino acid at a time. Protein assembly continues until the ribosome encounters a “stop” codon (a sequence of three nucleotides that does not code for an amino acid) [[2]](https://medlineplus.gov/genetics/understanding/howgeneswork/makingprotein/).

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025103629310.png?raw=True">
</div>


A protein’s shape is closely linked with its function, and the ability to predict this structure unlocks a greater understanding of what it does and how it works.  Many of the world’s greatest challenges, like developing treatments for diseases or finding enzymes that break down industrial waste, are fundamentally tied to proteins and the role they play. Figuring out what shapes proteins fold into is known as the protein folding problem, and has stood as a grand challenge in biology for the past 50 years. 

This has been a focus of intensive scientific research for many years, using a variety of experimental techniques to examine and determine protein structures, such as [nuclear magnetic resonance](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) and [X-ray crystallography](https://en.wikipedia.org/wiki/X-ray_crystallography). These techniques, as well as newer methods like [cryo-electron microscopy](https://en.wikipedia.org/wiki/Cryogenic_electron_microscopy), depend on extensive trial and error, which can [take years of painstaking and laborious work](https://www.youtube.com/watch?v=gLsC4wlrR2A) per structure, and require the use of multi-million dollar [specialised equipment](https://www.youtube.com/watch?v=WJKvDUo3KRk).



### Alpha Fold

#### Introduction

In a major scientific advance, the latest version of our AI system [AlphaFold](https://deepmind.com/research/case-studies/alphafold) has been recognized as a solution to this grand challenge by the organizers of the biennial Critical Assessment of protein Structure Prediction ([CASP](https://predictioncenter.org/)). This breakthrough demonstrates the impact AI can have on scientific discovery and its potential to dramatically accelerate progress in some of the most fundamental fields that explain and shape our world. 

#### Formulation

In detail, the input to Alphafold is an amino acid sequence, with each element representing one amino acid unit in the chain (there can be 21 amino acid units in total). For example, `PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGELASK`. And the output of AlphaFold is the predicted protein structure of the input amino acid sequence. The overall pipeline are figured below.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025112318555.png?raw=True">
</div>



We can divide the alpha into following parts:

- Sequence and Feature Exaction

- Deep neural Network

- Distance and Torsion Distribution Prediction

- Gradient descent on protein-specific potential

#### Sequence and Feature Exaction

Given an amino acid sequence, we first define and extract its features. Basiclly, we can encode the amino acid species features in one-hot format, Besides, HHblit  features (22 dimensions), MSA (Multiple Sequence Alignment)  features, etc. are also added as input, where HHblit features and MSA features  are extracted sequences from a large database that are similar to the input amino acid sequence and align them in passing. The reason for extracting this feature is that similar amino acid sequences are generally folded in a similar way, which is equivalent to adding similar sequence structure information can help prediction.

For example, searching the `PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGELASK` from the [HHblits](https://toolkit.tuebingen.mpg.de/jobs/4096655), we can get it's similarest amino acid sequences, and obtain their extracted features.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025114137219.png?raw=True">
</div>

#### Deep neural Network

The nerual network is consists of 220 residual convolution blocks. The details of block are illustrated in the figure below. 

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025115204662.png?raw=True">
</div>

#### Distance and Torsion Distribution Prediction

First, we need to know how AlphaFold present the 3D Protein structure,  instead of representing the protein structure in Cartersian space (x, y ,z),  AlphaFold represents 3D structures as a **pair of torsion angles between amino acids**. Even when a protein is in a folded state, the basic blocks, the amino acid structure remains unchanged. Yet, the torsion angle between one amino acid and the other changes. Besides, the AlphaFold also represent pairwise distance distribution between amino acids.

We can use the **pair of torsion angles and distances between amino acids** to represent a 3D  Protein structure.

##### Distance Distribution Prediction

AlphaFold predicts the **distance between amino acids** in a protein and the **probability distribution** of the predicted distance.

Let’s say that there are amino acids A, B, C, and D present in a protein.A distance map as below can be drawn:


<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025122448310.png?raw=True">
</div>

The distance between one amino acid with itself is zero, and the distance between A and B will be the same as that between B and A. You can draw a table like above by calculating the **distance between each amino acid pair**.  Furthermore, AlphaFold also predicts the **probability distribution of those distances**.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025122512990.png?raw=True">
</div>

Below figure is from the AlphaFold 1 [paper](https://www.nature.com/articles/s41586-019-1923-7). Here you can see that each pixel in the distance map represents a probability distribution. DeepMind calls this a distogram.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025122559803.png?raw=True">
</div>

##### Torsion Distribution Prediction

AlphaFold calculates the **probability distribution of the torsion angles** of the amino acids. 

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025122733667.png?raw=True">
</div>

One can say that AlphaFold 1 is a two-step process. In the first step, it receives data and amino acid sequence of a target protein and trains a CNN (with PDB dataset) to find the 1) **distogram of the protein** and 2) the **probability distribution of the torsion** angle.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025122823471.png?raw=True">
</div>

#### Protein-Specific Potential

 With the two outputs from step 1, AlphaFold 1 now starts the second step, where it tries to find the final folding structure. For this, it builds a **protein-specific potential** function of the protein folding structure. 


<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123031488.png?raw=True">
</div>

 First, the initially predicted protein structure is selected using the torsion angle distribution predicted by CNN.

- If this initial structure and the distogram estimated by the CNN are cross-produced, you can calculate how much the initially predicted protein structure fits the probability obtained from the distogram.
-  This potential is called the **distance potential**. The bigger the difference between the distance predicted in the distogram and the initial structure, the bigger the potential energy is in the proposed structure. 

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123139349.png?raw=True">
</div>

Similarly, if we cross-produce the initially predicted structure with the torsion distribution, it gives another potential function, which is the **geometric potential**. 

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123257816.png?raw=True">
</div>

Lastly, the algorithm also considers physical constraints. As mentioned above, the amino acid structure has a backbone structure and side chains. However, when AlphaFold 1 predicts the initial structure, it is done by using just the backbone structure, and whether a side chain exists or not is not considered. Thus, AlphaFold incorporates the **Van der Waals** **term to prevent steric clashes**, because residues do not bump into each other. 

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123342519.png?raw=True">
</div>

Therefore, AlphaFold calculates three potentials and sums them up in a single combined potential function, the **protein-specific potential**.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123405430.png?raw=True">
</div>

The last step is to go through iteration and find an optimal solution that minimizes the corresponding potential function. Alpha Fold 1 uses **gradient descent** **minimization** to obtain well-packed protein structures Thus, after repeated gradient descent process, the optimization converges and the lowest potential structure is stored as the best solution from the iteration as one of the expected answers.

<div align="center">
<img src="https://github.com/JiYuanFeng/applied_dl/blob/main/figures/image-20211025123510432.png?raw=True">
</div>

## Quiz
### Quiz 1 - Given two binary masks, compute dice score and dice loss
Descriptions: xx
### Quiz 2 - Give a fixed point set and moving point set , use ICP to compute the results in one iteraction.
Descriptions: xx
### Quiz 3 - Given two classifiers’ predictions (medical classification task), explain which classifier is better (compute their AUC-ROC curves).
Descriptions: xx
