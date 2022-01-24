# Lecture 5: Video Understanding

## Introduction
In this section we will introduce the Action Recognition (also known as Video Classfication) problem. This is the most fundamental topic in video understanding, which serves as the basis for various video related problems.

![](https://i.imgur.com/asGlnSA.jpg)



### From Image to Video

![](https://i.imgur.com/GTkvnX9.png)

A video is a sequence of images (called frames / snippets) captured and eventually displayed at a given frequency. **Image classification** aims to recognize the objects appearing in the image. And the goal of **action recognition** is to recognize actions of one or more agents occuring in the entire video. Compared with image classifiction, it is a more challenging task because:

- **Videos are big.** A video often plays at the rate of 30 frames per second (fps). Since a 3 color channels (RGB) pixel takes up 3 bytes, a one-minute video with small resolution (640 x 480) would still occupy ~1.5GB disk space. It is impossible for the network to process all the frames of one video at a time.
- **Video contains rich temporal information.** A naive solution to solve the action recognition problem is obtaining a single frame of the video, and then applying the image classification algorithm on it. However, a video not only contains rich appearance context but also crutial temporal dynamics. For example, to distinguish the actions of "high jump" and "long jump", it is not enough to just watch the begining of the video. Therefore, how to effectively model the spatio-temporal representations and capture long-range dependencies is the key to action recognition.

## Notions



## Temporal Segment Network

![](https://i.imgur.com/S51frWj.png)

### Motivation

Although the frames are densely recorded in the videos, the content changes relatively slowly, therefore, it is no need to utilize all the frames to train the network. How to design an effective sampling strategy remains an opening question. (1) Early approaches are designed to operate on a single frame or a stack of frames (e.g., 16 frames) with limited temporal duration, termed as **local sampling strategy**. These methods lack capacity of incorporating long-range temporal information of videos into the learning of action models. (2) Some previous approaches adopt the **dense sampling strategy** which samples more frames at a fixed rate. These methods suffer in huge computational cost especially when the video gets longer.

### Overview

Temporal Segment Network (TSN) is an effective and efficient video-level framework. It proposes a new **sparse and global sampling strategy**: segment-based sampling. Specifically, the network first divides the video into several segments of equal duration, and then one snippet is randomly sampled from its corresponding segment. Each snippet predicts the snippet-level action class, and an function is designed to aggregate the snippet-level predictions into video-level scores. 

Formally, given a video $V$, the network divides it into $K$ segments $\left \{ S_{1}, S_{2}, ..., S_{K}  \right \}$ of equal durations. One snippet $T_{K}$ is randomly selected from the corresponding segment $S_{K}$. Then, the network models the sequece of snipptets $\left ( T_{1}, T_{2}, ..., T_{K} \right )$ as follows:

\begin{split}
    TSN(T_{1}, T_{2}, ..., T_{K} ) = \mathcal{H}\left ( \mathcal{G}\left     ( \mathcal{F}(T_{1}; \textbf{W}), \mathcal{F}(T_{2};\textbf{W}),         ..., \mathcal{F}(T_{K};\textbf{W})  \right ) \right )
\end{split}

where $\mathcal{F}(T_{K};\textbf{W})$ is the function representing the ConvNet with the parameters $\textbf{W}$ operates on te snippet $T_{K}$ and produces the class scores. $\mathcal{G}$ is the consensus function which combines the predictions of all snipptets to obtain a consensus of class scores. Based on the consensus, the prediction function $\mathcal{H}$ (e.g., Softmax function) predicts the probability distribution of each action class.

The final training loss is a standard categorical cross-entropy loss. Denoted the segmental consensus as $\textbf{G} = \mathcal{G}\left     ( \mathcal{F}(T_{1}; \textbf{W}), \mathcal{F}(T_{2};\textbf{W}),         ..., \mathcal{F}(T_{K};\textbf{W})  \right )$, the final loss is formed as:

\begin{split}
    \mathcal{L}(y, \textbf{G}) = -\sum_{i=1}^{C}y_{i}\left ( g_{i} -         log\sum_{i=1}^{C}\exp{g_{j}} \right )
\end{split}


### Segment-based Sampling

**Formulation.** Given total frame number `num_frames` of a video and the expected segment number `num_clips`. Suppose a random frame is selected from each segment, then return the frame indices. 

Here we need to consider all the possible situations:
- The video has only one frame. Then, we always choose the first frame for `num_clips` times.
- The video frame number is larger than segment number. In this case,  we first get the start indices of each segment, and then calculate the random offsets for each segment.
- The video frame number is smaller than segment number. Then, we do a linear interpolation between [0, num_frames] for `num_clips` values. 

```python
def _get_train_frames(self, num_frames, num_clips):
    """Get frame indices.
    It will calculate the average interval for selected frames,
    and randomly shift them within offsets between [0, avg_interval].
    If the total number of frames is smaller than clips num or origin
    frames length, it will return all zero indices.
    Args:
        num_frames (int): Total number of frame in the video.
        num_clips (int): Total number.
    Returns:
        np.ndarray: Sampled frame indices in train mode.
    """
    
    if num_frames == 1:
        frame_indices = np.zeros((num_clips,), dtype=np.int)
        
    avg_interval = num_frmes // num_clips

    if avg_interval > 0:
        base_offsets = np.arange(self.num_clips) * avg_interval
        clip_offsets = np.random.randint(avg_interval, size=num_clips)
        frame_indices = base_offsets + clip_offsets
    elif avg_interval == 0:
        ratio = num_frames / num_clips
        frame_indices = np.around(np.arange(num_clips) * ratio)

    return frame_indices
```

### Segment Aggregation Function

There are five aggregation functions:
- Max pooling
- Average pooling
- Top-K pooling
- Linear weigting
- Attention weighting

**Max pooling.** $g_{i} = max_{k\in \left \{ 1,2,...,k \right \}}f_{i}^{k}$, where $f_{i}^{k}$ is the $i$-th element of $\textbf{F}^{K} =  \mathcal{F}(T_{K};\textbf{W})$. The basis idea of max pooling is to seek a single and most discriminative snippet for each action class and utilize this strongest activation as the video-level response.

**Average pooling.** $g_{i} = \frac{1}{K}\sum_{k=1}^{K}f_{i}^{k}$. The basic intuition behind average pooling is to leverage the responses of all snippets for action recognition, and use their mean activation as the video-level prediction.

**Top-K pooling.** Top-K pooling strikes a balance between the max pooling and average pooling, which first selects $K$ most discriminative snippets for each action category and then perform average pooling.

**Linear weighting.** $g_{i} = \sum_{k=1}^{K}\omega_{k}f_{i}^{k}$, where $\omega_{k}$ is the weight for the $k$-th snippet. The basic assumption underlying this consensus function is that action can be decomposed into several phases and differnent phases may play different roles in recognizing action classes.

**Attention weighting.** Linear weight is data independent, thus lacking the capacity of considering the difference between videos. Attention weighting is formed as $g_{i} = \sum_{k=1}^{K}\mathcal{A}(T_{k})f_{i}^{k}$, where $\mathcal{A}(T_{k})$ is the attention weight for snippet $T_{k}$ and calculated according to video content adaptively. Specifically, the visual feature $\textbf{R} = \mathcal{R}(T_{k})$ is first extracted from each snippet with the same ConvNet and then produce attention weights as:

\begin{split}
    e_{k} = \omega^{att}\mathcal{R}(T_{k}), \\
    \mathcal{A}(T_{k}) = \frac{\exp(e_{k})}{\sum_{l=1}^{K}\exp(e_{l})},
\end{split}

### Input Modalities Analysis

Compared with static images, the additinal temporal dimension of videos delivers an important cue for action understanding, namely motion. In addition to original input modalities of RGB frames and optical flow, TSN also investigates two other modalities (RGB differences and warped optical flow) in terms of speed and accuracy.

![](https://i.imgur.com/PfAIO48.png)

(1) RGB and optical flow, the basic combination in the two-strem ConvNets, yields strong performance of 94.9\%. The warped optical flow slightly increases the performance (94.9\% to 95.0\%) but severely slows down the speed to 5 FPS. (2) Using combination of RGB and RGB differences under TSN can provide competitive recognition performance (91.0\%) while running at a very fast speed at 340 FPS, which could serve well for building real-time action recognition systems with moderate accuracy requirement.



## Two-stream Inflated 3D ConvNet

### Motivation

Due to the high-dimensionality of the 3D ConvNets' parameterization and the lack of labeled video data, previous 3D ConvNets have been relatively shallow (up to 8 layers). Benefiting from the large-scale ImageNet dataset, there have been a lot of very deep 2D architectures for images. Is it possible to inflate the 2D ConvNet and reuse their initialization parameters for videos? Two-stream Inflated 3D ConvNet (I3D) gives the answer. It is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-tempoal features from video while leveraging successful ImageNet architecture designs and their parameters.

### Inflating 2D ConvNet into 3D ConvNet

Starting with a 2D architecture, and inflating all the filters and pooling kernels - endowing them with an additinal temporal dimension. Filters are typically square and here they are made to be cubic - $N\times N$ filters become $N \times N \times N$.

![](https://i.imgur.com/eEgoADX.png)

### Bootstrapping 3D Filters from 2D Filters

One may want to bootstrap parameters from the pretrained ImageNet models. The 3D models can be implicitly pretrained on ImgeNet by satisfying the condition (called **boring-video fixed point**): the pooled activations on a *boring* video should be the same as on the original single image input. This can be done by repeating the weights of the 2D filters $K_{t}$ times along the time dimension, and rescaling them by dividing by $K_{t}$.

![](https://i.imgur.com/DmROZfw.png)

**Sample code.** Parameters initialization in MMAction2.

```python3
@staticmethod
def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                         inflated_param_names):
    """Inflate a conv module from 2d to 3d.
    Args:
        conv3d (nn.Module): The destination conv3d module.
        state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
        module_name_2d (str): The name of corresponding conv module in the
            2d model.
        inflated_param_names (list[str]): List of parameters that have been
            inflated.
    """
    weight_2d_name = module_name_2d + '.weight'

    conv2d_weight = state_dict_2d[weight_2d_name]
    kernel_t = conv3d.weight.data.shape[2]
    
    # NOTE here: copy Kt times and divide by Kt
    new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
        conv3d.weight) / kernel_t
    conv3d.weight.data.copy_(new_weight)
    inflated_param_names.append(weight_2d_name)

    if getattr(conv3d, 'bias') is not None:
        bias_2d_name = module_name_2d + '.bias'
        conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
        inflated_param_names.append(bias_2d_name)
```

### Pacing Receptive Field Growth in Spcae and Time

3D convolutions applies a 3 dimentional filter to the 4-dimension tensor $(C \times T \times H \times W)$ and the filter moves 3-direction $(T, H, W)$ to calcuate the low level feature representations. Their output shape is a volume space such as cube or cuboid. 

![](https://i.imgur.com/H6qGvyg.gif =50%x)

Virtually all image models treat the two spatial dimensions (horiontal and vertical) equally - pooling kernels and strides are the same. This quite natural and means that features deeper in the networks are equally affected by image locations increasingly far away in both dimensions. Here, we show an example of late fusion for 2D ConvNet. Its receptive field grows slowly in space. For time dimension, it builds all-at-once at the end.

The shape of the input tensor is $3 \times 20 \times 64 \times 64$. It first passes a 2D convolutional layer with kernel size as $3 \times 3$, hence the receptive field becomes $1 \times 3 \times 3$. Then, a $4 \times 4$ 2D pooling layer is applied, its spatial receptive field keep getting larger to $1 \times 6 \times 6$. After that, it goes through a $3 \times 3$ 2D convolutional layer. For the center element in the below figure, we can calculate it covers 14 elements of input (i.e., from the 3-rd to 16-th element), which indicates the receptive field is $1 \times 14 \times 14$. In the end, the Global Average Pooling layer make the final output able to see all the elements, thus the final receptive field is $20 \times 64 \times 64$. The long-range temporal structure is only modeled in the last GlobalAvgPool layer.

![](https://i.imgur.com/k1iIeRd.png)

The 3D convolutional layer builds the receptive field growing both in space and time. And an optimal receptive field often depends on frame rate and image resolutions.


<!-- ### Two 3D Streams

While a 3D ConvNet should ale to learn motion features from RGB inputs directly, it still performs pure feedforward computation, whereas optical flow algorithms are in some sense recurrent. It is valuable to have a two-stream configuration - the I3D networks are trained on RGB and optical flow inputs separately and their predctions are averaged at test time. As the figure shown below, the accuracy of the network trained with two-stream inputs would be ~3\% higher than that with only RGB stream (71.1\% to 74.2\%).

![](https://i.imgur.com/DHnTEnh.png) -->




## Non-local Neural Networks

![](https://i.imgur.com/u6LJWMD.png)

### Motivation

Capturing long-range dependencies is of central important for video understanding. Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. The non-local operation computes the response at a position as a weighted sum of the features at all positions. As the figure shown above, a position $\textbf{x}_{i}$'s response is computed by the weighted average of the features of all positions $\textbf{x}_{j}$ (only the highest weighted ones are plotted). The non-local block can be plugged into various vision architectures.

### Formulation

A generic non-local operation in deep neural networks is defined as:

$$\begin{equation}
    \textbf{y}_{i} = \frac{1}{\mathcal{C}(\textbf{x})}\sum_{\forall       j}^{}f\left ( \textbf{x}_{i}, \textbf{x}_{j} \right                   )g(\textbf{x}_{j})
\end{equation} \tag{1}$$

here $i$ is the index of an output position (in space, time or spacetime) whose response is to be computed and $j$ is the index that enumerates all possible positions. $\textbf{x}$ is the input features and $\textbf{y}$ is the output signal of the same size as $\textbf{x}$. $f$ is a pairwise function that computes a scalar (representing relationship or affinity) between $i$ and all $j$. The unary function $g$ computes a representation of the input signal at the position $j$. The response is normalized by a factor $\mathcal{C}(\textbf{x})$.

### Pairwise Function

Next, we describe several versions of $f$ and $g$. For simplicity, $g$ is considered in the form of a linear embedding: $g(\textbf{x}_{j}) = W_{g}\textbf{x}_{j}$, where $W_{g}$ is a learned weight matrix. This can be implemented as, e.g., 1 x 1 x 1 3D convolution in spacetime. For pairwise function $f$, there are 4 versions:

- Gaussian
- Embedded Gaussian
- Dot product
- Concatenation

**Gaussian.** A natural choice of $f$ is Gaussian function, it has the from of:

\begin{split}
    f(\textbf{x}_{i}, \textbf{x}_{j}) =                 
    e^{\textbf{x}_{i}^{T}\textbf{x}_{j}}
\end{split}

Here $\textbf{x}_{i}^{T}\textbf{x}_{j}$ is dot-product similarity. The normalization factor is set as $\mathcal{C}(\textbf{x})= \sum_{\forall j}f(\textbf{x}_{i}, \textbf{x}_{j})$.

**Embedded Gaussian.** A simple extension of the Gaussian function is to compute similarity in an embedding space. It is defined as:

\begin{split}
    f(\textbf{x}_{i}, \textbf{x}_{j}) = e^{\theta         
    (\textbf{x}_{i}^{T})\phi (\textbf{x}_{j})}
\end{split}

Here $\theta (\textbf{x}_{i}) = W_{\theta}\textbf{x}_{i}$ and $\phi (\textbf{x}_{j}) = W_{\phi}\textbf{x}_{j}$ are two embeddings. As above, the normalization factor is set as $\mathcal{C}(\textbf{x})= \sum_{\forall j}f(\textbf{x}_{i}, \textbf{x}_{j})$. Note that *self-attention module* is a special case of non-local operations in the *embedded Gaussian* version. This can be seen from the fact that a give $i$, $\frac{1}{\mathcal{C}(\textbf{x})}f(\textbf{x}_{i}, \textbf{x}_{j})$ becomes the *softmax* computation along the dimension $j$.

**Dot product.** The form of dot product is written as:

\begin{split}
    f(\textbf{x}_{i}, \textbf{x}_{j}) = \theta         
    (\textbf{x}_{i}^{T})\phi (\textbf{x}_{j})
\end{split}

In this case, the normalization factor is set as $\mathcal{C}(\textbf{x})=N$, where $N$ is the number os positions in $\textbf{x}$, rather than the sum of $f$, because it simplifies gradient computation. The main difference between the dot product and embedded Gaussian version is the presence of softmax, which plays the role of an activatiion function.

**Concatenation.** The concatenation form of $f$ is:

\begin{split}
    f(\textbf{x}_{i}, \textbf{x}_{j}) = \text{ReLU}\left ( 
    \textbf{w}_{f}^{T}\left [ \theta(\textbf{x}_{i}), 
    \phi(\textbf{x}_{j}) \right ] \right )
\end{split}

Here $\left [ \cdot , \cdot  \right ]$ denotes concatenation and $\textbf{w}_{f}$ is a learned weight vector that projects the concatenated vector to a scalar. In this form, the normalization factor is $\mathcal{C}(\textbf{x})=N$.



### Non-local Block

The non-local operation in Eq.(1) can be wrapped into a non-local block that can be incorporated into many existing architectures. The non-local block is defined as:

$$\begin{equation}
    \textbf{z}_{i} = W_{z}\textbf{y}_{i} + \textbf{x}_{i}
\end{equation} \tag{2}$$

where $\textbf{y}_{i}$ is given in Eq.(1) and "$+\textbf{x}_{i}$" means a residual connection. An example of non-local block (embedded Gaussian version) is illustrated in the figure below. The feature map is shown as a tensor with the shape of $T \times H \times W \times 1024$, where 1024 denotes the channel number. $\otimes$ represents matrix multiplicatioin and $\oplus$ denotes element-wise sum. The softmax operation is performed on each row. The blue boxes denote 1 x 1 x 1 convolutions.

![](https://i.imgur.com/b6LLkkt.png)





## Quiz

### 1. Implementation of segment-based sampling for clips.

Based on the segment-based sampling, in this time, you need to obtain consecutive `clip_len` frames for each segment. Please complete the code below.

```python
def _get_train_clips(self, num_frames, num_clips, clip_len):
    """Get clip offsets in train mode.
    It will calculate the average interval for selected frames,
    and randomly shift them within offsets between [0, avg_interval].
    If the total number of frames is smaller than clips num or origin
    frames length, it will return all zero indices.
    Args:
        num_frames (int): Total number of frame in the video.
        num_clips (int): Total clip number.
        clip_len (int): Frame number of a clip.
        frame_interval (int): Frame interval between two frames.
    Returns:
        np.ndarray: Sampled frame indices in train mode.
    """
    
    # TODO: complete the sampling code here
    ori_clip_len = clip_len * frame_interval
    avg_interval = (num_frames - ori_clip_len + 1) // num_clips

    if avg_interval > 0:
        base_offsets = np.arange(num_clips) * avg_interval
        clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=num_clips)
    elif num_frames > max(num_clips, ori_clip_len):
        clip_offsets = np.sort(np.random.randint(
                    num_frames - ori_clip_len + 1, size=num_clips))
    elif avg_interval == 0:
        ratio = (num_frames - ori_clip_len + 1.0) / num_clips
        clip_offsets = np.around(np.arange(self.num_clips) * ratio)
    else:
        clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

    return clip_offsets
```

### 2. Computation of 3D convolution receptive field.

![](https://i.imgur.com/qTzVJXM.png)

**Answer**
![](https://i.imgur.com/xsOgJYt.png)







### 3. Implementation of non-local block.

**Sample code.** We provide the sample code for a 3D non-local block of *Gaussian* version. Please complete other version functions and the feedforward procedure. (Note: complete the TODO part)

```python
import torch
from torch import nn
from torch.nn import functional as F

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension == 3
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # define conv, pool, bn layer
        conv_layer = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn_layer = nn.BatchNorm3d
        

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn:
            self.W_z = nn.Sequential(
                    conv_layer(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn_layer(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_layer(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)
        
        # TODO: complete the definition code here
        # define the needed theta and phi for all operations,
        # "gaussian" version does not have learned parameters
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_layer(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
        
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; 
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        
        # TODO: complete the pairwise functions here
        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
            
        
        # normalization factor
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
 
        y = torch.matmul(f_div_C, g_x)
        
        # TODO: complete the residual connection of non-local block here
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z
```

## Reference
- Paper
1. https://arxiv.org/abs/1705.02953
2. https://arxiv.org/abs/1705.07750
3. https://arxiv.org/abs/1711.07971

- Slides
1. https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture18.pdf 
2. http://cs231n.stanford.edu/slides/2021/discussion_5_videos.pdf

- Lecture
1. https://www.youtube.com/watch?v=A9D6NXBJdwU&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=18 
- Code
1. https://github.com/open-mmlab/mmaction2
2. https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py



