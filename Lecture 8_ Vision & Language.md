# Lecture Note 8: Vision & Language

## 0. Introduction
Visual recognition and natural language understanding are two challenging tasks in artificial intelligence. In this section, we will focus on the skills to build deep learning models that can reason about images and text. To be specific, we will introduce background knowledge of recurrent neural networks and a representative task in Vision&Language, i.e., image captioning.

## 1. Recurrent Neural Networks (RNNs)

To understand a passage in natural language, humans don’t start their thinking from scratch every second but understand each word based on the previous words.  For example, if you see the phrase:

**"A man is sitting on a ?"**

it's easy to infer that ? is probably a word related to 'chairs' or 'bench' instead of something else like 'monkey' or 'cloud'. The words in a sentence are highly correlated and the temporal dependency of historical words is essential for human to understand the whole sentence.

Recurrent neural networks are a set of neural networks which could persist the historical information among the sequence,  allowing a global understanding of the whole sequence. Before the emergence of Transformer, RNN dominated the natural language processing tasks for many years due to their lightweight architecture and powerful representation ability for temporal dependency modeling.

We list the notations in this lecture for better understanding. 
| Notation                           | Name                         |
| ---------------------------------- | ---------------------------- |
| $D$                                  | size of hidden units in RNNs  |
| $C$                                  | size of input in RNNs |
| $C_I$                                  | dimension of image features |
| $T$                                  | length of sentence |
| $X=\{x_t\}_{t=1}^{T}$              | a sentence |
| $x_t\in \mathbb{R}^{C}$            | $t_{th}$ word       |
| $f_t\in \mathbb{R}^{D}$            | forget gate of LSTM     |
| $i_t\in \mathbb{R}^{D}$            | input gate of LSTM       |
| $o_t\in \mathbb{R}^{D}$            | output gate of LSTM      |
| $\tilde{C}_t \in \mathbb{R}^{D}$ |  candidate memory state of LSTM     |
| $C_t \in \mathbb{R}^{D}$ |  cell state of LSTM     |
| $h_t \in \mathbb{R}^{D}$ | hidden state of LSTM        |
| $W_{**}$ | matrix with learnable weights|
|  $b_{*}$ | vector with learnable weights|
|  $\sigma(\cdot)$ | Sigmoid function|
|  $I=\{r_i, ..., r_N\}$ | Image|
|  $r_i \in \mathbb{R}^{C_I}$ | regional features|

#### Why recurrent neural networks?

*FFNs for sequence modeling:*
![](https://i.imgur.com/Q1GR5sI.png)


*Vanilla RNN:*

![image alt](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png "An unrolled recurrent neural network.")

To mode a sequence $\{x_0, x_1, ..., x_T\}$ where $x_t\in \mathbb{R}^C$ (e.g., a sentence), a simple solution is to pass each element into feed-forward neural networks (FFNs). However, the outputs are independent of others, missing the relationships between each other. For example, $h_1$ can not see the content of $x_0$. 

Vanilla recurrent nerual networks extends FFNs by adding a temporal link between the $t_{th}$ input  and previous outputs $h_{t-1}$. The update rule of the output hidden state $h_t$:
$$h_t = f(x_t, h_{t-1}) = {\rm tanh}(W_x x_t + W_h h_{t-1} + b),$$
where $W_x\in \mathbb{R}^{D\times C}, W_h\in \mathbb{R}^{D\times D},b \in \mathbb{R}^{D}$ are learnable parameters. The activation function is usually set to Tanh. The $t_{th}$ hidden state carries the historical information and thus can be writeen as a function depentent on previous elements $\{x_0,...,x_t\}$:

$$h_t=f(x_t, ..., f(x_2, f(x_1, f(x_0, h_{init}))))=G(x_0,x_1, ...,x_t, h_{init})$$
where $h_{init}$ represents initial hidden state to calcuate the $h_0$, and we have $h_0=f(x_0, h_{init})$.


### Long-Short Term Memory (LSTM)
One of the appeals of vanilla RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, if we want to predict the last word of the paragraph "I am a 13-year-old student..., I speak fluent ?". It seems the context within the last four words is enough to reason that '?' is a word indicating the name of a language. However, when facing a new paragraph like “I grew up in France ..., I speak fluent ?”, inferring the correct answer "?=French" requires more context from further back. 

Unfortunately, vanilla RNNs gradually lost the connection between historical words and current words with the increasing sequence length. This phenomenon is mainly caused by the **gradient exploding/vanishing** during training. LSTM is a variant of vanilla RNN and it could alleviate this problem.


Illustration of long-shot term memory:
![](https://i.imgur.com/HWfXJut.gif)

The key to LSTMs is the cell state $C_t$. The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.


The updating rules for $C_t$ and $h_t$ are as follows:
$$f_t = \sigma(W_{fh} h_{t-1} + W_{fx} x_t + b_f)$$

$$i_t = \sigma(W_{ih} h_{t-1} + W_{ix} x_t + b_i)$$

$$o_t = \sigma(W_{oh} h_{t-1} + W_{ox} x_t + b_o)$$

$$\tilde{C}_t=tanh(W_C h_{t-1} + W_{Cx} x_t + b_C )$$

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

$$h_t = o_t * tanh(C_t)$$

Forget gate $f_t$ and input gate $i_t$ modulate the cell state at each time step. The gate is a vector made by a sigmoid layer, with each value between 0 and 1. The value 1 means "completely preserve this" while 0 means "completely drop this". Besides, the output gate $o_t$ controls what we are going to output by filtering the cell state.



### Reference
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## 2. Image Captioning (IC)
The goal of image captioning is to generate a natural sentence to describe the image. Different from traditional computer vision tasks like image classification and object detection which only predict one or some object labels, IC could generate a readable sentence with variable length to describe more details, including objects, actions, scenes, and their relationships. 

![](https://i.imgur.com/J5kupZC.jpg)

The challenges to write accurate captions for an image is two-fold: you need to get meaning out of the image by extracting relevant features, and you need to translate these features into a human-readable language.

![Picture1.png](https://s2.loli.net/2022/03/03/qQURafF19giNp63.png)

### Image Representation
We first extract the image features of the penultimate layer by an off-the-shelf visual backbone (e.g, VGG, ResNet, ViT), the resultant feature map contains $N$ vectors, each of which is a $C_I$-dimensional representation corresponding to a region of the image:
$$I=\{r_1, r_2, ..., r_N\}, r_i \in \mathbb{R}^{C_I}$$

### Soft Attention Mechanism: 



One of the most curious facets of the human visual system is the presence of attention. Rather than compress an entire image into a static representation, attention allows for salient features to dynamically come to the forefront as needed. This is especially important to deal with clutter in real-world images. 

When generating the $t_{th}$ word by RNNs, the context vector $z_t$ is a dynamic representation of the relevant part of the image input at time $t$. For each location i, the mechanism generates a positive weight $\alpha_{it}$ which can be interpreted either as the probability that location $i$ is the right place to focus for producing the next word:

$$\alpha_{ti} = w^T \tanh(W_{\alpha h} h_{t-1} + W_{\alpha r} r_{i} + b_{\alpha}),$$

where the vector $w, b_{\alpha}$ and matrix $W_{\alpha h},W_{\alpha r}$ are learnable. Then a softmax function is used to normalize $\sum_{i} a_{ti}=1$.
$$a_{ti} = softmax(\alpha_{ti})=\frac{exp(\alpha_{ti})}{\sum_{j} exp(\alpha_{tj})}$$

$$z_t = \Sigma_i a_{ti} r_i,$$


where $a_i$ represents the attention weight corresponding to region $r_i$.

The initial cell state $c_{init}$ and memory state $h_{init}$ of the LSTM are predicted by an average of the annotation vectors fed.

$$c_{init}=f_{init,c}(\frac{1}{N}\sum_i {r_i})$$

$$h_{init}=f_{init,h}(\frac{1}{N}\sum_i {r_i})$$




## 3. Quiz
### 1. Implement the inference code of an RNN model
```python
class VanillaRNN():
    def __init__(self, C, D):
        '''
        initialize a RNN module
        :param C: input size
        :param D: hidden size
        '''

        self.C = C
        self.D = D
        self.Wx = np.random.rand(D, C)
        self.Wh = np.random.rand(D, D)
        self.b = np.random.rand(D)

    def init_h(self, B):
        '''
        :param B: batch size
        :return: zero-initialized hidden state
        '''
        h0 = np.zeros((B, self.D))
        return h0

    def forward_one_step(self, x, prev_h):
        out = x @ self.Wx.T + prev_h @ self.Wh.T + self.b
        out = np.tanh(out)
        return out

    def forward(self, x):
        '''
        :param x: input sequence with shape [B, T, C]
        :return: a sequence of hidden states with shape [B, T, D]
        '''
        h_list = []
        B, T, C = x.shape
        assert self.C == C, 'error size of input feature'
        h = self.init_h(B)
        for t in range(T):
            h = self.forward_one_step(x[:,t], h)
            h_list.append(h)
        return h_list
```
### 2. Parameter Number and FLOPs calculation of RNNs.

$n$: number of feed-forward neural network (FFN) (Vanilla RNN has 1, LSTM has 4)
$C$: size of input
$D$: size of hidden unit
$T$: length of input sequence
Input shape: (B, T, C)

Take the update step for vanilla RNN as example, it's actually an FFN:
$$H_t = {\rm tanh}(XW_x + H_{t-1}W_h + b)$$ $$(B, C) \times (C, D) + (B, D) \times (D, D) + D \rightarrow (B, D) \overset{tanh}{\rightarrow}(B, D)$$

**Parameter Number of an FFN:** $CD+D^2+D$
**Parameter Number of RNNs** are the summation of all FFNs: 
$$n(CD+D^2+D)$$

**FLOPs of matrix product:** The matrix product operation $XW_x$: $(B, C)\times(C, D) \rightarrow (B, D)$ requires $BDC$ multiplication and $BD(C-1)$ addition:
$$BDC + BD(C-1) = BD(2C-1)$$

**FLOPs of a feed-forward layer** contain two matrix product operations and two matrix addition, which has FLOPs of: 

$$BD(2C-1) + BD(2D-1) + 2BD=BD(2C+2D)$$

**FLOPs of RNNs** are the summation of all feed-forward layers in all timestamps: 
$$nT\times BD(2 D+2 C )$$

### 3. Implement the soft attention mechanism.
- **Step 1**: Generate the importance $\alpha_{ti}$ of each region $r_i$ in the image conditioned on $h_{t-1}$:
$$\alpha_{ti} = w^T f(W_r r_i + W_h h_{t-1}+b)$$
- **Step 2**: Nomailize the importance values into attention scores:
$$a_{ti} = softmax(\alpha_{ti})=\frac{exp(\alpha_{ti})}{\sum_{j} exp(\alpha_{tj})}$$
- **Step 3**: Aggreate the all the region features by attention scores:
$$z_t = \Sigma_i a_{ti} r_{i},$$

We provide a simple NumPy implementation of the soft attention mechanism:
```python
def soft_attention(img, prev_h, W, w, b):
    '''
    # soft attention at one step
    img: regional features, shape [B, N, C_I]
    prev_h: previous hidden state at time (t-1), shape [B, D]
    W: learnable weights, shape [D+C_I, C_hidden]
    b: bias, shape [C_hidden]
    w: learnable vector, shape [C_hidden, 1]
    '''
    B, N, C_I= img.shape
    prev_h_ext = np.expand_dims(prev_h, 1).repeat(N, axis=1) # [B, N, D]
    input = np.concatenate([img, prev_h_ext], axis=-1) # [B, N, D+C_I]
    alpha = (input@W +  b) @ w #[B, N, 1]
    # softmax for normalization
    a = np.exp(alpha - np.min(alpha)) / (np.sum(np.exp(alpha-np.min(alpha)), axis=1, keepdims=True) + 1e-8) # [B, N, 1]
    # aggregation by attention scores
    output = np.sum(img * a, axis=1) # [B, C_I]
    return output
```