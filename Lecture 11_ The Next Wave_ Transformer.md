###### tags: `COMP3340 Applied DL`

# Lecture 11: The Next Wave: Transformer 
## Introduction

In this section, we will introduce the Transformer - an architecture that utilizes attention mechanism for modeling. The Transformer was first proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Although the Transformer model was initially proposed to solve the problem in NLP (Natural Language Processing), it has also shown its advantages in CV (Computer vision) to process the visual data (e.g., images and videos).

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/transformer_overview.png?raw=true" style="zoom:80%">
</div>

> Refenrece https://jalammar.github.io/illustrated-transformer/

### Motivation (why Transformers?)

- **Parallelization.** Conventional recurrent models typically factor computation along the symbol positions of the input and output sequences.  Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t−1}$ and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. The Transformer allows for significantly more parallelization due to its unique architecture.

- **Attention Mechanism.**  Attention mechanisms let a model draw from the state at any preceding point along the sequence. The attention layer can access all previous states and weighs them according to a learned measure of relevancy, providing relevant information about far-away tokens. Compared to convolutional operation, attention mechanism is able to provide the connection weights that are dynamically predicted according to each image instance.

- **Global Information Modeling.** The nice thing about Transformer is that we can model the global dependencies based on the attention mechanism. Compared to CNNS that use the neighbor context as inductive bias, Transformer-based utilized less inductive bias. The characteristics enable Transformer-based models to achieve better performances when trainied on the large-scale dataset.

## Transformer Basis

### Attention Mechanism （self-attention mechanism）
In fact, there are numerous attention mechanisms. In this section, we will mainly elaborate the classic qkv attention mechanism introduced in [Attention is All You Need](https://arxiv.org/abs/1706.03762).


#### Attention Input Parameters — Query, Key, and Value
The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value. Let’s first look at how to calculate self-attention using vectors, then proceed to look at how it’s actually implemented – using matrices. 

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/qkv_vector.png?raw=true" style="zoom:70%">
</div>

**The first step** in calculating self-attention is to create three vectors (i.e., the Query vector, the Key vector, and the Value vector) from each of the encoder’s input vectors (in this case, the embedding of each word). As introduced in this figure, we multiply $x_1$ by the $W^Q$ weight matrix produces $q_1$, multiply $x_1$ by the $W^K$ weight matrix produces $k_1$, and multiply $x_1$ by the $W^V$ weight matrix produces $v_1$. 

\begin{eqnarray}
q_1 & = x_1 \cdot W^Q,   q_2 & = x_2 \cdot W^Q \\
k_1 & = x_1 \cdot W^K,   k_2 & = x_2 \cdot W^K\\
v_1 & = x_1 \cdot W^V,   v_2 & = x_2 \cdot W^Vs
\end{eqnarray}

In the above formulations, the “query”, “key”, and “value” vectors are abstractions that are useful for calculating and thinking about attention.

#### Attention matric calculation

**The second step** in calculating self-attention is to calculate a score. Let's first calculate the score for word "Thinking" in this case. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position. The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. Note that to get stable gradients for training, the calculated scores are usually divided by the the square root of the dimension of the key vectors $\sqrt{d_k}$.

\begin{eqnarray}
score_{11} & = (q_1 \cdot k_1) / \sqrt{d_k}\\
score_{12} & = (q_1 \cdot k_2) / \sqrt{d_k} \\
score_{1n} & = (q_1 \cdot k_n) / \sqrt{d_k}
\end{eqnarray}

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/qkv_vector_att.png?raw=true" style="zoom:70%">
</div>

**The third step** is to pass the above results into *a softmax operation*. Softmax normalizes the scores so they’re all positive and add up to 1. This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

\begin{eqnarray}
att_{11} & = softmax(score_{11}) = softmax((q_1 \cdot k_1) / \sqrt{d_k})\\
att_{12} & = softmax(score_{12}) = softmax((q_1 \cdot k_2) / \sqrt{d_k}) \\
att_{1n} & = softmax(score_{1n}) = softmax((q_1 \cdot k_n) / \sqrt{d_k})
\end{eqnarray}

#### Attention Output
**The forth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny softmax score, 0.0001 for example). Afther that, we could sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word "Thinking"). 

#### Matrix Calculation 
Compared to the attention calculation on vectors, the first step for matrix calculation is to calculate the Query, Key, and Value matrices. Specifically, we need to pack the embeddings into a matrix $X$, and multiply it by the weight matrices we’ve trained ($W^Q, W^K, W^V$). In the following figure, each row in the $X$ denotes a word in the input sentence.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/qkv_matrix.png?raw=true" style="zoom:70%">
</div>

After obtaining the Query, Key, and Value matrices, we can follow the above steps two to four in one formula to calculate the self-attention layer output.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/qkv_matrix_attention.png?raw=true" style="zoom:70%">
</div>

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(\frac{QK^{\rm{T}}}{\sqrt{d_{k}}})V$$

Code is provided as in the following:
```python
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
```

### Multi-head self-attention (MHSA)

Since we have dealt with self-attention, we can easily extend the self-attention operation to the multi-head self-attention (MHSA) operation, which is the basic operator in Transformer model. The multi-head self-attention mechanism improves the performance of the attention layer from the following two perspectives.

- It expands the model’s ability to focus on different positions. 
- It gives the attention layer multiple “representation subspaces”. With multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices. Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace. This facilitates the model capacity in modeling.

In this case, we will introduce how to use eight attention heads (since the Transformer typically uses eight attention heads) for MHSA calculation. Simply speaking, we need do the same self-attention calculation outlined above, just eight different times with different weight matrices, we end up with eight different $Z$ matrices (i.e., $Z_0, Z_1, ..., Z_7$ in the following figure). After that, we concat the matrices then multiple them by an additional weights matrix $W^O$ to get the final output $Z$.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/multi_head_self_attention.png?raw=true" style="zoom:70%">
</div>

Code is provided in the following:
```python
class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.
    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
```


### Basis Architectuers
In this section, we mainly introduce four impornant components in Transformer architectures:
- Positional Encoding
- Encoder
- Decoder
- Head Layer

#### Positional Encoding
The positional encoding is responsible for accounting for the order of the words in the input sequence. To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during self-attention.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/positional_encoding_1.png?raw=true" style="zoom:50%">
</div>

Specifically, if we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/positional_encoding_2.png?raw=true" style="zoom:60%">
</div>

In fact, there are many ways to fomulate the positioanal encoding, like:
- Sin-Cos fixed positional encodings
- Learnable positional encodings

Take the sin-cos fixed positioanl encodings for example, the pattern looks like in the following figure. Each row corresponds the a positional encoding of a vector. 

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/sin_cos_2.png?raw=true" style="zoom:80%">
</div>


Code is provided in the following:
```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
```

#### Encoder
The encoder is typically composed of a stack of $N$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. The residual connection around each of the two sub-layers is employed, followed by layer normalization.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/encoder_block.png?raw=true" style="zoom:80%">
</div>

LayerNorm uses a similar scheme to BatchNorm, however, the normalization is not applied per dimension but per data point. Put it differently, with LayerNorm, we normalize each data point separately. Moreover, each data point’s mean and variance are shared over all hidden units (i.e. neurons) of the layer. For instance, in Image processing, we normalize each image independently of any other images, the mean and variance for each image is computed over all of its pixels and channels and neurons of the layer.

Below is the formula to compute the mean and standard deviation of one data point. $l$ indicates the current layer, $H$ is the number of neurons in layer $l$, and $a^l_i$ is the summed input from the layer $l-1$ to neuron $i$ of layer $l$.


\begin{eqnarray}
\mu & = & \frac{1}{H}\sum\limits_{i=1}^{H}\alpha^l_i \\
\sigma^l & = & \sqrt{ \frac{1}{H}\sum\limits_{i=1}^{H}\left(\alpha^l_i - \mu^l\right)^2}
\end{eqnarray}

Using this mean and standard deviation, the subsequent steps are the same as with BatchNorm: the input value is demeaned, then divided by standard deviation, and then affine transformed with learned $\gamma$ and $\beta$.


> Refenrece: https://arxiv.org/abs/1607.06450

Code is provided in the following:
```python
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
```

#### Decoder
The decoder is also composed of a stack of $N$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. 


<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/decoder_block.png?raw=true" style="zoom:50%">
</div>

Code is provided in the following:
```python
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
```

#### Head Layer
The head layer is composed of the final linear layers, which are followed by a Softmax Layer. The decoder stack outputs a vector of floats. Afther that, the final linear layers project the vector produced by the stack of decoders, into a much larger vector called a logits vector. The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). In the classification case, the cell with the highest probability is chosen, and the class associated with it is produced as the output for this time step.


<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/head_layer.png?raw=true" style="zoom:80%">
</div>

Code is provided in the following:
```python
class ClsHead(BaseHead):
    """classification head.
    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=False,
                 init_cfg=None):
        super(ClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        losses = self.loss(cls_score, gt_label)
        return losses
```


#### The overall architecture
In this sub-section, we will introduce how the overall Transformer model works. The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence:


<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/decoder.gif?raw=true" style="zoom:80%">
</div>

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/decoder_2.gif?raw=true" style="zoom:80%">
</div>

We provide the encoder-decoder sample codes for better illustration:
```python
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
```

## Classic Methods
In this section, we will mainly introduce the Transformer-baed models employed in computer vision.

### ViT
Now that we have a rough idea of how Multi-headed Self-Attention and Transformers work, let’s move on to the ViT. Generally speacking, ViT  suggests using a Transformer Encoder as a base model to extract features from the image, and passing these “processed” features into a Multilayer Perceptron (MLP) head model for classification. Vit roughlly works in the following steps:

- Split an image into patches
- Flatten the patches
- Produce lower-dimensional linear embeddings from the flattened patches
- Add positional embeddings
- Feed the sequence as an input to a standard transformer encoder
- Pretrain the model with image labels (fully supervised on a huge dataset)
- Finetune on the downstream dataset for image classification

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/vit.gif?raw=true" style="zoom:35%">
</div>

> Reference: https://arxiv.org/abs/2010.11929

#### Processing Images
The standard Transformer receives as input a 1D sequence of token embeddings. Special processing methods are needed for 2D images. For a standard $28 × 28$ MNIST image, we’d have to deal with 784 pixels. If we were to pass this flattened vector of length 784 through the Attention mechanism, we’d then obtain a $784 × 784$ Attention Matrix to see which pixels attend to one another. This is very costly even for modern-day hardware. Thus, ViT suggests breaking the image down into square patches as a form of lightweight “windowed” Attention. The so-called image patches are basically the sequence tokens (like words). 

Specifically, ViT reshapes the image $x \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \dot C)}$, where $(H,W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image patch, and $N=HW/P^2$ is the resulting number of patches.

Here's the sample code for patch processing.
```python
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
```

ViT prepends a learnable embedding, i.e., [class token], to the sequence of embedded patches,
$$(z^0_0 = x_\rm{class}),$$ whose state at the output of the Transformer encoder $(z^0_L)$ serves as the image representation.

The sample code for learnable embedding is as what follows,
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
cls_token = self.cls_token.expand(x.shape[0], -1, -1)  
x = torch.cat((cls_token, x), dim=1)
```

After that, ViT adds learnable/sin-cos positional encodings to the flatten patches. We provide the learnable positional encodings code for illustration.

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
x = x + self.pos_embed
```

To this end, we have processed the image to the 1D sequence of token embeddings that can be fed into Transformers. The overall images processing can be formulated by:

$$z_0 = [x_\rm{class};x^1_pE;x^2_pE;···;x^N_pE]+E_{pos}$$

#### Transformer Encoder Processing
The ViT is build on the well-known Transformer encoder. The Transformer encoder consists of alternating layers of multiheaded selfattention and MLP blocks. Layernorm is applied before every block, and residual connections after every block. The feature modelling process can be formulated by the MHSA operation.

\begin{eqnarray}
z^,_l & = & {\rm{MHSA}} ({\rm{LN}}(z_{l-1})) + z_{l-1} \\
z_l & = & {\rm{MLP}} ({\rm{LN}}(z^,_{l})) + z^,_{l} \\
 y & = & {\rm{LN}}(z^0_l)
\end{eqnarray}

<div  align="center">  
<img src="https://github.com/ChongjianGE/COMP3340_Applied_DL_Note/blob/main/trans_mlp/vit_encoder.png?raw=true" style="zoom:90%">
</div>

We provide the code of one single Transformer encoder block as what follows,
```python
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

#### Head Layer Processing
The outputs of the Transformer Encoder are then sent into a Multilayer Perceptron for image classification. The input features capture the essence of the image very well, hence making the MLP head’s classification task far simpler.

The Transformer gives out multiple outputs. Only the one related to the special [class token] embedding is fed into the classification head; the other outputs are ignored. The MLP, as expected, outputs a probability distribution of the classes the image could belong to.

```python
self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
x = self.head(x)
```


### PVT or others????
[WIP]

## Quiz
### Quiz 1 - Write the code for MHSA
Descriptions: Given the code of single-head self-attention layer, fill in the code of MHSA.
```python
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
```
### Quiz 2 - Write the code for Position Embedding
Descriptions: Given the code of learnable position embedding, fill in the code of sin-cos fixed position embedding.
### Quiz 3 - Writing the implementation of LayerNorm
Descriptions: Given the defination of LayerNorm operation, wite the implementation code.



## Refenrence 
1. https://jalammar.github.io/illustrated-transformer/
2. https://arxiv.org/abs/1706.03762
3. https://arxiv.org/abs/1607.06450
4. https://tungmphung.com/deep-learning-normalization-methods/
5. https://theaisummer.com/vision-transformer/
6. https://arxiv.org/abs/2010.11929
7. https://blog.paperspace.com/vision-transformers/
