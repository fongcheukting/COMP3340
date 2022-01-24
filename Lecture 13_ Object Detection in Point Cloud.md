# Lecture 13: Object Detection in Point Cloud

## 1. Introdution

In this lecture note, an overview of the Lidar-based 3D object detection methods is provided. We will start with the different representations of the Lidar sensor data to be processed by the deep neural networks. Then we will discuss some representative methods and elaborate on some commonly used techniques in Lidar-based 3D object detection. 

<center>
    <img src="https://i.imgur.com/XEhhErA.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">3D object detection in point cloud </div>
</center>


## 2. Different Representations for Processing Lidar Data
### Point Cloud Representation

The raw data returned by the Lidar sensor is an unordered set of lidar points. Each lidar point typically consists of  four elements, the (x,y,z) coordinates in the lidar coordinate frame and the reflection intensity which depends on the characteristics of the surface that the point belongs to. Below is a point cloud sample from the KITTI dataset, which uses a 64 lines 3D Lidar (Velodyne HDL-64E).

<center>
    <img src="https://i.imgur.com/93u0eAo.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">One point cloud returned by the Lidar sensor </div>
</center>

Each Lidar point cloud sample in the KITTI dataset on average contains 100k points and the number of each scene's returned lidar points depends on the scene charateristics such as the number of objects in the scene. 
* The point clouds are **unordered** and **irregularly distributed** in the 3D space. For instance, far objects have significantly less scanned lidar points comparing to the near objects with the same size. 
* Further, the **unordered property** of the point cloud requires the object detection network to have the **permutation invariance property**. 

These factors prevent the direct usage of CNN on point cloud data since it assume an densely ordered data structure such as the images.

### Range Image Representation
The range image representation interprets the returned Lidar sensor data as 360 degree photos of the 3d environment. Each pixel position on the range image corresponds to a certain elevation and azimuth angle, and each pixel value represents the distance from that captured point to the Lidar sensor. In a sense, it is more like the raw data format of the Lidar sensor comparing to the 3d point cloud since it aligns with the working principle of Lidar perfectly. 

<center>
    <img src="https://i.imgur.com/DuPlN0l.png">
    <img src="https://i.imgur.com/Slz214f.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Range image representation where the rows denote the elevation angle $\theta$ and the columns denote the azimuth angle $phi$. </div>
</center>

For Velodyne HDL-64E, its corresponding range image has 64 rows because of its 64 laser beams, and 4500 columns because of its azimuth resolution of 0.08 degree. There is a significant number of pixels in this image have no returned point since the laser beam being shoot to the sky without hitting any obstacle, and it is a standard pratice to assign those entries with zero value.
**The main advantage of range image representation is that it can be directly fed into the CNN networks.However, the direct usage of the conventional CNN networks may not work well for the range image since the scale ambiguity and occlusion are more severe comparing to the camera images.** For example, the CNN network usually have rectangular structure of convolutional filters with fixed parameters across the space, which assumes similar processing patterns with neighboring pixels. However, two adjacent pixels in the range image could vary significantly in the 3D space due to belonging to different objects. **As a result, the CNN filters should be carefully designed such that the filter parameters can be dynamically determined by explicitly using the depth information of the pixels inside the filter range.**

### 3D Voxelized Point Cloud
Another form of grid representation of point clouds can be achieved by voxelizing the 3D space into 3D cuboid subspaces along the x,y,z axes. The range of the voxelization is determined by both the sensor range and the application need. For example, given the range of Velodyne HDL-64E is 120m, the x and y dimensions of its corresponding 3d cuboid voxelization should not be greater than 240m assuming the lidar sensor is at the center of cuboid. And the typical ranges of z axis are from -2.5m to 1.5m considering that the Lidar sensors are mounted on the roof of the self-driving cars. 

<center>
    <img src="https://i.imgur.com/3r8Ds2w.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">3D voxelization of the 3D space to transform the point clouds into grid representation. </div>
</center>

For the quantization resolution, a typical choice is 0.1m x 0.1m x 0.1m. Within each voxel, the number of points can vary significantly. Note that most of voxels contain zero point (usually > 95% total voxels), in order to efficiently process the 3D voxelized point clouds, sparse convolutions are usually incorporated which only do the computations for the non-empty voxels. We will discuss the details of sparse convolution in later section. Besides, for the non-empty voxels, we also set a hyperparameter $K$ - number of points per voxel. For the voxels that have more points, we randomly drop the exceeding number of points. And for the voxels that have less points than $K$, we randomly repeat the points to match the requirement. As for the per-voxel features, they can be either hand-designed or machine-learned.  

<center>
    <img src="https://i.imgur.com/99gs3Ea.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Varying number of points in different voxels. </div>
</center>

To conclude, the main advantages of the 3D voxelized grid representation are:
* It can be directly processed by CNN due to its structured grid representation.
* Different from range image representation, 3d voxelization representation does not suffer from scale ambiguity and occlusion because they do not project lidar points on 2d point-of-view planes.

On the other hand, the main drawbacks are:
* quantization errors
* computation and memory inefficiency

### 2D Voxelized Point Cloud
Considering that for the application of autonomous driving, there are usually no overlapped objects of interest (such as cars and pedestrians) on the Bird Eye's View (BEV) plane. We can then only voxelize the raw point cloud along the x,y axes without differentiating the z axis. For each voxelized subspace, we often call it a "pillar".

<center>
    <img src="https://i.imgur.com/98aF5Q8.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">2D voxelization of the 3D space to transform the point clouds into grid representation. </div>
</center>

<center>
    <img src="https://i.imgur.com/pfDAibo.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Pillar representation after 2D voxelization on the BEV plane. </div>
</center>

The other pipelines are the same to the 3D voxelization except that instead of 3D convolution, only 2D convolution is needed now. Comparing to the 3D voxelization, 2D voxelization on the BEV plane is more efficient but will also introduce more quantization errors which hinder the detection performance.


## 3. Representative methods
Although there are some recent range image-based methods that can achieve comparable performance to the methods that levearge the  point cloud representation, it is beyond the scope of this lecture note and we will only focus on the point cloud-based methods. As for the point cloud-based methods, we can further classify them into point-based, voxel-based, and point-voxel-based methods.
### Point-based methods
Since the point-based methods diretly take the unordered set of Lidar points, they are required to be input-wise permutation invariant in the sense that changing the order in the input point set would not change the final detection results. To achive this, PointNet-like structure is often used. The idea behind PointNet is based on the following universal approximation theorem: any continuous input-wise permutation invariance function $f$ can be approximated by composition of two functions $h$ and $g$ where $g$ must be a symmetric function to ensure the permutation invariance property of $f$. In particular, first , function h is applied separately (point-wise) to each lidar point which transforms each lidar point to an embedding of dimension d. Then, function g takes the embeddings of lidar points generated by function h and generate a single global feature embedding of dimension d corresponding to the input point cloud. Examples of such symmetric function g are element-wise max-pooling and average-pooling.

<center>
    <img src="https://i.imgur.com/UPNdlJI.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Illustration of the proposed architecture by PointNet being applied to 4 lidar points. First, function h is applied separately to each lidar point and transforms them to embeddings with dimension d. Then, the symmetric function g takes these 4 embeddings and output a single global feature vector. </div>
</center>

Considering that the total amount of points is too huge to be processed directly, sampling methods are usually required to sample key points from the raw point set. We will discuss one representative sampling methd, Farthest Point Sampling (FPS) in the later section.

<center>
    <img src="https://i.imgur.com/iyg8xTx.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">PointNet network structure.</div>
</center>

<center>
    <img src="https://i.imgur.com/TX8kGrF.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Set abstraction operation.</div>
</center>

### Voxel-based methods
The voxelized point clouds can be directly processed by the CNN networks. Depending on the way of voxelization, these methods can be further divided into 3D voxel-based methods and 2D pillar-based methods.

As for the 3D voxel-based methods, VoxelNet is the representative method. For each voxel, it applies PointNet-like structure to extract voxel features from each point. After that, 3D convolution is leveraged to generate 3D features which is then collapsed along the z axis by concatenating the features along the z axis to the channel dimension. This generate a 2D Bird Eye's View (BEV) feature which can be further processed by the 2D backbone and region proposal network to generate the prediction results. 

<center>
    <img src="https://i.imgur.com/5LYhuiD.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">VoxelNet.</div>
</center>

In addtion, SECOND (Sparsely Embedded Convolutional Detection) further incorporates sparse convolution on top of VoxelNet to improve the computation efficiency.

<center>
    <img src="https://i.imgur.com/JQNVLko.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">SECOND: Sparsely Embedded Convolutional Detection.</div>
</center>

Regarding the 2D pillar-based methods, PointPillars is the most representative method. It also utilizes sparse convolution but only requires 2D convolution, thus is more efficient. The performance of it, however, is inevitablely worse than the 3D voxel-based methods.

<center>
    <img src="https://i.imgur.com/dRz9pRk.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">PointPillars: Fast Encoders for Object Detection from Point Clouds</div>
</center>

### Point-Voxel-based methods
There are also methods that utilize both the raw point clouds and their voxelized representations given the fact that the raw points perserve the precise geometry and the voxelized representations are more efficient to be processed by the CNN networks. 

PV-RCNN (Point-Voxel Feature Set Abstraction for 3D Object Detection) is the most representative method of this kind, it propose a Voxel Set Abstraction Module to leverage from the raw points, 3D voxel features, and 2D BEV features together to refine the initial bounding box predictions. 

<center>
    <img src="https://i.imgur.com/PJddJYq.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection</div>
</center>

```python=
# configuration of PV-RCNN's proposed Voxel Feature Set Abstraction
NAME: VoxelSetAbstraction
POINT_SOURCE: raw_points
NUM_KEYPOINTS: 2048
NUM_OUTPUT_FEATURES: 128
SAMPLE_METHOD: FPS

FEATURES_SOURCE: ['bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points']
SA_LAYER:
    raw_points:
        MLPS: [[16, 16], [16, 16]]
        POOL_RADIUS: [0.4, 0.8]
        NSAMPLE: [16, 16]
    x_conv1:
        DOWNSAMPLE_FACTOR: 1
        MLPS: [[16, 16], [16, 16]]
        POOL_RADIUS: [0.4, 0.8]
        NSAMPLE: [16, 16]
    x_conv2:
        DOWNSAMPLE_FACTOR: 2
        MLPS: [[32, 32], [32, 32]]
        POOL_RADIUS: [0.8, 1.2]
        NSAMPLE: [16, 32]
    x_conv3:
        DOWNSAMPLE_FACTOR: 4
        MLPS: [[64, 64], [64, 64]]
        POOL_RADIUS: [1.2, 2.4]
        NSAMPLE: [16, 32]
    x_conv4:
        DOWNSAMPLE_FACTOR: 8
        MLPS: [[64, 64], [64, 64]]
        POOL_RADIUS: [2.4, 4.8]
        NSAMPLE: [16, 32]
```


```python=
# forward function of PV-RCNN's proposed Voxel Feature Set Abstraction
def forward(self, batch_dict):
    """
    Args:
        batch_dict:
            batch_size:
            keypoints: (B, num_keypoints, 3)
            multi_scale_3d_features: {
                    'x_conv4': ...
                }
            points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
            spatial_features: optional
            spatial_features_stride: optional

    Returns:
        point_features: (N, C)
        point_coords: (N, 4)

    """
    keypoints = self.get_sampled_points(batch_dict)

    point_features_list = []
    if 'bev' in self.model_cfg.FEATURES_SOURCE:
        point_bev_features = self.interpolate_from_bev_features(
            keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
            bev_stride=batch_dict['spatial_features_stride']
        )
        point_features_list.append(point_bev_features)

    batch_size, num_keypoints, _ = keypoints.shape
    new_xyz = keypoints.view(-1, 3)
    new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

    if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
        raw_points = batch_dict['points']
        xyz = raw_points[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
        point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

        pooled_points, pooled_features = self.SA_rawpoints(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features,
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

    for k, src_name in enumerate(self.SA_layer_names):
        cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
        xyz = common_utils.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=self.downsample_times_map[src_name],
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

        pooled_points, pooled_features = self.SA_layers[k](
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
        )
        point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

    point_features = torch.cat(point_features_list, dim=2)

    batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
    point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

    batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
    point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

    batch_dict['point_features'] = point_features  # (BxN, C)
    batch_dict['point_coords'] = point_coords  # (BxN, 4)
    return batch_dict
```



## Farthest Point Sampling (FPS)

### Motivation

FPS process an input point set $\mathcal{P}_{in}$ of size [N,d] and output a set of keypoint points $\mathcal{P}_{out}$ of size [N',d], which maximally approximates the distribution of the original point set in the d dimensional space with N' points.

### Definition 

1. Distance between two points P<sub>1</sub> and P<sub>2</sub>:
d(P<sub>1</sub>,P<sub>2</sub>)=||P<sub>1</sub>-P<sub>2</sub>||<sub>p</sub>
where p<sub>1</sub>,p<sub>2</sub> are two separate points and ||•||<sub>p</sub> is the L<sub>p</sub>-norm which can be calculated as below:
||$\vec{x}$||<sub>p</sub>=$(\sum\limits_{i=1}^{n}|x_i|^p)^\frac{1}{p}$
2. Distance between a point P and a point set $\mathcal{P}$:
$d(P,\mathcal{P}) =\operatorname*{min}\limits_{P'\in\mathcal{P}}(d(P,P'))$

### The process of FPS

1. Initialize $\mathcal{P}_{out}=\emptyset$
2. Randomly select a point $P\in\mathcal{P}_{in}$, add $P$ to $\mathcal{P}_{out}$ and remove $P$ from $\mathcal{P}_{in}$.
3. From the remaining points in $\mathcal{P}_{in}$, select the point that is furthest to $\mathcal{P}_{out}$, add it to $\mathcal{P}_{out}$ and remove it from $\mathcal{P}_{in}$.
4. Repeat step 3 until the number of points in $\mathcal{P}_{out}$ equals to $N'$.

### A visualization of FPS

<!-- ![](https://i.imgur.com/QWLWnWu.png) -->

![](https://i.imgur.com/8gs5dr9.png)



### Example code for FPS with $L_2-norm$ in 3-D space


```python
def distance_point3d(p0, p1):
    d = (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p0[2])**2
    return math.sqrt(d)
    
def distance_point2pointset(p, P):
    d_min = 1000000
    for point_idx in range(P.shape[0]):
        d = distance_points3d(p - P[point_idx])
        if d < d_min:
            d_min = d
            
    return math.sqrt(d_min)

def furthest_point_sample(points, sample_count):
    points_index = np.arange(points.shape[0], dtype=np.int)
    # Randomly select a point for initialization
    output_point_index = np.array([np.random.choice(points_index)])
    # Delete the selected point from the original point set
    remaining_point_index = np.setdiff1d(points_index, init_point_index)
    
    while len(output_point_index) < sample_count:
        
        furthest_point_in_remaining_point_index = 0
        d_furthest_point_to_output_point_set = distance_point2pointset(points[remaining_point_index[0]], points[output_point_index])
        
        for idx_remaining in range(1, remaining_point_index.shape[0]):
            d_now = distance_point2pointset(points[remaining_point_index[idx_remaining]], points[output_point_index])
            if d_now > d_furthest_point_to_output_point_set:
                d_furthest_point_to_output_point_set = d_now
                furthest_point_in_remaining_point_index = idx_remaining
    
        # update output index and remaining index
        output_point_index = np.append(output_point_index, furthest_point_in_remaining_point_index)
        remaining_point_index = np.delete(remaining_point_index, furthest_point_in_remaining_point_index)
        
    return points[output_point_index]

```


## Sparse Convolution:

### Definition

Sparse tensors are those with many zeros (non-active sites) while only a small portion of sites are non-zeros (active sites).

### Motivation

Sparse convolution is designed for sparse input tensors for reducing computation cost.

1. Sparse convolution only applies convolution on output location when the convolution kernel contains active input sites.

2. Submanifold sparse convolution only applies convolution on output location when the center of convolution kernel is active in the input tensors.

### Input

We have a sparse input image as indicated in the figure below, where only $P_1$ and $P_2$ are non-zero pixels.


<center>
    <img src="https://i.imgur.com/tQ48o57.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Example input image</div>
</center>


### Output

Then for a (submanifold) sparse convolution with kernel size 3, the output can be illustrated as below.

![](https://i.imgur.com/NnxJFvW.png)

### Computation Pipeline

1. Build input/output hash table

The input hash table stores all active input sites. Then we estimated all possible active output sites, considering either regular output definition or submanifold output definition. At last, using the output hash table to record all involved active output sites.

<center>
    <img src="https://i.imgur.com/oom3wPT.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">Example for building input/output hash table</div>
</center>


2. Build rule book

The second step is to build the Rulebook. Below is an example of how to build Rulebook. Pᵢₙ has the input sites index. In this example, we have two nonzero data at position (2, 1) and (3, 2). Pₒᵤₜ has the corresponding output sites index. Then we collect the atomic operations from the convolution calculation process, i.e. consider the convolution process as many atomic operations w.r.t kernel elements. At last, we record all atomic operations into Rulebook. In Fig.6 the right table is an example of the Rulebook. The first column is the kernel element index. The second column is a counter and index about how many atomic operations this kernel element involves. The third and fourth column is about the source and result of this atomic operation.
![](https://i.imgur.com/bh2PhCg.png)

3. Computation Pipeline for GPU

![](https://i.imgur.com/8Cc1AvU.png)


As Fig.7 illustrates, we calculate the convolution, not like the sliding window approach but calculate all the atomic operations according to Rulebook. In Fig.7 red and blue arrows indicate two examples.
In Fig.7 red and blue arrows indicate two calculation instances. The red arrow processes the first atomic operation for kernel element (-1, -1). From Rulebook, we know this atomic operation has input from P1 with position (2, 1) and has output with position (2, 1). This entire atomic operation can be constructed as Fig. 8.

## Quiz

### Question1

Please apply Furthest Point Sampling algorithm to sample 4 keypoints in the given point set composed of 10 points with 2 dimensional features with the last point as the initialization point.

| $d_1$ | $d_2$ | 
| :-----:| :----: | 
|-8|-6|
|-8|-1|
|-18|11|
|-6|18|
|12|7|
|1|16|
|11|-4|
|19|-16|
|-3|-16|
|-9|5|
<!-- |-4|7|
|-3|-10|
|13|12|
|-15|5|
|16|-4|
|3|-9|
|2|-3|
|3|-12|
|17|3|
|-9|-17| -->

**Solution**:
1. Select (-9, 5) as the inital selected point.
2. Calculate the L2-distance between the remaining points and (-9,5).

<!-- | $\rm{}remaining\space points$ | $\rm{}minimum\space distance\space to\space the\\ selected\space points$ |  -->
| $\rm{}remaining\space points$ | $\rm{}minimum\space distance$ | 
| :-----:| :----: | 
|(-8, -6)|11.045361
|(-8, -1)|**6.08276253**|
|(-18, 11)|10.81665383|
|(-6, 18)|13.34166406|
|(12, 7)|21.09502311|
|(1, 16)|14.86606875|
|(11, -4)|21.9317122|
|(19, -16)|35.0|
|(-3, -16)|21.84032967|

(-8, -1) is selected as the next point.

3. Calculate the minimum L2-distance between the remaining points and the selected point set.

| $\rm{}remaining\space points$ | $\rm{}minimum\space distance$ | 
| :-----:| :----: | 
|(-8, -6)|**5.0**|
|(-18, 11)|10.81665383|
|(-6, 18)|13.34166406|
|(12, 7)|21.09502311|
|(1, 16)|14.86606875|
|(11, -4)|19.23538406|
|(19, -16)|30.88689042|
|(-3, -16)|15.8113883|

(-8, -6) is selected as the next point.

4. Repeat the above step with the updated point set of selected points.

| $\rm{}remaining\space points$ | $\rm{}minimum\space distance$ | 
| :-----:| :----: | 
|(-18, 11)|**10.81665383**|
|(-6, 18)|13.34166406|
|(12, 7)|21.09502311|
|(1, 16)|14.86606875|
|(11, -4)|19.10497317|
|(19, -16)|28.7923601|
|(-3, -16)|11.18033989|

(-18, 11) is selected as the final point.

5. The selected point set is {(-9, 5), (-8, -1), (-8, -6), (-18, 11)}.


### Question2

Considering the sparse matrix below, we are going to apply a (submanifold) sparse convolution with $3\times3$ convolutional kernel (with padding=1) on it to get a feature map. 

$$
\left[
\begin{matrix}
0&0&0&0&a_0\\
0&a_1&0&0&0\\
0&0&a_2&0&0\\
0&0&0&0&0\\
0&0&0&0&0\\
\end{matrix}
\right]
$$

(a) Please build the input hash table, output hash table as well as the rule book for sparse convolution and submanifold sparse convolution.

(b) Please compute the output of submanifold sparse convolution with a kernel below and $a_{0}=2$, $a_{1}=1$, $a_{2}=3$.

$$
\left[
\begin{matrix}
0.8&0.5&0.2\\
1&2&0.3\\
0.9&0.2&0.5\\
\end{matrix}
\right]
$$

**Solution**:
(a) 

Sparse Convolution:

The input hash table:

| $key$ | $value$ | 
| :-----:| :----: | 
|0|(4,0)|
|1|(1,1)|
|2|(2,2)|

The output hash table:

| $key$ | $value$ | 
| :-----:| :----: | 
|0|(3,0)|
|1|(4,0)|
|2|(3,1)|
|3|(4,1)|
|4|(0,0)|
|5|(1,0)|
|6|(2,0)|
|7|(0,1)|
|8|(1,1)|
|9|(2,1)|
|10|(0,2)|
|11|(1,2)|
|12|(2,2)|
|13|(3,2)|
|14|(1,3)|
|15|(2,3)|
|16|(3,3)|


The rule book:

| $kernel\ position$ | $count$ | $key_{in}$ | $key_{out}$ | 
| :-----:| :----: | :-----:| :----: | 
| (-1,-1) | 0 | 1 | 12 |
|  | 1 | 2 | 16 |
| (0,-1) | 0 | 0 | 3 |
|  | 1 | 1 | 11 |
|  | 2 |2 | 15 |
| (1,-1) | 0 | 0 | 2 |
| | 1 | 1 | 10 |
| | 2 | 2 | 14 |
| (-1,0) | 0 | 1 | 9 |
|  | 1 | 2 | 13 |
| (0,0) | 0 | 0 | 1 |
|  | 1 | 1 | 8 |
|  | 2 | 2 | 12 |
| (1,0) | 0 | 0 | 0 |
|  | 1 | 1 | 7 |
|  | 2 | 2 | 11 |
| (-1,1) | 0 | 1 | 6 |
| | 1 | 2 | 2 |
| (0,1) | 0 | 1 | 5 |
| | 1 | 2 | 9 |
| (1,1) | 0 | 1 | 4 |
| | 1 | 2 | 8 |

Submanifold Sparse Convolution:

The input hash table:

| $key$ | $value$ | 
| :-----:| :----: | 
|0|(4,0)|
|1|(1,1)|
|2|(2,2)|

The output hash table:

| $key$ | $value$ | 
| :-----:| :----: | 
|0|(4,0)|
|1|(1,1)|
|2|(2,2)|


The rule book:

| $kernel\ position$ | $count$ | $key_{in}$ | $key_{out}$ | 
| :-----:| :----: | :-----:| :----: | 
| (-1,-1) | 0 | 1 | 2 |
| (0,0) | 0 | 0 | 0 |
|  | 1 | 1 | 1 |
|  | 2 | 2 | 2 |
| (1,1) | 0 | 2 | 1 |

(b)

$$
\left[
\begin{matrix}
0&0&0&0&4\\
0&3.5&0&0&0\\
0&0&6.8&0&0\\
0&0&0&0&0\\
0&0&0&0&0\\
\end{matrix}
\right]
$$

## References

<text id="overview"> [1]: https://towardsdatascience.com/tagged/3d-object-detection?p=f34cf3227aea
