Convolutional Neural Networks
=============================

Convolutional neural networks (CNNs) are specialized neural networks for
data processing and are considered a grid-like structure. CNNs use the
convolution operator at least one layer, unlike the other types of
neural networks. Usage of the convolution operator provides three
advantages that play significant roles for improving the machine
learning system: sparse interactions, parameter sharing, and equivariant
representations [@10]. CNNs can be thought of as a series of layers. In
this study, convolutional layers, downsampling layers are used to
extract features, and a flatten layer is preferred to create vector
forms of feature maps before the classification part of the network
architecture.

CNN Layers
----------

### Convolution Layer

The convolution layer is based on a discrete convolution process.
Discrete convolution is given as the following;

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?y[t]%20=%20%20\sum_{a%20=%20-\infty%20}^{\infty%20}%20x[a]%20w[t-a],">
</p>

where <img src="https://latex.codecogs.com/svg.latex?x"> is the input and <img src="https://latex.codecogs.com/svg.latex?w"> is the kernel that shifts through the information in the input and kernel the parts that are summation to it and exclude the rest. Input data of convolution layers are generally multidimensional arrays. A convolution operator depends on tensor shape can be implemented in more dimension. The two-dimensional that is employed in our study can be defined as below

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Y[k,l]%20=%20%20\sum_{a_1%20=%20-%20\infty%20}^{%20\infty%20}%20\sum_{a_2%20=%20-%20\infty%20}^{\infty%20}%20X[k%20-a_1,%20l-a_2]%20W[%20a_1,%20a_2],">
</p>

where <img src="https://latex.codecogs.com/svg.latex?X_{n_1xn_2}"> represents two dimensional input matrix and <img src="https://latex.codecogs.com/svg.latex?W_{m_1xm_2}"> is a kernel matrix with <img src="https://latex.codecogs.com/svg.latex?m_1%20\leqslant%20n_1$%20and%20$m_2%20\leqslant%20%20n_2">. The main goal of the convolution operator usage is to reduce the input image to its essential features. A feature map is produced by sliding the convolution filter over the input signal. The sliding scale is a hyper-parameter known as a stride. The size of the feature map or convolution layer output length for each dimension can be realized using the following equation

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?o_d%20=%20\frac{n_d%20-%20m_d}{s_d}%20+1,">
</p>

where <img src="https://latex.codecogs.com/svg.latex?d"> is the number of dimensions, <img src="https://latex.codecogs.com/svg.latex?n_d$%20and%20$m_d"> represent the length of the input vector and the kernel length in <img src="https://latex.codecogs.com/svg.latex?d^{th}"> dimension, where <img src="https://latex.codecogs.com/svg.latex?s"> is the value of stride.

### Activation functions

In neural networks, when output data is generated from input data, activation functions are proposed to introduce non-linearity. The activation functions employed in our study are described below.

-   **Rectified Linear Unit (ReLu)** It offers much faster learning than sigmoid and tangent functions because of the simpler mathematical operations. Although it is continuous, it is not differentiable.

    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\phi%20(z)%20=%20max(0,%20z)">.
    </p>

-   **Softmax function:** It is the type of sigmoid function, and the
    softmax output can be considered a probability distribution over a
    finite set of outcomes [@10]. Therefore it is used in the output
    layer of the proposed architecture, especially for multiclass
    classification problems.
    
    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\phi%20(z_i)%20=%20\frac{e^{z_i}}{\sum_{k=1}^{K}%20e^{z_k}},">
    </p>

    where <img src="https://latex.codecogs.com/svg.latex?z_i"> is input of the softmax, <img src="https://latex.codecogs.com/svg.latex?i"> is the output index and <img src="https://latex.codecogs.com/svg.latex?K"> is the number of classes.

### Pooling layer

Another important part of CNNs is the pooling operation. A pooling layer
does not include learnable parameters like bias units or weights.
Pooling operations are used to decrease the size of feature maps with
some functions that calculate the average or the maximum value of each
distinct region of size <img src="https://latex.codecogs.com/svg.latex?a_1 \times a_2"> from the input. It helps the
representation become slightly invariant to small translations of the
input. A pooling layer solves disadvantages related to the probability
of over-fitting and computational complexity [@11].

### Flatten layer

A flatten layer is used between feature extraction and classification
sections to arrange tensor shape. The output tensor shape is mostly two
or more dimensional tensor. Therefore the tensor shape is decreased to a
one-dimensional vector with flatten layer to get a suitable input size
for dense layers.

### Fully-connected layer

Fully-connected layers are also called dense layers and correspond to
convolution layers with a kernel of size <img src="https://latex.codecogs.com/svg.latex?(1%20\times%201)">. In the
fully-connected layer, all units are connected with the units at the
previous layer. Outputs are multiplied with weight a and are given as
inputs to the units of the next layers. This processes can be
represented as follows

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{y}%20=%20{W^\top}\mathbf{x}%20+%20\mathbf{b},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}"> is the output vector of the fully connected layer, <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"> is the input vector, <img src="https://latex.codecogs.com/svg.latex?W"> denotes the matrix includes the weights of the connections between the neurons, and <img src="https://latex.codecogs.com/svg.latex?\mathbf{b}"> represents the bias term vector.

CNN Learning 
------------

CNNs occur a lot of layers and connections between these layers. As a
result of this, they consist of many parameters that are required to be
tuned. The main purpose of CNNs is to find out the best values for
parameters because they directly affect classification performance. The
learning ability of CNNs increases with tuning parameters. In the
following, we try to answer by explaining the main parts of the CNNs'
learning mechanism mathematically.

### Cross-Entropy Loss Function

A loss function quantifies the difference between the estimated output
of the model (the prediction) and the correct output (the ground truth)
to provide better convergence. In this study, we utilized the
cross-entropy loss function for multi-class classification. The
probability of each class is calculated according to the softmax
function ([\[softmax\]](#softmax){reference-type="ref"
reference="softmax"}) and the cross-entropy loss
([\[multi-cross\]](#multi-cross){reference-type="ref"
reference="multi-cross"}) for an instance is generated as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathrm{L(\mathbf{p},\mathbf{y})}%20=%20-{\sum_{k=1}^{K}%20y_k%20\text{log}(p_k)},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?y_k%20\in%20\left%20\{%20{0,1}%20\right%20\},">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{y}%20=%20[0,%20\cdots,y_k,%20%20\cdots,%200]"> is the output vector, <img src="https://latex.codecogs.com/svg.latex?K"> is
the number of classes and <img src="https://latex.codecogs.com/svg.latex?p_{k}"> is the estimated probability that the instance <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"> belongs to class <img src="https://latex.codecogs.com/svg.latex?k">. <img src="https://latex.codecogs.com/svg.latex?y_{k}"> is equal to 1 if the
target class is k. Otherwise, it is equal to 0.

### Optimization

Optimization algorithms are used when the closed-form equation can not
be preferable due to the presence of a singular matrix, a large dataset.
In this study, gradient descent based adaptive moment estimation (ADAM)
algorithm is utilized for training deep learning algorithms. Gradient
descent based optimization methods adjust the parameters iteratively to
minimize the cost function, which calculates a sum of loss functions
using the training data set. When cross-entropy loss function is used as
a loss function, the cost function is defined for multiclass
classification problems as follows: 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathit{J(\Theta)}%20%20=%20-\frac{1}{N}%20{\sum_{n=1}^{N}%20\sum_{k=1}^{K}%20y_k^{(n)}%20\text{log}(p_k^{(n)})},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?N"> is the number of instances, $\theta$ is the parameter vector
and <img src="https://latex.codecogs.com/svg.latex?\Theta"> is the parameter matrix. Gradient descent measures the local gradient of the cost function with regards to the parameter vector, and it goes in the direction of the descending gradient until
the algorithm converges to a minimum. At this point, an important
hyperparameter that must be determined carefully is the learning rate,
which specifies how often updating parameters occurs. The learning rate
changes depending on the gradient of the cost function which is
calculated at each iteration and each unit. At the output layer the
gradient of the cost function
([\[multi-cost\]](#multi-cost){reference-type="ref"
reference="multi-cost"}) can be defined for <img src="https://latex.codecogs.com/svg.latex?k^{th}"> class as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\nabla_{\theta_{k}}\mathit{J(\Theta)}%20=%20\frac{1}{N}%20{\sum_{n=1}^{N}%20\left%20(%20p_k^{(n)}%20-y_k^{(n)}%20\right%20)}%20\mathbf{z_k}^{(n)},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?\mathbf{z_k}^{(n)}"> is the score regarding <img src="https://latex.codecogs.com/svg.latex?n^{th}"> instance.
Learning rate is updated at each iteration according to the equations as
in below:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathit{\Theta}_\iota%20=\mathit{%20\Theta}_{\iota-1}%20-%20\eta%20\mathit{\Omega}_\iota,">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathit{\Omega}_\iota%20=%20%20\nabla_{\theta_k}%20%20\mathit{J(\Theta_\iota}),">
</p>

where <img src="https://latex.codecogs.com/svg.latex?\nabla(\cdot)"> indicates the partial derivation, <img src="https://latex.codecogs.com/svg.latex?\eta"> is the
learning rate and <img src="https://latex.codecogs.com/svg.latex?\Omega_\iota"> is the gradient matrix of the weight matrix at the iteration time <img src="https://latex.codecogs.com/svg.latex?\iota">. All weights are updated according
to the chain rule in the backpropagation algorithm in each iteration
from the output unit to inputs.

### Adaptive Moment Estimation

Adam can be defined as an adaptive learning rate method because of the
capability to compute individual learning rates for different
parameters. It calculates the learning rate depending on the gradient of
cost function and estimates the first moment (mean) and the second
moment (variance) of the gradient to update parameters. Estimations of
mean and variance can be calculated using the following equations:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{E}\left%20[%20\psi%20%20\right%20]_\iota%20=%20%20\frac{E\left%20[%20\psi%20%20\right%20]_\iota}{%201%20-\left%20(%20\gamma_1%20%20\right%20)^\iota},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{E}\left%20[%20\psi^2%20%20\right%20]_\iota%20=%20%20\frac{E\left%20[%20\psi^2%20%20\right%20]_\iota}{%201%20-\left%20(%20\gamma_2%20%20\right%20)^\iota},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?\psi"> indicates the gradient of the cost function, <img src="https://latex.codecogs.com/svg.latex?\gamma_1"> and
<img src="https://latex.codecogs.com/svg.latex?\gamma_2"> are values of the decay. After that, the weights are updated according to the following equation:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\theta_{\iota}^{\tau}%20=%20\theta_{\iota-1}^{\tau}%20-%20\frac{\eta%20}%20{{\sqrt{\widehat{E}\left%20[%20\psi^2%20%20\right%20]_\iota}}%20+%20\epsilon%20},">
</p>

where <img src="https://latex.codecogs.com/svg.latex?\tau"> denotes adaptive learning rate of each individual parameter and <img src="https://latex.codecogs.com/svg.latex?\epsilon"> is a small term which is used to prevent division to zero.

| ![Backpropagation_mlp.png](https://github.com/rcetin/Convolutional-Neural-Network-based-Signal-Classification-in-Real-Time/blob/main/figs/Backpropagation_mlp.png) | 
|:--:| 
| *Backpropagation algorithm on two layer neural network.* |

### Backpropogation

Backpropagation is the iterative procedure that decreases the cost in a
sequence by adjusting the weights. Backpropagation does not update the
weights of the model, and the optimizer's adjustment depends on the
gradient of the cost function. The backpropagation algorithm takes the
partial derivative of the cost function with respect to the weights in
accordance with the chain rule
([\[chain\_rule\]](#chain_rule){reference-type="ref"
reference="chain_rule"}) and propagates back to the network from the
outputs to the inputs. Regarding the multiclass classification problem,
we assume that softmax function and sigmoid function are used at the
output layer, hidden layer, respectively. Gradients are computed related
to the neural network architecture in Figure
[1](#backpropagation){reference-type="ref" reference="backpropagation"}
for an instance as follows. To update <img src="https://latex.codecogs.com/svg.latex?v_{it},"> a weight at the second
layer, the chain rule is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20\mathrm{L({p_i},{y_i})}%20}%20%20{\partial%20v_{it}}=%20%20\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_i}\frac{\partial%20p_i}{\partial%20z_i}\frac{\partial%20z_i}{\partial%20v_{it}},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_i}=%20%20-%20\sum_{k=1}^{K}%20\frac{y_{k}}{p_{k}},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\frac{\partial%20p_i}{\partial%20z_i}%20&%20=%20%20\left\{\begin{matrix}\tfrac{e^{z_i}}{\sum_{k=1}^{K}e^{z_k}}-%20\left%20(%20\tfrac{e^{z_i}}{\sum_{k=1}^{K}e^{z_k}}%20%20%20\right%20)%20^2%20,&%20i=k%20\\%20\\[0,05cm]-%20\frac{e^{z_i}%20e^{z_k}}{%20\left%20(%20%20\sum_{k=1}^{K}e^{z_k}%20%20%20\right%20)%20^2%20%20}%20%20\%20%20%20%20%20%20,&%20i\neq%20k\end{matrix}\right.\\%20%20\\[0,05cm]&=%20\left\{\begin{matrix}p_i%20-%20{p_k}^2%20\%20,&%20i=k%20\\-p_i%20\%20p_k%20%20\%20%20%20,&%20i%20\neq%20k\end{matrix}\right.\end{align*}">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20z_i}&=%20\sum_{k=1}^{K}%20\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_i}\frac{\partial%20p_i}{\partial%20z_i}%20,\\\vspace{0.3cm}&=%20\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_i}\frac{\partial%20p_i}{\partial%20z_i}%20-%20\sum_{k\neq%20i}^{K}%20%20%20\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_k}\frac{\partial%20p_k}{\partial%20z_i}%20,%20&%20\\\vspace{0.3cm}&=%20-y_i(1%20-%20p_i)%20+%20%20\sum_{k\neq%20i}^{K}%20y_k%20p_i%20,\\&=%20p_i%20-%20y_i.\end{align*}">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?{z_i}%20=%20\sum_{t=1}^{T}%20s_{t}v_{it}%20+%20b_{t},"> 
</p>

where <img src="https://latex.codecogs.com/svg.latex?b_{t}"> is the bias term, <img src="https://latex.codecogs.com/svg.latex?s_{t}"> is the output of <img src="https://latex.codecogs.com/svg.latex?t^{th}"> neuron at hidden layer.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial{z_i}}{\partial v_{it}} = \sum_{t=1}^{T} s_{t},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial \mathrm{L({p_i},{y_i})} }  {\partial v_{it}} = \left ( p_i - y_i \right )  \sum_{t=1}^{T} s_{t},">
</p>

To update <img src="https://latex.codecogs.com/svg.latex?w_{td},"> a weight at first layer chain rule is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20\mathrm{L({p_i},{y_i})}%20}%20%20{\partial%20w_{td}}=%20%20\frac{\partial%20\mathrm{L({p_i},{y_i})}}{\partial%20p_i}\frac{\partial%20p_i}{\partial%20z_i}\frac{\partial%20z_i}{\partial%20s_{t}}\frac{\partial%20s_{t}}{\partial%20u_{t}}\frac{\partial%20u_{t}}{\partial%20w_{td}}.">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\frac{\partial%20s_{t}}{\partial%20u_{t}}%20&=\left%20(\frac{1}{1+e^{-u_{t}}}\right%20)\left%20(%201-%20\frac{1}{1+e^{-u_{t}}}%20\right%20),%20%20\\[0,5cm]&=%20s_t\left%20(1-%20s_t%20\right%20).\end{align*}">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_t%20=%20\sum_{n=1}^{N}%20x_d%20w_{td},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20u_t}{\partial%20w_{td}}%20=%20\sum_{n=1}^{D}%20x_d%20,">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial%20\mathrm{L({p_i},{y_i})}%20}%20%20{\partial%20w_{td}}=%20%20\sum_{d=1}^{D}\left%20(%20p_i%20-%20y_i%20\right%20)%20v_{it}%20\%20s_t\left%20(1-%20s_t%20\right%20)x_d%20.">
</p>

### Regularization

There are many parameters in the training phase of deep CNNs, and
sometimes this causes over-fit means that the model is very successful
in the training data but fails when compared to new data. Regularization
techniques in our study are explained briefly in the following.

**Batch Normalization:** Batch normalization provides to learn a more
complex or flexible model by normalizing the mean and variance of the
output activations. The distribution of activations at each layer shows
a variation when the parameters are updated during training. It improves
learning by reducing this internal covariance shift. In this way, a
model is more resistant to problems such as vanishing, exploiting
problems. As a first step, given d-dimensional feature vector
<img src="https://latex.codecogs.com/svg.latex?\mathbf{f}=(f^{(1)}%20f^{(2)}%20\cdots%20f^{(d)}),"> all features are
normalized as in below;

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{\mathbf{f}}=\frac{f^{(d)}-\mathrm{E}[f^{(d)}]}%20%20{\sqrt{\mathrm{Var}[f^{(d)}]}}%20\%20.">
</p>

After the feature normalization, batch normalization can be defined for
each one of the batches as in below:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{\mu}_{B}%20=%20\frac{1}{R}%20\sum_{r=1}^{R}\mathbf{f_r},">
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{\sigma}_{B}^{2} = \frac{1}{R}\sum_{r=1}^{R}  \left ( \mathbf{f_r} - \mathbf{\mu}_{B}\right )^2,">
</p>

where <img src="https://latex.codecogs.com/svg.latex?R"> represents the total number of features at one batch, <img src="https://latex.codecogs.com/svg.latex?\mathbf{\mu}_{B}$ and $\mathbf{\sigma}_{B}^{2}"> are mean and variance of the batch respectively.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{\mathbf{f}_r}=\frac{\mathbf{f}_r-\mathbf{\mu}_{B}}  {\sqrt{\mathbf{\sigma}_{B}^{2}+\varepsilon}}  \ .">
</p>

All normalized features are scaled and shifted to hinder the network's
ability to utilize nonlinear transformations fully. Batch Normalization
allows us to use much higher learning rates and be less careful about
initialization. It also, in some cases eliminating the need for Dropout.

**Dropout:** Dropout can be seen as a stochastic regularization
technique. It prevents overfitting and provides a way of approximately
combining exponentially many different neural network architectures
efficiently. Dropout prevents to move some of the outputs of the
previous layer to the next layer. This can be considered as a masking
applied to cross-layer transitions. Dropout is applied to a unit in a
layer that must learn the pattern with randomly selected previous units'
outputs. In this way, the hidden units are enforced to extract valuable
features. Moreover, it reduces the risk of training data memorization.
Hyperparameters expressing the probability of the masking process are
called \"dropout rate\".
