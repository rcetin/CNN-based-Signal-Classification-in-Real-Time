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

<img src="https://latex.codecogs.com/svg.latex?y[t]%20=%20%20\sum_{a%20=%20-\infty%20}^{\infty%20}%20x[a]%20w[t-a]">,

where <img src="https://latex.codecogs.com/svg.latex?x"> is the input and <img src="https://latex.codecogs.com/svg.latex?w"> is the kernel that shifts through the information in the input and kernel the parts that are summation to it and exclude the rest. Input data of convolution layers are generally multidimensional arrays. A convolution operator depends on tensor shape can be implemented in more dimension. The two-dimensional that is employed in our study can be defined as below

<img src="https://latex.codecogs.com/svg.latex?Y[k,l]%20=%20%20\sum_{a_1%20=%20-%20\infty%20}^{%20\infty%20}%20\sum_{a_2%20=%20-%20\infty%20}^{\infty%20}%20X[k%20-a_1,%20l-a_2]%20W[%20a_1,%20a_2]">,

where <img src="https://latex.codecogs.com/svg.latex?X_{n_1xn_2}"> represents two dimensional input matrix and <img src="https://latex.codecogs.com/svg.latex?W_{m_1xm_2}"> is a kernel matrix with <img src="https://latex.codecogs.com/svg.latex?m_1%20\leqslant%20n_1$%20and%20$m_2%20\leqslant%20%20n_2">. The main goal of the convolution operator usage is to reduce the input image to its essential features. A feature map is produced by sliding the convolution filter over the input signal. The sliding scale is a hyper-parameter known as a stride. The size of the feature map or convolution layer output length for each dimension can be realized using the following equation

<img src="https://latex.codecogs.com/svg.latex?o_d%20=%20\frac{n_d%20-%20m_d}{s_d}%20+1">,

where <img src="https://latex.codecogs.com/svg.latex?d"> is the number of dimensions, <img src="https://latex.codecogs.com/svg.latex?n_d$%20and%20$m_d"> represent the length of the input vector and the kernel length in <img src="https://latex.codecogs.com/svg.latex?d^{th}"> dimension, where <img src="https://latex.codecogs.com/svg.latex?s"> is the value of stride.

### Activation functions

In neural networks, when output data is generated from input data, activation functions are proposed to introduce non-linearity. The activation functions employed in our study are described below.

-   **Rectified Linear Unit (ReLu)** It offers much faster learning than sigmoid and tangent functions because of the simpler mathematical operations. Although it is continuous, it is not differentiable.

    <img src="https://latex.codecogs.com/svg.latex?\phi%20(z)%20=%20max(0,%20z)">.

-   **Softmax function:** It is the type of sigmoid function, and the
    softmax output can be considered a probability distribution over a
    finite set of outcomes [@10]. Therefore it is used in the output
    layer of the proposed architecture, especially for multiclass
    classification problems.
    
    <img src="https://latex.codecogs.com/svg.latex?\phi%20(z_i)%20=%20\frac{e^{z_i}}{\sum_{k=1}^{K}%20e^{z_k}}">,
        
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
represented as follows <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}%20=%20{W^\top}\mathbf{x}%20+%20\mathbf{b}">,
where <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}"> is the output vector of the fully connected layer,
<img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"> is the input vector, <img src="https://latex.codecogs.com/svg.latex?W"> denotes the matrix includes the
weights of the connections between the neurons, and <img src="https://latex.codecogs.com/svg.latex?\mathbf{b}">
represents the bias term vector.

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

<img src="https://latex.codecogs.com/svg.latex?\mathrm{L(\mathbf{p},\mathbf{y})}%20=%20-{\sum_{k=1}^{K}%20y_k%20\text{log}(p_k)}">,

where <img src="https://latex.codecogs.com/svg.latex?y_k%20\in%20\left%20\{%20{0,1}%20\right%20\}">,
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

<img src="https://latex.codecogs.com/svg.latex?\mathit{J(\Theta)}%20%20=%20-\frac{1}{N}%20{\sum_{n=1}^{N}%20\sum_{k=1}^{K}%20y_k^{(n)}%20\text{log}(p_k^{(n)})}">,

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
<img src="https://latex.codecogs.com/svg.latex?\nabla_{\theta_{k}}\mathit{J(\Theta)}%20=%20\frac{1}{N}%20{\sum_{n=1}^{N}%20\left%20(%20p_k^{(n)}%20-y_k^{(n)}%20\right%20)}%20\mathbf{z_k}^{(n)}">,
where <img src="https://latex.codecogs.com/svg.latex?\mathbf{z_k}^{(n)}"> is the score regarding <img src="https://latex.codecogs.com/svg.latex?n^{th}"> instance.
Learning rate is updated at each iteration according to the equations as
in below:
<img src="https://latex.codecogs.com/svg.latex?\mathit{\Theta}_\iota%20=\mathit{%20\Theta}_{\iota-1}%20-%20\eta%20\mathit{\Omega}_\iota">,
<img src="https://latex.codecogs.com/svg.latex?\mathit{\Omega}_\iota%20=%20%20\nabla_{\theta_k}%20%20\mathit{J(\Theta_\iota})">,

where <img src="https://latex.codecogs.com/svg.latex?\nabla(\cdot)"> indicates the partial derivation, <img src="https://latex.codecogs.com/svg.latex?\eta"> is the
learning rate and $\Omega_\iota$ is the gradient matrix of the weight
matrix at the iteration time $\iota$. All weights are updated according
to the chain rule in the backpropagation algorithm in each iteration
from the output unit to inputs.

### Adaptive Moment Estimation

Adam can be defined as an adaptive learning rate method because of the
capability to compute individual learning rates for different
parameters. It calculates the learning rate depending on the gradient of
cost function and estimates the first moment (mean) and the second
moment (variance) of the gradient to update parameters. Estimations of
mean and variance can be calculated using the following equations:
$$\widehat{E}\left [ \psi  \right ]_\iota =  \frac{E\left [ \psi  \right ]_\iota}{ 1 -\left ( \gamma_1  \right )^\iota},\\$$
$$\widehat{E}\left [ \psi^2  \right ]_\iota =  \frac{E\left [ \psi^2  \right ]_\iota}{ 1 -\left ( \gamma_2  \right )^\iota},\\$$
where $\psi$ indicates the gradient of the cost function, $\gamma_1$ and
$\gamma_2$ are values of the decay. After that, the weights are updated
according to the following equation:
$$\theta_{\iota}^{\tau} = \theta_{\iota-1}^{\tau} - \frac{\eta } {{\sqrt{\widehat{E}\left [ \psi^2  \right ]_\iota}} + \epsilon },\\$$
where $\tau$ denotes adaptive learning rate of each individual parameter
and $\epsilon$ is a small term which is used to prevent division to
zero.

![Backpropagation algorithm on two layer neural
network.[]{label="backpropagation"}](./fig/Backpropagation_mlp.pdf){#backpropagation
width="380pt"}

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
for an instance as follows. To update $v_{it}$, a weight at the second
layer, the chain rule is: $$\label{chain_rule}
    \frac{\partial \mathrm{L({p_i},{y_i})} }  {\partial v_{it}}
       =  \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_i}
           \frac{\partial p_i}{\partial z_i}
           \frac{\partial z_i}{\partial v_{it}},$$
$$\frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_i}
   =  - \sum_{k=1}^{K} \frac{y_{k}}{p_{k}},$$ $$\begin{align*}
            \frac{\partial p_i}{\partial z_i} & =  \left\{\begin{matrix}
            \tfrac{e^{z_i}}{\sum_{k=1}^{K}e^{z_k}}
                              - \left ( \tfrac{e^{z_i}}{\sum_{k=1}^{K}e^{z_k}}   \right ) ^2 ,& i=k \\ \\[0,05cm]
            - \frac{e^{z_i} e^{z_k}}{ \left (  \sum_{k=1}^{K}e^{z_k}   \right ) ^2  }  \      ,& i\neq k 
            \end{matrix}\right.
        \\  \\[0,05cm]
            &= \left\{\begin{matrix}
            p_i - {p_k}^2 \ ,& i=k \\ 
            -p_i \ p_k  \   ,& i \neq k
            \end{matrix}\right.
    \end{align*}$$ $$\begin{align*}
     \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial z_i}  
               &= \sum_{k=1}^{K} \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_i}
                     \frac{\partial p_i}{\partial z_i} ,\\ 
                     \vspace{0.3cm}
               &= \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_i}
                     \frac{\partial p_i}{\partial z_i} - \sum_{k\neq i}^{K}   \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_k}
                     \frac{\partial p_k}{\partial z_i} , & \\ 
                     \vspace{0.3cm}
     &= -y_i(1 - p_i) +  \sum_{k\neq i}^{K} y_k p_i ,\\ 
     &= p_i - y_i.
    \end{align*}$$ $${z_i} = \sum_{t=1}^{T} s_{t}v_{it} + b_{t},$$ where
$b_{t}$ is the bias term, $s_{t}$ is the output of $t^{th}$ neuron at
hidden layer.
$$\frac{\partial{z_i}}{\partial v_{it}} = \sum_{t=1}^{T} s_{t},$$
$$\frac{\partial \mathrm{L({p_i},{y_i})} }  {\partial v_{it}} = \left ( p_i - y_i \right )  \sum_{t=1}^{T} s_{t},$$
To update $w_{td}$, a weight at first layer chain rule is:
$$\frac{\partial \mathrm{L({p_i},{y_i})} }  {\partial w_{td}}
       =  \frac{\partial \mathrm{L({p_i},{y_i})}}{\partial p_i}
           \frac{\partial p_i}{\partial z_i}
            \frac{\partial z_i}{\partial s_{t}}
             \frac{\partial s_{t}}{\partial u_{t}}
              \frac{\partial u_{t}}{\partial w_{td}}.$$ $$\begin{align*}
         \frac{\partial s_{t}}{\partial u_{t}} &=\left (\frac{1}{1+e^{-u_{t}}}\right )\left ( 1- \frac{1}{1+e^{-u_{t}}} \right ),  \\[0,5cm] 
         &= s_t\left (1- s_t \right ).
    \end{align*}$$ $$u_t = \sum_{n=1}^{N} x_d w_{td},$$
$$\frac{\partial u_t}{\partial w_{td}} = \sum_{n=1}^{D} x_d ,$$
$$\frac{\partial \mathrm{L({p_i},{y_i})} }  {\partial w_{td}}
       =  \sum_{d=1}^{D}\left ( p_i - y_i \right ) v_{it} \ s_t\left (1- s_t \right )x_d .$$

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
$\mathbf{f}=(f^{(1)} f^{(2)} \cdots f^{(d)})$, all features are
normalized as in below;
$$\widehat{\mathbf{f}}=\frac{f^{(d)}-\mathrm{E}[f^{(d)}]}  {\sqrt{\mathrm{Var}[f^{(d)}]}} \ .\\$$
After the feature normalization, batch normalization can be defined for
each one of the batches as in below:
$$\mathbf{\mu}_{B} = \frac{1}{R} \sum_{r=1}^{R}\mathbf{f_r},$$
$$\mathbf{\sigma}_{B}^{2} = \frac{1}{R}\sum_{r=1}^{R}  \left ( \mathbf{f_r} - \mathbf{\mu}_{B}\right )^2  ,$$
where $R$ represents the total number of features at one batch,
$\mathbf{\mu}_{B}$ and $\mathbf{\sigma}_{B}^{2}$ are mean and variance
of the batch respectively.
$$\widehat{\mathbf{f}_r}=\frac{\mathbf{f}_r-\mathbf{\mu}_{B}}  {\sqrt{\mathbf{\sigma}_{B}^{2}+\varepsilon}}  \ .$$
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