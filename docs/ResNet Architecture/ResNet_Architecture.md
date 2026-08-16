# ResNet Architecture

## Overview

**ResNet (Residual Network)** is a family of deep convolutional neural networks designed to make very deep networks easier to train.

ResNet was introduced in the paper:

> **Deep Residual Learning for Image Recognition**

by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

The key idea behind ResNet is **Residual Learning**.

Instead of requiring a group of layers to directly learn a complete transformation:

\[
H(x)
\]

ResNet reformulates the problem as learning a residual function:

\[
F(x)=H(x)-x
\]

The final output is then:

\[
\boxed{H(x)=F(x)+x}
\]

This is implemented using **Skip Connections**.

---

# 1. Why Was ResNet Introduced?

As neural networks became deeper, researchers expected that adding more layers would improve their ability to learn complex representations.

However, simply increasing network depth introduced several optimization challenges.

These included:

- Vanishing gradients
- Exploding gradients
- Optimization difficulty
- Slow convergence
- Degradation of training performance

A deeper network could sometimes have worse training performance than a shallower network.

This problem motivated the development of residual learning.

```text
Traditional Deep CNN
        ↓
Increase Depth
        ↓
Optimization Difficulty
        ↓
Degradation Problem
        ↓
Residual Learning
        ↓
ResNet
```

---

# 2. Main Idea of ResNet

A traditional neural network block attempts to learn:

\[
H(x)
\]

where (x) is the input and (H(x)) is the desired transformation.

ResNet introduces a residual function:

\[
F(x)=H(x)-x
\]

Therefore:

\[
H(x)=F(x)+x
\]

The input (x) is passed through a shortcut connection and added to the output of the learned transformation.

```
┌──────────────────┐
│                  │
│  Skip Connection │
│                  │
Input ────────────┤                  ├────→ Addition
  │              │                  │
  ↓              │                  │
Conv             │                  │
  ↓              │                  │
BN               │                  │
  ↓              │                  │
ReLU             │                  │
  ↓              │                  │
Conv             │                  │
  ↓              │                  │
BN ──────────────┴──────────────────┘
                    ↓
                  ReLU
```

The addition operation creates the residual connection.

---

# 3. Basic ResNet Architecture

A simplified ResNet architecture can be represented as:

- Input Image
- 7×7 Convolution
- Batch Normalization
- ReLU
- 3×3 Max Pooling
- Residual Stage 1
- Residual Stage 2
- Residual Stage 3
- Residual Stage 4
- Global Average Pooling
- Fully Connected Layer
- Output

The exact number of residual blocks inside each stage depends on the specific ResNet variant.

---

# 4. Main Components of ResNet

A ResNet architecture contains several important components:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Residual blocks
- Skip connections
- Downsampling
- Global Average Pooling
- Fully Connected classification layer

Each component contributes to the overall architecture.

---

# 5. Input Image

For image classification, a standard ResNet configuration commonly receives an RGB image such as:

\[
224\times224\times3
\]

where:

- 224 = image height
- 224 = image width
- 3 = RGB channels

The image is then processed by the initial convolutional layer.

```
224 × 224 × 3
       ↓
Initial Convolution
```

---

# 6. Initial Convolution

The original ResNet architecture begins with a:

\[
7\times7
\]

convolution with:

\[
64
\]

output channels and a stride of 2.

Conceptually:

```
Input
224 × 224 × 3
      ↓
7×7 Conv
64 channels
      ↓
112 × 112 × 64
```

The large convolutional kernel allows the network to extract initial visual patterns while reducing the spatial resolution.

---

# 7. Batch Normalization

After the initial convolution, ResNet uses Batch Normalization.

The simplified sequence is:

```
7×7 Conv
   ↓
BatchNorm
   ↓
ReLU
```

Batch Normalization helps stabilize neural network training.

A simplified formulation is:

\[
\hat{x}=
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
\]

followed by:

\[
y=\gamma\hat{x}+\beta
\]

where (\gamma) and (\beta) are learnable parameters.

---

# 8. ReLU Activation

ResNet uses the Rectified Linear Unit (ReLU) activation function.

\[
ReLU(x)=\max(0,x)
\]

Therefore:

- Negative values → 0
- Positive values → unchanged

ReLU introduces non-linearity into the network and allows the model to learn complex transformations.

---

# 9. Max Pooling

After the initial convolution and activation, ResNet uses a:

\[
3\times3
\]

Max Pooling layer with stride 2.

Conceptually:

```
112 × 112 × 64
       ↓
3×3 MaxPool
       ↓
56 × 56 × 64
```

The purpose is to reduce spatial resolution and computational cost while retaining strong local activations.

---

# 10. Residual Stages

After the initial layers, the network is organized into several Residual Stages.

A simplified structure is:

```
Stage 1
   ↓
Stage 2
   ↓
Stage 3
   ↓
Stage 4
```

Each stage contains multiple residual blocks.

The number of blocks depends on the ResNet variant.

For example, ResNet50 uses:

- Stage 1 → 3 Bottleneck Blocks
- Stage 2 → 4 Bottleneck Blocks
- Stage 3 → 6 Bottleneck Blocks
- Stage 4 → 3 Bottleneck Blocks

This gives a total of:

\[
3+4+6+3=16
\]

Bottleneck blocks.

---

# 11. Residual Block

The fundamental building block of ResNet is the Residual Block.

A basic residual block can be represented as:

```
┌──────────────────┐
│                  │
Input ───────┤   Skip Path      │
  │          │                  │
  ↓          │                  │
Conv         │                  │
  ↓          │                  │
BN           │                  │
  ↓          │                  │
ReLU         │                  │
  ↓          │                  │
Conv         │                  │
  ↓          │                  │
BN ──────────┴──────→ Add
                      ↓
                    ReLU
```

The main branch learns:

\[
F(x)
\]

while the shortcut provides:

\[
x
\]

The final output is:

\[
y=F(x)+x
\]

---

# 12. Identity Shortcut

When the input and output have the same dimensions, the shortcut can be a simple identity mapping.

```
Input
  │
  ├───────────────────┐
  │                   │
  ↓                   │
Residual Layers       │
  │                   │
  └────────→ Add ←────┘
```

Mathematically:

\[
y=F(x)+x
\]

No additional transformation is required for the shortcut.

---

# 13. Projection Shortcut

Sometimes the input and output dimensions are different.

For example:

```
56 × 56 × 64
       ↓
28 × 28 × 128
```

The input cannot be directly added to the output because their dimensions do not match.

In this situation, ResNet can use a projection shortcut, typically implemented using a:

\[
1\times1
\]

convolution.

```
Input
  │
  ├──→ 1×1 Conv ────────┐
  │                     │
  ↓                     │
Residual Branch         │
  │                     │
  └──────────→ Add ←────┘
```

The shortcut becomes:

\[
W_s x
\]

and the output is:

\[
y=F(x)+W_sx
\]

---

# 14. Downsampling

ResNet gradually reduces spatial resolution as the network becomes deeper.

A simplified progression is:

```
224 × 224
    ↓
112 × 112
    ↓
56 × 56
    ↓
28 × 28
    ↓
14 × 14
    ↓
7 × 7
```

At the same time, the number of channels generally increases:

```
64
 ↓
128 / 256
 ↓
256 / 512
 ↓
512 / 1024
 ↓
1024 / 2048
```

This creates an important architectural principle:

\[
Spatial\ Resolution \downarrow
\]

while:

\[
Feature\ Channels \uparrow
\]

---

# 15. Why Does ResNet Reduce Spatial Resolution?

Reducing spatial resolution has several purposes:

- **Computational Efficiency:** Smaller feature maps require fewer computations.
- **Larger Receptive Fields:** Deeper neurons can capture information from larger regions of the original image.
- **Hierarchical Representation:** The network gradually transforms detailed spatial information into higher-level semantic representations.

Conceptually:

```
High Resolution
     ↓
Local Features
     ↓
Medium Resolution
     ↓
Complex Structures
     ↓
Low Resolution
     ↓
High-Level Features
```

---

# 16. Bottleneck Architecture

ResNet50 and deeper variants such as ResNet101 and ResNet152 use Bottleneck Blocks.

A Bottleneck Block contains:

```
1×1 Conv
   ↓
3×3 Conv
   ↓
1×1 Conv
   ↓
Addition
   ↓
ReLU
```

The purpose of the three convolutions is different.

- **First 1×1 Convolution:** Reduces or transforms the channel dimension.
- **3×3 Convolution:** Extracts spatial features.
- **Second 1×1 Convolution:** Expands the representation back to the desired output dimension.

Conceptually:

```
Input
  ↓
1×1 Conv
  ↓
3×3 Conv
  ↓
1×1 Conv
  ↓
Add Shortcut
  ↓
ReLU
```

---

# 17. Why Use Bottleneck Blocks?

A direct sequence of multiple (3\times3) convolutions with a large number of channels would be computationally expensive.

The bottleneck design reduces computational cost by performing the spatial convolution at a smaller channel dimension.

Conceptually:

```
Large Representation
        ↓
1×1 Conv
        ↓
Reduced Representation
        ↓
3×3 Conv
        ↓
1×1 Conv
        ↓
Expanded Representation
```

This allows ResNet50 to become deeper without making the architecture prohibitively expensive.

---

# 18. Basic Block vs. Bottleneck Block

ResNet uses different block designs depending on the model variant.

**Basic Block**  

Used in architectures such as:
- ResNet18
- ResNet34

**Structure:**

```
3×3 Conv
   ↓
3×3 Conv
   ↓
Add Shortcut
```

**Bottleneck Block**  

Used in:
- ResNet50
- ResNet101
- ResNet152

**Structure:**

```
1×1 Conv
   ↓
3×3 Conv
   ↓
1×1 Conv
   ↓
Add Shortcut
```

The Bottleneck Block allows deeper architectures to remain computationally manageable.

---

# 19. ResNet Variants

The most commonly referenced ResNet architectures are:

| Model     | Depth | Block Type    |
|-----------|-------|----------------|
| ResNet18  | 18    | Basic Block    |
| ResNet34  | 34    | Basic Block    |
| ResNet50  | 50    | Bottleneck     |
| ResNet101 | 101   | Bottleneck     |
| ResNet152 | 152   | Bottleneck     |

The depth increases significantly across the family.

However, deeper does not automatically mean better.

The appropriate architecture depends on:

- Dataset size
- Computational resources
- Task complexity
- Training strategy
- Risk of overfitting

---

# 20. ResNet18 Architecture

ResNet18 uses Basic Residual Blocks.

Its residual stage configuration is:

```
Stage 1 → 2 Basic Blocks
Stage 2 → 2 Basic Blocks
Stage 3 → 2 Basic Blocks
Stage 4 → 2 Basic Blocks
```

Therefore:

\[
2+2+2+2=8
\]

residual blocks.

Each Basic Block contains two convolutional layers.

---

# 21. ResNet34 Architecture

ResNet34 also uses Basic Residual Blocks.

Its stage configuration is:

```
Stage 1 → 3 Basic Blocks
Stage 2 → 4 Basic Blocks
Stage 3 → 6 Basic Blocks
Stage 4 → 3 Basic Blocks
```

Therefore:

\[
3+4+6+3=16
\]

Basic Blocks.

Because each Basic Block contains two convolutional layers, the network becomes significantly deeper than ResNet18.

---

# 22. ResNet50 Architecture

ResNet50 uses Bottleneck Blocks.

Its stage configuration is:

```
Stage 1 → 3 Bottleneck Blocks
Stage 2 → 4 Bottleneck Blocks
Stage 3 → 6 Bottleneck Blocks
Stage 4 → 3 Bottleneck Blocks
```

Therefore:

\[
3+4+6+3=16
\]

Bottleneck Blocks.

However, each Bottleneck Block contains three convolutional layers.

Thus:

\[
16\times3=48
\]

convolutional layers.

When the initial convolution and final fully connected layer are included:

\[
48+1+1=50
\]

Therefore, the architecture is called ResNet50.

---

# 23. ResNet101 Architecture

ResNet101 also uses Bottleneck Blocks.

Its stage configuration is:

```
Stage 1 → 3 Bottleneck Blocks
Stage 2 → 4 Bottleneck Blocks
Stage 3 → 23 Bottleneck Blocks
Stage 4 → 3 Bottleneck Blocks
```

Total:

\[
3+4+23+3=33
\]

Bottleneck Blocks.

Since each block contains three convolutional layers:

\[
33\times3=99
\]

Adding the initial convolution and final fully connected layer:

\[
99+1+1=101
\]

Thus:

\[
\boxed{ResNet101}
\]

---

# 24. ResNet152 Architecture

ResNet152 uses an even deeper Bottleneck architecture.

Its stage configuration is:

```
Stage 1 → 3 Bottleneck Blocks
Stage 2 → 8 Bottleneck Blocks
Stage 3 → 36 Bottleneck Blocks
Stage 4 → 3 Bottleneck Blocks
```

Total:

\[
3+8+36+3=50
\]

Bottleneck Blocks.

Therefore:

\[
50\times3=150
\]

convolutional layers.

Adding the initial convolution and final fully connected layer:

\[
150+1+1=152
\]

Thus:

\[
\boxed{ResNet152}
\]

---

# 25. Architecture Comparison

| Architecture | Block     | Blocks per Stage       | Depth |
|--------------|-----------|-----------------------|-------|
| ResNet18     | Basic     | 2, 2, 2, 2           | 18    |
| ResNet34     | Basic     | 3, 4, 6, 3           | 34    |
| ResNet50     | Bottleneck| 3, 4, 6, 3           | 50    |
| ResNet101    | Bottleneck| 3, 4, 23, 3          | 101   |
| ResNet152    | Bottleneck| 3, 8, 36, 3          | 152   |

This demonstrates that deeper ResNet variants increase the number of residual blocks, especially in the middle stages.

---

# 26. ResNet50 Feature Dimensions

For a typical (224\times224) input, the spatial dimensions in ResNet50 approximately evolve as:

```
Input
224 × 224 × 3
        ↓
7×7 Conv
112 × 112 × 64
        ↓
Max Pool
56 × 56 × 64
        ↓
Stage 1
56 × 56 × 256
        ↓
Stage 2
28 × 28 × 512
        ↓
Stage 3
14 × 14 × 1024
        ↓
Stage 4
7 × 7 × 2048
        ↓
Global Average Pooling
2048
        ↓
Fully Connected
Classes
```

This illustrates the general relationship:

Spatial Resolution ↓  
Feature Channels  ↑

---

# 27. Global Average Pooling

After the final residual stage, ResNet applies Global Average Pooling.

For ResNet50:

```
7\times7\times2048
```

becomes:

```
2048
```

The operation calculates the average value of each feature map.

For a feature map (A_k):

```
z_k=
\frac{1}{H\times W}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
A_k(i,j)
```

The result is a 2048-dimensional feature vector.

---

# 28. Fully Connected Layer

The final feature vector is passed to a fully connected layer.

For ImageNet-style classification:

```
2048 → 1000
```

The output contains class scores.

For a custom classification task, the final dimension can be changed.

For example, for a 7-class skin lesion classification problem:

```
2048 → 7
```

The architecture can therefore be adapted to different datasets.

---

# 29. Softmax

For multi-class classification, the class scores can be converted into probabilities using Softmax.

For class (i):

\[
P(y=i)=
\frac{e^{z_i}}
{\sum_{j=1}^{C}e^{z_j}}
\]

where:

- (z_i) = score for class (i)
- (C) = number of classes
- (P(y=i)) = predicted probability

The probabilities sum to 1:

\[
\sum_{i=1}^{C}P(y=i)=1
\]

In many PyTorch implementations, Softmax is not explicitly included in the model during training when CrossEntropyLoss is used, because the loss function internally applies the required log-softmax operation.

---

# 30. Complete ResNet50 Pipeline

The complete conceptual architecture is:

- Input Image
- 7×7 Convolution
- Batch Normalization
- ReLU
- 3×3 Max Pooling
- Residual Stage 1
- Residual Stage 2
- Residual Stage 3
- Residual Stage 4
- Global Average Pooling
- 2048-D Feature Vector
- Fully Connected Layer
- Class Scores
- Prediction

---

# 31. Information Flow

The forward information flow can be represented as:

```
Input
  ↓
Initial Feature Extraction
  ↓
Residual Learning
  ↓
Progressively Deeper Representations
  ↓
High-Level Feature Representation
  ↓
Classification
```

At the same time, during backpropagation, gradients can use the shortcut paths:

```
Loss
  ↓
Classification Layer
  ↓
Residual Blocks
  ↓
Shortcut Connections
  ↓
Earlier Layers
```

This is one of the fundamental reasons why ResNet can effectively train very deep networks.

---

# 32. Residual Learning and Gradient Flow

For a residual block:

\[
y=F(x)+x
\]

the derivative with respect to (x) is:

\[
\frac{\partial y}{\partial x}
=
\frac{\partial F(x)}{\partial x}+1
\]

The identity term provides a direct contribution to the gradient.

This does not mean that ResNet completely eliminates all possible gradient problems.

Instead, the architecture provides a more favorable path for gradient propagation and optimization.

---

# 33. ResNet as a Feature Extractor

The classification layer is not the only useful part of ResNet.

The convolutional backbone can be used as a feature extractor:

- Input Image
- ResNet Backbone
- Residual Stages
- Global Average Pooling
- Feature Vector

This feature vector can then be used for:

- Transfer learning
- Classification
- Clustering
- Visualization
- Similarity analysis
- Representation learning
- Anomaly detection

---

# 34. ResNet and Transfer Learning

A pretrained ResNet can be adapted to a new dataset.

For example:

```
Pretrained ResNet50
        ↓
Remove Original Classifier
        ↓
Keep Feature Extractor
        ↓
Add New Classification Head
        ↓
Train on Target Dataset
```

This is particularly useful when the target dataset is smaller than the original training dataset.

Two common approaches are:

- **Feature Extraction:** Freeze the pretrained backbone.
- **Fine-Tuning:** Allow some or all of the pretrained layers to update.

---

# 35. ResNet for Medical Image Classification

ResNet architectures are widely used for image classification tasks, including medical imaging.

For a skin lesion classification task, the pipeline can be represented as:

- Dermoscopic Image
- ResNet50 Backbone
- Hierarchical Feature Extraction
- 2048-D Representation
- Classification Head
- Lesion Class

The network can learn visual representations related to:

- Color
- Texture
- Shape
- Border patterns
- Internal structures
- Complex visual patterns

The learned features are not necessarily direct equivalents of clinical diagnostic criteria; they are distributed representations learned from the training objective.

---

# 36. Advantages of ResNet Architecture

ResNet provides several important advantages.

1. **Easier Optimization:** Residual learning makes deep networks easier to optimize.
2. **Improved Gradient Flow:** Skip connections provide direct paths for information and gradients.
3. **Deep Representation Learning:** The network can learn increasingly complex feature representations.
4. **Reduced Degradation:** Residual connections help address the degradation problem associated with increasing depth.
5. **Transfer Learning:** Pretrained ResNet models can be adapted to many different tasks.
6. **Flexible Architecture:** Different variants provide different depth and computational requirements.

---

# 37. Limitations

Despite its advantages, ResNet also has limitations.

- **Computational Cost:** Deeper variants require more computation.
- **Memory Usage:** Large feature maps and many parameters can require significant GPU memory.
- **Training Time:** Deeper architectures generally require more computational resources.
- **Overfitting:** On small datasets, a very deep model may overfit without appropriate regularization and transfer learning strategies.
- **Interpretability:** Deep learned representations are not inherently easy to interpret.

Therefore, techniques such as Grad-CAM and other explainability methods can be useful.

---

# 38. ResNet Architecture Summary

The overall architecture can be summarized as:

```
RESNET
                    │
        ┌───────────┴───────────┐
        │                       │
   Initial Layers          Residual Learning
        │                       │
   Conv + BN + ReLU        Skip Connections
        │                       │
    Max Pooling             Residual Blocks
        │                       │
        └───────────┬───────────┘
                    ↓
             Residual Stages
                    ↓
        Global Average Pooling
                    ↓
             Feature Vector
                    ↓
          Fully Connected Layer
                    ↓
              Classification
```

---

# 39. Key Takeaways

The most important architectural concepts are:

- **Residual Learning**

\[
\boxed{H(x)=F(x)+x}
\]

- **Skip Connections:** Provide a shortcut path between the input and output of a residual block.
- **Residual Blocks:** The fundamental building blocks of ResNet.
- **Bottleneck Blocks:** Used by ResNet50, ResNet101, and ResNet152 to improve computational efficiency.
- **Downsampling:** Reduces spatial resolution while increasing feature channels.
- **Global Average Pooling:** Transforms the final feature maps into a compact feature vector.
- **Fully Connected Layer:** Maps the learned representation to class scores.
- **ResNet Family:** Includes architectures such as ResNet18, ResNet34, ResNet50, ResNet101, ResNet152.

---

## Conclusion

ResNet is a family of deep convolutional neural networks built around the principle of Residual Learning.

The central architectural idea is the addition of a shortcut connection:

\[
\boxed{y=F(x)+x}
\]

This allows the network to learn residual transformations while maintaining a direct path between earlier and later representations.

ResNet architectures progressively transform an input image from low-level visual features into high-level representations.

The general pipeline is:

- Input Image
- Initial Convolution
- Residual Stages
- Deep Feature Representation
- Global Average Pooling
- Feature Vector
- Fully Connected Layer
- Classification

Different ResNet variants increase depth by using different numbers of residual blocks.

ResNet18 and ResNet34 use Basic Blocks, while ResNet50, ResNet101, and ResNet152 use more computationally efficient Bottleneck Blocks.

The combination of residual learning, skip connections, hierarchical feature extraction, and efficient architectural design makes ResNet one of the most influential architectures in deep learning.
```

