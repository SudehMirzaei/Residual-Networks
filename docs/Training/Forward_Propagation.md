# Forward Propagation

**Forward Propagation**, also called the **Forward Pass**, is the process through which an input is passed through the neural network from the input layer to the output layer.

The purpose of forward propagation is to transform the input into a prediction.

The general process is:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
...
  ↓
Output
```

In a classification problem, the final output represents the model's prediction for the input.

---

1. **Why Forward Propagation Is Important**

A neural network learns a mapping between inputs and outputs.

For example, in image classification:

```
Input Image
     ↓
Neural Network
     ↓
Prediction
```

Suppose the input is a skin lesion image. The model may produce:

```
Melanoma      → 0.72
Nevus         → 0.18
Basal Cell    → 0.06
Other         → 0.04
```

The model uses these values to determine its prediction. Forward propagation is the process that generates these values.

---

2. **Forward Propagation vs Backpropagation**

Forward propagation and backpropagation are two complementary processes.

**Forward Propagation**
- Moves information: Input → Output
- Its purpose is to: 
  > Generate a prediction.

**Backpropagation**
- Moves gradients: Loss → Earlier Layers
- Its purpose is to:
  > Calculate how model parameters should be updated.

The complete training cycle is:

```
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Parameter Update
```

---

3. **Mathematical Definition**

A neural network can be represented as a sequence of functions.

Suppose:

```
x
 ↓
f₁
 ↓
f₂
 ↓
f₃
 ↓
ŷ
```

The final prediction can be written as:

```
ŷ = f₃(f₂(f₁(x)))
```

Each layer receives the output of the previous layer and applies a transformation. Therefore, forward propagation is essentially the sequential application of these transformations.

---

4. **A Simple Neural Network**

Consider a simple network:

```
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Output
```

Mathematically:

```
z₁ = W₁x + b₁
```

Then:

```
a₁ = ReLU(z₁)
```

Then:

```
z₂ = W₂a₁ + b₂
```

The final output is:

```
ŷ = z₂
```

The complete forward pass is:

```
x
 ↓
W₁x + b₁
 ↓
ReLU
 ↓
W₂a₁ + b₂
 ↓
ŷ
```

---

5. **Learnable Parameters**

Neural networks contain parameters that are learned during training. The most common parameters are:

- Weights
- Biases

For a linear layer:

```
z = Wx + b
```

where:

- W = Weight matrix
- x = Input
- b = Bias
- z = Output before activation

During forward propagation, these parameters are used to transform the input. During training, backpropagation calculates gradients for these parameters, and the optimizer updates them.

---

6. **Forward Propagation in a CNN**

For an image classification CNN, the forward pass is more complex. A simplified architecture is:

```
Input Image
    ↓
Convolution
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Pooling
    ↓
Convolutional Blocks
    ↓
Feature Representation
    ↓
Classifier
    ↓
Prediction
```

The network gradually transforms the raw image into a high-level representation.

---

7. **Input Image**

The forward pass begins with an image.

For example:

```
224 × 224 × 3
```

where:

- 224 = Height
- 224 = Width
- 3   = RGB Channels

The image is represented numerically as a tensor.

Conceptually:

```
Image
 ↓
Pixel Values
 ↓
Tensor
```

For example:

Input Shape:

```
224 × 224 × 3
```

---

8. **Initial Convolution in ResNet50**

The original ResNet50 architecture begins with:

```
7 × 7 Convolution
```

with:

- 64 Output Channels
- Stride = 2

The input can be represented as:

```
224 × 224 × 3
```

After the convolution:

```
112 × 112 × 64
```

Conceptually:

```
Input
224 × 224 × 3
       ↓
7×7 Conv
64 Channels
Stride 2
       ↓
112 × 112 × 64
```

The convolution extracts initial visual patterns from the image. These may include:

- Edges
- Corners
- Textures
- Simple Patterns

---

9. **Convolution During Forward Propagation**

A convolutional layer contains learnable kernels.

For example:

```
3 × 3 Kernel
```

The kernel slides across the input. At each location:

```
Input Region
     ×
Kernel Weights
     ↓
Summation
     ↓
One Activation
```

Repeating this operation across the image produces a feature map.

For example:

```
Input
  ↓
3×3 Kernel
  ↓
Feature Map
```

Multiple kernels produce multiple feature maps.

---

10. **Why Multiple Feature Maps Are Produced**

Suppose a convolutional layer has:

```
64 Output Channels
```

This means the layer contains 64 learned filters.

Conceptually:

```
Filter 1  → Feature Map 1
Filter 2  → Feature Map 2
Filter 3  → Feature Map 3
...
Filter 64 → Feature Map 64
```

Therefore:

```
64 Filters
      ↓
64 Feature Maps
```

Each filter can learn to respond to different patterns.

---

11. **Batch Normalization**

After the initial convolution, ResNet applies Batch Normalization.

The sequence is:

```
7×7 Conv
   ↓
BatchNorm
   ↓
ReLU
```

Batch Normalization normalizes and transforms the activations produced by the convolution.

Conceptually:

```
Convolution Output
       ↓
Batch Normalization
       ↓
Normalized Activation
```

This can help stabilize training.

---

12. **ReLU Activation**

After Batch Normalization, ResNet applies ReLU.

The ReLU function is:

```
ReLU(x) = max(0, x)
```

Therefore:

- **Negative Value**
  ↓
  Gradient becomes zero

- **Positive Value**
  ↓
  Unchanged

For example:

```
Input:

[-2, -1, 0, 2, 5]
```

After ReLU:

```
[0, 0, 0, 2, 5]
```

ReLU introduces non-linearity into the network.

---

13. **Max Pooling**

After the initial convolution, BatchNorm, and ReLU, ResNet50 uses:

```
3 × 3 Max Pooling
Stride = 2
```

The spatial resolution changes approximately from:

```
112 × 112 × 64
```

to:

```
56 × 56 × 64
```

Conceptually:

```
112 × 112 × 64
        ↓
3×3 MaxPool
Stride 2
        ↓
56 × 56 × 64
```

Max Pooling reduces spatial resolution while preserving strong local activations.

---

14. **Entering the Residual Stages**

After the initial layers, the representation is:

```
56 × 56 × 64
```

This tensor enters the Residual Stages. ResNet50 contains four main stages:

- Stage 1
- Stage 2
- Stage 3
- Stage 4

Each stage contains multiple Bottleneck Blocks.

ResNet50 uses:

- Stage 1 → 3 Blocks
- Stage 2 → 4 Blocks
- Stage 3 → 6 Blocks
- Stage 4 → 3 Blocks

Total:

```
3 + 4 + 6 + 3 = 16
```

Bottleneck Blocks.

---

15. **Forward Propagation Through a Residual Block**

A residual block has two paths.

**Main Path**

```
Input
  ↓
Convolutional Layers
  ↓
F(x)
```

**Shortcut Path**

```
Input
  ↓
Shortcut
```

The two paths are then added:

```
Shortcut
                    │
                    ↓
Input ───────────────────────┐
  │                          │
  ↓                          │
Main Path                    │
  ↓                          │
F(x)                         │
  │                          │
  └──────────→ Addition ←────┘
                    │
                    ↓
                  ReLU
                    │
                    ↓
                  Output
```

Mathematically:

```
y = F(x) + x
```

when an identity shortcut is used.

---

16. **Bottleneck Block in ResNet50**

ResNet50 uses Bottleneck Blocks.

The main path consists of:

```
1×1 Conv
   ↓
BatchNorm
   ↓
ReLU
   ↓
3×3 Conv
   ↓
BatchNorm
   ↓
ReLU
   ↓
1×1 Conv
   ↓
BatchNorm
```

Then the result is added to the shortcut.

The simplified structure is:

```
Input
  │
  ├──────────────────────────┐
  │                          │
  ↓                          │
1×1 Conv                     │
  ↓                          │
3×3 Conv                     │
  ↓                          │
1×1 Conv                     │
  ↓                          │
BatchNorm
  │
  └──────────→ Add ←─────────┘
                 │
                 ↓
                ReLU
                 │
                 ↓
               Output
```

---

17. **First 1×1 Convolution**

The first 1×1 convolution is mainly used to reduce the number of channels.

For example:

```
56 × 56 × 64
```

may become:

```
56 × 56 × 64
```

in some contexts, or in a standard ResNet50 stage the channel dimensions are configured according to the bottleneck stage.

The important concept is that a 1×1 convolution can control the channel dimension. It performs a learned combination of information across channels at each spatial position.

---

18. **3×3 Convolution**

The 3×3 convolution operates spatially.

It looks at local neighborhoods of the feature maps.

For example:

```
3 × 3 Region
```

contains neighboring spatial locations.

The convolution can learn:

- Edges
- Textures
- Local Shapes
- Spatial Patterns

Therefore:

```
1×1 Conv
   ↓
Channel Transformation

3×3 Conv
   ↓
Spatial Feature Extraction
```

---

19. **Final 1×1 Convolution**

The final 1×1 convolution expands the channel dimension to the desired output dimension of the Bottleneck Block.

For example:

```
56 × 56 × 64
       ↓
1×1 Conv
       ↓
56 × 56 × 256
```

The spatial dimensions remain:

```
56 × 56
```

while the number of channels changes:

```
64 → 256
```

This allows the output of the main path to match the shortcut representation.

---

20. **Element-Wise Addition**

After the main path and shortcut path have compatible dimensions, they are added.

For example:

```
Main Path:

56 × 56 × 256

        +

Shortcut:

56 × 56 × 256
```

After addition:

```
Output:

56 × 56 × 256
```

The addition happens element by element.

Mathematically:

```
y = F(x) + x
```

This operation combines:

- Existing Representation
- Newly Learned Residual Information

---

21. **Feature Hierarchy**

As the input passes through deeper layers, the network learns increasingly complex representations.

A simplified hierarchy is:

```
Input Image
     ↓
Edges
     ↓
Textures
     ↓
Patterns
     ↓
Shapes
     ↓
Complex Structures
     ↓
High-Level Features
```

Early layers focus on simpler visual patterns. Deeper layers combine these patterns into more meaningful representations.

---

22. **Spatial Resolution and Channel Depth**

A key characteristic of CNNs is that spatial resolution generally decreases while the number of channels increases.

For ResNet50, a simplified progression is:

```
Input
224 × 224 × 3
       ↓
Conv1
112 × 112 × 64
       ↓
MaxPool
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
```

This is a conceptual representation of the standard ResNet50 architecture.

---

23. **Why Does Spatial Resolution Decrease?**

Reducing spatial resolution provides several benefits.

- **Computational Efficiency**  
  Smaller feature maps require fewer operations.

- **Larger Effective Receptive Fields**  
  As the network becomes deeper, each activation corresponds to a larger region of the original image.

- **Hierarchical Representation**  
  The network gradually moves from detailed local information toward higher-level semantic information.

---

24. **Why Does the Number of Channels Increase?**

As spatial resolution decreases, the network can increase the number of feature channels.

For example:

```
56 × 56 × 256
        ↓
28 × 28 × 512
```

Although the spatial dimensions become smaller, the representation can contain more feature channels. Each channel can represent a different learned feature.

Conceptually:

```
More Spatial Detail
       ↓
Fewer Channels

Higher-Level Representation
       ↓
More Feature Channels
```

---

25. **Global Average Pooling**

After the final residual stage, ResNet50 produces approximately:

```
7 × 7 × 2048
```

The network then applies:

> Global Average Pooling

This converts each feature map into a single value. For one feature map:

```
7 × 7
```

the average of all 49 values is calculated.

Therefore:

```
7 × 7 × 2048
       ↓
Global Average Pooling
       ↓
1 × 1 × 2048
```

This can be represented as a feature vector:

```
2048-dimensional vector
```

---

26. **Why Global Average Pooling?**

Global Average Pooling:

- Reduces spatial dimensions
- Reduces the number of parameters
- Produces a compact representation
- Summarizes the activation of each feature map

Conceptually:

```
Feature Map
7 × 7
  ↓
Average
  ↓
One Value
```

This operation is performed for all 2048 feature maps.

---

27. **Feature Vector**

After Global Average Pooling, ResNet50 produces a feature vector:

```
[ f₁, f₂, f₃, ..., f₂₀₄₈ ]
```

This vector represents the image in a learned feature space.

Conceptually:

```
Image
  ↓
Convolutional Backbone
  ↓
High-Level Features
  ↓
2048-D Feature Vector
```

This representation can also be used for:

- Feature extraction
- Visualization
- Clustering
- Transfer Learning
- Classification
- Similarity analysis

---

28. **Fully Connected Layer**

The feature vector is passed to a Fully Connected layer.

For example, if the task contains 7 classes:

```
2048 Features
      ↓
Fully Connected Layer
      ↓
7 Outputs
```

The fully connected layer computes:

```
z = Wx + b
```

where:

- x = 2048-dimensional feature vector
- W = Learnable weights
- b = Bias
- z = Class Scores

---

29. **Logits**

The output of the final fully connected layer is commonly called logits.

For a 7-class classification problem:

```
Logits:

[2.4, 0.8, -1.2, 3.1, 0.4, -0.7, 1.2]
```

These values are not probabilities yet. They are raw class scores.

---

30. **Softmax**

For multi-class classification, Softmax can convert logits into probabilities.

The Softmax function is:

```
P(y=i) = eᶻⁱ / Σⱼ eᶻʲ
```

where:

- zᵢ = Logit for class i
- P(y=i) = Probability of class i

After Softmax:

```
Class 1 → 0.08
Class 2 → 0.03
Class 3 → 0.01
Class 4 → 0.72
Class 5 → 0.04
Class 6 → 0.02
Class 7 → 0.10
```

The probabilities sum to approximately:

```
1.0
```

---

31. **Softmax in PyTorch**

When using:

```
nn.CrossEntropyLoss()
```

you generally should not explicitly apply Softmax in the model's final layer. The model should return logits:

```python
outputs = model(inputs)
```

and then:

```python
loss = criterion(outputs, labels)
```

where:

- criterion = nn.CrossEntropyLoss()

CrossEntropyLoss internally applies the appropriate log-softmax operation. Therefore, this is generally preferred:

```
Linear
  ↓
Logits
  ↓
CrossEntropyLoss
```

rather than:

```
Linear
  ↓
Softmax
  ↓
CrossEntropyLoss
```

---

32. **Complete ResNet50 Forward Pass**

The complete conceptual forward pass can be represented as:

```
Input Image
224 × 224 × 3
       ↓
7×7 Conv
       ↓
112 × 112 × 64
       ↓
BatchNorm
       ↓
ReLU
       ↓
3×3 MaxPool
       ↓
56 × 56 × 64
       ↓
Residual Stage 1
       ↓
56 × 56 × 256
       ↓
Residual Stage 2
       ↓
28 × 28 × 512
       ↓
Residual Stage 3
       ↓
14 × 14 × 1024
       ↓
Residual Stage 4
       ↓
7 × 7 × 2048
       ↓
Global Average Pooling
       ↓
2048-D Feature Vector
       ↓
Fully Connected Layer
       ↓
Class Logits
       ↓
Prediction
```

---

33. **Forward Propagation During Training**

During training, forward propagation is only the first part of the process.

The complete training cycle is:

```
Input
                   ↓
          Forward Propagation
                   ↓
              Prediction
                   ↓
             Loss Function
                   ↓
                 Loss
                   ↓
           Backpropagation
                   ↓
               Gradients
                   ↓
               Optimizer
                   ↓
          Updated Parameters
```

The process repeats for many batches and epochs.

---

34. **Forward Propagation During Inference**

During inference, the model only needs to perform the forward pass.

There is no need to calculate gradients or update parameters.

The process becomes:

```
Input
  ↓
Forward Propagation
  ↓
Prediction
```

For example:

```python
model.eval()

with torch.no_grad():
    outputs = model(image)
```

The model generates predictions without performing parameter updates.

---

35. **Training Mode vs Evaluation Mode**

In PyTorch:

```python
model.train()
```

sets the model to training mode.

This affects layers such as:

- Batch Normalization
- Dropout

For evaluation:

```python
model.eval()
```

puts the model into evaluation mode. For inference, it is common to use:

```python
with torch.no_grad():
```

to disable gradient tracking.

---

36. **Forward Propagation and Feature Extraction**

The forward pass through the ResNet50 backbone can be used to extract representations.

Conceptually:

```
Input Image
     ↓
ResNet50 Backbone
     ↓
Global Average Pooling
     ↓
2048-D Feature Vector
```

Instead of using the final classifier, the feature vector can be stored.

For example:

```
Image
  ↓
ResNet50
  ↓
2048-dimensional embedding
```

These embeddings can then be used for:

- UMAP
- t-SNE
- Clustering
- Similarity Analysis
- Visualization
- Downstream Classification

---

37. **Forward Propagation and Representation Learning**

Forward propagation is also the mechanism through which a neural network creates representations.

At each stage:

```
Raw Input
   ↓
Representation 1
   ↓
Representation 2
   ↓
Representation 3
   ↓
High-Level Representation
```

The network gradually transforms the raw image into a representation that is useful for the target task. This is a central concept in Representation Learning.

---

38. **Forward Propagation and Residual Learning**

In ResNet, forward propagation through a residual block follows:

```
Input x
  │
  ├─────────────────────┐
  │                     │
  ↓                     │
Residual Function F(x)  │
  │                     │
  └────────→ Add ←──────┘
                │
                ↓
             F(x) + x
                │
                ↓
               ReLU
```

The residual block therefore does not simply replace the input. It combines:

- Original Representation
- Learned Residual Information

---

39. **Why Skip Connections Matter During Forward Propagation**

Skip connections are important during forward propagation because they allow information to bypass the transformation layers.

Without a shortcut:

```
Input
  ↓
Conv
  ↓
Conv
  ↓
Output
```

With a shortcut:

```
┌─────────────┐
                 │             │
Input ───────────┤             │
  │              │             │
  ↓              │             │
Conv             │             │
  ↓              │             │
Conv             │             │
  │              │             │
  └──────────────┴────→ Add
                         │
                         ↓
                       Output
```

Therefore, useful information can be preserved while new information is learned.

---

40. **Key Takeaways**

- Forward Propagation is the process of passing an input through a neural network to generate a prediction.
- It moves information from the input toward the output.
- Each layer transforms the representation produced by the previous layer.
- In CNNs, convolutional layers extract spatial features.
- Multiple convolutional filters produce multiple feature maps.
- Batch Normalization transforms and stabilizes activations.
- ReLU introduces non-linearity.
- Pooling reduces spatial resolution.
- ResNet uses residual blocks to transform and preserve representations.
- ResNet50 uses 16 Bottleneck Blocks distributed across four residual stages.
- Global Average Pooling converts the final feature maps into a compact feature vector.
- The final Fully Connected layer produces class logits.
- Softmax can convert logits into probabilities.
- When using CrossEntropyLoss, the model should normally output logits rather than applying Softmax explicitly.
- During training, forward propagation is followed by loss calculation and backpropagation.
- During inference, only the forward pass is required.
- The forward pass through the ResNet50 backbone can also be used to extract high-level feature representations.
```

