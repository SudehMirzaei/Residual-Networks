# Backpropagation

**Backpropagation**, short for **Backward Propagation of Errors**, is the fundamental algorithm used to calculate gradients in neural networks.

The purpose of backpropagation is to determine:

> How much should each parameter of the neural network change in order to reduce the loss?

Backpropagation works together with an optimization algorithm such as **Gradient Descent** or **Adam**.

The overall training process can be summarized as:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Parameter Update
  ↓
Repeat
```

---

1. **Why Do We Need Backpropagation?**

A neural network contains many learnable parameters.

For example, a convolutional neural network may contain:

- Millions of weights
- Thousands of biases

The network needs to determine the values of these parameters so that its predictions become more accurate.

Suppose the network produces:

```
Prediction = 0.7
```

while the correct target is:

```
Target = 1.0
```

The network calculates a loss that measures the difference between the prediction and the target. The next question is:

> Which weights caused this error, and in which direction should they change?

Backpropagation answers this question by calculating gradients.

---

2. **The Basic Training Loop**

Training a neural network consists of repeated iterations.

```
┌──────────────────────┐
                │                      │
                ↓                      │
Input → Forward Pass → Loss → Backward Pass
                                  ↓
                              Gradients
                                  ↓
                              Optimizer
                                  ↓
                           Update Weights
                                  │
                                  └─────────→
```

More explicitly:

1. Forward Propagation
2. Calculate Loss
3. Backpropagation
4. Calculate Gradients
5. Update Parameters
6. Repeat

---

3. **Forward Propagation**

Before backpropagation can happen, the network must perform a forward pass.

Suppose a simple neural network is:

```
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
```

During forward propagation:

```
x
↓
f₁(x)
↓
f₂(f₁(x))
↓
ŷ
```

where:

- x = Input
- ŷ = Prediction

The network transforms the input through multiple layers until it produces an output.

---

4. **Loss Function**

After obtaining the prediction, we compare it with the true target.

For example:

```
Prediction = ŷ
Target     = y
```

The loss function measures the error.

For a simple squared error:

```
L = (y - ŷ)²
```

For classification problems, a common choice is:

**Cross-Entropy Loss**

The loss can be represented as:

```
Prediction
    ↓
Compare with Target
    ↓
Loss
```

The objective of training is:

> Minimize Loss

---

5. **What Is a Gradient?**

A gradient tells us how sensitive the loss is to a parameter.

For a parameter `w`:

```
∂L / ∂w
```

means:

> How much does the loss change when `w` changes?

If:

```
∂L / ∂w > 0
```

increasing `w` tends to increase the loss locally. 

If:

```
∂L / ∂w < 0
```

increasing `w` tends to decrease the loss locally.

The magnitude also matters.

A large magnitude means:

```
Large gradient
↓
Parameter strongly affects loss
```

A small magnitude means:

```
Small gradient
↓
Parameter has a weaker local effect on loss
```

---

6. **Backpropagation**

Backpropagation calculates these gradients by propagating the error backward through the network.

Consider:

```
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
  ↓
Loss
```

The backward pass goes in the opposite direction:

```
Loss
  ↓
Layer 3
  ↓
Layer 2
  ↓
Layer 1
```

At each layer, the algorithm calculates how the loss depends on that layer's parameters.

---

7. **The Chain Rule**

The mathematical foundation of backpropagation is the Chain Rule from calculus.

Suppose:

```
x → f → g → L
```

Then:

```
y = f(x)
z = g(y)
L = L(z)
```

To calculate how the loss changes with respect to `x`:

```
∂L/∂x
```

we apply the chain rule:

```
∂L/∂x
=
∂L/∂z
×
∂z/∂y
×
∂y/∂x
```

This allows the gradient to be propagated backward through the computational graph.

---

8. **Simple Example of the Chain Rule**

Consider:

```
x = 2
y = 3x
z = y²
```

We want to calculate:

```
∂z/∂x
```

First:

```
∂z/∂y = 2y
```

Since:

```
y = 3x
```

and:

```
x = 2
```

we have:

```
y = 6
```

Therefore:

```
∂z/∂y = 12
```

Also:

```
∂y/∂x = 3
```

Using the chain rule:

```
∂z/∂x
=
∂z/∂y × ∂y/∂x

= 12 × 3

= 36
```

This is essentially the same principle used by backpropagation in large neural networks.

---

9. **Backpropagation Through a Neural Network**

Consider:

```
x
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
ŷ
 ↓
L
```

The gradient flows backward:

```
L
 ↓
∂L/∂Layer3
 ↓
∂L/∂Layer2
 ↓
∂L/∂Layer1
```

Each layer uses the gradient received from the next layer and computes the gradients required for its own parameters.

---

10. **A Simple Fully Connected Layer**

A fully connected layer can be represented as:

```
z = Wx + b
```

where:

- x = Input
- W = Weight matrix
- b = Bias
- z = Output

Suppose the next layers produce a loss:

```
L
```

Backpropagation calculates:

```
∂L/∂W
```

and:

```
∂L/∂b
```

These gradients tell the optimizer how the weights and biases should change.

---

11. **Parameter Update**

Once gradients have been calculated, an optimizer updates the parameters.

The simplest update rule is Gradient Descent:

```
w_new = w_old - η × ∂L/∂w
```

where:

- w = Parameter
- η = Learning Rate
- ∂L/∂w = Gradient

For example:

```
Current weight = 0.8
Gradient       = 0.2
Learning rate  = 0.1
```

Then:

```
New weight
=
0.8 - (0.1 × 0.2)

=
0.78
```

The weight moves in the direction that locally reduces the loss.

---

12. **Backpropagation vs Gradient Descent**

These two concepts are related but not identical.

**Backpropagation**

Calculates:

- Gradients

**Gradient Descent**

Uses those gradients to:

- Update Parameters

Therefore:

```
Backpropagation
       ↓
Calculate Gradients
       ↓
Optimizer
       ↓
Update Parameters
```

The optimizer does not need to be simple Gradient Descent. It can be:

- SGD
- Adam
- AdamW
- RMSprop

---

13. **Backpropagation in a CNN**

In a CNN, the network contains operations such as:

- Convolution
- Batch Normalization
- ReLU
- Pooling
- Fully Connected Layer

During the forward pass:

```
Input Image
    ↓
Convolution
    ↓
BatchNorm
    ↓
ReLU
    ↓
Pooling
    ↓
...
    ↓
Prediction
    ↓
Loss
```

During backpropagation, gradients travel backward through these operations.

```
Loss
  ↓
Classifier
  ↓
Global Average Pooling
  ↓
Residual Stages
  ↓
Pooling
  ↓
ReLU
  ↓
BatchNorm
  ↓
Convolution
  ↓
Input
```

---

14. **Backpropagation Through Convolution**

A convolution layer contains learnable kernels.

For example:

```
3 × 3 Kernel
```

may contain:

- 9 learnable weights

For multiple input and output channels, the number of parameters can become much larger.

During backpropagation, the network calculates how the loss depends on these kernel weights.

Conceptually:

```
Input
  ↓
Convolution
  ↓
Feature Map
  ↓
Loss
```

Backward:

```
Loss
  ↓
Gradient of Feature Map
  ↓
Gradient of Convolution
  ↓
Gradient of Kernel Weights
```

The convolutional filters are then updated by the optimizer.

---

15. **Backpropagation Through ReLU**

The ReLU function is:

```
ReLU(x) = max(0, x)
```

Its derivative is approximately:

```
ReLU'(x) = 1   if x > 0
           0   if x < 0
```

Therefore, during backpropagation:

- **Positive Activation**
  ↓
  Gradient can pass

- **Negative Activation**
  ↓
  Gradient becomes zero

This is one reason activation functions influence gradient flow.

---

16. **Backpropagation Through Batch Normalization**

Batch Normalization also participates in the computational graph.

During the forward pass:

```
Input
  ↓
Normalize
  ↓
Scale and Shift
  ↓
Output
```

During backpropagation, gradients are propagated through these operations. The learnable parameters of BatchNorm are also updated. These parameters are commonly represented as:

- γ = Scale
- β = Shift

---

17. **Backpropagation Through Pooling**

Pooling layers do not contain learnable weights in the same way convolutional layers do.

For Max Pooling:

```
3 × 3 Region

[ 1   3   2 ]
[ 4   9   5 ]
[ 2   6   1 ]
```

The maximum value is:

```
9
```

During the backward pass, the gradient is propagated primarily through the location that produced the maximum activation.

Conceptually:

**Forward:**

```
3×3 Region
   ↓
Maximum = 9
```

**Backward:**

```
Gradient
   ↓
Position of Maximum
```

---

18. **Backpropagation in ResNet**

ResNet introduces an important modification to the computational graph.

A residual block computes:

```
y = F(x) + x
```

where:

- F(x) is the residual transformation.

The forward path is:

```
Input x
  │
  ├──────────────────────┐
  │                      │
  ↓                      │
Residual Layers          │
  ↓                      │
F(x)                     │
  │                      │
  └──────────→ Addition ←┘
                    │
                    ↓
                    y
```

---

19. **Backpropagation Through a Residual Block**

For:

```
y = F(x) + x
```

the derivative is:

```
∂y/∂x = ∂F(x)/∂x + 1
```

During backpropagation, the gradient entering the addition operation is distributed through both branches.

Conceptually:

```
Gradient
                 │
                 ↓
              Addition
              ↙       ↘
             ↓         ↓
      Residual Path   Shortcut
             ↓         ↓
          Earlier    Earlier
           Layers     Layers
```

The shortcut therefore provides a direct gradient path.

---

20. **Why Skip Connections Help Backpropagation**

In a conventional deep network:

```
Gradient
   ↓
Layer
   ↓
Layer
   ↓
Layer
   ↓
Layer
   ↓
Earlier Layer
```

The gradient must pass through many transformations.

In ResNet:

```
Gradient
   │
   ├────────────────→ Shortcut
   │
   ↓
Residual Layers
   │
   └──────────────→ Earlier Layers
```

This creates a shorter path for gradient propagation.

As a result, the network becomes easier to optimize.

---

21. **Backpropagation and Vanishing Gradients**

The gradient through a deep sequence is based on repeated applications of the chain rule:

```
∂L/∂x = ∂L/∂xₙ × ∂xₙ/∂xₙ₋₁ × ... × ∂x₁/∂x
```

If many terms are smaller than 1:

```
0.8 × 0.7 × 0.6 × ...
```

the resulting gradient can become very small. This creates the Vanishing Gradient Problem. ResNet reduces this difficulty by introducing shortcut paths.

---

22. **Backpropagation and Exploding Gradients**

The opposite can happen if the derivatives are repeatedly large.

For example:

```
1.5 × 1.5 × 1.5 × ...
```

The gradient can grow rapidly.

This creates:

> Exploding Gradients

Possible consequences include:

- Very large parameter updates
- Unstable training
- Loss divergence
- Numerical overflow

---

23. **Backpropagation Through ResNet50**

ResNet50 contains:

- Stage 1 → 3 Bottleneck Blocks
- Stage 2 → 4 Bottleneck Blocks
- Stage 3 → 6 Bottleneck Blocks
- Stage 4 → 3 Bottleneck Blocks

Therefore:

```
3 + 4 + 6 + 3 = 16
```

Bottleneck Blocks are used. Each block contains a main path and a shortcut path.

Conceptually:

```
Input
  ↓
Bottleneck 1
  ↓
Bottleneck 2
  ↓
Bottleneck 3
  ↓
...
  ↓
Bottleneck 16
  ↓
Global Average Pooling
  ↓
Fully Connected Layer
  ↓
Loss
```

During backpropagation, the gradient travels backward through this entire computational graph. The shortcut connections provide additional paths through the residual blocks.

---

24. **Backpropagation in a ResNet50 Bottleneck**

A Bottleneck Block can be represented as:

```
Input
  │
  ├─────────────────────────────┐
  │                             │
  ↓                             │
1×1 Conv                        │
  ↓                             │
BatchNorm                       │
  ↓                             │
ReLU                            │
  ↓                             │
3×3 Conv                        │
  ↓                             │
BatchNorm                       │
  ↓                             │
ReLU                            │
  ↓                             │
1×1 Conv                        │
  ↓                             │
BatchNorm                       │
  │                             │
  └──────────────→ Addition ←───┘
                         │
                         ↓
                       ReLU
```

During the backward pass:

```
Gradient
    ↓
   ReLU
    ↓
 Addition
   ↙   ↘
  ↓     ↓
Main   Shortcut
Path     Path
  ↓       ↓
Earlier Layers
```

---

25. **Backpropagation and Learning**

Backpropagation itself does not "learn" the weights. It calculates the information required for learning.

The complete process is:

```
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
Weight Update
```

The optimizer performs the actual parameter update.

---

26. **Backpropagation in PyTorch**

In PyTorch, the backward pass is usually initiated with:

```python
loss.backward()
```

For example:

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

The sequence is:

```
optimizer.zero_grad()
        ↓
Forward Pass
        ↓
Calculate Loss
        ↓
loss.backward()
        ↓
Calculate Gradients
        ↓
optimizer.step()
        ↓
Update Parameters
```

---

27. **What Does loss.backward() Do?**

When we call:

```
loss.backward()
```

PyTorch uses its automatic differentiation system to traverse the computational graph backward.

It calculates gradients for parameters that have:

```
requires_grad=True
```

For example:

```
model.layer4[0].conv1.weight.grad
```

contains the gradient of the loss with respect to that parameter.

Mathematically:

```
∂L / ∂W
```

---

28. **What Does optimizer.step() Do?**

After:

```
loss.backward()
```

the gradients are available.

Then:

```
optimizer.step()
```

uses those gradients to update the parameters.

For a simplified SGD optimizer:

```
W_new = W_old - η × ∂L/∂W
```

Thus:

```
loss.backward()
```

calculates gradients, while:

```
optimizer.step()
```

uses them to update the weights.

---

29. **Why Do We Use optimizer.zero_grad()?**

Gradients in PyTorch accumulate by default. Therefore, before calculating gradients for the next iteration, we usually reset them:

```
optimizer.zero_grad()
```

The standard training loop is:

```python
for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

The order is important.

---

30. **Computational Graph**

Backpropagation operates on a computational graph.

For a simple operation:

```
x
↓
Multiply
↓
y
↓
Loss
```

the graph records the operations used to produce the output.

For a neural network:

```
Input
  ↓
Conv
  ↓
BatchNorm
  ↓
ReLU
  ↓
Residual Block
  ↓
Classifier
  ↓
Loss
```

Autograd uses this graph to calculate derivatives during the backward pass.

---

31. **Automatic Differentiation**

Modern deep learning frameworks such as PyTorch use Automatic Differentiation.

The developer does not need to manually calculate every derivative. Instead:

```
loss.backward()
```

automatically calculates the required gradients.

This is especially important for architectures such as ResNet, which contain millions of parameters and complex computational graphs.

---

32. **Backpropagation vs Forward Propagation**

| Property               | Forward Propagation   | Backpropagation          |
|-----------------------|-----------------------|---------------------------|
| Direction              | Input → Output        | Loss → Earlier Layers     |
| Purpose                | Generate Prediction    | Calculate Gradients       |
| Uses                   | Model Parameters       | Chain Rule                |
| Produces               | Prediction            | Gradients                 |
| Main Goal              | Compute Output        | Determine Parameter Updates|

The two processes work together.

```
Forward
   ↓
Prediction
   ↓
Loss
   ↓
Backward
   ↓
Gradients
```

---

33. **Backpropagation vs Gradient Flow**

These concepts are related but different.

**Backpropagation**  
The algorithmic process used to calculate gradients.

**Gradient Flow**  
The way those gradients propagate through the network.

Therefore:

```
Backpropagation
      ↓
Gradients propagate backward
      ↓
Gradient Flow
```

Problems in gradient flow can affect the effectiveness of backpropagation.

---

34. **Backpropagation and ResNet's Main Advantage**

One of the most important characteristics of ResNet is that its architecture changes the computational graph.

Instead of forcing the gradient to pass through every transformation:

```
Loss
 ↓
Layer
 ↓
Layer
 ↓
Layer
 ↓
Layer
```

ResNet provides shortcut paths:

```
Loss
 │
 ├──────────────→ Shortcut
 │
 ↓
Residual Layers
 │
 └──────────────→ Earlier Layers
```

This contributes to improved gradient propagation in very deep networks.

---

35. **Complete Training Process**

A complete ResNet training iteration can be represented as:

```
Input Image
                      │
                      ↓
              Forward Propagation
                      │
                      ↓
                ResNet50 Model
                      │
                      ↓
                  Prediction
                      │
                      ↓
                Loss Function
                      │
                      ↓
                Loss Value
                      │
                      ↓
                Backpropagation
                      │
                      ↓
                  Gradients
                      │
                      ↓
                  Optimizer
                      │
                      ↓
              Updated Parameters
                      │
                      └──────────────→ Next Iteration
```

---

36. **Key Takeaways**

- Backpropagation calculates gradients of the loss with respect to model parameters.
- It is based on the Chain Rule of calculus.
- Backpropagation works in the opposite direction of forward propagation.
- The output of backpropagation is a set of gradients.
- An optimizer uses these gradients to update model parameters.
- `loss.backward()` performs the backward pass in PyTorch.
- `optimizer.step()` updates the model parameters.
- `optimizer.zero_grad()` clears accumulated gradients.
- CNN convolutional filters are updated using gradients calculated during backpropagation.
- ReLU, BatchNorm, Pooling, and Convolution all participate in the computational graph.
- ResNet's Skip Connections provide additional paths for gradient propagation.
- The residual formulation:
  
  ```
  y = F(x) + x
  ```

  creates a direct identity contribution to the gradient:

  ```
  ∂y/∂x = ∂F(x)/∂x + 1
  ```

  This helps make optimization of very deep networks more stable.
  
- Backpropagation calculates the learning signal; the optimizer uses that signal to update the model.
```

