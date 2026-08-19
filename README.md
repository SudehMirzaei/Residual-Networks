# Residual Network

A structured study and documentation of **Residual Networks (ResNet)**, with a particular focus on **ResNet50**, residual learning, skip connections, gradient flow, feature extraction, and transfer learning.

This repository explains the concepts behind ResNet step by step, from the foundations of deep neural networks to the internal architecture and training mechanisms of residual networks.

---

## 📚 Repository Overview

The documentation is organized into six main sections:

```text
docs/
│
├── Foundations/
│   ├── Deep_Neural_Network.md
│   └── Vanishing_Gradient.md
│
├── Representation Learning/
│   └── Feature_Extraction.md
│
├── ResNet Architecture/
│   └── ResNet_Architecture.md
│
├── Residual Learning/
│   ├── Bottleneck_Block.md
│   └── Skip_Connection.md
│
├── Training/
│   ├── BackPropagation.md
│   ├── Forward_Propagation.md
│   └── Gradient_Flow.md
│
└── Transfer Learning/
    └── Transfer_Learning.md
```

---

📖 Documentation

1. **Foundations**

   This section introduces the fundamental concepts required to understand why very deep neural networks can become difficult to train and why architectures such as ResNet were introduced.

   **Deep Neural Networks**

   An introduction to deep neural networks, their layered structure, increasing depth, and the challenges associated with training deeper architectures.

   📄 [Deep Neural Networks](docs/Foundations/Deep_Neural_Network.md)

   ---

   **Vanishing Gradient**

   Explains the Vanishing Gradient Problem, why gradients can become extremely small in deep neural networks, and how this affects learning and optimization.

   📄 [Vanishing Gradient](docs/Foundations/Vanishing_Gradient.md)

   ---

2. **Representation Learning**

   This section focuses on how neural networks automatically learn meaningful representations from raw input data.

   **Feature Extraction**

   Explains how CNNs extract hierarchical visual features and how ResNet50 can be used as a feature extractor to obtain high-level image representations.

   📄 [Feature Extraction](docs/Representation%20Learning/Feature_Extraction.md)

   ---

3. **ResNet Architecture**

   This section explains the overall architecture of Residual Networks, with a particular focus on ResNet50.

   **ResNet Architecture**

   Covers the complete ResNet50 architecture, including:

   - Initial convolution
   - Batch Normalization
   - ReLU
   - Max Pooling
   - Residual Stages
   - Bottleneck Blocks
   - Global Average Pooling
   - Fully Connected Layer
   - Classification output

   📄 [ResNet Architecture](docs/ResNet%20Architecture/ResNet_Architecture.md)

   ---

4. **Residual Learning**

   This section focuses on the core idea behind ResNet: Residual Learning.

   Instead of directly learning a desired mapping:

   H(x)

   ResNet learns a residual function:

   F(x) = H(x) - x

   and reconstructs the desired mapping as:

   H(x) = F(x) + x

   This is implemented using Skip Connections.

   ---

   **Bottleneck Block**

   Explains the Bottleneck architecture used by ResNet50:

   1×1 Conv
   ↓
   3×3 Conv
   ↓
   1×1 Conv

   The document explains the purpose of each convolution and how the bottleneck reduces computational cost while allowing deeper networks.

   📄 [Bottleneck Block](docs/Residual%20Learning/Bottleneck_Block.md)

   ---

   **Skip Connection**

   Explains how shortcut connections bypass one or more layers and combine the original representation with the learned residual.

   The fundamental operation is:

   y = F(x) + x

   📄 [Skip Connection](docs/Residual%20Learning/Skip_Connection.md)

   ---

5. **Training**

   This section explains how ResNet is trained and how information and gradients move through the network.

   ---

   **Forward Propagation**

   Explains how an input image moves through ResNet50 from the initial convolution to the final classification output.

   The forward process can be summarized as:

   Input
   ↓
   Convolution
   ↓
   Residual Stages
   ↓
   Global Average Pooling
   ↓
   Fully Connected Layer
   ↓
   Prediction

   📄 [Forward Propagation](docs/Training/Forward_Propagation.md)

   ---

   **Backpropagation**

   Explains how gradients are calculated using the chain rule and propagated backward through the neural network.

   It also covers:

   - Gradient calculation
   - Loss functions
   - Parameter updates
   - Optimizers
   - Backpropagation through CNNs
   - Backpropagation through residual blocks

   📄 [Backpropagation](docs/Training/BackPropagation.md)

   ---

   **Gradient Flow**

   Explains how gradients propagate through deep neural networks and why maintaining effective gradient flow is important for training deep architectures.

   It also discusses how Skip Connections help provide shorter paths for gradient propagation.

   📄 [Gradient Flow](docs/Training/Gradient_Flow.md)

   ---

6. **Transfer Learning**

   This section explains how pretrained neural networks such as ResNet50 can be adapted to new datasets and tasks.

   **Transfer Learning**

   Covers:

   - Pretrained models
   - Feature extraction
   - Freezing layers
   - Fine-tuning
   - Replacing the classification head
   - Using ImageNet-pretrained ResNet50 for new classification tasks

   📄 [Transfer Learning](docs/Transfer%20Learning/Transfer_Learning.md)

   ---

🧠 **ResNet50 at a Glance**

ResNet50 is a deep convolutional neural network based on residual learning.

Its simplified architecture is:

Input Image: 224 × 224 × 3
       ↓
   7×7 Convolution
       ↓
   Batch Normalization
       ↓
   ReLU
       ↓
   3×3 Max Pooling
       ↓
   Residual Stage 1
       ↓
   Residual Stage 2
       ↓
   Residual Stage 3
       ↓
   Residual Stage 4
       ↓
   Global Average Pooling
       ↓
   2048-D Feature Vector
       ↓
   Fully Connected Layer
       ↓
   Classification

The four residual stages contain:

- Stage 1 → 3 Bottleneck Blocks
- Stage 2 → 4 Bottleneck Blocks
- Stage 3 → 6 Bottleneck Blocks
- Stage 4 → 3 Bottleneck Blocks

Therefore:

3 + 4 + 6 + 3 = 16

Bottleneck Blocks are used.

---

🔗 **Core Concepts**

The main concepts covered in this repository are:

- Deep Neural Networks
  ↓
- Vanishing Gradients
  ↓
- Residual Learning
  ↓
- Skip Connections
  ↓
- Bottleneck Blocks
  ↓
- ResNet Architecture
  ↓
- Gradient Flow
  ↓
- Feature Extraction
  ↓
- Transfer Learning

These concepts are interconnected and explain why ResNet can successfully train very deep convolutional neural networks.

---

🎯 **Learning Objectives**

This repository aims to provide a structured understanding of:

- Why deep neural networks become difficult to optimize
- What causes vanishing gradients
- How residual learning works
- How skip connections improve gradient propagation
- How Bottleneck Blocks reduce computational complexity
- How ResNet50 is structured
- How forward propagation works through ResNet50
- How backpropagation works through residual blocks
- How gradients flow through deep networks
- How ResNet can be used for feature extraction
- How pretrained ResNet models can be used through transfer learning

---

🏗️ **ResNet50 Residual Block**

The core idea of a residual block can be summarized as:

```
┌──────────────────────┐
│                      │
│      Skip Path       │
│                      │
Input ───────────┤──────────────────────┤
  │              │                      │
  ↓              │                      │
1×1 Conv         │                      │
  ↓              │                      │
3×3 Conv         │                      │
  ↓              │                      │
1×1 Conv         │                      │
  │              │                      │
  └──────────────┴──────→ Addition ←────┘
                              │
                              ↓
                             ReLU
                              │
                              ↓
                            Output
```

Mathematically:

y = F(x) + x

This formulation is the foundation of residual learning.

---

🛠️ **Technologies and Concepts**

This repository is primarily educational and focuses on deep learning concepts related to:

- Deep Neural Networks
- Convolutional Neural Networks
- Residual Networks
- ResNet50
- Residual Learning
- Skip Connections
- Bottleneck Blocks
- Backpropagation
- Gradient Flow
- Feature Extraction
- Transfer Learning


---

📚 **Documentation Map**

| Section                  | Topic                     | Documentation                                    |
|-------------------------|--------------------------|-------------------------------------------------|
| Foundations             | Deep Neural Networks     | Read                                            |
| Foundations             | Vanishing Gradient       | Read                                            |
| Representation Learning  | Feature Extraction       | Read                                            |
| ResNet Architecture      | ResNet Architecture      | Read                                            |
| Residual Learning        | Bottleneck Block         | Read                                            |
| Residual Learning        | Skip Connection          | Read                                            |
| Training                | Forward Propagation      | Read                                            |
| Training                | Backpropagation          | Read                                            |
| Training                | Gradient Flow            | Read                                            |
| Transfer Learning        | Transfer Learning        | Read                                            |

---

🚀 **Suggested Learning Path**

For a structured understanding of ResNet, the documentation can be studied in the following order:

- **Step 1 — Foundations**
  1. Deep Neural Networks
  2. Vanishing Gradient

- **Step 2 — Training Fundamentals**
  3. Forward Propagation
  4. Backpropagation
  5. Gradient Flow

- **Step 3 — Residual Learning**
  6. Skip Connection
  7. Bottleneck Block

- **Step 4 — ResNet Architecture**
  8. ResNet Architecture

- **Step 5 — Representation Learning**
  9. Feature Extraction

- **Step 6 — Transfer Learning**
  10. Transfer Learning

---

📌 **Summary**

The central idea behind this repository can be summarized as:

Deep Networks  
     ↓  
Optimization Difficulties  
     ↓  
Vanishing Gradients  
     ↓  
Residual Learning  
     ↓  
Skip Connections  
     ↓  
ResNet  
     ↓  
Deep and Effective Feature Learning  

ResNet demonstrates that increasing network depth does not necessarily have to make optimization harder when the architecture is designed to provide effective residual learning and gradient propagation.

