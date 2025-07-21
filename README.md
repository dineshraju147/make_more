# Make More: Character-Level Language Model

## 1. Introduction

**Make More** is an educational project inspired by Andrej Karpathy's "makemore." It aims to teach the fundamentals of language modeling through character-level modeling. This step-by-step project evolves from simple statistical bigram models to deeper modular neural networks. The ultimate goal is to provide intuition and implementation experience that helps bridge the understanding to more advanced models like ChatGPT.

## 2. Intuition Behind Language Modeling

Language modeling is the task of predicting the next element in a sequence. In this project, we work at the character level, predicting the next character based on previous ones. Starting with statistical frequency counts and eventually training neural networks, the project provides an intuitive understanding of:

- Sequence prediction
- Embedding representations
- Probability distributions and sampling
- Training dynamics and gradient flow

This is a foundational stepping stone for understanding large-scale language models like ChatGPT.

---

## 3. Make More v1: Bigram Character Model

### Overview

This version uses a simple statistical model based on bigram character probabilities. It constructs a 27x27 matrix (26 letters + 1 for start/end token) to capture character transitions.

### Implementation Details

- **Data**: A list of names.
- **Bigram Matrix (N)**: Counts transitions from character `i` to character `j`.
- **Sampling**: Uses `torch.multinomial()` to generate new names based on the probability distribution in matrix N.

```python
for ch1, ch2 in zip(chs, chs[1:]):
    ix1, ix2 = stoi[ch1], stoi[ch2]
    N[ix1, ix2] += 1
```

### Limitations

- Only captures direct character-to-character transitions
- No learning or generalization
- Doesn't use neural networks or gradients

### Output Example

```
avon.
carl.
elai.
nace.
tory.
```

---

## 4. Make More v2–v3.1: Neural Language Models (MLP)

### Key Concepts Introduced

- Character Embeddings
- Multi-Layer Perceptron (MLP)
- Cross-Entropy Loss
- Gradient Descent
- Batch Normalization
- Kaiming Initialization
- Tensor Manipulations

### Model Architecture

- **Input**: Block of 3 character indices → Embeddings → Concatenated vector
- **Layer 1**: Fully connected layer (6D → 100D) with tanh
- **Layer 2**: Fully connected layer (100D → 27D logits)
- **Loss**: Cross-entropy comparing predicted logits to true next characters

### Manual Training Loop

This stage focused on building every component from scratch without relying on high-level APIs:

- Embedding layer using manual weight matrix lookup
- Forward pass using raw matrix multiplications
- Tanh activations and softmax implementation
- Cross-entropy loss calculation with log-sum-exp trick for numerical stability
- **Manual Backward Pass**: Each parameter’s gradient was manually derived and computed.
  - Backpropagation equations for tanh and cross-entropy with softmax were hand-coded
  - Gradients were propagated through each layer
  - Backpropagation for BatchNorm manually derived and implemented
- Parameters were updated using basic SGD

This gives deep insight into how gradients flow and how weight updates occur without abstracting away any mechanics.

### Upgrades in v3.1

- Added BatchNorm with running mean/std
- Implemented Kaiming initialization
- Manually coded backward pass for all components
- Validated gradients using autograd comparison

### 📚 Key References:

- **MLP Architecture & Backpropagation**:
  - Bengio, Y., et al. (2003). [Neural Probabilistic Language Models](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- **Batch Normalization**:
  - Ioffe, S., & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- **Kaiming Initialization**:
  - He, K., et al. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

---

## 5. Make More v4: OOP Design + Training Diagnostics

### Architectural Enhancements

- Rewritten using PyTorch `nn.Module`
- Encapsulated model blocks: Embedding, MLP, Output Layer
- Modular and extensible design

### Training Enhancements

- Logit squashing for improved generalization
- Step-wise learning rate decay

### Diagnostic Tools

- **Gradient Histogram**: Tracks gradient distribution across layers
- **Update-to-Data Ratio Plot**: Visual feedback for tuning learning rate

---

## 6. Make More v5: WaveNet-Inspired Architecture

### Key Changes

- Inspired by WaveNet's local structure preservation
- Uses grouped processing via `FlattenConsecutive`
- Modular components:
  - `Embedding`
  - `FlattenConsecutive`
  - `Sequential` logic wrapper

### Design Rationale

- Inputs processed in bigram/trigram structure
- Mimics 1D convolutional behavior
- Maintains temporal locality before full flattening

### Visualization

- Loss plotted over 1000-iteration windows for clarity

---

## 7. How Make More Relates to ChatGPT

| Feature       | Make More                        | ChatGPT                              |
| ------------- | -------------------------------- | ------------------------------------ |
| Input Type    | Characters                       | Tokens (subwords)                    |
| Model Type    | MLP → Modular NN → WaveNet-style | Transformer-based decoder-only model |
| Training Data | Small dataset (names)            | Internet-scale corpus                |
| Goal          | Next-character prediction        | Next-token prediction                |
| Depth         | Shallow (1–3 layers)             | Deep (96+ layers in GPT-4)           |

**Why It Matters**: Understanding character-level modeling teaches the essential principles that scale up to ChatGPT:

- Sequential prediction
- Embeddings
- Loss minimization
- Neural architectures

---

## 8. Final Outputs

```
carman.
ambrie.
khismi.
xilah.
khalani.
emmahnee.
dellyn.
jarqui.
nermari.
chaiir.
kaleigh.
hamoni.
jaquinn.
```

These names are generated based on learned patterns, not memorized data.

---

## 9. Conclusion & Learning Outcomes

### What Was Learned

- Built language models **from scratch** using only NumPy and PyTorch basics
- Understood **every component manually**, from embedding to output layer
- Learned how to **manually implement a backward pass** for MLPs and BatchNorm
- Understood the importance of weight initialization (Kaiming)
- Practiced intensive **tensor manipulations** for reshaping and broadcasting
- Used batch normalization to stabilize training
- Learned how to plot diagnostics like gradient histograms and update ratios
- Explored **WaveNet-like** modular architectures and local structure preservation

### Future Roadmap

- Add residual connections for deeper learning
- Implement dilated convolutions (like full WaveNet)
- Explore attention mechanisms
- Transition to word/subword token modeling

Make More is not just a toy project — it's a microscope for understanding how deep language models like GPT truly work.

