# 📦 Structured Pruning of MobileNetV2 using Torch-Pruning

We apply **structured pruning** to a modified MobileNetV2 architecture using the [`torch-pruning`](https://pypi.org/project/torch-pruning/) library in PyTorch.

---

## 🔧 What is Torch-Pruning?

**Torch-Pruning** is a Python library designed to reduce model size and computation (FLOPs) by pruning channels, layers, or blocks in PyTorch models. It introduces a graph-based algorithm called **DepGraph** to identify dependencies and safely prune connected layers.

### 🔄 What is DepGraph?

**DepGraph** models dependencies between parameters across layers as a graph. It:
- Builds a dependency graph using the model's forward pass.
- Automatically groups dependent parameters for safe pruning.
- Applies pruning based on a global or local importance metric (e.g., L1-norm).

This approach supports CNNs, RNNs, GNNs, Transformers, and more — without manual pruning rules.

---

## ⚙️ Pruning MobileNetV2

### 🧠 Architecture Notes

MobileNetV2 uses **inverted residual blocks** with optional skip connections:
```python
def forward(self, x):
    if self.use_res_connect:
        return x + self.conv(x)  # Requires matching shapes
    else:
        return self.conv(x)
```
This introduces a pruning issue: After pruning, self.conv(x) may output fewer channels. But the skip connection x still has the original number. This leads to a shape mismatch: RuntimeError: size mismatch at tensor addition.

### 🛠️ Solution
To avoid this, residual blocks were excluded from pruning:
```python
# Ignore Conv2d layers in residual blocks
ignored_layers = []
for m in model.features.modules():
    if isinstance(m, InvertedResidual) and m.use_res_connect:
        for name, layer in m.named_modules():
            if isinstance(layer, nn.Conv2d):
                ignored_layers.append(layer)

# Also ignore the classifier
ignored_layers.append(model.classifier)
```
### 🧪 Pruning Configuration

| Parameter        | Value                                              |
| ---------------- | -------------------------------------------------- |
| **Method**       | `torch-pruning` with `MagnitudePruner`             |
| **Metric**       | L1-norm (`tp.importance.MagnitudeImportance(p=1)`) |
| **Mode**         | Global structured pruning (channel-wise)           |
| **Ratios Tried** | 30%, 50%, and 80% of prunable channels             |
| **Ignored**      | Residual blocks with skip connections + classifier |

## 📉 What Does L1-Norm Mean?

L1-norm is used to determine the "importance" of each convolutional filter:

$$\text{L1-norm} = \sum |w_i|$$

Where $w_i$ are the weights of the filter.  
Filters with lower L1-norms are considered less important and are pruned first.

---

## ❓ Does 80% Pruning Mean 80% Smaller Model and 80% FLOPs reduction?

Not exactly.

- **80% pruning** applies only to **prunable channels** — layers in `ignored_layers` are excluded.
- **FLOPs and model size** reductions depend on **which layers are pruned**:
  - Pruning late layers (which is common) leads to smaller FLOPs reduction.
  - FLOPs are **not evenly distributed** across layers.
  - Model size depends on **total parameters**, not just channel count.

---

## 📌 Summary

- ✅ Applied structured pruning to MobileNetV2 using `torch-pruning`.
- ✅ Handled residuals safely by excluding skip-connected blocks.
- ✅ Evaluated pruning at multiple ratios (30%, 50%, 80%).
- ✅ Learned that 80% pruning ≠ 80% FLOPs or size reduction due to pruning only a subset of layers and uneven FLOPs distribution.
