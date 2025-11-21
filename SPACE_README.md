---
title: SmolLM2-135M Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 SmolLM2-135M Text Generation

A **SmolLM2-135M** language model trained from scratch with **Grouped-Query Attention (GQA)**, demonstrating modern LLM architecture and training techniques.

## 🎯 Model Overview

This model is a complete implementation of the SmolLM2-135M architecture, trained as part of an advanced deep learning assignment.

### Model Specifications

| Specification | Value |
|--------------|-------|
| **Total Parameters** | ~135 Million |
| **Layers** | 30 |
| **Hidden Size** | 576 |
| **Attention Heads** | 9 |
| **KV Heads** | 3 (GQA) |
| **Intermediate Size** | 1536 |
| **Training Steps** | 5050 |
| **Final Loss** | 1.0297 |

## ⚡ Architecture Highlights

- ✅ **Grouped-Query Attention (GQA)**: 3x memory efficiency vs standard attention
- ✅ **RMSNorm**: Faster normalization than LayerNorm
- ✅ **SwiGLU Activation**: Superior to standard FFN
- ✅ **Rotary Position Embeddings (RoPE)**: Better position encoding
- ✅ **Flash Attention**: 2-4x faster attention computation
- ✅ **Mixed Precision (bfloat16)**: Reduced memory usage
- ✅ **torch.compile**: JIT optimization for faster inference

## 🚀 Training Details

The model was trained with state-of-the-art optimization techniques:

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 3e-4 with cosine decay
- **Batch Size**: 4
- **Sequence Length**: 256
- **Checkpoint Strategy**: Save/resume capability demonstrated
- **Hardware**: GPU P100 on Kaggle

### Training Progress

```
Step     0 | Loss: 4.2145 → Initial random state
Step  1000 | Loss: 1.5745 → Learning patterns
Step  3000 | Loss: 1.2657 → Improving coherence
Step  5000 | Loss: 1.1319 → Checkpoint saved
Step  5050 | Loss: 1.0297 → Final model
```

## 📊 Performance

- **Throughput**: ~5,600 tokens/sec on GPU
- **Inference Speed**: Fast (optimized with torch.compile)
- **Memory Usage**: Efficient (GQA reduces KV cache by 3x)

## 🎮 How to Use

1. **Click "🔄 Load Model"** to initialize the model
2. **Enter a prompt** in the text box
3. **Adjust parameters**:
   - **Max Tokens**: Length of generated text (10-500)
   - **Temperature**: Randomness (0.1=conservative, 2.0=creative)
   - **Top-K**: Sampling diversity (lower=focused, higher=diverse)
4. **Click "✨ Generate"** to create text

### Example Prompts

Try these prompts to see the model in action:

```
ROMEO:

KING RICHARD II:

Once upon a time

To be or not to be
```

## ⚠️ Important Notes

> **This is a demonstration model** trained for only 5,050 steps on Shakespeare text.
> 
> - The architecture is production-ready
> - The model needs significantly more training (100K+ steps) for high-quality outputs
> - Current outputs may be incoherent or repetitive
> - This serves as a proof-of-concept for the architecture and training pipeline

## 🔬 Technical Implementation

### Grouped-Query Attention (GQA)

GQA reduces memory usage while maintaining quality:

```
Query heads:  Q1, Q2, Q3 | Q4, Q5, Q6 | Q7, Q8, Q9
                  ↓            ↓            ↓
KV heads:        KV1          KV2          KV3
```

- 9 query heads grouped into 3 groups
- Each group shares 1 key-value head
- 3x reduction in KV cache size

### Architecture Diagram

```
Input Tokens
    ↓
Token Embeddings
    ↓
30x Transformer Blocks
    ├─ RMSNorm
    ├─ Grouped-Query Attention (GQA)
    ├─ Residual Connection
    ├─ RMSNorm
    ├─ MLP (SwiGLU)
    └─ Residual Connection
    ↓
Final RMSNorm
    ↓
LM Head (tied with embeddings)
    ↓
Output Logits
```

## 📚 Project Details

This model demonstrates:

1. **Architecture Reverse-Engineering**: Implemented SmolLM2 from HuggingFace config
2. **Modern Optimizations**: All Session 13 speedup techniques applied
3. **Checkpoint Management**: Proper save/resume implementation
4. **Deployment**: End-to-end pipeline from training to production

### Assignment Objectives ✅

- ✅ Reverse-engineer SmolLM2-135M architecture
- ✅ Train for 5000 steps with checkpointing
- ✅ Resume training from checkpoint for 50 additional steps
- ✅ Apply all optimization techniques
- ✅ Generate text every 500 steps
- ✅ Deploy to Hugging Face Spaces

## 🔗 Links

- **GitHub Repository**: [Add your repo link]
- **Training Notebook**: [Add Kaggle link]
- **Original Model**: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)

## 📄 License

MIT License - Free for educational and commercial use

---

**Built with ❤️ as part of ERA V4 Course - Session 13 Assignment**
