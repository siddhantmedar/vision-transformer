# Vision Transformer (ViT) from Scratch

A PyTorch implementation of Vision Transformer for CIFAR-100 classification, built from scratch for learning purposes.

## Architecture

- **Patch Embedding**: 4x4 patches → 64 patches per 32x32 image
- **Transformer Encoder**: 12 layers, 384 hidden dim, 3 attention heads
- **Classification**: Global average pooling → linear head

Custom implementations of:
- Layer Normalization
- Dropout
- Multi-Head Self-Attention

## Setup

```bash
# Install dependencies
uv sync

# Train
uv run python run.py --train --epochs 100

# Test
uv run python run.py
```

## Training

```bash
# With custom parameters
uv run python run.py --train --epochs 50 --lr 3e-4 --weight_decay 0.1

# Monitor with TensorBoard
tensorboard --logdir checkpoints/runs
```

## Project Structure

```
├── model.py      # ViT architecture
├── dataset.py    # CIFAR-100 data loading
├── run.py        # Training and testing
├── config.toml   # Model configuration
```

## Configuration

Edit `config.toml` to modify:
- Model architecture (d_model, num_heads, num_layers)
- Patch size
- Batch size

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Original ViT paper
