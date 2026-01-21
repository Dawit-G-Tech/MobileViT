# MobileViT for CIFAR-10 Classification

This project contains an implementation of MobileViT, a lightweight hybrid architecture that combines the strengths of CNNs and Vision Transformers, optimized for CIFAR-10 image classification.

MobileViT is designed to address the "lightweight" challenge by:
- Combining CNN speed with Transformer accuracy
- Using a novel "Unfold - Transform - Fold" mechanism for efficient global attention
- Maintaining spatial relationships while processing global context


### MobileViT Block Components:
1. **Local Representation (L)**: 3x3 convolutions for capturing low-level features (edges, textures)
2. **Global Representation (G)**: Sparse Global Attention via Transformer using the Unfold-Transform-Fold mechanism
3. **Fusion (F)**: Concatenation of local and global features

### The Core Mechanism: "Unfold - Transform - Fold"
- **Unfold**: Converts image tensor (B, C, H, W) into patches (B, N, P, d) where:
  - N = number of patches
  - P = pixels per patch
  - d = channels/embedding dimension
- **Transform**: Applies Transformer blocks to process "pixels at the same position across all patches"
- **Fold**: Reshapes back to (B, C, H, W) maintaining spatial integrity

## Files

- `mobilevit.py`: Complete MobileViT architecture implementation
- `train.py`: Training script for CIFAR-10
- `MobileViT_CIFAR10.ipynb`: Google Colab notebook for easy execution
- `requirements.txt`: Python dependencies

## Usage

### Google Colab

1. Open `MobileViT_CIFAR10.ipynb` in Google Colab
2. Upload `mobilevit.py` when prompted
3. Run all cells sequentially
4. The notebook will:
   - Install dependencies
   - Load CIFAR-10 dataset
   - Create and train MobileViT model
   - Visualize training progress
   - Evaluate on test set

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train MobileViT-XXS (smallest variant)
python train.py --model xxs --epochs 100 --batch-size 128

# Train MobileViT-XS
python train.py --model xs --epochs 100 --batch-size 128

# Train MobileViT-S
python train.py --model s --epochs 100 --batch-size 128
```

### Command Line Arguments

- `--model`: Model variant (`xxs`, `xs`, or `s`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 0.01)

## Model Variants

- **MobileViT-XXS**: Extra extra small (width_multiplier=0.5)
- **MobileViT-XS**: Extra small (width_multiplier=0.75)
- **MobileViT-S**: Small (width_multiplier=1.0)

## Key Features

- ✅ Handles arbitrary input dimensions (dynamic padding/reshaping)
- ✅ Efficient implementation using PyTorch operations (view, permute, contiguous)
- ✅ Optimized for CIFAR-10 (32x32 images)
- ✅ Includes data augmentation
- ✅ Cosine annealing learning rate scheduler
- ✅ Model checkpointing

## Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (with cosine annealing)
- **Weight Decay**: 0.01
- **Data Augmentation**: Random crop (padding=4) and horizontal flip
- **Normalization**: CIFAR-10 mean/std normalization


## Implementation Notes

The implementation follows the MobileViT paper's architecture:
- Uses inverted residual blocks (MobileNet-style) for efficient feature extraction
- Implements the Unfold-Transform-Fold mechanism manually using PyTorch operations
- Handles non-divisible dimensions with dynamic padding
- Maintains output shape matching input shape (B, C, H, W)

## References

- [MobileViT Paper](https://arxiv.org/abs/2110.02178)
- Original implementation concepts from Apple's MobileViT

