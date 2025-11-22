#!/usr/bin/env python3
"""
Main entry point for Vision Transformer training on ImageNet
"""

from vision_transformer.model import VisionTransformer
from data.dataset import get_dataloaders
from train import train, test


def main():
    """Train Vision Transformer on ImageNet"""

    # Create model
    print("Creating Vision Transformer model...")
    model = VisionTransformer(
        patch_size=16,
        img_size=224,
        in_channels=3,
        num_classes=1000
    )

    # Get data loaders
    print("\nLoading ImageNet data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=32,
        img_size=224,
        num_workers=4
    )

    # Train
    print("\nStarting training...")
    best_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer='adamw',
        learning_rate=3e-3,
        weight_decay=0.03,
        warmup_epochs=5,
        epochs=100,
        save_path='checkpoints'
    )

    print(f"\nTraining finished! Best validation loss: {best_loss:.4f}")

    # Test (optional)
    # test(model, val_loader, checkpoint_path='checkpoints/best.pt')


if __name__ == '__main__':
    main()
