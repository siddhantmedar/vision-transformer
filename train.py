#!/usr/bin/env python3
"""
Training script for Vision Transformer
Includes training loop, testing, and model wrapper
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    optimizer='adamw',
    learning_rate=3e-3,
    weight_decay=0.03,
    warmup_epochs=5,
    epochs=100,
    save_path='checkpoints',
    device=None
):
    """
    Train a Vision Transformer model

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer type ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_epochs: Number of warmup epochs
        epochs: Total number of training epochs
        save_path: Directory to save checkpoints
        device: Device to train on (None for auto-detect)

    Returns:
        best_val_loss: Best validation loss achieved
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Setup optimizer
    if optimizer == 'adamw':
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Setup scheduler
    scheduler = CosineAnnealingLR(opt, T_max=epochs - warmup_epochs, eta_min=1e-6)

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    print(f"Training on device: {device}")
    print(f"Optimizer: {optimizer}, LR: {learning_rate}, WD: {weight_decay}")
    print("-" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

            train_loss += loss.item()
            train_acc += acc
            num_batches += 1

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        train_loss /= num_batches
        train_acc /= num_batches

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        val_loss /= num_batches
        val_acc /= num_batches

        # === SCHEDULER ===
        current_epoch = epoch + 1
        if current_epoch <= warmup_epochs:
            # Linear warmup
            warmup_factor = current_epoch / warmup_epochs
            for param_group in opt.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
        else:
            scheduler.step()

        current_lr = opt.param_groups[0]['lr']

        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'lr': f'{current_lr:.6f}'
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_path = os.path.join(save_path, 'best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_path)
            tqdm.write(f"  → Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            last_path = os.path.join(save_path, 'last.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, last_path)

    # Save final checkpoint
    final_path = os.path.join(save_path, 'last.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs - 1,
    }, final_path)

    print(f"\n{'-' * 60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return best_val_loss


def test(model, test_loader, checkpoint_path=None, device=None):
    """
    Test a Vision Transformer model

    Args:
        model: The model to test
        test_loader: DataLoader for test data
        checkpoint_path: Path to checkpoint file to load (optional)
        device: Device to test on (None for auto-detect)

    Returns:
        dict: Dictionary with test metrics
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")

    print(f"Testing on device: {device}")
    print("-" * 60)

    # Test
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_top3_acc = 0.0
    num_batches = 0

    loss_fn = nn.CrossEntropyLoss()
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for x, y in test_pbar:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            # Top-1 accuracy
            acc = (torch.argmax(y_hat, dim=1) == y).float().mean().item()

            # Top-3 accuracy
            top3 = torch.topk(y_hat, min(3, y_hat.size(1)), dim=1)[1]
            top3_acc = (top3 == y.unsqueeze(-1)).float().sum() / x.shape[0]

            test_loss += loss.item()
            test_acc += acc
            test_top3_acc += top3_acc.item()
            num_batches += 1

            test_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'top3': f'{top3_acc.item():.4f}'
            })

    test_loss /= num_batches
    test_acc /= num_batches
    test_top3_acc /= num_batches

    print(f"\n{'-' * 60}")
    print(f"Test Results:")
    print(f"  Loss:        {test_loss:.4f}")
    print(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Top-3 Acc:   {test_top3_acc:.4f} ({test_top3_acc*100:.2f}%)")
    print("-" * 60)

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_top3_acc': test_top3_acc
    }


if __name__ == '__main__':
    print("Import this module to use train() and test() functions")
    print("\nExample:")
    print("  from train import train, test")
    print("  from vision_transformer.model import VisionTransformer")
    print("  from data.dataset import get_dataloaders")
    print()
    print("  model = VisionTransformer()")
    print("  train_loader, val_loader = get_dataloaders()")
    print("  train(model, train_loader, val_loader)")
