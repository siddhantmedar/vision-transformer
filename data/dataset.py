#!/usr/bin/env python3
"""
ImageNet dataset: download, load, and DataLoader utilities
All-in-one file for dataset management
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, load_from_disk


# ==================== Download ====================

def download_imagenet(split='train', subset=None):
    """
    Download ImageNet from Hugging Face
    Dataset is ready to use - labels are already single integers (0-999)

    Args:
        split: 'train' or 'validation'
        subset: Number of samples or percentage string (e.g., '1%', '10%')
    """
    output_path = Path(__file__).parent / 'raw' / 'imagenet'
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ImageNet {split} split...")
    print(f"Output: {output_path / split}")

    # Get HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("\nWARNING: Set HF_TOKEN environment variable")
        print("export HF_TOKEN='your_token'")
        print("Accept license at: https://huggingface.co/datasets/imagenet-1k")

    # Download
    dataset = load_dataset(
        'imagenet-1k',
        split=split,
        token=hf_token,
        trust_remote_code=True
    )

    # Apply subset if specified
    if subset:
        total = len(dataset)
        if isinstance(subset, str) and '%' in subset:
            num_samples = int(total * float(subset.rstrip('%')) / 100)
        else:
            num_samples = int(subset)
        num_samples = min(num_samples, total)
        print(f"Using {num_samples}/{total} samples")
        dataset = dataset.select(range(num_samples))

    # Save dataset
    dataset.save_to_disk(str(output_path / split))
    print(f"Saved {len(dataset)} samples to {output_path / split}")
    print(f"Ready for training - apply transforms in DataLoader")


def load_imagenet(split='train'):
    """Load downloaded ImageNet dataset"""
    path = Path(__file__).parent / 'raw' / 'imagenet' / split
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Download it first using: python data/dataset.py --split {split}"
        )
    return load_from_disk(str(path))


# ==================== DataLoader ====================

class ImageNetDataset(Dataset):
    """Wrapper for HuggingFace ImageNet dataset with PyTorch transforms"""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224, is_train=True):
    """Standard ImageNet transforms for Vision Transformer"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(batch_size=32, img_size=224, num_workers=4):
    """
    Create train and validation DataLoaders for ImageNet

    Args:
        batch_size: Batch size for training
        img_size: Image size (224, 384, etc.)
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader
    """
    print("Loading ImageNet datasets...")
    train_hf = load_imagenet('train')
    val_hf = load_imagenet('validation')

    # Get transforms
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)

    # Create PyTorch datasets
    train_dataset = ImageNetDataset(train_hf, train_transform)
    val_dataset = ImageNetDataset(val_hf, val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ImageNet dataset management')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download ImageNet')
    download_parser.add_argument('--split', default='train', choices=['train', 'validation'])
    download_parser.add_argument('--subset', default=None, help='e.g., "1%", "10%", or "1000"')
    download_parser.add_argument('--all-splits', action='store_true', help='Download both splits')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect dataset')
    inspect_parser.add_argument('--split', default='train', choices=['train', 'validation'])

    # Test command
    subparsers.add_parser('test', help='Test dataloader')

    args = parser.parse_args()

    if args.command == 'download':
        if args.all_splits:
            for split in ['train', 'validation']:
                print(f"\n{'='*60}")
                download_imagenet(split, args.subset)
                print(f"{'='*60}\n")
        else:
            download_imagenet(args.split, args.subset)

    elif args.command == 'inspect':
        dataset = load_imagenet(args.split)
        print(f"\nDataset: {args.split}")
        print(f"Samples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Image: {sample['image'].size}, mode={sample['image'].mode}")
        print(f"  Label: {sample['label']}")

    elif args.command == 'test':
        print("Testing DataLoader...")
        train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels}")
        print("DataLoader test successful!")

    else:
        parser.print_help()
