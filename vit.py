import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from datasets import load_dataset  
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Device definition
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if device.type == "mps":
    print("Using MPS device. Some operations may have limitations.")

# Load and prepare data
try:
    dataset = load_dataset('Maysee/tiny-imagenet',token=os.getenv('HF_TOKEN'))
except (FileNotFoundError, PermissionError, UnicodeDecodeError, IOError, Exception) as e:
    print(f"Error loading dataset: {e}")
    raise

@dataclass
class Config:
    n_layers: int = 6
    n_classes: int = len(set(dataset["train"]["label"]))
    d_model: int = 128
    num_head: int = 4
    d_head: int = d_model // num_head
    patch_size: int = 8
    n_channels: int = 3
    image_size: tuple = (64, 64)

config = Config()

def preprocess(tensor):
    # Convert to float and normalize to [0, 1]
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor - mean) / std
    
    return tensor

def get_dataset(dataset,split):

    if split == 'train':
        train_image_data = preprocess(torch.tensor(dataset["train"]["image"]))
        train_label_data = torch.tensor(dataset["train"]["label"])

        indices = torch.randperm(len(train_image_data))

        train_image_data = train_image_data[indices]
        train_label_data = train_label_data[indices]

        return train_image_data, train_label_data
    
    elif split in ['val', 'test']:
        valid_image_data = preprocess(torch.tensor(dataset["valid"]["image"]))
        valid_label_data = torch.tensor(dataset["valid"]["label"])

        valid_images_np = valid_image_data.numpy()
        valid_labels_np = valid_label_data.numpy()
        
        valid_indices, test_indices = train_test_split(
            np.arange(len(valid_images_np)), 
            test_size=0.1,  # 10% for test (1k out of 10k)
            stratify=valid_labels_np,
            random_state=313
        )
        
        # Split the validation data
        valid_image_data = torch.tensor(valid_images_np[valid_indices])
        valid_label_data = torch.tensor(valid_labels_np[valid_indices])
        test_image_data = torch.tensor(valid_images_np[test_indices])
        test_label_data = torch.tensor(valid_labels_np[test_indices])

        if split == 'val':
            return valid_image_data, valid_label_data
        elif split == 'test':
            return test_image_data, test_label_data
    else:
        raise ValueError(f"Invalid split: {split}")

class ViTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_dataset_images, train_dataset_labels = get_dataset(dataset, 'train')
val_dataset_images, val_dataset_labels = get_dataset(dataset, 'val')
test_dataset_images, test_dataset_labels = get_dataset(dataset, 'test')

train_dataset = ViTDataset(train_dataset_images, train_dataset_labels)
valid_dataset = ViTDataset(val_dataset_images, val_dataset_labels)
test_dataset = ViTDataset(test_dataset_images, test_dataset_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ViTEmbeddding(nn.Module):
    def __init__(self, n_channels, d_model, image_size, patch_size,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = (image_size[0] // self.patch_size) * (image_size[1] // self.patch_size)
        self.proj = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape # (batch_size, n_channels, height, width)
        out = self.proj(x) # (batch_size, d_model, height // patch_size, width // patch_size)
        out = out.flatten(2).transpose(1, 2) # (batch_size, n_patches, d_model)
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, d_model)
        out = torch.cat([cls_token, out], dim=1) # (batch_size, 1 + n_patches, d_model)
        out = out + self.pos_embedding # (batch_size, 1 + n_patches, d_model)
        return self.dropout(out) # (batch_size, 1 + n_patches, d_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        assert d_model % num_head == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_patches, d_model = x.shape # (batch_size, n_patches, d_model)
        Q = self.W_q(x) # (batch_size, n_patches, d_model)
        K = self.W_k(x) # (batch_size, n_patches, d_model)
        V = self.W_v(x) # (batch_size, n_patches, d_model)
        Q = Q.view(batch_size, n_patches, self.num_head, self.d_head).transpose(1, 2) # (batch_size, num_head, n_patches, d_head)
        K = K.view(batch_size, n_patches, self.num_head, self.d_head).transpose(1, 2) # (batch_size, num_head, n_patches, d_head)
        V = V.view(batch_size, n_patches, self.num_head, self.d_head).transpose(1, 2) # (batch_size, num_head, n_patches, d_head)
        attn_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head) # (batch_size, num_head, n_patches, n_patches)
        attn_weight = torch.softmax(attn_scores, dim=-1) # (batch_size, num_head, n_patches, n_patches)
        attn_weight = self.dropout(attn_weight) # (batch_size, num_head, n_patches, n_patches)
        attn = (attn_weight @ V).transpose(1, 2) # (batch_size, num_head, n_patches, d_head)
        attn = attn.contiguous().view(batch_size, n_patches, d_model) # (batch_size, n_patches, d_model)
        return x + attn # (batch_size, n_patches, d_model)

class FNN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.fnn(x)
        return out

class Encoder(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_head, dropout)
        self.fnn = FNN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (batch_size, 1 + n_patches, d_model)
        normalized_input = self.norm1(x) # (batch_size, 1 + n_patches, d_model)
        attention_output = self.mha(normalized_input) + x # (batch_size, 1 + n_patches, d_model)
        normalized_attention = self.norm2(attention_output) # (batch_size, 1 + n_patches, d_model)
        ffn_output = self.fnn(normalized_attention) + attention_output # (batch_size, 1 + n_patches, d_model)
        return self.dropout(ffn_output) # (batch_size, 1 + n_patches, d_model)

class EncoderBlock(nn.Module):
    def __init__(self, n_layers, n_channels, d_model, num_head, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding = ViTEmbeddding(n_channels, d_model, image_size, patch_size, dropout) # (batch_size, n_channels, height, width) -> (batch_size, 1 + n_patches, d_model)
        self.encoder_layers = nn.ModuleList(
            [Encoder(d_model, num_head, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x is (batch_size, n_channels, height, width)  
        x = self.embedding(x) # (batch_size, 1 + n_patches, d_model)
        for layer in self.encoder_layers:
            x = layer(x)
        
        cls_token = x[:, 0, :] # (batch_size, d_model)
        cls_token = self.norm(cls_token) # (batch_size, d_model)
        projected_output = self.proj(cls_token) # (batch_size, n_classes)
        return projected_output # (batch_size, n_classes)

def train(model, config, train_dataloader, valid_dataloader, device, epochs=20, batch_size=64, patience=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0

        model.train()

        for batch_images, batch_labels in train_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss_per_epoch.append(avg_train_loss)

        model.eval()

        for batch_images, batch_labels in valid_dataloader:
            with torch.no_grad():
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_dataloader)
        val_loss_per_epoch.append(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def evaluate(model):
    
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch_images, batch_labels in test_dataloader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            logits = model(batch_images)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    try:
        model = EncoderBlock(
            config.n_layers, config.n_channels, config.d_model, config.num_head, config.image_size, config.patch_size, config.n_classes, dropout=0.1
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model size: {total_params:,} parameters ({total_params / 1e6:.2f}M)")
        train(model, config, train_dataloader, valid_dataloader, device, epochs=20, batch_size=32, patience=3)
        evaluate(model)
    except Exception as e:
        print(f"Error during execution: {e}")
        raise