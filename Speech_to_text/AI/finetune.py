import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from train_step2 import ANN, fft_feat, DEVICE
import os

def expand_output_layer(model, new_output_dim):
    old_weight = model.cls.weight.data
    old_bias = model.cls.bias.data

    model.cls = nn.Linear(old_weight.shape[1], new_output_dim).to(DEVICE)

    with torch.no_grad():
        model.cls.weight[:old_weight.shape[0]] = old_weight
        model.cls.bias[:old_bias.shape[0]] = old_bias

    print(f"✅ Expanded output layer from {old_weight.shape[0]} to {new_output_dim} classes.")
    return model

# Load dữ liệu
wav_paths = np.load("models/wav_paths.npy", allow_pickle=True)
labels = np.load("models/labels.npy", allow_pickle=True)
id2sentence = np.load("models/id2sentence.npy", allow_pickle=True).item()

# Dataset
class AudioDataset(Dataset):
    def __init__(self, wav_paths, labels):
        self.wav_paths = wav_paths
        self.labels = labels

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        fft = fft_feat(self.wav_paths[idx])
        return torch.tensor(fft).float(), torch.tensor(self.labels[idx])

train_loader = DataLoader(AudioDataset(wav_paths, labels), batch_size=1, shuffle=True)

# Load model cũ
input_dim = 257
current_output_dim = 2204   # model cũ có 2204 classes
new_output_dim = max(id2sentence.keys()) + 1

model = ANN(input_dim, current_output_dim).to(DEVICE)
model.load_state_dict(torch.load(os.path.join("models", "ann_fft.pt"), map_location=DEVICE))

# Nếu cần mở rộng lớp output
if new_output_dim > current_output_dim:
    model = expand_output_layer(model, new_output_dim)

model.train()

# Fine-tune
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    running_loss = 0.0
    for fft, label in train_loader:
        fft = fft.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(fft)[0]
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.6f}")

# Save model mới
torch.save(model.state_dict(), os.path.join("models", "ann_fft.pt"))
print("Fine-tuning thành công! Model mới đã lưu: ann_fft.pt")
