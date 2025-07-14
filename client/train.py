#  MODEL_NAME = ""  # lightweight multilingual model
# client/train.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.preprocessing import LabelEncoder
import argparse

# -----------------------------
# Config
# -----------------------------
MODEL_NAME     = "bert-base-multilingual-cased"
MAX_LEN        = 128
BATCH_SIZE     = 8
EPOCHS         = 3
LR             = 2e-5
NOISE_MULT     = 1.0     # adjust for privacy vs. utility
MAX_GRAD_NORM  = 1.0
DELTA          = 1e-5    # target delta for ε calculation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------
class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------
# Model - Opacus Compatible
# -----------------------------
class AspectClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        # Disable dropout in encoder for Opacus compatibility
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Only train the classifier head
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))

# -----------------------------
# Training with DP
# -----------------------------
def train_local_model(region_csv, region_name):
    df = pd.read_csv(region_csv)
    texts = df["text"].tolist()
    labels = LabelEncoder().fit_transform(df["aspect"].tolist())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AspectDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AspectClassifier(num_labels=len(set(labels))).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # Calculate sample rate and initialize PrivacyEngine
    sample_rate = BATCH_SIZE / len(dataset)
    privacy_engine = PrivacyEngine()
    
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=NOISE_MULT,
        max_grad_norm=MAX_GRAD_NORM
    )
    print(f"🔐 DP Engine attached (σ={NOISE_MULT}, C={MAX_GRAD_NORM}, q={sample_rate:.4f})")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        with BatchMemoryManager(
            data_loader=dataloader, 
            max_physical_batch_size=BATCH_SIZE,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            
            for batch in memory_safe_data_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"[{region_name}] Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(dataloader):.4f}")

    # Compute epsilon
    epsilon = privacy_engine.get_epsilon(delta=DELTA)
    print(f"🔐 Trained with DP: ε = {epsilon:.2f}, δ = {DELTA}")

    # Save
    os.makedirs("client/weights", exist_ok=True)
    path = f"client/weights/{region_name}.pt"
    torch.save(model.state_dict(), path)
    print(f"✅ Saved DP-trained model at: {path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Region name (Region1, etc.)")
    parser.add_argument("--csv",    required=True, help="Path to regional CSV file")
    args = parser.parse_args()

    train_local_model(args.csv, args.region)
