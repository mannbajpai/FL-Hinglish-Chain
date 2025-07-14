# client/train.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from opacus import PrivacyEngine
from sklearn.preprocessing import LabelEncoder
import argparse

# -----------------------------
# Configurations
# -----------------------------
MODEL_NAME = "ai4bharat/indicBERTv2-mlm"  # lightweight multilingual model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3

# -----------------------------
# Dataset Class
# -----------------------------
class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------
# Model Class
# -----------------------------
class AspectClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(AspectClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.dropout(pooled_output)
        return self.classifier(x)

# -----------------------------
# Train Function with Differential Privacy
# -----------------------------
def train_local_model(region_csv, region_name):
    print(f"\n🔁 Training model for {region_name}...")

    df = pd.read_csv(region_csv)
    texts = df["text"].tolist()
    labels = LabelEncoder().fit_transform(df["aspect"])  # Encode to 0, 1, 2

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = AspectDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AspectClassifier(num_labels=len(set(labels))).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Enable Differential Privacy
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,      # tune this for privacy/accuracy trade-off
        max_grad_norm=1.0
    )
    print(f"🔐 DP Enabled | ε (epsilon) ≈ {privacy_engine.accountant.get_epsilon(delta=1e-5):.2f}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"📉 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    os.makedirs("client/weights", exist_ok=True)
    torch.save(model.state_dict(), f"client/weights/{region_name}.pt")
    print(f"✅ Saved DP-trained model: client/weights/{region_name}.pt")

# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Region name (e.g., Region1)")
    parser.add_argument("--csv", required=True, help="Path to regional CSV data")
    args = parser.parse_args()

    train_local_model(region_csv=args.csv, region_name=args.region)
