# server/evaluate.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# -----------------------------------------
# CONFIG
# -----------------------------------------
MODEL_NAME = "ai4bharat/indicBERTv2-mlm-only"
GLOBAL_MODEL_PATH = "server/global_model.pt"
TEST_DATA_PATH = "data/test.csv"  # Unified test set
NUM_LABELS = 3

# -----------------------------------------
# Dataset Class
# -----------------------------------------
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

# -----------------------------------------
# Model
# -----------------------------------------
class AspectClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

# -----------------------------------------
# Evaluation
# -----------------------------------------
def evaluate():
    # Load and encode test data
    df = pd.read_csv(TEST_DATA_PATH)
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["aspect_label"])

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ComplaintDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=4)

    # Load model
    model = AspectClassifier(MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Run evaluation
if __name__ == "__main__":
    evaluate()
