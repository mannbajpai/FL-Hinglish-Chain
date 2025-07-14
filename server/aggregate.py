# server/aggregate.py
import os
import torch
from transformers import AutoModel

MODEL_NAME = "ai4bharat/indicBERTv2-mlm-only"
WEIGHTS_DIR = "server/decrypted_weights"
OUTPUT_PATH = "server/global_model.pt"

class AspectClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

# 🔁 Strip out classifier weights and average only encoder

def federated_average(weight_files):
    encoder_keys = None
    accum = {}
    num_clients = len(weight_files)

    for i, path in enumerate(weight_files):
        print(f"🔄 Loading {path}")
        state_dict = torch.load(path, map_location=torch.device("cpu"))

        encoder_weights = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

        if encoder_keys is None:
            encoder_keys = encoder_weights.keys()
            for k in encoder_keys:
                accum[k] = encoder_weights[k].clone()
        else:
            for k in encoder_keys:
                accum[k] += encoder_weights[k]

    for k in accum:
        accum[k] = accum[k] / num_clients

    return accum


def aggregate_models():
    weight_files = [os.path.join(WEIGHTS_DIR, f) for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")]
    if len(weight_files) < 2:
        print("❌ Need at least 2 models for aggregation.")
        return

    print(f"📦 Aggregating {len(weight_files)} models (encoder only)...")
    averaged_encoder = federated_average(weight_files)

    model = AspectClassifier(MODEL_NAME, num_labels=3)
    model_dict = model.state_dict()
    model_dict.update(averaged_encoder)
    model.load_state_dict(model_dict)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"✅ Aggregated global model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    aggregate_models()
