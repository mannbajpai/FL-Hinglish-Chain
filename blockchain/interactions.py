import hashlib
from web3 import Web3
import json
import os
import argparse

# -------------------------------
# ✅ Blockchain Setup
# -------------------------------
# Connect to local blockchain
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "❌ Failed to connect to local blockchain"

# Load contract ABI from Hardhat artifacts
ARTIFACT_PATH = "contracts/artifacts/contracts/FLModelStore.sol/FLModelStore.json"
with open(ARTIFACT_PATH) as f:
    contract_json = json.load(f)
    abi = contract_json["abi"]

# Paste your deployed contract address here
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" 

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# Use first available local account
w3.eth.default_account = w3.eth.accounts[0]


# -------------------------------
# 🔐 Step 1: Hash the model weights
# -------------------------------
def hash_model(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()


# -------------------------------
# 🔗 Step 2: Send to Blockchain
# -------------------------------
def send_update(client_id, model_path):
    print(f"📦 Hashing model: {model_path}")
    model_hash = hash_model(model_path)
    print(f"🔐 SHA256: {model_hash}")

    print("📤 Sending to blockchain...")
    tx_hash = contract.functions.submitUpdate(client_id, model_hash).transact()
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Submitted! Tx hash: {receipt.transactionHash.hex()}")


# -------------------------------
# 📥 Step 3: View All Updates
# -------------------------------
def get_updates():
    count = contract.functions.getUpdateCount().call()
    print(f"📦 Total updates: {count}")
    for i in range(count):
        cid, hsh, ts = contract.functions.getUpdate(i).call()
        print(f"[{i}] Client: {cid}, Hash: {hsh}, Timestamp: {ts}")


# -------------------------------
# 🧪 CLI Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit model update to blockchain")
    parser.add_argument("--view", action="store_true", help="View updates")
    parser.add_argument("--region", type=str, help="Region/client ID")
    parser.add_argument("--model_path", type=str, help="Path to .pt file")

    args = parser.parse_args()

    if args.submit:
        if not args.region or not args.model_path:
            print("❌ --submit requires --region and --model_path")
        else:
            send_update(args.region, args.model_path)

    elif args.view:
        get_updates()

    else:
        print("⚠️ No action specified. Use --submit or --view")
