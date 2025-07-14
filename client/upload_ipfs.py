# client/upload_ipfs.py
import ipfshttpclient
import os
import argparse
from web3 import Web3
import json
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

# -----------------------------
# CONFIGURATION
# -----------------------------
IPFS_API = "/ip4/127.0.0.1/tcp/5001"
MODEL_DIR = "client/weights"
ENCRYPTED_DIR = "client/encrypted"
KEY_PATH = "client/secret.key"
ARTIFACT_PATH = "contracts/artifacts/contracts/FLModelStore.sol/FLModelStore.json"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"  # 🔁 Replace this

# Connect to Web3 local node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "❌ Web3 not connected"
w3.eth.default_account = w3.eth.accounts[0]

# Load contract ABI
with open(ARTIFACT_PATH) as f:
    abi = json.load(f)["abi"]
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# Connect to IPFS
client = ipfshttpclient.connect(IPFS_API)

# Ensure encrypted directory exists
os.makedirs(ENCRYPTED_DIR, exist_ok=True)

# -----------------------------
# Encrypt model with AES
# -----------------------------
def generate_key():
    key = get_random_bytes(16)  # AES-128
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    return key

def load_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return f.read()
    else:
        return generate_key()

def encrypt_file(input_path, output_path, key):
    cipher = AES.new(key, AES.MODE_CBC)
    with open(input_path, "rb") as f:
        data = f.read()
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    with open(output_path, "wb") as f:
        f.write(cipher.iv + ct_bytes)

# -----------------------------
# Upload model to IPFS and store CID on blockchain
# -----------------------------
def upload_model(region, model_path):
    assert os.path.exists(model_path), f"❌ Model not found: {model_path}"

    key = load_key()
    encrypted_path = os.path.join(ENCRYPTED_DIR, f"{region}.pt.enc")
    encrypt_file(model_path, encrypted_path, key)

    print(f"🚀 Uploading encrypted model to IPFS: {encrypted_path}")
    res = client.add(encrypted_path)
    cid = res["Hash"]
    print(f"✅ IPFS CID: {cid}")

    print(f"📤 Storing (region, CID) on blockchain...")
    tx = contract.functions.submitUpdate(region, cid).transact()
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    print(f"✅ Stored on chain in tx: {receipt.transactionHash.hex()}")

# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Client/region name")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    args = parser.parse_args()

    upload_model(args.region, args.model)
