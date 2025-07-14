import os
import ipfshttpclient
import json
from web3 import Web3
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import shutil

# -----------------------------
# CONFIGURATION
# -----------------------------
IPFS_API = "/ip4/127.0.0.1/tcp/5001"
ARTIFACT_PATH = "contracts/artifacts/contracts/FLModelStore.sol/FLModelStore.json"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"  # ← Replace this
KEY_PATH = "client/secret.key"

DOWNLOAD_DIR = "server/downloads"
DECRYPTED_DIR = "server/decrypted_weights"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(DECRYPTED_DIR, exist_ok=True)

# -----------------------------
# Load secret AES key
# -----------------------------
def load_key():
    with open(KEY_PATH, "rb") as f:
        return f.read()

# -----------------------------
# AES Decryption
# -----------------------------
def decrypt_file(enc_path, output_path, key):
    with open(enc_path, "rb") as f:
        iv = f.read(16)
        ciphertext = f.read()
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
    with open(output_path, "wb") as f:
        f.write(decrypted)

# -----------------------------
# Web3 and Contract Setup
# -----------------------------
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "❌ Web3 not connected"
with open(ARTIFACT_PATH) as f:
    abi = json.load(f)["abi"]
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

# -----------------------------
# IPFS Client
# -----------------------------
client = ipfshttpclient.connect(IPFS_API)

# -----------------------------
# Fetch and decrypt last N models
# -----------------------------
def fetch_and_decrypt_latest(n=2):
    key = load_key()
    total = contract.functions.getUpdateCount().call()

    if total < n:
        print(f"⚠️ Only {total} updates found, adjusting count...")
        n = total

    print(f"📦 Downloading and decrypting last {n} model(s)...")
    for i in range(total - n, total):
        client_id, cid, _ = contract.functions.getUpdate(i).call()

        tmp_dir = os.path.join(DOWNLOAD_DIR, f"{client_id}.pt.enc")
        dec_path = os.path.join(DECRYPTED_DIR, f"{client_id}.pt")

        if os.path.exists(dec_path):
            print(f"✅ Already decrypted: {client_id}")
            continue

        print(f"⬇️  Downloading from IPFS: {cid} for region: {client_id}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        try:
            client.get(cid, target=tmp_dir)
        except Exception as e:
            print(f"❌ IPFS download failed for {cid}: {e}")
            continue

        # Locate the actual encrypted file (inside CID folder)
        enc_inner = os.path.join(tmp_dir, cid)
        if not os.path.isfile(enc_inner):
            print(f"❌ No valid .pt.enc file found for {client_id}")
            continue

        try:
            print(f"🔓 Decrypting {enc_inner} → {dec_path}")
            decrypt_file(enc_inner, dec_path, key)
            print(f"✅ Decrypted: {client_id}")
        except Exception as e:
            print(f"❌ Decryption failed: {e}")

    print("✅ Done: Downloaded and decrypted recent models.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    fetch_and_decrypt_latest(n=2)
