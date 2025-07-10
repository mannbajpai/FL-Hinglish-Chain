from web3 import Web3
import json
import os
from datetime import datetime

# 1. Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "❌ Web3 connection failed"

# 2. Set default account for sending transactions
w3.eth.default_account = w3.eth.accounts[0]  # Replace if needed

# 3. Load contract ABI and address
with open("contracts/artifacts/contracts/FLModelStore.sol/FLModelStore.json") as f:
    contract_json = json.load(f)
    abi = contract_json["abi"]

# Replace with your deployed contract address
contract_address = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"

contract = w3.eth.contract(address=contract_address, abi=abi)


# 🔁 Submit model update
def submit_update(client_id: str, model_hash: str):
    tx_hash = contract.functions.submitUpdate(client_id, model_hash).transact()
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Update submitted in tx: {receipt.transactionHash.hex()}")


# 📥 Retrieve all updates
def get_all_updates():
    count = contract.functions.getUpdateCount().call()
    print(f"📦 Total updates: {count}")
    for i in range(count):
        cid, hsh, ts = contract.functions.getUpdate(i).call()
        print(f"👤 Client: {cid}, 🔗 Hash: {hsh}, 🕒 Time: {datetime.fromtimestamp(ts)}")


# 🧪 Example usage
if __name__ == "__main__":
    # Example 1: Submit dummy model update
    dummy_hash = "abc123def456"
    submit_update("Region1", dummy_hash)

    # Example 2: Read all updates
    get_all_updates()
