// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLModelStore {
    struct Update {
        string clientId;
        string ipfsHash; // or hash of weights
        uint256 timestamp;
    }

    Update[] public updates;

    event UpdateSubmitted(string clientId, string ipfsHash, uint256 timestamp);

    function submitUpdate(string memory clientId, string memory ipfsHash) public {
        updates.push(Update(clientId, ipfsHash, block.timestamp));
        emit UpdateSubmitted(clientId, ipfsHash, block.timestamp);
    }

    function getUpdateCount() public view returns (uint256) {
        return updates.length;
    }

    function getUpdate(uint256 index) public view returns (string memory, string memory, uint256) {
        require(index < updates.length, "Index out of bounds");
        Update memory update = updates[index];
        return (update.clientId, update.ipfsHash, update.timestamp);
    }

    function getAllUpdates() public view returns (Update[] memory) {
        return updates;
    }
}
