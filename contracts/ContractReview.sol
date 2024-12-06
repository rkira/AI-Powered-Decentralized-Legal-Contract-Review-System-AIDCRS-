// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ContractReview {
    struct Document {
        string ipfsHash;
        uint256 timestamp;
    }

    mapping(address => Document[]) public userDocuments;

    function addDocument(string memory _ipfsHash) public {
        userDocuments[msg.sender].push(Document(_ipfsHash, block.timestamp));
    }

    function getDocuments(address user) public view returns (Document[] memory) {
        return userDocuments[user];
    }
}
