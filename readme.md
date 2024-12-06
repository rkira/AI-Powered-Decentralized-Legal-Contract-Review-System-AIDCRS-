## DESCRIPTION ##
The AI-Powered Decentralized Legal Contract Review System (AIDCRS) is a tool that helps users review legal contracts quickly and securely. It uses AI to analyze contract clauses, identify risks, and provide recommendations. By using blockchain and IPFS, it ensures that all contract data is stored securely and cannot be tampered with. This makes reviewing contracts easier and safer for everyone.
OCR Functionality is implemented but not in action.

## prerequisite: ##
Solidity version: 0.8.19
GANACHE & IPFS Desktop App

libraries and dependencies:
`pip install web3 pdfplumber pytesseract pdf2image requests flask tesseract ipfshttpclient transformers torch flask_cors fuzzywuzzy tf-keras accelerate>=0.26.0 transformers[torch] pdfminer.six levenshtein`


## HOW TO RUN ##
1)START GANACHE & IPFS

2)Configure Ganashe :
- Add project(truffle-config.js) to Ganache and check configurations.

3)Copy the contract address from the deployed contract in Ganache.
- compile and deploy the contract(ContractReview.sol) with "truffle compile" and "truffle migrate"
- In server.py replace "Your contract address" with your actual deployed contract address in ganache.

4)train AI model.
- In Directory ..\model, run "train_bert_model.py" to train model and change path locations according to your system.

5)Start Application
NOTE:CHECK and EDIT all path location and add your path in required fields. 
- Execute server.py
- Open localhost site and test.

## Extra INFO ##
- "server.py" and "train_bert_model.py" can be edited to change Output results and fix model accuracy.
- CSV file(risk.csv) contains only 535 rows of data.
- UI interface is avaliable in ..\templates\index.html.
- after training the model will be saved in the ..\model\new_model.
- pip install levenshtein
 removes UserWarning: Using slow pure-python SequenceMatcher. 
 - Uploaded files are stored locally in ..\uploads
 - Once succesfully uploaded the transaction will consist of an ipfs hash.
 For Example:QmYuPPQuuc9ezYQtgTAupLDcLCBn9ZJgsPjG7mUx7qbN8G.
 It can be used with `http://127.0.0.1:8080/ipfs/"insert ipfs hash" ` to view information about the contract.
 - Sample Contracts are avaliable in `..\Legal_Samples` to test are from `https://www.signwell.com/contracts`.

