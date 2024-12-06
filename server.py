import os
import json
from flask import Flask, request, jsonify, render_template
from web3 import Web3
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import requests
import re
from uuid import uuid4  # Import uuid for generating unique identifiers
import time  # To ensure that IPFS hash changes with every upload
from uuid import uuid4

app = Flask(__name__)
CORS(app)

# Initialize Web3 for blockchain interactions
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
web3.eth.default_account = web3.eth.accounts[0]

# Load Contract
with open("build/contracts/ContractReview.json") as f:
    contract_json = json.load(f)
    contract_abi = contract_json["abi"]

contract_address = "Your contract address"
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Load your AI model and tokenizer
model_name = "add your path/model/saved_model"  # Path to the trained AI model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the dataset from CSV for contract risk analysis
dataset_path = "add your path/model/training_data/risk.csv"
df = pd.read_csv(dataset_path)

# List of labels corresponding to your model's outputs
labels = ["Legal Risk", "Compliance Risk", "Operational Risk"]  # Example categories, update as needed

# Threshold for match score to filter out irrelevant clauses
THRESHOLD_SCORE = 70  # Only consider clauses with a match score greater than this

def analyze_contract_with_ai(text):
    analysis_results = []

    # Split the input text into individual clauses (assuming each clause is a sentence or part of the contract)
    clauses = text.split('.')
    
    # Iterate over each clause in the document
    for clause in clauses:
        clause = clause.strip()  # Remove any leading/trailing whitespace
        
        if len(clause) == 0 or not is_meaningful_clause(clause):
            continue
        
        best_match = None
        highest_score = 0
        
        # Check each clause against the dataset to find the best match
        for index, row in df.iterrows():
            # Calculate the similarity score between the clause and the clause in the dataset
            score = fuzz.partial_ratio(clause.lower(), row['Clause Text'].lower())
            
            # If the score is better than the current best match, update it
            if score > highest_score:
                best_match = row
                highest_score = score
        
        # Only include if match score is above 0.6 and if the clause is not a single word (filter short, meaningless clauses)
        if best_match is not None and highest_score > 60 and len(clause.split()) > 2:  # More than 2 words
            # Append the analysis result
            analysis_results.append({
                "sentence": clause, 
                "risk_category": best_match['Risk Category'],
                "actionable_risk_points": best_match['Actionable Risk Points'],
                "match_score": highest_score / 100  # Normalize to a score between 0 and 1
            })
    
    return analysis_results

def is_meaningful_clause(clause):
    # Regular expressions to filter out clauses that are metadata, numbers, or references
    meaningless_patterns = [
        r"\b\d+\b",              # Clause contains only numbers
        r"\b[A-Za-z]+\s+[0-9]{1,2}\b",  # Clauses with words followed by small numbers
        r"^\s*$",                # Empty clauses
    ]
    
    for pattern in meaningless_patterns:
        if re.search(pattern, clause):
            return False
    
    # Also, include clauses that contain key legal terms or are structured like legal sentences
    legal_keywords = ['agreement', 'contract', 'terms', 'clause', 'shall', 'obligations', 'compensation', 'consent', 'breach', 'confidentiality', 'parties']
    if any(keyword in clause.lower() for keyword in legal_keywords):
        return True
    
    return False

def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_image(image_file_path):
    text = pytesseract.image_to_string(image_file_path)
    return text

def upload_to_ipfs(file):
    try:
        # Create a unique file name by appending a timestamp and UUID
        unique_filename = f"{str(uuid4())}_{int(time.time())}_{file.filename}"
        file_path = os.path.join('uploads', unique_filename)
        
        # Save the file locally
        os.makedirs('uploads', exist_ok=True)  # Ensure the 'uploads' directory exists
        file.save(file_path)
        
        # Optional: Modify the file content to ensure unique hash (e.g., append metadata)
        with open(file_path, "a") as f:
            f.write(f"\n# Unique Metadata: {unique_filename}")
        
        # Upload the file to IPFS
        with open(file_path, "rb") as f:
            ipfs_response = requests.post("http://127.0.0.1:5001/api/v0/add", files={"file": f})
            ipfs_response.raise_for_status()  # Raise an error for any HTTP issues
            ipfs_hash = ipfs_response.json()["Hash"]
        
        return ipfs_hash
    except Exception as e:
        print(f"Error uploading to IPFS: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_contract', methods=['POST'])
def upload_contract():
    try:
        file = request.files['file']
        file_extension = file.filename.split('.')[-1].lower()

        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Extract text based on file type
        contract_text = ""
        if file_extension == 'pdf':
            contract_text = extract_text_from_pdf(file_path)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            contract_text = extract_text_from_image(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Analyze the contract text with AI
        analysis_results = analyze_contract_with_ai(contract_text)

        # Step 4: Upload to IPFS
        ipfs_hash = upload_to_ipfs(file)

        # Step 5: Store IPFS hash on blockchain
        tx_hash = contract.functions.addDocument(ipfs_hash).transact()
        web3.eth.wait_for_transaction_receipt(tx_hash)

        # Return the results with IPFS hash
        return jsonify({
            "status": "success",
            "analysis": analysis_results,
            "ipfs_hash": ipfs_hash  # Provide the IPFS hash of the uploaded file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deploy_contract', methods=['POST'])
def deploy_contract():
    try:
        # Load contract ABI and bytecode
        with open('ContractReview.json') as f:
            contract_data = json.load(f)
            abi = contract_data['abi']
            bytecode = contract_data['bytecode']

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create 'uploads' folder if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    app.run(debug=True)
