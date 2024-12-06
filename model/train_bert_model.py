from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Load and preprocess your dataset
df = pd.read_csv("C:/Users/major/OneDrive/Desktop/Github_AIDCRS/model/training_data/risk.csv", on_bad_lines='skip')

# Check the first few rows to ensure the data is correctly loaded
print(df.head())

# Create label dictionary to map 'Risk Category' to integers
label_dict = {category: idx for idx, category in enumerate(df['Risk Category'].unique())}

# Map 'Risk Category' to numeric labels
df['Risk Category'] = df['Risk Category'].map(label_dict)

# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the 'Clause Text' column (input text)
tokenized_dataset = tokenizer(df['Clause Text'].tolist(), padding=True, truncation=True, max_length=512)

# Convert to Hugging Face dataset
tokenized_dataset = Dataset.from_dict(tokenized_dataset)

# Add the 'labels' column
tokenized_dataset = tokenized_dataset.add_column('labels', df['Risk Category'])

# Split the dataset into train and test sets (80/20 split)
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Extract train and eval datasets
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

# Start training
trainer.train()

# Save the trained model and tokenizer to a specified folder after training
save_path = "./new_model"  # specify the path where you want to save the model
model.save_pretrained(save_path)  # Save the model's weights and configuration
tokenizer.save_pretrained(save_path)  # Save the tokenizer files

print(f"Model and tokenizer saved to {save_path}")
