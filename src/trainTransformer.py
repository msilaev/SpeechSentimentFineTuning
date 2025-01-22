import sys
import os
import yaml
import joblib
from dotenv import load_dotenv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# from mlflow.exceptions import MlflowException

import pandas as pd

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def train_transformer():
    
    # Load the IMDB dataset
    dataset = load_dataset('imdb')
    print(dataset.keys())


    # Split the data into train and test
    train_data = dataset['train']
    test_data = dataset['test']    

    # Tokenize the datasets
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    train_data = train_data.rename_column("label", "labels")
    test_data = test_data.rename_column("label", "labels")


    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    # Load the BERT model for binary classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Define the scheduler for learning rate adjustment
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    print("transformer training")
    num_epochs = 3

    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        
            print(batch.keys())

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            progress_bar.set_postfix({"loss": loss.item()})

    # Evaluation loop
    # Evaluation loop
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'label']}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
            all_labels.extend(batch['label'].cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_bert_imdb")
    tokenizer.save_pretrained("./fine_tuned_bert_imdb")






if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    raise Exception("Usage: python train.py <path to params.yaml>")
    #    sys.exit(1)

    #param_yaml_path = sys.argv[1]

    #with open(param_yaml_path) as f:
    #    params_yaml = yaml.safe_load(f)

    train_transformer()
