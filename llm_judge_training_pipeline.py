import torch
from datasets import load_dataset

### Data preparation
# Load the dataset from a CSV file
# Replace the path with your actual dataset location
dataset = load_dataset("csv", data_files="/Users/rickytan/Documents/Learn/Kaggle/LLM_judge/llm-classification-finetuning/train.csv")
print(dataset)

# Split the dataset into train (80%) and test (20%) sets
# Setting a seed ensures reproducibility of the split
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Function to process labels based on winner columns
# Assigns numeric labels: 0 for winner_model_a, 1 for winner_model_b, 2 for winner_tie
def process_labels(example):
    if example["winner_model_a"]:
        return {"label": 0}
    elif example["winner_model_b"]:
        return {"label": 1}
    elif example["winner_tie"]:
        return {"label": 2}

# Map the label processing function to the train and test datasets
train_dataset = train_dataset.map(process_labels)
test_dataset = test_dataset.map(process_labels)

# Use embeddings for the categorical features `model_a` and `model_b`
# LabelEncoder converts categorical data to numeric IDs
from sklearn.preprocessing import LabelEncoder

# Initialize label encoders for `model_a` and `model_b`
label_encoder_a = LabelEncoder()
label_encoder_b = LabelEncoder()

# Fit the encoders on the train dataset values
model_a_values = train_dataset["model_a"]
model_b_values = train_dataset["model_b"]

label_encoder_a.fit(model_a_values)  # Fit encoder for `model_a`
label_encoder_b.fit(model_b_values)  # Fit encoder for `model_b`

# Function to encode `model_a` and `model_b` to numeric IDs using the fitted label encoders
def encode_ids(example):
    return {
        "model_a_id": label_encoder_a.transform([example["model_a"]])[0],  # Encode `model_a`
        "model_b_id": label_encoder_b.transform([example["model_b"]])[0],  # Encode `model_b`
    }

# Map the encoding function to the train and test datasets
train_dataset = train_dataset.map(encode_ids)
test_dataset = test_dataset.map(encode_ids)

# Define embeddings for `model_a_id` and `model_b_id`
# Embeddings are used to represent categorical variables in a dense, low-dimensional space
import torch.nn as nn

embedding_size = 16  # Size of the embedding vector for each categorical ID

# Create embeddings for `model_a_id` and `model_b_id`
# The number of embeddings corresponds to the number of unique values in each categorical variable
model_a_embedding = nn.Embedding(num_embeddings=len(label_encoder_a.classes_), embedding_dim=embedding_size)
model_b_embedding = nn.Embedding(num_embeddings=len(label_encoder_b.classes_), embedding_dim=embedding_size)

# Embeddings will be used later in a model to learn relationships between categorical features and textual features



from transformers import AutoTokenizer

### Load the tokenizer for the DistilBERT model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.model_max_length = 512  # Set the maximum token length for the tokenizer

# Print the column names in the dataset to confirm the structure
print(train_dataset.column_names)

# Define a function to tokenize the input data
def tokenize_function(batch):
    # Combine the prompt, response_a, and response_b into a single string with [SEP] tokens as separators
    inputs = [
        f"{prompt} [SEP] {response_a} [SEP] {response_b}"
        for prompt, response_a, response_b in zip(
            batch["prompt"], batch["response_a"], batch["response_b"]
        )
    ]
    # Tokenize the combined inputs with truncation and padding
    # `truncation=True` ensures the sequence doesn't exceed `max_length`
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    # Include additional features in the tokenized dataset
    tokenized["model_a_id"] = batch["model_a_id"]  # Add raw model_a IDs
    tokenized["model_b_id"] = batch["model_b_id"]  # Add raw model_b IDs
    tokenized["labels"] = batch["label"]           # Add labels for classification
    return tokenized

# Apply the tokenize function to the train and test datasets
# `batched=True` processes multiple rows at once for efficiency
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the format of the dataset to PyTorch tensors
# Include only the specified columns for training and testing
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "model_a_id", "model_b_id", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "model_a_id", "model_b_id", "labels"])



### Initialize the combined model including both textual and categorical embedding
import torch
import torch.nn as nn
import os
import json

class CombinedModel(nn.Module):
    def __init__(self, transformer, model_a_embedding, model_b_embedding, embedding_size, num_labels):
        """
        Initializes the combined model, which integrates a transformer with embeddings
        for categorical data and a classifier for final predictions.

        Args:
        - transformer: Hugging Face transformer model (e.g., DistilBERT).
        - model_a_embedding: Embedding layer for model_a categorical feature.
        - model_b_embedding: Embedding layer for model_b categorical feature.
        - embedding_size: Size of the embedding vectors.
        - num_labels: Number of output classes for classification.
        """
        super(CombinedModel, self).__init__()
        self.transformer = transformer
        self.model_a_embedding = model_a_embedding
        self.model_b_embedding = model_b_embedding
        # Define a classifier that combines the transformer output and embeddings
        self.classifier = nn.Linear(transformer.config.hidden_size + 2 * embedding_size, num_labels)

    def forward(self, input_ids, attention_mask, model_a_id, model_b_id, labels=None):
        """
        Forward pass through the combined model.

        Args:
        - input_ids: Tokenized input IDs for the transformer.
        - attention_mask: Attention mask for the transformer.
        - model_a_id: Encoded categorical IDs for model_a.
        - model_b_id: Encoded categorical IDs for model_b.
        - labels: Optional, ground truth labels for calculating loss.

        Returns:
        - Loss and logits if labels are provided, else just logits.
        """
        # Process text data through the transformer
        text_output = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        text_cls = text_output[:, 0, :]  # Extract [CLS] token representation

        # Process categorical data through embedding layers
        a_emb = self.model_a_embedding(model_a_id)  # Embedding for model_a IDs
        b_emb = self.model_b_embedding(model_b_id)  # Embedding for model_b IDs

        # Concatenate the transformer output with embeddings
        combined = torch.cat([text_cls, a_emb, b_emb], dim=1)

        # Pass the combined features through the classifier
        logits = self.classifier(combined)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()  # Define cross-entropy loss function
            loss = loss_fn(logits, labels)

        # Return loss and logits if labels are provided, else only logits
        return (loss, logits) if labels is not None else logits

    def save_pretrained(self, save_directory):
        """
        Save the model to a specified directory in a format compatible with Hugging Face.

        Args:
        - save_directory: Directory path to save the model.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the Hugging Face transformer model
        self.transformer.save_pretrained(save_directory)

        # Save the embeddings and classifier weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save the entire configuration for model reconstruction
        self.transformer.config.to_json_file(os.path.join(save_directory, "config.json"))



### Training and save the model
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel
import os
from tqdm import tqdm

# Load the pretrained transformer model
transformer = AutoModel.from_pretrained("distilbert-base-uncased")

# Extend the positional embeddings to handle longer sequences
# Step 1: Get the original position embedding weights from the transformer
original_position_embeddings = transformer.embeddings.position_embeddings.weight
# Step 2: Create a new position embedding layer to handle 1024 tokens
new_position_embeddings = torch.nn.Embedding(1024, transformer.config.hidden_size)
# Step 3: Copy the original weights for the first 512 positions
new_position_embeddings.weight.data[:512] = original_position_embeddings.data
# Step 4: Initialize the extra positions with the mean of the original weights
new_position_embeddings.weight.data[512:] = original_position_embeddings.data.mean(dim=0)

# Replace the transformer's position embeddings with the new embeddings
transformer.embeddings.position_embeddings = new_position_embeddings
transformer.config.max_position_embeddings = 1024  # Update max position embeddings in the config

# Number of output classes for classification
num_labels = 3
# Initialize the CombinedModel with transformer and embeddings
model = CombinedModel(transformer, model_a_embedding, model_b_embedding, embedding_size, num_labels)

# Set up DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16)

# Define the optimizer (AdamW is commonly used for transformer models)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Detect device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the appropriate device
model.to(device)

# Validation function
def validate(model, val_loader, device):
    """
    Perform validation on the validation dataset.

    Args:
    - model: The combined model.
    - val_loader: DataLoader for the validation set.
    - device: Device to perform computations (CPU/GPU).

    Returns:
    - Average validation loss and accuracy.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification

    # Disable gradient calculations during validation
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", unit="batch"):
            # Move inputs and labels to the correct device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_a_id = batch["model_a_id"].to(device)
            model_b_id = batch["model_b_id"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            _, logits = model(input_ids, attention_mask, model_a_id, model_b_id, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)  # Average loss
    accuracy = correct / total  # Overall accuracy
    return avg_loss, accuracy

# Directory to save model checkpoints
check_point_dir = "./check_point/"
os.makedirs(check_point_dir, exist_ok=True)

# Directory to save the final trained model
model_dir = "./trained_model"

# Training loop
epochs = 1000  # Set to 1000 so that when the accuracy reachs a satisfied level, can maunally stop training.
try:
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        # Iterate through training batches with tqdm for progress display
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            # Move inputs and labels to the correct device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_a_id = batch["model_a_id"].to(device)
            model_b_id = batch["model_b_id"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()  # Reset gradients
            # Forward pass
            loss, logits = model(input_ids, attention_mask, model_a_id, model_b_id, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate loss

        avg_train_loss = total_loss / len(train_loader)  # Average training loss
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        # Perform validation
        val_loss, val_accuracy = validate(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), os.path.join(check_point_dir, f"model_epoch_{epoch + 1}.pt"))
        print(f"Model of epoch {epoch + 1} is saved at {check_point_dir}")

except KeyboardInterrupt:
    # Handle manual interruption (e.g., Ctrl+C) by saving the model
    model_save_path = "/content/drive/My Drive/Colab Notebooks/LLM_judge/ModelTrained"
    os.makedirs(model_save_path, exist_ok=True)

    # Save the trained model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


