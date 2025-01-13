from google.colab import drive
drive.mount('/content/drive')

# Input Text
import csv
import ast

texts = []
with open("/content/drive/My Drive/Colab Notebooks/LLM_judge/test.csv") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header

    for row in csv_reader:
        row_list = []
        for item in row:
            try:
                # Safely evaluate each item as a Python literal
                inner_list = ast.literal_eval(item)
                if isinstance(inner_list, list):  # Check if the evaluated item is a list
                    row_list.append("".join(inner_list))  # Concatenate list items into a single string
            except (ValueError, SyntaxError):  # Skip invalid or non-literal items
                continue
        texts.append(row_list)

print(texts[2])  # Verify the processed data


import torch
from torch.nn.functional import softmax

def inference(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    # Ensure the model and inputs are on the same device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run inference without calculating gradients (faster and uses less memory)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probs = softmax(logits, dim=1)

    # Predict the class with the highest probability
    predicted_classes = torch.argmax(probs, dim=1)

    print("Probabilities:", probs)
    print("Predicted class:", predicted_classes)

# Run inference for each processed text
for string in texts:
    inference(string)
