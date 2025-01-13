# Categ_Text_EMB_FineTuned-distilbert-base-uncased-LLM-Judge

# LLM Preference Prediction

This repository contains my solution to the Kaggle competition **"Predicting User Preferences in LLM Responses"**. The goal of the competition is to predict which responses users prefer in a head-to-head battle between two chatbot models, focusing on improving reward models for reinforcement learning from human feedback (RLHF).

## Problem Description

Large Language Models (LLMs) are increasingly shaping the way we interact with AI systems. However, user satisfaction is influenced by biases like position preference, verbosity, or self-promotion, making preference prediction a challenging task. This competition provides real-world data collected from Chatbot Arena to help bridge the gap between LLM capability and human preference. The task is to predict user preferences based on provided input prompts and chatbot responses.

## Solution Overview

### Approach

The key insight I observed from the dataset is that the categorical variables `model_a` and `model_b` serve as predictive features but cannot be directly treated as textual data. Unlike the textual features `prompt`, `response_a`, and `response_b`, the categorical features represent discrete chatbot models and require a different processing strategy. To address this, I used **embedding layers** to represent `model_a` and `model_b` in a dense, low-dimensional space, allowing the model to learn relationships between categorical and textual features effectively.

### Model Architecture

The solution combines the power of a pretrained **transformer model** with categorical embeddings:

1. **Textual Features**: The `prompt`, `response_a`, and `response_b` were tokenized using the `distilbert-base-uncased` tokenizer and processed through a pretrained **DistilBERT** model. 
   
2. **Categorical Features**: The categorical variables `model_a` and `model_b` were encoded into numeric IDs and processed using separate embedding layers.

3. **Combined Features**: The outputs from the transformer model (textual features) and the embeddings (categorical features) were concatenated and passed through a fully connected classifier to predict user preference.

### Why **DistilBERT**?

I selected the pretrained **DistilBERT** model for the following reasons:
- **Efficiency**: DistilBERT is a lightweight model derived from BERT that maintains 97% of its language understanding capabilities while being 60% faster and using 40% less memory.
- **Versatility**: Its ability to handle diverse NLP tasks makes it a solid foundation for this preference prediction task.
- **Token Limit**: DistilBERT supports long input sequences, and I extended its positional embeddings to accommodate the maximum token length in this dataset.

### Training Details

- **Datasets**: The dataset was split into **80% training** and **20% testing**. The categorical embeddings were trained alongside the transformer model for joint optimization.
- **Loss Function**: Cross-entropy loss was used to handle the multi-class classification task (0 for `model_a` preferred, 1 for `model_b` preferred, 2 for `tie`).
- **Optimizer**: The model was trained using the AdamW optimizer with a learning rate of `5e-5`.
- **Validation**: At the end of each epoch, validation loss and accuracy were calculated to evaluate the model's performance. These metrics are included in the repository.

### Results

- The model was trained for multiple epochs, and accuracy metrics were calculated for each epoch. The final trained model and checkpoint files are included in this repository. You are welcome to experiment with these results or fine-tune the model further.

## Repository Structure
├── model/ # Directory containing the final trained model 
├── checkpoints/ # Checkpoints for each epoch 
├── scripts/ # Python scripts for data preprocessing, training, and inference 
├── README.md # Project documentation


### Key Scripts
- **Data Preprocessing**: The dataset is processed to encode `model_a` and `model_b` as embeddings, while tokenizing textual data using DistilBERT's tokenizer.
- **Model Definition**: The combined model integrates a pretrained DistilBERT with categorical embeddings.
- **Training**: Includes training and validation loops, along with saving checkpoints for each epoch.
- **Inference**: Allows prediction on new data and outputs probabilities for each class.

## How to Use
1. The training data can be found on Kaggle competition page: https://www.kaggle.com/competitions/llm-classification-finetuning/data
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-preference-prediction.git
   cd llm-preference-prediction
2. Install dependencies:
   pip install -r requirements.txt
3. Train the model:
   python scripts/train.py
4. Run inference:
   python scripts/inference.py --input test.csv

Challenge This Solution

The datasets and the training logs, including accuracy and validation loss per epoch, are included in this repository. I invite you to analyze, fine-tune, or challenge these results by experimenting with different architectures, hyperparameters, or datasets. Your contributions are welcome!
Acknowledgments

Special thanks to Kaggle and the competition organizers for providing this exciting real-world dataset.
Hugging Face for making state-of-the-art models like DistilBERT accessible and easy to integrate.
