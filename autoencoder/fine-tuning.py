# Summary: Fine-tuning a BERT Model for Single-Sentence Binary Classification

# In this section, we demonstrate how to fine-tune a pre-trained BERT model for sentiment analysis using the IMDb dataset. The pre-trained model used is DistilBertForSequenceClassification, a distilled version of BERT with a classification head at the top.
# Key Steps

# 1. Setup and Initialization:
# Determine whether to use a GPU or CPU for training.
# Load the DistilBertForSequenceClassification model and tokenizer.

# 2. Dataset Preparation:
# Load the IMDb dataset for training, validation, and testing.
# Tokenize the datasets using the DistilBertTokenizerFast class.

# 3. Training Setup:
# Use the Trainer and TrainingArguments classes from the transformers library for training.
# Specify hyperparameters like the batch size, number of epochs, warmup steps, and weight decay.
# Define a compute_metrics function to calculate evaluation metrics like accuracy and F1 score.

# 4. Training and Evaluation:
# Create a Trainer object with the model, training arguments, datasets, and metrics function.
# Train the model and monitor metrics using logs.

# 5. Inference:
# Test the model's predictions using a get_prediction function.
# Save the fine-tuned model and use the Pipeline API to simplify predictions.

# 6. Results:
# The fine-tuned model achieved high accuracy and F1 scores on the IMDb dataset.
# The model is capable of classifying positive and negative sentiment reliably.
# Setup device


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Import model and tokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(
    model_path, id2label={0: "NEG", 1: "POS"}, label2id={"NEG": 0, "POS": 1}
)

# Load IMDb dataset
from datasets import load_dataset
imdb_train = load_dataset('imdb', split="train")
imdb_test = load_dataset('imdb', split="test[:6250]+test[-6250:]")
imdb_val = load_dataset('imdb', split="test[6250:12500]+test[-12500:-6250]")

# Tokenize datasets
enc_train = imdb_train.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True)
enc_test = imdb_test.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True)
enc_val = imdb_val.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True)

# Define training arguments
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir='./MyIMDBModel',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./logs',
    logging_steps=200,
    evaluation_strategy='steps',
    fp16=cuda.is_available(),
    load_best_model_at_end=True
)

# Define metrics calculation function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

# Initialize and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    compute_metrics=compute_metrics
)
results = trainer.train()

# Save the best model
model_save_path = "MyBestIMDBModel"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Perform inference using the saved model
from transformers import pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(nlp("The movie was very impressive")) 
