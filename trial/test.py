from newspaper import Article

def fetch_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


url = "https://air-news.com/article"
news_text = fetch_article(url)


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(news_text, padding=True, truncation=True, max_length=512, return_tensors="pt")


from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

#  dataset from a collection of daily news
train_dataset = [{"input_ids": tokens["input_ids"], "labels": 1}]

# Training arguments
training_args = TrainingArguments(
    output_dir="./news_model",
    num_train_epochs=1,  # Adjust as needed
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train the model incrementally
trainer.train()


############### Daily Trining Loop ##############

import schedule
import time

def train_daily():
    new_data = fetch_and_prepare_daily_news()
    trainer.train(new_data)
 
schedule.every().day.at("02:00").do(train_daily)

while True:
    schedule.run_pending()
    time.sleep(1)
