import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

nltk.download('stopwords')

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

#preprocessing data
path_train = r"C:\Users\Laura\OneDrive\Desktop\ethical_ai\train_150k.txt"
path_test = r"C:\Users\Laura\OneDrive\Desktop\ethical_ai\test_62k.txt"


train_df = pd.read_csv(path_train, sep="\t", header=None, nrows=15000)
test_df = pd.read_csv(path_test, sep="\t", header=None, nrows=15000)

train_df.columns = ["label", "text"]
test_df.columns = ["label", "text"]
stop_words = set(stopwords.words("english"))

def preprocess_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


train_df["clean_tweet"] = train_df["text"].apply(preprocess_tweet)
test_df["clean_tweet"] = test_df["text"].apply(preprocess_tweet)


X_train = train_df["clean_tweet"]
y_train = train_df["label"]
X_test = test_df["clean_tweet"]
y_test = test_df["label"]

#hugging face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

#BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

#training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    
    
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,    
    gradient_accumulation_steps=8,   
    
    fp16=True,                       
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none"                
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

#evaluate
trainer.evaluate()