from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import pandas as pd
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
import nltk


# sets from the model
model_path = r"C:\Users\Laura\OneDrive\Desktop\ethical_ai\results\checkpoint-2814"
path_test = r"C:\Users\Laura\OneDrive\Desktop\ethical_ai\test_62k.txt"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# test dataset and pre process
test_df = pd.read_csv(path_test, sep="\t", header=None, nrows=15000)
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

test_df["clean_tweet"] = test_df["text"].apply(preprocess_tweet)

# dataset for BERT
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize_function, batched=True)


test_dataset = test_dataset.remove_columns(["label", "text", "clean_tweet"])
test_dataset.set_format("torch")

# load model
model = BertForSequenceClassification.from_pretrained(model_path)
trainer = Trainer(model=model)

output = trainer.predict(test_dataset)
y_pred = np.argmax(output.predictions, axis=-1)
y_true = test_df["label"].values

#confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.show()




#model metrics
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "rmse": rmse
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(test_dataset)

print(results)