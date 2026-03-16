import os
import torch
import numpy as np
import shap
from shap import Explanation
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt

# sets from the model
model_path = r"C:\Users\Laura\OneDrive\Desktop\ethical_ai\results\checkpoint-2814"
output_folder = "xai_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load dataset
dataset = load_dataset('csv', data_files='X_test_limpo.csv')
dataset = dataset.filter(lambda x: x["clean_tweet"] is not None and len(str(x["clean_tweet"]).strip()) > 0)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# model prediction
def predict(texts):
    if not isinstance(texts, list):
        texts = texts.tolist() if hasattr(texts, 'tolist') else [str(texts)]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

#xai
def evaluateXAI(dataset):
    print("starting xai")
    explainer = shap.Explainer(predict, tokenizer)
    
   
    sample_texts = dataset["train"].shuffle(seed=42).select(range(200))["clean_tweet"]
    
    
    shap_values = explainer(sample_texts)

    def save_path(filename):
        return os.path.join(output_folder, filename)

    
    shap_values_pos = shap_values[:, :, 1]

  #global plot
    print("shap summary plot")
    try:
        
        import pandas as pd
        vals = pd.DataFrame(shap_values_pos.values).fillna(0).values
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, show=False)
        plt.savefig(save_path("shap_summary_global.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Summary Plot failed: {e}")

    #force plot
    print("SHAP Force Plots...")
    
    
    try:
        shap.force_plot(
            shap_values_pos.base_values[0], 
            shap_values_pos.values[0], 
            shap_values_pos.data[0],
            matplotlib=True,
            show=False
        )
        plt.savefig(save_path("exemple_0_force.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error Force Plot Local: {e}")

    
    print("HTML")
    try:
        
        html_content = shap.plots.text(shap_values_pos, display=False)
        with open(save_path("index_xai_interativo.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error HTML {e}")

if __name__ == "__main__":
    evaluateXAI(dataset)