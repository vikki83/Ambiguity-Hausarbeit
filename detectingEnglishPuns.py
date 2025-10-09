import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

tree = ET.parse("./src/subtask1-homographic-test.xml")
root = tree.getroot()
sentences = []

for text_node in root.findall("text"):
    text_id = text_node.get("id")

    words = []
    for word_node in text_node.findall("word"):
        words.append(word_node.text)
    
    full_sentence = ' '.join(words)

    sentences.append({
        "id": text_id,
        "text": full_sentence
    })

# id and text
df_sentences = pd.DataFrame(sentences) 

# id and label
df_labels = pd.read_csv("./src/subtask1-homographic-test.gold", sep='\t', header=None, names=['id', 'label']) 

# merge 2 dataframes into one
df = pd.merge(df_sentences, df_labels, on="id")
#print(df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(
    df["text"].tolist(),     
    truncation=True,          
    padding=True,             
    max_length=128,           
    return_tensors="pt"      
)

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(df["label"].values) 

dataset = TensorDataset(input_ids, attention_mask, labels)
#print(dataset)
#print(len(dataset))
#print(dataset[0])
#print("input_ids:", dataset[0])
#print("attention_mark:", dataset[1])
#print("label:", dataset[2])