import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW

BATCH_SIZE = 16
MAX_LENGTH = 128

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def create_dataloader_from_data(x_data, y_data, tokenizer, maxlen, batch_size, shuffle=False):
    encodings = tokenizer(
        x_data['text'].tolist(),     
        truncation=True,          
        padding=True,             
        max_length=maxlen,           
        return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(y_data.values) 
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    #print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

tree = ET.parse('./src/subtask1-homographic-test.xml')
root = tree.getroot()
sentences = []

for text_node in root.findall('text'):
    text_id = text_node.get('id')

    words = []
    for word_node in text_node.findall('word'):
        words.append(word_node.text)
    
    full_sentence = ' '.join(words)

    sentences.append({
        'id': text_id,
        'text': full_sentence
    })

# id and text
df_sentences = pd.DataFrame(sentences) 

# id and label
df_labels = pd.read_csv('./src/subtask1-homographic-test.gold', sep='\t', header=None, names=['id', 'label']) 

# merge 2 dataframes into one
df = pd.merge(df_sentences, df_labels, on='id')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

x = df.drop('label', axis=1) 
y = df['label'] 

# split Training+Validation (85) and Test (15)
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42, stratify=y)

# split Training+Validation to Training (70) and Validation (15)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=15/85, random_state=42, stratify=y_train_val)

training_dataloader = create_dataloader_from_data(x_train, 
    y_train, 
    tokenizer, 
    MAX_LENGTH,
    BATCH_SIZE,
    shuffle=True
)

validation_dataloader = create_dataloader_from_data(x_val, 
    y_val, 
    tokenizer, 
    MAX_LENGTH,
    BATCH_SIZE
)

testing_dataloader = create_dataloader_from_data(x_test, 
    y_test, 
    tokenizer, 
    MAX_LENGTH,
    BATCH_SIZE
)

#batches len
#print(len(training_dataloader)) 
#print(len(validation_dataloader)) 
#print(len(testing_dataloader)) 

optimizer = AdamW(bert_model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

loss_function = torch.nn.CrossEntropyLoss()