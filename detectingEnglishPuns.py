import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # accuracy_score, recall_score, precision_score, f1_score

NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128

OUTPUT_DIRECTORY = "./pun_detection_model/"
TRAINING = False  # true if Model is not trained yet

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
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if os.path.isdir(OUTPUT_DIRECTORY) and TRAINING is True:  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5,
                    eps = 1e-8
                    )

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
    df = pd.merge(df_sentences, df_labels, on='id')  # merge 2 dataframes into one

    # class weights for imbalanced data
    class_counts = df['label'].value_counts()
    total = len(df)
    class_weights = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])]).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)  # FIX: Use weighted loss


    x = df.drop('label', axis=1) 
    y = df['label'] 

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42, stratify=y)
    
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

    # training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch + 1} of {NUM_EPOCHS}")

        # training
        model.train() 
        total_training_loss = 0
        for i, batch in enumerate(training_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_training_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_training_loss = total_training_loss / len(training_dataloader)
        print(f"Average Training Loss: {avg_training_loss:.4f}")
        
        # validation
        model.eval()  
        total_correct = 0 
        total_samples = 0 

        with torch.no_grad(): 
            for batch in validation_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}\n")

    # testing
    print("Starting Testing")
    all_predictions = []
    all_true_labels = []
    model.eval()

    with torch.no_grad():
        for batch in testing_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            true_labels = batch[2] 
            outputs = model(input_ids, attention_mask=attention_mask)
            # get predictions with index of the max logit
            predictions = torch.argmax(outputs.logits, dim=-1)
            # move to cpu and convert to numpy
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.numpy())

    with open("classification_report.txt", "w") as file:
        file.write(classification_report(all_true_labels, all_predictions, target_names=['Non-Pun', 'Pun']))

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    model.save_pretrained(OUTPUT_DIRECTORY)
    tokenizer.save_pretrained(OUTPUT_DIRECTORY)
    print(f"Model and tokenizer saved to {OUTPUT_DIRECTORY}")

# to check the new saved model
else:
    loaded_model = BertForSequenceClassification.from_pretrained(OUTPUT_DIRECTORY)
    loaded_tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIRECTORY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    new_sentences = [
        "I'm reading a book on anti-gravity. It's impossible to put down.", # Pun
        "The wedding for the two antennas was terrible, but the reception was amazing.", # pun
        "The past, the present, and the future walked into a bar. It was tense.", #'pun'
        "The weather is very nice today.", # Not a pun
        "Sven lost interest in reading.", # Not a pun
        "I asked the AI for a joke, and it responded so promptly.", # pun
        "Never tell a chemistry joke to a volatile substance; it might overreact.", # pun
        "I need to remember to buy milk on my way home from work.", # non pun
        "The mother is feeding a baby but it doesn't like the food.", # non pun
        "My cat is a good hunter. It is currently hunting my socks.", # non pun
        "Sue was scared of speaking in public so she took a public speaking class.", # non-pun
        "The butcher backed into his meat grinder and got a little behind in his work.", # Pun
        "Why was the obtuse angle so upset? Because it was never right.", # pun
        "A horse walks into a bar. Several patrons are startled and leave, as a horse in a bar is an unexpected and potentially dangerous situation.", # pun
        "If at first you don't succeed, skydiving is probably not for you.", # non pun
        "I ate all the apples.",#non pun
        "The monkey are at home.", #non pun
        "Teachers do not like correcting homework."#non pun
    
    ]

    class_names = ['Non-Pun', 'Pun']
    loaded_model.eval()

    for sentence in new_sentences:
        inputs = loaded_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)

        prediction_index = torch.argmax(outputs.logits, dim=1).item()
        predicted_class = class_names[prediction_index]
        
        print(f"Sentence: '{sentence}'")
        print(f"Prediction: {predicted_class}\n")