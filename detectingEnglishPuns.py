from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datetime import datetime

import xml.etree.ElementTree as ET
import pandas as pd
import torch
import os

NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
DATASET_PATH = "./src/subtask1-homographic-test.xml"
OUTPUT_DIRECTORY = f"./pun_detection_model-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels for Classifications
CLASSIFICATION_NAMES = ['Non-Pun', 'Pun']

# Enable Testing using custom Dataset
CUSTOM_TESTING = True  # True = Custom Testing | False = No Custom Testing

# Setup Folder Structure for Model and Report saving
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
report_path = os.path.join(OUTPUT_DIRECTORY, "classification_report.txt")


# Create standard Dataloader using Dataset Splits
def create_dataloader_from_data(x_data, y_data, tokenizer, maxlen, batch_size, shuffle=False):
    encodings = tokenizer(
        x_data['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=maxlen,
        return_tensors='pt'
    )

    ids = encodings['input_ids']
    mask = encodings['attention_mask']
    labs = torch.tensor(y_data.values)
    dataset = TensorDataset(ids, mask, labs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_dataset(dataset_source_file=None):
    if dataset_source_file is not None:
        tree = ET.parse(dataset_source_file)
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
        df_sentences = pd.DataFrame(sentences)
        df_labels = pd.read_csv('./src/subtask1-homographic-test.gold', sep='\t', header=None, names=['id', 'label'])
        return pd.merge(df_sentences, df_labels, on='id')  # merge 2 dataframes into one
    else:
        return None


# for custom testing after the model has been fine-tuned (to check the predictions on new sentences)
def custom_testing():
    fine_tuned_bert_model = BertForSequenceClassification.from_pretrained(OUTPUT_DIRECTORY)
    fine_tuned_bert_tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIRECTORY)
    fine_tuned_bert_model.to(DEFAULT_DEVICE)

    custom_testing_true_labels = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    custom_testing_sentences = [
        "I'm reading a book on anti-gravity. It's impossible to put down.",  # Pun
        "The wedding for the two antennas was terrible, but the reception was amazing.",  # pun
        "The past, the present, and the future walked into a bar. It was tense.",  # 'pun'
        "The weather is very nice today.",  # Not a pun
        "Sven lost interest in reading.",  # Not a pun
        "I asked the AI for a joke, and it responded so promptly.",  # pun
        "Never tell a chemistry joke to a volatile substance; it might overreact.",  # pun
        "I need to remember to buy milk on my way home from work.",  # non pun
        "The mother is feeding a baby but it doesn't like the food.",  # non pun
        "My cat is a good hunter. It is currently hunting my socks.",  # non pun
        "Sue was scared of speaking in public so she took a public speaking class.",  # non-pun
        "The butcher backed into his meat grinder and got a little behind in his work.",  # Pun
        "Why was the obtuse angle so upset? Because it was never right.",  # pun
        "A horse walks into a bar. Several patrons are startled and leave, as a horse in a bar is an unexpected and potentially dangerous situation.",
        # pun
        "If at first you don't succeed, skydiving is probably not for you.",  # non pun
        "I ate all the apples.",  # non pun
        "The monkeys are at home.",  # non pun
        "Teachers do not like correcting homework.",  # non pun
        "I wanted to learn how to drive a stick shift, but I couldn't find the manual.",  # pun
        "I'm trying to organize a hide-and-seek tournament, but it's a disaster. Good players are hard to find."  # pun
    ]

    custom_testing_predictions = []
    fine_tuned_bert_model.eval()
    for sentence, true_label in zip(custom_testing_sentences, custom_testing_true_labels):
        inputs = fine_tuned_bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        ids = inputs['input_ids'].to(DEFAULT_DEVICE)
        mask = inputs['attention_mask'].to(DEFAULT_DEVICE)
        with torch.no_grad():
            out = fine_tuned_bert_model(input_ids=ids, attention_mask=mask)

        prediction_index = torch.argmax(out.logits, dim=1).item()
        custom_testing_predictions.append(prediction_index)
        predicted_class = CLASSIFICATION_NAMES[prediction_index]
        true_class: str = CLASSIFICATION_NAMES[true_label]

        # Document Predictions and Correct Labels
        with open(report_path, "a") as report_file:
            report_file.write(f"\nSentence: {sentence} - Prediction: {predicted_class} - Correct: {true_class}")

        print(f"Sentence: '{sentence}'")
        print(f"Prediction: {predicted_class}\n")

    # classification report for new sentences
    with open(report_path, "a") as report_file:
        report_file.write(f"\n")

    with open(report_path, "a") as report_file:
        report_file.write("=" * 45 + "\n")
        report_file.write("Performance of the model with new sentences\n")
        report_file.write(classification_report(custom_testing_true_labels, custom_testing_predictions,
                                                target_names=['Non-Pun', 'Pun']))
        report_file.write("=" * 45 + "\n")


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_model.to(DEFAULT_DEVICE)
optimizer = AdamW(bert_model.parameters(), lr=LEARNING_RATE, eps=1e-8)

# dataset preparation
df = prepare_dataset(DATASET_PATH)
if df is None:
    exit(f"No Dataset was defined under Path: {DATASET_PATH}")

x = df.drop('label', axis=1)
y = df['label']

#x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
#x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=15 / 85, random_state=42,
#                                                  stratify=y_train_val)

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42,
                                                  stratify=y_train_val)

# Class weights for imbalanced data
class_counts = df['label'].value_counts()
total = len(df)
class_weights = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])]).to(DEFAULT_DEVICE)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
with open(report_path, "w") as file:
    file.write("=" * 70 + "\n")
    file.write(
        f"NUM_EPOCHS: {NUM_EPOCHS}, "
        f"BATCH_SIZE: {BATCH_SIZE}, "
        f"TRAINING_SIZE: {len(y_train)}, "
        f"VAL_SIZE: {len(y_val)}, "
        f"TEST_SIZE: {len(y_test)},"
        f"LEARNING_RATE: {LEARNING_RATE}\n"
    )
    file.write("=" * 70 + "\n\n")

training_dataloader = create_dataloader_from_data(x_train, y_train, bert_tokenizer, MAX_LENGTH, BATCH_SIZE,
                                                  shuffle=True)
validation_dataloader = create_dataloader_from_data(x_val, y_val, bert_tokenizer, MAX_LENGTH, BATCH_SIZE)
testing_dataloader = create_dataloader_from_data(x_test, y_test, bert_tokenizer, MAX_LENGTH, BATCH_SIZE)

# Training and Validation
for epoch in range(NUM_EPOCHS):
    print(f"Starting Epoch {epoch + 1} of {NUM_EPOCHS}")
    # TRAINING
    bert_model.train()
    total_training_loss = 0
    for i, batch in enumerate(training_dataloader):
        input_ids = batch[0].to(DEFAULT_DEVICE)
        attention_mask = batch[1].to(DEFAULT_DEVICE)
        labels = batch[2].to(DEFAULT_DEVICE)
        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_training_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_training_loss = total_training_loss / len(training_dataloader)
    print(f"Average Training Loss for epoch {epoch + 1}: {avg_training_loss:.4f}")
    with open(report_path, "a") as file:
        file.write(f"Average Training Loss for epoch {epoch + 1}: {avg_training_loss:.4f}")
        file.write("\n" + "-" * 45 + "\n")

    # VALIDATION
    bert_model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch[0].to(DEFAULT_DEVICE)
            attention_mask = batch[1].to(DEFAULT_DEVICE)
            labels = batch[2].to(DEFAULT_DEVICE)
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {avg_val_accuracy:.4f}\n")
    with open(report_path, "a") as file:
        file.write(f"Validation Accuracy: {avg_val_accuracy:.4f}\n")
        file.write("\n" + "-" * 45 + "\n")

# TESTING
all_predictions = []
all_true_labels = []
bert_model.eval()
with torch.no_grad():
    for batch in testing_dataloader:
        input_ids = batch[0].to(DEFAULT_DEVICE)
        attention_mask = batch[1].to(DEFAULT_DEVICE)
        true_labels = batch[2]
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        # get predictions with index of the max logit
        predictions = torch.argmax(outputs.logits, dim=-1)
        # move to cpu and convert to numpy
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(true_labels.numpy())

with open(report_path, "a") as file:
    file.write("=" * 45 + "\n")
    file.write("Validation for the new model\n\n")
    file.write(classification_report(all_true_labels, all_predictions, target_names=CLASSIFICATION_NAMES))
    file.write("=" * 45 + "\n")

bert_model.save_pretrained(OUTPUT_DIRECTORY)
bert_tokenizer.save_pretrained(OUTPUT_DIRECTORY)
print(f"Model and tokenizer saved to {OUTPUT_DIRECTORY}")

if CUSTOM_TESTING:
    custom_testing()