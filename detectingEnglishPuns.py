from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datetime import datetime
from src.custom_testing_sentences import CUSTOM_TESTING_SENTENCES

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import random
import torch
import time
import os


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

def custom_testing():
    fine_tuned_bert_model = BertForSequenceClassification.from_pretrained(OUTPUT_DIRECTORY)
    fine_tuned_bert_tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIRECTORY)
    fine_tuned_bert_model.to(DEFAULT_DEVICE)

    custom_testing_true_labels = [
        # 50 puns
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 50 non-puns
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]

    custom_testing_predictions = []

    fine_tuned_bert_model.eval()
    output_sentences = ""
    correct_puns = 0
    total_puns = 0
    correct_non_puns = 0
    total_non_puns = 0

    for sentence, true_label in zip(CUSTOM_TESTING_SENTENCES, custom_testing_true_labels):
        inputs = fine_tuned_bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        ids = inputs['input_ids'].to(DEFAULT_DEVICE)
        mask = inputs['attention_mask'].to(DEFAULT_DEVICE)
        with torch.no_grad():
            out = fine_tuned_bert_model(input_ids=ids, attention_mask=mask)

        prediction_index = torch.argmax(out.logits, dim=1).item()
        custom_testing_predictions.append(prediction_index)
        predicted_class = CLASSIFICATION_NAMES[prediction_index]
        true_class: str = CLASSIFICATION_NAMES[true_label]

        output_sentences += f"\nSentence: {sentence} - Prediction: {predicted_class} - Correct: {true_class}"

    for t_label, pred_label in zip(custom_testing_true_labels, custom_testing_predictions):
        if t_label == 1:
            total_puns += 1
            if t_label == pred_label:
                correct_puns += 1
        elif t_label == 0:
            total_non_puns += 1
            if t_label == pred_label:
                correct_non_puns += 1

    puns_predictions = f"Puns: ({correct_puns}/{total_puns} correct)"
    non_puns_predictions = f"Non-Puns: ({correct_non_puns}/{total_non_puns} correct)"

    with open(report_path, "a") as report_file:
        report_file.write("\n\n")
        report_file.write("=" * 75 + "\n")
        report_file.write("Custom Testing Summary\n")
        report_file.write(f"{puns_predictions}\n")
        report_file.write(f"{non_puns_predictions}\n")

    with open(report_path, "a") as report_file:
        report_file.write("=" * 45 + "\n")
        report_file.write("Performance of the model with new sentences\n")
        report_file.write(classification_report(custom_testing_true_labels, custom_testing_predictions, target_names=['Non-Pun', 'Pun']))
        report_file.write("=" * 45 + "\n")

    with open(report_path, "a") as report_file:
        report_file.write(output_sentences)


def add_simple_non_puns(df, SIMPLE_NON_PUNS):
    if os.path.exists(SIMPLE_NON_PUNS):
        with open(SIMPLE_NON_PUNS, 'r', encoding='utf-8') as f:
            simple_sentences = [line.strip() for line in f if line.strip()]

        df_simple = pd.DataFrame({
            'id': [f'simple_{i}' for i in range(len(simple_sentences))],
            'text': simple_sentences,
            'label': 0
        })

        df = pd.concat([df, df_simple], ignore_index=True)
        print(f"Added {len(df_simple)} simple non-puns to the dataset.")
    else:
        print("Error: Simple non-puns file not found.")
    return df


def balance_dataset(df):
    df_puns = df[df['label'] == 1]
    df_non_puns = df[df['label'] == 0]
    print(f"Original distribution: {len(df_puns)} puns, {len(df_non_puns)} non-puns")
    df_puns_sampled = df_puns.sample(n=len(df_non_puns), random_state=42)
    df_balanced = pd.concat([df_puns_sampled, df_non_puns]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(
        f"Balanced distribution: {len(df_balanced[df_balanced['label'] == 1])} puns, {len(df_balanced[df_balanced['label'] == 0])} non-puns")
    return df_balanced


def cross_validation(x, y, bert_tokenizer, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_VALUE)
    cv_results = []

    x_array = x.reset_index(drop=True)
    y_array = y.reset_index(drop=True)

    with open(report_path, "a") as file:
        file.write("=" * 70 + "\n")
        file.write(f"CROSS VALIDATION ({n_splits}-Fold)\n")
        file.write(f"NUM_EPOCHS: {NUM_EPOCHS}, BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}\n")
        file.write("=" * 70 + "\n\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_array, y_array)):
        print(f"\n{'='*30} FOLD {fold+1}/{n_splits} {'='*30}")

        x_fold_train = x_array.iloc[train_idx]
        y_fold_train = y_array.iloc[train_idx]
        x_fold_val   = x_array.iloc[val_idx]
        y_fold_val   = y_array.iloc[val_idx]

        # Reset for each fold
        fold_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        fold_model.to(DEFAULT_DEVICE)
        fold_optimizer = AdamW(fold_model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.01)

        fold_train_loader = create_dataloader_from_data(x_fold_train, y_fold_train, bert_tokenizer, MAX_LENGTH, BATCH_SIZE, shuffle=True)
        fold_val_loader   = create_dataloader_from_data(x_fold_val,   y_fold_val,   bert_tokenizer, MAX_LENGTH, BATCH_SIZE)

        # Training
        for epoch in range(NUM_EPOCHS):
            fold_model.train()
            total_loss = 0
            for batch in fold_train_loader:
                input_ids     = batch[0].to(DEFAULT_DEVICE)
                attention_mask = batch[1].to(DEFAULT_DEVICE)
                labels        = batch[2].to(DEFAULT_DEVICE)
                fold_optimizer.zero_grad()
                outputs = fold_model(input_ids, attention_mask=attention_mask, labels=labels)
                outputs.loss.backward()
                total_loss += outputs.loss.item()
                fold_optimizer.step()
            print(f"  Fold {fold+1} | Epoch {epoch+1} | Avg Train Loss: {total_loss/len(fold_train_loader):.4f}")

        # Evaluation
        fold_model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for batch in fold_val_loader:
                input_ids      = batch[0].to(DEFAULT_DEVICE)
                attention_mask = batch[1].to(DEFAULT_DEVICE)
                true_labels    = batch[2]
                outputs = fold_model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(true_labels.numpy())

        fold_report = classification_report(fold_labels, fold_preds, target_names=CLASSIFICATION_NAMES, output_dict=True)
        fold_accuracy = fold_report['accuracy']
        cv_results.append(fold_report)

        print(f"  Fold {fold+1} Accuracy: {fold_accuracy:.4f}")
        with open(report_path, "a") as file:
            file.write(f"--- Fold {fold+1}/{n_splits} ---\n")
            file.write(classification_report(fold_labels, fold_preds, target_names=CLASSIFICATION_NAMES))
            file.write("\n")

    # CV results

    cv_accuracies = []
    cv_f1_pun = []
    cv_f1_nonpun = []

    for result in cv_results:
        cv_accuracies.append(result['accuracy'])

    for result in cv_results:
        cv_f1_pun.append(result['Pun']['f1-score'])

    for result in cv_results:
        cv_f1_nonpun.append(result['Non-Pun']['f1-score'])


    with open(report_path, "a") as file:
        file.write("=" * 70 + "\n")
        file.write("CROSS VALIDATION SUMMARY\n")
        file.write(f"Accuracy:       {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}\n")
        file.write(f"F1 (Pun):       {np.mean(cv_f1_pun):.4f} ± {np.std(cv_f1_pun):.4f}\n")
        file.write(f"F1 (Non-Pun):   {np.mean(cv_f1_nonpun):.4f} ± {np.std(cv_f1_nonpun):.4f}\n")
        file.write("=" * 70 + "\n\n")

    print(f"\nCV Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")


start_time = time.time()
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

NUM_EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
DATASET_PATH = "./src/subtask1-homographic-test.xml"
SIMPLE_NON_PUNS = "./src/simple_non_puns.txt"
OUTPUT_DIRECTORY = f"./pun_detection_model-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEFAULT_DEVICE}")
print(f"Cuda is {torch.cuda.is_available()}")
endtime = time.time()
delta = endtime - start_time
print(f"Time elapsed: {delta:.2f} seconds")

CLASSIFICATION_NAMES = ['Non-Pun', 'Pun']

ADD_SIMPLE_NON_PUNS = True   # True = add simple non-puns to SemEval dataset    | False = skip
BALANCE_DATASET = True       # True = use equal amount of puns and non puns     | False = use the imbalanced dataset
CROSS_VALIDATION = True      # True = run Cross Validation                      | False = do not run Cross Validation
CUSTOM_TESTING = True        # True = run Custom Testing                        | False = do not run Custom Testing

# Setup Folder Structure for Model and Report Saving
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
report_path = os.path.join(OUTPUT_DIRECTORY, "classification_report.txt")
with open(report_path, "w") as file:
    file.write("")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_model.to(DEFAULT_DEVICE)
optimizer = AdamW(bert_model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay = 0.01)

# Dataset Preparation
df = prepare_dataset(DATASET_PATH)
if df is None:
    exit(f"No Dataset was defined under Path: {DATASET_PATH}")

if ADD_SIMPLE_NON_PUNS:
    df = add_simple_non_puns(df, SIMPLE_NON_PUNS)

if BALANCE_DATASET:
    df = balance_dataset(df)

x = df.drop('label', axis=1)
y = df['label']

if CROSS_VALIDATION:
    cross_validation(x, y, bert_tokenizer)

#Training 70%, Validation 15%, Test 15%
# x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=15 / 85, random_state=42,
#                                                  stratify=y_train_val)

# Training 60%, Validation 20%, Test 20%
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42,
                                                  stratify=y_train_val)

# Class weights for imbalanced data
# class_counts = df['label'].value_counts()
# total = len(df)
# class_weights = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])]).to(DEFAULT_DEVICE)
# loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

with open(report_path, "a") as file:
    file.write("=" * 70 + "\n")
    file.write("HOLD-OUT EVALUATION\n")
    file.write(
        f"NUM_EPOCHS: {NUM_EPOCHS}, "
        f"BATCH_SIZE: {BATCH_SIZE}, "
        f"TRAINING_SIZE: {len(y_train)}, "
        f"VAL_SIZE: {len(y_val)}, "
        f"TEST_SIZE: {len(y_test)},"
        f"LEARNING_RATE: {LEARNING_RATE}\n"
    )
    file.write("=" * 70 + "\n\n")

training_dataloader = create_dataloader_from_data(x_train, y_train, bert_tokenizer, MAX_LENGTH, BATCH_SIZE,shuffle=True)
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