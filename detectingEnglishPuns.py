import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import random
import torch
import time
import os

from transformers import BertTokenizer, BertForSequenceClassification
from src.custom_testing_sentences import CUSTOM_TESTING_SENTENCES
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datetime import datetime


SEED_VALUE = 42
NUM_EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
DATASET_PATH = "./src/subtask1-homographic-test.xml"
SIMPLE_NON_PUNS = "./src/simple_non_puns.txt"
OUTPUT_DIRECTORY = f"./pun_detection_model-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFICATION_NAMES = ['Non-Pun', 'Pun']
ADD_SIMPLE_NON_PUNS = True   # True = add simple non-puns to SemEval dataset    | False = skip
BALANCE_DATASET = True       # True = use equal amount of puns and non puns     | False = use the imbalanced dataset
CROSS_VALIDATION = True      # True = run Cross Validation                      | False = do not run Cross Validation
CUSTOM_TESTING = True        # True = run Custom Testing                        | False = do not run Custom Testing


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


def split_data(x, y):
    #Training 70%, Validation 15%, Test 15%
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=15 / 85, random_state=42,
                                                     stratify=y_train_val)

    # Training 60%, Validation 20%, Test 20%
    # x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    # x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42,
    #                                                   stratify=y_train_val)

    return x_train, x_val, x_test, y_train, y_val, y_test


def train_epoch(model, optimizer, train_loader, device):
    model.train()
    total_training_loss = 0
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_training_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_training_loss / len(train_loader)


def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            true_labels = batch[2]

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.numpy())

    return all_true_labels, all_predictions


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
            avg_train_loss = train_epoch(fold_model, fold_optimizer, fold_train_loader, DEFAULT_DEVICE)
            print(f"  Fold {fold+1} | Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        fold_labels, fold_preds = evaluate(fold_model, fold_val_loader, DEFAULT_DEVICE)

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


def holdout_validation(model, optimizer, train_loader, val_loader, test_loader, num_epochs, device, report_path, class_names):
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1} of {num_epochs}")
        # TRAINING
        avg_training_loss = train_epoch(model, optimizer, train_loader, device)

        print(f"Average Training Loss for epoch {epoch + 1}: {avg_training_loss:.4f}")
        with open(report_path, "a") as file:
            file.write(f"Average Training Loss for epoch {epoch + 1}: {avg_training_loss:.4f}")
            file.write("\n" + "-" * 45 + "\n")

        # VALIDATION
        val_labels, val_preds = evaluate(model, val_loader, device)
        avg_val_accuracy = sum(p == t for p, t in zip(val_preds, val_labels)) / len(val_labels) if val_labels else 0

        print(f"Validation Accuracy: {avg_val_accuracy:.4f}\n")
        with open(report_path, "a") as file:
            file.write(f"Validation Accuracy: {avg_val_accuracy:.4f}\n")
            file.write("\n" + "-" * 45 + "\n")

    # TESTING
    all_true_labels, all_predictions = evaluate(model, test_loader, device)

    with open(report_path, "a") as file:
        file.write("=" * 45 + "\n")
        file.write("Validation for the new model\n\n")
        file.write(classification_report(all_true_labels, all_predictions, target_names=class_names))
        file.write("=" * 45 + "\n")


if __name__ == '__main__':
    start_time = time.time()
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Using Device: {DEFAULT_DEVICE}")
    print(f"Cuda is {torch.cuda.is_available()}")
    endtime = time.time()
    delta = endtime - start_time
    print(f"Time elapsed: {delta:.2f} seconds")


    # Model and Report Saving
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

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

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

    holdout_validation(
        model=bert_model,
        optimizer=optimizer,
        train_loader=training_dataloader,
        val_loader=validation_dataloader,
        test_loader=testing_dataloader,
        num_epochs=NUM_EPOCHS,
        device=DEFAULT_DEVICE,
        report_path=report_path,
        class_names=CLASSIFICATION_NAMES
    )

    bert_model.save_pretrained(OUTPUT_DIRECTORY)
    bert_tokenizer.save_pretrained(OUTPUT_DIRECTORY)
    print(f"Model and tokenizer saved to {OUTPUT_DIRECTORY}")

    if CUSTOM_TESTING:
        custom_testing()
