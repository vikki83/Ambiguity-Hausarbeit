from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import AdamW
from datetime import datetime

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import random
import torch
import os

SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NUM_EPOCHS = 2
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
    custom_testing_sentences = [
        # puns
        "Why don't scientists trust atoms? Because they make up everything.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "I used to be a baker, but I couldn't make enough dough.",
        "I don't trust stairs. They're always up to something.",
        "Did you hear about the restaurant on the moon? Great food, no atmosphere.",
        "Why are skeletons so calm? Because nothing gets under their skin.",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
        "I stayed up all night to see where the sun went. Then it dawned on me.",
        "The lumberjack loved his new computer. He especially liked logging in.",
        "A ghost walked into a bar and ordered a spirit.",
        "I wondered why the frisbee was getting bigger. Then it hit me.",
        "What do you get from a pampered cow? Spoiled milk.",
        "The invention of the shovel was groundbreaking.",
        "I'm trying to learn to handle my fear of elevators. I'm taking steps to avoid them.",
        "I'm reading a horror story in Braille. Something bad is about to happen, I can feel it.",
        "A bee's hair is always sticky because it uses a honeycomb.",
        "What do you call a pig that does karate? A pork chop.",
        "Why are fish so smart? Because they live in schools.",
        "My sea sickness comes in waves.",
        "I'm glad I know sign language, it's pretty handy.",
        "I'm terrified of clocks. They're very alarming.",
        "I told my computer I needed a break, and it said, 'No problem, I'm already in bits.'",
        "A midget psychic escaped from prison. He was a small medium at large.",
        "Why did the coffee file a police report? It got mugged.",
        "I asked the AI for a joke, and it responded so promptly.",
        "That must be a loud musician. He's got a lot of amps.",
        "The calendar maker's days are numbered.",
        "Why did the biologist break up with the physicist? They had no chemistry.",
        "I asked a sign painter for a good pun. He gave me a sign.",
        "The life of a lighthouse keeper is very isolated, but they're always in the spotlight.",
        "The optometrist fell into his lens-grinding machine and made a spectacle of himself.",
        "A police officer ticketed me for my fantastic driving. He said, 'It's a fine.'",
        "I wasn't originally going to get a brain transplant, but then I changed my mind.",
        "He was a carpenter, but he was fired. He just wasn't nailing it.",
        "The magician was walking down the street and turned into a drugstore.",
        "The soldier who survived mustard gas and pepper spray is now a seasoned veteran.",
        "My girlfriend said she was leaving me because I'm obsessed with baseball. I told her she was way off base.",
        "He had a photographic memory, but it was never developed.",
        "I used to be a historian, but I realized there was no future in it.",
        "A prisoner's favorite punctuation mark is a period. It marks the end of his sentence.",
        "The duck told the waiter to put the drink on his bill.",
        "I'm thinking of becoming a pilot. My career would really take off.",
        "The man who survived a fall from a 10-story building was not worried. He was just a little shaken.",
        "Why was the math book sad? It had too many problems.",
        "The past, the present, and the future walked into a bar. It was tense.",
        "The geography teacher was fired for getting lost. He couldn't find his bearings.",
        "What's the difference between a zookeeper and a gambler? One has a stake in the lions, the other has lions at stake.",
        "Why are musicians so bad at fishing? They're always dropping the bass.",
        "I tried to sue the airline for losing my luggage. I lost my case.",
        "The tailor was fired from his job. It seems he just wasn't suited for the work.",

        # non-puns
        # "The train is scheduled to arrive at 5:00 PM.",
        # "My favorite color is blue, but I also like green.",
        # "I need to buy groceries after work.",
        # "The library is closed on Sundays.",
        # "This new software update has several security patches.",
        # "The Earth revolves around the Sun in approximately 365 days.",
        # "I am planning to visit my family next month.",
        # "The dog barked at the mail carrier.",
        # "This book provides a detailed history of the Roman Empire.",
        # "She is learning to play the piano.",
        # "The meeting was postponed due to a scheduling conflict.",
        # "Remember to charge your phone before you leave.",
        # "The recipe requires two cups of flour and one cup of sugar.",
        # "Mount Everest is the tallest mountain in the world.",
        # "He wore a heavy coat because it was cold outside.",
        # "The children are playing in the park.",
        # "I prefer to drink coffee in the morning.",
        # "The car's engine is making a strange noise.",
        # "We watched a documentary about penguins last night.",
        # "The final exam will cover all chapters from the textbook.",
        # "The sky is very clear tonight, so we can see the stars.",
        # "My computer is running slowly; I should probably restart it.",
        # "The restaurant's special of the day is grilled salmon.",
        # "She forgot her umbrella and got wet in the rain.",
        # "The presentation will begin in ten minutes.",
        # "Many people commute to the city for work.",
        # "The museum's new exhibit features modern art.",
        # "He is studying to become a mechanical engineer.",
        # "The traffic was heavy on the highway this morning.",
        # "I have a dentist appointment on Friday.",
        # "The new policy will take effect on the first of the month.",
        # "A horse walks into a bar. Several patrons are startled and leave, as a horse in a bar is an unexpected and potentially dangerous situation.",
        # "The company reported its quarterly earnings today.",
        # "She is painting the walls of her living room.",
        # "The store will close at 9:00 PM tonight.",
        # "I need to get my car's oil changed soon.",
        # "The national park is home to many different species of wildlife.",
        # "He is allergic to peanuts.",
        # "The flight was delayed for two hours.",
        # "The package should be delivered by tomorrow afternoon.",
        # "I finished reading the novel yesterday.",
        # "The project deadline is next Wednesday.",
        # "The wind is blowing strongly from the west.",
        # "She is training for a marathon.",
        # "The phone's battery is almost dead.",
        # "The baby is sleeping in his crib.",
        # "The monkeys are at home",
        # "The government announced new environmental regulations.",
        # "He is taking a course on statistics and data analysis.",
        # "The cat is sitting on the windowsill."
        "Actions speak louder than words.",
        "The early bird catches the worm.",
        "A stitch in time saves nine.",
        "Don't count your chickens before they hatch.",
        "A watched pot never boils.",
        "The grass is always greener on the other side.",
        "Don't put all your eggs in one basket.",
        "Every cloud has a silver lining.",
        "A bird in the hand is worth two in the bush.",
        "Better late than never.",
        "Birds of a feather flock together.",
        "Haste makes waste.",
        "Curiosity killed the cat.",
        "Don't judge a book by its cover.",
        "If it ain't broke, don't fix it.",
        "Practice makes perfect.",
        "An apple a day keeps the doctor away.",
        "Absence makes the heart grow fonder.",
        "It's raining cats and dogs.",
        "He let the cat out of the bag.",
        "She spilled the beans.",
        "You have to bite the bullet.",
        "He hit the nail on the head.",
        "That's the best of both worlds.",
        "Don't cry over spilled milk.",
        "Every dog has its day.",
        "That ship has sailed.",
        "We'll cross that bridge when we come to it.",
        "To kill two birds with one stone.",
        "He's sitting on the fence.",
        "The pen is mightier than the sword.",
        "A mind is like a parachute; it doesn't work if it's not open.",
        "A clear conscience is the sign of a bad memory.",
        "We are all in the gutter, but some of us are looking at the stars.",
        "I can resist everything except temptation.",
        "Always forgive your enemies; nothing annoys them so much.",
        "A committee is a group that keeps minutes and wastes hours.",
        "Experience is the name everyone gives to their mistakes.",
        "A man's home is his castle.",
        "Politics makes strange bedfellows.",
        "Misery loves company.",
        "Why do they call it rush hour when nothing moves?",
        "A cynic knows the price of everything and the value of nothing.",
        "The road to success is always under construction.",
        "Money is the root of all evil.",
        "An ounce of prevention is worth a pound of cure.",
        "I'm not superstitious, but I am a little stitious.",
        "A closed mouth gathers no foot.",
        "Time flies",
        "They wanted to paint the town red",
        "War is too important to be left to the generals."
    ]

    custom_testing_predictions = []

    fine_tuned_bert_model.eval()
    output_sentences = ""
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

        output_sentences += f"\nSentence: {sentence} - Prediction: {predicted_class} - Correct: {true_class}"

        print(f"Sentence: '{sentence}'")
        print(f"Prediction: {predicted_class}\n")

        correct_puns = 0
        total_puns = 0
        correct_non_puns = 0
        total_non_puns = 0

        for true_label, pred_label in zip(custom_testing_true_labels, custom_testing_predictions):
            if true_label == 1:
                total_puns += 1
                if true_label == pred_label:
                    correct_puns += 1
            elif true_label == 0:
                total_non_puns += 1
                if true_label == pred_label:
                    correct_non_puns += 1

    puns_predictions = f"Puns: ({correct_puns}/{total_puns} correct)"
    non_puns_predictions = f"Non-Puns: ({correct_non_puns}/{total_non_puns} correct)"

    with open(report_path, "a") as report_file:
        report_file.write("\n\n")
        report_file.write("=" * 75 + "\n")
        report_file.write("Custom Testing Summary\n")
        report_file.write(f"{puns_predictions}\n")
        report_file.write(f"{non_puns_predictions}\n")

    # classification report for new sentences
    with open(report_path, "a") as report_file:
        report_file.write(f"\n")

    with open(report_path, "a") as report_file:
        report_file.write("=" * 45 + "\n")
        report_file.write("Performance of the model with new sentences\n")
        report_file.write(classification_report(custom_testing_true_labels, custom_testing_predictions,
                                                target_names=['Non-Pun', 'Pun']))
        report_file.write("=" * 45 + "\n")

    with open(report_path, "a") as report_file:
        report_file.write(output_sentences)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_model.to(DEFAULT_DEVICE)
optimizer = AdamW(bert_model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay = 0.01)

# dataset preparation
df = prepare_dataset(DATASET_PATH)
if df is None:
    exit(f"No Dataset was defined under Path: {DATASET_PATH}")

# для не балансированного датасета
x = df.drop('label', axis=1)
y = df['label']

# df_puns = df[df['label'] == 1]
# df_non_puns = df[df['label'] == 0]
#
# print(f"Original distribution: {len(df_puns)} puns, {len(df_non_puns)} non-puns")
#
# # equal  amount of puns and non puns
# df_puns_sampled = df_puns.sample(n=len(df_non_puns), random_state=42)
# df_balanced = pd.concat([df_puns_sampled, df_non_puns]).sample(frac=1, random_state=42).reset_index(drop=True)
#
# print(
#     f"Balanced distribution: {len(df_balanced[df_balanced['label'] == 1])} puns, {len(df_balanced[df_balanced['label'] == 0])} non-puns")
#
# x = df_balanced.drop('label', axis=1)
# y = df_balanced['label']

# Training 70%, Validation 15%, Test 15%
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=15 / 85, random_state=42,
                                                 stratify=y_train_val)

# Training 60%, Validation 20%, Test 20%
# x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42,
#                                                   stratify=y_train_val)

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