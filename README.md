# Ambiguity in Humor: Detecting English Puns

Term paper project for the course **Ambiguity**.

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) on the
[SemEval-2017 Task 7](https://alt.qcri.org/semeval2017/task7/) homographic pun dataset
to classify English sentences as **pun** or **non-pun**. The data is split into
training, development and test sets; hyperparameters are selected on the development
set, and the held-out test set is evaluated only once, on the final model. In addition,
the classifier is evaluated on a hand-written control set of 100 sentences to test how
well it generalizes beyond the SemEval data.
