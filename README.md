# Darija N-Gram Language Model

This project implements a probabilistic n-gram language model for Darija.

## Dataset

I used the `music-data` subfolder from the provided Darija corpus.  
The assignment allows choosing the amount of data used for training, so I trained the model on the first 50,000 tokens to keep the program efficient.

## Model

I used a trigram model, where:

P(w3 | w1, w2)

This means the model predicts the next word based on the previous two words.

## Preprocessing

The preprocessing step:
- converts text to lowercase
- removes URLs
- removes emails
- removes standalone numbers and symbols
- keeps Arabic words and Latin/Arabizi words such as `b7al`, `3lach`, and `9lbi`
- removes very short tokens

This is useful because Darija online text often appears in both Arabic script and Latin transliteration.

## Probability

The model uses Laplace smoothing:

P(w3 | w1, w2) = (Count(w1, w2, w3) + 1) / (Count(w1, w2) + Vocabulary Size)

Laplace smoothing prevents zero probabilities for unseen n-grams.

## Features

The program can:
- load the corpus
- preprocess the text
- train a trigram language model
- calculate word probabilities
- predict the next word
- generate Darija-like text
- change model settings from the command line

## How to Run

```bash
python main.py
```

Useful options:

```bash
python main.py --random-seed 7
python main.py -n 4 --token-limit 100000 --length 20
python main.py --seed "dima dima" --seed "hadchi li"
```

## Tests

```bash
python -m unittest discover
```
