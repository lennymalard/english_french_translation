import torch
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import sys

def load_dataset(path, batch_slice, input_slice, output_slice):
    dataset_df = pd.read_csv(path)
    dataset = dataset_df.to_numpy()[batch_slice]

    train_inputs = dataset[batch_slice, input_slice]
    train_targets = dataset[batch_slice, output_slice]

    return train_inputs, train_targets

def tokenize(texts, tokenizer, verbose=True):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif isinstance(texts, list):
        pass
    else:
        raise TypeError("texts must be a list or ndarray")

    tokenized_texts = []
    print("\nStarting tokenization...") if verbose else None

    for i in tqdm(range(len(texts)), disable=not verbose, file=sys.stdout):
        doc = tokenizer(texts[i])
        tokens = [token.text.lower() for token in doc]
        tokenized_texts.append(tokens)

    return tokenized_texts

def map_tokens(tokenized_texts):
    value_index = {}
    index_value = {}
    special_tokens = ["<EOS>", "<SOS>", "<PAD>", "<UKN>"]

    for text in tokenized_texts:
        for token in text:
            if token not in value_index.keys():
                value_index[token] = len(value_index)
                index_value[len(index_value)] = token

    for token in special_tokens:
        value_index[token] = len(value_index)
        index_value[len(index_value)] = token

    return {"value_index": value_index, "index_value": index_value}

def index_texts(tokenized_texts, vocab):
    for text in tokenized_texts:
        for i in range(len(text)):
            text[i] = vocab["value_index"][text[i]]
    return tokenized_texts

def add_special_tokens(tokenized_texts, vocab):
    max_length = len(max(tokenized_texts, key=len))

    for i in range(len(tokenized_texts)):
        for _ in range(max_length - len(tokenized_texts[i])):
            tokenized_texts[i].append(vocab["value_index"]["<PAD>"])

        tokenized_texts[i].insert(0, vocab["value_index"]["<SOS>"])
        tokenized_texts[i].append(vocab["value_index"]["<EOS>"])

    return tokenized_texts

def preprocessing_pipeline(path, batch_slice, input_slice, output_slice, tokenizers, device):
    print("\nPreprocessing data...")
    train_inputs, train_targets = load_dataset(path, batch_slice, input_slice, output_slice)

    en_tokenizer = tokenizers["en_tokenizer"]
    fr_tokenizer = tokenizers["fr_tokenizer"]

    tokenized_train_inputs = tokenize(train_inputs, en_tokenizer)
    tokenized_train_targets = tokenize(train_targets, fr_tokenizer)

    en_vocab = map_tokens(tokenized_train_inputs)
    fr_vocab = map_tokens(tokenized_train_targets)

    indexed_train_inputs = index_texts(tokenized_train_inputs, en_vocab)
    indexed_train_targets = index_texts(tokenized_train_targets, fr_vocab)

    padded_train_inputs = add_special_tokens(indexed_train_inputs, en_vocab)
    padded_train_targets = add_special_tokens(indexed_train_targets, fr_vocab)

    train_inputs, train_targets = torch.tensor(padded_train_inputs).to(device), torch.tensor(padded_train_targets).to(device)

    training_data = {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "en_vocab": en_vocab,
        "fr_vocab": fr_vocab,
    }

    torch.save(training_data, "./preprocessed_data.pt")
    print("\nData preprocessing completed.")
    return training_data

def main():
    path = "/Users/lenny/Documents/DEV/python/torch_projects/french_english/eng_-french.csv"
    tokenizers = {"en_tokenizer": spacy.load('en_core_web_sm'), "fr_tokenizer": spacy.load('fr_core_news_sm')}
    batch_slice = slice(None)
    input_slice = 0
    output_slice = 1

    preprocessing_pipeline(path, batch_slice, input_slice, output_slice, tokenizers, device)

if __name__ == "__main__":
    main()