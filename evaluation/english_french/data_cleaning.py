import re
import spacy
import torch
from tqdm import tqdm
import sys

preprocessed_data = torch.load('../../TrED1/training_data/preprocessed_data_01.pt', weights_only=True)

tokenizers = {'en_tokenizer': spacy.load('en_core_web_sm'), 'fr_tokenizer': spacy.load('fr_core_news_sm')}
vocabs = {'en_vocab': preprocessed_data['en_vocab'], 'fr_vocab': preprocessed_data['fr_vocab']}

def clean_data(path, tokenizers, vocabs, verbose=True):
    cleaned_rows = []

    digit_pattern = re.compile(r'\d{1,3}\.')
    newline_pattern = re.compile(r'\n')
    punctuation_space_pattern = re.compile(r'(\.!?)(\s+)')
    space_capital_pattern = re.compile(r'(\s)([A-Z])')

    with open(path, 'r') as file:
        print("\nStarting data cleaning...")
        for row in tqdm(file, disable=not verbose, file=sys.stdout):
            if row.startswith('#'):
                continue

            row = digit_pattern.sub('', row)
            row = newline_pattern.sub('', row)
            row = punctuation_space_pattern.sub(r'\1', row)
            row = re.sub(r'\(e\)', '', row)
            row = re.sub(r'\(-euse\)', '', row)
            row = re.sub(r'\(-ère\)', '', row)
            row = re.sub(r'\(ne\)', '', row)
            row = re.sub(r'\(-ève\)', '', row)

            row = re.findall(r'[^|]+', row)
            if len(row) < 2:
                continue

            row[0] = space_capital_pattern.sub(r'\2', row[0], count=1)
            row[1] = space_capital_pattern.sub(r'\2', row[1], count=1)

            en_doc = tokenizers['en_tokenizer'](row[0])
            fr_doc = tokenizers['fr_tokenizer'](row[1])

            en_valid = all(token.text.lower() in vocabs['en_vocab']['index_value'].values() for token in en_doc)
            fr_valid = all(token.text.lower() in vocabs['fr_vocab']['index_value'].values() for token in fr_doc)
            if not (en_valid and fr_valid):
                continue

            cleaned_rows.append(row)

    return cleaned_rows

def split_data(data):
    data1, data2 = [], []
    for list in data:
        data1.append(list[0])
        data2.append(list[1])
    return data1, data2

def tokenize(texts, tokenizer, verbose=True):
    tokenized_texts = []
    print("\nStarting tokenization...") if verbose else None

    for i in tqdm(range(len(texts)), disable=not verbose, file=sys.stdout):
        doc = tokenizer(texts[i])
        tokens = [token.text.lower() for token in doc]
        tokenized_texts.append(tokens)

    return tokenized_texts

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

texts1 = clean_data('data/english-french-translations1.md', tokenizers, vocabs)
texts2 = clean_data('data/english-french-translations2.md', tokenizers, vocabs)
texts3 = clean_data('data/english-french-translations3.md', tokenizers, vocabs)

full_samples = texts1 + texts2 + texts3
inputs, targets = split_data(full_samples)

tokenized_inputs = tokenize(inputs, tokenizers['en_tokenizer'], vocabs['en_vocab'])
tokenized_targets = tokenize(targets, tokenizers['fr_tokenizer'], vocabs['en_vocab'])

"""indexed_inputs = index_texts(tokenized_inputs, vocabs['en_vocab'])
indexed_targets = index_texts(tokenized_targets, vocabs['fr_vocab'])

padded_inputs = add_special_tokens(indexed_inputs, vocabs['en_vocab'])
padded_targets = add_special_tokens(indexed_targets, vocabs['fr_vocab'])

torch_inputs = torch.tensor(padded_inputs)
torch_targets = torch.tensor(padded_targets)"""

dataset = {'inputs': inputs, 'targets': tokenized_targets}

torch.save(dataset, 'data/test_data.pt')