from torchmetrics.text import BLEUScore
from nltk.translate import meteor
from nltk.tokenize import word_tokenize
import torch
from training import get_lengths
from inference import translate, token2text, predict_next_token, preprocess, check_sentence, normalize_output
from EncoderDecoder import *
import spacy
from tqdm import tqdm
import sys
from data_preprocessing import tokenize

def translate_test_data(texts, vocabs, tokenizers, verbose=True):
    candidates = []
    print("\nStarting translation...")
    for i in tqdm(range(len(texts)), disable=not verbose, file=sys.stdout):
        candidates.append(translate(texts[i]))
    candidates = tokenize(candidates, tokenizers['fr_tokenizer'])
    return candidates

def add_dim(texts):
    texts = texts.copy()
    for i in range(len(texts)):
        texts[i] = [texts[i]]
    return texts

def token2string(tokens):
    tokens = tokens.copy()
    for i in range(len(tokens)):
       tokens[i] = ' '.join(tokens[i])
    return tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache() if torch.cuda.is_available() else None

preprocessed_data = torch.load('training_data/preprocessed_data_01.pt', weights_only=True)
cleaned_data = torch.load("../evaluation/english_french/data/test_data.pt", weights_only=True)

test_inputs = cleaned_data['inputs']
test_targets = cleaned_data['targets']

en_vocab = preprocessed_data['en_vocab']
fr_vocab = preprocessed_data['fr_vocab']

vocabs = {
    'en_vocab': en_vocab,
    'fr_vocab': fr_vocab
}

en_tokenizer = spacy.load('en_core_web_sm')
fr_tokenizer = spacy.load('fr_core_news_sm')

tokenizers = {
    'en_tokenizer': en_tokenizer,
    'fr_tokenizer': fr_tokenizer,
}

EN_VOCAB_SIZE = len(en_vocab['index_value'])
FR_VOCAB_SIZE = len(fr_vocab['index_value'])

encoder = Encoder(vocab_size=EN_VOCAB_SIZE, pad_token=en_vocab['value_index']['<PAD>'])
decoder = Decoder(vocab_size=FR_VOCAB_SIZE, pad_token=fr_vocab['value_index']['<PAD>'])
model = EncoderDecoder(encoder, decoder)

model.load_state_dict(torch.load("parameters/parameters_02_25_2025-06_16_58.pt", weights_only=True, map_location=device))
model.eval()

candidates = translate_test_data(test_inputs, vocabs, tokenizers)

candidates = token2string(candidates)
test_targets = token2string(test_targets)

test_targets = add_dim(test_targets)

bleu = BLEUScore()
bleu_score = bleu(candidates, test_targets)

print(f"BLEU Score: {format(bleu_score, '.4f')}")

meteor_scores = [meteor([word_tokenize(test_target[0])], word_tokenize(candidate)) for test_target, candidate in zip(test_targets, candidates)]
meteor_score = sum(meteor_scores) / len(meteor_scores)

print(f"METEOR Score: {format(meteor_score, '.4f')}")





