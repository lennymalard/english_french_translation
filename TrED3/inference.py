import torch
from EncoderDecoder import *
from data_preprocessing import tokenize, index_texts, add_special_tokens
from training import get_lengths
from torch.nn.functional import softmax
import spacy
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache() if torch.cuda.is_available() else None

preprocessed_data = torch.load('training_data/preprocessed_data_01.pt', weights_only=True, map_location=device)

en_tokenizer = spacy.load('en_core_web_sm')
fr_tokenizer = spacy.load('fr_core_news_sm')

tokenizers = {
    'en_tokenizer': en_tokenizer,
    'fr_tokenizer': fr_tokenizer,
}

en_vocab = preprocessed_data['en_vocab']
fr_vocab = preprocessed_data['fr_vocab']

vocabs = {
    'en_vocab': en_vocab,
    'fr_vocab': fr_vocab,
}

EN_VOCAB_SIZE = len(en_vocab['index_value'])
FR_VOCAB_SIZE = len(fr_vocab['index_value'])

encoder = Encoder(vocab_size=EN_VOCAB_SIZE, pad_token=en_vocab['value_index']['<PAD>'])
decoder = Decoder(vocab_size=FR_VOCAB_SIZE, pad_token=fr_vocab['value_index']['<PAD>'])
model = EncoderDecoder(encoder, decoder)

model.load_state_dict(
    torch.load('parameters/parameters_02_23_2025-00_24_02.pt', weights_only=True, map_location=device))
model.eval()

def preprocess(text, tokenizer, vocab):
    text = [text]
    tokens = tokenize(text, tokenizer, False)
    indexed_tokens = index_texts(tokens, vocab)
    padded_tokens = add_special_tokens(indexed_tokens, vocab)
    return padded_tokens

def token2text(tokens, vocab):
    tokens = tokens.tolist()
    text = []
    for token in tokens[0]:
        text.append(vocab['index_value'][token])
    return text

def temperature_scaling(logits, temperature):
    if temperature > 0:
        return softmax(logits/temperature, dim=-1)
    return softmax(logits, dim=-1)

def predict_next_token(last_token, decoder_output, memory_state, vocab, temperature=0.0):
    token_logits, (h_t, c_t) = model.decoder(last_token, decoder_output, memory_state, get_lengths(last_token, vocab))
    token_scores = temperature_scaling(logits=token_logits, temperature=temperature)
    if temperature > 0:
        next_token = torch.multinomial(torch.squeeze(token_scores), 1, replacement=True)[None, :]
    else:
        next_token = torch.argmax(token_scores, dim=-1)
    return next_token, (h_t, c_t)

def normalize_output(text_tokens):
    text = " ".join(token for token in text_tokens if token not in ('<SOS>', '<EOS>'))
    text = re.sub(r'\s+([,.!?])', r'\1', text) # Captures ws + punct and replaces with punct.
    text = text[0].upper() + text[1:]
    return text

def check_sentence(text):
    eos_marks = ('!', '?', '.')
    ctr = 0
    for i in range(len(text)):
        if text[i] in eos_marks:
            ctr+=1
        if ctr == 2:
            raise ValueError("text must be a sentence.")
    if ctr == 0:
        raise ValueError("text must end with an end-of-sentence mark.")

def translate(text, temperature=0.6, token_limit=1000):
    with torch.no_grad():
        check_sentence(text)

        decoder_inputs = torch.tensor([[vocabs['fr_vocab']['value_index']['<SOS>']]]).to(device)
        encoder_inputs = torch.tensor(preprocess(text, tokenizers['en_tokenizer'], vocabs['en_vocab'])).to(device)
        encoder_lengths = get_lengths(encoder_inputs, en_vocab)
        encoder_outputs, hidden = model.encoder(encoder_inputs, encoder_lengths)

        eos_marks = (
            vocabs['fr_vocab']['value_index']['!'],
            vocabs['fr_vocab']['value_index']['?'],
            vocabs['fr_vocab']['value_index']['.']
        )

        for _ in range(token_limit):
            next_token, hidden = predict_next_token(decoder_inputs[:, -1:], encoder_outputs, hidden, vocabs['fr_vocab'], temperature=temperature)
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=1)
            if decoder_inputs[0, -1].item() in eos_marks:
                break

        text_list = token2text(decoder_inputs, vocabs['fr_vocab'])
        return normalize_output(text_list)