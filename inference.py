import torch

from EncoderDecoder import *
from data_preprocessing import tokenize, index_texts, add_special_tokens
from training import get_lengths
from torch.nn.functional import softmax
import spacy
import string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache() if torch.cuda.is_available() else None

preprocessed_data = torch.load('data/preprocessed_data_01.pt', weights_only=True, map_location=device)

en_tokenizer = spacy.load('en_core_web_sm')
fr_tokenizer = spacy.load('fr_core_news_sm')

tokenizer = {
    'en_tokenizer': en_tokenizer,
    'fr_tokenizer': fr_tokenizer,
}

en_vocab = preprocessed_data['en_vocab']
fr_vocab = preprocessed_data['fr_vocab']

vocab = {
    'en_vocab': en_vocab,
    'fr_vocab': fr_vocab,
}

EN_VOCAB_SIZE = len(en_vocab['index_value'])
FR_VOCAB_SIZE = len(fr_vocab['index_value'])

encoder = Encoder(vocab_size=EN_VOCAB_SIZE, pad_token=en_vocab['value_index']['<PAD>'])
decoder = Decoder(vocab_size=FR_VOCAB_SIZE, pad_token=fr_vocab['value_index']['<PAD>'])
model = EncoderDecoder(encoder, decoder)

model.load_state_dict(torch.load('parameters/parameters_02_16_2025-22_42_34.pt', weights_only=True, map_location=device))
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
    if temperature > 0 and not None:
        return softmax(logits/temperature, dim=-1)
    else:
        return softmax(logits, dim=-1)

def predict_next_token(last_token, memory_state, vocab, temperature=0.0):
    token_logits, (h_t, c_t) = model.decoder(last_token, *memory_state, get_lengths(last_token, vocab))
    token_scores = temperature_scaling(logits=token_logits, temperature=temperature)
    if temperature > 0:
        next_token = torch.multinomial(torch.squeeze(token_scores), 1, replacement=True)[None, :]
    else:
        next_token = torch.argmax(token_scores, dim=-1)
    return next_token, (h_t, c_t)

def normalize_output(text_list, vocab):
    for i in range(len(text_list)-1, -1, -1):
        if text_list[i] in ('<SOS>', '<EOS>'):
            text_list.pop(i)

        if text_list[i-1] in string.punctuation and text_list[i].isalpha():
            text_list[i] = str(list(text_list[i])[0].upper()) + "".join(c for c in list(text_list[i])[1:])

        if text_list[i] not in string.punctuation and i != 1:
            text_list.insert(i, " ")

    text_list.pop(0)
    text_list[0] = str(list(text_list[0])[0].upper()) + "".join(c for c in list(text_list[0])[1:])
    return text_list

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

def translate(text, tokenizer=tokenizer, vocab=vocab, model=model, temperature=0.0, token_limit=1000):
    with torch.no_grad():
        check_sentence(text)

        decoder_inputs = torch.tensor([[vocab['fr_vocab']['value_index']['<SOS>']]]).to(device)
        encoder_inputs = torch.tensor(preprocess(text, tokenizer['en_tokenizer'], vocab['en_vocab'])).to(device)
        encoder_lengths = get_lengths(encoder_inputs, en_vocab)
        h_0, c_0 = model.encoder(encoder_inputs, encoder_lengths)

        eos_marks = (
            vocab['fr_vocab']['value_index']['!'],
            vocab['fr_vocab']['value_index']['?'],
            vocab['fr_vocab']['value_index']['.']
        )

        i = 0
        h_t, c_t = h_0, c_0
        while decoder_inputs[0, -1] not in eos_marks:
            next_token, (h_t, c_t) = predict_next_token(decoder_inputs[:, -1:], (h_t, c_t), vocab['fr_vocab'], temperature=temperature)
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=1)

            if i == token_limit:
                break
            else:
                i+=1

        text_list = token2text(decoder_inputs, vocab['fr_vocab'])
        normalized_text_list = normalize_output(text_list, vocab['fr_vocab'])
        generated_text = "".join(text_token for text_token in normalized_text_list)
        return generated_text

print(translate("I am fine.", tokenizer, vocab, model, temperature=0.0, token_limit=10))