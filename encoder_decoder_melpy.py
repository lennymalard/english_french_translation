# %%

import sys
sys.path.insert(0,'/Users/lenny/Documents/DEV/git/melpy-project')

import melpy.tensor as mt
import numpy as np
import pandas as pd

dataset_df = pd.read_csv("/Users/lenny/Documents/DEV/git/melpy-project/examples/french_english/eng_-french.csv")
dataset = dataset_df.to_numpy()

X = dataset[:100,0]
y = dataset[:100,1]

from melpy.preprocessing import Tokenizer

english_tokenizer = Tokenizer(strategy="word")
french_tokenizer = Tokenizer(strategy="word")

X = np.array(["<SOS> " + data + " <EOS>" for data in X])
y = np.array(["<SOS> " + data + " <EOS>" for data in y])

english_tokenizer.fit_on_texts(X.tolist())
french_tokenizer.fit_on_texts(y.tolist())

X_tokenized = english_tokenizer.texts_to_sequences(X.tolist())
y_tokenized = french_tokenizer.texts_to_sequences(y.tolist())

english_tokenizer.value_index["<pad>"] = len(english_tokenizer.value_index)
english_tokenizer.index_value[len(english_tokenizer.index_value)] = '<pad>'

max_X_len = len(max(X_tokenized, key=len))
max_y_len = len(max(y_tokenized, key=len))

french_tokenizer.value_index["<pad>"] = len(french_tokenizer.value_index)
french_tokenizer.index_value[len(french_tokenizer.index_value)] = '<pad>'

for i in range(len(X_tokenized)):
    for _ in range(max_X_len-len(X_tokenized[i])):
        X_tokenized[i].insert(-1, english_tokenizer.value_index["<pad>"])

    for _ in range(max_y_len-len(y_tokenized[i])):
        y_tokenized[i].insert(-1, french_tokenizer.value_index["<pad>"])

encoder_input = np.array(X_tokenized)
decoder_input = np.array([seq[:-1] for seq in y_tokenized])
decoder_target = np.array([seq[1:] for seq in y_tokenized])

french_vocab_size = len(french_tokenizer.value_index)
english_vocab_size = len(english_tokenizer.value_index)
batch_size = encoder_input.shape[0]

encoder_input = english_tokenizer.one_hot_encode(encoder_input)
decoder_input = french_tokenizer.one_hot_encode(decoder_input)
decoder_target = french_tokenizer.one_hot_encode(decoder_target)

encoder_input = mt.Tensor(encoder_input.reshape(batch_size, max_X_len, -1))
decoder_input = mt.Tensor(decoder_input.reshape(batch_size, max_y_len-1, -1))
decoder_target = mt.Tensor(decoder_target.reshape(batch_size, max_y_len-1, -1))

# %%
from melpy import LSTM, Embedding, Dense, Adam, CategoricalCrossEntropy

class Encoder:
    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = None

        self.embedding = Embedding(inputs.shape[-1], 128)
        self.embedding.inputs = self.inputs

        self.lstm = LSTM(128, 256, return_cell_state=True)

    def forward(self, mask):
        self.lstm.inputs = self.embedding.forward(mask)
        self.outputs = self.lstm.forward(mask)
        return self.outputs

    def backward(self, dX, cell_state_grad, mask):
        dX = self.lstm.backward(dX, cell_state_grad, mask)
        self.embedding.backward(dX, mask)

    def zero_grad(self):
        self.lstm.zero_grad()
        self.embedding.zero_grad()

class Decoder:
    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = None

        self.embedding = Embedding(inputs.shape[-1], 128)
        self.embedding.inputs = self.inputs

        self.lstm = LSTM(128, 256, return_sequences=True)
        self.dense = Dense(256, inputs.shape[-1], activation="softmax")

    def forward(self, context, mask):
        self.lstm.initial_hidden_state = context[0]
        self.lstm.initial_cell_state = context[1]
        self.lstm.inputs = self.embedding.forward(mask)
        self.dense.inputs = self.lstm.forward(mask)
        self.outputs = self.dense.forward(mask)
        return self.outputs

    def backward(self, dX, mask):
        dX = self.dense.backward(dX, mask=mask)
        dX = self.lstm.backward(dX, mask=mask)
        self.embedding.backward(dX, mask=mask)
        return self.lstm.cells[0].sequence_hidden_states[0].grad, self.lstm.cells[0].sequence_cell_states[0].grad

    def zero_grad(self):
        self.lstm.zero_grad()
        self.embedding.zero_grad()
        self.dense.zero_grad()

class Model:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.outputs = None

    def forward(self, encoder_mask=None, decoder_mask=None):
        context = self.encoder.forward(encoder_mask)
        self.outputs = self.decoder.forward(context, decoder_mask)
        return self.outputs

    def backward(self, dX, encoder_mask=None, decoder_mask=None):
        hidden_state_grad, cell_state_grad = self.decoder.backward(dX, decoder_mask)
        self.encoder.backward(hidden_state_grad, cell_state_grad, encoder_mask)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

def create_padding_mask(tokens, tokenizer):
    encoded_pad = tokenizer.one_hot_encode(tokenizer.value_index["<pad>"])
    mask = ~np.all(tokens.array == encoded_pad, axis=-1).astype(bool)
    return mask.astype(np.float64)

encoder_mask = create_padding_mask(encoder_input, english_tokenizer)
decoder_mask = create_padding_mask(decoder_input, french_tokenizer)

encoder = Encoder(encoder_input)
decoder = Decoder(decoder_input)

model = Model(encoder, decoder)

optimizer = Adam(learning_rate=0.01)
loss_function = CategoricalCrossEntropy()

EPOCHS = 10
BATCH_SIZE = 32

STEPS = encoder_input.shape[0] // BATCH_SIZE
if STEPS * BATCH_SIZE < encoder_input.shape[0]:
    STEPS += 1

for epoch in range(EPOCHS):
    for step in range(STEPS):
        input1 = mt.Tensor(encoder_input.array[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                        requires_grad=True)
        input2 = mt.Tensor(decoder_input.array[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                           requires_grad=True)
        target = mt.Tensor(decoder_target.array[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         requires_grad=True)

        encoder_mask = create_padding_mask(input1, english_tokenizer)
        decoder_mask = create_padding_mask(input2, french_tokenizer)

        model.inputs = input1
        model.encoder.inputs = input1
        model.decoder.inputs = input2
        model.encoder.embedding.inputs = model.encoder.inputs
        model.decoder.embedding.inputs = model.decoder.inputs

        model.forward(encoder_mask, decoder_mask)

        loss = loss_function.forward(target, model.outputs, mask=decoder_mask)
        dX = loss_function.backward(mask=decoder_mask)

        model.zero_grad()
        model.backward(dX, encoder_mask, decoder_mask)

        optimizer.update_layer(model.decoder.dense)
        optimizer.update_layer(model.decoder.lstm)
        optimizer.update_layer(model.decoder.embedding)

        optimizer.update_layer(model.encoder.lstm)
        optimizer.update_layer(model.encoder.embedding)

        optimizer.step += 1

        print(f"Epoch: {epoch}/{EPOCHS}; Step: {step}/{STEPS}; Loss: {loss}")

# %%
def predict(X):
    model.inputs = X
    model.encoder.inputs = X
    model.decoder.inputs = mt.Tensor(french_tokenizer.one_hot_encode(french_tokenizer.value_index["<sos>"]).reshape(1, -1, french_vocab_size))
    model.encoder.embedding.inputs = model.encoder.inputs
    model.decoder.embedding.inputs = model.decoder.inputs
    print(model.encoder.inputs.shape)
    print(model.encoder.inputs.grad.shape)
    output = model.forward()
    return output

def translate(text):
    tokens = english_tokenizer.texts_to_sequences(text)[0]
    encoded_tokens = mt.Tensor(english_tokenizer.one_hot_encode(tokens).reshape(1, -1, english_vocab_size))

    prediction = predict(encoded_tokens)

    print(prediction.shape)
    print(np.argmax(prediction.array[-1]))

    sentence = french_tokenizer.index_value[int(np.argmax(prediction.array))]

    return sentence




