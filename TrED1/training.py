# %%
import torch
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

torch.cuda.empty_cache() if torch.cuda.is_available() else None

from utils.data import *
from EncoderDecoder import *
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader

FR_VOCAB_SIZE = None
EN_VOCAB_SIZE = None
BATCH_SIZE = 64

def create_padding_mask(tensor, vocab):
    return tensor == vocab["value_index"]["<PAD>"]

def get_lengths(tensor, vocab):
    return (tensor != vocab["value_index"]["<PAD>"]).int().sum(dim=1)

def train(model, dataloader, criterion, optimizer, epochs, device):
    model.train()

    print("\nTraining...")
    for epoch in range(epochs):
        for batch in (step_bar := tqdm(dataloader, disable=False ,file=sys.stdout)):
            step_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            step_bar.set_postfix({"loss": loss.item() if 'loss' in locals() else 1e7, "device": device})

            enc_input_batch = batch['training_data']['encoder_inputs'].to(device)
            dec_input_batch = batch['training_data']['decoder_inputs'].to(device)
            dec_target_batch = batch['training_data']['decoder_targets'].to(device)

            enc_lengths_batch = batch['lengths']['encoder_lengths'].to('cpu')
            dec_lengths_batch = batch['lengths']['decoder_lengths'].to('cpu')

            outputs = model.forward(enc_input_batch, dec_input_batch, enc_lengths_batch, dec_lengths_batch).float()

            outputs= outputs.view(-1, outputs.size(-1))
            dec_target_batch = dec_target_batch.view(-1)

            loss = criterion(outputs, dec_target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training complete.")
    return model.state_dict()

def main():
    global FR_VOCAB_SIZE
    global EN_VOCAB_SIZE
    global BATCH_SIZE

    preprocessed_data = torch.load('training_data/preprocessed_data_01.pt', weights_only=True, map_location=torch.device(device))

    train_inputs = preprocessed_data["train_inputs"]
    train_targets = preprocessed_data["train_targets"]

    en_vocab = preprocessed_data['en_vocab']
    fr_vocab = preprocessed_data['fr_vocab']

    encoder_inputs = train_inputs
    decoder_inputs = train_targets[:, :-1]
    decoder_targets = train_targets[:, 1:]

    if FR_VOCAB_SIZE is None:
        FR_VOCAB_SIZE = len(fr_vocab["value_index"])
        print(f"FR vocab size: {FR_VOCAB_SIZE}.")

    if EN_VOCAB_SIZE is None:
        EN_VOCAB_SIZE = len(en_vocab["value_index"])
        print(f"EN vocab size: {EN_VOCAB_SIZE}.")

    encoder_max_seq_length = encoder_inputs.size(1)
    decoder_max_seq_length = decoder_inputs.size(1)

    encoder_sequence_lengths =  get_lengths(encoder_inputs, en_vocab).flatten().to(torch.int64).cpu()
    decoder_sequence_lengths = get_lengths(decoder_inputs, fr_vocab).flatten().to(torch.int64).cpu()

    training_dataset = EncDecDataset(
        data = {
            'encoder_inputs': encoder_inputs,
            'decoder_inputs': decoder_inputs,
            'decoder_targets': decoder_targets,
        },
        lengths = {
            'encoder_lengths': encoder_sequence_lengths,
            'decoder_lengths': decoder_sequence_lengths,
        }
    )

    training_dataloader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE)

    encoder = Encoder(EN_VOCAB_SIZE, en_vocab["value_index"]["<PAD>"]).to(device)
    decoder = Decoder(FR_VOCAB_SIZE, fr_vocab["value_index"]["<PAD>"]).to(device)
    model = EncoderDecoder(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab['value_index']['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters())

    parameters = train(model, training_dataloader, criterion, optimizer, 35, device)

    torch.save(parameters, f"parameters/parameters_{datetime.now().strftime("%m_%d_%Y-%H_%M_%S")}.pt")

if __name__ == "__main__":
    main()





