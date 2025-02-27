import torch
from torch.utils.data import Dataset

class EncDecDataset(Dataset):
    def __init__(self, data : dict, lengths : dict):
        self.data = self._check_data(data)
        self.lengths = self._check_lengths(lengths)
        self._check_batch_size()

    def _check_batch_size(self):
        if list(self.data.values())[0].size(0) !=  list(self.lengths.values())[0].size(0):
            raise ValueError("Batch sizes do not match between training_data and lengths.")

    @staticmethod
    def _check_data_keys(data):
        if len(data.keys()) > 3:
            raise ValueError("'training_data' must have at most three keys.")

        if list(data.keys()) != ['encoder_inputs', 'decoder_inputs', 'decoder_targets']:
            raise ValueError("'training_data' keys must be 'encoder_inputs', 'decoder_inputs' and 'decoder_targets'.")

    @staticmethod
    def _check_lengths_keys(lengths):
        if len(lengths.keys()) > 2:
            raise ValueError("'lengths' must have at most two keys.")

        if list(lengths.keys()) != ['encoder_lengths', 'decoder_lengths']:
            raise ValueError("'lengths' keys must be 'encoder_lengths' and 'decoder_lengths'.")

    @staticmethod
    def _check_data_values(data):
        dataset_length = None
        for value in data.values():
            if not torch.is_tensor(value):
                raise ValueError("'training_data' values must be of type torch.Tensor.")

            if dataset_length is None:
                dataset_length = value.size(0)
            elif dataset_length != value.size(0):
                raise ValueError("'training_data' values must have the same batch size.")

    @staticmethod
    def _check_lengths_values(lengths):
        dataset_length = None
        for value in lengths.values():
            if not torch.is_tensor(value):
                raise ValueError("'lengths' values must be of type torch.Tensor.")

            if dataset_length is None:
                dataset_length = value.size(0)
            elif dataset_length != value.size(0):
                raise ValueError("'lengths' values must have the same batch size.")

    def _check_data(self, data):
        if isinstance(data, dict):
            self._check_data_keys(data)
            self._check_data_values(data)
            return data
        else:
            raise ValueError("'training_data' must be of type torch.Tensor or dict.")

    def _check_lengths(self, lengths):
        if isinstance(lengths, dict):
            self._check_lengths_keys(lengths)
            self._check_lengths_values(lengths)
            return lengths
        else:
            raise ValueError("'lengths' must be of type torch.Tensor or dict.")


    def __len__(self):
        return self.data['encoder_inputs'].size(0)

    def __getitem__(self, idx):
        return {
            'training_data': {
                'encoder_inputs': self.data['encoder_inputs'][idx],
                'decoder_inputs': self.data['decoder_inputs'][idx],
                'decoder_targets': self.data['decoder_targets'][idx]
            },
            'lengths': {
                'encoder_lengths': self.lengths['encoder_lengths'][idx],
                'decoder_lengths': self.lengths['decoder_lengths'][idx]
            }
        }

def enc_dec_collate(batch):
    encoder_inputs = [item['training_data']['encoder_inputs'] for item in batch]
    decoder_inputs = [item['training_data']['decoder_inputs'] for item in batch]
    decoder_targets = [item['training_data']['decoder_targets'] for item in batch]

    encoder_lengths = [item['lengths']['encoder_lengths'] for item in batch]
    decoder_lengths = [item['lengths']['decoder_lengths'] for item in batch]

    return {
        'training_data': {
            'encoder_inputs': torch.stack(encoder_inputs),
            'decoder_inputs': torch.stack(decoder_inputs),
            'decoder_targets': torch.stack(decoder_targets)
        },
        'lengths': {
            'encoder_lengths': torch.stack(encoder_lengths),
            'decoder_lengths': torch.stack(decoder_lengths)
        }
    }