import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.pad_sequences import get_seq_length_from_padded_seq


class ICU_LSTM(nn.Module):
    def __init__(self, input_size):
        super(ICU_LSTM, self).__init__()
        hidden_size = 256
        num_layers = 1
        output_size = 1

        self.attention_layer = nn.Linear(input_size, input_size, bias=False)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

        self.attention = None

        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'lstm.weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'lstm.weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'lstm.bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
            # Reproducing Keras' unit_forget_bias parameter
            n = t.size(0)
            start, end = n // 4, n // 2
            t[start:end].fill_(1.)

    def forward(self, x, h_c=None):
        # x is of shape batch_size x seq_length x n_features
        a = self.attention_layer(x)
        a = torch.softmax(a, dim=1)
        # Save a to attention variable for being able to return it later
        self.attention = a.clone().detach().cpu().numpy()
        x = a * x

        #seq_lengths = get_seq_length_from_padded_seq(x.clone().detach().cpu().numpy())
        #x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        if h_c is None:
            intermediate, h_c = self.lstm(x)
        else:
            h, c = h_c
            intermediate, h_c = self.lstm(x, h, c)
        #intermediate, _ = pad_packed_sequence(intermediate, batch_first=True, padding_value=0, total_length=14)

        intermediate = self.dense(intermediate)
        output = torch.sigmoid(intermediate)
        return output, h_c
