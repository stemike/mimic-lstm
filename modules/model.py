import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.pad_sequences import get_seq_length_from_padded_seq


class ICU_LSTM(nn.Module):
    def __init__(self, input_size, seq_length=14):
        super(ICU_LSTM, self).__init__()
        hidden_size = 256
        num_layers = 1
        output_size = 1

        self.attention_layer = nn.Linear(seq_length, seq_length, bias=False)
        # x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x) use Padded sequence instead
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dense = nn.Linear(hidden_size, output_size)

        self.attention = None

    def forward(self, x, h_c=None):
        # x is of shape batch_size x seq_length x n_features
        # Swap Axes because attention goes over one features seq_length i.e. 14 days in this case
        a = self.attention_layer(torch.transpose(x, 1, 2))
        a = torch.softmax(a, dim=2)
        a = torch.transpose(a, 1, 2)
        # Save a to attention variable for being able to return it later
        self.attention = a.clone().detach().cpu().numpy()
        x = a * x

        # seq_lengths = get_seq_length_from_padded_seq(x.clone().detach().cpu().numpy())
        # x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        if h_c is None:
            intermediate, hidden = self.lstm(x)
        else:
            h, c = h_c
            intermediate, h_c = self.lstm(x, h, c)
        # intermediate, _ = pad_packed_sequence(intermediate, batch_first=True, padding_value=0, total_length=14)

        intermediate = self.dense(intermediate)
        output = torch.sigmoid(intermediate)
        return output, h_c
