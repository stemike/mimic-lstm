import torch
from torch import nn


#RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
#model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])
class ICU_LSTM(nn.Module):
    def __init__(self, input_size):
        super(ICU_LSTM, self).__init__()
        hidden_size = 256
        num_layers = 1
        output_size = 1

        self.attention = None #Attention(input_layer, time_steps)
        # x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x) use Padded sequence instead
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dense = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return None

    def forward(self, x, h_c):
        intermediate, hidden = self.lstm(x)
        intermediate = self.dense(intermediate)
        output = torch.sigmoid(intermediate)
        return output





