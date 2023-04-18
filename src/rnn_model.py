import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, dropout):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer creation
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer creation
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, dropout = dropout, batch_first = True)

        # Output fully connected layer creation
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):

        # x input through embedding layer
        x = self.embedding(x)

        # x input through RNN layer
        output, hidden = self.rnn(x, hidden)

        # x input through Output fully connected layer
        output = self.fc(output)

        return output, hidden



