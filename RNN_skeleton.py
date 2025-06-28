import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Weights for input to hidden connection
        self.Wx = nn.Linear(embedding_dim, hidden_size)
        # Weights for hidden to hidden connection
        self.Wh = nn.Linear(hidden_size, hidden_size)
        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

        # Activation function (tanh) for the hidden state
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state with zeros
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Iterate over each time step
        for t in range(seq_len):
            xt = x[:, t, :]  # Select the t-th time step input
            h = torch.tanh(self.Wx(xt) + self.Wh(h))  # Update hidden state

        # Use the hidden state from the last time step to predict the output
        out = self.fc(h)
        return out