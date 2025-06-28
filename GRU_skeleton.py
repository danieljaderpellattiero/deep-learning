import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)

        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # Update gate
        # hint : The update gate controls how much of the previous hidden state (h_prev)
        # should be carried forward to the next hidden state.
        z_t = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))

        # Reset gate
        # hint : The reset gate determines how much of the previous hidden state
        # should be "reset" or ignored when computing the candidate hidden state
        r_t = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))

        # Candidate hidden state
        # hint : The candidate hidden state is computed using a combination of the reset hidden state and the current input.
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r_t * h_prev))

        # New final hidden state
        # The final hidden state is a blend of the previous hidden state and the candidate hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru_cell = GRUCell(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)

        out = self.fc(h_t)
        return out
