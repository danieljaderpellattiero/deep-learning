import torch
import torch.nn as nn

class Attention1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention1D, self).__init__()
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)

        self.last_linear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
        )

    def attention(self, x):
        k = self.k_linear(x).unsqueeze(-1)  # B x output_dim x 1
        q = self.q_linear(x).unsqueeze(-1)  # B x output_dim x 1
        v = self.v_linear(x).unsqueeze(-1)  # B x output_dim x 1

        attention_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # B x output_dim x output_dim
        attention_weights = torch.softmax(attention_scores, dim=-1)

        context = torch.bmm(attention_weights, v).squeeze()  # B x output_dim
        context = self.last_linear(context) + context  # B x output_dim
        return context

    def forward(self, x):
        return self.attention(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, attention=False):
        super(MLPModel, self).__init__()

        if attention:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU(),
                nn.Dropout(0.3),
                Attention1D(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU(),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim),
            )
        else:
            if num_layers == 3:
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.SiLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, output_dim),
                )
            elif num_layers == 5:
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.SiLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, output_dim),
                )
            elif num_layers == 7:
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.SiLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 2048),
                    nn.BatchNorm1d(2048),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, output_dim),
                )

    def forward(self, x):
        return self.layers(x)
