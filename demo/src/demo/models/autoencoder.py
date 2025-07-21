import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        *encoder_dims, bottleneck = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dims[0], bottleneck),
            nn.ReLU(),
        )
        decoder_dims = list(reversed(encoder_dims))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, decoder_dims[0]),
            nn.ReLU(),
            nn.Linear(decoder_dims[0], input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z  # reconstruction & latent
