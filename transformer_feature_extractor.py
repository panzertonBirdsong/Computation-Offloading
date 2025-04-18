import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, d_model=32, nhead=2, num_layers=2):
        super().__init__(observation_space, features_dim)
        
        self.n_input = observation_space.shape[0]

        self.embedding = nn.Linear(self.n_input, d_model)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_input, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        x = observations.unsqueeze(1)
        x = self.embedding(x)
        
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]

        x = x.squeeze(1)  # [batch_size, d_model]

        features = self.fc_out(x)  # [batch_size, features_dim]
        
        return features


if __name__ == "__main__":
    print("All good.")