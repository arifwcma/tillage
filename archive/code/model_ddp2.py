import torch
from torch import nn


BOTTLENECK_HIDDEN_UNITS = 32


class Ddp2Preprocessor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(n_features)
        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(n_features, BOTTLENECK_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(BOTTLENECK_HIDDEN_UNITS, n_features),
        )
        self.output_batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, spectra):
        x = self.input_batch_norm(spectra)
        x = self.bottleneck_mlp(x)
        x = self.output_batch_norm(x)
        return x

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def transform_with_frozen_running_statistics(self, spectra_numpy_matrix):
        self.eval()
        spectra_tensor = torch.from_numpy(spectra_numpy_matrix)
        with torch.no_grad():
            transformed_tensor = self.forward(spectra_tensor)
        return transformed_tensor.cpu().numpy()
