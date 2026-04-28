import torch
from torch import nn


class DdpPreprocessor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, spectra):
        return self.batch_norm(spectra)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def transform_with_frozen_running_statistics(self, spectra_numpy_matrix):
        self.eval()
        spectra_tensor = torch.from_numpy(spectra_numpy_matrix)
        with torch.no_grad():
            transformed_tensor = self.batch_norm(spectra_tensor)
        return transformed_tensor.cpu().numpy()


class DdpPlusMlp(nn.Module):
    def __init__(self, preprocessor, regression_head):
        super().__init__()
        self.preprocessor = preprocessor
        self.regression_head = regression_head

    def forward(self, spectra):
        normalised_spectra = self.preprocessor(spectra)
        return self.regression_head(normalised_spectra)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
