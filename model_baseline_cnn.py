import torch
from torch import nn


CONV_OUT_CHANNELS_FIRST = 16
CONV_OUT_CHANNELS_SECOND = 32
CONV_KERNEL_SIZE = 11
CONV_PADDING = CONV_KERNEL_SIZE // 2
POOL_KERNEL_SIZE = 2


class BaselineSocCnn(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, CONV_OUT_CHANNELS_FIRST, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE),
            nn.Conv1d(
                CONV_OUT_CHANNELS_FIRST, CONV_OUT_CHANNELS_SECOND,
                kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE),
        )
        flatten_size = self._compute_flatten_size(n_features)
        self.regression_head = nn.Linear(flatten_size, 1)

    def _compute_flatten_size(self, n_features):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_features)
            return self.feature_extractor(dummy_input).flatten(1).shape[1]

    def forward(self, input_spectra):
        spectra_with_channel_axis = input_spectra.unsqueeze(1)
        feature_map = self.feature_extractor(spectra_with_channel_axis)
        flat_features = feature_map.flatten(1)
        return self.regression_head(flat_features).squeeze(-1)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
