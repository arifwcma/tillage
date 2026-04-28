import torch
from torch import nn


CONV_CHANNELS = 8
CONV_KERNEL_SIZE = 7
CONV_PADDING = (CONV_KERNEL_SIZE - 1) // 2


class Ddp3Preprocessor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(n_features)
        self.encoder_conv = nn.Conv1d(
            in_channels=1,
            out_channels=CONV_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING,
        )
        self.encoder_activation = nn.ReLU()
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=CONV_CHANNELS,
            out_channels=1,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING,
        )
        self.output_batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, spectra):
        x = self.input_batch_norm(spectra)
        x = x.unsqueeze(1)
        x = self.encoder_conv(x)
        x = self.encoder_activation(x)
        x = self.decoder_conv(x)
        x = x.squeeze(1)
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
