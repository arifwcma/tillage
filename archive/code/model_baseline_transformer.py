import torch
from torch import nn
import torch.nn.functional as functional


PATCH_SIZE = 32
TRANSFORMER_D_MODEL = 32
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FEEDFORWARD_DIM = 64
TRANSFORMER_N_LAYERS = 2


class BaselineSocTransformer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.padding_amount = self._compute_padding_amount(n_features, PATCH_SIZE)
        self.n_patches = (n_features + self.padding_amount) // PATCH_SIZE

        self.patch_embedding = nn.Linear(PATCH_SIZE, TRANSFORMER_D_MODEL)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.n_patches, TRANSFORMER_D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_N_HEADS,
            dim_feedforward=TRANSFORMER_FEEDFORWARD_DIM,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=TRANSFORMER_N_LAYERS
        )
        flatten_size = self.n_patches * TRANSFORMER_D_MODEL
        self.regression_head = nn.Linear(flatten_size, 1)

    def _compute_padding_amount(self, n_features, patch_size):
        remainder = n_features % patch_size
        return 0 if remainder == 0 else patch_size - remainder

    def _patchify(self, padded_spectra):
        return padded_spectra.unfold(dimension=1, size=PATCH_SIZE, step=PATCH_SIZE)

    def forward(self, input_spectra):
        padded_spectra = functional.pad(input_spectra, (0, self.padding_amount))
        patched_spectra = self._patchify(padded_spectra)
        embedded_patches = self.patch_embedding(patched_spectra) + self.positional_embedding
        encoded_patches = self.transformer_encoder(embedded_patches)
        flat_features = encoded_patches.flatten(1)
        return self.regression_head(flat_features).squeeze(-1)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
