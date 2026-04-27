from torch import nn


ENCODER_HIDDEN_DIMENSION = 256
BOTTLENECK_DIMENSION = 64
REGRESSION_HEAD_HIDDEN = 32


class P2bnSocAnn(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.preprocessing_batchnorm = nn.BatchNorm1d(n_features)
        self.regression_head = nn.Sequential(
            nn.Linear(n_features, REGRESSION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGRESSION_HEAD_HIDDEN, 1),
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_features, ENCODER_HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN_DIMENSION, BOTTLENECK_DIMENSION),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIMENSION, ENCODER_HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN_DIMENSION, n_features),
        )

    def forward(self, raw_spectra):
        normalised_spectra = self.preprocessing_batchnorm(raw_spectra)
        soc_prediction = self.regression_head(normalised_spectra).squeeze(-1)
        latent_code = self.encoder(normalised_spectra)
        reconstructed_spectra = self.decoder(latent_code)
        return soc_prediction, reconstructed_spectra
