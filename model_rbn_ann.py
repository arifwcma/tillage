from torch import nn


REGRESSION_HEAD_HIDDEN = 32


class RbnSocAnn(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.preprocessing_batchnorm = nn.BatchNorm1d(n_features)
        self.regression_head = nn.Sequential(
            nn.Linear(n_features, REGRESSION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGRESSION_HEAD_HIDDEN, 1),
        )

    def forward(self, raw_spectra):
        normalised_spectra = self.preprocessing_batchnorm(raw_spectra)
        return self.regression_head(normalised_spectra).squeeze(-1)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
