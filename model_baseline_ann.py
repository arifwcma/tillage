from torch import nn


REGRESSION_HEAD_HIDDEN = 32


class BaselineSocAnn(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(n_features, REGRESSION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGRESSION_HEAD_HIDDEN, 1),
        )

    def forward(self, input_spectra):
        return self.regression_head(input_spectra).squeeze(-1)
