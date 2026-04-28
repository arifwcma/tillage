from torch import nn


HIDDEN_UNITS = 32
DROPOUT_PROBABILITY = 0.3


class MlpSocAnn(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(n_features, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROBABILITY),
            nn.Linear(HIDDEN_UNITS, 1),
        )

    def forward(self, spectra):
        return self.regression_head(spectra).squeeze(-1)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
