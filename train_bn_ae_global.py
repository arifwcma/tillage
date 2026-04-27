from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "bn_ae"

DATASET_NAME = "global"
SOC_COLUMN = "Org C"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOTTLENECK_DIMENSION = 64
HIDDEN_DIMENSION = 256
REGRESSION_HEAD_HIDDEN = 32
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def load_train_and_test_dataframes():
    train_dataframe = pd.read_csv(PREPROCESSED_DIR / f"{DATASET_NAME}_none_train.csv", low_memory=False)
    test_dataframe = pd.read_csv(PREPROCESSED_DIR / f"{DATASET_NAME}_none_test.csv", low_memory=False)
    return train_dataframe, test_dataframe


def extract_spectra_and_target(dataframe):
    spectra_columns = [c for c in dataframe.columns if WAVENUMBER_COLUMN_PATTERN.match(c) is not None]
    spectra_matrix = dataframe[spectra_columns].to_numpy(dtype=np.float32)
    target_vector = dataframe[SOC_COLUMN].to_numpy(dtype=np.float32)
    return spectra_matrix, target_vector


def to_device_tensor(numpy_array):
    return torch.from_numpy(numpy_array).to(DEVICE)


class LearnedPreprocessingAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.preprocessing_batchnorm = nn.BatchNorm1d(n_features)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, BOTTLENECK_DIMENSION),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, n_features),
        )

    def forward(self, raw_spectra):
        normalised_spectra = self.preprocessing_batchnorm(raw_spectra)
        latent_code = self.encoder(normalised_spectra)
        reconstructed_spectra = self.decoder(latent_code)
        return reconstructed_spectra, normalised_spectra


class SocRegressorOnPretrainedBatchnorm(nn.Module):
    def __init__(self, pretrained_autoencoder, n_features):
        super().__init__()
        self.preprocessing_batchnorm = pretrained_autoencoder.preprocessing_batchnorm
        self.regression_head = nn.Sequential(
            nn.Linear(n_features, REGRESSION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(REGRESSION_HEAD_HIDDEN, 1),
        )

    def forward(self, raw_spectra):
        normalised_spectra = self.preprocessing_batchnorm(raw_spectra)
        soc_prediction = self.regression_head(normalised_spectra).squeeze(-1)
        return soc_prediction


def build_loader_features_only(spectra_matrix, shuffle):
    dataset = TensorDataset(to_device_tensor(spectra_matrix))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def build_loader_features_and_targets(spectra_matrix, target_vector, shuffle):
    dataset = TensorDataset(to_device_tensor(spectra_matrix), to_device_tensor(target_vector))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def pretrain_autoencoder(autoencoder_model, train_spectra):
    autoencoder_model.train()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)
    reconstruction_loss = nn.MSELoss()
    train_loader = build_loader_features_only(train_spectra, shuffle=True)

    for epoch_index in range(PRETRAIN_EPOCHS):
        epoch_loss_sum = 0.0
        for (batch_spectra,) in train_loader:
            optimizer.zero_grad()
            reconstructed_spectra, _ = autoencoder_model(batch_spectra)
            loss = reconstruction_loss(reconstructed_spectra, batch_spectra)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * batch_spectra.size(0)
        if epoch_index == 0 or (epoch_index + 1) % 10 == 0:
            mean_loss = epoch_loss_sum / len(train_spectra)
            print(f"  [pretrain] epoch {epoch_index + 1:>3d}/{PRETRAIN_EPOCHS}  recon MSE: {mean_loss:.6f}")


def finetune_soc_regressor(regressor_model, train_spectra, train_target):
    regressor_model.train()
    optimizer = torch.optim.Adam(regressor_model.parameters(), lr=LEARNING_RATE)
    soc_loss = nn.MSELoss()
    train_loader = build_loader_features_and_targets(train_spectra, train_target, shuffle=True)

    for epoch_index in range(FINETUNE_EPOCHS):
        epoch_loss_sum = 0.0
        for batch_spectra, batch_target in train_loader:
            optimizer.zero_grad()
            predicted_soc = regressor_model(batch_spectra)
            loss = soc_loss(predicted_soc, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * batch_spectra.size(0)
        if epoch_index == 0 or (epoch_index + 1) % 20 == 0:
            mean_loss = epoch_loss_sum / len(train_spectra)
            print(f"  [finetune] epoch {epoch_index + 1:>3d}/{FINETUNE_EPOCHS}  SOC MSE: {mean_loss:.6f}")


def predict_soc_for_set(regressor_model, spectra_matrix):
    regressor_model.eval()
    with torch.no_grad():
        predictions = regressor_model(to_device_tensor(spectra_matrix)).cpu().numpy()
    return predictions


def compute_rmse(observed, predicted):
    residual = observed.ravel() - predicted.ravel()
    return float(np.sqrt((residual ** 2).mean()))


def compute_r_squared(observed, predicted):
    observed = observed.ravel()
    predicted = predicted.ravel()
    sum_squared_residuals = float(((observed - predicted) ** 2).sum())
    sum_squared_total = float(((observed - observed.mean()) ** 2).sum())
    if sum_squared_total == 0:
        return float("nan")
    return 1.0 - sum_squared_residuals / sum_squared_total


def compute_mean_bias_deviation(observed, predicted):
    return float((predicted.ravel() - observed.ravel()).mean())


def compute_rpiq(observed, predicted):
    rmse_value = compute_rmse(observed, predicted)
    if rmse_value == 0:
        return float("inf")
    interquartile_range = float(
        np.percentile(observed.ravel(), 75) - np.percentile(observed.ravel(), 25)
    )
    return interquartile_range / rmse_value


def print_metrics(label, observed, predicted):
    rmse_value = compute_rmse(observed, predicted)
    r2_value = compute_r_squared(observed, predicted)
    mbd_value = compute_mean_bias_deviation(observed, predicted)
    rpiq_value = compute_rpiq(observed, predicted)
    print(
        f"  {label}: RMSE={rmse_value:.4f}  R2={r2_value:.4f}  "
        f"MBD={mbd_value:+.4f}  RPIQ={rpiq_value:.3f}  n={len(observed)}"
    )


def save_test_predictions(test_dataframe, test_predictions):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spectra_columns = [c for c in test_dataframe.columns if WAVENUMBER_COLUMN_PATTERN.match(c) is not None]
    reference_columns = [c for c in test_dataframe.columns if c not in spectra_columns]
    output_dataframe = test_dataframe[reference_columns].copy()
    output_dataframe["observed"] = test_dataframe[SOC_COLUMN].to_numpy()
    output_dataframe["predicted"] = test_predictions.ravel()
    output_path = RESULTS_DIR / f"{DATASET_NAME}_bn_ae_test_predictions.csv"
    output_dataframe.to_csv(output_path, index=False)
    print(f"  wrote {output_path}")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"device: {DEVICE}")

    train_dataframe, test_dataframe = load_train_and_test_dataframes()
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]
    print(f"train rows: {len(train_target)}  test rows: {len(test_target)}  features: {n_features}")

    print("\n[Phase A] pretraining autoencoder (Input -> BN -> Encoder -> Decoder -> Input')")
    autoencoder_model = LearnedPreprocessingAutoencoder(n_features).to(DEVICE)
    pretrain_autoencoder(autoencoder_model, train_spectra)

    print("\n[Phase B] finetuning SOC regressor on pretrained BN only")
    regressor_model = SocRegressorOnPretrainedBatchnorm(autoencoder_model, n_features).to(DEVICE)
    finetune_soc_regressor(regressor_model, train_spectra, train_target)

    print("\n[Evaluation]")
    train_predictions = predict_soc_for_set(regressor_model, train_spectra)
    test_predictions = predict_soc_for_set(regressor_model, test_spectra)
    print_metrics("train", train_target, train_predictions)
    print_metrics("test ", test_target, test_predictions)

    save_test_predictions(test_dataframe, test_predictions)


if __name__ == "__main__":
    main()
