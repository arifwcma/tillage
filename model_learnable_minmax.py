import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn


class LearnableMinMax(nn.Module):
    def __init__(self, dataset_matrix):
        super().__init__()
        per_band_minimum = dataset_matrix.min(dim=0).values
        per_band_maximum = dataset_matrix.max(dim=0).values
        self.low = nn.Parameter(per_band_minimum.clone())
        self.high = nn.Parameter(per_band_maximum.clone())

    def forward(self, dataset_matrix):
        return (dataset_matrix - self.low) / (self.high - self.low)

    def count_learnable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


BELOW_ZERO_PENALTY_WEIGHT = 0.1
ABOVE_ONE_PENALTY_WEIGHT = 0.1


def compute_total_loss(transformed_dataset_matrix):
    band_wise_mean = transformed_dataset_matrix.mean(dim=0)
    band_centring_term = torch.mean(torch.square(band_wise_mean - 0.5))
    below_zero_penalty = torch.mean(torch.relu(-transformed_dataset_matrix))
    above_one_penalty = torch.mean(torch.relu(transformed_dataset_matrix - 1.0))
    return (
        band_centring_term
        + BELOW_ZERO_PENALTY_WEIGHT * below_zero_penalty
        + ABOVE_ONE_PENALTY_WEIGHT * above_one_penalty
    )


def load_global_none_train_spectra():
    csv_path = Path(__file__).resolve().parent / "data" / "preprocessed" / "global_none_train.csv"
    table = pd.read_csv(csv_path, low_memory=False)
    wavenumber_column_pattern = re.compile(r"^m(\d+(?:\.\d+)?)$")
    spectra_columns = []
    wavenumber_values = []
    for column_name in table.columns:
        match_object = wavenumber_column_pattern.match(column_name)
        if match_object is None:
            continue
        spectra_columns.append(column_name)
        wavenumber_values.append(float(match_object.group(1)))
    spectra_tensor = torch.from_numpy(table[spectra_columns].to_numpy(dtype=np.float32))
    wavenumber_array = np.array(wavenumber_values)
    return spectra_tensor, wavenumber_array


def classical_minmax_per_feature(dataset_matrix):
    per_band_minimum = dataset_matrix.min(dim=0).values
    per_band_maximum = dataset_matrix.max(dim=0).values
    per_band_range = per_band_maximum - per_band_minimum
    per_band_range = torch.where(per_band_range == 0, torch.ones_like(per_band_range), per_band_range)
    return (dataset_matrix - per_band_minimum) / per_band_range


def plot_three_mean_spectra(wavenumber_array, original_dataset, minmax_dataset, model_dataset, output_figure_path):
    figure, axes_grid = plt.subplots(1, 3, figsize=(18, 5))

    axes_grid[0].plot(wavenumber_array, original_dataset.mean(dim=0).detach().numpy(), color="#222222", linewidth=1.0)
    axes_grid[0].set_title("original mean reflectance")
    axes_grid[1].plot(wavenumber_array, minmax_dataset.mean(dim=0).detach().numpy(), color="#9467bd", linewidth=1.0)
    axes_grid[1].set_title("classical minmax mean")
    axes_grid[2].plot(wavenumber_array, model_dataset.mean(dim=0).detach().numpy(), color="#1f77b4", linewidth=1.0)
    axes_grid[2].set_title("LearnableMinMax forward mean")

    for panel_axes in axes_grid:
        panel_axes.set_xlim(4000, 600)
        panel_axes.set_xlabel("wavenumber (cm$^{-1}$)")
        panel_axes.set_ylabel("mean reflectance")
        panel_axes.grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_figure_path, dpi=120)
    plt.close(figure)


def write_band_parameters_to_csv(wavenumber_array, model, output_csv_path):
    learned_low = model.low.detach().cpu().numpy()
    learned_high = model.high.detach().cpu().numpy()
    band_parameter_table = pd.DataFrame({
        "wavenumber": wavenumber_array,
        "down": learned_low,
        "up": learned_high,
    })
    band_parameter_table.to_csv(output_csv_path, index=False)


def summarise_per_band_mean(transformed_dataset_matrix):
    band_wise_mean = transformed_dataset_matrix.mean(dim=0).detach()
    return {
        "min":  float(band_wise_mean.min()),
        "max":  float(band_wise_mean.max()),
        "mean": float(band_wise_mean.mean()),
        "std":  float(band_wise_mean.std(unbiased=False)),
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parent
    output_figure_path = project_root / "results" / "learnable_minmax_spectra.png"
    output_csv_path = project_root / "results" / "learnable_minmax_band_parameters.csv"
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dataset_matrix, wavenumber_array = load_global_none_train_spectra()
    print(f"loaded global_none_train: shape = {tuple(raw_dataset_matrix.shape)}")
    print(f"raw value range: [{float(raw_dataset_matrix.min()):.4f}, {float(raw_dataset_matrix.max()):.4f}]")

    model = LearnableMinMax(raw_dataset_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_loss = float(compute_total_loss(model(raw_dataset_matrix)))
    print(f"initial loss: {initial_loss:.6f}  (low=min, high=max → transformed in [0,1])")
    print(f"initial per-band mean stats: {summarise_per_band_mean(model(raw_dataset_matrix))}")

    for epoch_index in range(500):
        optimizer.zero_grad()
        transformed_dataset_matrix = model(raw_dataset_matrix)
        loss = compute_total_loss(transformed_dataset_matrix)
        loss.backward()
        optimizer.step()

    final_loss = float(compute_total_loss(model(raw_dataset_matrix)))
    print(f"\nfinal loss: {final_loss:.6f}")
    print(f"final per-band mean stats: {summarise_per_band_mean(model(raw_dataset_matrix))}")
    print(f"learnable params: {model.count_learnable_parameters()}")

    classical_minmax_dataset = classical_minmax_per_feature(raw_dataset_matrix)
    final_transformed_dataset = model(raw_dataset_matrix)
    plot_three_mean_spectra(
        wavenumber_array,
        raw_dataset_matrix,
        classical_minmax_dataset,
        final_transformed_dataset,
        output_figure_path,
    )
    print(f"wrote {output_figure_path}")

    write_band_parameters_to_csv(wavenumber_array, model, output_csv_path)
    print(f"wrote {output_csv_path}")
