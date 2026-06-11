# Finetuning Pangu-Weather Model on IMD GFS Data

This repository contains the codebase, logs, and evaluation metrics for the finetuning of the Pangu-Weather model using Indian Meteorological Department (IMD) Global Forecast System (GFS) data. 

## 1. Explanation of the Finetuning Project
The original Pangu-Weather model, developed by Huawei, was trained on 40 years of global ECMWF ERA5 reanalysis data. While it performs exceptionally well globally, applying it directly to regional datasets like IMD GFS introduces a domain gap. 

This project aims to bridge that gap. We converted the original 24-hour ONNX model to PyTorch (`.pth`) format using a corrected version of the conversion script provided by https://github.com/zhaoshan2/pangu-pytorch, and then successfully finetuned this model on 3.5 years of IMD GFS data. The goal was to teach the model the specific thermodynamic and kinetic dynamics present in the IMD datasets.

## 2. Files and File Structure

```text
├── IMD_GFS_data_3yr/          # Data directory (Surface, Upper, and Aux Data)
├── Pangu_Finetune_Single/     # Core training and utility scripts
│   ├── finetune_entire.py     # Main finetuning script
│   ├── compute_mean_std.py    # Generates standard deviation/mean (aux_data)
│   ├── compare_models_metrics.py # Script for generating evaluation metrics
│   └── pangu_finetune_gfs.pbs # HPC job submission script
├── final_inference/           # Inference scripts
│   └── inference_multi_model.py # Autoregressive n-day inference script
├── models/                    # Saved checkpoints (e.g., best_model.pth)
└── README.md                  # Project documentation
```

## 3. Infrastructure
* **Compute:** High-Performance Computing (HPC) Cluster
* **Hardware:** 8x NVIDIA RTX A5000, split across 2 Nodes
* **Environment:** Managed via Conda (`pangu_env`). See `requirements.txt` for exact Python dependencies.

## 4. Inputs / Data
The dataset consists of 3.5 years of IMD GFS data, spanning from **January 2023 to May 2026**.
* **Preprocessing:** Original files were converted from GRIB2 to NetCDF (`.nc`) at a 0.25-degree resolution.
* **Structure:** Separated into day-wise surface files (`surface_YYYY_MM_DD.nc`) and upper-air files (`upper_air_YYYY_MM_DD.nc`) but it also supports monthly files (`surface_YYYY_MM.nc`) and (`upper_air_YYYY_MM.nc`) for surface and upper air files respectively.
* **Normalization:** Mean and standard deviation tensors were computed across the dataset using `compute_mean_std.py` script and stored in the `aux_data/` directory to normalize inputs during training.

## 5. Logs and Metrics of Finetuning
The model was trained for a maximum of 100 epochs, utilizing an Early Stopping mechanism monitoring the validation loss with a patience of 20 epochs.

* **Training Duration:** Training halted at Epoch 51 due to early stopping.
* **Best Epoch:** The optimal weights were achieved at **Epoch 31**.
* **Best Validation Loss:** `0.131063`

## 6. Output File
The final optimized weights from Epoch 31 were extracted and saved. This checkpoint is exclusively used for all subsequent forecasting.
* **Location:** `Pangu_Finetune_Single/finetuned_GFS_final/finetune_fully/24/models/best_model.pth`

## 7. Inference and Scripts
To run a forecast using the finetuned model, we utilize a unified multi-model inference script that supports both 1-day and autoregressive n-day forecasting.

```bash
python final_inference/inference_multi_model.py \
    --model_type finetuned \
    --model_path path/to/best_model.pth \
    --data_root path/to/IMD_GFS_data_3yr \
    --output_root path/to/output_dir \
    --start_time "20240428 00:00:00" \
    --horizon_hours 24 \
    --steps 1
```

## 8. Inference Output
The output of the inference script is a standard NetCDF file (e.g., `forecast_20240429_0000.nc`) containing the predicted grid values for both surface and upper-air variables exactly 24 hours from the initialization time.

## 9. Final Evaluation Metrics (Validation Set)
Below are the comprehensive final evaluation metrics calculated on the validation set for the best model (Epoch 31).

### Surface Variables
| Variable | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| **msl** | 0.073886 | 0.118954 | 0.015700 | 0.002193 |
| **u10** | 0.185723 | 0.272680 | 0.073880 | 0.000375 |
| **v10** | 0.220025 | 0.320830 | 0.103115 | 0.001338 |
| **t2m** | 0.032998 | 0.051937 | 0.002503 | 0.001301 |

### Upper-Air Variables

<details>
<summary><strong>Geopotential Height (Z)</strong></summary>

| Level (hPa) | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| 1000 | 0.072679 | 0.114007 | 0.014779 | -0.000714 |
| 925 | 0.060814 | 0.094486 | 0.010556 | -0.000575 |
| 850 | 0.048441 | 0.074273 | 0.006381 | 0.000889 |
| 700 | 0.033425 | 0.049883 | 0.002691 | 0.002235 |
| 600 | 0.028970 | 0.042768 | 0.001932 | -0.000951 |
| 500 | 0.026258 | 0.038921 | 0.001583 | -0.000391 |
| 400 | 0.024287 | 0.036014 | 0.001350 | 0.000721 |
| 300 | 0.022000 | 0.032169 | 0.001084 | 0.002624 |
| 250 | 0.020066 | 0.028477 | 0.000867 | 0.000151 |
| 200 | 0.017620 | 0.024314 | 0.000668 | 0.000843 |
| 150 | 0.015370 | 0.020583 | 0.000531 | 0.000390 |
| 100 | 0.015149 | 0.020056 | 0.000638 | 0.001261 |
| 50 | 0.017115 | 0.022148 | 0.001347 | -0.003105 |

</details>

<details>
<summary><strong>Specific Humidity (Q)</strong></summary>

| Level (hPa) | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| 1000 | 0.057498 | 0.093964 | 0.008429 | -0.001874 |
| 925 | 0.073778 | 0.119966 | 0.013345 | 0.006099 |
| 850 | 0.117857 | 0.191651 | 0.036182 | 0.004525 |
| 700 | 0.150290 | 0.248336 | 0.064604 | -0.000847 |
| 600 | 0.163390 | 0.274222 | 0.082499 | -0.006342 |
| 500 | 0.180746 | 0.312607 | 0.107082 | -0.001595 |
| 400 | 0.185728 | 0.340238 | 0.121986 | -0.022950 |
| 300 | 0.173535 | 0.342318 | 0.119333 | -0.019806 |
| 250 | 0.159722 | 0.325065 | 0.115841 | -0.015793 |
| 200 | 0.149626 | 0.290380 | 0.117516 | 0.003764 |
| 150 | 0.130919 | 0.215997 | 0.131480 | -0.019835 |
| 100 | 0.073541 | 0.105966 | 0.047325 | -0.016877 |
| 50 | 0.141486 | 0.260057 | 0.040752 | -0.073893 |

</details>

<details>
<summary><strong>Temperature (T)</strong></summary>

| Level (hPa) | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| 1000 | 0.040247 | 0.061388 | 0.003465 | 0.000484 |
| 925 | 0.046887 | 0.067637 | 0.004180 | 0.000839 |
| 850 | 0.047664 | 0.067615 | 0.004175 | 0.005231 |
| 700 | 0.042838 | 0.061302 | 0.003526 | 0.001913 |
| 600 | 0.043232 | 0.060753 | 0.003669 | 0.002458 |
| 500 | 0.044410 | 0.061441 | 0.003825 | 0.003403 |
| 400 | 0.045013 | 0.062069 | 0.004001 | 0.007358 |
| 300 | 0.055112 | 0.076136 | 0.006726 | 0.003522 |
| 250 | 0.076332 | 0.107386 | 0.018501 | 0.004651 |
| 200 | 0.089716 | 0.123998 | 0.034661 | -0.002250 |
| 150 | 0.060681 | 0.083190 | 0.009915 | -0.002570 |
| 100 | 0.042565 | 0.058353 | 0.003902 | -0.006956 |
| 50 | 0.066705 | 0.098120 | 0.024671 | -0.005213 |

</details>

<details>
<summary><strong>U-Wind Component (U)</strong></summary>

| Level (hPa) | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| 1000 | 0.187488 | 0.269946 | 0.072480 | -0.004291 |
| 925 | 0.178837 | 0.259880 | 0.069784 | -0.006674 |
| 850 | 0.185204 | 0.259067 | 0.070476 | -0.004503 |
| 700 | 0.185468 | 0.256267 | 0.070106 | -0.011347 |
| 600 | 0.185824 | 0.256339 | 0.070676 | -0.011824 |
| 500 | 0.183563 | 0.253114 | 0.068457 | -0.009073 |
| 400 | 0.183034 | 0.251675 | 0.067852 | -0.004083 |
| 300 | 0.171120 | 0.232806 | 0.060858 | -0.005977 |
| 250 | 0.151333 | 0.204488 | 0.049647 | -0.009946 |
| 200 | 0.137866 | 0.186742 | 0.043262 | -0.009769 |
| 150 | 0.131951 | 0.182324 | 0.044485 | -0.005861 |
| 100 | 0.119462 | 0.163879 | 0.039493 | -0.001967 |
| 50 | 0.137343 | 0.216046 | 0.067010 | -0.003718 |

</details>

<details>
<summary><strong>V-Wind Component (V)</strong></summary>

| Level (hPa) | MAE | RMSE | SMSE | Bias |
|---|---|---|---|---|
| 1000 | 0.219780 | 0.315665 | 0.099930 | -0.000575 |
| 925 | 0.221592 | 0.322655 | 0.106627 | -0.000397 |
| 850 | 0.239954 | 0.340497 | 0.117914 | -0.005278 |
| 700 | 0.242971 | 0.340473 | 0.114645 | -0.005305 |
| 600 | 0.243377 | 0.340176 | 0.112703 | 0.000782 |
| 500 | 0.238721 | 0.333327 | 0.106190 | -0.002801 |
| 400 | 0.233180 | 0.325294 | 0.098357 | -0.003267 |
| 300 | 0.218310 | 0.300224 | 0.084443 | -0.005698 |
| 250 | 0.200775 | 0.272890 | 0.072629 | 0.000353 |
| 200 | 0.199732 | 0.270453 | 0.073886 | -0.005473 |
| 150 | 0.215073 | 0.296998 | 0.093659 | -0.005613 |
| 100 | 0.207381 | 0.282123 | 0.090840 | -0.003995 |
| 50 | 0.235091 | 0.364035 | 0.182881 | 0.001212 |

</details>

## 10. Comparison with Original Model (Improvements)
To quantify the domain gap bridged by our finetuning, we compared a 24-hour forecast from both the original Pangu-Weather model and our finetuned version against ground truth IMD GFS observations.

The improvements are highly significant across all key thermodynamic and kinematic variables.

![Performance Comparison](finetune_performance_comparison.png)

### Key Improvements
* **Pressure & Height:** MSL error (RMSE) was reduced by **66%**, and Geopotential Height (Z) error fell by **69%**. The finetuned model almost completely removed the strong negative biases present in the original model.
* **Temperature:** Surface temperature (T2M) error was reduced by **55%**, and upper-air temperature (T) error saw a similar reduction of **56%**.
* **Winds:** Both surface (U10, V10) and upper-air winds (U, V) experienced error reductions of approximately **20% to 25%**. 

## 11. Summary
By systematically finetuning the Pangu-Weather model on regional IMD GFS data, we successfully eliminated large systematic biases inherited from its global ERA5 pre-training. The resulting weights yield a model that is vastly superior for regional forecasting within the IMD data distribution, ensuring that downstream meteorological applications relying on this model will be substantially more accurate.
