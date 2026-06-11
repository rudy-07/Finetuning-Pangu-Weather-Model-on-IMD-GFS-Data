import xarray as xr
import numpy as np
from tqdm import tqdm

import os
from os.path import join
from os import listdir
import argparse
import warnings

# Suppress xarray FutureWarning about timedelta decoding
warnings.simplefilter(action='ignore', category=FutureWarning)


def cal_mean(filenames, surface_variables, pLevel=None):
    surface_map = {v: {"batch": 0, "sumV": 0} for v in surface_variables}
    for file in tqdm(filenames):
        ds = xr.open_dataset(file)
        # upper_air
        if pLevel is not None:
            if 'level' in ds.dims or 'level' in ds.coords:
                ds = ds.sel(level=pLevel)
            elif 'pressure_level' in ds.dims or 'pressure_level' in ds.coords:
                ds = ds.sel(pressure_level=pLevel)
            else:
                raise KeyError(f"No valid level dimension found. Available dims: {list(ds.dims)}")

        for v in surface_variables:
            data = ds[v].data
            batch = (data != np.nan).sum()
            sumV = np.nansum(data)

            surface_map[v]["batch"] += batch
            surface_map[v]["sumV"] += sumV

    surface_mean = {v: surface_map[v]["sumV"] / surface_map[v]["batch"] for v in surface_map}
    if pLevel is None:
        return surface_mean
    
    # upper_air
    return {pLevel: surface_mean}


def cal_std(filenames, surface_variables, surface_mean, pLevel=None):
    surface_map = {v: {"batch": 0, "sumV": 0} for v in surface_variables}
    for file in tqdm(filenames):
        ds = xr.open_dataset(file)
        # upper_air
        if pLevel is not None:
            if 'level' in ds.dims or 'level' in ds.coords:
                ds = ds.sel(level=pLevel)
            elif 'pressure_level' in ds.dims or 'pressure_level' in ds.coords:
                ds = ds.sel(pressure_level=pLevel)
            else:
                raise KeyError(f"No valid level dimension found. Available dims: {list(ds.dims)}")

        for v in surface_variables:
            data = ds[v].data
            batch = (data != np.nan).sum()
            sumV = np.nansum(np.abs(data - surface_mean[v]) ** 2)

            surface_map[v]["batch"] += batch
            surface_map[v]["sumV"] += sumV

    surface_std = {v: np.sqrt(surface_map[v]["sumV"] / surface_map[v]["batch"]) for v in surface_map}
    if pLevel is None:
        return surface_std
    
    # upper_air
    return {pLevel: surface_std}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Mean and Std of NetCDF dataset")
    parser.add_argument("--dataset_root", type=str, default="data", help="Path to the dataset containing train/valid/test folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the .npy files. Defaults to dataset_root/aux_data")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir if args.output_dir else join(dataset_root, "aux_data")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")

    # cal surface mean std
    filenames = []
    surface_dir = join(dataset_root, "surface")
    if os.path.exists(surface_dir):
        for file in listdir(surface_dir):
            if file.startswith("surface") and file.endswith(".nc"):
                filenames.append(join(surface_dir, file))
    
    if not filenames:
        raise ValueError(f"No surface netCDF files found in {dataset_root}!")
    
    print(f"Found {len(filenames)} surface netCDF files.")

    # Calculate values
    # NOTE: The loop calculation uses ["u10", "v10", "t2m", "msl"]
    calc_surface_vars = ["u10", "v10", "t2m", "msl"]
    
    print("\nCalculating surface means...")
    surface_mean_dict = cal_mean(filenames, calc_surface_vars)
    print("Calculating surface stds...")
    surface_std_dict = cal_std(filenames, calc_surface_vars, surface_mean_dict)

    # Convert to Pangu's expected order and shape: (4,)
    # Order must strictly be: msl, u10, v10, t2m
    target_surface_vars = ['msl', 'u10', 'v10', 't2m']
    
    surface_mean_arr = np.array([surface_mean_dict[v] for v in target_surface_vars], dtype=np.float32)
    surface_std_arr = np.array([surface_std_dict[v] for v in target_surface_vars], dtype=np.float32)

    ## save surface npy
    np.save(join(output_dir, "surface_mean.npy"), surface_mean_arr)
    np.save(join(output_dir, "surface_std.npy"), surface_std_arr)
    print(f"Saved surface_mean.npy and surface_std.npy to {output_dir}")

    # cal upper_air 
    filenames = []
    upper_dir = join(dataset_root, "upper")
    if os.path.exists(upper_dir):
        for file in listdir(upper_dir):
            if (file.startswith("upper_air") or file.startswith("upper")) and file.endswith(".nc"):
                filenames.append(join(upper_dir, file))
                
    if not filenames:
        raise ValueError(f"No upper_air netCDF files found in {dataset_root}!")
        
    print(f"\nFound {len(filenames)} upper_air netCDF files.")

    pLevels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    upper_air_variables = ['z', 'q', 't', 'u', 'v']

    ## cal upper_air mean and std
    upper_air_mean = {}
    upper_air_std = {}
    for pl in pLevels:
        print(f"\nCalculating upper_air mean for level {pl}...")
        upper_air_mean = {**upper_air_mean, **cal_mean(filenames, upper_air_variables, pl)}
        print(f"Calculating upper_air std for level {pl}...")
        upper_air_std = {**upper_air_std, **cal_std(filenames, upper_air_variables, upper_air_mean[pl], pl)}

    # Convert to Pangu's expected shape: (13, 1, 1, 5)
    upper_mean_arr = np.zeros((13, 1, 1, 5), dtype=np.float32)
    upper_std_arr = np.zeros((13, 1, 1, 5), dtype=np.float32)
    
    for i, pl in enumerate(pLevels):
        for j, v in enumerate(upper_air_variables):
            upper_mean_arr[i, 0, 0, j] = upper_air_mean[pl][v]
            upper_std_arr[i, 0, 0, j] = upper_air_std[pl][v]

    ## save upper_air npy
    np.save(join(output_dir, "upper_mean.npy"), upper_mean_arr)
    np.save(join(output_dir, "upper_std.npy"), upper_std_arr)
    print(f"Saved upper_mean.npy and upper_std.npy to {output_dir}")