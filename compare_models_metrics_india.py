import xarray as xr
import numpy as np
import argparse

def calculate_metrics(true_vals, pred_vals):
    true_vals = true_vals.flatten()
    pred_vals = pred_vals.flatten()
    
    # Remove NaNs
    mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]
    
    if len(true_vals) == 0:
        return {k: np.nan for k in ["rmse", "mae", "mse", "bias", "corr", "r2"]}
    
    error = pred_vals - true_vals
    
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    bias = np.mean(error)
    
    # Pearson Correlation and R-squared
    if np.std(true_vals) == 0 or np.std(pred_vals) == 0:
        corr = np.nan
        r2 = np.nan
    else:
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "bias": bias,
        "corr": corr,
        "r2": r2
    }

def subset_india(ds):
    lat_name = None
    lon_name = None
    for name in ["latitude", "lat", "y"]:
        if name in ds.coords:
            lat_name = name
            break
    for name in ["longitude", "lon", "x"]:
        if name in ds.coords:
            lon_name = name
            break

    if lat_name and lon_name:
        # Handle coordinate slicing carefully, accounting for decreasing latitude arrays
        lat_vals = ds[lat_name].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            ds = ds.sel({lat_name: slice(40.0, 0.0), lon_name: slice(60.0, 100.0)})
        else:
            ds = ds.sel({lat_name: slice(0.0, 40.0), lon_name: slice(60.0, 100.0)})
            
    return ds

def main():
    parser = argparse.ArgumentParser(description="Compare two inference models and true data (INDIA REGION ONLY).")
    parser.add_argument("--true_surface", help="Path to observed surface NC", required=True)
    parser.add_argument("--true_upper", help="Path to observed upper NC", required=True)
    parser.add_argument("--pred_original", help="Path to original predicted NC", required=True)
    parser.add_argument("--pred_finetuned", help="Path to finetuned predicted NC", required=True)
    args = parser.parse_args()

    print("Loading datasets...")
    ds_true_surf = xr.open_dataset(args.true_surface)
    ds_true_upper = xr.open_dataset(args.true_upper)
    ds_orig = xr.open_dataset(args.pred_original)
    ds_fine = xr.open_dataset(args.pred_finetuned)

    # Subset to India region (Lat 0-40, Lon 60-100)
    print("Subsetting to India Region (Lat: 0 to 40, Lon: 60 to 100)...")
    ds_true_surf = subset_india(ds_true_surf)
    ds_true_upper = subset_india(ds_true_upper)
    ds_orig = subset_india(ds_orig)
    ds_fine = subset_india(ds_fine)
    
    # Try to align time for true datasets based on forecast valid time
    try:
        target_time = ds_orig.time.values[0]
        print(f"Target forecast valid time: {target_time}")
        
        for ds_name, ds in [("surface", ds_true_surf), ("upper", ds_true_upper)]:
            if "valid_time" in ds.coords or "valid_time" in ds.dims:
                ds = ds.sel(valid_time=target_time, method="nearest")
            elif "time" in ds.coords or "time" in ds.dims:
                ds = ds.sel(time=target_time, method="nearest")
            
            if ds_name == "surface":
                ds_true_surf = ds
            else:
                ds_true_upper = ds
    except Exception as e:
        print(f"Time alignment info (might use first timestep): {e}")

    surf_vars = ['msl', 'u10', 'v10', 't2m']
    upper_vars = ['z', 'q', 't', 'u', 'v']
    
    # Helper to print metrics
    def print_metrics(var_name, metrics):
        print(f"Variable: {var_name.upper()}")
        for k, v in metrics.items():
            print(f"  {k.upper():4s}: {v:.4f}")
        print("-" * 30)

    print("\n" + "="*50)
    print("1. OBSERVED vs ORIGINAL (INDIA REGION)")
    print("="*50)
    for var in surf_vars:
        if var in ds_true_surf and var in ds_orig:
            arr_true = ds_true_surf[var].values.squeeze()
            arr_pred = ds_orig[var].values.squeeze()
            if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
            if arr_true.shape != arr_pred.shape:
                print(f"Shape mismatch for {var}: True {arr_true.shape}, Pred {arr_pred.shape}. Skipping.")
                continue
            metrics = calculate_metrics(arr_true, arr_pred)
            print_metrics(var, metrics)
            
    for var in upper_vars:
        if var not in ds_true_upper:
            continue
        if 'level' in ds_true_upper.coords:
            for lvl in ds_true_upper['level'].values:
                if lvl not in ds_orig['level'].values: continue
                arr_true = ds_true_upper[var].sel(level=lvl).values.squeeze()
                arr_pred = ds_orig[var].sel(level=lvl).values.squeeze()
                if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                if arr_true.shape != arr_pred.shape: continue
                metrics = calculate_metrics(arr_true, arr_pred)
                print_metrics(f"{var} (Level {lvl})", metrics)
        else:
            arr_true = ds_true_upper[var].values.squeeze()
            arr_pred = ds_orig[var].values.squeeze()
            if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
            if arr_true.shape != arr_pred.shape: continue
            metrics = calculate_metrics(arr_true, arr_pred)
            print_metrics(var, metrics)

    print("\n" + "="*50)
    print("2. OBSERVED vs FINETUNED (INDIA REGION)")
    print("="*50)
    for var in surf_vars:
        if var in ds_true_surf and var in ds_fine:
            arr_true = ds_true_surf[var].values.squeeze()
            arr_pred = ds_fine[var].values.squeeze()
            if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
            if arr_true.shape != arr_pred.shape: continue
            metrics = calculate_metrics(arr_true, arr_pred)
            print_metrics(var, metrics)

    for var in upper_vars:
        if var not in ds_true_upper:
            continue
        if 'level' in ds_true_upper.coords:
            for lvl in ds_true_upper['level'].values:
                if lvl not in ds_fine['level'].values: continue
                arr_true = ds_true_upper[var].sel(level=lvl).values.squeeze()
                arr_pred = ds_fine[var].sel(level=lvl).values.squeeze()
                if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                if arr_true.shape != arr_pred.shape: continue
                metrics = calculate_metrics(arr_true, arr_pred)
                print_metrics(f"{var} (Level {lvl})", metrics)
        else:
            arr_true = ds_true_upper[var].values.squeeze()
            arr_pred = ds_fine[var].values.squeeze()
            if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
            if arr_true.shape != arr_pred.shape: continue
            metrics = calculate_metrics(arr_true, arr_pred)
            print_metrics(var, metrics)

    print("\n" + "="*50)
    print("3. ORIGINAL vs FINETUNED (INDIA REGION)")
    print("="*50)
    for var in surf_vars + upper_vars:
        if var in ds_orig and var in ds_fine:
            arr_orig = ds_orig[var].values.squeeze()
            arr_fine = ds_fine[var].values.squeeze()
            if arr_orig.shape != arr_fine.shape: continue
            metrics = calculate_metrics(arr_orig, arr_fine)
            print_metrics(var, metrics)

if __name__ == "__main__":
    main()
