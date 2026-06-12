import xarray as xr
import numpy as np
import argparse
import os
from datetime import datetime, timedelta

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

def _find_nc_path(data_root, kind, dt):
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    hour = dt.strftime("%H")
    if kind == "surface":
        candidates = [
            os.path.join(data_root, "surface", f"surface_{year}_{month}_{day}_{hour}.nc"),
            os.path.join(data_root, "surface", f"surface_{year}_{month}_{day}.nc"),
            os.path.join(data_root, f"surface_{year}_{month}_{day}_{hour}.nc"),
            os.path.join(data_root, f"surface_{year}_{month}_{day}.nc"),
            os.path.join(data_root, "surface", f"surface_{year}_{month}.nc"),
            os.path.join(data_root, f"surface_{year}_{month}.nc"),
        ]
    else:  # upper
        candidates = [
            os.path.join(data_root, "upper", f"upper_air_{year}_{month}_{day}_{hour}.nc"),
            os.path.join(data_root, "upper", f"upper_air_{year}_{month}_{day}.nc"),
            os.path.join(data_root, f"upper_air_{year}_{month}_{day}_{hour}.nc"),
            os.path.join(data_root, f"upper_air_{year}_{month}_{day}.nc"),
            os.path.join(data_root, "upper", f"upper_air_{year}_{month}.nc"),
            os.path.join(data_root, f"upper_air_{year}_{month}.nc"),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare two inference models and true data for multi-step autoregressive forecasts.")
    parser.add_argument("--data_root", help="Path to observed data root (contains surface/ upper/ dirs)", required=True)
    parser.add_argument("--pred_original", help="Path to original predicted NC directory (e.g. output_root/inference)", required=True)
    parser.add_argument("--pred_finetuned", help="Path to finetuned predicted NC directory (e.g. output_root/inference)", required=True)
    parser.add_argument("--start_time", help="Start time (e.g. '20260527 00:00:00')", required=True)
    parser.add_argument("--steps", type=int, default=10, help="Number of forecast steps")
    parser.add_argument("--horizon_hours", type=int, default=24, help="Hours per step")
    parser.add_argument("--output_txt", default="evaluation_metrics_10day.txt", help="Path to save output text file")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_time, "%Y%m%d %H:%M:%S")

    surf_vars = ['msl', 'u10', 'v10', 't2m']
    upper_vars = ['z', 'q', 't', 'u', 'v']

    with open(args.output_txt, 'w') as f_out:
        def log(msg=""):
            print(msg)
            f_out.write(msg + "\n")

        def print_metrics(var_name, metrics):
            log(f"  Variable: {var_name.upper()}")
            for k, v in metrics.items():
                log(f"    {k.upper():4s}: {v:.4f}")
            log("-" * 30)

        for step in range(1, args.steps + 1):
            target_dt = start_dt + timedelta(hours=args.horizon_hours * step)
            log("\n" + "#"*80)
            log(f"### STEP {step} | FORECAST TARGET: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} ###")
            log("#"*80)

            # File paths
            pred_orig_path = os.path.join(args.pred_original, f"forecast_{target_dt:%Y%m%d_%H%M}.nc")
            pred_fine_path = os.path.join(args.pred_finetuned, f"forecast_{target_dt:%Y%m%d_%H%M}.nc")
            
            if not os.path.exists(pred_orig_path):
                log(f"Original prediction missing: {pred_orig_path}")
                continue
            if not os.path.exists(pred_fine_path):
                log(f"Finetuned prediction missing: {pred_fine_path}")
                continue

            true_surface_path = _find_nc_path(args.data_root, "surface", target_dt)
            true_upper_path = _find_nc_path(args.data_root, "upper", target_dt)

            if not true_surface_path or not true_upper_path:
                log(f"Missing true data for step {step}. Skipping.")
                continue

            ds_true_surf = xr.open_dataset(true_surface_path)
            ds_true_upper = xr.open_dataset(true_upper_path)
            ds_orig = xr.open_dataset(pred_orig_path)
            ds_fine = xr.open_dataset(pred_fine_path)

            # Align time for true datasets
            try:
                target_time = ds_orig.time.values[0]
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
                pass

            log("\n" + "="*50)
            log(f"1. OBSERVED vs ORIGINAL | STEP {step}")
            log("="*50)
            for var in surf_vars:
                if var in ds_true_surf and var in ds_orig:
                    arr_true = ds_true_surf[var].values.squeeze()
                    arr_pred = ds_orig[var].values.squeeze()
                    if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                    if arr_true.shape != arr_pred.shape: continue
                    metrics = calculate_metrics(arr_true, arr_pred)
                    print_metrics(var, metrics)
                    
            for var in upper_vars:
                if var not in ds_true_upper: continue
                arr_true = ds_true_upper[var].values.squeeze()
                arr_pred = ds_orig[var].values.squeeze()
                if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                if arr_true.shape != arr_pred.shape: continue
                metrics = calculate_metrics(arr_true, arr_pred)
                print_metrics(var, metrics)

            log("\n" + "="*50)
            log(f"2. OBSERVED vs FINETUNED | STEP {step}")
            log("="*50)
            for var in surf_vars:
                if var in ds_true_surf and var in ds_fine:
                    arr_true = ds_true_surf[var].values.squeeze()
                    arr_pred = ds_fine[var].values.squeeze()
                    if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                    if arr_true.shape != arr_pred.shape: continue
                    metrics = calculate_metrics(arr_true, arr_pred)
                    print_metrics(var, metrics)

            for var in upper_vars:
                if var not in ds_true_upper: continue
                arr_true = ds_true_upper[var].values.squeeze()
                arr_pred = ds_fine[var].values.squeeze()
                if arr_true.ndim > arr_pred.ndim: arr_true = arr_true[0]
                if arr_true.shape != arr_pred.shape: continue
                metrics = calculate_metrics(arr_true, arr_pred)
                print_metrics(var, metrics)
                
            log("\n" + "="*50)
            log(f"3. ORIGINAL vs FINETUNED | STEP {step}")
            log("="*50)
            for var in surf_vars + upper_vars:
                if var in ds_orig and var in ds_fine:
                    arr_orig = ds_orig[var].values.squeeze()
                    arr_fine = ds_fine[var].values.squeeze()
                    if arr_orig.shape != arr_fine.shape: continue
                    metrics = calculate_metrics(arr_orig, arr_fine)
                    print_metrics(var, metrics)

            ds_true_surf.close()
            ds_true_upper.close()
            ds_orig.close()
            ds_fine.close()

if __name__ == "__main__":
    main()
