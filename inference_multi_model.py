import os
import sys
import ctypes

# ---------------------------------------------------------------------------
# RHEL/CentOS libstdc++ workaround
# ---------------------------------------------------------------------------
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _libstdcxx = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(_libstdcxx):
        try:
            ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
            print(f"[init] Preloaded {_libstdcxx}", flush=True)
        except OSError as _e:
            print(f"[init] WARNING: could not preload libstdc++: {_e}", flush=True)

import argparse
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# NFS mmap SIGBUS workaround for onnxruntime
# ---------------------------------------------------------------------------
_ORT_LOCAL = "/dev/shm/pylocal"
if os.path.isdir(_ORT_LOCAL) and _ORT_LOCAL not in sys.path:
    sys.path.insert(0, _ORT_LOCAL)

os.environ.setdefault("ONNXRUNTIME_PREFER_NO_MMAP", "1")

print("[init] importing onnxruntime...", flush=True)
try:
    import onnxruntime as ort
    print(f"[init] onnxruntime {ort.__version__} loaded from: {ort.__file__}", flush=True)
    _ORT_AVAILABLE = True
except Exception as e:
    print(f"[init] WARNING: onnxruntime import failed: {e}", flush=True)
    _ORT_AVAILABLE = False

import numpy as np
import xarray as xr
import torch

from finetune_entire_normalized import (
    PanguModel,
    load_model_state,
    loadAllConstants,
    normBackData,
    cfg,
)
from train14_small_era5 import Pangu_lite

EPSILON = 0.622

# ---------------------------------------------------------------------------
# Helper: open a NetCDF file WITHOUT memory-mapping.
# ---------------------------------------------------------------------------
def _open_nc_nommap(path):
    try:
        ds = xr.open_dataset(path, engine="scipy", decode_timedelta=False)
    except Exception:
        ds = xr.open_dataset(path, decode_timedelta=False)
        ds.load()
    return ds


def _pick_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name, ds.coords[name].values
        if name in ds.variables:
            return name, ds[name].values
    return None, None


def _ensure_descending_lat(arr, lat_vals):
    if lat_vals[0] < lat_vals[-1]:
        arr = arr[..., ::-1, :].copy()
        lat_vals = lat_vals[::-1].copy()
    return arr, lat_vals


def _ensure_descending_levels(arr, level_vals):
    if level_vals[0] < level_vals[-1]:
        arr = arr[:, ::-1, :, :].copy()
        level_vals = level_vals[::-1].copy()
    return arr, level_vals


def _relative_humidity_to_specific_humidity(relative_humidity, temperature, pressure_hpa):
    rh = np.asarray(relative_humidity, dtype=np.float32)
    temp = np.asarray(temperature, dtype=np.float32)
    rh_max = np.nanmax(rh)
    rh_fraction = rh if rh_max <= 1.5 else rh / 100.0
    rh_fraction = np.clip(rh_fraction, 0.0, 1.0)
    temp_c = temp - 273.15
    saturation_vapor_pressure = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    vapor_pressure = rh_fraction * saturation_vapor_pressure
    pressure_pa = np.asarray(pressure_hpa, dtype=np.float32) * 100.0
    vapor_pressure = np.minimum(vapor_pressure, pressure_pa * 0.99)
    q = (EPSILON * vapor_pressure) / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)
    return np.asarray(q, dtype=np.float32)


def _broadcast_pressure(pressure_vals, dims, level_name):
    if level_name not in dims:
        raise KeyError(f"Pressure level '{level_name}' not found in data dims {dims}")
    shape = [1] * len(dims)
    shape[dims.index(level_name)] = len(pressure_vals)
    return np.asarray(pressure_vals, dtype=np.float32).reshape(shape)


def _resolve_upper_q(ds_upper):
    if "q" in ds_upper:
        q = ds_upper["q"].values.astype(np.float32)
        return q
    if "r" not in ds_upper:
        raise KeyError("Upper-air file has neither q nor r for humidity.")
    if "t" not in ds_upper:
        raise KeyError("Cannot convert r to q because temperature t is missing.")

    level_name, level_vals = _pick_coord(ds_upper, ("level", "isobaricInhPa", "pressure_level"))
    if level_vals is None:
        raise KeyError("Cannot convert r to q because pressure levels are missing.")

    rh = ds_upper["r"].values
    temp = ds_upper["t"].values
    pressure = _broadcast_pressure(level_vals, ds_upper["r"].dims, level_name)
    q = _relative_humidity_to_specific_humidity(rh, temp, pressure)
    return q


def _ensure_time_index(ds, target_time, label):
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    if "time" not in ds.dims:
        return ds

    has_time_index = "time" in ds.coords and "time" in ds.indexes
    if has_time_index:
        return ds

    if ds.sizes.get("time") == 1:
        print(f"  [time-fix] {label}: assigning missing time coordinate {target_time}", flush=True)
        return ds.assign_coords(time=("time", [np.datetime64(target_time)]))

    raise ValueError(
        f"{label} has a time dimension but no time coordinate/index. "
        "Cannot select a timestep from a multi-time file without timestamps."
    )


def _find_nc_path(data_root, kind, dt):
    """Return the NetCDF path for surface or upper-air data at datetime dt. Supports daily and monthly."""
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
    raise FileNotFoundError(
        f"No {kind} NetCDF found for {dt:%Y-%m-%d %H:00}. Tried: {candidates}"
    )


def _load_single_step(data_root, start_time, horizon_hours):
    surface_path = _find_nc_path(data_root, "surface", start_time)
    upper_path = _find_nc_path(data_root, "upper", start_time)

    print(f"Loading surface: {surface_path}")
    print(f"Loading upper:   {upper_path}")

    ds_surface = _open_nc_nommap(surface_path)
    ds_upper = _open_nc_nommap(upper_path)

    ds_surface = _ensure_time_index(ds_surface, start_time, "surface")
    ds_upper = _ensure_time_index(ds_upper, start_time, "upper")

    sel_kw = dict(method="nearest")
    ds_surface = ds_surface.sel(time=start_time, **sel_kw)
    ds_upper = ds_upper.sel(time=start_time, **sel_kw)

    def _resolve_expver(ds):
        if "expver" in ds.dims:
            if "expver" in ds.indexes:
                if 5 in ds.indexes["expver"]:
                    return ds.sel(expver=5)
                elif 1 in ds.indexes["expver"]:
                    return ds.sel(expver=1)
            return ds.isel(expver=0)
        return ds

    ds_surface = _resolve_expver(ds_surface)
    ds_upper = _resolve_expver(ds_upper)

    lat_name, lat_vals = _pick_coord(ds_surface, ("latitude", "lat", "y"))
    lon_name, lon_vals = _pick_coord(ds_surface, ("longitude", "lon", "x"))
    level_name_raw, level_vals = _pick_coord(ds_upper, ("level", "isobaricInhPa", "pressure_level"))

    upper_z = ds_upper["z"].values.astype(np.float32)
    upper_q = _resolve_upper_q(ds_upper)
    upper_t = ds_upper["t"].values.astype(np.float32)
    upper_u = ds_upper["u"].values.astype(np.float32)
    upper_v = ds_upper["v"].values.astype(np.float32)
    z_max = float(np.nanmax(np.abs(upper_z)))
    if z_max < 20000:
        upper_z = upper_z * 9.80665

    input_upper = np.stack([upper_z, upper_q, upper_t, upper_u, upper_v], axis=0)

    if level_vals is not None:
        input_upper, level_vals = _ensure_descending_levels(input_upper, level_vals)

    input_upper_stack = input_upper.reshape(-1, len(lat_vals), len(lon_vals))
    input_upper_stack, lat_vals_fixed = _ensure_descending_lat(input_upper_stack, lat_vals)
    input_upper = input_upper_stack.reshape(input_upper.shape)

    surface_msl = ds_surface["msl"].values.astype(np.float32)
    surface_u10 = ds_surface["u10"].values.astype(np.float32)
    surface_v10 = ds_surface["v10"].values.astype(np.float32)
    surface_t2m = ds_surface["t2m"].values.astype(np.float32)
    input_surface = np.stack([surface_msl, surface_u10, surface_v10, surface_t2m], axis=0)
    input_surface_stack = input_surface.reshape(-1, len(lat_vals), len(lon_vals))
    input_surface_stack, _ = _ensure_descending_lat(input_surface_stack, lat_vals)
    input_surface = input_surface_stack.reshape(input_surface.shape)

    ds_surface.close()
    ds_upper.close()

    end_time = start_time + timedelta(hours=horizon_hours)
    return input_upper, input_surface, end_time, lat_vals_fixed, lon_vals


def _build_coords(data_root, ref_time):
    surface_path = _find_nc_path(data_root, "surface", ref_time)
    upper_path = _find_nc_path(data_root, "upper", ref_time)

    ds_surface = _open_nc_nommap(surface_path)
    ds_upper = _open_nc_nommap(upper_path)

    lat_name, lat_vals = _pick_coord(ds_surface, ("latitude", "lat", "y"))
    lon_name, lon_vals = _pick_coord(ds_surface, ("longitude", "lon", "x"))
    level_name, level_vals = _pick_coord(ds_upper, ("level", "isobaricInhPa", "pressure_level"))

    if level_vals is None:
        level_name = "level"
        level_vals = np.array([int(x) for x in cfg.ERA5_UPPER_LEVELS], dtype=np.int32)

    if lat_vals is not None and lat_vals[0] < lat_vals[-1]:
        lat_vals = lat_vals[::-1].copy()

    ds_surface.close()
    ds_upper.close()

    return lat_name, lat_vals, lon_name, lon_vals, level_name, level_vals


def _save_forecast(output_upper, output_surface, lat_vals, lon_vals, data_root, start_time, forecast_time, output_dir):
    lat_name, lat_vals_file, lon_name, lon_vals_file, level_name, level_vals = _build_coords(
        data_root, start_time
    )
    if lat_vals is None:
        lat_vals = lat_vals_file
    if lon_vals is None:
        lon_vals = lon_vals_file
    time_vals = np.array([np.datetime64(forecast_time)])

    if level_vals[0] < level_vals[-1]:
        output_upper_save = output_upper[:, ::-1, :, :]
    else:
        output_upper_save = output_upper

    ds_out = xr.Dataset(
        data_vars={
            "z": (("time", level_name, lat_name, lon_name), output_upper_save[0][None, ...]),
            "q": (("time", level_name, lat_name, lon_name), output_upper_save[1][None, ...]),
            "t": (("time", level_name, lat_name, lon_name), output_upper_save[2][None, ...]),
            "u": (("time", level_name, lat_name, lon_name), output_upper_save[3][None, ...]),
            "v": (("time", level_name, lat_name, lon_name), output_upper_save[4][None, ...]),
            "msl": (("time", lat_name, lon_name), output_surface[0][None, ...]),
            "u10": (("time", lat_name, lon_name), output_surface[1][None, ...]),
            "v10": (("time", lat_name, lon_name), output_surface[2][None, ...]),
            "t2m": (("time", lat_name, lon_name), output_surface[3][None, ...]),
        },
        coords={
            "time": time_vals,
            level_name: level_vals,
            lat_name: lat_vals,
            lon_name: lon_vals,
        },
        attrs={"forecast_time": forecast_time.strftime("%Y-%m-%d %H:%M:%S")},
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"forecast_{forecast_time:%Y%m%d_%H%M}.nc")
    ds_out.to_netcdf(out_path)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Pangu-lite Normalization Helpers
# ---------------------------------------------------------------------------
def load_lite_statistics(aux_path, device):
    s_mean = np.load(os.path.join(aux_path, "surface_mean.npy")).astype(np.float32)
    s_std = np.load(os.path.join(aux_path, "surface_std.npy")).astype(np.float32)
    u_mean = np.load(os.path.join(aux_path, "upper_mean.npy")).astype(np.float32)
    u_std = np.load(os.path.join(aux_path, "upper_std.npy")).astype(np.float32)

    # u_mean/std are stored as [13, 1, 1, 5] with levels 50 -> 1000
    # Model expects [5, 13, 1, 1] with levels 1000 -> 50
    u_mean = u_mean[::-1, :, :, :].copy() # 1000 -> 50
    u_std = u_std[::-1, :, :, :].copy()

    u_mean = u_mean.transpose(3, 0, 1, 2) # [5, 13, 1, 1]
    u_std = u_std.transpose(3, 0, 1, 2)
    
    s_mean = s_mean.reshape(4, 1, 1)
    s_std = s_std.reshape(4, 1, 1)

    return (
        torch.from_numpy(s_mean).unsqueeze(0).to(device),
        torch.from_numpy(s_std).unsqueeze(0).to(device),
        torch.from_numpy(u_mean).unsqueeze(0).to(device),
        torch.from_numpy(u_std).unsqueeze(0).to(device),
    )


def norm_lite(upper, surface, stats):
    s_m, s_s, u_m, u_s = stats
    return (upper - u_m) / u_s, (surface - s_m) / s_s


def norm_back_lite(upper, surface, stats):
    s_m, s_s, u_m, u_s = stats
    return (upper * u_s) + u_m, (surface * s_s) + s_m


def get_lite_masks(aux_path, device):
    mask_path = os.path.join(aux_path, "constantMask24.npy")
    mask = np.load(mask_path).astype(np.float32)
    mask = mask[0, :, :721, :] # [3, 721, 1440]
    surface_mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    const_h_path = os.path.join(aux_path, "Constant_17_output_0.npy")
    const_h = np.load(const_h_path).astype(np.float32)
    const_h = np.squeeze(const_h)[:, :721, :] # [1, 721, 1440]
    const_h = torch.from_numpy(const_h).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 721, 1440]
    return surface_mask, const_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["original", "finetuned", "lite"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--aux_data_dir", default="", help="Path to aux_data. Defaults to data_root/aux_data.")
    parser.add_argument("--start_time", default="20230605 00:00:00")
    parser.add_argument("--horizon_hours", type=int, default=24)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output_subdir", default="inference")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="ORT intra_op threads (0 = use all CPU cores)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.PG_INPUT_PATH = args.data_root
    cfg.PG.HORIZON = args.horizon_hours
    
    aux_path = args.aux_data_dir if args.aux_data_dir else os.path.join(args.data_root, "aux_data")

    start_time = datetime.strptime(args.start_time, "%Y%m%d %H:%M:%S")

    print("Loading initial input data...")
    input_upper, input_surface, _, start_lat_vals, start_lon_vals = _load_single_step(
        args.data_root, start_time, args.horizon_hours
    )
    sys.stdout.flush()

    curr_upper = torch.from_numpy(input_upper).unsqueeze(0).to(device)
    curr_surface = torch.from_numpy(input_surface).unsqueeze(0).to(device)

    is_onnx = args.model_path.endswith(".onnx")
    
    if args.model_type in ["original", "finetuned"]:
        if is_onnx:
            # ONNX Setup (Original model could be ONNX format)
            if not _ORT_AVAILABLE:
                raise RuntimeError("onnxruntime not available.")
            num_threads = args.num_threads if args.num_threads > 0 else min(8, os.cpu_count() or 1)
            options = ort.SessionOptions()
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.enable_mem_reuse = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            with open(args.model_path, "rb") as _f:
                model_bytes = _f.read()
            ort_session = ort.InferenceSession(model_bytes, sess_options=options, providers=providers)
            model = None
            constants = None
        else:
            # PyTorch PanguModel
            model = PanguModel(device=device).to(device)
            state = torch.load(args.model_path, map_location=device)
            load_model_state(model, state, strict=True)
            model.eval()
            
            # Temporary override of cfg to load constants from correct aux path if different
            original_input_path = cfg.PG_INPUT_PATH
            cfg.PG_INPUT_PATH = os.path.dirname(aux_path)
            constants = loadAllConstants(device=device)
            cfg.PG_INPUT_PATH = original_input_path
            
    elif args.model_type == "lite":
        if is_onnx:
            raise NotImplementedError("ONNX is not supported for Pangu-lite.")
        model = Pangu_lite(residual=True).to(device)
        state = torch.load(args.model_path, map_location=device)
        # Handle state dict if nested
        if 'model' in state: state = state['model']
        model.load_state_dict(state, strict=False)
        model.eval()
        
        # Load Lite specific stats and masks
        lite_stats = load_lite_statistics(aux_path, device)
        lite_mask_surface, lite_mask_const_h = get_lite_masks(aux_path, device)
    
    forecast_time = start_time
    output_dir = os.path.join(args.output_root, args.output_subdir)

    for step in range(1, args.steps + 1):
        forecast_time = forecast_time + timedelta(hours=args.horizon_hours)
        print(f"\n--- Step {step}/{args.steps}: forecasting {forecast_time} ---")

        if args.model_type in ["original", "finetuned"]:
            if is_onnx:
                upper_np = np.ascontiguousarray(curr_upper.cpu().squeeze(0).float().numpy())
                surface_np = np.ascontiguousarray(curr_surface.cpu().squeeze(0).float().numpy())
                ort_inputs = {"input": upper_np, "input_surface": surface_np}
                outputs = ort_session.run(None, ort_inputs)
                out_upper = torch.from_numpy(np.array(outputs[0])).unsqueeze(0).to(device)
                out_surface = torch.from_numpy(np.array(outputs[1])).unsqueeze(0).to(device)
            else:
                with torch.no_grad():
                    out_upper, out_surface = model(
                        curr_upper,
                        curr_surface,
                        constants["weather_statistics"],
                        constants["constant_maps"],
                        constants["const_h"],
                    )
                    out_upper, out_surface = normBackData(
                        out_upper, out_surface, constants["weather_statistics_last"]
                    )
        elif args.model_type == "lite":
            with torch.no_grad():
                # Normalize input
                norm_upper, norm_surface = norm_lite(curr_upper, curr_surface, lite_stats)
                
                # Forward pass
                out_surface, out_upper = model(
                    norm_surface,
                    lite_mask_surface,
                    norm_upper,
                    lite_mask_const_h
                )
                
                # Inverse normalize
                out_upper, out_surface = norm_back_lite(out_upper, out_surface, lite_stats)

        out_upper_np = out_upper.cpu().squeeze(0).numpy()
        out_surface_np = out_surface.cpu().squeeze(0).numpy()

        _save_forecast(
            output_upper=out_upper_np,
            output_surface=out_surface_np,
            lat_vals=start_lat_vals,
            lon_vals=start_lon_vals,
            data_root=args.data_root,
            start_time=start_time,
            forecast_time=forecast_time,
            output_dir=output_dir,
        )

        curr_upper = out_upper
        curr_surface = out_surface

    print("\nAutoregressive forecast complete.")

if __name__ == "__main__":
    main()
