import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import Ngl as ng
import shapefile

def _find_nc_path(data_root, mode, valid_date, file_type):
    if mode == "forecast":
        cand = os.path.join(data_root, f"forecast_{valid_date.strftime('%Y%m%d_%H%M')}.nc")
        if os.path.exists(cand): return cand
        cand = os.path.join(data_root, f"forecast_{valid_date.strftime('%Y%m%d')}.nc")
        if os.path.exists(cand): return cand
    else:
        year = valid_date.strftime("%Y")
        month = valid_date.strftime("%m")
        day = valid_date.strftime("%d")
        
        prefix = "surface" if file_type == "surface" else "upper_air"
        subdir = "surface" if file_type == "surface" else "upper"
        
        candidates = [
            os.path.join(data_root, subdir, f"{prefix}_{year}_{month}_{day}.nc"),
            os.path.join(data_root, f"{prefix}_{year}_{month}_{day}.nc"),
            os.path.join(data_root, subdir, f"{prefix}_{year}_{month}.nc"),
            os.path.join(data_root, f"{prefix}_{year}_{month}.nc"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
    return None

def _resolve_expver(ds):
    if "expver" in ds.dims:
        if "expver" in ds.indexes:
            if 5 in ds.indexes["expver"]:
                return ds.sel(expver=5)
            elif 1 in ds.indexes["expver"]:
                return ds.sel(expver=1)
        return ds.isel(expver=0)
    return ds

def overlay_shapefile(wks, plot, shapefile_path):
    if os.path.exists(shapefile_path):
        try:
            sf = shapefile.Reader(shapefile_path)
            for shape_rec in sf.shapeRecords():
                x = np.array([pt[0] for pt in shape_rec.shape.points], dtype="f")
                y = np.array([pt[1] for pt in shape_rec.shape.points], dtype="f")
                plres = ng.Resources()
                plres.gsMarkerSizeF = 0.000003
                plres.gsLineColor = "black"
                ng.add_polymarker(wks, plot, x, y, plres)
        except Exception as e:
            print(f"[WARN] Failed to overlay shapefile: {e}")
    else:
        print(f"[WARN] Shapefile not found at {shapefile_path}. Skipping overlay.")

def plot_t2m(ds_sel, out_path, main_title, second_line, shapefile_path, lat, lon):
    temp = ds_sel["t2m"].values.astype(np.float32)
    if np.nanmax(temp) > 100: temp = temp - 273.15

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        temp = temp[::-1, :].copy()

    wks = ng.open_wks("png", out_path)
    res = ng.Resources()
    res.nglDraw = False
    res.nglFrame = False

    res.cnFillOn = True
    res.cnFillMode = "AreaFill"
    res.cnFillPalette = "BlAqGrYeOrReVi200"
    res.cnLinesOn = True
    res.cnLineLabelsOn = False

    res.sfXArray = lon
    res.sfYArray = lat

    res.mpLimitMode = "LatLon"
    res.mpMinLatF = 0.0
    res.mpMaxLatF = 40.0
    res.mpMinLonF = 60.0
    res.mpMaxLonF = 100.0

    res.vpXF = 0.15
    res.vpYF = 0.80
    res.vpWidthF = 0.7
    res.vpHeightF = 0.7

    res.cnLevelSelectionMode = "ManualLevels"
    res.cnMinLevelValF = 18.0
    res.cnMaxLevelValF = 42.0
    res.cnLevelSpacingF = 2.0

    res.lbTitleOn = False
    res.pmLabelBarWidthF = 0.1
    res.pmLabelBarHeightF = 0.6
    res.pmLabelBarOrthogonalPosF = 0.0001
    res.lbLabelFontHeightF = 0.01

    res.tiMainOn = True
    res.tiMainString = f"{main_title}~C~{second_line}"
    res.tiMainFontHeightF = 0.015
    res.tiMainOffsetYF = 0.02

    x_border = [0.01, 0.99, 0.99, 0.01, 0.01]
    y_border = [0.01, 0.01, 0.99, 0.99, 0.01]
    border_res = ng.Resources()
    border_res.gsLineColor = "black"
    border_res.gsLineThicknessF = 3.0
    ng.polyline_ndc(wks, x_border, y_border, border_res)

    plot = ng.contour_map(wks, temp, res)
    overlay_shapefile(wks, plot, shapefile_path)

    ng.draw(plot)
    ng.frame(wks)
    del plot
    del wks

def plot_msl(ds_sel, out_path, main_title, second_line, shapefile_path, lat, lon):
    temp = ds_sel["msl"].values.astype(np.float32)
    if np.nanmean(temp) > 10000: temp = temp / 100.0

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        temp = temp[::-1, :].copy()

    wks = ng.open_wks("png", out_path)
    res = ng.Resources()
    res.nglDraw = False
    res.nglFrame = False

    res.cnFillOn = True
    res.cnFillMode = "AreaFill"
    res.cnLinesOn = True
    res.cnLineLabelsOn = False

    res.sfXArray = lon
    res.sfYArray = lat

    res.mpLimitMode = "LatLon"
    res.mpMinLatF = 0.0
    res.mpMaxLatF = 40.0
    res.mpMinLonF = 60.0
    res.mpMaxLonF = 100.0
    res.mpGridAndLimbOn = False

    res.vpXF = 0.15
    res.vpYF = 0.80
    res.vpWidthF = 0.7
    res.vpHeightF = 0.7

    res.cnLevelSelectionMode = "ExplicitLevels"
    res.cnLevels = [998, 1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018]
    res.cnFillColors = [ "#F00082", "#FA3C3C", "#F08228", "#E6AF2D", "#E6DC32", "#A0E632", "#00DC00", "#00D28C", "#00C8C8", "#00A0FF", "#1E3CFF", "#8200DC" ]

    res.lbTitleOn = False
    res.pmLabelBarWidthF = 0.1
    res.pmLabelBarHeightF = 0.6
    res.pmLabelBarOrthogonalPosF = 0.0001
    res.lbLabelFontHeightF = 0.01

    res.tiMainOn = True
    res.tiMainString = f"{main_title}~C~{second_line}"
    res.tiMainFontHeightF = 0.015
    res.tiMainOffsetYF = 0.02

    x_border = [0.01, 0.99, 0.99, 0.01, 0.01]
    y_border = [0.01, 0.01, 0.99, 0.99, 0.01]
    border_res = ng.Resources()
    border_res.gsLineColor = "black"
    border_res.gsLineThicknessF = 3.0
    ng.polyline_ndc(wks, x_border, y_border, border_res)

    plot = ng.contour_map(wks, temp, res)
    overlay_shapefile(wks, plot, shapefile_path)

    ng.draw(plot)
    ng.frame(wks)
    del plot
    del wks

def plot_wind(ds_sel, out_path, main_title, second_line, shapefile_path, lat, lon, inLevel):
    is_surface = (inLevel == 10)
    if is_surface:
        u = ds_sel["u10"].values.astype(np.float32)
        v = ds_sel["v10"].values.astype(np.float32)
    else:
        level_name = "level" if "level" in ds_sel.coords else "isobaricInhPa" if "isobaricInhPa" in ds_sel.coords else "pressure_level"
        ds_lev = ds_sel.sel({level_name: inLevel}, method="nearest")
        u = ds_lev["u"].values.astype(np.float32)
        v = ds_lev["v"].values.astype(np.float32)

    wind_speed = np.sqrt(u**2 + v**2) * 1.94384

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        u = u[::-1, :].copy()
        v = v[::-1, :].copy()
        wind_speed = wind_speed[::-1, :].copy()

    wks = ng.open_wks("png", out_path)
    
    # Map
    map_res = ng.Resources()
    map_res.nglDraw = False
    map_res.nglFrame = False
    map_res.mpLimitMode = "LatLon"
    map_res.mpMinLatF = 0.0
    map_res.mpMaxLatF = 40.0
    map_res.mpMinLonF = 60.0
    map_res.mpMaxLonF = 100.0
    map_res.mpGridAndLimbOn = False
    map_res.vpXF = 0.15
    map_res.vpYF = 0.80
    map_res.vpWidthF = 0.7
    map_res.vpHeightF = 0.7

    map_res.tiMainOn = True
    map_res.tiMainString = f"{main_title}~C~{second_line}"
    map_res.tiMainFontHeightF = 0.015
    map_res.tiMainOffsetYF = 0.02

    x_border = [0.01, 0.99, 0.99, 0.01, 0.01]
    y_border = [0.01, 0.01, 0.99, 0.99, 0.01]
    border_res = ng.Resources()
    border_res.gsLineColor = "black"
    border_res.gsLineThicknessF = 3.0
    ng.polyline_ndc(wks, x_border, y_border, border_res)

    map_plot = ng.map(wks, map_res)

    # Contour Plot
    res = ng.Resources()
    res.nglDraw = False
    res.nglFrame = False
    res.sfXArray = lon
    res.sfYArray = lat
    res.cnFillOn = True
    res.cnFillMode = "AreaFill"
    res.cnLinesOn = False
    res.cnLineLabelsOn = False
    
    res.cnLevelSelectionMode = "ExplicitLevels"
    if inLevel in [400, 300, 250, 200, 150, 100, 50]:
        res.cnLevels = [100, 80, 60, 50, 40]
        res.cnFillColors = ["#FFFFFF", "#00DC00", "#00D28C", "#00C8C8", "#00A0FF", "#1E3CFF"]
    elif inLevel == 500:
        res.cnLevels = [60, 40, 30, 20]
        res.cnFillColors = ["#FFFFFF", "#64F0F0", "#00C8C8", "#00A0FF", "#1E3CFF"]
    else:
        res.cnLevels = [40, 30, 20]
        res.cnFillColors = ["#FFFFFF", "#00C8C8", "#00A0FF", "#1E3CFF"]

    res.lbOrientation = "Vertical"
    res.lbLabelFontHeightF = 0.012
    res.lbTitleString = "Wind Speed (kt)"
    res.lbTitleFontHeightF = 0.014
    res.lbTitleAngleF = 90
    res.lbTitlePosition = "Right"
    res.lbTitleDirection = "Across"
    res.lbTitleOffsetF = 0.3
    res.pmLabelBarDisplayMode = "Always"
    res.pmLabelBarSide = "Right"
    res.pmLabelBarWidthF = 0.10
    res.pmLabelBarHeightF = 0.55
    res.pmLabelBarOrthogonalPosF = -0.05
    res.pmLabelBarParallelPosF = 0.5

    cplot = ng.contour(wks, wind_speed, res)

    # Vector Plot
    vec_res = ng.Resources()
    vec_res.nglDraw = False
    vec_res.nglFrame = False
    vec_res.vcGlyphStyle = "WindBarb"
    vec_res.vcRefLengthF = 0.015
    vec_res.vcRefMagnitudeF = 20.0
    vec_res.vcLineArrowThicknessF = 1.5
    vec_res.vcMinMagnitudeF = 0.5
    vec_res.vcMonoLineArrowColor = False

    skip = 4
    vec_res.vfXArray = lon[::skip]
    vec_res.vfYArray = lat[::skip]
    u_plot = u[::skip, ::skip]
    v_plot = v[::skip, ::skip]
    vplot = ng.vector(wks, u_plot, v_plot, vec_res)

    ng.overlay(map_plot, cplot)
    ng.overlay(map_plot, vplot)
    overlay_shapefile(wks, map_plot, shapefile_path)

    ng.draw(map_plot)
    ng.frame(wks)
    del map_plot
    del cplot
    del vplot
    del wks

def process_dataset(nc_path, valid_date):
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    if "valid_time" in ds.dims: ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.dims: ds = ds.rename({"valid_time": "time"})
         
    if "time" in ds.dims and not "time" in ds.coords:
        ds = ds.assign_coords(time=("time", [np.datetime64(valid_date.strftime('%Y-%m-%dT00:00:00'))]))
        
    ds_sel = ds.sel(time=valid_date, method="nearest")
    ds_sel = _resolve_expver(ds_sel)

    lat = None; lon = None
    for name in ("latitude", "lat", "y"):
        if name in ds_sel.coords: lat = ds_sel[name].values; break
    for name in ("longitude", "lon", "x"):
        if name in ds_sel.coords: lon = ds_sel[name].values; break
        
    return ds, ds_sel, lat, lon

def main():
    parser = argparse.ArgumentParser(description="Unified PyNGL Script: Generate T2M, MSL, and Wind plots natively for April 29.")
    parser.add_argument("--obs_dir", default="data/", help="Path to observed data directory")
    parser.add_argument("--fine_dir", default="output/29april2024/april28", help="Path to finetuned forecast directory")
    parser.add_argument("--orig_dir", default="output/29april2024/april28zhaoshan", help="Path to original forecast directory")
    parser.add_argument("--base_date", default="20240428", help="Base date for the forecast (YYYYMMDD)")
    parser.add_argument("--shapefile", default="/home/administrator/Documents/Ramesh_data/Inference/India_shape/India_State.shp", help="Path to India State shapefile")
    parser.add_argument("--output_dir", default="plots_april29", help="Output directory for all generated plots")
    args = parser.parse_args()

    base_date = datetime.strptime(args.base_date, "%Y%m%d")
    lead_hours = 24
    valid_date = base_date + timedelta(hours=lead_hours)
    
    base_str = base_date.strftime("%d-%m-%Y")
    valid_str = valid_date.strftime("%d-%m-%Y")
    date_str_compact = valid_date.strftime('%Y%m%d')

    models = [
        {"name": "Observed", "dir": args.obs_dir, "mode": "observed"},
        {"name": "finetuned", "dir": args.fine_dir, "mode": "forecast"},
        {"name": "original", "dir": args.orig_dir, "mode": "forecast"}
    ]

    wind_levels = [10, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    os.makedirs(os.path.join(args.output_dir, "t2m"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "msl"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "wind"), exist_ok=True)

    for model in models:
        m_name = model["name"]
        m_dir = model["dir"]
        m_mode = model["mode"]

        print(f"\n{'='*60}\nGenerating native plots for: {m_name}\n{'='*60}")

        # --- PROCESS SURFACE VARIABLES (T2M, MSL, Wind 10m) ---
        surf_path = _find_nc_path(m_dir, m_mode, valid_date, "surface")
        if surf_path:
            print(f"Loading Surface Data: {surf_path}")
            ds_surf, ds_sel_surf, lat, lon = process_dataset(surf_path, valid_date)
            if lat is not None and lon is not None:
                # Plot T2M
                print("  -> Plotting T2M...")
                t2m_title = f"AIML Model ({m_name}) TEMP. (DEG. C) at 2m (t2m) FORECAST ({lead_hours} HR)" if m_mode == "forecast" else f"Observed ({m_name}) TEMP. (DEG. C) at 2m (t2m)"
                t2m_line2 = f"based on 00 UTC of {base_str} valid for 00 UTC of {valid_str}" if m_mode == "forecast" else f"based on 00UTC of {valid_str}"
                t2m_out = os.path.join(args.output_dir, "t2m", f"t2m_{m_name}_{m_mode}_{date_str_compact}_{lead_hours}HR" if m_mode == "forecast" else f"t2m_{m_name}_{m_mode}_{date_str_compact}")
                plot_t2m(ds_sel_surf, t2m_out, t2m_title, t2m_line2, args.shapefile, lat, lon)

                # Plot MSL
                print("  -> Plotting MSL...")
                msl_title = f"AIML Model ({m_name}) MSL Pressure (hPa) FORECAST ({lead_hours} HR)" if m_mode == "forecast" else f"Observed ({m_name}) MSL Pressure (hPa)"
                msl_line2 = f"based on 00 UTC of {base_str} valid for 00 UTC of {valid_str}" if m_mode == "forecast" else f"based on 00 UTC of {valid_str}"
                msl_out = os.path.join(args.output_dir, "msl", f"mslp_{m_name}_{m_mode}_{date_str_compact}_{lead_hours}HR" if m_mode == "forecast" else f"mslp_{m_name}_{m_mode}_{date_str_compact}")
                plot_msl(ds_sel_surf, msl_out, msl_title, msl_line2, args.shapefile, lat, lon)

                # Plot Surface Wind (Level 10)
                if 10 in wind_levels:
                    print("  -> Plotting Surface Wind (10m)...")
                    w_title = f"AIML Model ({m_name}) 10m WIND (kt) FORECAST ({lead_hours} HR)" if m_mode == "forecast" else f"Observed ({m_name}) 10m WIND (kt)"
                    w_out = os.path.join(args.output_dir, "wind", f"wind_10_{m_name}_{m_mode}_{date_str_compact}_{lead_hours}HR" if m_mode == "forecast" else f"wind_10_{m_name}_{m_mode}_{date_str_compact}")
                    plot_wind(ds_sel_surf, w_out, w_title, msl_line2, args.shapefile, lat, lon, 10)
            ds_surf.close()
        else:
            print(f"[WARN] Surface NetCDF for {valid_date} not found in {m_dir}")

        # --- PROCESS UPPER-AIR VARIABLES (Wind at pressure levels) ---
        upper_levels = [l for l in wind_levels if l != 10]
        if upper_levels:
            upper_path = _find_nc_path(m_dir, m_mode, valid_date, "upper")
            # If using 'forecast' mode, surface and upper might be the same file
            if m_mode == "forecast" and not upper_path:
                upper_path = surf_path 
            
            if upper_path:
                print(f"Loading Upper Air Data: {upper_path}")
                ds_up, ds_sel_up, lat, lon = process_dataset(upper_path, valid_date)
                if lat is not None and lon is not None:
                    for lvl in upper_levels:
                        print(f"  -> Plotting Wind at {lvl} hPa...")
                        w_title = f"AIML Model ({m_name}) {lvl} hPa WIND (kt) FORECAST ({lead_hours} HR)" if m_mode == "forecast" else f"Observed ({m_name}) {lvl} hPa WIND (kt)"
                        w_line2 = f"based on 00 UTC of {base_str} valid for 00 UTC of {valid_str}" if m_mode == "forecast" else f"based on 00 UTC of {valid_str}"
                        w_out = os.path.join(args.output_dir, "wind", f"wind_{lvl}_{m_name}_{m_mode}_{date_str_compact}_{lead_hours}HR" if m_mode == "forecast" else f"wind_{lvl}_{m_name}_{m_mode}_{date_str_compact}")
                        plot_wind(ds_sel_up, w_out, w_title, w_line2, args.shapefile, lat, lon, lvl)
                ds_up.close()
            else:
                print(f"[WARN] Upper NetCDF for {valid_date} not found in {m_dir}")

    print(f"\nAll plots successfully generated in the '{args.output_dir}' directory.")

if __name__ == "__main__":
    main()
