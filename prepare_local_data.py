# -*- coding: utf-8 -*-
"""從原始 NODASS 資料庫萃取「專案實際使用」的最小資料集，複製到本機。

用途
----
原始資料位於外接硬碟（F:\\20260907_塑源，共 424.63 GB），
但專案實際只用到其中極小一部分：

  POM 海流：POM/2d/*.nc 的 us / vs（表層東向/北向流速）
  WRF 風場：WRF/**/*_0000.grib2 的 10 公尺 u10 / v10

本腳本將這些欄位萃取、合併、壓縮後寫入專案內的 slim_data/，
使專案不再依賴外接硬碟。

輸出
----
  slim_data/pom_currents_slim.nc   （us, vs, lat, lon, time）
  slim_data/wrf_wind_slim.nc       （u10, v10, latitude, longitude, time）

用法
----
  python prepare_local_data.py                # 自動偵測來源
  python prepare_local_data.py --source F:\\20260907_塑源
  python prepare_local_data.py --force        # 覆寫既有輸出
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import xarray as xr

# ------------------------------------------------------------
# 路徑設定
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "slim_data")

DEFAULT_SOURCE_CANDIDATES = [
    os.environ.get("PLASTIC_DATA_ROOT"),
    r"F:\20260907_塑源",
]

# 專案實際使用的欄位（與 ck.py 的候選清單一致）
POM_U_CANDIDATES = ["us", "u", "water_u", "u_eastward"]
POM_V_CANDIDATES = ["vs", "v", "water_v", "v_northward"]
POM_LAT_CANDIDATES = ["lat", "latitude", "lat_rho", "y"]
POM_LON_CANDIDATES = ["lon", "longitude", "lon_rho", "x"]
POM_TIME_CANDIDATES = ["time", "ocean_time", "valid_time"]

WRF_U_CANDIDATES = ["u10", "10u", "u"]
WRF_V_CANDIDATES = ["v10", "10v", "v"]
WRF_LAT_CANDIDATES = ["latitude", "lat"]
WRF_LON_CANDIDATES = ["longitude", "lon"]
WRF_TIME_CANDIDATES = ["valid_time", "time"]

COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


def _pick(candidates, available):
    lower = {str(k).lower(): str(k) for k in available}
    for cand in candidates:
        if str(cand).lower() in lower:
            return lower[str(cand).lower()]
    return None


def _human(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _detect_engines():
    """偵測可用的 NetCDF 引擎。

    注意：netCDF4 引擎在含中文的路徑（例如 F:\\20260907_塑源）會失敗，
    因此 scipy 引擎必須列為候選，確保中文路徑仍可讀取。
    """
    engines = []
    for mod_name, engine_name in (("netCDF4", "netcdf4"), ("scipy", "scipy"), ("h5netcdf", "h5netcdf")):
        try:
            __import__(mod_name)
            engines.append(engine_name)
        except ImportError:
            continue
    return engines


def _open_nc(path, engines):
    """依序嘗試可用引擎開啟 NetCDF；全部失敗才拋出錯誤。"""
    errors = []
    for engine in [None] + engines:
        try:
            if engine is None:
                return xr.open_dataset(path, decode_times=True)
            return xr.open_dataset(path, engine=engine, decode_times=True)
        except Exception as exc:  # noqa: BLE001 - 需逐一嘗試所有引擎
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"無法開啟 {path}；錯誤：{'; '.join(errors)}")


def resolve_source(explicit=None):
    candidates = [explicit] + DEFAULT_SOURCE_CANDIDATES
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "POM")):
            return path
    raise SystemExit(
        "找不到原始資料根目錄（需包含 POM/ 子目錄）。\n"
        "請以 --source 指定，或設定環境變數 PLASTIC_DATA_ROOT。"
    )


# ------------------------------------------------------------
# POM 海流
# ------------------------------------------------------------
def extract_pom(source_root, engines, force=False):
    out_path = os.path.join(OUT_DIR, "pom_currents_slim.nc")
    if os.path.isfile(out_path) and not force:
        print(f"[POM] 已存在，略過（--force 可覆寫）：{out_path}")
        return out_path

    pom_dir = os.path.join(source_root, "POM")
    pom_2d = os.path.join(pom_dir, "2d")
    search_dir = pom_2d if os.path.isdir(pom_2d) else pom_dir
    files = sorted(glob.glob(os.path.join(search_dir, "**", "*.nc"), recursive=True))
    if not files:
        raise SystemExit(f"[POM] 找不到 .nc 檔案：{search_dir}")

    print(f"[POM] 找到 {len(files)} 個檔案，來源：{search_dir}")
    t0 = time.time()

    datasets = []
    try:
        for idx, path in enumerate(files, 1):
            ds = _open_nc(path, engines)
            names = list(ds.variables)
            u_name = _pick(POM_U_CANDIDATES, names)
            v_name = _pick(POM_V_CANDIDATES, names)
            lat_name = _pick(POM_LAT_CANDIDATES, names)
            lon_name = _pick(POM_LON_CANDIDATES, names)
            time_name = _pick(POM_TIME_CANDIDATES, names)
            if not all([u_name, v_name, lat_name, lon_name, time_name]):
                raise SystemExit(
                    f"[POM] {os.path.basename(path)} 缺少必要欄位；"
                    f"目前欄位={names}"
                )
            # 只保留專案需要的欄位，避免載入 ele/wx/wy/slp/sla 等無用變數。
            keep = [u_name, v_name, lat_name, lon_name, time_name]
            datasets.append(ds[keep])
            if idx % 10 == 0 or idx == len(files):
                print(f"  已載入 {idx}/{len(files)}")

        try:
            merged = xr.combine_by_coords(datasets, combine_attrs="override")
        except Exception:
            merged = xr.concat(
                datasets, dim=time_name, data_vars="minimal",
                coords="minimal", compat="override", join="override",
            )

        merged = merged.sortby(time_name)
        times = np.asarray(merged[time_name].values)
        keep_mask = ~np.asarray(__import__("pandas").Index(times).duplicated(keep="first"))
        merged = merged.isel({time_name: np.flatnonzero(keep_mask)})

        # 統一變數名稱，讓下游（ck.py）能以固定名稱讀取。
        rename = {u_name: "us", v_name: "vs", lat_name: "lat", lon_name: "lon", time_name: "time"}
        merged = merged.rename({k: v for k, v in rename.items() if k != v})

        os.makedirs(OUT_DIR, exist_ok=True)
        encoding = {name: COMPRESSION for name in ("us", "vs")}
        merged.to_netcdf(out_path, encoding=encoding)
        size = os.path.getsize(out_path)
        print(
            f"[POM] 完成：{out_path}\n"
            f"      時間步={merged.sizes['time']}，網格={merged.sizes['lat']}x{merged.sizes['lon']}，"
            f"大小={_human(size)}，耗時={time.time() - t0:.1f}s"
        )
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
    return out_path


# ------------------------------------------------------------
# WRF 風場
# ------------------------------------------------------------
def extract_wrf(source_root, engines, force=False):
    out_path = os.path.join(OUT_DIR, "wrf_wind_slim.nc")
    if os.path.isfile(out_path) and not force:
        print(f"[WRF] 已存在，略過（--force 可覆寫）：{out_path}")
        return out_path

    wrf_dir = os.path.join(source_root, "WRF")
    all_files = [
        f for f in glob.glob(os.path.join(wrf_dir, "**", "*"), recursive=True)
        if os.path.isfile(f)
        and f.lower().endswith((".grib2", ".grib", ".grb", ".gri"))
        and not f.lower().endswith(".idx")
    ]
    if not all_files:
        raise SystemExit(f"[WRF] 找不到 GRIB2 檔案：{wrf_dir}")

    # 每個預報循環的 _0000 檔代表「分析時刻」，構成歷史風場時間軸。
    analysis = [p for p in all_files if os.path.basename(p).lower().endswith("_0000.grib2")]
    if not analysis:
        analysis = all_files
    analysis = sorted(analysis)
    print(f"[WRF] 掃描 {len(all_files)} 個 GRIB2，選用 {len(analysis)} 個分析時刻檔")
    t0 = time.time()

    records = []
    failures = []
    for idx, path in enumerate(analysis, 1):
        try:
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                decode_times=True,
                backend_kwargs={
                    "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
                    "indexpath": "",
                },
            )
            u_name = _pick(WRF_U_CANDIDATES, list(ds.data_vars))
            v_name = _pick(WRF_V_CANDIDATES, list(ds.data_vars))
            lat_name = _pick(WRF_LAT_CANDIDATES, list(ds.variables))
            lon_name = _pick(WRF_LON_CANDIDATES, list(ds.variables))
            time_name = _pick(WRF_TIME_CANDIDATES, list(ds.variables))
            if not all([u_name, v_name, lat_name, lon_name, time_name]):
                raise ValueError(f"缺少 10m U/V 或經緯度；變數={list(ds.variables)}")

            ua = np.asarray(ds[u_name].values, dtype=np.float32)
            va = np.asarray(ds[v_name].values, dtype=np.float32)
            while ua.ndim > 2:
                ua = ua[0]
            while va.ndim > 2:
                va = va[0]
            lat_vals = np.asarray(ds[lat_name].values, dtype=np.float32)
            lon_vals = np.asarray(ds[lon_name].values, dtype=np.float32)
            time_value = np.asarray(ds[time_name].values).reshape(-1)[0]
            records.append((np.datetime64(time_value), lat_vals, lon_vals, ua, va))
            ds.close()
        except Exception as exc:  # noqa: BLE001 - 個別檔案失敗不應中斷整體
            failures.append(f"{os.path.basename(path)}: {exc}")
        if idx % 20 == 0 or idx == len(analysis):
            print(f"  已處理 {idx}/{len(analysis)}")

    if not records:
        raise SystemExit("[WRF] 無任何可用檔案；代表性錯誤：" + "；".join(failures[:5]))

    records.sort(key=lambda item: item[0])
    seen = set()
    unique = []
    for item in records:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])

    times = np.array([item[0] for item in unique], dtype="datetime64[ns]")
    u_stack = np.stack([item[3] for item in unique]).astype(np.float32)
    v_stack = np.stack([item[4] for item in unique]).astype(np.float32)
    lat_grid = unique[-1][1]
    lon_grid = unique[-1][2]

    ds_out = xr.Dataset(
        data_vars={
            "u10": (("time", "y", "x"), u_stack),
            "v10": (("time", "y", "x"), v_stack),
            "latitude": (("y", "x"), lat_grid),
            "longitude": (("y", "x"), lon_grid),
        },
        coords={"time": times},
        attrs={"source": "WRF SFC _0000 analysis, 10m heightAboveGround"},
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    encoding = {name: COMPRESSION for name in ("u10", "v10", "latitude", "longitude")}
    ds_out.to_netcdf(out_path, encoding=encoding)
    size = os.path.getsize(out_path)
    print(
        f"[WRF] 完成：{out_path}\n"
        f"      時間步={ds_out.sizes['time']}，網格={ds_out.sizes['y']}x{ds_out.sizes['x']}，"
        f"大小={_human(size)}，耗時={time.time() - t0:.1f}s"
    )
    if failures:
        print(f"[WRF] 注意：{len(failures)} 個檔案讀取失敗（已略過）")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="萃取專案所需的最小資料集到本機")
    parser.add_argument("--source", default=None, help="原始資料根目錄（含 POM/ 與 WRF/）")
    parser.add_argument("--force", action="store_true", help="覆寫既有輸出檔案")
    parser.add_argument("--skip-pom", action="store_true", help="略過 POM 萃取")
    parser.add_argument("--skip-wrf", action="store_true", help="略過 WRF 萃取")
    args = parser.parse_args()

    source_root = resolve_source(args.source)
    engines = _detect_engines()
    print("=" * 70)
    print(f"來源：{source_root}")
    print(f"輸出：{OUT_DIR}")
    print(f"可用 NetCDF 引擎：{engines or ['(無)']}")
    print("=" * 70)

    if not args.skip_pom:
        extract_pom(source_root, engines, force=args.force)
    if not args.skip_wrf:
        extract_wrf(source_root, engines, force=args.force)

    print("=" * 70)
    total = 0
    for name in ("pom_currents_slim.nc", "wrf_wind_slim.nc"):
        path = os.path.join(OUT_DIR, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            total += size
            print(f"  {name}: {_human(size)}")
    print(f"  總計: {_human(total)}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
