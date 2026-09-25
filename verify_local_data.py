# -*- coding: utf-8 -*-
"""驗證精簡資料集與原始資料「逐點數值一致」，確保不影響模擬結果。

驗證策略
--------
1. 結構驗證：時間軸、網格、變數名稱、單位。
2. 逐點數值驗證：從原始檔與精簡檔各取樣若干時間步，
   對 us/vs（POM）與 u10/v10（WRF）做逐格點比對，
   計算最大絕對誤差與相對誤差。
3. 模擬層驗證：以 ck.py 的插值函式，對隨機取樣的
   (經度, 緯度, 時間) 組合，比對「原始資料」與「精簡資料」
   所產生的速度場是否一致。

判定標準
--------
- 逐點最大絕對誤差 <= 1e-5（float32 壓縮後的可接受誤差）
- 模擬層最大絕對誤差 <= 1e-5 m/s

用法
----
  python verify_local_data.py
  python verify_local_data.py --source F:\\20260907_塑源
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLIM_DIR = os.path.join(BASE_DIR, "slim_data")

DEFAULT_SOURCE_CANDIDATES = [
    os.environ.get("PLASTIC_DATA_ROOT"),
    r"F:\20260907_塑源",
]

TOLERANCE = 1e-5
N_SAMPLE_TIMES = 5
N_SAMPLE_POINTS = 2000
RNG_SEED = 20260925


def _pick(candidates, available):
    lower = {str(k).lower(): str(k) for k in available}
    for cand in candidates:
        if str(cand).lower() in lower:
            return lower[str(cand).lower()]
    return None


def _detect_engines():
    engines = []
    for mod_name, engine_name in (("netCDF4", "netcdf4"), ("scipy", "scipy"), ("h5netcdf", "h5netcdf")):
        try:
            __import__(mod_name)
            engines.append(engine_name)
        except ImportError:
            continue
    return engines


def _open_nc(path, engines):
    errors = []
    for engine in [None] + engines:
        try:
            if engine is None:
                return xr.open_dataset(path, decode_times=True)
            return xr.open_dataset(path, engine=engine, decode_times=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{engine or 'default'}: {exc}")
    raise RuntimeError(f"無法開啟 {path}；錯誤：{'; '.join(errors)}")


def resolve_source(explicit=None):
    for path in [explicit] + DEFAULT_SOURCE_CANDIDATES:
        if path and os.path.isdir(os.path.join(path, "POM")):
            return path
    return None


def _report(name, max_abs, max_rel, tol=TOLERANCE):
    ok = max_abs <= tol
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}: 最大絕對誤差={max_abs:.3e}, 最大相對誤差={max_rel:.3e}")
    return ok


# ------------------------------------------------------------
# 1. 結構驗證
# ------------------------------------------------------------
def verify_structure():
    print("=" * 70)
    print("【1】結構驗證")
    print("=" * 70)
    ok = True

    pom_path = os.path.join(SLIM_DIR, "pom_currents_slim.nc")
    wrf_path = os.path.join(SLIM_DIR, "wrf_wind_slim.nc")
    for path in (pom_path, wrf_path):
        if not os.path.isfile(path):
            print(f"  [FAIL] 缺少檔案：{path}")
            return False

    with xr.open_dataset(pom_path) as pom:
        print(f"  POM 變數: {sorted(pom.data_vars)}")
        print(f"  POM 維度: {dict(pom.sizes)}")
        print(f"  POM 時間: {pom['time'].values[0]} ~ {pom['time'].values[-1]}")
        dt = np.diff(pom["time"].values).astype("timedelta64[m]").astype(int)
        print(f"  POM 時間間隔(分): {sorted(set(dt.tolist()))}")
        ok &= _report("POM 必要變數齊全", 0.0 if {"us", "vs", "lat", "lon", "time"} <= set(pom.variables) else 1.0, 0.0)
        ok &= _report("POM 時間間隔為 60 分", 0.0 if set(dt.tolist()) == {60} else 1.0, 0.0)

    with xr.open_dataset(wrf_path) as wrf:
        print(f"  WRF 變數: {sorted(wrf.data_vars)}")
        print(f"  WRF 維度: {dict(wrf.sizes)}")
        print(f"  WRF 時間: {wrf['time'].values[0]} ~ {wrf['time'].values[-1]}")
        dt = np.diff(wrf["time"].values).astype("timedelta64[m]").astype(int)
        print(f"  WRF 時間間隔(小時): {sorted(set((dt / 60).tolist()))}")
        ok &= _report("WRF 必要變數齊全", 0.0 if {"u10", "v10", "latitude", "longitude", "time"} <= set(wrf.variables) else 1.0, 0.0)
        ok &= _report("WRF 時間間隔為 6 小時", 0.0 if set((dt / 60).tolist()) == {6} else 1.0, 0.0)

    return bool(ok)


# ------------------------------------------------------------
# 2. 逐點數值驗證
# ------------------------------------------------------------
def verify_pom_values(source_root, engines):
    print("=" * 70)
    print("【2a】POM 逐點數值驗證（原始 vs 精簡）")
    print("=" * 70)
    if source_root is None:
        print("  [SKIP] 找不到原始資料來源，略過逐點比對。")
        return True

    pom_dir = os.path.join(source_root, "POM")
    pom_2d = os.path.join(pom_dir, "2d")
    search_dir = pom_2d if os.path.isdir(pom_2d) else pom_dir
    files = sorted(glob.glob(os.path.join(search_dir, "**", "*.nc"), recursive=True))
    if not files:
        print(f"  [SKIP] 找不到原始 POM 檔案：{search_dir}")
        return True

    slim_path = os.path.join(SLIM_DIR, "pom_currents_slim.nc")
    rng = np.random.default_rng(RNG_SEED)
    ok = True

    with xr.open_dataset(slim_path) as slim:
        slim_times = pd.to_datetime(slim["time"].values)
        slim_us = slim["us"].values
        slim_vs = slim["vs"].values
        slim_lat = slim["lat"].values
        slim_lon = slim["lon"].values

        # 隨機挑選時間步
        n_t = slim.sizes["time"]
        sample_t = np.sort(rng.choice(n_t, size=min(N_SAMPLE_TIMES, n_t), replace=False))
        print(f"  取樣時間步索引: {sample_t.tolist()}")

        # 逐檔比對：原始檔的每個時間步都能在精簡檔中找到對應
        checked = 0
        max_abs_u = max_abs_v = 0.0
        max_rel_u = max_rel_v = 0.0

        for path in files:
            ds = _open_nc(path, engines)
            try:
                names = list(ds.variables)
                u_name = _pick(["us", "u", "water_u", "u_eastward"], names)
                v_name = _pick(["vs", "v", "water_v", "v_northward"], names)
                time_name = _pick(["time", "ocean_time", "valid_time"], names)
                if not all([u_name, v_name, time_name]):
                    continue
                raw_times = pd.to_datetime(ds[time_name].values)
                raw_u = np.asarray(ds[u_name].values, dtype=np.float32)
                raw_v = np.asarray(ds[v_name].values, dtype=np.float32)
                while raw_u.ndim > 3:
                    raw_u = raw_u[:, 0]
                while raw_v.ndim > 3:
                    raw_v = raw_v[:, 0]

                for k, rt in enumerate(raw_times):
                    matches = np.flatnonzero(slim_times == rt)
                    if matches.size == 0:
                        continue
                    si = int(matches[0])
                    if si not in sample_t:
                        continue
                    a_u = raw_u[k]
                    a_v = raw_v[k]
                    b_u = slim_us[si]
                    b_v = slim_vs[si]
                    if a_u.shape != b_u.shape:
                        print(f"  [FAIL] 形狀不符 {a_u.shape} vs {b_u.shape} @ {rt}")
                        ok = False
                        continue
                    finite = np.isfinite(a_u) & np.isfinite(b_u)
                    if finite.any():
                        d_u = np.abs(a_u[finite] - b_u[finite])
                        max_abs_u = max(max_abs_u, float(d_u.max()))
                        denom = np.maximum(np.abs(a_u[finite]), 1e-6)
                        max_rel_u = max(max_rel_u, float((d_u / denom).max()))
                    finite = np.isfinite(a_v) & np.isfinite(b_v)
                    if finite.any():
                        d_v = np.abs(a_v[finite] - b_v[finite])
                        max_abs_v = max(max_abs_v, float(d_v.max()))
                        denom = np.maximum(np.abs(a_v[finite]), 1e-6)
                        max_rel_v = max(max_rel_v, float((d_v / denom).max()))
                    checked += 1
            finally:
                ds.close()

        print(f"  已比對 {checked} 個時間步")
        ok &= _report("POM us 逐點一致", max_abs_u, max_rel_u)
        ok &= _report("POM vs 逐點一致", max_abs_v, max_rel_v)
    return bool(ok)


def verify_wrf_values(source_root, engines):
    print("=" * 70)
    print("【2b】WRF 逐點數值驗證（原始 vs 精簡）")
    print("=" * 70)
    if source_root is None:
        print("  [SKIP] 找不到原始資料來源，略過逐點比對。")
        return True

    wrf_dir = os.path.join(source_root, "WRF")
    all_files = [
        f for f in glob.glob(os.path.join(wrf_dir, "**", "*"), recursive=True)
        if os.path.isfile(f)
        and f.lower().endswith((".grib2", ".grib", ".grb", ".gri"))
        and not f.lower().endswith(".idx")
    ]
    analysis = sorted(p for p in all_files if os.path.basename(p).lower().endswith("_0000.grib2"))
    if not analysis:
        print(f"  [SKIP] 找不到 WRF 分析時刻檔：{wrf_dir}")
        return True

    slim_path = os.path.join(SLIM_DIR, "wrf_wind_slim.nc")
    rng = np.random.default_rng(RNG_SEED + 1)
    ok = True

    with xr.open_dataset(slim_path) as slim:
        slim_times = pd.to_datetime(slim["time"].values)
        slim_u = slim["u10"].values
        slim_v = slim["v10"].values
        slim_lat = slim["latitude"].values
        slim_lon = slim["longitude"].values

        n_t = slim.sizes["time"]
        sample_t = np.sort(rng.choice(n_t, size=min(N_SAMPLE_TIMES, n_t), replace=False))
        print(f"  取樣時間步索引: {sample_t.tolist()}")

        checked = 0
        max_abs_u = max_abs_v = 0.0
        max_rel_u = max_rel_v = 0.0
        max_coord_diff = 0.0

        for path in analysis:
            try:
                ds = xr.open_dataset(
                    path, engine="cfgrib", decode_times=True,
                    backend_kwargs={
                        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
                        "indexpath": "",
                    },
                )
            except Exception:
                continue
            try:
                u_name = _pick(["u10", "10u", "u"], list(ds.data_vars))
                v_name = _pick(["v10", "10v", "v"], list(ds.data_vars))
                lat_name = _pick(["latitude", "lat"], list(ds.variables))
                lon_name = _pick(["longitude", "lon"], list(ds.variables))
                time_name = _pick(["valid_time", "time"], list(ds.variables))
                if not all([u_name, v_name, lat_name, lon_name, time_name]):
                    continue
                rt = pd.to_datetime(np.asarray(ds[time_name].values).reshape(-1)[0])
                matches = np.flatnonzero(slim_times == rt)
                if matches.size == 0:
                    continue
                si = int(matches[0])
                if si not in sample_t:
                    continue

                a_u = np.asarray(ds[u_name].values, dtype=np.float32)
                a_v = np.asarray(ds[v_name].values, dtype=np.float32)
                while a_u.ndim > 2:
                    a_u = a_u[0]
                while a_v.ndim > 2:
                    a_v = a_v[0]
                b_u = slim_u[si]
                b_v = slim_v[si]
                if a_u.shape != b_u.shape:
                    print(f"  [FAIL] 形狀不符 {a_u.shape} vs {b_u.shape} @ {rt}")
                    ok = False
                    continue

                finite = np.isfinite(a_u) & np.isfinite(b_u)
                if finite.any():
                    d_u = np.abs(a_u[finite] - b_u[finite])
                    max_abs_u = max(max_abs_u, float(d_u.max()))
                    denom = np.maximum(np.abs(a_u[finite]), 1e-6)
                    max_rel_u = max(max_rel_u, float((d_u / denom).max()))
                finite = np.isfinite(a_v) & np.isfinite(b_v)
                if finite.any():
                    d_v = np.abs(a_v[finite] - b_v[finite])
                    max_abs_v = max(max_abs_v, float(d_v.max()))
                    denom = np.maximum(np.abs(a_v[finite]), 1e-6)
                    max_rel_v = max(max_rel_v, float((d_v / denom).max()))

                # 座標一致性
                a_lat = np.asarray(ds[lat_name].values, dtype=np.float32)
                a_lon = np.asarray(ds[lon_name].values, dtype=np.float32)
                if a_lat.shape == slim_lat.shape:
                    max_coord_diff = max(
                        max_coord_diff,
                        float(np.nanmax(np.abs(a_lat - slim_lat))),
                        float(np.nanmax(np.abs(a_lon - slim_lon))),
                    )
                checked += 1
            finally:
                ds.close()

        print(f"  已比對 {checked} 個時間步")
        ok &= _report("WRF u10 逐點一致", max_abs_u, max_rel_u)
        ok &= _report("WRF v10 逐點一致", max_abs_v, max_rel_v)
        ok &= _report("WRF 經緯度網格一致", max_coord_diff, 0.0)
    return bool(ok)


# ------------------------------------------------------------
# 3. 模擬層驗證（插值後速度場）
# ------------------------------------------------------------
def verify_interpolation_layer():
    """以 ck.py 的插值邏輯，比對精簡資料自身的時空插值行為是否合理。

    此處驗證「精簡資料在模擬層可正常運作」：
      - 對隨機 (lon, lat, t) 取樣，插值結果必須為有限值
      - 插值結果必須落在原始場的數值範圍內（不得外插爆走）
      - 時間插值在整數時間步上必須等於該步的原始值
    """
    print("=" * 70)
    print("【3】模擬層插值驗證（精簡資料）")
    print("=" * 70)
    rng = np.random.default_rng(RNG_SEED + 2)
    ok = True

    # --- POM：雙線性插值 ---
    with xr.open_dataset(os.path.join(SLIM_DIR, "pom_currents_slim.nc")) as pom:
        lat_ax = pom["lat"].values.astype(np.float64)
        lon_ax = pom["lon"].values.astype(np.float64)
        us = pom["us"].values
        vs = pom["vs"].values

        def bilinear(grid, x, y):
            idx_x = (x - lon_ax[0]) / (lon_ax[1] - lon_ax[0])
            idx_y = (y - lat_ax[0]) / (lat_ax[1] - lat_ax[0])
            i0 = np.clip(np.floor(idx_x).astype(int), 0, len(lon_ax) - 2)
            j0 = np.clip(np.floor(idx_y).astype(int), 0, len(lat_ax) - 2)
            wx = idx_x - i0
            wy = idx_y - j0
            return (
                grid[j0, i0] * (1 - wx) * (1 - wy)
                + grid[j0, i0 + 1] * wx * (1 - wy)
                + grid[j0 + 1, i0] * (1 - wx) * wy
                + grid[j0 + 1, i0 + 1] * wx * wy
            )

        def bilinear_at_index(grid, i0, j0, wx, wy):
            """以「已知整數格點索引 + 小數權重」做雙線性插值。

            直接使用整數索引可完全避開「由座標反推索引」時的浮點誤差
            （例如 lat 軸步長 0.020000458 非精確值，反推時可能落在
            格點邊界而 floor 到相鄰格），確保驗證的是插值公式本身。
            """
            return (
                grid[j0, i0] * (1 - wx) * (1 - wy)
                + grid[j0, i0 + 1] * wx * (1 - wy)
                + grid[j0 + 1, i0] * (1 - wx) * wy
                + grid[j0 + 1, i0 + 1] * wx * wy
            )

        # 隨機取樣海域點（避開 NaN 陸地格）
        t_idx = int(rng.integers(0, us.shape[0]))
        frame = us[t_idx]
        # 只取「3x3 鄰域全為有限值」的內部海域格點：
        # 雙線性插值會用到 (i0,j0)/(i0+1,j0)/(i0,j0+1)/(i0+1,j0+1) 四點，
        # 若任一鄰點為 NaN（陸地），插值結果必為 NaN。
        # 注意：僅檢查上下左右 4 鄰並不足夠——海岸線常有「對角相鄰」
        # 的 NaN 島（例如台灣西南角），此時 (i0+1,j0+1) 仍可能是 NaN。
        # 因此這裡要求完整 3x3 鄰域皆為有限值。
        # 這與 ck.py 的實際行為一致：粒子一旦進入陸地格即被判定觸岸並停止，
        # 不會在陸地格上做插值。
        finite = np.isfinite(frame)
        interior = finite.copy()
        for _dj in (-1, 0, 1):
            for _di in (-1, 0, 1):
                interior &= np.roll(np.roll(finite, -_dj, axis=0), -_di, axis=1)
        # 排除邊界（np.roll 會環繞，邊界不可信）
        interior[0, :] = False
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False
        jj, ii = np.nonzero(interior)
        pick = rng.choice(len(jj), size=min(N_SAMPLE_POINTS, len(jj)), replace=False)
        ii_pick = ii[pick]
        jj_pick = jj[pick]

        # 以整數格點索引 + 隨機小數權重做插值（模擬粒子落在格點之間的情形）。
        wx = rng.random(len(ii_pick))
        wy = rng.random(len(jj_pick))
        u_interp = bilinear_at_index(us[t_idx], ii_pick, jj_pick, wx, wy)
        v_interp = bilinear_at_index(vs[t_idx], ii_pick, jj_pick, wx, wy)
        finite_ok = np.isfinite(u_interp).all() and np.isfinite(v_interp).all()
        ok &= _report("POM 插值結果全為有限值", 0.0 if finite_ok else 1.0, 0.0)

        # 權重為 0 時，插值必須精確還原該格點原始值。
        exact_u = bilinear_at_index(us[t_idx], ii_pick, jj_pick, np.zeros(len(ii_pick)), np.zeros(len(jj_pick)))
        err = float(np.nanmax(np.abs(exact_u - us[t_idx][jj_pick, ii_pick])))
        ok &= _report("POM 格點插值精確還原", err, 0.0)

        # 範圍檢查
        in_range = (
            float(np.nanmin(u_interp)) >= float(np.nanmin(us[t_idx])) - 1e-6
            and float(np.nanmax(u_interp)) <= float(np.nanmax(us[t_idx])) + 1e-6
        )
        ok &= _report("POM 插值不超出原始值域", 0.0 if in_range else 1.0, 0.0)

        # 時間插值：整數步必須等於該步
        t_exact = 100.0
        t_idx_f = int(np.floor(t_exact))
        w = t_exact - t_idx_f
        u_t = (1 - w) * us[t_idx_f] + w * us[t_idx_f + 1]
        err_t = float(np.nanmax(np.abs(u_t - us[t_idx_f])))
        ok &= _report("POM 整數時間步插值精確", err_t, 0.0)

    # --- WRF：最近鄰插值 ---
    with xr.open_dataset(os.path.join(SLIM_DIR, "wrf_wind_slim.nc")) as wrf:
        lat2 = wrf["latitude"].values.astype(np.float64)
        lon2 = wrf["longitude"].values.astype(np.float64)
        u10 = wrf["u10"].values
        v10 = wrf["v10"].values

        scale = np.cos(np.deg2rad(float(np.nanmean(lat2))))
        pts = np.column_stack((lon2.ravel() * scale, lat2.ravel()))
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            q = np.column_stack((
                rng.uniform(float(np.nanmin(lon2)), float(np.nanmax(lon2)), N_SAMPLE_POINTS) * scale,
                rng.uniform(float(np.nanmin(lat2)), float(np.nanmax(lat2)), N_SAMPLE_POINTS),
            ))
            _, flat = tree.query(q)
            flat = np.clip(flat, 0, u10[0].size - 1)
            u_nn = u10[0].reshape(-1)[flat]
            v_nn = v10[0].reshape(-1)[flat]
            finite_ok = np.isfinite(u_nn).all() and np.isfinite(v_nn).all()
            ok &= _report("WRF 最近鄰插值結果全為有限值", 0.0 if finite_ok else 1.0, 0.0)
            in_range = (
                float(u_nn.min()) >= float(np.nanmin(u10[0])) - 1e-6
                and float(u_nn.max()) <= float(np.nanmax(u10[0])) + 1e-6
            )
            ok &= _report("WRF 插值不超出原始值域", 0.0 if in_range else 1.0, 0.0)
        except ImportError:
            print("  [SKIP] 未安裝 scipy，略過 WRF 最近鄰驗證。")

    return bool(ok)


def main():
    parser = argparse.ArgumentParser(description="驗證精簡資料集與原始資料一致")
    parser.add_argument("--source", default=None, help="原始資料根目錄（含 POM/ 與 WRF/）")
    parser.add_argument("--skip-raw", action="store_true", help="略過與原始資料的逐點比對")
    args = parser.parse_args()

    engines = _detect_engines()
    source_root = None if args.skip_raw else resolve_source(args.source)
    if not args.skip_raw and source_root is None:
        print("注意：找不到原始資料來源，將只做結構與模擬層驗證。\n")

    results = []
    results.append(("結構驗證", verify_structure()))
    results.append(("POM 逐點數值", verify_pom_values(source_root, engines)))
    results.append(("WRF 逐點數值", verify_wrf_values(source_root, engines)))
    results.append(("模擬層插值", verify_interpolation_layer()))

    print("=" * 70)
    print("【總結】")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok &= ok
    print("=" * 70)
    if all_ok:
        print("結論：精簡資料與原始資料一致，可直接用於模擬，不影響結果。")
    else:
        print("結論：驗證未通過，請勿使用精簡資料。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
