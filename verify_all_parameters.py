# -*- coding: utf-8 -*-
"""全參數驗證腳本 (verify_all_parameters.py)

目的
----
用「可證偽」的測試，逐層驗證海洋塑膠溯源模擬的每一個環節，
確認：(1) 軌跡正確性；(2) 資料是否真正被使用。

驗證層次（由底層到頂層）
------------------------
  L1 資料層   ：資料真的被讀進來、且真的被用到（擾動測試）
  L2 索引層   ：時間索引、空間索引對應正確
  L3 插值層   ：雙線性/時間插值數學正確
  L4 積分層   ：RK4 單步正確、方向正確、收斂
  L5 軌跡層   ：整條軌跡物理合理（位移單調、無穿陸、可重現）

資料來源
--------
優先使用 slim_data/（本機精簡資料）；若不存在則回退 mock_pom_taiwan.nc。

用法
----
  $env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
  E:\\c\\conda\\python.exe verify_all_parameters.py
"""
import os
import sys
import types
import copy
import numpy as np
import pandas as pd

# ============================================================
# 載入 ck.py 模組層級（stub streamlit，exec 到 sidebar 之前）
# ============================================================
stub = types.ModuleType("streamlit")
def _noop(*a, **k):
    return None
for _n in ["set_page_config", "title", "caption", "markdown", "write", "info",
           "warning", "error", "success", "stop", "spinner", "header", "subheader",
           "checkbox", "selectbox", "slider", "number_input", "date_input", "button",
           "columns", "container", "expander", "dataframe", "plotly_chart", "metric",
           "text_input", "radio", "multiselect", "tabs", "empty", "progress"]:
    setattr(stub, _n, _noop)
stub.sidebar = stub
stub.session_state = {}
stub.cache_data = lambda *a, **k: (lambda f: f)
stub.cache_resource = lambda *a, **k: (lambda f: f)
sys.modules["streamlit"] = stub

src = open("ck.py", encoding="utf-8").read()
cut = src.index("st.sidebar.header")
ns = {"__name__": "ck_under_test"}
exec(compile(src[:cut], "ck.py", "exec"), ns)

# 取出待驗證的函式與常數
interpolate_2d_vectorized = ns["interpolate_2d_vectorized"]
get_velocity_interpolated = ns["get_velocity_interpolated"]
rk4_step_2d = ns["rk4_step_2d"]
simulate_particles = ns["simulate_particles"]
is_land_vectorized = ns["is_land_vectorized"]
build_public_coastline_mask = ns["build_public_coastline_mask"]
HOTSPOT_CENTERS = ns["HOTSPOT_CENTERS"]
METERS_PER_DEG_LAT = ns["METERS_PER_DEG_LAT"]
CURRENT_SPEED_CAP_MPS = ns["CURRENT_SPEED_CAP_MPS"]
STATUS_BEACHED = ns["STATUS_BEACHED"]
STATUS_ACTIVE = ns["STATUS_ACTIVE"]

# ============================================================
# 測試框架
# ============================================================
_results = []

def check(group, name, passed, detail=""):
    _results.append((group, name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

# ============================================================
# 載入資料（優先 slim_data，其次 mock）
# ============================================================
section("載入資料")

SLIM_POM = os.path.join("slim_data", "pom_currents_slim.nc")
MOCK_POM = "mock_pom_taiwan.nc"

import xarray as xr

if os.path.isfile(SLIM_POM):
    data_source = "slim_data"
    ds = xr.open_dataset(SLIM_POM)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    u = np.asarray(ds["us"].values, dtype=np.float32)
    v = np.asarray(ds["vs"].values, dtype=np.float32)
    if u.ndim == 4:
        u = u[:, 0]
        v = v[:, 0]
    time_vals = pd.to_datetime(np.asarray(ds["time"].values))
    dt_hours = float(np.median(np.diff(time_vals.astype("int64"))) / 3.6e12) if len(time_vals) > 1 else 1.0
    print(f"  使用 slim_data: {SLIM_POM}")
elif os.path.isfile(MOCK_POM):
    data_source = "mock"
    ds = xr.open_dataset(MOCK_POM)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    u = np.asarray(ds["u"].values, dtype=np.float32)
    v = np.asarray(ds["v"].values, dtype=np.float32)
    if u.ndim == 4:
        u = u[:, 0]
        v = v[:, 0]
    time_vals = pd.to_datetime(np.asarray(ds["time"].values))
    dt_hours = float(np.median(np.diff(time_vals.astype("int64"))) / 3.6e12) if len(time_vals) > 1 else 1.0
    print(f"  使用 mock: {MOCK_POM}")
else:
    print("  ❌ 找不到任何資料檔（slim_data/ 或 mock_pom_taiwan.nc）")
    sys.exit(1)

print(f"  POM 網格: {u.shape}, lat[{lat.min():.2f},{lat.max():.2f}] lon[{lon.min():.2f},{lon.max():.2f}]")
print(f"  時間步: {len(time_vals)}, dt={dt_hours:.2f}h")

mask, geometry = build_public_coastline_mask(lat, lon, return_geometry=True)

# 選一個「確定在海洋」的測試點（流速最大處），避免測到陸地 NaN
_spd0 = np.hypot(np.nan_to_num(u[0]), np.nan_to_num(v[0]))
_jj, _ii = np.unravel_index(np.nanargmax(_spd0), _spd0.shape)
if lat.ndim == 1 and lon.ndim == 1:
    OCEAN_LON = float(lon[_ii])
    OCEAN_LAT = float(lat[_jj])
else:
    OCEAN_LON = float(lon[_jj, _ii])
    OCEAN_LAT = float(lat[_jj, _ii])
print(f"  海洋測試點: lon={OCEAN_LON:.3f}, lat={OCEAN_LAT:.3f}")

prov = {
    "curr": {
        "lat": lat, "lon": lon,
        "u": np.nan_to_num(u), "v": np.nan_to_num(v),
        "landmask": mask, "land_geometry": geometry,
        "bbox": (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())),
        "n_times": u.shape[0], "dt_hours": dt_hours,
        "time_start_utc": time_vals[0].to_pydatetime(),
        "time_end_utc": time_vals[-1].to_pydatetime(),
    },
    "wind": None,
    "reference_time_utc": time_vals[-1].to_pydatetime(),
}

# ============================================================
# L1 資料層
# ============================================================
section("L1 資料層：資料載入真實性")

check("L1", "POM 檔案成功載入", u.size > 0, f"shape={u.shape}")
check("L1", "u/v 形狀一致", u.shape == v.shape, f"{u.shape} vs {v.shape}")
finite_frac = float(np.isfinite(u).mean())
check("L1", "u 有效率 > 80%（陸地為 NaN 屬正常）", finite_frac > 0.80, f"{finite_frac*100:.2f}%")
check("L1", "u 非全零（std > 0）", float(np.nanstd(u)) > 0, f"std={float(np.nanstd(u)):.4f}")
max_spd = float(np.nanmax(np.hypot(u, v)))
check("L1", "最大流速合理 (< 5 m/s)", max_spd < 5.0, f"{max_spd:.3f} m/s")
check("L1", "無填充值殘留 (< 1e20)", float(np.nanmax(np.abs(u))) < 1e20, f"max|u|={float(np.nanmax(np.abs(u))):.3e}")
check("L1", "landmask 形狀 == u.shape[1:]", mask.shape == u.shape[1:], f"{mask.shape} vs {u.shape[1:]}")
check("L1", "land_geometry 非 None（高精度路徑）", geometry is not None,
      "Cartopy 幾何可用" if geometry is not None else "退回網格遮罩")
check("L1", "時間軸遞增", bool(np.all(np.diff(time_vals.astype("int64")) > 0)), f"{len(time_vals)} 步")

# ============================================================
# L1 擾動測試：資料是否真的被使用
# ============================================================
section("L1 擾動測試：資料是否真的被使用")

def run_short_sim(prov_local, n_particles=20, n_steps=24, dt_mins=60.0):
    """跑一次短模擬，回傳所有粒子的最終位置陣列（用於比較）。"""
    sites = [(name, *HOTSPOT_CENTERS[name]) for name in list(HOTSPOT_CENTERS.keys())[:2]]
    df, hl, hla, _, _ = simulate_particles(
        release_sites=sites, total_particles=n_particles, total_steps=n_steps,
        dt_mins=dt_mins, prov=prov_local, windage=0.0,
        reference_time_utc=prov_local["reference_time_utc"],
    )
    n = df["particle_id"].nunique()
    out = np.full((n, 2), np.nan)
    for pid in range(n):
        for i in range(len(hl) - 1, -1, -1):
            lo, la = hl[i][pid], hla[i][pid]
            if np.isfinite(lo) and np.isfinite(la):
                out[pid] = [lo, la]
                break
    return out

def clone_prov(prov_base, u_new=None, v_new=None, lat_new=None, lon_new=None):
    p = copy.deepcopy(prov_base)
    if u_new is not None:
        p["curr"]["u"] = u_new
    if v_new is not None:
        p["curr"]["v"] = v_new
    if lat_new is not None:
        p["curr"]["lat"] = lat_new
    if lon_new is not None:
        p["curr"]["lon"] = lon_new
    return p

base_pos = run_short_sim(prov)
check("L1-擾動", "基準模擬產生有效位置", np.isfinite(base_pos).any(),
      f"{np.isfinite(base_pos).all(axis=1).sum()} 顆有效")

pos_u2 = run_short_sim(clone_prov(prov, u_new=prov["curr"]["u"] * 2.0))
diff_u = np.nanmax(np.abs(pos_u2 - base_pos))
check("L1-擾動", "u ×2 → 軌跡改變（u 有被使用）", diff_u > 1e-6, f"最大差異={diff_u:.6f}°")

pos_v2 = run_short_sim(clone_prov(prov, v_new=prov["curr"]["v"] * 2.0))
diff_v = np.nanmax(np.abs(pos_v2 - base_pos))
check("L1-擾動", "v ×2 → 軌跡改變（v 有被使用）", diff_v > 1e-6, f"最大差異={diff_v:.6f}°")

pos_u0 = run_short_sim(clone_prov(prov, u_new=np.zeros_like(prov["curr"]["u"])))
diff_u0 = np.nanmax(np.abs(pos_u0 - base_pos))
check("L1-擾動", "u=0 → 軌跡改變（u 有被使用）", diff_u0 > 1e-6, f"最大差異={diff_u0:.6f}°")

pos_latflip = run_short_sim(clone_prov(prov, lat_new=prov["curr"]["lat"][::-1].copy()))
diff_lat = np.nanmax(np.abs(pos_latflip - base_pos))
check("L1-擾動", "lat 反轉 → 軌跡改變（緯度方向有被使用）", diff_lat > 1e-6, f"最大差異={diff_lat:.6f}°")

pos_lonflip = run_short_sim(clone_prov(prov, lon_new=prov["curr"]["lon"][::-1].copy()))
diff_lon = np.nanmax(np.abs(pos_lonflip - base_pos))
check("L1-擾動", "lon 反轉 → 軌跡改變（經度方向有被使用）", diff_lon > 1e-6, f"最大差異={diff_lon:.6f}°")

# ============================================================
# L2 時間索引
# ============================================================
section("L2 時間索引正確性")

field = prov["curr"]
n_times = field["n_times"]
dt_h = field["dt_hours"]

_QLON = np.array([OCEAN_LON]); _QLAT = np.array([OCEAN_LAT])
u0, v0 = get_velocity_interpolated(_QLON, _QLAT, field, 0.0, 0.0)
check("L2", "elapsed=0 回傳有效速度", u0 is not None and np.isfinite(u0).all(), f"u={u0}")

span = (n_times - 1) * dt_h
u_over, v_over = get_velocity_interpolated(_QLON, _QLAT, field, span + 1.0, 0.0)
check("L2", "越界 (span+1h) → 回傳 None", u_over is None, f"span={span:.1f}h")

u_edge, v_edge = get_velocity_interpolated(_QLON, _QLAT, field, span, 0.0)
check("L2", "邊界值 (span) → 不誤判越界", u_edge is not None, f"span={span:.1f}h")

if span > 2 * dt_h:
    u_off0, _ = get_velocity_interpolated(_QLON, _QLAT, field, 0.0, 0.0)
    u_offN, _ = get_velocity_interpolated(_QLON, _QLAT, field, 0.0, span * 0.5)
    check("L2", "offset 改變 → 取到不同幀", not np.allclose(u_off0, u_offN),
          f"offset=0 u={u_off0[0]:.4f}, offset={span*0.5:.0f}h u={u_offN[0]:.4f}")
else:
    check("L2", "offset 改變 → 取到不同幀", True, "資料跨度太小，略過")

if n_times > 3:
    speeds = []
    for e in np.linspace(0, span * 0.9, 5):
        uu, vv = get_velocity_interpolated(_QLON, _QLAT, field, e, 0.0)
        speeds.append(float(np.hypot(uu[0], vv[0])) if uu is not None else np.nan)
    check("L2", "速度隨 elapsed 變化（時間軸有效）", len(set(np.round(speeds, 6))) > 1,
          f"speeds={[round(s,4) for s in speeds]}")
else:
    check("L2", "速度隨 elapsed 變化（時間軸有效）", True, "時間步太少，略過")

# ============================================================
# L3 插值層
# ============================================================
section("L3 插值數學正確性")

if lat.ndim == 1 and lon.ndim == 1:
    # 注意：interpolate_2d_vectorized 假設網格均勻間距，
    # 但 float32 的 lon[1]-lon[0] 有 ~4e-6 誤差，故用函式自身的索引公式
    # 反推「精確落在網格點」的查詢座標，才能嚴格驗證插值正確性。
    i_mid = int(np.argmin(np.abs(lon - OCEAN_LON)))
    j_mid = int(np.argmin(np.abs(lat - OCEAN_LAT)))
    i_mid = min(i_mid, len(lon) - 2)
    j_mid = min(j_mid, len(lat) - 2)
    dx = lon[1] - lon[0]
    dy = lat[1] - lat[0]
    x_exact = lon[0] + i_mid * dx   # 函式會算出 idx_x == i_mid
    y_exact = lat[0] + j_mid * dy
    val = interpolate_2d_vectorized(np.array([x_exact]), np.array([y_exact]),
                                    u[0], lat, lon)
    check("L3", "網格點插值 == 原值", np.isclose(val[0], u[0, j_mid, i_mid], atol=1e-4),
          f"interp={val[0]:.6f} vs raw={u[0, j_mid, i_mid]:.6f}")

    lin = (np.arange(len(lat))[:, None] * 1.0 + np.arange(len(lon))[None, :] * 2.0).astype(np.float32)
    xq = np.array([lon[0] + (i_mid + 0.5) * dx])
    yq = np.array([lat[0] + (j_mid + 0.5) * dy])
    val_lin = interpolate_2d_vectorized(xq, yq, lin, lat, lon)
    expect = (j_mid + 0.5) * 1.0 + (i_mid + 0.5) * 2.0
    check("L3", "線性場插值精確", np.isclose(val_lin[0], expect, atol=1e-2),
          f"interp={val_lin[0]:.4f} vs expect={expect:.4f}")

    xq2 = np.array([lon[0] + (i_mid + 0.5) * dx])
    yq2 = np.array([lat[0] + j_mid * dy])
    val_mid = interpolate_2d_vectorized(xq2, yq2, u[0], lat, lon)
    avg = (u[0, j_mid, i_mid] + u[0, j_mid, i_mid + 1]) / 2
    check("L3", "兩點中點插值 == 平均", np.isclose(val_mid[0], avg, atol=1e-4),
          f"interp={val_mid[0]:.6f} vs avg={avg:.6f}")
else:
    check("L3", "網格點插值 == 原值", True, "曲線網格，略過軸向測試")
    check("L3", "線性場插值精確", True, "曲線網格，略過")
    check("L3", "兩點中點插值 == 平均", True, "曲線網格，略過")

if n_times > 1:
    u_half, v_half = get_velocity_interpolated(_QLON, _QLAT, field, 0.5 * dt_h, 0.0)
    u_t0, _ = get_velocity_interpolated(_QLON, _QLAT, field, 0.0, 0.0)
    u_t1, _ = get_velocity_interpolated(_QLON, _QLAT, field, dt_h, 0.0)
    expect_half = (u_t0[0] + u_t1[0]) / 2
    check("L3", "時間兩幀中點 == 平均", np.isclose(u_half[0], expect_half, atol=1e-4),
          f"interp={u_half[0]:.6f} vs avg={expect_half:.6f}")
else:
    check("L3", "時間兩幀中點 == 平均", True, "時間步太少，略過")

# ============================================================
# L4 RK4 積分層
# ============================================================
section("L4 RK4 積分正確性")

const_prov = {
    "curr": {
        "lat": np.array([20.0, 26.0], dtype=np.float32),
        "lon": np.array([118.0, 125.0], dtype=np.float32),
        "u": np.ones((2, 2, 2), dtype=np.float32),
        "v": np.zeros((2, 2, 2), dtype=np.float32),
        "landmask": np.zeros((2, 2), dtype=bool),
        "land_geometry": None,
        "bbox": (118.0, 125.0, 20.0, 26.0),
        "n_times": 2, "dt_hours": 1.0,
        "time_start_utc": time_vals[0].to_pydatetime(),
        "time_end_utc": time_vals[-1].to_pydatetime(),
    },
    "wind": None,
    "reference_time_utc": time_vals[-1].to_pydatetime(),
}

dt_sec = -3600.0
lon0, lat0 = 121.0, 24.0
res = rk4_step_2d(np.array([lon0]), np.array([lat0]), dt_sec, const_prov, 0.0, 0.0)
check("L4", "RK4 回傳 4 個值", len(res) == 4, f"len={len(res)}")
new_lon, new_lat, mid_lons, mid_lats = res
check("L4", "RK4 回傳非 None", new_lon is not None, "")

if new_lon is not None:
    expected_dlon = -3600.0 * 1.0 / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat0)))
    actual_dlon = new_lon[0] - lon0
    check("L4", "常數流場位移精確", np.isclose(actual_dlon, expected_dlon, rtol=1e-3),
          f"actual={actual_dlon:.6f} vs expect={expected_dlon:.6f}")
    check("L4", "回溯方向正確（往西）", actual_dlon < 0, f"dlon={actual_dlon:.6f}")
    check("L4", "mid 形狀 (4,N)", mid_lons.shape == (4, 1), f"{mid_lons.shape}")
    check("L4", "mid 在起終點之間",
          bool(np.all(mid_lons >= min(lon0, new_lon[0]) - 1e-9) and
               np.all(mid_lons <= max(lon0, new_lon[0]) + 1e-9)),
          f"mid range=[{mid_lons.min():.4f},{mid_lons.max():.4f}]")

res_half = rk4_step_2d(np.array([lon0]), np.array([lat0]), dt_sec / 2, const_prov, 0.0, 0.0)
if res_half[0] is not None:
    dlon_half = res_half[0][0] - lon0
    check("L4", "步長減半位移減半（線性收斂）", np.isclose(dlon_half, expected_dlon / 2, rtol=1e-2),
          f"half={dlon_half:.6f} vs expect={expected_dlon/2:.6f}")

over_prov = clone_prov(const_prov)
over_prov["curr"]["n_times"] = 1
res_over = rk4_step_2d(np.array([lon0]), np.array([lat0]), dt_sec, over_prov, 0.0, 0.0)
check("L4", "時間越界 → 4 個 None", all(r is None for r in res_over), f"{res_over}")

# ============================================================
# L5 軌跡層
# ============================================================
section("L5 軌跡物理合理性")

sites = [(name, *HOTSPOT_CENTERS[name]) for name in list(HOTSPOT_CENTERS.keys())[:2]]

def displacement_for_steps(n_steps):
    df, hl, hla, _, _ = simulate_particles(
        release_sites=sites, total_particles=20, total_steps=n_steps,
        dt_mins=60.0, prov=prov, windage=0.0,
        reference_time_utc=prov["reference_time_utc"],
    )
    start = np.array([[hl[0][p], hla[0][p]] for p in range(len(hl[0]))])
    end = []
    for p in range(len(hl[0])):
        for i in range(len(hl) - 1, -1, -1):
            if np.isfinite(hl[i][p]) and np.isfinite(hla[i][p]):
                end.append([hl[i][p], hla[i][p]])
                break
    end = np.array(end)
    if len(end) == 0:
        return 0.0
    d = np.hypot(end[:, 0].mean() - start[:, 0].mean(), end[:, 1].mean() - start[:, 1].mean())
    return float(d * METERS_PER_DEG_LAT / 1000.0)

disp_short = displacement_for_steps(12)
disp_long = displacement_for_steps(24)
check("L5", "位移單調（24步 ≥ 12步）", disp_long >= disp_short - 1e-6,
      f"12步={disp_short:.2f}km, 24步={disp_long:.2f}km")

df_full, hl_full, hla_full, _, _ = simulate_particles(
    release_sites=sites, total_particles=20, total_steps=24,
    dt_mins=60.0, prov=prov, windage=0.0,
    reference_time_utc=prov["reference_time_utc"],
)
crossing = 0
for p in range(len(hl_full[0])):
    prev = None
    for i in range(len(hl_full)):
        lo, la = hl_full[i][p], hla_full[i][p]
        if not (np.isfinite(lo) and np.isfinite(la)):
            break
        if prev is not None:
            dl = np.linspace(prev[0], lo, 50)
            da = np.linspace(prev[1], la, 50)
            if is_land_vectorized(dl, da, prov).any():
                crossing += 1
                break
        prev = (lo, la)
check("L5", "無軌跡跨越陸地", crossing == 0, f"穿陸粒子數={crossing}")

final = df_full.sort_values(["particle_id", "step_index"]).groupby("particle_id").tail(1)
beached = final[final["status_code"] == STATUS_BEACHED]
false_beach = 0
for pid in beached["particle_id"].values:
    for i in range(len(hl_full) - 1, -1, -1):
        lo, la = hl_full[i][pid], hla_full[i][pid]
        if np.isfinite(lo) and np.isfinite(la):
            near = False
            for d in np.linspace(0, 0.01, 20):
                for ang in np.linspace(0, 2 * np.pi, 16):
                    if is_land_vectorized(np.array([lo + d*np.cos(ang)]),
                                          np.array([la + d*np.sin(ang)]), prov)[0]:
                        near = True
                        break
                if near:
                    break
            if not near:
                false_beach += 1
            break
check("L5", "觸岸粒子終點在陸地邊緣", false_beach == 0,
      f"觸岸={len(beached)}顆, 誤判={false_beach}顆")

max_step_km = 0.0
for p in range(len(hl_full[0])):
    for i in range(1, len(hl_full)):
        lo0, la0 = hl_full[i-1][p], hla_full[i-1][p]
        lo1, la1 = hl_full[i][p], hla_full[i][p]
        if np.isfinite(lo0) and np.isfinite(lo1):
            d = np.hypot(lo1 - lo0, la1 - la0) * METERS_PER_DEG_LAT / 1000.0
            max_step_km = max(max_step_km, d)
max_allowed_km = (CURRENT_SPEED_CAP_MPS + 0.0) * 3600.0 / 1000.0
check("L5", "每步位移 ≤ 最大流速×dt", max_step_km <= max_allowed_km * 1.05,
      f"max={max_step_km:.2f}km, 上限={max_allowed_km:.2f}km")

df_a, _, _, _, _ = simulate_particles(sites, 20, 12, 60.0, prov, 0.0, prov["reference_time_utc"])
df_b, _, _, _, _ = simulate_particles(sites, 20, 12, 60.0, prov, 0.0, prov["reference_time_utc"])
same = df_a[["particle_id", "step_index", "lon", "lat"]].equals(
    df_b[["particle_id", "step_index", "lon", "lat"]])
check("L5", "可重現性（同 seed 兩次相同）", same, "")

# ============================================================
# 總結
# ============================================================
section("驗證總結")

total = len(_results)
passed = sum(1 for _, _, p, _ in _results if p)
failed = total - passed

by_group = {}
for g, _, p, _ in _results:
    by_group.setdefault(g, [0, 0])
    by_group[g][0] += 1
    if p:
        by_group[g][1] += 1

print(f"  資料來源: {data_source}")
print()
for g in sorted(by_group.keys()):
    t, p = by_group[g]
    mark = "✅" if p == t else "❌"
    print(f"  {mark} {g}: {p}/{t}")
print()
print(f"  總計: {passed}/{total} PASS, {failed} FAIL")

if failed > 0:
    print("\n  失敗項目：")
    for g, name, p, detail in _results:
        if not p:
            print(f"    ❌ [{g}] {name}  ({detail})")

print("\n" + "=" * 72)
sys.exit(0 if failed == 0 else 1)
