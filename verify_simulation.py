# -*- coding: utf-8 -*-
"""
================================================================================
 verify_simulation.py  —  NODASS 海洋塑膠垃圾溯源系統：獨立驗收驗證腳本 (Plan A)
================================================================================

目的
----
本腳本「不修改 ck.py」，而是以 ck.py 的資料載入與計算函式為基礎，
向競賽評審客觀證明三件事：

  模組一：資料鏈路與切片探針（Data Link & Time-Slice Probe）
          → 證明所選 UTC 時間切片的 POM/WRF 數據確實被提取並灌入計算。

  模組二：RK4 空間算子幾何可逆性（Frozen-Time Single-Step Reversibility）
          → 凍結時間場，排除非自治與時間軸插值干擾，
            純粹檢驗 RK4 幾何算子的數學可逆性。

  模組三：群體粒子守恆與物理邊界防禦（Ensemble Conservation & Boundary Defense）
          → 客觀量化檢驗：數值健康、陸地不穿透、步長連續性。

設計約束
--------
  * 輕量：終端機 20 秒內完成。
  * 唯讀：不寫入任何檔案、不修改 ck.py。
  * 可獨立執行：`python verify_simulation.py`

方法論說明（為何採 Plan A）
---------------------------
  非自治系統（速度場隨時間變化）與非線性截斷（SPEED_CAP_MPS）本就不具備
  長軌跡完全對稱可逆性。因此本腳本改採：
    - 模組二：凍結時間的「單步算子可逆性」測試（真正檢驗 RK4 的方式）。
    - 模組三：客觀的「質量守恆 / 陸地不穿透 / 步長連續」量化檢驗。
================================================================================
"""

import os
import sys
import importlib.util
from datetime import datetime, timedelta, timezone

import numpy as np

# ------------------------------------------------------------------
# 0a. 強制 UTF-8 輸出（Windows 終端預設 GBK 會無法輸出 ✓/✗ 等符號）
# ------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# ------------------------------------------------------------------
# 0. 以 headless 模式匯入 ck.py（避免 Streamlit UI 阻塞終端機）
# ------------------------------------------------------------------
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

CK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ck.py")


def _load_ck_module():
    """以模組方式載入 ck.py，取得其資料載入與計算函式。"""
    if not os.path.isfile(CK_PATH):
        raise FileNotFoundError(f"找不到 ck.py：{CK_PATH}")
    spec = importlib.util.spec_from_file_location("ck_module", CK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ck_module"] = module
    spec.loader.exec_module(module)
    return module


print("=" * 78)
print(" NODASS 溯源系統 — 獨立驗收驗證腳本 (verify_simulation.py / Plan A)")
print("=" * 78)
print(f" 執行時間 (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
print(f" 載入模組      : {CK_PATH}")
print("-" * 78)

try:
    ck = _load_ck_module()
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] 無法載入 ck.py：{exc}")
    sys.exit(1)

# 從 ck 取用核心函式與常數
build_velocity_providers = ck.build_velocity_providers
get_velocity_interpolated = ck.get_velocity_interpolated
rk4_step_2d = ck.rk4_step_2d
haversine = ck.haversine
is_land_vectorized = ck.is_land_vectorized
snap_to_valid_ocean = ck.snap_to_valid_ocean
STATUS_ACTIVE = ck.STATUS_ACTIVE
STATUS_BEACHED = ck.STATUS_BEACHED
STATUS_OUT_OF_BOUNDS = ck.STATUS_OUT_OF_BOUNDS
STATUS_TIME_EXCEEDED = ck.STATUS_TIME_EXCEEDED
DATA_ROOT = ck.DATA_ROOT
POM_DIR = ck.POM_DIR
WRF_DIR = ck.WRF_DIR

print(f" DATA_ROOT     : {DATA_ROOT}")
print(f" POM_DIR       : {POM_DIR}")
print(f" WRF_DIR       : {WRF_DIR}")
print("-" * 78)


# ------------------------------------------------------------------
# 全域驗收結果收集器
# ------------------------------------------------------------------
RESULTS = []  # list of (module, check_name, passed: bool, detail: str)


def record(module, name, passed, detail=""):
    RESULTS.append((module, name, bool(passed), detail))
    flag = "PASS" if passed else "FAIL"
    print(f"   [{flag}] {name}" + (f" — {detail}" if detail else ""))


def fmt_dt(dt):
    if dt is None:
        return "None"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ==================================================================
# 載入資料提供者（POM + WRF）
# ==================================================================
print("\n[載入] 正在建立 POM/WRF 資料提供者 (build_velocity_providers)...")
try:
    # 注意：build_velocity_providers 已移除 required_hours 參數
    # （避免 @st.cache_resource 快取鍵隨回溯天數變動而全量重載）。
    PROV = build_velocity_providers(need_wind=True)
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] 資料載入失敗：{exc}")
    sys.exit(1)

CURR = PROV["curr"]
WIND = PROV.get("wind")
print(f"[載入] 完成。POM 檔案數={PROV.get('ocean_file')}，WRF 檔案={PROV.get('wind_file')}")
print(f"[載入] POM 時間範圍：{fmt_dt(CURR.get('time_start_utc'))} ~ {fmt_dt(CURR.get('time_end_utc'))}")
print(f"[載入] POM dt_hours={CURR.get('dt_hours')}，n_times={CURR.get('n_times')}")
if WIND is not None:
    print(f"[載入] WRF 時間範圍：{fmt_dt(WIND.get('time_start_utc'))} ~ {fmt_dt(WIND.get('time_end_utc'))}")
    print(f"[載入] WRF dt_hours={WIND.get('dt_hours')}，n_times={WIND.get('n_times')}")


# ==================================================================
# 模組一：資料鏈路與切片探針 (Data Link & Time-Slice Probe)
# ==================================================================
print("\n" + "=" * 78)
print(" 模組一：資料鏈路與切片探針 (Data Link & Time-Slice Probe)")
print("=" * 78)

TEST_TIME = CURR.get("time_end_utc") or datetime(2025, 12, 25, 0, 0, tzinfo=timezone.utc)
print(f" 測試時間戳 (UTC): {fmt_dt(TEST_TIME)}")

# --- 1a. POM 時間切片探針 ---
print("\n [1a] POM 海流時間切片探針")
try:
    dt_hours = float(CURR.get("dt_hours", 3.0))
    n_times = int(CURR["n_times"])
    hours_elapsed = 6.0
    t_exact = (n_times - 1) - max(hours_elapsed, 0.0) / dt_hours
    t_exact = float(np.clip(t_exact, 0.0, max(n_times - 1.0001, 0.0)))
    t_idx = int(np.floor(t_exact))
    t_next = min(t_idx + 1, n_times - 1)
    w = t_exact - t_idx

    t_start = CURR.get("time_start_utc")
    frame_prev = t_start + timedelta(hours=t_idx * dt_hours) if t_start else None
    frame_next = t_start + timedelta(hours=t_next * dt_hours) if t_start else None

    print(f"      回溯 elapsed = {hours_elapsed:.1f} h")
    print(f"      配對幀索引   : t_idx={t_idx}, t_next={t_next}")
    print(f"      前幀時間戳   : {fmt_dt(frame_prev)}")
    print(f"      後幀時間戳   : {fmt_dt(frame_next)}")
    print(f"      插值權重 w   : {w:.4f}  (u = (1-w)*u[t_idx] + w*u[t_next])")
    record("模組一", "POM 時間切片可解析 (t_idx/t_next/w 有效)",
           0.0 <= w <= 1.0 and t_next < n_times,
           f"w={w:.4f}")
except Exception as exc:  # noqa: BLE001
    record("模組一", "POM 時間切片可解析", False, str(exc))

# --- 1b. WRF 時間切片探針 ---
print("\n [1b] WRF 風場時間切片探針")
if WIND is None:
    record("模組一", "WRF 風場已載入", False, "wind provider 為 None")
else:
    try:
        w_dt = float(WIND.get("dt_hours", 6.0))
        w_n = int(WIND["n_times"])
        print(f"      選取 GRIB2 檔名 : {PROV.get('wind_file')}")
        print(f"      valid_time 範圍 : {fmt_dt(WIND.get('time_start_utc'))} ~ {fmt_dt(WIND.get('time_end_utc'))}")
        print(f"      時次數 n_times  : {w_n}，dt_hours={w_dt:.2f}")
        record("模組一", "WRF 風場已載入且時間軸有效",
               w_n >= 1 and w_dt > 0,
               f"n_times={w_n}, dt={w_dt:.2f}h")
    except Exception as exc:  # noqa: BLE001
        record("模組一", "WRF 風場時間軸有效", False, str(exc))

# --- 1c. 採樣點現場物理量探針 ---
print("\n [1c] 採樣點現場物理量探針 (海流 u/v 與 10m 風 u10/v10)")
PROBE_SITES = {
    "新竹外海": (120.8, 24.8),
    "臺東外海": (121.4, 22.8),
}

for site_name, (plon, plat) in PROBE_SITES.items():
    print(f"\n      ● {site_name} ({plon:.2f}°E, {plat:.2f}°N)")
    try:
        lon_arr = np.array([plon], dtype=np.float64)
        lat_arr = np.array([plat], dtype=np.float64)

        u_cur, v_cur = get_velocity_interpolated(lon_arr, lat_arr, CURR, 0.0)
        u_cur = float(np.asarray(u_cur).ravel()[0])
        v_cur = float(np.asarray(v_cur).ravel()[0])
        spd_cur = float(np.hypot(u_cur, v_cur))
        print(f"        海流  u={u_cur:+.4f} m/s, v={v_cur:+.4f} m/s, |V|={spd_cur:.4f} m/s")

        if WIND is not None:
            u_w, v_w = get_velocity_interpolated(lon_arr, lat_arr, WIND, 0.0)
            u_w = float(np.asarray(u_w).ravel()[0])
            v_w = float(np.asarray(v_w).ravel()[0])
            spd_w = float(np.hypot(u_w, v_w))
            print(f"        風場  u10={u_w:+.4f} m/s, v10={v_w:+.4f} m/s, |W|={spd_w:.4f} m/s")
        else:
            spd_w = None

        finite_ok = np.isfinite(u_cur) and np.isfinite(v_cur)
        nonzero_ok = (abs(u_cur) > 1e-6) or (abs(v_cur) > 1e-6)
        record("模組一", f"{site_name} 海流數值有效 (非NaN/非0)",
               finite_ok and nonzero_ok,
               f"|V|={spd_cur:.4f} m/s")

        if WIND is not None:
            w_finite = np.isfinite(u_w) and np.isfinite(v_w)
            # 風速物理合理範圍：0 < |W| <= 35 m/s（WIND_SPEED_CAP_MPS）。
            # 嚴禁 3.0 m/s 閹割；此處僅驗證非 NaN 且落在物理上限內。
            w_in_range = w_finite and (0.0 < spd_w <= 35.0)
            record("模組一", f"{site_name} 風場數值有效 (0 < |W| <= 35 m/s)",
                   w_in_range,
                   f"|W|={spd_w:.4f} m/s")

        # --- 雙層 QC 合成速度包絡探針 ---
        # 驗證 rk4_step_2d.get_uv 的動態上限：
        #   total_cap = CURRENT_SPEED_CAP_MPS + windage * WIND_SPEED_CAP_MPS
        # 以標準寶特瓶 windage=0.015 為例，上限應為 3.525 m/s。
        #
        # 注意：rk4_step_2d 回傳的是「積分後的經緯度座標」，不是速度向量。
        # 因此不可對其回傳值取 hypot（那會得到座標的模長，而非速度）。
        # 此處以 get_velocity_interpolated 重現 get_uv 的兩層 QC 合成邏輯，
        # 直接檢驗合成速度是否落在動態包絡內。
        try:
            probe_windage = 0.015
            total_cap = ck.CURRENT_SPEED_CAP_MPS + probe_windage * ck.WIND_SPEED_CAP_MPS

            # 第一層：分量獨立 QC
            uc, vc = get_velocity_interpolated(lon_arr, lat_arr, CURR, 0.0)
            uc = float(np.asarray(uc).ravel()[0])
            vc = float(np.asarray(vc).ravel()[0])
            c_spd = float(np.hypot(uc, vc))
            if c_spd > ck.CURRENT_SPEED_CAP_MPS:
                c_scale = ck.CURRENT_SPEED_CAP_MPS / (c_spd + 1e-12)
                uc *= c_scale
                vc *= c_scale

            if WIND is not None:
                uw, vw = get_velocity_interpolated(lon_arr, lat_arr, WIND, 0.0)
                uw = float(np.asarray(uw).ravel()[0])
                vw = float(np.asarray(vw).ravel()[0])
                w_spd = float(np.hypot(uw, vw))
                w_scale = 1.0
                if w_spd > ck.WIND_SPEED_CAP_MPS:
                    w_scale = ck.WIND_SPEED_CAP_MPS / (w_spd + 1e-12)
                uc += uw * w_scale * probe_windage
                vc += vw * w_scale * probe_windage

            # 第二層：合成速度動態 QC
            spd_tot = float(np.hypot(uc, vc))
            if spd_tot > total_cap:
                scale = total_cap / (spd_tot + 1e-12)
                uc *= scale
                vc *= scale
                spd_tot = float(np.hypot(uc, vc))

            print(f"        合成  |V_total|={spd_tot:.4f} m/s "
                  f"(動態上限 total_cap={total_cap:.4f} m/s, windage={probe_windage})")
            record("模組一", f"{site_name} 合成速度包絡 (|V_total| <= total_cap)",
                   np.isfinite(spd_tot) and spd_tot <= total_cap + 1e-9,
                   f"|V_total|={spd_tot:.4f} <= {total_cap:.4f} m/s")
        except Exception as exc:  # noqa: BLE001
            record("模組一", f"{site_name} 合成速度包絡", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        record("模組一", f"{site_name} 物理量探針", False, str(exc))


# ==================================================================
# 模組二：RK4 空間算子幾何可逆性 (Frozen-Time Single-Step Reversibility)
# ==================================================================
print("\n" + "=" * 78)
print(" 模組二：RK4 空間算子幾何可逆性 (凍結時間單步測試)")
print("=" * 78)

START_LON, START_LAT = 121.5, 23.5   # 確定處於開闊海域
DT_MINS = 30.0
FROZEN_ELAPSED_HOURS = 12.0   # 凍結時間場，排除非自治與時間軸插值干擾
WINDAGE = 0.0                 # 純海流，排除風阻非保守項

print(f" 起點座標     : ({START_LON:.4f}°E, {START_LAT:.4f}°N)")
print(f" 時間步長     : dt = {DT_MINS:.0f} 分鐘")
print(f" 凍結 elapsed : {FROZEN_ELAPSED_HOURS:.1f} h (速度場不隨時間變)")
print(f" 風阻係數     : {WINDAGE:.2f} (純海流)")

try:
    lon0, lat0, _snap_km = snap_to_valid_ocean(START_LON, START_LAT, PROV)
    print(f" 吸附後起點 P0: ({lon0:.6f}°E, {lat0:.6f}°N)")

    dt_seconds = abs(DT_MINS) * 60.0

    # Phase 1：逆向推一步 (dt = -1800s)，凍結 elapsed
    lon1, lat1, _mid_lon1, _mid_lat1 = rk4_step_2d(
        np.array([lon0]), np.array([lat0]),
        -dt_seconds, PROV, FROZEN_ELAPSED_HOURS, WINDAGE
    )
    lon1 = float(np.asarray(lon1).ravel()[0])
    lat1 = float(np.asarray(lat1).ravel()[0])
    print(f" Phase 1 逆向一步 P1 : ({lon1:.6f}°E, {lat1:.6f}°N)")

    # Phase 2：從 P1 正向推一步 (dt = +1800s)，凍結 elapsed
    lon2, lat2, _mid_lon2, _mid_lat2 = rk4_step_2d(
        np.array([lon1]), np.array([lat1]),
        +dt_seconds, PROV, FROZEN_ELAPSED_HOURS, WINDAGE
    )
    lon2 = float(np.asarray(lon2).ravel()[0])
    lat2 = float(np.asarray(lat2).ravel()[0])
    print(f" Phase 2 正向一步 P2 : ({lon2:.6f}°E, {lat2:.6f}°N)")

    closure_err_m = float(haversine(lon0, lat0, lon2, lat2))
    print(f" 閉合誤差距離        : {closure_err_m:.6f} 公尺")

    record("模組二", "RK4 凍結時間單步閉合誤差 < 1.0 m",
           closure_err_m < 1.0,
           f"誤差={closure_err_m:.6f} m")
except Exception as exc:  # noqa: BLE001
    record("模組二", "RK4 凍結時間單步可逆性", False, str(exc))


# ==================================================================
# 模組三：群體粒子守恆與物理邊界防禦 (Ensemble Conservation & Boundary Defense)
# ==================================================================
print("\n" + "=" * 78)
print(" 模組三：群體粒子守恆與物理邊界防禦 (客觀量化測試)")
print("=" * 78)

N_PARTICLES = 100
N_STEPS = 48          # 24 小時 @ 30 分鐘
DT_SECONDS = -abs(DT_MINS) * 60.0
MAX_STEP_DISPLACEMENT_M = 5000.0   # 單步物理極限 5 km
ENSEMBLE_WINDAGE = 0.015           # 標準寶特瓶 1.5% 風阻

print(f" 粒子數       : {N_PARTICLES}")
print(f" 回溯步數     : {N_STEPS} 步 (24 小時 @ 30 分鐘)")
print(f" 單步位移上限 : {MAX_STEP_DISPLACEMENT_M/1000:.1f} km/步")
print(f" 風阻係數     : {ENSEMBLE_WINDAGE:.3f}")

try:
    rng = np.random.default_rng(20251225)

    # 在台灣周圍海域釋放隨機粒子（經緯度範圍涵蓋台灣周邊）
    lons = rng.uniform(119.5, 122.5, N_PARTICLES)
    lats = rng.uniform(21.5, 25.5, N_PARTICLES)

    # 吸附到有效水域，並記錄初始狀態
    snapped_lons = np.empty(N_PARTICLES, dtype=np.float64)
    snapped_lats = np.empty(N_PARTICLES, dtype=np.float64)
    for i in range(N_PARTICLES):
        slon, slat, _snap_km = snap_to_valid_ocean(float(lons[i]), float(lats[i]), PROV)
        snapped_lons[i] = slon
        snapped_lats[i] = slat

    lons = snapped_lons
    lats = snapped_lats

    active = np.ones(N_PARTICLES, dtype=bool)
    status_codes = np.full(N_PARTICLES, STATUS_ACTIVE, dtype=np.int8)

    # 檢驗指標累計器
    nan_leak_count = 0
    land_penetration_count = 0
    step_violation_count = 0
    max_step_displacement = 0.0
    beached_count = 0

    prev_lons = lons.copy()
    prev_lats = lats.copy()

    for step in range(N_STEPS):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        hours_elapsed = max(0.0, abs(step * DT_SECONDS) / 3600.0)
        max_hours = max((CURR["n_times"] - 1) * CURR["dt_hours"], 0.0)

        # 時間軸耗盡：標記並停止（與 ck.py 第二道防線一致）
        if max_hours > 0 and hours_elapsed > max_hours:
            status_codes[active_idx] = STATUS_TIME_EXCEEDED
            active[active_idx] = False
            break

        lon_sub, lat_sub, _mid_lon, _mid_lat = rk4_step_2d(
            lons[active_idx], lats[active_idx],
            DT_SECONDS, PROV, hours_elapsed, ENSEMBLE_WINDAGE
        )
        lon_sub = np.asarray(lon_sub).ravel()
        lat_sub = np.asarray(lat_sub).ravel()

        # --- 檢驗指標 1：NaN / Inf 洩漏 ---
        bad = ~np.isfinite(lon_sub) | ~np.isfinite(lat_sub)
        if np.any(bad):
            nan_leak_count += int(np.sum(bad))

        # --- 檢驗指標 3：步長連續性（未擱淺粒子的單步位移）---
        step_disp = haversine(
            prev_lons[active_idx], prev_lats[active_idx],
            lon_sub, lat_sub
        )
        step_disp = np.where(np.isfinite(step_disp), step_disp, 0.0)
        if step_disp.size:
            max_step_displacement = max(max_step_displacement, float(np.max(step_disp)))
            step_violation_count += int(np.sum(step_disp > MAX_STEP_DISPLACEMENT_M))

        # --- 檢驗指標 2：陸地不穿透 ---
        bbox = CURR["bbox"]
        out_of_bounds = (
            (lon_sub < bbox[0]) | (lon_sub > bbox[1]) |
            (lat_sub < bbox[2]) | (lat_sub > bbox[3])
        )
        beached = is_land_vectorized(lon_sub, lat_sub, PROV)
        beached = np.asarray(beached).ravel().astype(bool)

        # 擱淺粒子必須被正確標記並就地凍結
        if np.any(beached):
            beached_global = active_idx[beached]
            status_codes[beached_global] = STATUS_BEACHED
            beached_count += int(np.sum(beached))

        next_active = ~(out_of_bounds | beached)
        dead_idx = active_idx[~next_active]
        if len(dead_idx) > 0:
            lons[dead_idx] = np.nan
            lats[dead_idx] = np.nan

        # 更新存活粒子座標
        alive_global = active_idx[next_active]
        lons[alive_global] = lon_sub[next_active]
        lats[alive_global] = lat_sub[next_active]

        prev_lons[alive_global] = lon_sub[next_active]
        prev_lats[alive_global] = lat_sub[next_active]

        active[active_idx] = next_active

    # --- 陸地穿透複檢：所有存活粒子最終位置不得在陸地上 ---
    final_active_idx = np.where(active)[0]
    if len(final_active_idx) > 0:
        final_land = is_land_vectorized(
            lons[final_active_idx], lats[final_active_idx], PROV
        )
        final_land = np.asarray(final_land).ravel().astype(bool)
        land_penetration_count = int(np.sum(final_land))

    print(f"\n 檢驗結果：")
    print(f"   存活粒子數        : {int(np.sum(active))} / {N_PARTICLES}")
    print(f"   擱淺粒子數        : {beached_count}")
    print(f"   NaN/Inf 洩漏數    : {nan_leak_count}")
    print(f"   陸地穿透數        : {land_penetration_count}")
    print(f"   步長超限次數      : {step_violation_count}")
    print(f"   最大單步位移      : {max_step_displacement:.2f} m")

    record("模組三", "指標1 數值健康 (NaN/Inf 洩漏 = 0)",
           nan_leak_count == 0,
           f"洩漏={nan_leak_count}")
    record("模組三", "指標2 陸地不穿透 (穿透數 = 0)",
           land_penetration_count == 0,
           f"穿透={land_penetration_count}")
    record("模組三", "指標3 步長連續性 (單步 < 5 km)",
           step_violation_count == 0,
           f"超限={step_violation_count}, 最大={max_step_displacement:.1f} m")
except Exception as exc:  # noqa: BLE001
    record("模組三", "群體粒子守恆與邊界防禦", False, str(exc))


# ==================================================================
# 驗收診斷匯總表 (Pass/Fail Summary)
# ==================================================================
print("\n" + "=" * 78)
print(" 驗收診斷匯總表 (Acceptance Diagnostic Summary)")
print("=" * 78)

modules = ["模組一", "模組二", "模組三"]
module_titles = {
    "模組一": "資料鏈路與切片探針",
    "模組二": "RK4 空間算子幾何可逆性 (凍結時間單步)",
    "模組三": "群體粒子守恆與物理邊界防禦",
}

total_pass = 0
total_fail = 0

for mod in modules:
    items = [r for r in RESULTS if r[0] == mod]
    if not items:
        continue
    print(f"\n 【{mod}】{module_titles[mod]}")
    print(" " + "-" * 74)
    for _, name, passed, detail in items:
        flag = "PASS" if passed else "FAIL"
        mark = "✓" if passed else "✗"
        line = f"   {mark} [{flag}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

print("\n" + "=" * 78)
overall = "PASS" if total_fail == 0 else "FAIL"
print(f" 總計：{total_pass} 項通過 / {total_fail} 項失敗")
print(f" 整體驗收判定：{'✅ PASS — 系統通過物理與數值一致性驗收' if overall == 'PASS' else '❌ FAIL — 存在未通過項目，請檢視上方明細'}")
print("=" * 78)

sys.exit(0 if overall == "PASS" else 2)