# -*- coding: utf-8 -*-
"""
================================================================================
 verify_land_crossing_fix.py  —  驗證「零穿陸」修法（唯讀）
================================================================================

目的
----
比較三種觸岸判定策略，確認哪一種能保證「碰到陸地一定停止」：

  策略 A（現行）：僅檢查 RK4 終點
  策略 B（中間點）：檢查終點 + 4 個 RK4 中間點
  策略 C（線段採樣）：將 RK4 折線路徑（起點→k1→k2→k3→終點）以
                      < 網格間距 的間距等距採樣，逐點檢查

判定「穿陸」的黃金標準（ground truth）
--------------------------------------
對每一步，用極密的採樣（間距 200 m，遠小於網格 2.2 km）檢查整條路徑，
若發現任何採樣點落在陸地格，即為「真實穿陸」。

然後檢查各策略是否漏判（真實穿陸但策略未判觸岸）。

用法
----
  python verify_land_crossing_fix.py
  python verify_land_crossing_fix.py --particles 100 --steps 96
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CK_PATH = os.path.join(BASE_DIR, "ck.py")


def load_ck_module():
    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def _inner(fn):
                return fn

            return _inner

        def __iter__(self):
            return iter([])

        def __bool__(self):
            return False

    class _FakeStreamlit:
        def __getattr__(self, name):
            return _Any()

    sys.modules["streamlit"] = _FakeStreamlit()

    with open(CK_PATH, encoding="utf-8") as fh:
        src = fh.read()

    ui_start = src.find("st.sidebar.header")
    if ui_start < 0:
        raise RuntimeError("找不到 ck.py 的 Streamlit UI 主體起點")

    namespace = {"__name__": "ck_module", "__file__": CK_PATH}
    exec(compile(src[:ui_start], CK_PATH, "exec"), namespace)
    return namespace


def rk4_step_with_path(lon, lat, dt_seconds, prov, hours_elapsed, windage,
                       time_offset_hours=0.0):
    """與 ck.py 的 rk4_step_2d 數學一致，回傳終點 + 4 個中間點。"""
    ck = load_ck_module.__globals__["_CK_NS"]
    METERS_PER_DEG_LAT = ck["METERS_PER_DEG_LAT"]
    CURRENT_SPEED_CAP_MPS = ck["CURRENT_SPEED_CAP_MPS"]
    WIND_SPEED_CAP_MPS = ck["WIND_SPEED_CAP_MPS"]
    get_velocity_interpolated = ck["get_velocity_interpolated"]

    if isinstance(time_offset_hours, dict):
        curr_offset = float(time_offset_hours.get("curr", 0.0))
        wind_offset = float(time_offset_hours.get("wind", 0.0))
    else:
        curr_offset = wind_offset = float(time_offset_hours)

    def to_dlon(u, lat_here):
        return u / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat_here)) + 1e-12)

    def to_dlat(v):
        return v / METERS_PER_DEG_LAT

    def get_uv(l, la, t_elap):
        uc, vc = get_velocity_interpolated(l, la, prov["curr"], t_elap, curr_offset)
        c_spd = np.hypot(uc, vc)
        c_scale = np.where(c_spd > CURRENT_SPEED_CAP_MPS,
                           CURRENT_SPEED_CAP_MPS / (c_spd + 1e-12), 1.0)
        uc, vc = uc * c_scale, vc * c_scale

        if prov["wind"] is not None:
            uw, vw = get_velocity_interpolated(l, la, prov["wind"], t_elap, wind_offset)
            w_spd = np.hypot(uw, vw)
            w_scale = np.where(w_spd > WIND_SPEED_CAP_MPS,
                               WIND_SPEED_CAP_MPS / (w_spd + 1e-12), 1.0)
            uc = uc + uw * w_scale * windage
            vc = vc + vw * w_scale * windage

        total_cap = CURRENT_SPEED_CAP_MPS + windage * WIND_SPEED_CAP_MPS
        total_spd = np.hypot(uc, vc)
        scale = np.where(total_spd > total_cap, total_cap / (total_spd + 1e-12), 1.0)
        return uc * scale, vc * scale

    u1, v1 = get_uv(lon, lat, hours_elapsed)
    dlon1, dlat1 = to_dlon(u1, lat), to_dlat(v1)

    t2 = hours_elapsed + abs(0.5 * dt_seconds) / 3600.0
    lon2 = lon + 0.5 * dt_seconds * dlon1
    lat2 = lat + 0.5 * dt_seconds * dlat1

    u2, v2 = get_uv(lon2, lat2, t2)
    dlon2, dlat2 = to_dlon(u2, lat2), to_dlat(v2)

    lon3 = lon + 0.5 * dt_seconds * dlon2
    lat3 = lat + 0.5 * dt_seconds * dlat2

    u3, v3 = get_uv(lon3, lat3, t2)
    dlon3, dlat3 = to_dlon(u3, lat3), to_dlat(v3)

    t4 = hours_elapsed + abs(dt_seconds) / 3600.0
    lon4 = lon + dt_seconds * dlon3
    lat4 = lat + dt_seconds * dlat3

    u4, v4 = get_uv(lon4, lat4, t4)
    dlon4, dlat4 = to_dlon(u4, lat4), to_dlat(v4)

    new_lon = lon + (dt_seconds / 6.0) * (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4)
    new_lat = lat + (dt_seconds / 6.0) * (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4)

    # 路徑節點：起點 → k1 → k2 → k3 → 終點
    path_lons = np.vstack([lon, lon2, lon3, lon4, new_lon])
    path_lats = np.vstack([lat, lat2, lat3, lat4, new_lat])
    return new_lon, new_lat, path_lons, path_lats


def sample_polyline(path_lons, path_lats, spacing_km):
    """將折線路徑以 spacing_km 間距等距採樣，回傳 (lons, lats) 扁平陣列。

    path_lons/path_lats 形狀 (n_nodes, n_particles)。
    """
    n_nodes, n_particles = path_lons.shape
    out_lons = []
    out_lats = []

    for p in range(n_particles):
        pl = path_lons[:, p]
        pa = path_lats[:, p]
        # 逐段採樣
        seg_lons = [pl[0]]
        seg_lats = [pa[0]]
        for k in range(n_nodes - 1):
            x0, y0 = pl[k], pa[k]
            x1, y1 = pl[k + 1], pa[k + 1]
            # 段長（km，近似）
            dlat_km = (y1 - y0) * 111.32
            dlon_km = (x1 - x0) * 111.32 * np.cos(np.deg2rad((y0 + y1) * 0.5))
            seg_km = float(np.hypot(dlon_km, dlat_km))
            n_sub = max(1, int(np.ceil(seg_km / spacing_km)))
            for s in range(1, n_sub + 1):
                f = s / n_sub
                seg_lons.append(x0 + f * (x1 - x0))
                seg_lats.append(y0 + f * (y1 - y0))
        out_lons.append(np.array(seg_lons))
        out_lats.append(np.array(seg_lats))

    return out_lons, out_lats


def main():
    parser = argparse.ArgumentParser(description="驗證零穿陸修法（唯讀）")
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--dt-mins", type=float, default=30.0)
    parser.add_argument("--windage", type=float, default=0.03)
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    if args.source:
        os.environ["PLASTIC_DATA_ROOT"] = args.source

    print("=" * 74)
    print(" 零穿陸修法驗證（唯讀）")
    print("=" * 74)

    ck = load_ck_module()
    load_ck_module.__globals__["_CK_NS"] = ck

    is_land_vectorized = ck["is_land_vectorized"]
    snap_to_valid_ocean = ck["snap_to_valid_ocean"]
    meters_to_deg_lon = ck["meters_to_deg_lon"]
    meters_to_deg_lat = ck["meters_to_deg_lat"]

    print(f" 資料模式 : {ck['DATA_MODE']}   根目錄: {ck['DATA_ROOT']}")
    print(" 載入速度場...")
    prov = ck["build_velocity_providers"](need_wind=True)

    lon_ax = prov["curr"]["lon"]
    lat_ax = prov["curr"]["lat"]
    grid_km = float((lat_ax[1] - lat_ax[0]) * 111.32)
    print(f" 網格間距 : {grid_km:.3f} km")
    print(f" 採樣間距 : 1.0 km（策略 C） / 0.2 km（黃金標準）")
    print()

    # 建立粒子
    rng = np.random.default_rng(42)
    sites = [(k, v[0], v[1]) for k, v in ck["HOTSPOT_RELEASE_POINTS"].items()]
    base = args.particles // len(sites)
    rem = args.particles % len(sites)
    counts = [base + (1 if i < rem else 0) for i in range(len(sites))]

    lons_list, lats_list = [], []
    for idx, (name, lon0, lat0) in enumerate(sites):
        n_site = counts[idx]
        if n_site <= 0:
            continue
        lon0, lat0, snapped = snap_to_valid_ocean(lon0, lat0, prov)
        if snapped < 0:
            continue
        ang = rng.random(n_site) * 2 * np.pi
        rs = np.sqrt(rng.random(n_site)) * 500.0
        lons_list.append(lon0 + meters_to_deg_lon(rs * np.cos(ang), lat0))
        lats_list.append(lat0 + meters_to_deg_lat(rs * np.sin(ang)))

    lons0 = np.concatenate(lons_list)
    lats0 = np.concatenate(lats_list)
    n = len(lons0)
    print(f" 粒子數 : {n}")
    print()

    # 三策略狀態
    lon_a, lat_a = lons0.copy(), lats0.copy()
    lon_b, lat_b = lons0.copy(), lats0.copy()
    lon_c, lat_c = lons0.copy(), lats0.copy()
    act_a = np.ones(n, bool)
    act_b = np.ones(n, bool)
    act_c = np.ones(n, bool)

    dt_seconds = -abs(args.dt_mins) * 60.0
    base_time = prov["reference_time_utc"] or datetime.now(timezone.utc)

    def _off(field):
        if field is None or field.get("time_end_utc") is None:
            return 0.0
        return max(0.0, (field["time_end_utc"] - base_time).total_seconds() / 3600.0)

    offsets = {"curr": _off(prov["curr"]), "wind": _off(prov["wind"])}
    bbox = prov["curr"]["bbox"]

    # 統計：真實穿陸（黃金標準）vs 各策略漏判
    truth_crossings = 0
    miss_a = 0
    miss_b = 0
    miss_c = 0
    truth_particles = set()
    miss_a_particles = set()
    miss_b_particles = set()
    miss_c_particles = set()

    for step in range(args.steps):
        hours_elapsed = max(0.0, abs(step * dt_seconds) / 3600.0)
        max_hours = max((prov["curr"]["n_times"] - 1) * prov["curr"]["dt_hours"]
                        - offsets["curr"], 0.0)
        if (max_hours > 0) and (hours_elapsed > max_hours):
            break

        for tag, (lon_arr, lat_arr, act) in {
            "A": (lon_a, lat_a, act_a),
            "B": (lon_b, lat_b, act_b),
            "C": (lon_c, lat_c, act_c),
        }.items():
            idx = np.where(act)[0]
            if len(idx) == 0:
                continue

            new_lon, new_lat, path_lons, path_lats = rk4_step_with_path(
                lon_arr[idx], lat_arr[idx], dt_seconds, prov, hours_elapsed,
                args.windage, offsets,
            )
            oob = ((new_lon < bbox[0]) | (new_lon > bbox[1])
                   | (new_lat < bbox[2]) | (new_lat > bbox[3]))

            # 黃金標準：0.2 km 採樣
            gl, ga = sample_polyline(path_lons, path_lats, 0.2)
            truth_land = np.zeros(len(idx), bool)
            for p in range(len(idx)):
                if np.any(is_land_vectorized(gl[p], ga[p], prov)):
                    truth_land[p] = True

            # 策略 A：僅終點
            land_a = is_land_vectorized(new_lon, new_lat, prov)

            # 策略 B：終點 + 4 中間點
            land_b = land_a.copy()
            for k in range(1, path_lons.shape[0] - 1):
                land_b |= is_land_vectorized(path_lons[k], path_lats[k], prov)

            # 策略 C：1.0 km 線段採樣
            cl, ca = sample_polyline(path_lons, path_lats, 1.0)
            land_c = np.zeros(len(idx), bool)
            for p in range(len(idx)):
                if np.any(is_land_vectorized(cl[p], ca[p], prov)):
                    land_c[p] = True

            # 真實穿陸 = 黃金標準判定為陸，但該策略未判觸岸
            for p in range(len(idx)):
                pid = int(idx[p])
                if truth_land[p]:
                    truth_crossings += 1
                    truth_particles.add(pid)
                    if not land_a[p]:
                        miss_a += 1
                        miss_a_particles.add(pid)
                    if not land_b[p]:
                        miss_b += 1
                        miss_b_particles.add(pid)
                    if not land_c[p]:
                        miss_c += 1
                        miss_c_particles.add(pid)

            # 更新狀態（各策略用自己的判定）
            if tag == "A":
                lon_a[idx], lat_a[idx] = new_lon, new_lat
                act_a[idx] = ~(oob | land_a)
            elif tag == "B":
                lon_b[idx], lat_b[idx] = new_lon, new_lat
                act_b[idx] = ~(oob | land_b)
            else:
                lon_c[idx], lat_c[idx] = new_lon, new_lat
                act_c[idx] = ~(oob | land_c)

    print("=" * 74)
    print(" 結果")
    print("=" * 74)
    print(f" 黃金標準（0.2 km 採樣）偵測到的真實穿陸步數 : {truth_crossings}")
    print(f" 涉及粒子數                                  : {len(truth_particles)}")
    print()
    print(f" 【漏判（真實穿陸但未停止）】")
    print(f"   策略 A（僅終點）    : {miss_a} 步 / {len(miss_a_particles)} 粒子")
    print(f"   策略 B（+4 中間點） : {miss_b} 步 / {len(miss_b_particles)} 粒子")
    print(f"   策略 C（1 km 採樣） : {miss_c} 步 / {len(miss_c_particles)} 粒子")
    print()
    print(f" 【最終存活數】")
    print(f"   策略 A : {int(np.sum(act_a))}")
    print(f"   策略 B : {int(np.sum(act_b))}")
    print(f"   策略 C : {int(np.sum(act_c))}")
    print()
    print("=" * 74)
    print(" 結論")
    print("=" * 74)
    if miss_c == 0:
        print(" ✅ 策略 C（1 km 線段採樣）達成零漏判：所有真實穿陸都被正確停止。")
    else:
        print(f" ⚠️ 策略 C 仍有 {miss_c} 步漏判，需縮小採樣間距。")
    if miss_b > 0:
        print(f" ⚠️ 策略 B（僅加 4 中間點）仍有 {miss_b} 步漏判，不足以保證零穿陸。")
    print("=" * 74)


if __name__ == "__main__":
    main()
