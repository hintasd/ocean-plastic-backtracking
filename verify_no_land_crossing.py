# -*- coding: utf-8 -*-
"""
================================================================================
 verify_no_land_crossing.py  —  驗證修改後的 ck.py 已無穿陸（唯讀）
================================================================================

目的
----
直接呼叫「修改後的 ck.py」的 simulate_particles，取得實際輸出軌跡，
然後對每一條軌跡的每一段（相鄰兩步之間）做極密採樣（0.2 km），
檢查是否有任何採樣點落在陸地格。

若有 → 表示仍有「碰到陸地但沒停止」的情況（穿陸 bug 未修好）。
若無 → 表示零穿陸達成。

用法
----
  python verify_no_land_crossing.py
  python verify_no_land_crossing.py --particles 240 --steps 96
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys

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


def main():
    parser = argparse.ArgumentParser(description="驗證修改後 ck.py 無穿陸（唯讀）")
    parser.add_argument("--particles", type=int, default=240)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--dt-mins", type=float, default=30.0)
    parser.add_argument("--windage", type=float, default=0.03)
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    if args.source:
        os.environ["PLASTIC_DATA_ROOT"] = args.source

    print("=" * 74)
    print(" 驗證修改後 ck.py 無穿陸（唯讀）")
    print("=" * 74)

    ck = load_ck_module()
    load_ck_module.__globals__["_CK_NS"] = ck

    is_land_vectorized = ck["is_land_vectorized"]

    print(f" 資料模式 : {ck['DATA_MODE']}   根目錄: {ck['DATA_ROOT']}")
    print(" 載入速度場...")
    prov = ck["build_velocity_providers"](need_wind=True)

    sites = [(k, v[0], v[1]) for k, v in ck["HOTSPOT_RELEASE_POINTS"].items()]
    print(f" 熱點數 : {len(sites)}   粒子數 : {args.particles}   步數 : {args.steps}")
    print()

    print(" 執行 simulate_particles（修改後的 ck.py）...")
    df, hist_lon, hist_lat, run_id, particle_sites = ck["simulate_particles"](
        sites, args.particles, args.steps, args.dt_mins, prov, args.windage,
        reference_time_utc=prov["reference_time_utc"],
    )
    print(f" 完成。軌跡歷史長度 : {len(hist_lon)} 步")
    print()

    # 對每條軌跡的每一段做極密採樣檢查
    n_steps = len(hist_lon)
    n_particles = len(hist_lon[0]) if n_steps > 0 else 0
    print(f" 檢查 {n_particles} 個粒子 × {n_steps} 步的軌跡...")

    crossing_segments = 0
    crossing_particles = set()
    crossing_detail = []

    for s in range(1, n_steps):
        lon0 = hist_lon[s - 1]
        lat0 = hist_lat[s - 1]
        lon1 = hist_lon[s]
        lat1 = hist_lat[s]

        for p in range(n_particles):
            x0, y0 = lon0[p], lat0[p]
            x1, y1 = lon1[p], lat1[p]
            if not (np.isfinite(x0) and np.isfinite(y0)
                    and np.isfinite(x1) and np.isfinite(y1)):
                continue

            # 段長（km）
            dlat_km = (y1 - y0) * 111.32
            dlon_km = (x1 - x0) * 111.32 * np.cos(np.deg2rad((y0 + y1) * 0.5))
            seg_km = float(np.hypot(dlon_km, dlat_km))
            n_sub = max(1, int(np.ceil(seg_km / 0.2)))

            f = np.linspace(0.0, 1.0, n_sub + 1)
            sl = x0 + f * (x1 - x0)
            sa = y0 + f * (y1 - y0)
            land = is_land_vectorized(sl, sa, prov)

            if np.any(land):
                crossing_segments += 1
                crossing_particles.add(p)
                crossing_detail.append((s, p, seg_km, float(np.sum(land))))

    print()
    print("=" * 74)
    print(" 結果")
    print("=" * 74)
    print(f" 穿陸段數（相鄰兩步之間跨過陸地）: {crossing_segments}")
    print(f" 涉及粒子數                      : {len(crossing_particles)}")
    print()

    if crossing_segments == 0:
        print(" ✅ 零穿陸：所有軌跡段皆未跨過陸地格。")
    else:
        print(" ⚠️ 仍有穿陸段，前 10 筆：")
        for s, p, km, nland in crossing_detail[:10]:
            print(f"    步 {s} 粒子 {p}: 段長 {km:.2f} km, 陸地採樣點 {nland}")
    print("=" * 74)


if __name__ == "__main__":
    main()
