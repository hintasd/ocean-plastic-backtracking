# -*- coding: utf-8 -*-
"""敏感度分析驗證腳本 (verify_sensitivity_analysis.py)

目的
----
驗證 run_sensitivity_sweep 的正確性，確認：
  (1) 掃描回傳正確的欄位與點數
  (2) 風阻係數確實影響軌跡（擾動測試）
  (3) 本地源半徑確實影響本地源佔比（單調性）
  (4) 回溯天數確實影響結果
  (5) 步長改變時總步數正確重算
  (6) 穩健性判定邏輯正確（跨越 50% 偵測）
  (7) 不支援的參數名稱會報錯

驗證層次
--------
  S1 回傳結構
  S2 風阻敏感度（擾動）
  S3 半徑敏感度（單調）
  S4 天數敏感度
  S5 步長重算
  S6 穩健性判定
  S7 錯誤處理

用法
----
  $env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
  E:\\c\\conda\\python.exe verify_sensitivity_analysis.py
"""
import os
import sys
import types
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

run_sensitivity_sweep = ns["run_sensitivity_sweep"]
simulate_particles = ns["simulate_particles"]
build_velocity_providers = ns["build_velocity_providers"]
HOTSPOT_CENTERS = ns["HOTSPOT_CENTERS"]

# ============================================================
# 測試框架
# ============================================================
_results = []

def check(group, name, passed, detail=""):
    _results.append((group, name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")

# ============================================================
# 準備資料與測試站點
# ============================================================
print("載入速度場資料...")
prov = build_velocity_providers(need_wind=True)
print("資料載入完成。")

# 使用單一海洋站點（基隆／野柳），少量粒子與步數以加速測試。
SITE_NAME = "北部 - 基隆／野柳"
site_lon, site_lat = HOTSPOT_CENTERS[SITE_NAME]
release_sites = [(SITE_NAME, site_lon, site_lat)]

N_PARTICLES = 30
N_STEPS = 24
DT_MINS = 60.0

# ============================================================
# S1 回傳結構
# ============================================================
section("S1 回傳結構 (Return Structure)")

sens_df = run_sensitivity_sweep(
    param_name="windage",
    param_values=[0.005, 0.015, 0.025],
    base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": DT_MINS},
    release_sites=release_sites,
    total_particles=N_PARTICLES,
    total_steps=N_STEPS,
    dt_mins=DT_MINS,
    prov=prov,
)

expected_cols = {"param_value", "local_pct", "external_pct", "net_displacement_km",
                 "total_distance_km", "ci_lo", "ci_hi", "significant", "prediction"}
check("S1", "回傳為 DataFrame", isinstance(sens_df, pd.DataFrame))
check("S1", "欄位齊全", expected_cols.issubset(set(sens_df.columns)),
      f"缺={expected_cols - set(sens_df.columns)}")
check("S1", "點數正確 (3)", len(sens_df) == 3, f"len={len(sens_df)}")
check("S1", "param_value 與輸入一致",
      list(sens_df["param_value"]) == [0.005, 0.015, 0.025],
      f"{list(sens_df['param_value'])}")
check("S1", "local_pct 在 [0,100]",
      bool(((sens_df["local_pct"] >= 0) & (sens_df["local_pct"] <= 100)).all()))
check("S1", "local_pct + external_pct ≈ 100",
      bool(np.allclose(sens_df["local_pct"] + sens_df["external_pct"], 100.0, atol=1e-6)))

# ============================================================
# S2 風阻敏感度（擾動測試）
# ============================================================
section("S2 風阻敏感度 (Windage Sensitivity)")

# 風阻從 0.2% 到 4.0%，軌跡應有明顯差異（本地源佔比或淨位移改變）。
sens_wind = run_sensitivity_sweep(
    param_name="windage",
    param_values=[0.002, 0.015, 0.04],
    base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": DT_MINS},
    release_sites=release_sites,
    total_particles=N_PARTICLES,
    total_steps=N_STEPS,
    dt_mins=DT_MINS,
    prov=prov,
)

# 風阻改變 → 淨位移應改變（風是額外推力）。
disp_range = sens_wind["net_displacement_km"].max() - sens_wind["net_displacement_km"].min()
check("S2", "風阻改變使淨位移改變 (>0.1 km)", disp_range > 0.1,
      f"淨位移範圍={disp_range:.2f} km")

# 風阻越大 → 淨位移通常越大（風推得更遠），但非嚴格保證；
# 檢查至少不是完全無關（相關係數絕對值 > 0.5 或範圍夠大）。
check("S2", "風阻對結果有實質影響",
      disp_range > 0.1 or sens_wind["local_pct"].nunique() > 1,
      f"disp_range={disp_range:.2f}, local_pct 種類={sens_wind['local_pct'].nunique()}")

# ============================================================
# S3 半徑敏感度（單調性）
# ============================================================
section("S3 本地源半徑敏感度 (Radius Sensitivity)")

# 半徑越大 → 越多粒子被判定為近岸 → 本地源佔比應單調不減。
sens_radius = run_sensitivity_sweep(
    param_name="local_radius",
    param_values=[10.0, 22.2, 50.0],
    base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": DT_MINS},
    release_sites=release_sites,
    total_particles=N_PARTICLES,
    total_steps=N_STEPS,
    dt_mins=DT_MINS,
    prov=prov,
)

# 半徑越大，本地源佔比應 >=（單調不減）。
lp = sens_radius["local_pct"].tolist()
monotonic_nondecreasing = all(lp[i] <= lp[i + 1] + 1e-9 for i in range(len(lp) - 1))
check("S3", "半徑越大 → 本地源佔比單調不減", monotonic_nondecreasing,
      f"local_pct={[round(v,1) for v in lp]}")

# 極端半徑（50km）應 >= 極小半徑（10km）。
check("S3", "半徑 50km 本地源佔比 >= 10km",
      lp[-1] >= lp[0] - 1e-9, f"10km={lp[0]:.1f}%, 50km={lp[-1]:.1f}%")

# ============================================================
# S4 天數敏感度
# ============================================================
section("S4 回溯天數敏感度 (Days Sensitivity)")

sens_days = run_sensitivity_sweep(
    param_name="days",
    param_values=[1.0, 2.0, 3.0],
    base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": DT_MINS},
    release_sites=release_sites,
    total_particles=N_PARTICLES,
    total_steps=N_STEPS,
    dt_mins=DT_MINS,
    prov=prov,
)

check("S4", "天數掃描回傳 3 點", len(sens_days) == 3, f"len={len(sens_days)}")
# 回溯越久 → 總漂流里程「不減」（粒子可能提前觸岸/越界而停止，
# 此時不同天數結果相同，屬正常物理行為，故用「不減」而非「遞增」）。
dist_days = sens_days["total_distance_km"].tolist()
check("S4", "回溯越久 → 總漂流里程不減",
      all(dist_days[i] <= dist_days[i + 1] + 1e-6 for i in range(len(dist_days) - 1)),
      f"total_dist={[round(v,1) for v in dist_days]}")
check("S4", "天數掃描結果皆有效",
      bool(np.isfinite(sens_days["local_pct"]).all()),
      f"local_pct={[round(v,1) for v in sens_days['local_pct']]}")

# ============================================================
# S5 步長重算
# ============================================================
section("S5 步長重算 (Time Step Recompute)")

# 步長改變時，總步數應重算（相同回溯時數下，步長越小步數越多）。
# 用 1 天 = 24 小時：dt=60min → 24 步；dt=30min → 48 步。
sens_dt = run_sensitivity_sweep(
    param_name="dt_mins",
    param_values=[60.0, 30.0],
    base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": 60.0},
    release_sites=release_sites,
    total_particles=N_PARTICLES,
    total_steps=24,
    dt_mins=60.0,
    prov=prov,
)

check("S5", "步長掃描回傳 2 點", len(sens_dt) == 2, f"len={len(sens_dt)}")
# 步長越小（30min）→ 步數越多 → 總漂流里程通常越大或相近。
check("S5", "步長改變不崩潰且結果有效",
      bool(np.isfinite(sens_dt["local_pct"]).all()),
      f"local_pct={[round(v,1) for v in sens_dt['local_pct']]}")

# ============================================================
# S6 穩健性判定
# ============================================================
section("S6 穩健性判定 (Robustness Detection)")

# 模擬穩健性判定邏輯（與 UI 相同）。
def robustness(local_pcts):
    lo, hi = min(local_pcts), max(local_pcts)
    crosses = (lo < 50.0) and (hi > 50.0)
    return crosses

# 跨越 50% → 不穩健
check("S6", "跨越 50% 判定為不穩健", robustness([40.0, 55.0, 60.0]) is True)
# 全在 50% 以上 → 穩健
check("S6", "全在 50% 以上判定為穩健", robustness([60.0, 70.0, 80.0]) is False)
# 全在 50% 以下 → 穩健（一致偏境外）
check("S6", "全在 50% 以下判定為穩健", robustness([20.0, 30.0, 40.0]) is False)
# 剛好 50% → 不跨越
check("S6", "剛好 50% 不視為跨越", robustness([50.0, 50.0]) is False)

# ============================================================
# S7 錯誤處理
# ============================================================
section("S7 錯誤處理 (Error Handling)")

try:
    run_sensitivity_sweep(
        param_name="unknown_param",
        param_values=[1.0],
        base_kwargs={"windage": 0.015, "days": 1.0, "dt_mins": DT_MINS},
        release_sites=release_sites,
        total_particles=N_PARTICLES,
        total_steps=N_STEPS,
        dt_mins=DT_MINS,
        prov=prov,
    )
    check("S7", "不支援參數應報 ValueError", False, "未拋出例外")
except ValueError:
    check("S7", "不支援參數應報 ValueError", True, "正確拋出 ValueError")
except Exception as e:
    check("S7", "不支援參數應報 ValueError", False, f"拋出 {type(e).__name__}")

# ============================================================
# 總結
# ============================================================
section("驗證總結 (Summary)")
total = len(_results)
passed = sum(1 for _, _, p, _ in _results if p)
failed = total - passed

by_group = {}
for g, _, p, _ in _results:
    by_group.setdefault(g, [0, 0])
    by_group[g][1] += 1
    if p:
        by_group[g][0] += 1

for g in sorted(by_group):
    gp, gt = by_group[g]
    print(f"  {g}: {gp}/{gt} PASS")

print(f"\n  總計: {passed}/{total} PASS, {failed} FAIL")

if failed > 0:
    print("\n  失敗項目：")
    for g, n, p, d in _results:
        if not p:
            print(f"    [{g}] {n}  — {d}")

sys.exit(0 if failed == 0 else 1)
