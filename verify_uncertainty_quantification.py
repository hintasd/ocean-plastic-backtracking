# -*- coding: utf-8 -*-
"""不確定性量化驗證腳本 (verify_uncertainty_quantification.py)

目的
----
驗證 bootstrap 信賴區間實作的正確性與統計性質，確認：
  (1) 點估計正確（等於樣本比例）
  (2) 信賴區間涵蓋真值（覆蓋率測試）
  (3) 樣本數越大，CI 越窄（收斂性）
  (4) 顯著性判定正確（CI 是否跨越 50%）
  (5) 邊界情況處理正確（空陣列、極小樣本、全 True/全 False）
  (6) 可重現性（固定 seed 結果一致）

驗證層次
--------
  U1 點估計正確性
  U2 信賴區間覆蓋率（統計正確性）
  U3 收斂性（樣本數 vs CI 寬度）
  U4 顯著性判定
  U5 邊界情況
  U6 可重現性

用法
----
  $env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
  E:\\c\\conda\\python.exe verify_uncertainty_quantification.py
"""
import os
import sys
import types
import numpy as np

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

bootstrap_proportion_ci = ns["bootstrap_proportion_ci"]

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
# U1 點估計正確性
# ============================================================
section("U1 點估計正確性 (Point Estimate)")

# 已知比例：60% True
flags = np.array([True] * 60 + [False] * 40)
r = bootstrap_proportion_ci(flags, n_boot=2000, seed=42)
check("U1", "點估計 = 樣本比例 (60%)", abs(r["point_pct"] - 60.0) < 1e-9,
      f"point={r['point_pct']:.4f}%")

# 已知比例：25% True
flags2 = np.array([True] * 25 + [False] * 75)
r2 = bootstrap_proportion_ci(flags2, n_boot=2000, seed=42)
check("U1", "點估計 = 樣本比例 (25%)", abs(r2["point_pct"] - 25.0) < 1e-9,
      f"point={r2['point_pct']:.4f}%")

# 樣本數回報正確
check("U1", "樣本數 n 回報正確", r["n"] == 100, f"n={r['n']}")

# ============================================================
# U2 信賴區間覆蓋率（統計正確性）
# ============================================================
section("U2 信賴區間覆蓋率 (Coverage Rate)")

# 真值 p=0.6，每次抽 n=100，重複 500 次，檢查 95% CI 是否涵蓋真值
true_p = 0.6
n_per = 100
n_trials = 500
covered = 0
rng = np.random.default_rng(123)
for t in range(n_trials):
    sample = rng.random(n_per) < true_p
    res = bootstrap_proportion_ci(sample, n_boot=1000, ci=95.0, seed=t)
    if res["lo_pct"] <= true_p * 100.0 <= res["hi_pct"]:
        covered += 1
coverage = covered / n_trials
# 95% CI 的理論覆蓋率應接近 0.95（允許 0.90~0.99 的抽樣波動）
check("U2", "95% CI 覆蓋率 ≈ 95% (0.90~0.99)", 0.90 <= coverage <= 0.99,
      f"coverage={coverage:.3f} ({covered}/{n_trials})")

# 真值 p=0.3 的覆蓋率
true_p2 = 0.3
covered2 = 0
for t in range(n_trials):
    sample = rng.random(n_per) < true_p2
    res = bootstrap_proportion_ci(sample, n_boot=1000, ci=95.0, seed=t + 10000)
    if res["lo_pct"] <= true_p2 * 100.0 <= res["hi_pct"]:
        covered2 += 1
coverage2 = covered2 / n_trials
check("U2", "p=0.3 時 95% CI 覆蓋率 ≈ 95%", 0.90 <= coverage2 <= 0.99,
      f"coverage={coverage2:.3f} ({covered2}/{n_trials})")

# ============================================================
# U3 收斂性（樣本數 vs CI 寬度）
# ============================================================
section("U3 收斂性 (Convergence: n vs CI Width)")

widths = {}
for n in [25, 100, 400, 1600]:
    # 固定真值 0.6，用固定 seed 產生樣本
    rng_n = np.random.default_rng(7)
    sample = rng_n.random(n) < 0.6
    res = bootstrap_proportion_ci(sample, n_boot=2000, seed=42)
    widths[n] = res["hi_pct"] - res["lo_pct"]

# CI 寬度應隨 n 增加而單調遞減
monotonic = all(widths[a] > widths[b] for a, b in zip([25, 100, 400], [100, 400, 1600]))
check("U3", "CI 寬度隨樣本數單調遞減", monotonic,
      " | ".join(f"n={n}: {w:.1f}%" for n, w in widths.items()))

# 理論上 CI 寬度 ∝ 1/sqrt(n)，n 變 4 倍寬度應約減半
ratio = widths[100] / widths[400]
check("U3", "n×4 → CI 寬度約減半 (1.5~2.5x)", 1.5 <= ratio <= 2.5,
      f"width(100)/width(400)={ratio:.2f}")

# ============================================================
# U4 顯著性判定
# ============================================================
section("U4 顯著性判定 (Significance)")

# 明確偏離 50%：80% True，n=200 → 應顯著
flags_sig = np.array([True] * 160 + [False] * 40)
r_sig = bootstrap_proportion_ci(flags_sig, n_boot=2000, seed=42)
check("U4", "80% (n=200) 判定為顯著", r_sig["significant"] is True,
      f"CI=[{r_sig['lo_pct']:.1f}, {r_sig['hi_pct']:.1f}]")

# 接近 50%：52% True，n=100 → 應不顯著（CI 跨越 50%）
flags_ns = np.array([True] * 52 + [False] * 48)
r_ns = bootstrap_proportion_ci(flags_ns, n_boot=2000, seed=42)
check("U4", "52% (n=100) 判定為不顯著", r_ns["significant"] is False,
      f"CI=[{r_ns['lo_pct']:.1f}, {r_ns['hi_pct']:.1f}]")

# 極端：100% True → 應顯著
flags_all = np.array([True] * 50)
r_all = bootstrap_proportion_ci(flags_all, n_boot=2000, seed=42)
check("U4", "100% True 判定為顯著", r_all["significant"] is True,
      f"CI=[{r_all['lo_pct']:.1f}, {r_all['hi_pct']:.1f}]")

# 極端：0% True → 應顯著（CI 完全 < 50%）
flags_none = np.array([False] * 50)
r_none = bootstrap_proportion_ci(flags_none, n_boot=2000, seed=42)
check("U4", "0% True 判定為顯著", r_none["significant"] is True,
      f"CI=[{r_none['lo_pct']:.1f}, {r_none['hi_pct']:.1f}]")

# ============================================================
# U5 邊界情況
# ============================================================
section("U5 邊界情況 (Edge Cases)")

# 空陣列
r_empty = bootstrap_proportion_ci(np.array([], dtype=bool))
check("U5", "空陣列回傳 n=0 且不崩潰", r_empty["n"] == 0 and r_empty["point_pct"] == 0.0,
      f"n={r_empty['n']}, point={r_empty['point_pct']}")

# 極小樣本 (n=3) → 退化為點估計
r_tiny = bootstrap_proportion_ci(np.array([True, True, False]), n_boot=2000, seed=42)
check("U5", "n<5 退化為點估計 (CI=point)",
      abs(r_tiny["lo_pct"] - r_tiny["point_pct"]) < 1e-9 and
      abs(r_tiny["hi_pct"] - r_tiny["point_pct"]) < 1e-9,
      f"point={r_tiny['point_pct']:.1f}, CI=[{r_tiny['lo_pct']:.1f}, {r_tiny['hi_pct']:.1f}]")

# 單一元素
r_one = bootstrap_proportion_ci(np.array([True]))
check("U5", "單一元素不崩潰", r_one["n"] == 1 and r_one["point_pct"] == 100.0,
      f"n={r_one['n']}, point={r_one['point_pct']}")

# CI 上下界合理性：lo <= point <= hi
flags_mid = np.array([True] * 60 + [False] * 40)
r_mid = bootstrap_proportion_ci(flags_mid, n_boot=2000, seed=42)
check("U5", "CI 上下界包含點估計 (lo ≤ point ≤ hi)",
      r_mid["lo_pct"] <= r_mid["point_pct"] <= r_mid["hi_pct"],
      f"lo={r_mid['lo_pct']:.1f}, point={r_mid['point_pct']:.1f}, hi={r_mid['hi_pct']:.1f}")

# CI 上下界在 [0, 100] 範圍內
check("U5", "CI 上下界落在 [0, 100]",
      0.0 <= r_mid["lo_pct"] <= 100.0 and 0.0 <= r_mid["hi_pct"] <= 100.0,
      f"lo={r_mid['lo_pct']:.1f}, hi={r_mid['hi_pct']:.1f}")

# ============================================================
# U6 可重現性
# ============================================================
section("U6 可重現性 (Reproducibility)")

flags_rep = np.array([True] * 60 + [False] * 40)
r_a = bootstrap_proportion_ci(flags_rep, n_boot=2000, seed=42)
r_b = bootstrap_proportion_ci(flags_rep, n_boot=2000, seed=42)
check("U6", "相同 seed 結果完全一致",
      r_a["lo_pct"] == r_b["lo_pct"] and r_a["hi_pct"] == r_b["hi_pct"],
      f"CI_a=[{r_a['lo_pct']:.4f}, {r_a['hi_pct']:.4f}]")

# 不同 seed 結果接近（但不完全相同）
r_c = bootstrap_proportion_ci(flags_rep, n_boot=2000, seed=999)
check("U6", "不同 seed 結果接近 (差異 < 2%)",
      abs(r_a["lo_pct"] - r_c["lo_pct"]) < 2.0 and abs(r_a["hi_pct"] - r_c["hi_pct"]) < 2.0,
      f"seed42=[{r_a['lo_pct']:.1f}, {r_a['hi_pct']:.1f}], seed999=[{r_c['lo_pct']:.1f}, {r_c['hi_pct']:.1f}]")

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
