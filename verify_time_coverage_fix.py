"""驗證「時間凍結偽漂流」修復。

背景（bug）：
    get_velocity_interpolated 舊版把 t_exact 用 np.clip(..., 0.0, ...) 夾住。
    當 offset + elapsed 超過資料總跨度時，速度場凍結在資料第一幀，
    粒子在靜止流場中繼續積分 → 偽漂流。
    症狀：回溯 20 天的位移反而比 18 天少、軌跡完全不同。

本腳本驗證三件事：
    1. get_velocity_interpolated 在越界時回傳 (None, None)，不再靜默 clip。
    2. 邊界內仍正常回傳速度（未誤殺）。
    3. rk4_step_2d 在越界時回傳 4 個 None。
    4. simulate_particles 在越界時把粒子標記為 STATUS_TIME_EXCEEDED 並停用，
       且「回溯越久位移越少」的異常不再發生（位移單調不減）。

執行：
    $env:PYTHONIOENCODING="utf-8"; E:\\c\\conda\\python.exe verify_time_coverage_fix.py
"""
import sys
import types
import numpy as np

# ------------------------------------------------------------
# 載入 ck.py 模組層級（stub streamlit，只 exec 到 sidebar 之前）
# ------------------------------------------------------------
stub = types.ModuleType("streamlit")


def _noop(*args, **kwargs):
    return None


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


for _name in [
    "set_page_config", "title", "caption", "markdown", "write", "info",
    "warning", "error", "success", "stop", "spinner", "header", "subheader",
    "checkbox", "selectbox", "slider", "number_input", "date_input", "button",
    "columns", "container", "expander", "dataframe", "plotly_chart", "metric",
    "text_input", "radio", "multiselect", "tabs", "empty", "progress",
]:
    setattr(stub, _name, _noop)
stub.sidebar = stub
stub.session_state = {}
stub.cache_data = lambda *a, **k: (lambda f: f)
stub.cache_resource = lambda *a, **k: (lambda f: f)
sys.modules["streamlit"] = stub

src = open("ck.py", encoding="utf-8").read()
cut = src.index("st.sidebar.header")
ns = {"__name__": "ck_under_test"}
exec(compile(src[:cut], "ck.py", "exec"), ns)

get_velocity_interpolated = ns["get_velocity_interpolated"]
rk4_step_2d = ns["rk4_step_2d"]
simulate_particles = ns["simulate_particles"]
STATUS_TIME_EXCEEDED = ns["STATUS_TIME_EXCEEDED"]
STATUS_ACTIVE = ns["STATUS_ACTIVE"]

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ------------------------------------------------------------
# 建立合成速度場：u 隨時間變化，讓「凍結幀」可被偵測
# ------------------------------------------------------------
NY, NX = 20, 20
NT = 745          # 模擬 POM 745 幀 @1h
DT_H = 1.0
lat = np.linspace(21.0, 25.0, NY, dtype=np.float32)
lon = np.linspace(119.0, 123.0, NX, dtype=np.float32)

# u[t] = 0.1 * t  → 不同時間幀速度明顯不同，凍結幀會產生可辨識的偽漂流
u = np.zeros((NT, NY, NX), dtype=np.float32)
v = np.zeros((NT, NY, NX), dtype=np.float32)
for t in range(NT):
    u[t] = 0.1 * t
    v[t] = 0.05 * t

field = {
    "lat": lat, "lon": lon, "u": u, "v": v,
    "n_times": NT, "dt_hours": DT_H,
}

lon_q = np.array([121.0], dtype=np.float32)
lat_q = np.array([23.0], dtype=np.float32)

# ------------------------------------------------------------
# 1. 邊界內：正常回傳
# ------------------------------------------------------------
u_in, v_in = get_velocity_interpolated(lon_q, lat_q, field, hours_elapsed=0.0, time_offset_hours=0.0)
check("邊界內 (offset=0, elapsed=0) 回傳有效速度",
      u_in is not None and np.isfinite(u_in).all(),
      f"u={u_in}")

# 資料總跨度 = (745-1)*1 = 744h；elapsed=744 恰好在邊界上
u_edge, v_edge = get_velocity_interpolated(lon_q, lat_q, field, hours_elapsed=744.0, time_offset_hours=0.0)
check("邊界上 (elapsed=744h) 仍回傳有效速度（不誤殺）",
      u_edge is not None and np.isfinite(u_edge).all(),
      f"u={u_edge}")

# ------------------------------------------------------------
# 2. 越界：回傳 None（不再凍結）
# ------------------------------------------------------------
u_over, v_over = get_velocity_interpolated(lon_q, lat_q, field, hours_elapsed=745.0, time_offset_hours=0.0)
check("越界 (elapsed=745h) 回傳 None（不再凍結幀）",
      u_over is None and v_over is None,
      f"u={u_over}")

# 重現原始 bug 場景：offset=738h（起算 12/01，資料末端 12/31 18:00）
u18, _ = get_velocity_interpolated(lon_q, lat_q, field, hours_elapsed=432.0, time_offset_hours=738.0)
u20, _ = get_velocity_interpolated(lon_q, lat_q, field, hours_elapsed=480.0, time_offset_hours=738.0)
check("原始 bug 場景：offset=738h 時 18 天與 20 天皆越界 → 皆回傳 None",
      u18 is None and u20 is None,
      f"18d={u18}, 20d={u20}")

# ------------------------------------------------------------
# 3. rk4_step_2d 越界時回傳 4 個 None
# ------------------------------------------------------------
prov = {
    "curr": field,
    "wind": None,
    "reference_time_utc": None,
}
r = rk4_step_2d(lon_q, lat_q, -1800.0, prov, hours_elapsed=745.0, windage=0.0,
                time_offset_hours={"curr": 0.0, "wind": 0.0})
check("rk4_step_2d 越界時回傳 4 個 None",
      all(x is None for x in r),
      f"len={len(r)}")

r_ok = rk4_step_2d(lon_q, lat_q, -1800.0, prov, hours_elapsed=0.0, windage=0.0,
                   time_offset_hours={"curr": 0.0, "wind": 0.0})
check("rk4_step_2d 邊界內回傳 4 個有效值",
      r_ok[0] is not None and r_ok[2] is not None and r_ok[2].shape[0] == 4,
      f"mid shape={None if r_ok[2] is None else r_ok[2].shape}")

# ------------------------------------------------------------
# 4. simulate_particles：越界粒子被標記 TIME_EXCEEDED 並停用
# ------------------------------------------------------------
# 讓粒子在固定流場中往西漂（u>0 表示向東；此處 u 隨時間遞增）
# 使用 offset 使回溯很快越界，驗證粒子被正確停用。
from datetime import datetime, timezone, timedelta

# 資料末端 = 2025-12-31 18:00 UTC；起算時刻設在資料末端前 738h
# → 只能回溯 6h，超過即越界。
ref = datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)
time_end = ref
time_start = ref - timedelta(hours=(NT - 1) * DT_H)

prov_sim = {
    "curr": dict(field, bbox=(119.0, 123.0, 21.0, 25.0),
                 landmask=np.zeros((NY, NX), dtype=bool),
                 land_geometry=None,
                 time_start_utc=time_start,
                 time_end_utc=time_end),
    "wind": None,
    "reference_time_utc": time_end,
}

# 起算時刻 = 資料末端前 738h（模擬使用者選 12/01）
start_time = time_end - timedelta(hours=738)

df, hist_lon, hist_lat, run_id, sites = simulate_particles(
    release_sites=[("test", 121.0, 23.0)],
    total_particles=10,
    total_steps=20,          # 20 步 × 1h = 20h > 可用 6h
    dt_mins=60.0,
    prov=prov_sim,
    windage=0.0,
    reference_time_utc=start_time,
)

exceeded = df[df["status_code"] == STATUS_TIME_EXCEEDED]
check("simulate_particles 產生 STATUS_TIME_EXCEEDED 記錄",
      len(exceeded) > 0,
      f"count={len(exceeded)}")

# 越界後不應再有 ACTIVE 記錄（粒子已停用）
last_step = df["step_index"].max()
active_at_last = df[(df["step_index"] == last_step) & (df["status_code"] == STATUS_ACTIVE)]
check("越界後粒子不再 ACTIVE（已停用）",
      len(active_at_last) == 0,
      f"active_at_last={len(active_at_last)}")

# 越界粒子的座標必須是 NaN（不得保留凍結幀的偽漂流座標）
exceeded_coords = exceeded[["lon", "lat"]].to_numpy()
check("越界粒子座標為 NaN（無偽漂流座標）",
      np.isnan(exceeded_coords).all(),
      f"finite_count={np.isfinite(exceeded_coords).sum()}")

# ------------------------------------------------------------
# 5. 位移單調性：回溯越久，淨位移不應減少（核心症狀回歸測試）
# ------------------------------------------------------------
def run_displacement(days):
    steps = int(days * 24)
    d, _, _, _, _ = simulate_particles(
        release_sites=[("test", 121.0, 23.0)],
        total_particles=10,
        total_steps=steps,
        dt_mins=60.0,
        prov=prov_sim,
        windage=0.0,
        reference_time_utc=ref,
    )
    # 取每顆粒子最後有效位置
    valid = d[np.isfinite(d["lon"]) & np.isfinite(d["lat"])]
    if valid.empty:
        return 0.0
    last = valid.sort_values(["particle_id", "step_index"]).groupby("particle_id").tail(1)
    start = d[d["step_index"] == 0]
    return float(np.hypot(last["lon"].mean() - start["lon"].mean(),
                          last["lat"].mean() - start["lat"].mean()))

# 在「資料充足」情境下（offset=0，可回溯 744h），位移應隨天數單調不減
prov_full = {
    "curr": dict(field, bbox=(119.0, 123.0, 21.0, 25.0),
                 landmask=np.zeros((NY, NX), dtype=bool),
                 land_geometry=None,
                 time_start_utc=time_start,
                 time_end_utc=time_end),
    "wind": None,
    "reference_time_utc": time_end,
}
prov_sim_backup = prov_sim
prov_sim = prov_full
ref_full = time_end

d18 = run_displacement(18)
d20 = run_displacement(20)
check("資料充足時：20 天位移 >= 18 天位移（單調不減）",
      d20 >= d18 - 1e-9,
      f"18d={d18:.6f}, 20d={d20:.6f}")

prov_sim = prov_sim_backup

# ------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("失敗項目：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通過 ✅")
