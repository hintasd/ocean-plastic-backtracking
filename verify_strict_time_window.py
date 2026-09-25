"""驗證「時間視窗嚴格檢查」：越界直接報錯，不再自動平移。

背景：
    舊版 UI 在回溯區間超出資料範圍時會「自動平移」整個區間，
    導致使用者選的起算時刻與實際模擬的起算時刻不一致，
    且平移後仍可能取到凍結幀造成偽漂流。
    新版改為：越界直接 st.error + st.stop，絕不自動調整。

本腳本以純函式重現 UI 的三道檢查邏輯，驗證：
    1. 正常區間 → 通過
    2. 回溯天數 > 資料總長度 → 報錯
    3. 起算時刻 > 資料末端 → 報錯
    4. 回溯區間起點 < 資料開端（時間越界）→ 報錯
    5. 邊界值（恰好等於資料開端/末端）→ 通過（不誤殺）

執行：
    $env:PYTHONIOENCODING="utf-8"; E:\\c\\conda\\python.exe verify_strict_time_window.py
"""
import sys
from datetime import datetime, timezone, timedelta

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def evaluate_window(reference_time_utc, days, available_start, available_end):
    """重現 ck.py UI 的嚴格時間視窗檢查。

    回傳 (ok, error_kind)：
        ok=True  → 通過，error_kind=None
        ok=False → 報錯，error_kind ∈ {"range_invalid", "too_long", "after_end", "before_start"}
    """
    requested_days = float(days)
    requested_hours = requested_days * 24.0
    available_hours = (available_end - available_start).total_seconds() / 3600.0

    if available_hours <= 0:
        return False, "range_invalid"

    simulation_start = reference_time_utc - timedelta(hours=requested_hours)

    # 檢查一：回溯天數超過資料總長度
    if requested_hours > available_hours + 1e-9:
        return False, "too_long"

    # 檢查二：起算時刻超出資料末端
    if reference_time_utc > available_end + timedelta(seconds=1):
        return False, "after_end"

    # 檢查三：回溯區間起點早於資料開端
    if simulation_start < available_start - timedelta(seconds=1):
        return False, "before_start"

    return True, None


# ------------------------------------------------------------
# 測試資料：模擬 POM 745 幀 @1h
# ------------------------------------------------------------
available_start = datetime(2025, 12, 1, 0, 0, tzinfo=timezone.utc)
available_end = datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)
# 資料總長度 = 30 天 18 小時 = 738 小時

# 1. 正常區間：起算 12/31 18:00，回溯 18 天（432h）→ 起點 12/13 18:00，在範圍內
ok, kind = evaluate_window(available_end, 18, available_start, available_end)
check("正常區間（起算=末端，回溯 18 天）→ 通過", ok, f"kind={kind}")

# 2. 回溯天數超過資料總長度：回溯 40 天 > 30.75 天
ok, kind = evaluate_window(available_end, 40, available_start, available_end)
check("回溯 40 天 > 資料總長度 → 報錯 too_long", (not ok) and kind == "too_long", f"kind={kind}")

# 3. 起算時刻超出資料末端
ok, kind = evaluate_window(available_end + timedelta(hours=5), 5, available_start, available_end)
check("起算時刻超出資料末端 → 報錯 after_end", (not ok) and kind == "after_end", f"kind={kind}")

# 4. 時間越界（原始 bug 場景）：起算 12/01，回溯 20 天 → 起點 11/11，早於資料開端
start_1201 = datetime(2025, 12, 1, 0, 0, tzinfo=timezone.utc)
ok, kind = evaluate_window(start_1201, 20, available_start, available_end)
check("原始 bug 場景（起算 12/01 回溯 20 天）→ 報錯 before_start",
      (not ok) and kind == "before_start", f"kind={kind}")

# 5. 同一場景回溯 18 天 → 起點 11/13，仍早於 12/01 → 也應報錯
ok, kind = evaluate_window(start_1201, 18, available_start, available_end)
check("起算 12/01 回溯 18 天（起點 11/13）→ 報錯 before_start",
      (not ok) and kind == "before_start", f"kind={kind}")

# 6. 邊界值：起算 12/31 18:00，回溯恰好 30.75 天（738h）→ 起點 = 資料開端 → 通過
ok, kind = evaluate_window(available_end, 738 / 24.0, available_start, available_end)
check("邊界值（回溯恰好 738h，起點=資料開端）→ 通過（不誤殺）", ok, f"kind={kind}")

# 7. 邊界值：起算時刻恰好 = 資料末端 → 通過
ok, kind = evaluate_window(available_end, 1, available_start, available_end)
check("邊界值（起算=資料末端，回溯 1 天）→ 通過", ok, f"kind={kind}")

# 8. 起算 12/20，回溯 18 天 → 起點 12/02，在範圍內 → 通過
start_1220 = datetime(2025, 12, 20, 0, 0, tzinfo=timezone.utc)
ok, kind = evaluate_window(start_1220, 18, available_start, available_end)
check("起算 12/20 回溯 18 天（起點 12/02）→ 通過", ok, f"kind={kind}")

# 9. 起算 12/20，回溯 20 天 → 起點 11/30，早於 12/01 → 報錯
ok, kind = evaluate_window(start_1220, 20, available_start, available_end)
check("起算 12/20 回溯 20 天（起點 11/30）→ 報錯 before_start",
      (not ok) and kind == "before_start", f"kind={kind}")

# 10. 資料範圍無效
ok, kind = evaluate_window(available_end, 5, available_end, available_start)
check("資料範圍無效（起點晚於終點）→ 報錯 range_invalid",
      (not ok) and kind == "range_invalid", f"kind={kind}")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("失敗項目：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("全部通過 ✅")
