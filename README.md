# 海洋塑膠垃圾溯源分析系統 (Ocean Plastic Source Backtracking)

以 **拉格朗日逆向數值積分 (Lagrangian Reverse Integration)** 為核心的海洋塑膠垃圾溯源系統。
結合 POM 三維海流模式與 WRF 大氣風場，透過 **RK4 四階龍格-庫塔法** 將海漂垃圾的觀測位置
沿時間軸反向推演，判定其可能的來源（本地源 vs 境外源），並提供不確定性量化與參數敏感度分析。

---

## 目錄

- [核心功能](#核心功能)
- [科學方法](#科學方法)
- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [使用說明](#使用說明)
- [驗證腳本](#驗證腳本)
- [專案結構](#專案結構)

---

## 核心功能

| 功能 | 說明 |
|------|------|
| **反向軌跡推演** | 從海岸熱點釋放粒子，沿時間軸反向積分，重建漂流路徑 |
| **來源判定** | 依「本地源關聯半徑」判定粒子最終是否滯留於本地領海（預設 22.2 km = 12 浬） |
| **不確定性量化** | Bootstrap 重抽樣估計「本地源佔比」的 95% 信賴區間與統計顯著性 |
| **參數敏感度分析** | 對單一參數掃描一組值，觀察結論是否穩健（曲線是否跨越 50%） |
| **全台熱點同步反演** | 一次對全部 12 個海岸熱點釋放粒子，比較全台各地的來源傾向 |
| **互動式視覺化** | pydeck 動態軌跡圖層、來源密度熱圖、plotly 統計圖表 |

---

## 科學方法

### 1. 漂流速度合成

粒子在海面的運動速度由海流與風生漂流合成：

$$
\vec{V}_{\text{total}} = \vec{V}_{\text{current}} + w \cdot \vec{V}_{\text{wind}}
$$

其中 $w$ 為**風阻係數 (windage)**，代表物體受風影響的比例（例如寶特瓶約 1.5%）。

### 2. RK4 逆向積分

以四階龍格-庫塔法對位置進行時間反向積分：

$$
\frac{d\vec{x}}{dt} = -\vec{V}_{\text{total}}(\vec{x}, t)
$$

每一步計算 4 個中間點（$k_1, k_2, k_3, k_4$），並檢查**終點與所有中間點**是否觸陸，
以避免大步長跨越狹窄陸地（海岬、沙洲）造成的漏判。

### 3. 來源判定

粒子回溯至起算時刻後，計算其與釋放熱點的距離：

- **本地源 (Local Source)**：最終位置落在關聯半徑內
- **境外源 (External Source)**：最終位置超出關聯半徑

### 4. 不確定性量化

以 Bootstrap 重抽樣（$n_{\text{boot}} = 2000$）估計本地源佔比的 95% 信賴區間：

- **顯著 (Significant)**：CI 完全落在 50% 同一側
- **不顯著 (Not Significant)**：CI 跨越 50%

---

## 系統架構

```mermaid
flowchart TD
    A[POM 海流資料<br/>us/vs] --> C[速度場提供者<br/>build_velocity_providers]
    B[WRF 風場資料<br/>u10/v10] --> C
    C --> D[雙線性時空插值<br/>get_velocity_interpolated]
    D --> E[RK4 逆向積分<br/>rk4_step_2d]
    E --> F{觸陸判定<br/>終點 + 4 中間點}
    F -->|觸陸| G[STATUS_BEACHED]
    F -->|越界| H[STATUS_OUT_OF_BOUNDS]
    F -->|逾時| I[STATUS_TIME_EXCEEDED]
    F -->|繼續| E
    G --> J[軌跡資料集]
    H --> J
    I --> J
    J --> K[來源判定<br/>compute_replay_metrics]
    J --> L[不確定性量化<br/>bootstrap_proportion_ci]
    J --> M[敏感度分析<br/>run_sensitivity_sweep]
    K --> N[Streamlit UI]
    L --> N
    M --> N
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 準備資料

本專案需要 POM 海流與 WRF 風場資料。請執行 `prepare_local_data.py` 從原始資料庫
萃取精簡資料集（約 1.15 GB）：

```bash
python prepare_local_data.py
```

或設定環境變數指向原始資料庫：

```powershell
$env:PLASTIC_DATA_ROOT = "F:\20260907_塑源"
```

### 3. 啟動應用

```bash
streamlit run ck.py
```

瀏覽器會自動開啟 `http://localhost:8501`。

---

## 使用說明

### 側邊欄參數

| 區塊 | 參數 | 說明 |
|------|------|------|
| **1. 追溯起始熱點** | 選擇目標海岸熱點 | 12 個預設熱點、🌏 全部熱點、或自訂座標 |
| **2. 時間維度** | 回溯模擬天數 | 1~30 天 |
| | 數值積分步長 | 30 min ~ 3 hr |
| **3. 本地源關聯半徑** | 判定半徑 | 10~50 km（預設 22.2 km） |
| **4. 漂流動力學** | 模擬粒子總數 | 50~1000 |
| | 海廢材質特性 | 寶特瓶 / 漁網 / 保麗龍等預設 |
| | 海面風阻係數 | 0~5% |
| **5. 起始時間** | 回溯起算日期 | 須落在資料涵蓋範圍內 |

### 三個分析分頁

1. **溯源推演 (Trajectory & Heatmap)** — 動態軌跡圖層、來源密度熱圖、核心 KPI
2. **群體統計分析 (Ensemble Analytics)** — 漂流里程分佈、粒子最終歸宿分佈
3. **敏感度分析 (Sensitivity Analysis)** — 參數掃描曲線、穩健性判定

---

## 驗證腳本

本專案以「可證偽」的測試逐層驗證每個環節：

| 腳本 | 驗證內容 | 結果 |
|------|----------|------|
| `verify_all_parameters.py` | 六層全參數驗證（資料層/時間索引/插值/RK4/軌跡） | 37/37 PASS |

執行方式：

```bash
python verify_all_parameters.py
```

---

## 專案結構

```
plastic/
├── ck.py                              # 主應用程式（Streamlit）
├── prepare_local_data.py              # 從原始資料萃取精簡資料集
├── verify_all_parameters.py           # 全參數驗證
├── requirements.txt                   # Python 依賴
├── README.md                          # 本文件
├── .gitignore                         # Git 忽略規則
│
└── slim_data/                         # 精簡資料集（.gitignore 排除）
    ├── pom_currents_slim.nc
    └── wrf_wind_slim.nc
```

---

## 授權

本專案為學術研究用途。資料來源為 NODASS（國家海洋資料庫及共享平台）。
