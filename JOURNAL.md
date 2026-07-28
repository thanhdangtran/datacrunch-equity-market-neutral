# Nhật ký thí nghiệm — DataCrunch #2

> Mỗi lần chạy có điểm số đều ghi vào đây (tự động qua `src/explog.py`) và vào
> `reports/leaderboard.csv`. CSV để sắp xếp; file này để ghi **kết luận** — thứ mà
> con số đứng một mình không bao giờ nói ra.
>
> Chiến lược ở [PLAN.md](PLAN.md). Cơ chế vận hành ở [SETUP.md](SETUP.md).

---

## Bối cảnh bài toán — đọc cái này trước

Ba điều quyết định mọi lựa chọn thiết kế, và cả ba đều không hiển nhiên:

1. **Thang điểm là Pearson trên đúng MỘT moon mỗi tuần** (~2.100 mã). Sàn nhiễu của
   một tuần là `1/√2096 ≈ 0.022`, trong khi tín hiệu tốt nhất đo được chỉ ~0.03.
   **Nhiễu tuần lớn gần bằng toàn bộ tín hiệu** → không bao giờ kết luận từ một moon,
   và `ic_std` là thông tin quan trọng ngang `mean_ic`.

2. **Thưởng theo `e**20`** → chỉ top ~3–5% có tiền, top 25% nhận 0. Payoff cực lồi,
   sàn bằng 0 ⇒ **phương sai có giá trị** nếu không đánh đổi bằng mean. Đây là lý do
   chạy 4 model khác nhau về *cấu trúc* thay vì 4 seed của cùng một model.

3. **Data local dừng ở moon 781, moon được chấm là 1092** — cách nhau 311 moons (~6 năm).
   Mọi model phải đo bằng `gap_folds` (train → bỏ 311 moons → validate), không chỉ
   walk-forward thường. Chỉ số này đã bắt được điểm yếu của Model B ngay lần đầu.

Ràng buộc kỹ thuật đáng nhớ:
- Features có **đúng 7 mức** → int8 lossless (đã kiểm: sai số **chính xác 0.00**), `max_bin=7`.
- `id` **không nối được** giữa các moon → không có feature time-series theo mã. Cross-sectional thuần.
- Target **88,2% bằng đúng 0** → **không được rank-transform target** (xem mục 2026-07-28 bên dưới).
- Máy chỉ còn ~3,5 GB RAM trong khi ma trận đầy đủ là 7,5 GB → mọi thứ chạy qua memmap int8.

---

## 2026-07-28 — Khảo sát dữ liệu (`src/survey.py`)

IC per-moon của cả 1150 features trên toàn bộ 781 moons.

| Baseline | mean IC | Sharpe | hit | last104 |
|---|---|---|---|---|
| `−Feature_43` (một feature) | 0.0232 | 1.13 | 86,6% | 0.0223 |
| Trung bình 20 feature mạnh nhất | 0.0238 | 1.16 | 86,6% | 0.0211 |
| **Trung bình 100 feature mạnh nhất** | **0.0251** | **1.19** | 88,7% | 0.0242 |

**Kết luận:** Champion khởi điểm = `0.0251 / Sharpe 1.19`. Một feature duy nhất đảo dấu
đã cho Sharpe 1.13 — bất kỳ model nào không beat được ngưỡng này là model sai.

**Hai giả định bị bác bỏ:**
- ~~"Redundancy khổng lồ, nén xuống 150 chiều"~~ → **SAI**. Trung vị `|corr|` giữa các cặp
  feature chỉ **0.021**; chỉ 0,09% số cặp có `|r|>0.95`; cần **258 PC** cho 90% phương sai.
  Feature set **không** dư thừa → đã sửa lại thiết kế Model A.
- ~~Top Mutual Information trong EDA có sẵn~~ → **nhiễu**. Không một feature nào trong
  danh sách đó xuất hiện ở top IC per-moon tính trên toàn bộ data.

**Tin tốt:** top-20 feature chọn ở nửa đầu lịch sử giữ nguyên IC ở nửa sau
(−0.01364 → −0.01370). Tín hiệu cốt lõi có tính dừng.

---

## 2026-07-28 — Rank-transform target là có hại (kiểm chứng bằng số)

Submission cũ #64783 dùng `gauss_rank` lên target. Đo trên moon 700 (87,6% target = 0):

- Nhóm zero bị trải trên **1,75 đơn vị** dải giá trị dù target thật **giống hệt nhau**
- Thứ hạng gán cho chúng có **`corr = +0.72` với vị trí dòng trong dataframe**
- **67% phương sai** của target-đã-rank đến từ nhóm dòng đồng nhất đó

**Kết luận:** rank hoá target = train trên nhiễu thứ tự dòng cho 88% dữ liệu.
Dùng **L2 trên target thô**; Pearson được tối đa hoá bởi `E[y|x]` mà L2 ước lượng đúng nó.
Rank-transform *prediction* thì khác — đó là siêu tham số cần test, không phải điều cấm.

---

## 2026-07-28 — `exp_ridge` · Model A · Ridge 1150 features

Quét alpha, 6 walk-forward fold + 3 gap311 fold. Fit bằng streaming Gram (X'X là 1150×1150
bất kể bao nhiêu dòng) nên không bao giờ dựng ma trận 7 GB.

| alpha | mean_ic | Sharpe |
|---|---|---|
| 1e5 | 0.0312 | 1.16 |
| **1e6** | **0.0329** | **1.25** |
| 1e7 | 0.0309 | 1.21 |
| 1e8 | 0.0253 | 1.08 |
| 1e10 | 0.0201 | 0.94 |

**Chọn `alpha = 1e6`**: walk-forward `0.0329 / 1.25`, hit 90%, **6/6 fold dương**;
gap311 `0.0304 / 1.31`, 3/3 fold dương. Vượt champion `0.0251 / 1.19`.

**Kết luận 1:** đỉnh alpha cực lớn (1e6) xác nhận tín hiệu yếu và rải rác trên 1150 features
— cần shrink rất mạnh **nhưng không được cắt bỏ feature**.

**Kết luận 2 (quan trọng nhất tới giờ):** khoảng trống 311 moons **không nghiêm trọng**.
Chỉ mất 8% mean IC và Sharpe còn cao hơn. Cụ thể: fold train moons **1–314** validate 626–677
đạt **+0.0297**, còn fold train moons **1–621** (nhiều hơn 307 moons, gần validation hơn hẳn)
validate cùng đoạn chỉ đạt **+0.0291** — **dữ liệu cũ 6 năm tốt ngang dữ liệu mới**.
→ Hạ rủi ro §2.7 từ "nghiêm trọng" xuống "trung bình"; **rút lại** luận điểm "recency weighting
là trọng tâm" vì dữ liệu không ủng hộ.

---

## 2026-07-28 — `exp_lgbm` · Model B · LightGBM `max_bin=7`

`num_leaves=15, max_depth=5, lr=0.02, feature_fraction=0.15, lambda_l2=50, 800 cây`,
lấy mẫu mỗi 4 moon (giữ nguyên vẹn cross-section — không bao giờ bagging theo dòng).

| family | mean_ic | ic_std | Sharpe | hit |
|---|---|---|---|---|
| walk_forward (3 fold) | **0.0429** | 0.0400 | 1.07 | 0.86 |
| gap311 (2 fold) | 0.0267 | 0.0335 | 0.80 | 0.79 |

So trên **cùng fold** với Ridge:

| Fold | Ridge | LightGBM |
|---|---|---|
| wf4 | +0.0291 | **+0.0470** |
| wf5 | +0.0451 | +0.0459 |
| wf6 | +0.0273 | **+0.0359** |
| gap311_2 | **+0.0390** | +0.0288 |
| gap311_3 | **+0.0225** | +0.0245 |

**Kết luận:** B có mean IC cao hơn 30% nhưng phương sai gấp rưỡi → **Sharpe thấp hơn A**.
`ic_std = 0.0400` trong khi sàn nhiễu chỉ ~0.022 và Ridge đạt 0.0264 → **thừa capacity**.
Và B suy giảm **−38%** qua gap311 so với Ridge chỉ **−8%** → lợi thế phi tuyến của B
**phụ thuộc regime**, không sống sót qua 6 năm.

**Quyết định:** B **không** qua promotion gate, **không** được thay A. Nhưng B decorrelated
về cấu trúc với A nên vẫn giữ slot riêng làm cửa cược độc lập (§1.3: payoff lồi ⇒ phương sai
có giá trị). Việc tiếp theo cho B: **hạ capacity**, mỗi lần một thay đổi.
