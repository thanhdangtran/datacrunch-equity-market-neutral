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

---

## 2026-07-28 — `crunch test` + SUBMISSION #1 (slot `secure-ladybug`)

**`crunch test` pass, determinism check pass.** 1 phút 33 giây, tiêu thụ 1,41 GB.

Hai bài học vận hành:

1. **`crunch test` với data `default` làm cạn RAM.** Harness của platform nạp `X_train`
   thành DataFrame đầy đủ (1,64M × 1150 float32 = 7,5 GB) *trước khi* gọi vào `train()`,
   nên việc `train()` của mình tiết kiệm bộ nhớ không cứu được. RAM về 0, phải kill.
   → **Cách làm đúng: test hợp đồng platform trên biến thể `small`** trong một bản sao
   workspace ở scratchpad; đo số liệu thì dùng harness riêng trên memmap int8.

2. **Biến thể `small` chỉ có 130 moons (652–781), 251k dòng** — bằng 1/6 bản `default`
   (781 moons, 1,64M dòng). Đừng đọc điểm từ `crunch test` như điểm thật.

Điểm trên 9 moons của `crunch test` (773–781): mean IC **0.0152**. Nhìn thì thấp, nhưng:

| Train range | n dòng | mean IC trên moons 773–781 |
|---|---|---|
| 652–772 (giống `small`) | 234k | 0.0197 |
| 1–772 (đầy đủ) | 1.62M | **0.0226** |

→ Đoạn 773–781 **vốn là đoạn yếu** — ngay cả model train trên toàn bộ dữ liệu cũng chỉ
đạt 0.0226 ở đó so với 0.0329 trung bình. Và 9 moons có sai số chuẩn ~0.007 nên không
kết luận được gì. **Không phải suy giảm.**
→ Đồng thời xác nhận **nhiều dữ liệu vẫn hơn** (0.0226 > 0.0197), nên cloud thấy toàn bộ
dataset sẽ tốt hơn local.

**Đã push submission #1 vào slot `secure-ladybug`** (Model A).
Trước khi push đã xoá `resources/model.npz` — đó là artifact từ smoke test chỉ train trên
41 moons, gửi lên sẽ có nguy cơ platform dùng lại thay vì train mới.
`requirements.txt` để trống version (chỉ `numpy`, `pandas`) cho tương thích môi trường cloud.

## 2026-07-28 — `exp_lgbm_leaves7`

| family | mean_ic | ic_std | sharpe | hit | last104 |
|---|---|---|---|---|---|
| walk_forward | 0.0400 | 0.0401 | 1.00 | 0.84 | 0.0375 |
| gap311 | 0.0260 | 0.0333 | 0.78 | 0.76 | 0.0260 |

**Config:** `{"model": "lgbm", "stride": 4, "n_estimators": 800, "num_leaves": 7, "lambda_l2": 50.0, "max_bin": 7, "feature_fraction": 0.15}`

**Kết luận:** Hạ num_leaves 15->7 (một thay đổi duy nhất) để kiểm tra chẩn đoán 5.5 #2: ic_std 0.0400 vượt xa sàn nhiễu 0.022 => nghi thừa capacity.

**SỬA KẾT LUẬN cho `exp_lgbm_leaves7`** (dòng tự động ở trên ghi giả thuyết, không phải kết quả):

| | mean_ic | **ic_std** | Sharpe | hit |
|---|---|---|---|---|
| `num_leaves=15` | 0.0429 | **0.0400** | 1.07 | 0.86 |
| `num_leaves=7` | 0.0400 | **0.0401** | 1.00 | 0.84 |

**Giả thuyết SAI.** Giảm một nửa số lá làm mean IC giảm nhưng phương sai **y nguyên**
(0.0400 → 0.0401). Chẩn đoán §5.5 #2 "thừa capacity" **không giải thích được** `ic_std` cao
của LightGBM. Đừng lặp lại hướng này — hạ `num_leaves` chỉ mất tín hiệu, không được gì.

**Giả thuyết thay thế đang test:** prediction của GBM có đuôi dày, vài mã nhận giá trị cực đoan
nên Pearson bị chi phối bởi một nhúm tên; ridge thì cho phân phối gần Gauss. Nếu đúng,
rank-transform *prediction* theo moon (đơn điệu nhưng **không** affine, nên **có** đổi Pearson)
sẽ hạ phương sai. → `exp_lgbm_rankpred`.

Lưu ý phân biệt: rank *prediction* là siêu tham số hợp lệ; rank *target* đã chứng minh có hại.

---

## 2026-07-28 — Cloud run KHÔNG tự động

`crunch push` chỉ **upload**, không kích hoạt chạy. Bằng chứng:
- `crunch runner` chỉ có `cloud` / `cloud-executor` (đều ghi *"do not directly run!"* — platform
  tự gọi) và `local`. **Không có lệnh CLI nào tạo cloud run.**
- Quickstarter: *"1. Download Notebook 2. Upload 3. **Create a run to validate it**"*.
- `datacrunch-2.md:90`: *"If the **run completes** before Sunday 12pm UTC, it will be taken into
  account for the week. Otherwise, the run will be terminated and ignored."*

→ **Phải vào dashboard ấn Run thủ công.** Deadline tính theo lúc run **xong**, không phải
lúc submit — mà cloud train lại trên toàn bộ dataset nên đừng để sát Chủ nhật 12:00 UTC.

## 2026-07-28 — `exp_lgbm_rankpred`

| family | mean_ic | ic_std | sharpe | hit | last104 |
|---|---|---|---|---|---|
| walk_forward | 0.0262 | 0.0298 | 0.88 | 0.81 | 0.0228 |
| gap311 | 0.0203 | 0.0262 | 0.78 | 0.83 | 0.0203 |

**Config:** `{"model": "lgbm", "stride": 4, "n_estimators": 800, "num_leaves": 15, "lambda_l2": 50.0, "max_bin": 7, "feature_fraction": 0.15, "rank_pred": true}`

**Kết luận:** PLACEHOLDER

**KẾT QUẢ `exp_lgbm_rankpred`** — giả thuyết đuôi dày **đúng về cơ chế, sai về hướng đi**:

| | mean_ic | ic_std | Sharpe |
|---|---|---|---|
| gốc (`num_leaves=15`) | **0.0429** | 0.0400 | **1.07** |
| rank prediction theo moon | 0.0262 | **0.0298** | 0.88 |
| gap311 gốc | **0.0267** | 0.0335 | **0.80** |
| gap311 rank prediction | 0.0203 | 0.0262 | 0.78 |

Rank-transform **đúng là hạ phương sai mạnh** (0.0400 → 0.0298), xác nhận phương sai đến từ
một nhúm prediction cực đoan. **Nhưng nó giết mean IC còn mạnh hơn** (−39%), nên Sharpe tệ đi.

**Kết luận quan trọng:** *lợi thế của GBM nằm chính ở những prediction cực đoan.* Làm phẳng
chúng thì mất cả nhiễu lẫn tín hiệu. Điều này hợp lý với cấu trúc target: 88% bằng 0 và khối
lượng dồn ở ±1 — **tín hiệu thật chính là việc nhận ra mã nào sẽ đi cực đoan**, nên model nào
bắt được điều đó tất yếu có phương sai cao. Phương sai của Model B là **giá phải trả**, không
phải khuyết tật cần sửa.

**Hai hướng đã đóng cho Model B** (đừng thử lại):
1. Hạ `num_leaves` → mất mean IC, phương sai không đổi.
2. Rank-transform prediction → hạ phương sai nhưng mất mean IC nhiều hơn.

→ Giữ Model B ở cấu hình gốc, để nguyên phương sai cao, dùng slot riêng làm cửa cược độc lập
(§1.3: payoff `e**20` lồi ⇒ phương sai có giá trị khi không đánh đổi mean).
→ Quan sát này **củng cố tiền đề của Model D**: cấu trúc 88% zeros là chỗ có tín hiệu thật,
và mô hình hoá thẳng nó (hai tầng) đáng giá hơn là làm mượt prediction.

---

## 2026-07-28 — 🔴 BẪY LỚN: `crunch push` tự pin version theo máy local

Submission #1 **fail ngay ở bước cài đặt** trên cloud:

```
ERROR: Could not find a version that satisfies the requirement numpy==2.4.2
       (from versions: ..., 2.2.5, 2.2.6)
ERROR: Ignored the following versions that require a different python version:
       2.3.0 Requires-Python >=3.11; ... 2.5.1 Requires-Python >=3.12
```

**Nguyên nhân:** `crunch push` **mặc định chạy pip freeze** và ghi đè `requirements.txt`
bằng version của máy local, rồi đổi tên file gốc thành `requirements.original.txt`.
File local của mình để trống version (`numpy\npandas`, 13 byte) nhưng gói gửi lên là
60 byte có pin `numpy==2.4.2`, `pandas==3.0.0`.

**Môi trường cloud: Python 3.10** (suy ra từ log — mọi bản numpy cần Python ≥3.11 đều bị
bỏ qua, cao nhất khả dụng là **numpy 2.2.6**). Máy local đang Python **3.14.6** + numpy 2.4.2
→ pin local **không bao giờ** cài được trên cloud.

**Cách sửa: luôn push với `--no-pip-freeze`.**

```bash
crunch push --no-pip-freeze -m "..."
```

Kiểm tra bằng `crunch push --dry --no-pip-freeze` — phải thấy
`found code file: requirements.txt (13 bytes)` chứ **không** thấy dòng `froze file:` hay
`rename original file:`.

⚠️ **Áp dụng cho cả 4 slot.** Đây là bẫy im lặng: `--dry` thường vẫn báo thành công, và
`crunch test` local hoàn toàn không đụng tới requirements nên không bắt được lỗi này.

⚠️ **Hệ quả cho Model C:** cloud là Python 3.10, nên torch bản mới có thể không có. Phải
kiểm tra ràng buộc version của cloud trước khi thiết kế Model C phụ thuộc torch, hoặc viết
MLP + gradient của Pearson loss bằng numpy thuần.

→ Đã push lại thành **submission #2** với requirements không pin.

---

## 2026-07-28 — Môi trường cloud thật (từ log run submission #2)

Chạy được sau khi bỏ pin version. Ghi lại để không phải đoán nữa:

| Mục | Giá trị |
|---|---|
| Runtime chọn | **Powerful** — 16 core CPU, **không GPU**, **120 GB RAM**, 80 GB storage, AWS Fargate |
| Python | **3.10** |
| numpy | **1.26.4** (cài sẵn) |
| pandas | **2.3.3** (cài sẵn) |
| crunch-cli trên cloud | **11.7.0** (local đang 11.8.0) |
| Quota hiển thị | **15h29m46s** |
| Cờ thực thi | `--gpu false` |

**Data trên cloud là `X.parquet` / `y.parquet`** (không phải `.reduced.`), tải từ cùng
`data-releases/210`. Kích thước **725 MB** so với `X.reduced.parquet` 718 MB — chênh rất ít,
nên đừng giả định cloud có nhiều dữ liệu hơn hẳn cho tới khi đo trong `train()`.

### 🔴 SỬA LẠI ước tính `train_frequency`

Log ghi `looping moon=782 train=True (1/9)` → **vòng lặp OOS công khai chỉ có 9 moons
(782–790)**, không phải ~310 như tôi ước tính từ `live_to_predict = 1092`.

→ Ước tính cũ ("`train_frequency=1` tốn ~8,6h") **sai hoàn toàn**. Với 9 vòng lặp thì kể cả
train mỗi moon cũng chỉ là 9 lần train.
→ `train_frequency` gần như **không phải ràng buộc ngân sách** ở run OOS công khai.
→ Vẫn chưa biết run chấm điểm hằng tuần (moon 1092) lặp bao nhiêu vòng. **Cần xem log của
run đó rồi mới kết luận.**

**Bài học:** đừng suy ngân sách compute từ khoảng cách moon trong `moons_split.json`;
đọc thẳng số `(i/N)` trong log của lần run đầu.

### Chốt `train_frequency = 26` (dùng cho mọi run về sau)

Vòng lặp OOS công khai có 9 moons, mà 26 > 9 → model **train đúng một lần tại moon đầu (782)**,
khớp log `train=True (1/9)`.

So với `train_frequency=1`: train tại moon 782 dùng dữ liệu tới ~778, train tại moon 790 dùng
tới ~786 — chênh **8 moons trên tổng ~780**, không đáng kể. Nên gợi ý "để 1–2" là thừa.

**26 bền hơn cả 1 lẫn 0:**
- Vòng lặp ngắn → tự thành train-một-lần, không tốn gì.
- Vòng lặp dài (run chấm điểm tuần, nếu quả thật dài) → tự chặn chi phí compute.
- Không phải đổi tham số theo từng loại run.

Cấu hình UI đang dùng: **Train frequency = 26**, **Force first train = Yes**.
Force first train **bắt buộc phải Yes** vì ta không gửi model artifact kèm (đã xoá
`resources/model.npz` trước khi push) — UI ghi *"Submission without a model must be trained
at least once"*. Còn *"0 means never calling the train method"*.

---

## 2026-07-28 — ĐIỂM THẬT từ hub (qua `src/fetch_scores.py`)

Đọc được bằng **API key tài khoản** (`CRUNCH_API_KEY`). Lưu ý: project token trong workspace
**chỉ push được** — mọi endpoint đọc trả `Access Denied`. Điểm nằm ở `prediction.primaryMean`,
không phải ở `run`.

### ✅ Câu hỏi treo từ đầu dự án đã có lời giải: CHỈ CÓ MỘT LEADERBOARD

Endpoint metric (đọc công khai, không cần key) trả về **đúng 1 metric**:
`name=score, primary=true, rankOrder=DESCENDING, weight=100,
scorerFunction=CUSTOM__BROAD__SCORING, reducerFunction=NONE, cumulative=false`.

→ Câu *"for both metrics"* / *"leaderboards"* số nhiều trong `datacrunch-2.md` là chữ thừa.
→ **Pool là 1.000 USDC, không phải 2×.** Kỳ vọng thưởng bằng một nửa kịch bản lạc quan.
→ **Không có leaderboard Spearman** ⇒ rank-transform prediction không bắt buộc (và đã đo là có hại).
→ `reducerFunction=NONE`, `cumulative=false` ⇒ điểm là một giá trị đơn, không tích luỹ.

### Điểm OOS công khai (moons 782–790 = 2 tháng đầu 2020, đầu COVID)

| Slot | Prediction | Ngày | **Điểm** | Ghi chú |
|---|---|---|---|---|
| `secure-ladybug` | 78980 | 28/07 | **+0.01622** | **Model A — Ridge** |
| `fantastic-snipe` | 71815 | 24/06 | +0.02341 | submission cũ, tốt nhất |
| `fantastic-snipe` | 71798 | 24/06 | +0.00813 | submission cũ "first" |
| `fantastic-snipe` | 71972 | 26/06 | **−0.00367** | **submission #64783 — bản gauss-rank target** |

**Đọc số cho đúng:** 9 moons, sai số chuẩn ≈ `0.022/√9` ≈ **0.0073**.
- Model A (0.0162) vs bản cũ tốt nhất (0.0234): chênh **1 SE** → **không phân biệt được**. Đừng kết luận.
- Model A (0.0162) vs bản gauss-rank (−0.0037): chênh **~2,7 SE** → **có ý nghĩa**.

**Bản gauss-rank ăn điểm ÂM trên OOS thật.** Khớp đúng dự đoán từ phân tích số: rank hoá
target có 88% giá trị bằng 0 thì 88% dữ liệu bị train trên nhiễu thứ tự dòng. Không phải suy luận
suông nữa — nó đã lỗ thật trên out-of-sample.

⚠️ **Model đang chạy hằng tuần chính là bản gauss-rank này** (submission #5 = #64783).
5 run hằng tuần (29/06, 05/07, 12/07, 19/07, 26/07) đều `primaryMean = None` — **chưa có điểm**,
đúng với độ trễ 6 tuần đã ghi trong tài liệu.

### Model A chạy cloud thành công
Run 97650: `status=COMPLETED, success=true`. Run 97629 (submission #1, bản pin numpy) `success=false`.
