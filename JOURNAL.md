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

---

## 2026-07-28 — `participate.md` xác nhận / sửa mấy điểm

**1. Clone token: cửa sổ 3 phút.** Tài liệu ghi thẳng: *"The site generates new tokens every
minute, and each token can only be used once within a 3-minute timeframe."*
→ Giải thích chính xác tỷ lệ 4/10 token dùng được: 4 cái sống là 4 cái chạy trong ~1 phút.
→ Không phải "token hỏng", chỉ là hết hạn. Quy trình đúng: lấy 1 token → chạy trong 3 phút.

**2. Run có tính CHUỖI — model được kế thừa giữa các run.**
*"they are all linked together by using an iterative version of your model to continue the work
of the previous one"*, và `resources/` được *"persisted across runs"*.
→ Run 1 train → sinh model → Run 2 kế thừa model đó.
→ ⚠️ **Với `train_frequency=26` và vòng lặp ngắn, model có thể KHÔNG BAO GIỜ train lại** sau run
đầu — nó đóng băng ở dữ liệu tại thời điểm run 1. Theo đo đạc gap311 (mất 8% qua 6 năm) thì
trôi vài tháng là không đáng kể, nhưng **cần theo dõi**: nếu điểm tụt dần qua nhiều tuần thì
đây là nghi phạm đầu tiên, và cách sửa là push submission mới để reset chuỗi.

**3. Run cuối cùng thành công được chọn tự động.** *"If you do not select anything, the last
successful run will be selected."* → run 97650 của `secure-ladybug` đang là bản được chọn.

**4. `--no-pip-freeze` là cờ chính thức**, và *"Your original requirements will always be
preserved"* — khớp với những gì đã gặp.

**5. Run trên cloud phải bấm tay** — nút *"Run in the Cloud"* trên trang submission. Xác nhận
kết luận trước đó: `crunch push` không tự chạy.

**6. DataCrunch 2 không có thời gian ân hạn** — *"even terminate still running runs when the
Submission Phase ends, because they are too time sensitive"*.

---

## 2026-07-28 — Model B đóng gói + push (`lovely-fowl` submission #1)

`crunch test --train-frequency 26` **pass, determinism check pass**.

⚠️ **Mặc định `crunch test --train-frequency` là 1** → train lại ở *cả 9 moon*. Với LightGBM
800 cây thì quá 10 phút và bị timeout. Luôn truyền `--train-frequency 26` khi test Model B/D.

### Xác nhận độc lập đặc tính A vs B

Điểm trên **cùng 9 moons (773–781)** qua harness của platform, data `small`, tập train khác hẳn
CV local:

| | mean IC | ic_std | Sharpe |
|---|---|---|---|
| Model A (ridge) | 0.0152 | — | — |
| Model B (lgbm) | **0.0248** | 0.0387 | 0.64 |

Tái hiện đúng đặc tính đo được ở CV local (B mean cao hơn, phương sai ~1,5×, Sharpe thấp hơn).
Hai đường đo hoàn toàn độc lập cho cùng một kết luận → đặc tính này là thật, không phải artefact
của harness riêng.

### Chi phí

`crunch test` Model B: **22 phút** trên data `small` (251k dòng), so với Model A **1m33s**.
Cloud có ~1,65M dòng ≈ 6,5× → ước tính train **~2 giờ**. Nằm trong quota 15h nhưng không tầm thường.
Nếu cần cắt: hạ `n_estimators` 800 → 500, nhưng như vậy là đổi model đã validate, phải đo lại.

### Dọn workspace trước khi push
`ws-fantastic-snipe/` còn `notebook.ipynb` của submission cũ và **nó bị gom vào gói** (`crunch push`
coi mọi file code là một phần submission). Đã chuyển ra scratchpad. **Kiểm tra `crunch push --dry`
trước mỗi lần push** — nó liệt kê đúng những file sẽ gửi.

---

## 2026-07-29 — Cả 2 run cloud thành công, điểm thật khớp dự đoán

### Model B chạy full data trên cloud — điểm nhảy vọt so với test small

| Nguồn | mean IC |
|---|---|
| Local CV (stride-4, đầy đủ moon range) | 0.0429 |
| `crunch test` small (251k dòng, 9 moons) | 0.0248 |
| **Cloud run thật (1,64M dòng, 9 moons)** | **0.0386** |

Cloud gần khớp CV local (0.0386 vs 0.0429) hơn hẳn `crunch test` nhỏ (0.0248). Xác nhận điều
đã suy đoán: `crunch test` trên data `small` **đánh giá thấp** model dùng nhiều dữ liệu, vì bản
thân nó chỉ có 251k dòng thay vì 1,64M. **Đừng dùng điểm `crunch test` để so sánh model — chỉ
dùng để xác nhận không lỗi.**

### Model A: determinism xác nhận qua 2 slot độc lập

`secure-ladybug` (78980) và `fantastic-snipe` (79099) chạy **cùng code, cùng data** →
cả hai ra đúng **+0.01622**, khớp tuyệt đối. Xác nhận `train()`/`infer()` không có nguồn
ngẫu nhiên bị rò rỉ (đúng như `deterministic=True` yêu cầu, dù ridge vốn không cần cờ đó).

### Bảng điểm cloud hiện tại (9 moons OOS công khai, 782–790, đầu COVID 2020)

| Slot | Model | Điểm |
|---|---|---|
| `lovely-fowl` | B — LightGBM | **+0.0386** |
| `secure-ladybug` | A — Ridge | +0.0162 |
| `fantastic-snipe` | A — Ridge (thay bản gauss-rank) | +0.0162 |
| `scornful-trout` | — | chưa có run |

**B đang dẫn điểm rõ rệt trên OOS thật** dù Sharpe local thấp hơn A. Nhất quán với thiết kế:
payoff `e**20` thưởng mean cao hơn khi không phải đánh đổi hoàn toàn, và 9 moons quá ít để
Sharpe khác biệt thể hiện ra theo hướng có lợi cho A.

### `fantastic-snipe` weekly run vẫn treo 5 prediction "pending"
29/06, 05/07, 12/07, 19/07, 26/07 — bản gauss-rank cũ, đúng độ trễ 6 tuần, sẽ resolve dần.
Submission #6 (Model A) mới thay hôm 28/07 nên chưa vào chu kỳ weekly nào.

---

## 2026-07-29 — Đọc bảng "Best Score" trên hub: KHÔNG đuổi theo

Người dùng gửi screenshot leaderboard: top 1 = **0.1535**, top 13 đều nằm 0.10–0.15,
cột "Best Score" / "Last subm. Score" / "Run success X/Y".

**API `comp.leaderboards.list()` vẫn `Access Denied` kể cả với API key tài khoản** — không
đọc trực tiếp được cấu trúc bảng. Suy luận từ dữ kiện gián tiếp:

Cột **"Run success X/Y"** khớp đúng cơ chế mình vừa trải qua — số lần Run cloud thành công/
thất bại, đúng loại prediction `primaryMean` mà `fetch_scores.py` đọc trên public OOS
(782–790, 9 moons). → Đây gần chắc là bảng xếp theo **Best Score trên public OOS**, không phải
điểm live hằng tuần.

**Đối chiếu độ lớn — bằng chứng overfit rõ ràng:**

| | mean IC |
|---|---|
| Feature đơn mạnh nhất, đo trên **toàn bộ 781 moons lịch sử** (`src/survey.py`) | 0.0232 |
| Blend 100 feature mạnh nhất, toàn lịch sử | 0.0251 |
| Model A/B trên cloud, 9 moons OOS | 0.0162 / 0.0386 |
| **Top 1 leaderboard "Best Score"** | **0.1535** |

Top 1 cao gấp **~6,6 lần** feature mạnh nhất đo trên 15 năm dữ liệu. Sai số chuẩn một moon
≈0,022, 9 moons → mean 0,15 lệch **~20 SE** khỏi 0. Không thể là may mắn trên một lần thử.

Nhưng đây là **Best Score** — max qua nhiều lần submit trên **đúng 9 moon cố định, công khai,
biết trước**. Công thức kinh điển của multiple-comparison overfitting: 1150 features × nhiều
lần thử × validation set nhỏ cố định = dễ khớp ngẫu nhiên lên 0,10+ mà không có kỹ năng thật.
`crunch push` không giới hạn số lần, điểm public OOS lộ ngay sau mỗi run → hillclimb trên
đúng 9 moon đó gần như tất yếu xảy ra trong sân chơi này.

**Kết luận, không đổi chiến lược:** không đuổi theo bảng này. Xác nhận bằng số cụ thể cho
cảnh báo đã có sẵn ở PLAN.md §1.4. Model ăn 0.15 trên 9 moon công khai nhiều khả năng sập về
gần 0 hoặc âm ở moon 1092 (chưa từng thấy, không peek được). Vị trí trên bảng này không dự báo
được kỳ vọng thưởng thật — tiếp tục dựa vào CV lịch sử + gap311 làm căn cứ duy nhất.

## 2026-07-29 — `exp_twostage`

| family | mean_ic | ic_std | sharpe | hit | last104 |
|---|---|---|---|---|---|
| walk_forward | 0.0479 | 0.0416 | 1.15 | 0.87 | 0.0452 |
| gap311 | 0.0375 | 0.0359 | 1.04 | 0.87 | 0.0375 |

**Config:** `{"model": "twostage", "stride": 4, "clf_estimators": 500, "reg_estimators": 500, "max_bin": 7}`

**Kết luận:** (điền tay sau khi xem số)

---

## 2026-07-29 — `exp_twostage` · Model D · Two-stage classifier + regressor

Stage 1: LightGBM binary `P(target != 0 | x)`. Stage 2: LightGBM regression chỉ trên
193.664 dòng `target != 0`. Kết hợp: `pred = P(nonzero) * E[target | nonzero]`.
Cùng fold, cùng stride với B (3 walk-forward + 2 gap311) để so trực tiếp.

| | mean_ic | ic_std | Sharpe | gap311 |
|---|---|---|---|---|
| A — Ridge | 0.0329 | 0.0264 | **1.25** | 0.0304 |
| B — LightGBM | 0.0429 | 0.0400 | 1.07 | 0.0267 |
| **D — Two-stage** | **0.0479** | 0.0416 | **1.15** | **0.0375** |

**D thắng trên mọi mặt trận cùng lúc — không phải trade-off.** Vượt B cả về mean IC (+12%)
lẫn Sharpe (1.15 > 1.07), và vượt cả A lẫn B về gap311 (0.0375 — chịu khoảng trống 311 moons
tốt nhất trong 3 model). Chi phí: 2 lần train LightGBM mỗi fold nên chậm hơn B (~100s/fold
walk-forward, ~70s/fold gap so với B's ~55s/~37s), nhưng không đáng kể so với quota 15h.

**Xác nhận giả thuyết đặt ra sau thí nghiệm rank-pred trên B** (JOURNAL 2026-07-28): tín hiệu
thật nằm ở việc nhận diện mã sẽ đi cực đoan, không nằm ở việc ước lượng chính xác giá trị khi
target = 0. Tách tường minh classification (có đi cực đoan không) khỏi regression (đi bao nhiêu,
chỉ fit trên 12% dữ liệu có tín hiệu) khai thác cấu trúc 88%-zero tốt hơn một regression đơn L2
trên toàn bộ target như A/B đang làm.

**Quyết định:** D là ứng viên mạnh nhất tới giờ. Cần: (1) `crunch test` xác nhận API + không lỗi,
(2) đóng gói `main.py` cho `scornful-trout`, (3) `--no-pip-freeze` khi push (bài học đã có).

**Đã push Model D vào `scornful-trout` — submission #1.**
Smoke test API: train 41 moons 28,2s, infer 11 moons mean IC 0,0426 (khớp tầm CV harness).
`--dry` xác nhận gói sạch (không dính `resources/` cũ như bài học ở `fantastic-snipe`).

**Cần bạn: ấn Run in the Cloud cho `scornful-trout`.**

---

## 2026-07-29 — Model D chạy cloud thành công + ma trận tương quan A/B/D

### Điểm cloud thật, cả 4 slot giờ đều có run thành công

| Slot | Model | Điểm cloud (9 moons OOS) |
|---|---|---|
| `scornful-trout` | **D — Two-stage** | **+0.04835** (cao nhất) |
| `lovely-fowl` | B — LightGBM | +0.03864 |
| `secure-ladybug` | A — Ridge | +0.01622 |
| `fantastic-snipe` | A — Ridge (tạm) | +0.01622 |

D dẫn đầu cả cloud lẫn local CV — nhất quán, không phải may mắn ngẫu nhiên trên 9 moons.

### Ma trận tương quan prediction (train ≤725, validate 730–781, cùng fold)

|  | A | B | D |
|---|---|---|---|
| A | — | 0.515 | 0.533 |
| B | 0.515 | — | **0.835** |
| D | 0.533 | 0.835 | — |

**corr(B,D) = 0.835 — cao, đúng như lo ngại.** Cả hai đều là LightGBM, D chỉ là B được tách
thêm một tầng phân loại. Theo gate ở PLAN §4 (`corr > 0.95` → không đóng góp gì), 0.835 **chưa
chạm ngưỡng loại bỏ** nhưng đã mất phần lớn giá trị decorrelation — không nên coi B và D là
hai "vé số" độc lập theo đúng nghĩa của §1.3.

A với B/D chỉ ~0.52 — **A là nguồn diversity thật sự duy nhất** trong 3 model đã có.

### Ensemble đều trọng số A+B+D (rank/scale rồi trung bình, cùng val window)

| | mean IC | Sharpe |
|---|---|---|
| A riêng | 0.0273 | — |
| B riêng | 0.0359 | — |
| **D riêng** | **0.0405** | — |
| Ensemble đều A+B+D | 0.0387 | 1.10 |

**Ensemble KHÔNG vượt được D một mình trên cửa sổ này.** Đúng như dự đoán từ corr(B,D)=0.835:
trộn D với B tương quan cao chỉ pha loãng D bằng một bản gần giống nhưng yếu hơn, còn A kéo mean
xuống dù có đóng góp diversity. Kết luận tạm thời: **đừng vội ensemble đều trọng số** — nếu làm,
phải hạ trọng số B (dư thừa so với D) hoặc bỏ hẳn, và cân trọng số theo Sharpe/IC thay vì đều nhau.
Cần đo lại trên nhiều fold trước khi quyết định dứt khoát (đây mới 1 cửa sổ).

### Quyết định: `fantastic-snipe` nên chạy D hoặc chờ C, không giữ A trùng lặp

`secure-ladybug` và `fantastic-snipe` hiện chạy **cùng một model A** — hai vé số giống hệt
nhau, lãng phí một slot. Ba lựa chọn:
1. Chuyển `fantastic-snipe` sang D (model tốt nhất) → nhưng D và B trùng 0,835, D và
   `scornful-trout` cũng trùng luôn (cùng code) → mất diversity kiểu khác.
2. Giữ A ở `fantastic-snipe` cho tới khi Model C (MLP IC-loss) xong — C là nguồn decorrelation
   thật sự vì khác cấu trúc loss, không chỉ khác vài tầng LightGBM.
3. Xây một biến thể D khác cấu hình (feature_fraction/seed khác) — decorrelation yếu, không
   đáng công.

**Chọn (2).** A hiện là nguồn diversity thật, giữ nó ở `fantastic-snipe` trong lúc xây Model C
thay vì đổi sang D chỉ để thắng con số ngắn hạn trên một cửa sổ 9 moons.

---

## 2026-08-03 — SỬA LẠI: bản gauss-rank KHÔNG BAO GIỜ chạy live hằng tuần

Kết luận ngày 2026-07-29 ("bản đang chạy hằng tuần chính là bản gauss-rank, submission #5 =
#64783") **sai**. Đào field `submission` trong từng `run` object (không chỉ `prediction`) của
`fantastic-snipe` qua API key tài khoản cho thấy:

| Run (crunch/round) | Ngày tạo prediction | `submission.number` | Điểm |
|---|---|---|---|
| 90339 (round 122→123) | 29/06 | **#4** (id 64435, "") | **+0.02350** ✅ đã resolve |
| 91790 (round 123) | 05/07 | #4 | pending |
| 93286 (round 124) | 12/07 | #4 | pending |
| 94892 (round 125) | 19/07 | #4 | pending |
| 97031 (round 126) | 26/07 | #4 | pending |
| 97793 (round 127) | — (test thủ công) | **#6** (Model A, thay gauss-rank) | +0.01622 |
| 100574 (round 127→) | 03/08 | #6 | pending |

**Submission #5 (gauss-rank, #64783) không xuất hiện trong bất kỳ run hằng tuần nào.** Chuỗi
weekly giữ nguyên **#4** (bản trước gauss-rank, `gen:88973`, điểm test thủ công +0.02341) suốt
5 tuần liên tiếp, rồi nhảy thẳng sang **#6** (Model A) khi nó được push 28/07. #5 chỉ có đúng một
lần chạy thủ công (run 89191, cho điểm public OOS −0.00367) và chưa bao giờ được chọn cho run
hằng tuần — có thể vì nền tảng chỉ tự chọn "last successful run" theo nghĩa run *thủ công* được
bấm, và #5 chưa từng được bấm "Run in the Cloud" như #4 đã từng.

**Hệ quả:** điểm live +0.02350 ngày 29/06 **không mâu thuẫn** với phân tích "rank-transform target
có hại" ở §3.2 — nó đo submission #4 (không rank-transform), không phải bản gauss-rank. Bản
gauss-rank chưa bao giờ bị chấm điểm thật ngoài phép đo thủ công đã có. Không có bằng chứng mới
nào cần xem lại kết luận §3.2.

**Bài học vận hành:** đọc `prediction.name` (`"out-of-sample crunch #1"`) không đủ để biết code
nào tạo ra nó — phải tra field `submission.number` trong `run` object tương ứng.

## 2026-08-03 — Xác nhận torch có wheel cho Python 3.10 (môi trường cloud)

Rủi ro treo từ 2026-07-28 ("cloud là Python 3.10, torch bản mới có thể không có") — đã đo trực
tiếp thay vì đoán: `pip download torch --python-version 310 --platform manylinux2014_x86_64
--only-binary=:all:` trả về **torch 2.6.0, wheel `cp310-cp310-manylinux1_x86_64`** thành công.
→ Không cần viết MLP thuần numpy để né phụ thuộc torch. Model C dùng torch bình thường, nhưng
`requirements.txt` của `ws-fantastic-snipe` phải ghim một version có wheel cho cp310 (không để
trống như numpy/pandas) và vẫn push bằng `--no-pip-freeze` để tránh bẫy đã gặp ở §"BẪY LỚN".

## 2026-08-03 — `exp_mlp_ic` · Model C · MLP tối ưu trực tiếp IC

`src/models/mlp_ic.py` + `src/run_mlp.py`. 3 lớp ẩn (256/128/64), dropout 0.3, AdamW
weight_decay 1e-3, batch = một cross-section moon, loss = `-Pearson(pred, target)` trực tiếp,
early stop trên 52 moon cuối của tập train (tách riêng khỏi val thật để không rò rỉ), stride 4
giống B/D để so cùng ngân sách fold.

| family | mean_ic | ic_std | sharpe | hit | last104 |
|---|---|---|---|---|---|
| walk_forward | 0.0333 | 0.0374 | **0.89** | 0.80 | 0.0363 |
| gap311 | 0.0155 | 0.0377 | **0.41** | 0.63 | 0.0155 |

**Config:** `{"model": "mlp_ic", "stride": 4, "hidden": [256, 128, 64], "dropout": 0.3, "weight_decay": 0.001, "epochs_max": 60, "patience": 8}`

So với 3 model đã có (cùng loại fold, không hoàn toàn cùng cửa sổ vì C chỉ chạy 3wf+2gap như B/D):

| | wf mean_ic | wf Sharpe | gap311 mean_ic | gap311 Sharpe |
|---|---|---|---|---|
| A — Ridge | 0.0329 | **1.25** | 0.0304 | **1.31** |
| B — LightGBM | 0.0429 | 1.07 | 0.0267 | 0.80 |
| D — Two-stage | **0.0479** | 1.15 | **0.0375** | 1.04 |
| **C — MLP IC-loss** | 0.0333 | 0.89 | **0.0155** | **0.41** |

**Kết luận: C không qua promotion gate, và không chỉ "không qua" — nó là model yếu nhất trên
gap311 trong cả 4, kém rõ rệt so với cả B (model yếu thứ nhì).** `ic_gap300` (§5.4, chỉ số dự báo
live tốt nhất) sụp mạnh nhất trong 4 model: Sharpe 0.41, và riêng fold `gap311_3` chỉ **+0.0035**
— gần như bằng 0, nằm trong sàn nhiễu. Đúng chẩn đoán §5.5 #2 (train IC ổn, val IC ≈ 0 qua khoảng
trống lớn → overfit, giảm capacity trước): walk-forward vẫn ổn (0.0333, gần A) nhưng khi bị đẩy
xa 311 moon thì rơi tệ hơn hẳn cây quyết định (B, D) lẫn ridge (A). Ridge sống sót tốt nhất qua
gap311 (chỉ mất 8%) chính vì alpha cực lớn ép nó không học đặc thù regime; MLP 3 lớp + dropout 0.3
+ weight_decay 1e-3 hoá ra vẫn đủ capacity để bám vào cấu trúc cross-section của giai đoạn train
gần, thứ không sống sót được 6 năm.

**Không promote, không thay slot nào.** Hai hướng để thử tiếp nếu muốn cứu Model C (mỗi lần một
biến, theo nguyên tắc §5.2):
1. **Giảm capacity mạnh hơn**: hidden nhỏ hơn (vd. 64/32) hoặc weight_decay cao hơn 1 bậc (1e-2).
2. **Rolling window thay vì toàn bộ lịch sử** cho tập train — nếu MLP đang học đặc thù của cả
   cửa sổ train dài thay vì tín hiệu bền, cắt bớt lịch sử xa có thể ép nó tổng quát hoá tốt hơn
   (ngược với A, nơi §2.7 đã đo dữ liệu xa không hại — nhưng MLP có thể phản ứng khác ridge).

Nếu cả hai không cứu được C, kết luận tạm thời là **A vẫn là nguồn diversity thực sự duy nhất**,
và `fantastic-snipe` nên tiếp tục giữ Model A thay vì chờ C — decorrelation về *loss function*
không tự động có nghĩa là generalize tốt qua khoảng trống 6 năm.
