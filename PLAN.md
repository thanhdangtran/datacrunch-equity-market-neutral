# DataCrunch #2 — Kế hoạch thi đấu

> Tài liệu này là nguồn sự thật duy nhất cho chiến lược. Mọi thay đổi hướng đi phải được cập nhật vào đây.
> Cập nhật lần cuối: 2026-07-28

---

## 0. TL;DR — 7 điều phải nhớ

1. **Chỉ top ~3–5% mỗi tuần mới có tiền.** Top 25% nhận ~0. Đây là trò chơi thứ hạng cực đoan, không phải trò chơi "trung bình khá".
2. 🔴 **Data local dừng ở moon 781. Moon được chấm điểm là 1092.** Khoảng trống **311 moons ≈ 6 năm** mà ta không bao giờ nhìn thấy local. Đây là rủi ro lớn nhất của cả cuộc thi — xem §2.7.
3. **Nhiễu tuần (~0.022) lớn gần bằng tín hiệu (~0.03).** Điểm một tuần gần như là xổ số. Không bao giờ đánh giá model bằng một moon.
4. **`id` không nối được giữa các moon** (đã xác minh trên data thật) → không có feature time-series theo mã. Mọi thứ là cross-sectional thuần.
5. **Features có đúng 7 mức** → lưu `int8`, LightGBM `max_bin=7`. Không mất thông tin, nhanh hơn nhiều lần, RAM giảm 4×.
6. **4 model phải khác nhau về *cấu trúc*** (loss / định nghĩa target / dạng hàm), không phải 4 seed. Decorrelation là nguồn alpha thứ hai.
7. **Submit trong tuần 1.** Điểm mất 6 tuần mới hiện. Không submit sớm = mù đến tuần 7.

---

## 1. Luật chơi

### 1.1 Chấm điểm
- Metric: **Pearson correlation** giữa prediction và target, tính trên **đúng moon cuối cùng** của dataset, mỗi tuần một lần.
- Cỡ mẫu mỗi lần chấm: ~2.000–2.400 cổ phiếu.
- Prediction phải nằm trong `[-1, 1]`. Prediction hằng số → correlation undefined → **điểm = 0**.

### 1.2 Cơ chế thưởng
```python
weekly_rewards = 1000  # USDC
percentile_rank = your_rank / nb_participants   # 0 = tệ nhất, 1 = tốt nhất
if percentile_rank <= 0.5:
    reward = 0
else:
    e = 2 * (percentile_rank - 0.5)
    weight = e ** 20
    reward = weekly_rewards * (weight / sum(participants_weights))
```

Số mũ 20 làm payoff cực kỳ lồi:

| Thứ hạng | e | w = e²⁰ | % so với #1 | Ước tính $/tuần (N=500) |
|---|---|---|---|---|
| #1 | 1.00 | 1.000 | 100% | ~$84 |
| Top 1% | 0.98 | 0.668 | 67% | ~$56 |
| Top 5% | 0.90 | 0.122 | 12% | ~$10 |
| Top 10% | 0.80 | 0.0115 | 1.2% | ~$1 |
| Top 25% | 0.50 | 0.0000010 | ~0% | ~$0 |
| Median | 0.00 | 0 | 0% | $0 |

*(Ước tính giả định N=500 và phân bố e đều trên nửa trên: sum(w) ≈ (N/2)/21 ≈ 11.9)*

### 1.3 Ba hệ quả chiến lược
1. **Payoff lồi + sàn bằng 0 ⇒ phương sai có giá trị.** Ở cùng mean IC, model *khác biệt* có kỳ vọng thưởng cao hơn model *bám consensus*.
2. **Nhưng phương sai chỉ đáng giá nếu không đánh đổi bằng mean IC.** Thêm nhiễu thuần làm giảm mean mà không được gì. Diversity phải đến từ *góc nhìn khác*, không phải từ nhiễu.
3. Vì mọi người dùng chung 1150 features, điểm các participant sẽ tương quan cao. **Người thắng tuần là người có thành phần lệch khỏi đám đông đúng hướng.**

### 1.4 Ràng buộc vận hành
| Mục | Giá trị |
|---|---|
| Embargo | **4 moons** (target = forward return 28 ngày) |
| Compute | **15h GPU hoặc CPU / tuần**, reset Chủ nhật 12:00 UTC |
| Deadline tuần | Run phải xong trước Chủ nhật 12:00 UTC |
| Độ trễ điểm | **6 tuần** (1 ngày + 4 tuần để target resolve) |
| Cloud data | Code chạy trên cloud thấy **toàn bộ** dataset (tới ~moon 1088), không chỉ 781 moons local |
| Public OOS | moons **782–790** = 2 tháng đầu 2020 — **đúng đầu COVID, không đại diện, đừng tune theo** |
| Moon chấm điểm | **1092** — cách data local **311 moons (~6 năm)** |

### ⚠️ Điều cần xác minh trên hub
`datacrunch-2.md` viết *"calculated as follows for **both metrics**"* và *"computed on the **leaderboards**"* (số nhiều), nhưng phần Scoring chỉ nêu Pearson.
→ **Rất có thể có 2 leaderboard** (Pearson + Spearman hoặc + metric tích luỹ), mỗi bảng một pool.
→ Nếu có leaderboard Spearman thì **rank-transform prediction là bắt buộc**, và tổng pool thực tế là 2×.
**Việc cần làm: vào hub.crunchdao.com/competitions/datacrunch-2 kiểm tra số leaderboard.**

---

## 2. Dữ liệu — sự thật đã xác lập

Nguồn: `financial-eda-deep-dive.ipynb` chạy trên `data/X.reduced.parquet` + `data/y.reduced.parquet`.

### 2.1 Kích thước
- **1.637.276 dòng × 1152 cột** (`id`, `moon`, `Feature_1..Feature_1150`)
- moon **1 → 781** (~15 năm, tần suất tuần)
- ~2.400–2.600 cổ phiếu mỗi moon
- float32 → **7.5 GB RAM**. Không NaN. Không duplicate.

### 2.2 Features
- **Tất cả 1150 features được quantize thành đúng 7 bins**: `{0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0}`
- Thống kê **giống hệt nhau cho cả 1150 cột**: mean = 0.499999, std = 0.231955, skew ≈ 0, kurtosis = −0.527
  → features đã được **rank-normalize cross-sectional theo từng moon**. Không cần chuẩn hoá lại.
- ~~**Redundancy khổng lồ**~~ → **SAI, đã bác bỏ bằng data thật.** Xem §2.8. `Feature_1..5` đúng là gần trùng nhau, nhưng đó là ngoại lệ chứ không đại diện: trung vị `|corr|` giữa các cặp feature chỉ **0.021**, và chỉ **0.09%** số cặp có `|r| > 0.95`. Cần **258 PC** để giải thích 90% phương sai, **411 PC** cho 95%. Feature set **không** dư thừa.

### 2.3 Target
| Chỉ số | Giá trị |
|---|---|
| Range | `[-1, 1]` |
| Mean tổng | ~5e-08 (≈ 0) |
| Std tổng | 0.2989 |
| Std per-moon | ~0.27 |
| Mean per-moon | ≈ 0 → **đã neutralize cross-sectional** |
| Kurtosis | 7.63 |
| **Số giá trị = 0** | **1.443.612 / 1.637.276 = 88,2%** |
| Số giá trị ≠ 0 | 193.664 = 11,8% |

Quantiles: `1% = -1.00`, `5% = -0.3705`, `10%..90% = 0.00`, `95% = +0.3530`, `99% = +1.00`

→ Target là dạng **có vùng chết ở giữa + clip ở hai đuôi**. Chỉ ~12% số mã mang tín hiệu; ~2% nằm ở ±1.

### 2.4 Sức mạnh tín hiệu
*(Số liệu EDA gốc dùng Spearman pooled trên sample 5% — đã được thay bằng IC per-moon trên toàn bộ 781 moons ở §2.8. Giữ lại đây để đối chiếu.)*

Kết luận EDA gốc vẫn đúng về mặt định tính: nhóm `Feature_1..7`, `Feature_43..49` âm nhất quán → một **family** (khả năng cao: short-term reversal).

**Mutual Information** top trong EDA: `Feature_999, 1061, 524, 529, 383, 241, 993, 478, 244, 1067`.
⚠️ Không một feature nào trong danh sách này xuất hiện ở top IC per-moon tính trên toàn bộ data (§2.8). **MI trong EDA là nhiễu — đã xác nhận, bỏ qua hoàn toàn.**

### 2.5 Ràng buộc lớn nhất: `id` không nối được
- `id` tăng tuần tự theo moon (moon 1: id ~600–1900; moon 781: id ~1.636.000).
- Tài liệu xác nhận: *"the same asset has a different `id` in different `moon`"*.

**Hệ quả**: không thể tạo lag feature, momentum, rolling stat theo từng mã. Toàn bộ thông tin time-series đã được nhúng sẵn trong 1150 features. **Mọi model đều là cross-sectional thuần.** Giá trị gia tăng chỉ nằm ở: nén feature, chọn dạng hàm, chọn loss, và ensemble.

### 2.6 Lưu ý về file dữ liệu
`crunch_tools.load_data()` trong quickstarter trả về moon **635–777** (276k dòng, ~2.75 năm), trong khi `data/X.reduced.parquet` có đủ **781 moons**.
→ **Đọc parquet trực tiếp**, đừng dựa vào `load_data()`.
→ `--size large` **bị Access Denied** với tài khoản này; `default` chính là bản `reduced` 781 moons. Không mất gì. Chi tiết ở [SETUP.md](SETUP.md) §4.2.

### 2.7 🔴 Split chính thức — và khoảng trống 6 năm

`data/moons_split.json` (11 KB, tải kèm data) định nghĩa split thật:

| Nhóm | Moons | Số moon | Ghi chú |
|---|---|---|---|
| `train` | 1 – 772 | 772 | Huấn luyện |
| `reduced_local` | 773 – 781 | 9 | Test local (`crunch test`) |
| `reduced_cloud` | 782 – 790 | 9 | Public OOS = 2 tháng đầu 2020 |
| **`live_to_predict`** | **1092** | **1** | **Moon được chấm điểm** |

**Data local dừng ở moon 781. Moon chấm điểm là 1092.**
Khoảng cách: **311 moons ≈ 6 năm** (moon 790 ≈ đầu 2020; moon 1092 ≈ giữa 2026).

#### ✅ ĐÃ ĐO — khoảng trống này KHÔNG nghiêm trọng như lo ngại

`src/cv.py::gap_folds` mô phỏng đúng tình huống: train tới moon T, bỏ **311 moons**, rồi mới validate.
Kết quả Model A (§4):

| | mean IC | Sharpe |
|---|---|---|
| Walk-forward thường (gap 4 moons) | 0.0329 | 1.25 |
| **Gap 311 moons** | **0.0304** | **1.31** |

**Chỉ mất 8% mean IC qua khoảng trống 6 năm, và Sharpe còn cao hơn.**
Cụ thể hơn: fold `gap311_1` train trên moons 1–314 rồi validate 626–677 đạt **+0.0297**, trong khi
fold `wf4` train trên moons 1–621 (nhiều hơn 307 moons, gần validation hơn hẳn) validate cùng đoạn
đó chỉ đạt **+0.0291**. Dữ liệu cũ 6 năm **tốt ngang** dữ liệu mới.

→ Tín hiệu có tính dừng cao. **Điều chỉnh: hạ rủi ro này từ "nghiêm trọng" xuống "trung bình"**, và
luận điểm "recency weighting là trọng tâm" ở dưới **không được dữ liệu ủng hộ** — dữ liệu xa vẫn
có giá trị, không nên vứt. Vẫn giữ `ic_gap311` làm chỉ số bắt buộc để bắt model nào học thuộc regime.

#### Hệ quả — đây là điều chỉnh lớn nhất của kế hoạch

1. **Ta validate trên dữ liệu trước 2020 nhưng bị chấm điểm ở 2026.** Sáu năm đó chứa COVID crash, đợt lạm phát 2022, chu kỳ AI 2023–2025 — toàn bộ đều **vô hình** với validation local. Khoảng cách local→live lớn hơn nhiều so với ước lượng ban đầu.
2. **Model artifact train local gần như vô giá trị.** Thứ thực sự được chấm là **hàm `train()` khi nó được chạy trên cloud với dữ liệu tới ~moon 1088**. Ta không tune một model — ta tune một **thủ tục huấn luyện** phải tự xoay xở với 6 năm chưa từng thấy.
3. **Recency weighting và rolling window chuyển từ "thử nghiệm" thành trọng tâm.** Model bám chặt vào đặc thù 2005–2020 sẽ chết. Phải ưu tiên thủ tục thích nghi được với dữ liệu gần.
4. **`train_frequency` trở thành siêu tham số quan trọng**, không phải chi tiết vận hành. Retrain trên cloud là cách duy nhất chạm được vào dữ liệu 2020–2026.
5. **Cách validate đúng: mô phỏng chính khoảng trống này.** Train tới moon T, rồi đo IC ở T+300 chứ không chỉ T+5. Nếu model giữ được IC qua khoảng cách 300 moon trong lịch sử, nó mới có cơ hội ở live. **Đây phải là một cột riêng trong báo cáo — xem §5.4.**

### 2.8 Khảo sát trên data thật — `src/survey.py` → `reports/survey.json`

Tính IC **per-moon** của cả 1150 features trên **toàn bộ 781 moons** (không sample, không pooled).
Đây là cách đo đúng với metric chấm điểm; mọi con số ở §2.4 nên nhường cho phần này.

#### a) Baseline — con số phải beat

| Chiến lược | mean IC | ic_std | **Sharpe** | hit rate | IC 104 moon gần nhất |
|---|---|---|---|---|---|
| `−Feature_43` (một feature duy nhất) | 0.0232 | 0.0206 | **1.13** | 86,6% | 0.0223 |
| Trung bình 20 feature mạnh nhất (đã căn dấu) | 0.0238 | 0.0205 | 1.16 | 86,6% | 0.0211 |
| **Trung bình 100 feature mạnh nhất** | **0.0251** | 0.0211 | **1.19** | **88,7%** | 0.0242 |

> 🔴 **Một feature duy nhất, đảo dấu, đạt Sharpe 1.13.** Gộp 100 feature chỉ nâng mean IC từ
> 0.0232 lên 0.0251 (+8%). **Đây là champion khởi điểm — mọi model phải beat `mean_ic 0.0251 / Sharpe 1.19`,
> nếu không thì vô nghĩa.** Một model phức tạp thua con số này là model sai.

#### b) IC per-feature
- Mạnh nhất: **`Feature_43` mean IC = −0.0232, t = −31.5**. Kế đó `Feature_1` (−0.0232), `Feature_44`, `Feature_2`, `Feature_3`, `Feature_45`, `Feature_46`, `Feature_4`, `Feature_47`, `Feature_5` — **toàn bộ top 10 là một family duy nhất, tất cả đều âm**.
- Dương mạnh nhất: `Feature_83` (+0.0168, t = 19.7), rồi `Feature_99`, `Feature_1090`, `Feature_1087`, `Feature_68`, `Feature_1089`, `Feature_1088`, `Feature_64`, `Feature_1124`, `Feature_1096`. Thấy rõ 2 cụm: `64–99` và `1087–1096`.
- **268 / 1150 features có |t| > 3** (kỳ vọng ngẫu nhiên ≈ 3). Tín hiệu tồn tại trên diện rộng, không chỉ ở vài feature.
- Nhưng chỉ **17 features** có `|mean IC| > 0.01`. Phần lớn tín hiệu rất yếu và nằm rải rác.

#### c) Tính ổn định qua thời gian — tin tốt cho §2.7
| Phép đo | Giá trị |
|---|---|
| corr(vector IC nửa đầu, nửa sau) | **0.591** |
| corr(vector IC quý 1, quý 4) | **0.406** |
| IC của top-20 feature (chọn ở nửa đầu) đo tại nửa đầu | −0.01364 |
| IC của top-20 feature (chọn ở nửa đầu) đo tại **nửa sau** | **−0.01370** |

→ Vector IC toàn cục drift khá mạnh (0.41 giữa quý 1 và quý 4), **nhưng nhóm feature mạnh nhất
giữ nguyên sức mạnh gần như tuyệt đối** qua thời gian. Tín hiệu cốt lõi là bền.
Đây là bằng chứng đầu tiên cho thấy khoảng trống 311 moons ở §2.7 **có thể vượt qua được** —
miễn là bám vào nhóm feature bền thay vì đuổi theo feature có IC cao nhất trong một cửa sổ.

#### d) Universe đang co lại và target đang biến động mạnh hơn
| Phép đo | 50 moon đầu | 50 moon cuối |
|---|---|---|
| Số mã / moon | 2.533 | **1.900** |
| Tỷ lệ target = 0 | 90,2% | **87,0%** |
| Std target | 0.272 | **0.314** |

→ Ít mã hơn ⇒ **nhiễu tuần tăng**: `1/√1900 = 0.0229` (so với `1/√2533 = 0.0199`).
→ Nhiễu tuần trung bình toàn lịch sử đo được: **0.0219**, khớp với `ic_std` thực nghiệm 0.0206–0.0211 của các baseline.
→ Xu hướng này kéo dài 6 năm tới moon 1092 thì universe live có thể còn nhỏ hơn nữa ⇒ nhiễu còn cao hơn ⇒ **thứ hạng tuần càng giống xổ số** ⇒ luận điểm "cần diversity" ở §1.3 càng mạnh.

---

## 3. Ba quyết định thiết kế bị ép bởi dữ liệu

### 3.1 Dùng `int8`, `max_bin=7`
Features có đúng 7 mức → map `{0, .17, .33, .5, .67, .83, 1.0}` → `{0,1,2,3,4,5,6}` dạng `int8`.
- RAM: 7.5 GB → **1.9 GB**
- LightGBM `max_bin=7` khớp chính xác → **không mất một chút thông tin nào**, binning nhanh hơn nhiều lần
- Quan trọng vì trên cloud code nhận **toàn bộ** dataset, không chỉ 15 năm

### 3.2 Loss mặc định là L2 trên target thô
Pearson được tối đa hoá bởi `E[y|x]` (sai khác một phép affine). L2 chính là estimator của `E[y|x]`.
→ **88% zeros KHÔNG phải lý do để đổi loss.** Nó chỉ có nghĩa prediction sẽ tập trung quanh 0 — điều đó không ảnh hưởng Pearson.
→ **Rank-transform *target* là có hại — đã kiểm chứng bằng số.** Áp `gauss_rank` lên target của moon 700
(87,6% giá trị bằng đúng 0): nhóm zero bị trải trên **1,75 đơn vị** dải giá trị dù target thật giống hệt nhau;
thứ hạng gán cho chúng có **`corr = +0.72` với vị trí dòng trong dataframe**; và **67% phương sai** của
target-đã-rank đến từ nhóm dòng đồng nhất đó. Rank hoá target = train trên nhiễu thứ tự dòng cho 88% dữ liệu.
Submission cũ #64783 mắc đúng lỗi này.
→ Rank-transform *prediction* là phép đơn điệu **không affine** nên **có** đổi Pearson. Là một siêu tham số cần test, không phải mặc định.

### 3.3 Bagging theo moon, không theo dòng
Trong cross-section, các dòng cùng một moon không độc lập. Bagging theo dòng làm rò rỉ thông tin trong cùng cross-section. **Luôn resample ở cấp moon.**

---

## 4. Bốn model

Trục thiết kế: mỗi model phải khác ở **ít nhất một trong ba thứ** — không gian feature, dạng hàm, hoặc **loss / định nghĩa target**.

### 4.0 Sổ đăng ký model ↔ slot

Mỗi project slot = một workspace = một dòng leaderboard riêng (cùng `userId 13215`).
Workspace giữ project token dài hạn → **không bao giờ cần clone token nữa**. Submit bằng `crunch push`.

| Slot | Workspace | Data | Model được gán | Trạng thái |
|---|---|---|---|---|
| `secure-ladybug` | `ws-secure-ladybug/` | ✅ | **A** — Ridge 1150 features, alpha 1e6 | ✅ đã có kết quả, chờ đóng gói |
| `lovely-fowl` | `ws-lovely-fowl/` | ✅ | **B** — LightGBM `max_bin=7` | chưa làm |
| `fantastic-snipe` | `ws-fantastic-snipe/` | ✅ | **C** — MLP tối ưu trực tiếp IC | chưa làm — *đang giữ submission cũ #64783* |
| `scornful-trout` | `ws-scornful-trout/` | ✅ | **D** — Two-stage 88% zeros | chưa làm |

✅ **Đủ 4 slot.** Từ giờ không cần clone token nữa — mọi submission đi qua `crunch push` trong workspace tương ứng.

> ⚠️ `ws-fantastic-snipe/` chứa submission cũ #64783 (Ridge+LGBM blend, PCA 40, gauss-rank target,
> neutralization 0.5). **Không dùng làm nền.** Nó rank-transform target, mà §3.2 + kiểm chứng thực
> nghiệm cho thấy điều đó biến 88% dữ liệu thành nhiễu theo thứ tự dòng (xem §3.2).
> Giữ file lại để đối chiếu, code mới viết từ đầu.

### Model A — Ridge có ràng buộc ổn định  *(neo an toàn)* — **ĐÃ SỬA THIẾT KẾ**
| | |
|---|---|
| **Khác biệt** | Tuyến tính + chọn feature theo *độ bền*, không theo *độ mạnh* |
| **~~Thiết kế cũ~~** | ~~Gom cụm về 120–200 chiều~~ — **bỏ**. §2.8a cho thấy feature set không dư thừa (cần 258 PC cho 90% var, trung vị `\|corr\|` = 0.021). Nén xuống 150 chiều là **vứt đi tín hiệu thật**. |
| **Thiết kế mới** | Ridge trên **toàn bộ 1150 features**, alpha lớn (tự nó đã là cơ chế nén). Trọng số feature ưu tiên theo **độ ổn định IC giữa các era** (§2.8c) chứ không theo `\|mean IC\|` toàn cục — đây chính là phòng thủ cho khoảng trống 311 moons. Expanding window + recency weighting. |
| **Vai trò** | Sàn không bao giờ sập. Deterministic, chạy vài phút, bền qua đổi regime. |
| **Phải beat** | `mean_ic 0.0251 / Sharpe 1.19` (§2.8a) |
| **✅ KẾT QUẢ** | **`alpha = 1e6`: mean_ic 0.0329, Sharpe 1.25, hit 90%, last104 0.0362, cả 6/6 fold dương. Gap311: 0.0304 / Sharpe 1.31, 3/3 fold dương.** Vượt champion. → `reports/exp_ridge.json` |
| **Rủi ro** | Bỏ lỡ hoàn toàn phi tuyến |

**Quét alpha** (`src/run_ridge.py`), walk-forward mean_ic: `1e5→0.0312`, **`1e6→0.0329`**, `1e7→0.0309`, `1e8→0.0253`, `1e10→0.0201`.
Đỉnh rõ ràng tại **1e6** — cực lớn, xác nhận §2.8: 1150 features mang tín hiệu yếu và rải rác nên cần
shrink rất mạnh, nhưng **không** nên cắt bỏ feature (nén xuống 150 chiều sẽ mất phần đuôi này).

### Model B — LightGBM `max_bin=7`  *(phi tuyến, tương tác bậc thấp)*
| | |
|---|---|
| **Khác biệt** | Dạng hàm (cây, tương tác) |
| **Chi tiết** | Features `int8` 0–6 nguyên bản. `num_leaves` nhỏ, `feature_fraction` 0.1–0.3, lr thấp + nhiều cây, **bagging theo moon**. Loss L2 trên target thô. |
| **Vai trò** | Bắt tương tác giữa features |
| **Rủi ro** | **Đây là model đa số participant cũng dùng** → mean IC tốt nhưng correlated cao với đám đông. Một mình nó khó lọt top 5%. |
| **✅ KẾT QUẢ** | walk-forward **mean_ic 0.0429 / Sharpe 1.07**; gap311 **0.0267 / Sharpe 0.80**. → `reports/exp_lgbm.json` |

#### So sánh A vs B trên **cùng** fold (fold khớp nhau, tránh so lệch)

Model B chỉ chạy 3 fold walk-forward gần nhất và 2 gap fold, nên phải đối chiếu đúng những fold đó:

| Fold | Ridge (A) | LightGBM (B) |
|---|---|---|
| wf4 | +0.0291 | **+0.0470** |
| wf5 | +0.0451 | +0.0459 |
| wf6 | +0.0273 | **+0.0359** |
| gap311_2 | **+0.0390** | +0.0288 |
| gap311_3 | **+0.0225** | +0.0245 |

| | mean IC | ic_std | **Sharpe** | hit |
|---|---|---|---|---|
| A — Ridge (6 fold) | 0.0329 | **0.0264** | **1.25** | 90% |
| B — LightGBM (3 fold) | **0.0429** | 0.0400 | 1.07 | 86% |
| A — gap311 | 0.0304 | 0.0232 | **1.31** | 90% |
| B — gap311 | 0.0267 | 0.0335 | 0.80 | 79% |

**Kết luận: B có mean IC cao hơn hẳn (+30%) nhưng phương sai gấp rưỡi, nên Sharpe thấp hơn.**

Hai điều quan trọng:
1. **`ic_std` của B là 0.0400, trong khi sàn nhiễu lý thuyết chỉ ~0.022 và Ridge đạt 0.0264.** B đang tạo ra phương sai vượt xa mức nhiễu không tránh được → dấu hiệu thừa capacity. Theo §5.5 #2, việc cần làm là **giảm capacity trước**, không phải thêm feature.
2. **B suy giảm mạnh hơn nhiều qua khoảng trống 311 moons**: −38% (0.0429→0.0267) so với Ridge chỉ −8% (0.0329→0.0304). Phần lợi thế phi tuyến của B **phụ thuộc regime** và không sống sót qua 6 năm.

→ **B chưa qua promotion gate §5.6** (Sharpe 1.07 < 1.25 của A; gap311 tệ hơn hẳn).
→ Nhưng §1.3 nói phương sai *có giá trị* khi payoff lồi, và B decorrelated về mặt cấu trúc với A.
   Nên B **vẫn xứng đáng một slot riêng** như một cửa cược khác, chứ không phải để thay A.
   Trước đó phải thử hạ capacity (ít cây hơn / `lambda_l2` cao hơn / `num_leaves` nhỏ hơn) — mỗi exp một thay đổi.

### Model C — MLP tối ưu trực tiếp IC  *(nguồn decorrelation lớn nhất)*
| | |
|---|---|
| **Khác biệt** | **Hàm loss** |
| **Chi tiết** | Batch = **toàn bộ cross-section của một moon** (~2.400 dòng). **Loss = −Pearson(pred, target)** trên batch đó. 2–3 lớp ẩn, dropout + weight decay mạnh, early stop theo validation IC. |
| **Vì sao khác** | A và B ước lượng `E[y|x]`. C tối ưu **thẳng metric chấm điểm** — nó tự do bỏ qua scale và chỉ tập trung vào thứ tự tương đối trong cross-section. Prediction sẽ lệch đáng kể khỏi A/B. |
| **Vai trò** | Nguồn diversity chính |
| **Rủi ro** | Bất ổn khi train; cần early stopping kỷ luật |

### Model D — Hai tầng khai thác cấu trúc 88% zeros  *(định nghĩa lại target)*
| | |
|---|---|
| **Khác biệt** | **Định nghĩa target** |
| **Chi tiết** | **Tầng 1**: LightGBM binary → `P(target ≠ 0 \| x)` ("mã này có phải extreme mover không"). **Tầng 2**: LightGBM chỉ train trên 193.664 dòng `target ≠ 0` → dự đoán giá trị/dấu. **Kết hợp**: `pred = P(≠0) × E[target \| ≠0]`. |
| **Vì sao khác** | 88,2% zeros là **cấu trúc thật**, không phải nhiễu — và A/B/C đều bỏ qua nó. Cùng features nhưng bài toán khác → sai số độc lập ở mức cao. |
| **Vai trò** | Diversity + khai thác cấu trúc target |
| **Rủi ro** | Tầng 2 chỉ có 12% dữ liệu → dễ overfit |

### Ensemble E
- **Rank-average theo từng moon** — không average giá trị thô, vì scale của 4 model khác nhau hoàn toàn.
- Trọng số: **bắt đầu bằng đều nhau (0.25)**. Chỉ fit trọng số khi có ≥5 fold độc lập, nếu không sẽ overfit.
- **Kiểm tra bắt buộc**: ma trận tương quan giữa prediction của A/B/C/D.
  → Cặp nào `corr > 0.95` thì model đó **không đóng góp gì** — phải sửa hoặc thay.

---

## 5. Quy trình: chạy → đánh giá → sửa → đánh giá → submit

### 5.1 Cấu trúc repo
```
src/
  data.py           # load parquet, map 7 bins -> int8, cache .npy
  cv.py             # purged walk-forward splitter (gap = embargo = 4)
  metrics.py        # per-moon Pearson, Sharpe, drawdown, report
  models/
    ridge.py        # Model A
    lgbm.py         # Model B
    mlp_ic.py       # Model C
    twostage.py     # Model D
  ensemble.py       # Ensemble E
  run.py            # entrypoint: 1 config -> 1 report
  report.py         # gộp reports/ -> leaderboard.csv
  submission.py     # sinh train()/infer() từ config thắng cuộc
experiments/
  exp_001.yaml ...
reports/
  exp_001.json, leaderboard.csv
```

### 5.2 Vòng lặp
```
1. RUN       python -m src.run --exp experiments/exp_017.yaml
2. EVALUATE  python -m src.report              # bảng so sánh với champion
3. DIAGNOSE  theo checklist 5.5
4. FIX       sửa ĐÚNG MỘT thứ, tăng số exp
5. GOTO 1    cho đến khi qua promotion gate 5.6
6. SUBMIT    crunch test  ->  crunch push
```

> **Nguyên tắc cứng: mỗi exp chỉ đổi một biến.**
> Với signal ~0.03 và nhiễu ~0.022, đổi 2 thứ cùng lúc là **mất vĩnh viễn khả năng quy kết nguyên nhân**.

### 5.3 Thiết kế validation (quan trọng nhất — làm trước mọi model)
- **Purged walk-forward theo moon**: train `[0, T]` → **gap 4 moons (embargo)** → validate `[T+5, T+5+H]`
- 5–6 fold trải trên ~10 năm gần nhất
- **Không bao giờ** dùng random split. **Không bao giờ** để moon validation lọt vào train.

### 5.4 Báo cáo chuẩn mỗi lần chạy
Không nhìn một con số. Mỗi exp phải in ra:

| Trường | Ý nghĩa |
|---|---|
| `mean_ic` | Pearson per-moon trung bình |
| `ic_std` | Độ lệch chuẩn |
| **`sharpe = mean/std`** | **Chỉ số ra quyết định chính**, không phải mean |
| `hit_rate` | % moon có IC dương |
| `ic_by_fold` | 5–6 fold riêng biệt — **một fold âm là cờ đỏ** |
| `ic_last_104` | Chỉ 2 năm gần nhất (regime hiện tại > năm 2010) |
| **`ic_gap300`** | **IC khi train tới T rồi đo ở T+300 — mô phỏng khoảng trống local→live (§2.7). Đây là chỉ số dự báo live tốt nhất ta có.** |
| `corr_with_champion` | Có thêm được diversity không |
| `runtime`, `peak_ram` | Ngân sách 15h/tuần |

**Kỳ vọng thực tế: `mean_ic` tốt nằm khoảng 0.02–0.05.**

⚠️ `ic_gap300` gần như chắc chắn sẽ **thấp hơn nhiều** `mean_ic`. Đừng hoảng — điều cần
là *so sánh* `ic_gap300` giữa các model, vì nó đo đúng thứ live sẽ đo: khả năng sống sót
qua 6 năm không nhìn thấy. Model có `mean_ic` cao nhưng `ic_gap300` ≈ 0 là model đã học
thuộc regime cũ.

### 5.5 Chẩn đoán khi kết quả xấu — theo đúng thứ tự này
| # | Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|---|
| 1 | `mean_ic > 0.10` | **Leak** | Kiểm tra embargo 4 moons; kiểm tra không moon validation nào lọt vào train |
| 2 | Train IC cao, val IC ≈ 0 | Overfit | **Giảm capacity trước**, đừng thêm feature |
| 3 | Fold gần nhất tệ hơn fold cũ | Non-stationarity | Thử recency weighting hoặc rolling window thay expanding |
| 4 | IC dương nhưng Sharpe < 0.3 | Tín hiệu dồn vào vài moon | Kiểm tra có phải chỉ ăn được ở giai đoạn biến động cao |
| 5 | Ensemble không hơn model tốt nhất | Các model correlated quá | Xem ma trận corr; sửa hoặc thay model trùng lặp |

### 5.6 Promotion gate — điều kiện để được submit
Tất cả phải đúng, **không thoả hiệp**:

- [ ] `mean_ic` > champion — **champion khởi điểm = 0.0251** (§2.8a)
- [ ] `sharpe` > champion — **champion khởi điểm = 1.19** — *nếu mean cao hơn nhưng Sharpe thấp hơn thì **KHÔNG** promote, đó thường là may mắn*
- [ ] Không fold nào có mean IC âm
- [ ] `ic_last_104` không tệ hơn champion quá 20%
- [ ] `crunch test` pass với **determinism check bật**
- [ ] Prediction không hằng số, nằm trong `[-1, 1]`
- [ ] Runtime train+infer < ~10h (chừa biên trên quota 15h)
- [ ] Nếu thêm vào ensemble để lấy diversity: `corr_with_champion < 0.95`

---

## 6. Lịch trình

| Tuần | Việc | Đầu ra |
|---|---|---|
| **1** | Phase 0 (data → int8 cache) + harness CV + **Model A** + **Model B** | **Submit cuối tuần 1 — bắt buộc** |
| **2** | **Model C** (MLP IC-loss) + **Model D** (two-stage) | 4 model có số đo |
| **3** | **Ensemble E**, ma trận decorrelation, tuning | Champion đầu tiên |
| **4+** | Lặp: 1 thay đổi / exp, promote khi qua gate | Cải thiện tăng dần |

> **Submit trong tuần 1 là bắt buộc.** Điểm mất 6 tuần mới hiện ra. Không có submission tuần 1 = mù hoàn toàn đến tuần 7.

---

## 7. Rủi ro đã biết

| Rủi ro | Mức độ | Xử lý |
|---|---|---|
| **Khoảng trống 311 moons (~6 năm) giữa data local và moon chấm điểm** | 🔴 **Nghiêm trọng — rủi ro số 1** | Đo `ic_gap300` (§5.4); ưu tiên recency weighting + rolling window; tune *thủ tục train* chứ không tune *model artifact* (§2.7) |
| Overfit 15 năm local rồi gãy trên regime mới | **Cao** — failure mode phổ biến nhất | Quyết định bằng Sharpe qua các fold, không phải mean IC |
| Public OOS = 2 tháng đầu 2020 (đầu COVID) | Cao | **Không tune theo nó.** Điểm ở đó sẽ tệ và không đại diện cho gì cả |
| MI scores trong EDA là noise | Trung bình | Coi là chưa xác thực cho tới khi validate trên full data |
| Model trung bình khá → $0 | Chắc chắn | Tối đa hoá mean IC **và** giữ diversity; đừng bám consensus |
| 4 model hoá ra correlated cao | Trung bình | Ma trận corr là gate bắt buộc trước khi ensemble |
| Cloud thấy toàn bộ dataset → OOM | Trung bình | `int8` từ đầu; test memory với data lớn hơn local |

---

## 8. Môi trường

### Đã xác lập
| Gói | Phiên bản | Trạng thái |
|---|---|---|
| Python | 3.14.6 | ✅ |
| numpy | 2.4.2 | ✅ |
| pandas | 3.0.0 | ✅ |
| pyarrow | 24.0.0 | ✅ |
| scikit-learn | 1.8.0 | ✅ |
| lightgbm | 4.6.0 | ✅ |
| scipy | 1.17.0 | ✅ |
| crunch-cli | 11.8.0 | ✅ |
| torch | 2.13.0+cpu | ✅ có wheel cho py3.14, không cần env riêng |

**CUDA không khả dụng trên máy local** (`torch.cuda.is_available() = False`). Không phải vấn đề: Model C dùng batch = 1 cross-section (~2.400×1150), CPU thừa sức. Cloud cho 15h *GPU hoặc CPU* — nếu chọn GPU thì code phải `.to(device)` theo biến, đừng hardcode `cpu`.

### Smoke test IC loss — đã chạy, đã pass
`scratchpad/smoke_ic_loss.py`: MLP + loss `−Pearson` trên dữ liệu tổng hợp mô phỏng cấu trúc thật (7 bins, 87,9% zeros, 2.400 mã/moon, 300 moons).

```
epoch 2: val mean IC=+0.0389  std=0.0199  sharpe=1.95  hit=0.98
300 moons x 6 epochs in 20.6s on CPU
```

Hai điều xác nhận được:
1. **Pipeline Model C chạy được** trên torch 2.13/py3.14, và rất nhanh. Ngoại suy sang 1150 features × 781 moons: ~2–3 phút cho 6 epoch. Ngân sách compute **không** phải ràng buộc cho Model C.
2. **Ước lượng nhiễu tuần trong §1.3 được kiểm chứng bằng thực nghiệm**: std của per-moon IC ra **0.020–0.022**, khớp với dự đoán lý thuyết `1/√2400 ≈ 0.0204`. Con số "nhiễu tuần ≈ tín hiệu" không phải phỏng đoán.

⚠️ Mean IC 0.039 ở trên là trên **dữ liệu tổng hợp có tín hiệu cấy sẵn** — nó chỉ chứng minh pipeline hoạt động, **không** nói gì về IC đạt được trên dữ liệu thật.

### Workspace & project slot
Chi tiết vận hành (lệnh, bẫy, layout) ở **[SETUP.md](SETUP.md)**. Token ở `TOKENS.local.md` (gitignored — repo có remote GitHub công khai).

| Workspace | Project slot | Data | Trạng thái |
|---|---|---|---|
| `ws-lovely-fowl/` | `lovely-fowl` | ✅ 718 MB đã tải | sẵn sàng |
| `ws-secure-ladybug/` | `secure-ladybug` | ❌ chưa tải | sẵn sàng (chạy `crunch download`) |

**Data đã về và đã xác minh** — không còn blocker.

### ⚠️ Chỉ có 2 slot, không phải 4
Trong 4 dòng lệnh được cung cấp: dòng 3 và 4 trùng nhau → chỉ **3 token phân biệt**, trong đó **1 đã chết**. Còn **2 slot sống**.

**Mỗi project slot là một dòng leaderboard riêng** (cùng `userId: 13215`). Với payoff lồi e²⁰, nhiều slot decorrelated = nhiều vé số độc lập cho cùng một tuần — về mặt kỳ vọng thì đây là đòn bẩy rất mạnh.

→ **Cần 2 clone token nữa** để chạy đủ 4 model thành 4 submission riêng.
→ **Câu hỏi cần xác minh trên hub**: nền tảng tính thưởng theo *submission* hay theo *user*? Nếu theo user thì nhiều slot không nhân được kỳ vọng, và chiến lược đúng là gộp 4 model thành 1 ensemble trên 1 slot. **Điều này quyết định §4 kết thúc bằng Ensemble E hay bằng 4 submission riêng — phải trả lời trước tuần 3.**

---

## 9. Việc cần làm tiếp theo

### Đã xong
- [x] `src/data.py` — int8 memmap cache (85s, sai số khỏi lưới 7 mức = **0.00 tuyệt đối**)
- [x] `src/survey.py` — khảo sát data thật → `reports/survey.json` (§2.8)
- [x] Xác minh `id` unique toàn cục → §2.5 đúng
- [x] Xác minh split chính thức → phát hiện khoảng trống 311 moons (§2.7)
- [x] 2 project slot sống, có data riêng: `ws-lovely-fowl`, `ws-secure-ladybug`
- [x] Môi trường đủ, torch 2.13 + smoke test IC loss pass (§8)

### Đang chặn
- [ ] 🔴 **Cần 2 clone token nữa** cho slot #3, #4. Token có TTL rất ngắn — 4/6 token đã chết vì để lâu. **Lấy 1 token → gửi ngay → chạy ngay → xong mới lấy cái tiếp theo.**
- [ ] Xác minh số leaderboard trên hub (§1.4) — quyết định có bắt buộc rank-transform không
- [ ] Xác minh nền tảng tính thưởng theo *submission* hay theo *user* (§8) — quyết định 4 submission riêng hay 1 ensemble

### Tiếp theo (không bị chặn)
- [ ] `src/cv.py` + `src/metrics.py` — harness purged walk-forward + `ic_gap300`
- [ ] Model A (Ridge 1150 features, thiết kế đã sửa ở §4) → beat `0.0251 / Sharpe 1.19`
- [ ] Model B (LightGBM `max_bin=7`) → submission đầu tiên vào `ws-lovely-fowl`
