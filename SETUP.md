# Setup & vận hành workspace

> Chiến lược nằm ở [PLAN.md](PLAN.md). File này chỉ nói về cơ chế: môi trường, workspace, lệnh.
> Token nằm ở `TOKENS.local.md` (gitignored).

## 1. Môi trường

Python **3.14.6**, không cần venv riêng — mọi thứ chạy trên interpreter hệ thống.

| Gói | Phiên bản |
|---|---|
| numpy | 2.4.2 |
| pandas | 3.0.0 |
| pyarrow | 24.0.0 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.6.0 |
| scipy | 1.17.0 |
| torch | 2.13.0+cpu |
| crunch-cli | 11.8.0 |

**CUDA không khả dụng local** (`torch.cuda.is_available() == False`). Không sao — Model C
dùng batch = 1 cross-section (~2.100×1150), CPU thừa sức (đo được: 300 moons × 6 epoch = 20,6s
với 200 features). Cloud cho *15h GPU hoặc CPU*; nếu chọn GPU thì code phải dùng biến `device`,
đừng hardcode `cpu`.

## 2. Workspace

```
ws-lovely-fowl/        # project slot "lovely-fowl"  — CÓ data
  .crunchdao/
    project.json       # {competitionName, projectName, userId, sizeVariant}
    token              # project token dài hạn — tái dùng được, ĐỪNG MẤT
  data/
    X.reduced.parquet  # 718 MB, 1.637.276 × 1152
    y.reduced.parquet  # 7,8 MB
    moons_split.json   # 11 KB — định nghĩa split chính thức, xem PLAN.md §2.7
  prediction/          # output của `crunch test`
  resources/           # model artifacts (model_directory_path)

ws-secure-ladybug/     # project slot "secure-ladybug" — CHƯA có data
```

Mỗi workspace = **một project slot riêng trên platform** = một dòng leaderboard riêng.
Cùng `userId: 13215`.

## 3. Lệnh

### Tạo workspace mới (tiêu thụ 1 clone token — chỉ chạy khi chắc chắn)
```bash
crunch setup datacrunch-2 <TÊN_THƯ_MỤC> --token <CLONE_TOKEN> --no-quickstarter
```
- `--no-quickstarter` **bắt buộc** khi chạy trong shell không tương tác, nếu không sẽ treo ở prompt
- `--no-data` để bỏ qua tải data (nhưng vẫn tiêu thụ token — xem cảnh báo dưới)

### Tải data vào workspace đã có (KHÔNG tốn clone token)
```bash
cd ws-lovely-fowl
crunch download --size-variant default
```

### Test / submit
```bash
cd ws-lovely-fowl
crunch test              # chạy local, mô phỏng cloud, có determinism check
crunch push              # gửi submission
```

## 4. Ba cái bẫy đã gặp

### 4.1 Clone token dùng một lần
Token trong `crunch setup` **bị tiêu thụ ngay lần dùng đầu**, kể cả khi chạy với `--no-data`.
Lần thứ hai → *"Your token seems to have expired or is invalid"*.
→ **Không probe, không chạy nháp.** Sau khi setup xong, dùng `<ws>/.crunchdao/token` cho mọi
việc tiếp theo.

### 4.2 `--size large` không có quyền
```
crunch download --size-variant large
→ You do not have permission to access this resource: Access Denied
```
Tài khoản chỉ truy cập được `default` (và `small`).
**`default` chính là bản `reduced`**: `X.reduced.parquet` với đủ **781 moons** — đúng file mà
`financial-eda-deep-dive.ipynb` đã dùng. Không mất gì cả; `--size large` trong tài liệu là
tuỳ chọn không dành cho tier này.

`--size-variant` cũng ghi vào `project.json` (`sizeVariant`), nên một lần đặt sai sẽ dính lại.

### 4.3 `crunch_tools.load_data()` trả về ít data hơn file trên đĩa
`load_data()` trong quickstarter trả moon 635–777 (~2,75 năm). File `X.reduced.parquet` có đủ
781 moons. → **Đọc parquet trực tiếp**, đừng dựa vào `load_data()` cho việc nghiên cứu.

## 5. Số liệu data đã xác minh

Chạy trực tiếp trên `ws-lovely-fowl/data/`:

| | |
|---|---|
| X | 1.637.276 dòng × 1152 cột, moons 1–781 |
| y | moons 1–781, target zero fraction **0,8817** |
| Số mã / moon | min 1.698, max 2.624, **trung bình 2.096** |
| `id` unique toàn cục | **True** → xác nhận không nối được giữa các moon |

Lưu ý: trung bình 2.096 mã/moon (thấp hơn con số "~2.400" ước lượng ban đầu từ EDA).
→ nhiễu Pearson mỗi tuần ≈ `1/√2096` ≈ **0,0218**.
