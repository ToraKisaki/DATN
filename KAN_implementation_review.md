# Rà soát phương án tích hợp KAN vào W-NETR & đối chiếu với Chương 3

> Tài liệu nội bộ phục vụ viết luận. Tổng hợp **chính xác những gì code thực sự làm** (đọc từ
> `models/`, `networks_kan.py`, `train_wnetr_networks.py`) so với **W-NETR gốc dùng MLP**, rồi đối
> chiếu với `Chuong/3_DeXuat_ThietLap.tex` để chỉ ra phần cần viết / sửa lại.
>
> Ngày rà soát: 2026-06-26 (đã hiệu chỉnh sau khi đối chiếu `ablation_7_5_findings.md`). Nguồn sự thật
> là **mã nguồn** + script train thật **`train_wnetr_networks_v4.py`** (KHÔNG phải `train_wnetr_networks.py`
> bản cũ). CLAUDE.md có vài chỗ lệch, đã ghi chú bên dưới.
>
> **2 điểm chốt do người dùng xác nhận:**
> 1. Mọi run KAN báo cáo đều train bằng **v4 ⇒ `use_base_update=True`** (nhánh SiLU residual của FasterKAN bật).
> 2. Mô hình KAN đưa vào kết quả/ablation là biến thể **`kan_wnetr_base`** (≈38.87M tham số), **không phải**
>    `kan_wnetr` mặc định (≈110M). Xem B.5.

---

## Phần A — Baseline để so sánh: W-NETR gốc (MLP) nằm ở đâu trong repo

| Thành phần | Bản gốc (MLP) | Bản đề xuất (KAN) |
|---|---|---|
| Mô hình W (2 nhánh) | `models/unetr.py` → `UNETR` | `models/unetr_kan_wnetr.py` → `KANWNETR` |
| Encoder ViT | `models/vit.py` → `ViT` | `models/kanvit.py` → `KANViT` |
| Transformer block | `models/transformerblock.py` → `TransformerBlock` | `models/kantransformerblock.py` → `KANTransformerBlock` |
| Khối phi tuyến (FFN) | `MLPBlock` (Linear→GELU→Linear) | `FasterKANBlock` → `FasterKAN` (`models/fasterkan.py`) |

**Kết luận quan trọng:** `KANWNETR` là **bản sao gần như nguyên văn** của `UNETR` gốc; phần khung
hình chữ W (2 nhánh, skip-connection, phép trừ đặc trưng + `tanh`, decoder CNN) **giữ nguyên 100%**.
Khác biệt thực sự **chỉ nằm bên trong FFN của mỗi Transformer encoder block**. Câu khẳng định ở
Chương 3 "*Điểm thay đổi duy nhất là thay MLP … bằng KAN block*" về macro-architecture là **đúng**.
Phần cần bổ sung là **mô tả đúng KAN đó là loại gì** (xem Phần C).

---

## Phần B — Các thay đổi cụ thể so với MLP-WNETR gốc

### B.1 Khối FFN: MLPBlock → FasterKANBlock

**Gốc (`transformerblock.py`):**
```
MLPBlock: Linear(hidden→mlp_dim) → GELU → Dropout → Linear(mlp_dim→hidden) → Dropout
TransformerBlock.forward:
    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))
```

**Đề xuất (`kantransformerblock.py`):**
```
FasterKANBlock: KAN([hidden, mlp_dim, hidden]) → Dropout
KANTransformerBlock.forward:
    x = x + attn(norm1(x))
    x = x + kan(norm2(x))      # cấu trúc residual + LN y hệt bản gốc
```
→ Macro y hệt. GELU-MLP cố định được thay bằng KAN có hàm kích hoạt **học được trên cạnh**.

### B.2 Bên trong KAN: dùng **FasterKAN (RBF)**, KHÔNG phải B-spline

Đây là chi tiết quan trọng nhất và Chương 3 đang nói chưa đúng.

- `FasterKAN` = stack **2 lớp** `FasterKANLayer`: `[hidden → mlp_dim]` rồi `[mlp_dim → hidden]`.
- Mỗi `FasterKANLayer` gồm:
  1. **LayerNorm nội bộ** trên đầu vào (`self.layernorm`),
  2. **`ReflectionalSwitchFunction`** — hàm cơ sở dạng RBF,
  3. `SplineLinear` — `nn.Linear(input_dim * num_grids, output_dim, bias=False)`, init Xavier-uniform.
- `ReflectionalSwitchFunction` (grid_min=−2, grid_max=2, `denominator=0.33`; `num_grids` mặc định lớp
  = 8 nhưng **mô hình báo cáo `kan_wnetr_base` truyền `num_grids=4`**):
  với mỗi điểm lưới \(g_k\):
  \[
    \phi_k(x) = 1 - \tanh^2\!\Big(\frac{x-g_k}{h}\Big), \quad h=0.33
  \]
  tức hàm "chuông" \(\operatorname{sech}^2\) (đạo hàm của tanh), **không phải B-spline** như KAN gốc
  của Liu et al. Đây là biến thể *FasterKAN* (RBF reflectional switch) chọn vì rẻ và nhanh hơn spline.
- Mỗi token có chiều `hidden` được khai triển thành `hidden × num_grids` đặc trưng cơ sở rồi chiếu
  tuyến tính → đây là chỗ "hàm kích hoạt học được" thực sự nằm.

### B.3 Đường residual SiLU — BẬT trong mọi run báo cáo (`use_base_update=True`, v4)

- `FasterKANLayer` có nhánh phụ: đầu ra = `spline(x) + base_linear(SiLU(LN(x)))`, tức cộng thêm một
  biến đổi tuyến tính trên hàm kích hoạt **SiLU** song song với nhánh cơ sở RBF (lai RBF + base).
- ✅ **Đã chốt:** script train thật là `train_wnetr_networks_v4.py`, và mọi nhánh build KAN ở đó
  (`kan_wnetr`, `kan_wnetr_base`, `kan_wnetr_small`) đều truyền **`use_base_update=True`**
  (dòng 1330, 1349, 1370). Vì vậy **toàn bộ kết quả KAN trong luận dùng nhánh SiLU residual**.
- Hệ quả: số tham số KAN tăng (mỗi `FasterKANLayer` thêm một `nn.Linear(in→out)` có bias). Đây là
  lý do `kan_wnetr_base` = **38.87M** params (xem B.5).
- ⚠️ Đánh giá phải dùng **`evaluate_by_run_id_v4.py`** (dựng model với `use_base_update=True` →
  38,871,394 params). Bản v3 dựng thiếu nhánh này (32.56M) và `load_weights(strict=False)` load lệch
  âm thầm → SSIM rơi về ~0.14–0.46 (rác). Đây là bug đã được phát hiện (mục 4 của `ablation_7_5_findings.md`).

### B.4 Hệ quả về tham số / chuẩn hoá

- Có **2 LayerNorm** trên đường FFN của KAN (norm2 ở block + layernorm nội bộ mỗi FasterKANLayer);
  bản MLP chỉ có **1** (norm2). Đây là khác biệt nhỏ nhưng có thật, ảnh hưởng chuẩn hoá/độ ổn định.
- `SplineLinear` **không có bias**; lớp đầu của KAN có kích thước hiệu dụng `(hidden×8) × mlp_dim` →
  số tham số FFN tăng đáng kể so với `hidden × mlp_dim` của MLP.
- `spline_weight_init_scale = 0.667`, init Xavier-uniform (khác init normal 0.02 mà `init_weights`
  áp cho Conv/Linear thường — `SplineLinear.reset_parameters` tự ghi đè bằng Xavier).

### B.5 Cấu hình thực dùng — mô hình BÁO CÁO là `kan_wnetr_base` (KHÔNG phải `kan_wnetr` default)

Trong `train_wnetr_networks_v4.py` có 3 biến thể KAN. Mô hình đưa vào ablation/kết quả (theo
`ablation_7_5_findings.md`, 38.87M params) là **`kan_wnetr_base`**:

```
# kan_wnetr_base  (ĐÃ DÙNG cho kết quả — dòng 1356-1376)  → ≈ 38.87M tham số
hidden_size = 256,  mlp_dim = 512,  num_heads = 8,  num_layers = 12
num_grids = 4,  use_base_update = True
feature_size = 16,  patch_size = 16,  dropout_rate = 0.2,  spatial_dims = 1
pos_embed = "conv",  norm_name = "instance",  res_block = True

# kan_wnetr  (mặc định, ≈110M — KHÔNG dùng cho kết quả — dòng 1318-1336)
hidden_size = 360,  mlp_dim = 700,  num_heads = 12,  num_grids = 8,  use_base_update = True
```

Baseline so sánh: **MLP-WNETR (`unetr`) = 25.79M** tham số. Như vậy ablation thực tế là:
**MLP 25.8M  ↔  KAN-base 38.9M** (cùng 12 layer, cùng khung W; KAN dùng hidden/heads/grids nhỏ hơn
default để "đúng kích cỡ" triển khai thiết bị).
Skip indices = `[3, 6, 9]` + đầu ra ViT cuối (sau norm) → 4 mức skip, khớp ý "{3,6,9,12}".

### B.6 Phép trừ đặc trưng — ĐÚNG vị trí (cần nói rõ trong luận)

Phép `tanh(z_f − z_m)` **không** xảy ra bên trong vòng lặp 12 encoder layer. Nó xảy ra ở **4 mức
fusion skip-connection của UNETR** (enc1…enc4) khi đưa vào decoder, và **chỉ áp cho nhánh fECG**
(`forward` của `KANWNETR`, dòng ~287–296):
```
dec3_f = decoder5_f(dec4_f, tanh(enc4_f - enc4_m))   # nhánh fECG: có trừ
dec3_m = decoder5_m(dec4_m, enc4_m)                  # nhánh mECG: KHÔNG trừ
... lặp lại cho enc3, enc2, enc1
```
→ Chương 3 (công thức 162 + caption hình) đang ngụ ý phép trừ ở "mỗi bước encoder của Transformer";
cần sửa lại cho đúng: trừ tại **các mức skip CNN của UNETR**, một chiều (m→f).

### B.7 Biến thể phụ (không phải mô hình chính)

- `kan_wnetr_networks` (`networks_kan.py::My_build_KANUNETR`) là biến thể **2 lượt tuần tự**:
  chạy mô hình lấy `mecg_pred`, lấy phần dư `y=x−mecg_pred`, chạy lần 2 lấy `fecg_pred`. Biến thể này
  **bỏ qua** cơ chế trừ đặc trưng nội tại → khác hẳn mô hình đề xuất. Nếu không dùng thì không cần
  nhắc trong luận; nếu có dùng để so sánh thì phải mô tả tách bạch.
- `kan_wnetr_small`: cấu hình nhẹ (hidden=128, mlp=256, heads=8, layers=6, num_grids=4) cho thử nghiệm.

---

## Phần C — Đối chiếu với Chương 3 (`3_DeXuat_ThietLap.tex`): cần viết/sửa gì

### ✅ Phần đang ĐÚNG, giữ nguyên
- Macro-architecture W-NETR (2 nhánh + khử đặc trưng mECG) — đúng.
- "Chỉ thay MLP trong encoder block bằng KAN" ở mức ý tưởng — đúng.
- Hai phương trình block KAN-ViT (MSA residual + KAN residual, công thức 154–159) — khớp code.
- Tiền xử lý: **lọc 3–90 Hz** (code `BANDPASS=(3.0,90.0)`, Butterworth order 3) — **đúng**
  (CLAUDE.md ghi "0.5–90 Hz" là **sai**, bỏ qua).
- Chuẩn hoá theo \(\sigma_y\) = std của tín hiệu nhiễu (code `sigma = mix.std()`, `NORMALIZE_MODE="std"`)
  — **đúng**. Cửa sổ 992 mẫu, 250 Hz — đúng.
- FS=250, độ dài 992 (~3.97 s), patch 16 — đúng.

### ⚠️ Phần SAI / THIẾU — cần sửa

**C.1 (QUAN TRỌNG NHẤT) — "spline activation" là sai loại KAN.**
Mục 3.4.3 và 3.4.5 nói chung chung "KAN / spline activation". Thực tế dùng **FasterKAN** với cơ sở
**RBF Reflectional Switch Function** \(1-\tanh^2((x-g_k)/h)\), **không dùng B-spline**. Cần:
- Thêm một tiểu mục "Cơ sở RBF của FasterKAN" với công thức \(\phi_k\) ở B.2.
- Nêu rõ tham số (theo `kan_wnetr_base`): `num_grids=4`, lưới \([-2,2]\), `denominator=0.33`,
  2 lớp KAN `[256→512→256]`. (Biến thể default dùng `num_grids=8`, `[360→700→360]` nhưng không báo cáo.)
- Sửa câu "spline activation có thể tự thích nghi" → "hàm kích hoạt học được trên nền cơ sở RBF".
- Giải thích lý do chọn FasterKAN thay KAN-spline gốc: chi phí tính/bộ nhớ thấp hơn, ổn định hơn.

**C.2 — Hàm mất mát: Chương 3 chỉ mô tả Huber, thực tế là COMPOSITE 3 số hạng.**
Code v4 (`LOSS_MODE="composite"`, dòng 1672 & 1701) với trọng số mặc định:
\[
  \mathcal{L} = 0.5\,\mathcal{L}_{\text{Huber}} + 0.5\,(1-\text{SSIM}) + 0.2\,\mathcal{L}_{\text{deriv}}
\]
trong đó \(\mathcal{L}_{\text{deriv}}\) là L1 trên đạo hàm thời gian (làm sắc sườn QRS → tăng F1 phát
hiện đỉnh). Hai số hạng `corr`, `sisnr`, `stft` có sẵn nhưng **mặc định = 0** (tắt). Huber còn nhân
trọng số theo nhánh: `LOSS_W_MECG=4.0`, `LOSS_W_FECG=4500.0` (hoặc `--fecg_only` trọng số 4500). Cần:
- Bổ sung số hạng **(1−SSIM)** (0.5) và **đạo hàm thời gian** (0.2) vào mục 3.5; nêu rõ Huber = 0.5.
- Nêu trọng số mất cân bằng mECG/fECG (4 vs 4500) và lý do (fECG biên độ rất nhỏ).
- (Khớp với mô tả "Huber+SSIM+đạo hàm" trong `ablation_7_5_findings.md`. KHÔNG mô tả STFT — đang tắt.)

**C.3 — Bảng siêu tham số (Bảng 3.x) cần sửa và bổ sung — dùng config `kan_wnetr_base`.**
- Optimizer: ghi **AdamW** (weight_decay=0.01), không phải "Adam".
- Kiến trúc đề xuất (kan_wnetr_base): `hidden_size=256`, `mlp_dim=512`, `num_heads=8`,
  `num_layers=12`, `num_grids=4`, `use_base_update=True`, `dropout=0.2`, `patch_size=16`,
  `feature_size=16`. (≈38.87M params; baseline MLP 25.79M.) **Đừng** ghi 360/700/8 (đó là biến thể
  default không dùng cho kết quả).
- Phác đồ huấn luyện theo run báo cáo (finetune ADFECGDB): **AdamW, lr=2e-5, 150 epoch, batch=64,
  scheduler cosine + warmup, chuẩn hoá `std`** (theo `ablation_7_5_findings.md`). Bỏ con số 80 epoch
  / batch 32 đang mâu thuẫn trong TODO dòng 250.
- Nếu mô tả cả pretrain (mô phỏng) thì nêu riêng (10-sub, 8ch) — khác recipe finetune.

**C.4 — Vị trí phép trừ đặc trưng (công thức 162 + caption Hình KAN-WNETR).**
Sửa cho đúng: phép \(\tanh(z^{(f)}-z^{(m)})\) áp tại **4 mức skip-connection CNN của UNETR** (không
phải "mỗi bước Transformer encoder"), và **chỉ một chiều** (trừ vào nhánh fECG; nhánh mECG giữ skip
nguyên). Xem B.6.

**C.5 — Cờ `use_base_update` (nhánh SiLU) — ĐÃ CHỐT: BẬT.**
Mọi run báo cáo dùng `use_base_update=True`, nên **bắt buộc** mô tả FasterKAN ở dạng lai trong luận:
\[
  \text{KAN}(x) = \underbrace{W_s\,\Phi_{\text{RBF}}(\text{LN}(x))}_{\text{nhánh cơ sở RBF}}
                + \underbrace{W_b\,\text{SiLU}(\text{LN}(x))}_{\text{nhánh residual}} .
\]
Đây là điểm khác biệt kiến trúc thật so với MLP (và so với FasterKAN thuần RBF). Nên giải thích lý do:
nhánh SiLU residual ổn định tối ưu hoá, tăng tốc hội tụ, để nhánh RBF lo chi tiết cục bộ.

**C.6 — Hai LayerNorm trên đường FFN của KAN.**
Có thể nêu ngắn: FasterKANLayer tự chuẩn hoá đầu vào (LN nội bộ) nên đường KAN có thêm 1 LN so với
MLP gốc — giúp ổn định huấn luyện KAN.

### 📝 Việc khác (chất lượng trình bày)
- Mục 3.4.1 "Ý tưởng tổng thể" còn ở dạng gạch đầu dòng (TODO) → viết thành văn xuôi.
- Mục FHR error còn TODO → viết thành văn xuôi.
- Đồng bộ thuật ngữ "KAN-ViT" / "FasterKAN" xuyên suốt; định nghĩa FasterKAN ngay lần đầu xuất hiện.

---

## Phần D — Checklist sửa Chương 3 (ưu tiên giảm dần)

1. [ ] Thêm tiểu mục cơ sở RBF FasterKAN + công thức \(\phi_k\); sửa mọi chỗ "B-spline/spline". **(C.1)**
2. [ ] Viết lại mục 3.5: composite loss Huber + (1−SSIM), trọng số 0.5/0.5 và 4 vs 4500. **(C.2)**
3. [ ] Sửa bảng siêu tham số: AdamW, bổ sung 360/700/12/12/8/0.2/16; chốt epochs. **(C.3)**
4. [ ] Sửa mô tả + caption phép trừ đặc trưng (4 mức skip CNN, một chiều). **(C.4)**
5. [ ] Mô tả FasterKAN dạng LAI (RBF + SiLU residual) vì `use_base_update=True`. **(C.5)**
6. [ ] (tuỳ) Nêu LN nội bộ của FasterKANLayer. **(C.6)**
7. [ ] Hoàn thiện các đoạn TODO thành văn xuôi (3.4.1, FHR, mâu thuẫn epochs/batch).

---

## Phụ lục — Trích dẫn vị trí mã nguồn

- FFN KAN: `models/kantransformerblock.py:20-45` (FasterKANBlock), `:48-84` (KANTransformerBlock).
- Cơ sở RBF: `models/fasterkan.py:17-40` (ReflectionalSwitchFunction), `:43-84` (FasterKANLayer),
  `:100-131` (FasterKAN stack).
- Mô hình W: `models/unetr_kan_wnetr.py:265-300` (forward, phép trừ tanh ở skip).
- Baseline MLP: `models/transformerblock.py:16-93`, `models/vit.py:18,94`, `models/unetr.py:289-326`.
- Build & cấu hình (script thật v4): `train_wnetr_networks_v4.py:1318-1336` (kan_wnetr),
  `:1356-1376` (**kan_wnetr_base — model báo cáo**, `use_base_update=True`),
  `:117-124` (định nghĩa composite loss + trọng số), `:1672-1673` & `:1701-1702` (lắp composite).
- Nhánh SiLU residual: `models/fasterkan.py:64-67, 81-83`.
- Eval đúng version: `evaluate_by_run_id_v4.py` (v3 load lệch → rác); xem `ablation_7_5_findings.md` mục 4.
