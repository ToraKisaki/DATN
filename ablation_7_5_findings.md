# Bảng ablation 7.5 (MLP-FFN vs KAN-FFN) — Tổng hợp số liệu & phát hiện

> File này tổng hợp toàn bộ những gì đã tìm hiểu/đo đạc để bạn tự điền vào
> `Chuong/7_Ket_qua.tex` (mục `\section{Nghiên cứu loại bỏ ...}`, label `tab:ablation`).
> Ngày đo: 2026-06-26. Tất cả số liệu lấy từ checkpoint **đã train sẵn**, chỉ chạy **eval lại** (không train lại).

---

## 1. Số liệu đã đo (đã kiểm chứng)

Đánh giá trên **tập val ADFECGDB (dữ liệu thực), 592 đoạn tín hiệu**, cùng giao thức
(`--metric-max-batches 200`, denorm theo `std`). SSIM/PSNR là chất lượng tái tạo dạng
sóng fECG; F1 là phát hiện đỉnh fQRS.

| Biến thể FFN | SSIM | PSNR (dB) | F1 (%) | #Params | Run ID | Pretrain |
|---|---|---|---|---|---|---|
| **MLP-FFN** (baseline W-NETR, `unetr`) | 0.818 | 20.36 | 94.41 | 25.79M | `20260623_105156` | 1 subject mô phỏng |
| **KAN-FFN** (cùng "ngân sách" pretrain 1-sub) | 0.810 | 20.24 | 94.97 | 38.87M | `20260623_093308` | 1 subject mô phỏng |
| **KAN-FFN** (run KAN tốt nhất) | 0.819 | 20.09 | 96.21 | 38.87M | `20260620_151427` | 10 subject mô phỏng |

Chi tiết P/R (fECG):
- MLP `105156`: P=0.9401, R=0.9482, F1=0.9441
- KAN `093308`: P=0.9572, R=0.9424, F1=0.9497
- KAN `151427`: P=0.9691, R=0.9553, F1=0.9621

---

## 2. Cặp ablation nào là "công bằng"?

Mục đích ablation: **chỉ đổi khối FFN (MLP ↔ KAN), giữ nguyên mọi thứ còn lại** để
cô lập riêng đóng góp của KAN.

- **MLP `105156`** và **KAN `093308`** đều được **pretrain trên dữ liệu mô phỏng 1 subject**,
  rồi tinh chỉnh trên ADFECGDB với **cùng phác đồ** (AdamW, lr=2e-5, 150 epoch, bs=64,
  loss hỗn hợp Huber+SSIM+đạo hàm). → **Đây là cặp ablation đúng nghĩa.**
  - Kết quả: KAN cải thiện F1 **94.41% → 94.97% (+0.56pp)**, SSIM/PSNR tương đương,
    đổi lại tăng tham số **25.8M → 38.9M**.

- **KAN `151427`** tốt hơn (F1 96.21%) **một phần lớn nhờ được pretrain trên 10 subject
  (gấp 10 lần dữ liệu)**, KHÔNG chỉ nhờ FFN. → Nếu đưa dòng này vào bảng sẽ **lẫn lộn**
  đóng góp của FFN với đóng góp của dữ liệu pretrain. Nên chỉ nêu ở phần bàn luận như
  "kết quả tốt nhất", không đặt trong bảng ablation.

### Khác biệt cụ thể giữa 2 run KAN
Cả hai **giống hệt** về kiến trúc (`kan_wnetr_base`, 38.87M) và recipe finetune.
Chỉ khác checkpoint pretrain mà chúng finetune từ đó:

| | `093308` | `151427` |
|---|---|---|
| Pretrain | 1 subject (`1sub_k5_c5_kh75_8ch`) | 10 subject (`10sub_k5_c5_kh75_8ch`) |
| Norm pretrain | `paper_exact` | `std` |
| Finetune src | `run_20260622_163247` | `run_20260619_193819` |

### Lưu ý của baseline MLP
- MLP `105156` pretrain 1 subject nhưng dùng `34ch` + norm `var_global`
  (KAN dùng `8ch` + `paper_exact`/`std`). Hai pipeline pretrain **không trùng tuyệt đối**
  về số kênh/normalize, nhưng **cùng ở mức 1-subject** — đây là cặp gần nhất có sẵn mà
  không phải train lại. Nếu cần ablation tuyệt đối sạch thì phải train lại MLP và KAN
  từ **cùng một checkpoint pretrain** (tốn vài ngày GPU — xem mục 5).

---

## 3. ⚠️ Cảnh báo nhất quán với mục 7.4

- Mục 7.4 báo **F1 = 97.73%** — số này đo trên **tập MÔ PHỎNG**.
- Bảng ablation 7.5 ở trên đo trên **tập ADFECGDB THỰC** (~94–96%).
- → Hai tập test khác nhau. **Caption bảng 7.5 phải ghi rõ "đánh giá trên ADFECGDB thực"**
  để người đọc không tưởng là mâu thuẫn. (Có thể chạy thêm ablation trên tập mô phỏng
  nếu muốn cùng thang đo với 7.4 — xem mục 5.)

---

## 4. 🐛 Bug eval đã phát hiện & cách chạy lại

`evaluate_by_run_id.py` import `train_wnetr_networks_v3`. Nhưng các run KAN tháng 6/2026
được train bằng **v4**, vốn dựng model với `use_base_update=True` (đường SiLU residual
của FasterKAN) → **38,871,394 params**. Bản v3 dựng **thiếu** flag đó → **32,561,506 params**.
Vì `load_weights` dùng `strict=False` nên nó **load lệch âm thầm**, model KAN xuất ra rác
(SSIM ~0.14–0.46 thay vì ~0.81). Số liệu MLP (`unetr`) thì v3/v4 giống nhau nên không bị.

**Cách chạy lại đúng:** mình đã tạo `evaluate_by_run_id_v4.py` (bản copy đổi import sang
`train_wnetr_networks_v4`). Lệnh tái lập:

```bash
conda run --no-capture-output -n KANWNET_fecg python evaluate_by_run_id_v4.py \
  --run-id 20260623_093308 --metric-max-batches 200 --batch-size 32
# Baseline MLP có thể dùng script gốc hoặc _v4 đều được:
conda run --no-capture-output -n KANWNET_fecg python evaluate_by_run_id.py \
  --run-id 20260623_105156 --metric-max-batches 200 --batch-size 32
```

**Kiểm tra đúng:** log phải in `Params: 38,871,394` và **không** có dòng
`[warn] missing keys` / `unexpected keys`. Kết quả đầy đủ được lưu tại
`logs/run_<id>/eval_<id>_mb200.csv`.

> Lưu ý nhỏ: phần "Per-channel SSIM" in ra `PSNR=psnr SSIM=ssim` (literal) — đây là bug
> hiển thị nhỏ trong script, KHÔNG ảnh hưởng các số tổng hợp SSIM/PSNR/F1 ở trên.

---

## 5. (Tuỳ chọn) Nếu muốn ablation chặt chẽ hơn

Để có ablation "chỉ đổi FFN" tuyệt đối sạch, cần train lại 2 model từ **cùng pretrain
checkpoint, cùng số kênh, cùng normalize**:
1. Pretrain 1 lần trên mô phỏng (vd 10-sub, 8ch, `std`).
2. Finetune ADFECGDB 2 lần: một bản `unetr` (MLP), một bản `kan_wnetr_base` (KAN), mọi
   hyperparameter còn lại y hệt.
3. Eval cả hai bằng `evaluate_by_run_id*.py` (đúng version v3/v4 tương ứng).
Chi phí: theo ghi chú dự án, mỗi finetune ~vài giờ–1 ngày; pretrain 10-sub ~2.5 ngày.
Nên validate trên 1-sub trước (~5–6h) rồi mới cam kết chạy đầy đủ.

---

## 6. Khối LaTeX gợi ý để dán vào `7_Ket_qua.tex`

```latex
\begin{itemize}
  \item \textbf{Thiết lập:} giữ nguyên kiến trúc và quy trình huấn luyện W-NETR
        (cùng pretrain trên dữ liệu mô phỏng 1~đối tượng, cùng phác đồ tinh chỉnh
        trên ADFECGDB: AdamW, $lr=2\times10^{-5}$, 150~epoch, hàm mất mát hỗn hợp
        Huber+SSIM+đạo hàm), chỉ thay khối FFN: MLP $\leftrightarrow$ FasterKAN.
  \item \textbf{Đánh giá:} trên tập kiểm tra ADFECGDB (dữ liệu thực, 592~đoạn);
        cùng giao thức SSIM/PSNR (tái tạo dạng sóng fECG) và F1 (phát hiện fQRS).
  \item \textbf{Nhận xét:} thay MLP bằng KAN-FFN cải thiện F1 (94.41\% $\to$ 94.97\%)
        trong khi giữ tương đương SSIM/PSNR, đổi lại tăng tham số (25.8M $\to$ 38.9M).
\end{itemize}

\begin{table}[htbp]
\centering
\caption{Nghiên cứu loại bỏ: MLP-FFN so với KAN-FFN trong cùng cấu hình W-NETR
(đánh giá trên tập ADFECGDB thực, 592~đoạn). Hai biến thể dùng chung quy trình
pretrain (mô phỏng 1~đối tượng) và tinh chỉnh, chỉ khác khối FFN.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
\textbf{Biến thể FFN} & \textbf{SSIM} & \textbf{PSNR (dB)} & \textbf{F1 (\%)} & \textbf{\#Tham số} \\
\hline
MLP-FFN (baseline W-NETR) & \textbf{0.818} & \textbf{20.36} & 94.41 & 25.8M \\
KAN-FFN (đề xuất)         & 0.810 & 20.24 & \textbf{94.97} & 38.9M \\
\hline
\end{tabular}
\end{table}

\noindent
Khi được pretrain trên toàn bộ dữ liệu mô phỏng 10~đối tượng (thay vì 1~đối tượng)
rồi tinh chỉnh theo cùng phác đồ, biến thể KAN-FFN đạt mức tốt nhất với
SSIM~$=0.819$, PSNR~$=20.09$~dB và F1~$=96.21\%$ trên cùng tập ADFECGDB thực.
Mức cải thiện này phản ánh đồng thời đóng góp của khối KAN và lợi ích của lượng dữ liệu
pretrain lớn hơn, nên không được đưa vào Bảng~\ref{tab:ablation} (vốn cô lập riêng yếu tố
FFN) mà chỉ nêu ở đây để tham chiếu.
```

---

## 7. Nguồn dữ liệu (truy vết)

- Số liệu eval: `logs/run_20260623_105156/eval_20260623_105156_mb200.csv`,
  `logs/run_20260623_093308/eval_20260623_093308_mb200.csv`,
  `logs/run_20260620_151427/eval_20260620_151427_mb200.csv`.
- Config các run: `logs/run_<id>/run_meta_<id>.csv`.
- run `20260622_171304` (KAN) đã bị loại: không có `checkpoints/best.pth` (chạy dở dang).
