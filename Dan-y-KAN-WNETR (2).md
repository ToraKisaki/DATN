# Dàn ý đồ án — Hệ thống trích xuất điện tim thai không xâm lấn KAN-WNETR

> Khung được thiết kế lại theo hướng **khoa học**: tách bạch rõ *kiến thức nền đã có* khỏi *đóng góp mới*, đi theo mạch IMRaD mở rộng (Vấn đề → Nền tảng → Phương pháp → Thực nghiệm → Hệ thống → Kết luận). Đồ án có **ba đóng góp**: (1) thuật toán KAN-WNETR; (2) tối ưu & triển khai biên thời gian thực; (3) hệ thống theo dõi lâm sàng. Mỗi chương đều có *mở đầu chương* và *kết luận chương*.

---

## Phần đầu (Front matter)

- Bìa chính / bìa phụ (có ô chữ ký GVHD)
- **Lời cảm ơn**
- **Lời cam đoan** *(bắt buộc với ĐATN, hội đồng có chấm)*
- **Tóm tắt** (tiếng Việt)
- **Abstract** (tiếng Anh)
- **Mục lục**
- **Danh mục từ viết tắt** — fECG, mECG, maECG/aECG, NI-fECG, FHR, MHR, SNR, CTG, QRS, KAN, MLP/FFN, MSA, ViT, UNETR, W-NETR, SSIM, PSNR, MSE, ONNX, INT8, SPI, ADS1293, FIGO/NICHD…
- **Danh mục hình vẽ**
- **Danh mục bảng biểu**

---

## Chương 1 — Giới thiệu chung

Định vị vấn đề, phát biểu rõ mục tiêu và phạm vi, nêu đóng góp. Đây là chương hội đồng đọc kỹ nhất.

- **1.1 Đặt vấn đề và bối cảnh** — tử vong chu sinh & dị tật tim bẩm sinh; vai trò theo dõi fHR/hình thái; hạn chế của CTG/Doppler (mất tín hiệu, an toàn sóng siêu âm dài hạn); xâm lấn vs không xâm lấn; bài toán cốt lõi: tách fECG khỏi maECG đơn kênh trong điều kiện **SNR thấp** và **mECG–fECG chồng lấp**.
- **1.2 Tính cấp thiết** — nhu cầu giải pháp an toàn, chi phí thấp, theo dõi dài hạn/tại nhà; khoảng trống của thiết bị thương mại.
- **1.3 Mục tiêu nghiên cứu** — phát biểu **3 mục tiêu** tường minh: (a) đề xuất mô hình KAN-WNETR nâng độ chính xác & bảo toàn hình thái; (b) tối ưu mô hình để chạy **thời gian thực trên thiết bị biên**; (c) xây dựng **hệ thống theo dõi** đầu giường + dashboard lâm sàng.
- **1.4 Đối tượng và phạm vi** — đối tượng: tín hiệu maECG đơn kênh. Phạm vi: huấn luyện/đánh giá trên **dữ liệu mô phỏng FECGSYNDB**, triển khai trên **Raspberry Pi 5**, kiểm thử bằng mô phỏng/playback; **chưa** dùng dữ liệu lâm sàng thực (nêu thẳng để tránh bị hỏi).
- **1.5 Phương pháp nghiên cứu** — quy trình: nghiên cứu lý thuyết → đề xuất kiến trúc → thực nghiệm có đối chứng → tối ưu/triển khai → kiểm thử hệ thống.
- **1.6 Đóng góp chính** — liệt kê gạch đầu dòng 3 đóng góp ở trên (đây là phần làm nổi "tính mới").
- **1.7 Bố cục báo cáo** — đảm bảo số chương khớp đúng nội dung (bản cũ bị lệch số).

---

## Chương 2 — Cơ sở về điện tim thai và theo dõi thai nhi

Nền tảng y–sinh. Tham chiếu khung Sarafan (Ch2).

- **2.1 Giới thiệu chương**
- **2.2 Nguyên lý và đặc tính tín hiệu ECG/fECG** — chu trình P–QRS–T; khác biệt fECG vs mECG (biên độ, SNR); vì sao fECG khó tách.
- **2.3 Ý nghĩa lâm sàng của theo dõi tim thai** — fHR, hình thái, phát hiện sớm bất thường.
- **2.4 Các phương thức theo dõi thai** — CTG, Doppler, PCG, fECG xâm lấn vs không xâm lấn; ưu/nhược từng loại.
- **2.5 Tổng quan thiết bị fECG thương mại** — Monica/Nemo/… và khoảng trống (chi phí, phổ thông, chỉ cho fHR).
- **2.6 Kết luận chương**

---

## Chương 3 — Tổng quan các phương pháp tách fECG không xâm lấn

Khảo sát SOTA, kết thúc bằng việc **định vị đề tài**. Tham chiếu Sarafan (Ch3).

- **3.1 Giới thiệu chương** — bài toán tách & vai trò tiền xử lý.
- **3.2 Nhóm phương pháp cổ điển**
  - 3.2.1 Lọc thích nghi (LMS/NLMS/RLS)
  - 3.2.2 Kalman/Extended Kalman Filter
  - 3.2.3 Tách nguồn mù — ICA và biến thể (FastICA, RobustICA, JADE)
  - 3.2.4 Khử mẫu mECG — Template Subtraction & TSc
  - 3.2.5 Phương pháp lai TS–ICA
- **3.3 Nhóm phương pháp học sâu**
  - 3.3.1 CNN
  - 3.3.2 RNN (LSTM/GRU)
  - 3.3.3 Lai CNN–RNN
  - 3.3.4 Mô hình sinh (GAN/CycleGAN)
  - 3.3.5 Autoencoder & họ U-Net; hướng kiến trúc hai nhánh (dual Res-UNet, W-Net)
- **3.4 Nhận xét, khoảng trống nghiên cứu và định vị đề tài** — chỉ rõ W-NETR là SOTA hiện tại, hạn chế của MLP-FFN cố định, và **lý do chọn KAN** làm hướng cải tiến.
- **3.5 Kết luận chương**

---

## Chương 4 — Cơ sở lý thuyết học sâu *(CHƯƠNG MỚI — cần bổ sung)*

Chương "kiến thức nền đã có" — **chỉ chứa lý thuyết chuẩn**, không chứa đóng góp. Tham chiếu document.pdf (Ch2). Tách phần này ra giúp Chương 5 chỉ còn nói về cái mới ⇒ rất "khoa học".

- **4.1 Giới thiệu chương**
- **4.2 Mạng nơ-ron và khối MLP/FFN** — kích hoạt phi tuyến cố định, vai trò channel-mixing.
- **4.3 Tích chập 1D (CNN) cho tín hiệu thời gian** — receptive field và hạn chế.
- **4.4 Cơ chế Attention và Transformer encoder** — Multi-Head Self-Attention, LayerNorm, positional encoding, residual; kèm công thức.
- **4.5 Kiến trúc U-Net và UNETR** — encoder Transformer + decoder CNN + skip connection; patchify 1D.
- **4.6 Kolmogorov–Arnold Networks (KAN)** — định lý biểu diễn KA; learnable activation trên cạnh; tham số hóa spline (B-spline); residual activation; so sánh KAN vs MLP (biểu diễn, diễn giải, độ phức tạp, tối ưu hóa).
- **4.7 Hàm mất mát cho hồi quy tín hiệu** — MSE/MAE/Huber và lý do dùng Huber.
- **4.8 Kết luận chương**

---

## Chương 5 — Mô hình đề xuất KAN-WNETR

Chương "đóng góp thuật toán" — **chỉ nói cái mới**, dựa trên nền Chương 4.

- **5.1 Giới thiệu chương và ý tưởng tổng thể**
- **5.2 Kiến trúc nền W-NETR** — hai nhánh UNETR 1D (mECG / fECG); cơ chế *mECG feature elimination* (trừ đặc trưng + tanh).
- **5.3 KAN-ViT: thay MLP/FFN bằng KAN trong encoder block** — công thức khối encoder sau thay thế.
- **5.4 Tích hợp KAN vào W-NETR → KAN-WNETR** — giữ macro-architecture (L=12, skip {3,6,9,12}, patchify), điểm ghép W, hàm tanh.
- **5.5 Phân tích thiết kế và kỳ vọng** — vì sao can thiệp đúng vào MLP; lợi ích bảo toàn hình thái P/QRS/T và độ bền với nhiễu.
- **5.6 Tổng hợp đóng góp kiến trúc**
- **5.7 Kết luận chương**

---

## Chương 6 — Dữ liệu, tiền xử lý và thiết lập thực nghiệm

Tách riêng phần "vật liệu & phương pháp thực nghiệm" để Chương 7 chỉ trình bày kết quả.

- **6.1 Giới thiệu chương**
- **6.2 Bộ dữ liệu FECGSYNDB** — mô phỏng FECGSYN; 10 thai kỳ; 250 Hz/16-bit; 34 kênh; ngữ cảnh C0–C5; 5 mức SNR; minh họa hình học vị trí điện cực.
- **6.3 Chiến lược chia dữ liệu** — subject-independent (đối tượng 10 làm test); chọn kênh maECG 11/19/22/25.
- **6.4 Tiền xử lý** — lọc thông dải 3–90 Hz; cửa sổ 992 mẫu (~3.97 s); chuẩn hóa z-score theo độ lệch chuẩn; khôi phục biên độ.
- **6.5 Phát hiện QRS** — thuật toán Pan–Tompkins (các bước + công thức).
- **6.6 Chỉ số đánh giá** — Precision/Recall/F1 (phát hiện fQRS); PSNR/SSIM/MSE (chất lượng tín hiệu); **bổ sung sai số FHR (bpm)** vì hệ thống đã tính được FHR.
- **6.7 Cấu hình huấn luyện** — hyperparameters (chốt một bộ số nhất quán), optimizer Adam, loss Huber, môi trường phần cứng/thư viện.
- **6.8 Kết luận chương**

---

## Chương 7 — Kết quả thực nghiệm và đánh giá

Phần "Results & Discussion". Đây là nơi ăn điểm "hàm lượng khoa học".

- **7.1 Giới thiệu chương**
- **7.2 Kết quả định tính** — dạng sóng fECG tái tạo theo SNR/case *(dùng hình thật của fECG, thay đoạn võng mạc cũ)*.
- **7.3 Chất lượng tín hiệu định lượng** — bảng SSIM/PSNR theo C0–C5; biện luận vì sao tốt ở nhiễu khó (C3–C4).
- **7.4 Phát hiện fQRS và ước lượng FHR** — F1 + **sai số FHR (bpm)**.
- **7.5 So sánh với SOTA** — bảng EKF/TSPCA/RCED-Net/Res-UNet/CycleGAN/W-net/W-NETR.
- **7.6 Nghiên cứu loại bỏ (ablation): MLP-FFN vs KAN-FFN** — *điểm mấu chốt*: chỉ thay đúng khối FFN trong cùng cấu hình W-NETR để **định lượng riêng đóng góp của KAN** (nếu chưa chạy thì đây là việc ưu tiên số 1).
- **7.7 Phân tích độ phức tạp** — số tham số, FLOPs, kích thước mô hình, thời gian suy luận; làm cầu nối sang Chương 8.
- **7.8 Bàn luận** — điểm mạnh, hạn chế, giải thích xu hướng; **đối chiếu chênh lệch SSIM giữa FP32 và bản INT8 triển khai**.
- **7.9 Kết luận chương**

---

## Chương 8 — Thiết kế và triển khai hệ thống theo dõi fECG thời gian thực

Chương "đóng góp hệ thống" — phần lớn nhất, dựa trên `raspi-deploy` + `raspi-fecg-server`. Trình bày theo phân hệ kiểu đồ án hệ thống điểm cao.

- **8.1 Giới thiệu chương và yêu cầu hệ thống** — yêu cầu chức năng/phi chức năng (thời gian thực, gọn nhẹ, an toàn dữ liệu).
- **8.2 Kiến trúc tổng quan** — sơ đồ 3 tầng: **thiết bị (Pi 5) ↔ server (PC) ↔ dashboard web**; làm rõ ranh giới Pi/PC (monitor PyQt5 chạy cục bộ trên Pi; server FastAPI chạy trên PC; dashboard xem được trên LCD Pi qua kiosk hoặc bất kỳ trình duyệt LAN).
- **8.3 Tối ưu mô hình cho thiết bị biên**
  - 8.3.1 Xuất ONNX
  - 8.3.2 Lượng tử hóa INT8 động (MatMul/Gemm)
  - 8.3.3 ONNX Runtime trên Pi 5 & benchmark — latency ~80–120 ms/cửa sổ, ~8–12× thời gian thực; đánh đổi chất lượng INT8 vs FP32.
- **8.4 Phân hệ thu nhận & thiết bị (Raspberry Pi 5)**
  - 8.4.1 Phần cứng — Pi 5, ADS1293, LCD 3.5", nguồn/battery HAT.
  - 8.4.2 Driver ADS1293 — SPI mềm trên header trên để tránh xung đột chân LCD; cấu hình 3-lead; **RLD trên IN4**; 24-bit→mV.
  - 8.4.3 Xử lý thời gian thực — ring buffer 992, tái suy luận mỗi 200 ms, replay mẫu mới.
  - 8.4.4 Giao diện monitor đầu giường — PyQt5/pyqtgraph 480×320, hai đường cuộn raw + fECG.
  - 8.4.5 Cơ chế dự phòng — simulator & bộ lọc thay thế khi thiếu phần cứng/mô hình.
- **8.5 Phân hệ server & phân tích lâm sàng (FastAPI + SQLite)**
  - 8.5.1 Giao tiếp WebSocket & xác thực token thiết bị.
  - 8.5.2 Ước lượng FHR (từ fECG) / MHR (từ raw) / chỉ số chất lượng tín hiệu.
  - 8.5.3 Cảnh báo lâm sàng — ngưỡng NICHD/ACOG/FIGO (brady/tachy/mất tín hiệu).
  - 8.5.4 Lưu trữ & mô hình dữ liệu — schema SQLite, lưu 1/4 mẫu.
- **8.6 Phân hệ giao diện web cho bác sĩ** — login, dashboard đa bệnh nhân (FHR + sparkline + cảnh báo), màn hình bệnh nhân (tile FHR/MHR, thanh chất lượng, banner cảnh báo, freeze); thiết kế cảm ứng 480×320.
- **8.7 Thiết kế cơ khí/vỏ hộp** *(nếu đưa vào)* — Fusion 360, in 3D, ràng buộc kích thước/nhiệt.
- **8.8 Kiểm thử và đánh giá hệ thống** — kịch bản end-to-end, độ trễ chuỗi, độ ổn định, kiểm thử xác thực/ngắt kết nối.
- **8.9 Kết luận chương**

---

## Chương 9 — Kết luận và hướng phát triển

- **9.1 Kết luận** — đối chiếu kết quả đạt được với **3 mục tiêu** ở Chương 1.
- **9.2 Hạn chế** — dữ liệu mô phỏng; khoảng cách fQRS so với SOTA; chất lượng tụt sau INT8; chưa kiểm chứng lâm sàng.
- **9.3 Hướng phát triển** — thu **dữ liệu lâm sàng thực** (chính bộ thu ADS1293 của đề tài có thể đảm nhiệm); đa kênh; ablation sâu hơn; quantization-aware training; tiến tới chuẩn thiết bị y tế.
- **9.4 Bài học kinh nghiệm**

---

## Tài liệu tham khảo

Thống nhất **một kiểu trích dẫn duy nhất** (IEEE đánh số) — bản cũ đang trộn tên–năm ở Chương 1 với số ở phần sau.

## Phụ lục

- **A.** Sơ đồ chân & bảng đấu nối ADS1293 ↔ Pi 5
- **B.** Bảng register ADS1293 đã cấu hình
- **C.** Danh sách REST API & giao thức WebSocket
- **D.** Lược đồ cơ sở dữ liệu SQLite
- **E.** Hướng dẫn cài đặt & chạy hệ thống
- **F.** *(tuỳ chọn)* Trích đoạn mã nguồn quan trọng

---

## Nguyên tắc khiến bố cục này "khoa học"

1. **Tách bạch đã-biết và cái-mới**: Ch2–4 chỉ là nền tảng/khảo sát; đóng góp gói gọn ở Ch5 (thuật toán), Ch7 (kết quả + ablation), Ch8 (hệ thống). Đây là khác biệt lớn nhất so với bản cũ (vốn nhồi lý thuyết nền vào chương "mô hình đề xuất").
2. **Mạch lập luận IMRaD mở rộng**: Vấn đề → Nền tảng → Phương pháp → Thực nghiệm → Hệ thống → Kết luận.
3. **Mỗi chương có mở đầu + kết luận chương** (đúng tiêu chí hội đồng).
4. **Mọi khẳng định có trích dẫn; mọi hình/bảng được đánh số và *được nhắc tới* trong văn bản.**
5. **Ba trụ "hàm lượng khoa học"**: ablation (định lượng đóng góp KAN), phân tích độ phức tạp, và đối chiếu FP32↔INT8 — những thứ phản biện hay hỏi.

> **Co giãn theo phạm vi**: nếu là đồ án môn (không phải ĐATN), có thể gộp Ch2+Ch3, rút gọn Ch4 vào phần đầu Ch5, và nén các mục con của Ch8 — vẫn giữ nguyên mạch logic 6 khối lớn.
