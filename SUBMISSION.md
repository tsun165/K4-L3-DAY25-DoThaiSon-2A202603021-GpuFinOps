# Báo cáo Kỹ thuật Tối ưu hóa Chi phí GPU (GPU FinOps Report) — NimbusAI

**Học viên:** Đỗ Thái Sơn  
**Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
**Đơn vị:** NimbusAI  
**Thời điểm đánh giá:** Snapshot Tháng 6/2026  

---

## 1. Tổng quan Baseline vs. Optimized

Trong vai trò FinOps Lead tại startup LLM NimbusAI, chúng tôi đã tiến hành kiểm toán toàn diện hóa đơn hạ tầng GPU, tập trung vào đơn vị kinh tế cốt lõi là **`$/1M-token`** thay vì đơn thuần theo dõi `$/GPU-giờ`.

### Số liệu tổng hợp

| Chỉ số | Baseline (Trước tối ưu) | Optimized (Sau tối ưu) | Mức độ cải thiện / Tiết kiệm |
|---|---|---|---|
| **Tổng ngân sách hàng tháng** | **$27,133** | **$14,577** | **-$12,556 (-46.3%)** |
| **Đơn giá inference (`$/1M-token`)** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **-82.6%** |
| **Chi phí inference hàng ngày** | $48.87 / ngày | $8.48 / ngày | -$40.39 / ngày |
| **Chi phí workload/compute hàng tháng** | $25,667 / tháng | $15,578 / tháng | -$10,089 / tháng (-39.3%) |
| **Lãng phí GPU nhàn rỗi (Idle Waste)** | $600 / tháng | $0 / tháng | Triệt tiêu 100% |
| **Lãng phí do Over-provisioning (Util-lies)**| $655 / tháng | $0 / tháng | Tiết kiệm $655 / tháng |

---

## 2. Phân tích chi tiết các đòn bẩy tiết kiệm (Savings Breakdown by Lever)

Hóa đơn chi phí được tối ưu thông qua 4 đòn bẩy chính (hiển thị trực quan qua biểu đồ Waterfall `outputs/savings.png`):

| Đòn bẩy (Lever) | Tiết kiệm (USD/tháng) | Tỷ trọng đóng góp | Phân tích cơ chế & Lý do đóng góp |
|---|---|---|---|
| **Purchasing Strategy (Spot & Reserved)** | **$10,089** | **80.3%** | **Đóng góp lớn nhất.** Tận dụng đặc tính workload: các job training/finetune/batch có thể ngắt quãng (`interruptible=1`) được chuyển sang Spot instance (tiết kiệm 40–60% sau khi bù trừ chi phí checkpointing). Các job inference 24/7 ổn định (`duty cycle >= 55%`) được cam kết Reserved 3 năm để hưởng mức chiết khấu 45%. |
| **Inference Levers (Cascade, Cache, Batch)** | **$1,212** | **9.7%** | Kết hợp 3 kỹ thuật: định tuyến tầng (Cascade) đẩy các tác vụ cơ bản sang model `small` ($0.20/$0.40 thay vì $3/$15), Prompt Caching giảm 90% chi phí input lặp lại, và Batch API giảm 50% cho các tác vụ phi thời gian thực. Stack discount đạt tới 0.05 (giảm 95%). |
| **Right-size Util-lies** | **$655** | **5.2%** | Chuyển đổi GPU bị định danh sai hiệu năng thực tế (`gpu-h100-4`, `gpu-a10g-1`) sang GPU đúng kích cỡ (right-sizing từ H100 sang A100, A10G sang L4), tránh trả tiền cho phần cứng thừa thãi. |
| **Kill Idle GPUs** | **$600** | **4.8%** | Tắt tự động các GPU chạy dưới 10% công suất trong nhiều giờ (phát hiện `gpu-h100-5` bị bỏ không 8h/ngày lãng phí $20/ngày). |

---

## 3. Bản chất của hiện tượng "GPU-Util Lie"

### Hiện tượng thực tế trong dữ liệu
Qua kiểm toán M1, telemetry ghi nhận:
- **`gpu-h100-4`**: `nvidia-smi` báo **98.2% GPU-Util**, nhưng **MFU chỉ đạt 0.194 (19.4%)** và MBU chỉ đạt **0.207 (20.7%)**.
- **`gpu-a10g-1`**: `GPU-Util = 96.9%`, nhưng **MFU chỉ đạt 0.268 (26.8%)**.

### Giải thích cơ chế kỹ thuật
1. **Bản chất của `nvidia-smi`:** Chỉ báo `gpu_util_pct` phản ánh **tỷ lệ thời gian mà ít nhất một kernel đang active trên GPU** trong cửa sổ đo mẫu (clock đang chạy), hoàn toàn **không phản ánh số lượng tensor core hay toán tử số học thực sự được thực thi**.
2. **Nghẽn băng thông bộ nhớ (Memory-bound / Memory Stall):** Trong các tác vụ decode LLM, arithmetic intensity cực thấp (~1–2 FLOP/byte). Các Tensor Core liên tục phải chờ nạp trọng số và KV Cache từ HBM. GPU clock luôn bận (100% util) nhưng thực tế các Core đang rơi vào trạng thái *stall/idle chờ I/O bộ nhớ*.
3. **Overhead khởi chạy Kernel (Kernel Launch Overhead):** Các chuỗi kernel nhỏ chạy tuần tự khiến GPU bận chuyển đổi ngữ cảnh mà không sinh ra FLOPs hiệu dụng.
4. **Tác động tài chính:** Doanh nghiệp trả tiền trọn gói theo giờ cho GPU flagship đắt đỏ (H100 với giá $2.50/giờ) nhưng chỉ thu lại hiệu năng của một dòng card tầm trung (A100 $1.79/giờ hoặc thấp hơn). Nếu mua sắm dựa vào GPU-Util, công ty sẽ ảo tưởng rằng cụm máy chủ đang quá tải và tiếp tục chi tiền mua thêm phần cứng không cần thiết.

---

## 4. Báo cáo kết quả các phần mở rộng ("Your Turn" Extensions)

Chúng tôi đã cài đặt, tích hợp vào pipeline và đo lường định lượng 3 phần mở rộng nâng cao:

### 4.1. Extension 1: Cải thiện chính sách `recommend_tier()` đa yếu tố
- **Cải tiến:** Mở rộng `finops.pricing.recommend_tier()` kết hợp `gpu_type`, `job_days` và ma trận rủi ro gián đoạn thực tế (`GPU_SPOT_INTERRUPT_RISK`: H100 rủi ro thấp 3% nhờ pool riêng biệt, A10G rủi ro cao 15%). Đồng thời đưa logic đánh giá giữa Reserved 1 năm và 3 năm theo vòng đời job.
- **Đo lường định lượng:**
  - M3 phân bổ chuẩn xác: các job huấn luyện ngắn hạn như `job-train-llm` (14 ngày), `job-finetune` (3 ngày) tận dụng Spot instance kết hợp checkpointing, giảm chi phí từ $12,000 xuống $7,524 (tiết kiệm 37.3%).
  - Các job suy luận 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`) đạt duty cycle 75–100% (vượt xa điểm hòa vốn 55%) được cam kết Reserved, đưa mức tiết kiệm purchasing đạt **39.3% ($10,089/tháng)**.

### 4.2. Extension 2: Right-sizing theo MBU và kinh tế học VRAM
- **Cải tiến:** Xây dựng module phân tích trong M1 tính toán `$/GB-VRAM` và `peak_bw_tbs` của toàn bộ catalog. Với các GPU có MBU thấp (<0.25) và MFU thấp (<0.30), thuật toán tự động tìm GPU thay thế rẻ nhất có băng thông đỉnh vẫn đảm bảo tối thiểu 1.5× băng thông thực tế đang dùng.
- **Kết quả đo lường định lượng:**
  - `gpu-h100-4` (H100, tiêu tốn $2.50/h, đạt bw thực tế 0.69 TB/s) được khuyến nghị chuyển sang **A100** ($1.79/h, peak bw 2.00 TB/s). Tiết kiệm **$511/tháng**.
  - `gpu-a10g-0` (A10G, tiêu tốn $1.00/h, đạt bw thực tế 0.14 TB/s) được chuyển sang **L4** ($0.80/h, peak bw 0.30 TB/s). Tiết kiệm **$144/tháng**.
  - Tổng tiềm năng tiết kiệm từ MBU Right-sizing: **$655/tháng**.

### 4.3. Extension 3: Mô hình kinh tế học Prompt Caching (`cache_is_worth_it`)
- **Cải tiến:** Xây dựng hàm `cache_is_worth_it(avg_cache_reads, write_cost_per_m, read_discount, base_read_price_per_m)`. Caching chỉ sinh lời khi tổng số tiền tiết kiệm từ các lượt đọc tiếp theo lớn hơn chi phí lưu/ghi cache ban đầu:
  $$\text{Break-even reads} = \frac{\text{write\_cost}}{\text{base\_read\_price} \times (1 - \text{read\_discount})}$$
- **Kết quả đo lường định lượng:**
  - Với tỷ lệ chiết khấu đọc 90% (`read_discount = 0.10`), điểm hòa vốn cần tối thiểu **1.11 lần đọc lại**.
  - Workload thực tế tại NimbusAI có tần suất tái sử dụng system prompt trung bình là **3.5 lần**, hoàn toàn vượt ngưỡng hòa vốn (`cache_is_worth_it = True`), bảo đảm tính hợp lý của khoản tiết kiệm trong M2.

---

## 5. Phân tích Tính bền vững (Sustainability)

Telemetry năng lượng và mô hình phát thải:
- **Năng lượng trung bình trên mỗi query:** `0.24 Wh`
- **Lượng phát thải carbon trung bình:** `0.091 gCO2e` (tại `us-east-1` với 380 gCO2/kWh)
- **Vùng triển khai tối ưu:** **`europe-north1`** (Na Uy/Phần Lan) đạt mức phát thải thấp nhất nhờ nguồn thủy điện sạch (chỉ ~20 gCO2/kWh), đồng thời giá điện cạnh tranh nhất.
- **Đánh đổi (Trade-off):** Chuyển các job training/batch (không nhạy cảm với độ trễ người dùng) sang `europe-north1` giúp giảm hơn 90% phát thải carbon cho cùng một khối lượng tính toán. Ngược lại, traffic inference phục vụ khách hàng trực tiếp cần giữ ở vùng gần người dùng để duy trì SLA về độ trễ round-trip.

---

## 6. Khuyến nghị hành động cho Ban Lãnh đạo NimbusAI (Top 3 Actions)

Nếu đảm nhiệm vai trò FinOps Lead tại NimbusAI, 3 hành động triển khai ngay theo thứ tự ưu tiên ROI:

1. **Hành động 1 — Áp dụng ngay Spot Instance & Checkpointing cho Training Workloads (ROI cao nhất):**
   - *Thực thi:* Cấu hình framework checkpointing tự động (chu kỳ 30–60 phút) cho các job `job-train-llm` và `job-finetune`.
   - *Kết quả:* Cắt giảm ngay lập tức hơn $5,000/tháng mà không ảnh hưởng tới tiến độ bàn giao model.
2. **Hành động 2 — Chuẩn hóa API Gateway với Cascade Routing & Prompt Caching:**
   - *Thực thi:* Bắt buộc các đội sản phẩm định tuyến qua LiteLLM proxy: tự động cache prefix prompt hệ thống và chuyển câu hỏi ngắn/dễ sang model small.
   - *Kết quả:* Giảm chi phí phục vụ token từ $6.488 xuống $1.126 / 1M-token (giảm hơn 80% hóa đơn suy luận).
3. **Hành động 3 — Kích hoạt chính sách Chargeback dựa trên chuẩn FOCUS:**
   - *Thực thi:* Với tỷ lệ gắn tag đã đạt 92% (vượt ngưỡng 80%), chuyển từ giai đoạn Showback sang Chargeback thực tế; xuất báo cáo theo chuẩn FOCUS để từng team chịu trách nhiệm ngân sách trực tiếp.
   - *Kết quả:* Chấm dứt triệt để tình trạng bỏ quên GPU chạy ngầm không sử dụng ($600/tháng lãng phí).
