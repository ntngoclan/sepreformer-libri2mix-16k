# Protocol đánh giá SepReformer baseline và PARR

## 1. Bộ chỉ số chính thức

| Nhóm | Chỉ số báo cáo | Vai trò |
|---|---|---|
| Tách nguồn | SI-SNR, SI-SNRi | Chỉ số chính, tương thích cách báo cáo SepReformer |
| Tương thích kết quả cũ | BSS-Eval SDR, SDRi | Không gọi nhầm là SI-SDR/SDRi |
| Chẩn đoán lỗi | BSS-SIR/i, BSS-SAR/i | Phân biệt nhiễu giọng còn sót và artifact |
| Tính nhất quán | Mixture-consistency error | Kiểm tra tổng hai nguồn có tái tạo mixture hay không |
| Bảo toàn biên độ | SNR, SNRi | Bổ sung cho metric scale-invariant |
| Chất lượng cảm nhận | WB-PESQ | PESQ wideband cho audio 16 kHz |
| Độ dễ hiểu | STOI, ESTOI | Mức bảo toàn thông tin lời nói |
| Hiệu quả | Params, MACs, latency, RTF, peak VRAM | Giá phải trả cho PARR |
| Độ tin cậy | median, std, IQR, bootstrap CI 95% | Thể hiện phân bố, không chỉ mean |

Không báo cáo đồng thời SI-SNR và zero-mean SI-SDR như hai bằng chứng độc lập,
vì dưới quy ước projection hiện dùng chúng tương đương về bản chất.

PESQ P.862.2 hiện là chuẩn legacy/superseded, nhưng vẫn được tính để so sánh
với literature speech separation. Trong báo cáo phải ghi rõ implementation
`pesq==0.0.4`, chế độ `wb`, sampling rate 16 kHz.

## 2. Quy tắc permutation

Với mỗi utterance, evaluator tìm permutation tối đa tổng SI-SNR giữa hai nguồn.
Permutation đó được giữ cố định cho SNR, BSS-SDR, PESQ, STOI và ESTOI. Không
được cho mỗi metric tự chọn một permutation khác, vì điều đó tạo kết quả lạc
quan không công bằng.

## 3. Chạy đánh giá

Đảm bảo mỗi model có checkpoint tốt nhất trong thư mục `scratch_weights`, sau
đó chạy riêng:

```bash
python run.py --model SepReformer_Base_Libri2Mix_16K --engine-mode test
python run.py --model SepReformer_PARR_Libri2Mix_16K --engine-mode test
```

Mỗi model sinh:

```text
evaluation/checkpoint_epoch_XXXX/
├── metrics_per_utterance.csv
└── metrics_summary.json
```

CSV chứa metric trung bình hai speaker, metric từng speaker, permutation, độ dài
audio, latency, RTF và peak VRAM của từng utterance. JSON chứa mean, standard
deviation, median, Q1/Q3, bootstrap CI 95%, số mẫu hợp lệ, device, params và MACs.

Inference time chỉ đo synchronized model forward với batch size 1; không gồm
đọc file, PESQ/STOI hoặc ghi CSV. Baseline và PARR phải chạy trên cùng GPU, cùng
precision và cùng tiến trình nền. Nên chạy benchmark tốc độ ít nhất ba lần và
báo cáo median giữa các lần nếu cần số liệu hiệu năng chính xác.

Hai config mặc định dùng `seed: 0` và deterministic mode. Nếu đủ tài nguyên,
train cả baseline và PARR với cùng tập seed, ví dụ `{0,1,2}`, đồng thời giữ
checkpoint của mỗi seed trong thư mục riêng. Bootstrap theo utterance đo độ bất
định trên test set nhưng không thay thế độ biến thiên giữa các lần train.

## 4. So sánh paired baseline–PARR

```bash
python -m utils.compare_separation_metrics \
  --baseline models/SepReformer_Base_Libri2Mix_16K/evaluation/checkpoint_epoch_XXXX/metrics_per_utterance.csv \
  --candidate models/SepReformer_PARR_Libri2Mix_16K/evaluation/checkpoint_epoch_XXXX/metrics_per_utterance.csv \
  --output-dir comparison/base_vs_parr
```

Công cụ chỉ ghép các hàng có cùng `key`, sau đó sinh:

- `paired_comparison.json`;
- `paired_comparison.md`.

Mỗi metric có mean baseline, mean PARR, paired delta, bootstrap CI 95%, tỷ lệ
utterance được cải thiện và Wilcoxon signed-rank p-value đã hiệu chỉnh
Benjamini-Hochberg cho nhiều phép thử.

Đối với quality metric, delta dương tốt hơn. Đối với latency, RTF và VRAM,
delta âm tốt hơn.

## 5. Bảng nên đưa vào khóa luận

### Bảng chất lượng

| Model | SI-SNRi ↑ | BSS-SDRi ↑ | SNRi ↑ | WB-PESQ ↑ | STOI ↑ | ESTOI ↑ |
|---|---:|---:|---:|---:|---:|---:|
| SepReformer-B | mean ± CI | mean ± CI | mean ± CI | mean ± CI | mean ± CI | mean ± CI |
| SepReformer-PARR | mean ± CI | mean ± CI | mean ± CI | mean ± CI | mean ± CI | mean ± CI |

BSS-SIR, BSS-SAR và mixture-consistency error nên đặt trong bảng phân tích lỗi
hoặc phụ lục thay vì làm bảng chính quá rộng.

### Bảng hiệu quả

| Model | Params ↓ | MACs ↓ | Peak VRAM ↓ | RTF ↓ | Latency ↓ |
|---|---:|---:|---:|---:|---:|
| SepReformer-B | | | | | |
| SepReformer-PARR | | | | | |

### Bảng paired improvement

Đưa paired delta và CI 95% cho SI-SNRi, PESQ, STOI và RTF. Nếu CI của quality
delta không chứa 0 và hiệu quả tăng không quá lớn, lập luận về PARR sẽ thuyết
phục hơn đáng kể so với chỉ báo cáo hai mean.

## 6. Nguyên tắc sử dụng test set

Không theo dõi test set ở nhiều epoch để chọn mô hình. Chọn checkpoint bằng
validation loss, khóa kiến trúc/hyperparameter, rồi chạy test đúng một lần cho
thí nghiệm báo cáo. Vì vậy `test_epochs` của hai cấu hình 16 kHz đã được đặt
thành danh sách rỗng.
