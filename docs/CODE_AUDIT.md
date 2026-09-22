# Audit code và triển khai

> Báo cáo lịch sử: các nhắc đến L4 bên dưới mô tả môi trường dự kiến ở thời điểm
> audit, không phải giới hạn phần cứng hiện tại. Tên file đã được cập nhật.
> Quy trình triển khai hiện hành: [DEPLOYMENT.md](DEPLOYMENT.md).

Ngày audit: 2026-09-05.

> Cập nhật 2026-09-07: các kiểm tra cũ bỏ sót scheduler serialization,
> optimizer.step và RNG resume. Xem [bản tái kiểm tra](PARR_REAUDIT_2026_09_07.md)
> để biết sửa chữa và trạng thái xác minh hiện tại; không diễn giải checklist cũ
> thành bằng chứng đã chạy thành công trên GPU.

## Phạm vi

Audit tập trung vào hai thí nghiệm phải so sánh trực tiếp:

- `SepReformer_Base_Libri2Mix_16K`;
- `SepReformer_PARR_Libri2Mix_16K`;
- shared utilities, loss, evaluator, checkpoint và deployment scripts;
- Libri2Mix 16 kHz `min/train-100`, `dev`, `test`.

Code WSJ0 và thư mục `SourceCode` lịch sử không nằm trên đường chạy L4 này và
không được dùng để kết luận về baseline/PARR 16 kHz.

## Các lỗi/rủi ro đã xử lý

1. Checkpoint trước đây chỉ được lưu khi validation cải thiện. Engine nay luôn
   ghi `latest.pth` để resume và ghi riêng `best.pth` để test.
2. Checkpoint được ghi vào file tạm rồi `os.replace`, tránh làm hỏng bản hợp lệ
   nếu instance dừng trong lúc ghi.
3. `best_valid_loss` được giữ trong `latest.pth`, nên resume không thể vô tình
   thay `best.pth` bằng một checkpoint kém hơn.
4. DataLoader có generator riêng và seed rõ cho Python/NumPy trong từng worker.
   Việc PARR có thêm tham số không còn làm lệch thứ tự batch/crop so với baseline.
5. CUDA availability và GPU ID được kiểm tra trước khi tạo criterion/engine.
6. Ba profiler MACs/params không còn làm toàn bộ training dừng nếu một profiler
   không hỗ trợ một operator; lỗi profiler được ghi vào summary.
7. Profiling chạy ở eval mode rồi khôi phục đúng training state.
8. Các lệnh `functional.upsample`, `autograd.Variable`, `.data.numpy()` trên
   đường chạy 16 kHz đã được thay bằng API hiện hành nhưng giữ nguyên phép toán.
9. Peak normalization không còn chia cho 0 khi inference ra tín hiệu im lặng.
10. STFT nhánh tensor 3-D tạo padding đúng device và dtype.
11. Evaluator baseline và PARR dùng cùng implementation và cùng permutation
    SI-SNR cho mọi quality metric.

## Kết quả kiểm tra tại workspace

- Python `compileall`: đạt cho `run.py`, `utils`, hai model 16 kHz và `scripts`.
- YAML parse: đạt.
- Shared config parity: đạt sau khi bỏ đúng khối `parr` duy nhất.
- File parity: `dataset.py`, `main.py`, `model.py` và `modules/network.py` của
  baseline/PARR có hash giống nhau; khác biệt kiến trúc nằm trong separator PARR.
- Bash syntax: đạt cho setup/train/test scripts.
- Dockerfile static check: đúng CUDA 12.1.1, cu121 và PyTorch 2.1.2.
- Requirements L4: không trùng package và không cài đè torch CUDA wheel.
- Dataset archive:
  - train-100: 13.900 mixture;
  - dev: 3.000 mixture;
  - test: 3.000 mixture;
  - tất cả mixture đều có đủ file cùng tên trong `s1` và `s2`;
  - các WAV lấy mẫu ngẫu nhiên đều mono, 16.000 Hz và không rỗng.
- SVG XML parse: đạt cho toàn bộ bốn file SVG hiện có.

## Kiểm tra bắt buộc trên chính L4

Máy audit hiện tại không có môi trường PyTorch/CUDA của máy train, do đó không
thể trung thực tuyên bố CUDA runtime đã đạt. Trên L4 phải chạy:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preflight.py
```

Mặc định preflight thực hiện:

- phép nhân ma trận CUDA;
- PARR identity/controller/gradient smoke test;
- checkpoint atomic round-trip và lựa chọn best/latest;
- đọc mẫu thật từ cả ba partition;
- chạy SI-SNR, BSS-SDR, PESQ, STOI, ESTOI và ghi thử CSV/JSON;
- forward, đúng objective train, bốn spectral auxiliary loss và backward cho
  cả baseline và PARR ở batch 2 × 64.000 samples;
- kiểm tra NaN/Inf, output shape, gradient stage scale và peak VRAM.

## Điều kiện GO/NO-GO

Chỉ **GO** cho full training khi:

1. preflight exit code bằng 0;
2. báo cáo nhận đúng một L4 và CUDA 12.1 runtime;
3. cả hai full training-step simulation không OOM và không có NaN/Inf;
4. dataset count đúng 13.900/3.000/3.000;
5. persistent volume chứa được `latest.pth`, `best.pth` và log;
6. sau epoch đầu tiên có đủ hai checkpoint và có thể khởi động lại từ
   `latest.pth`.

Nếu một điều kiện không đạt thì **NO-GO**: giữ log và
`run_logs/preflight_report.json`, sửa nguyên nhân trước khi bắt đầu 200 epoch.

## Giới hạn có chủ ý

- Chưa bật AMP để tránh thay đổi điều kiện số học ngay trước thí nghiệm đối
  chứng. L4 vẫn chạy FP32 với batch size 2 theo config đã chốt.
- `torch.use_deterministic_algorithms(..., warn_only=True)` ghi cảnh báo thay vì
  dừng khi gặp operator không deterministic. Vì vậy cần báo cáo nhiều seed khi
  tài nguyên cho phép; không tuyên bố bitwise reproducibility.
- MACs từ ptflops/THOP/torchinfo là estimates theo công cụ. Báo cáo phải ghi rõ
  tool và input length, không coi ba con số là hoàn toàn tương đương.
- Static audit không thể bảo đảm mọi lỗi phần cứng/driver. Preflight trên đúng
  instance L4 là cổng xác nhận runtime cuối cùng.
