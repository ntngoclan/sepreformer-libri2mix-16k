# Tái kiểm tra PARR sau phản biện — 2026-09-07

## Phạm vi và kết luận tạm thời

Bản này cập nhật các kết luận quá mạnh trong audit ngày 2026-09-05. Kiểm tra
cú pháp hoặc vài mẫu dữ liệu không đủ để kết luận sẵn sàng train 200 epoch.
Không thay đổi kiến trúc PARR, vị trí chèn, gamma/gate, loss hay hyperparameter.
Các sửa chữa đường train/test được áp dụng đồng bộ cho baseline và PARR 16 kHz.

## Ánh xạ phát hiện → sửa chữa

| Phát hiện | Thay đổi và bài kiểm tra |
|---|---|
| Local closure trong scheduler không pickle được | `utils/implements/schedulers.py`: không lưu closure vào thuộc tính ngoài cơ chế LambdaLR; kiểm tra checkpoint với AdamW và cả hai scheduler thật. |
| Warm-up gọi scheduler trước optimizer | Chuyển `scheduler.step()` ra sau `optimizer.step()` theo API PyTorch; dùng `(step+1)/warmup_steps` để giữ dãy LR mỗi update là 1e-6, 2e-6, ... |
| Preflight import sai khi chạy script trực tiếp | Đưa repository root vào `sys.path` trước các import dự án. |
| Preflight chưa cập nhật trọng số | Hai bước forward/loss/backward/clip/AdamW cho mỗi mô hình; kiểm tra finite và trọng số thực sự thay đổi. |
| Gamma smoke test bị mở thủ công | Test bổ sung giữ gamma=0 tự nhiên: exact identity và gradient gate bằng 0 trước bước đầu, khác 0 sau khi optimizer mở gamma. Test cũ gamma=0.1 chỉ còn là kiểm tra riêng, không thay thế test này. |
| Resume thiếu RNG | Checkpoint lưu Python, NumPy, torch CPU/CUDA, generator và cấu hình DataLoader; khôi phục sau profiling; bỏ validation thừa lúc resume. |
| Bỏ sót trường hợp multiworker | `scripts/runtime_checks.py` so sánh epoch kế tiếp của chạy liên tục và resume với 0/2 worker, gồm batch, crop ngẫu nhiên, dropout, loss, trọng số, AdamW state và LR. |
| Loader âm thầm cắt độ dài không khớp | Kiểm tra chiều dài WAV gốc trước crop/stride trimming; từ chối audio rỗng hoặc NaN/Inf. |
| Chỉ kiểm tra một mẫu mỗi split | Preflight mặc định đọc toàn bộ mixture và sources; `--quick-data` chỉ nhận trạng thái `partial`. |
| CSV thiếu mẫu vẫn được so sánh | Mặc định yêu cầu tập key giống nhau và chiều dài khớp; CLI yêu cầu 3.000 mẫu. Subset phải bật rõ `--allow-subset`; số cặp loại bỏ theo metric hiện trong JSON/Markdown. |
| Auxiliary làm sai phạm vi benchmark | `Model.forward(return_aux=False)` không chạy auxiliary waveform heads; train giữ mặc định cũ. Test/infer chỉ nạp model weights, không khởi tạo optimizer/criterion. |
| Profiler có thể để lại hook/buffer | Mỗi profiler dùng một bản sao riêng, không chạm model đang train. |
| Báo cáo VRAM/params không rõ phạm vi | Ghi metadata thời gian forward, peak allocated VRAM, tổng params và params ngoài auxiliary. Auxiliary weights vẫn resident; không tuyên bố đây là deployment tối giản. |

Optimizer được tạo sau khi model chuyển device. Training dừng khi loss hoặc
gradient clipping gặp NaN/Inf. Log được nối tiếp thay vì ghi đè lúc resume.
Docker thêm alias `python` để các Bash script chạy với Ubuntu image hiện tại.

## Kết quả đã chạy tại workspace

### Dữ liệu thật — đạt

Lệnh: `python -u scripts/audit_audio_archive.py`.

Đọc toàn bộ nội dung từng WAV và kiểm tra CRC ZIP, header PCM, số byte payload,
mono 16 kHz, không rỗng, đủ nguồn và độ dài raw bằng nhau. Thời gian 355,90 giây.

| Partition | Mixture | WAV thực sự đọc | Giờ mixture | Kết quả |
|---|---:|---:|---:|---|
| train-100 | 13.900 | 41.700 | 43,2700 | Đạt |
| dev | 3.000 | 9.000 | 4,5135 | Đạt |
| test | 3.000 | 9.000 | 4,1872 | Đạt |

Tổng 59.700 WAV thuộc `mix_clean/s1/s2`, tất cả PCM 16-bit. Đây không phải kiểm
tra các biến thể noisy/max/8 kHz và không thay thế kiểm tra DataLoader trên L40.

### Runtime và cú pháp

- Cú pháp Python: compileall thành công trên bản code cuối.
- Bash: `bash -n` thành công cho setup/train/test scripts.
- Bộ test CPU PyTorch 2.1.2: hai lượt xác nhận cuối đều đạt 10/10 test sau
  sửa warm-up (34,46 và 37,57 giây),
  không còn cảnh báo gọi scheduler sai thứ tự. Gồm hai update với objective thật
  của cả hai model, checkpoint/resume 0 và 2 worker, gamma/gate và inference.
- Dataset class/SoundFile đã đọc thành công một mẫu ở từng split trực tiếp từ ZIP.
- PESQ 0.0.4 chưa cài được trên Windows vì thiếu Microsoft Visual C++ 14+;
  evaluator PESQ đầy đủ phải được kiểm tra trong môi trường Linux L40.
- L40/CUDA, VRAM thực, PESQ runtime trên máy train và chạy hết epoch: chưa xác minh.

Do đó cả baseline và PARR hiện **sẵn sàng để đưa lên L40 và chạy full
preflight**, nhưng **chưa được đánh dấu sẵn sàng chạy thẳng 200 epoch** cho tới
khi full preflight trên chính L40 trả `status: passed`.

## Điều kiện sử dụng

Resume được thiết kế ở **ranh giới epoch**, không phải giữa minibatch. Giữ cùng
dataset, batch size, số worker, sampler, phần mềm và GPU topology; không dùng
persistent workers. Checkpoint cũ không có RNG không thể tái lập đầy đủ.
CUDA có thể không deterministic nên không cam kết bitwise trên L40.

Trên máy L40 chạy:

```bash
source .venv/bin/activate
python -m scripts.runtime_checks
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_l40.py --require-l40
```

Preflight phải trả exit code 0 **và** `status: passed`; báo cáo cũ bị vô hiệu hóa
thành `incomplete` khi một lượt kiểm tra mới bắt đầu. Hai bước optimizer không
phải hai epoch. Kiểm tra checkpoint và resume sau epoch thực đầu tiên trước
khi để train dài. Chỉ load checkpoint từ nguồn tin cậy vì torch.load dùng pickle.

## Giới hạn khoa học

Các bản sửa trên xử lý tính đúng đắn triển khai; không chứng minh PARR tốt hơn
baseline, tính mới học thuật, hoặc khả năng khái quát tiếng Việt. Ablation và
đánh giá có kiểm soát vẫn là thí nghiệm chưa thực hiện. Không dùng checkpoint
hoặc kết quả paper khác điều kiện để thay thế baseline 16 kHz có kiểm soát.
