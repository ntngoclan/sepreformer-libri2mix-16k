# Sửa lỗi pipeline SepReformer-PARR và baseline

Các thay đổi áp dụng đồng bộ cho hai package Libri2Mix 16 kHz và utilities dùng chung. Vị trí chèn PARR, hai nhánh Conv-U, gamma/gate, trọng số loss và cấu hình kiến trúc được giữ nguyên. Đây là sửa tính đúng đắn và vận hành của phép so sánh; chưa phải bằng chứng PARR tốt hơn baseline.

| Phát hiện trong review | Thay đổi |
|---|---|
| F1: STFT phụ thuộc padding batch | Số frame hợp lệ được tính từ từng `input_sizes`, frame ngoài phạm vi bị loại khỏi cả tử và mẫu loss. Giữ quy ước STFT không center và padding lên bội số hop của mỗi câu. |
| F2: resume không phát hiện dilation/dropout khác | Checkpoint có `run_contract`: full-config hash, architecture hash, source fingerprint chuẩn hóa newline, pipeline revision và config snapshot. Kiểm tra trước khi nạp trạng thái. |
| F3: câu ngắn hơn kernel | STFT có tối thiểu một frame. Model pad waveform đủ kernel/stride rồi cắt output về chiều dài gốc. Khi train, câu rất ngắn có đủ frame cho BatchNorm ở bottleneck. |
| F4: estimate im lặng nhận 0 dB | SI-SNR v2 chuẩn hóa năng lượng, giữ bất biến gain với tín hiệu rất nhỏ; estimate zero/DC nhận floor −80 dB và được ghi nhận trong failures. Reference zero/DC bị từ chối rõ ràng. |
| F5: bootstrap tắt trả CI giả | `bootstrap_samples=0` trả CI không khả dụng (`null` trong JSON). Dùng `ci_low/high`; alias `ci95_*` chỉ có ý nghĩa khi confidence=0,95. Paired comparison dùng cùng quy tắc. |
| F6: padding ảnh hưởng backbone | `Model.forward(..., input_sizes=...)` nhóm các mẫu cùng độ dài thật, crop trước backbone, ghép output về thứ tự gốc và pad sau inference. Engine train/valid truyền lengths. Validation/test dùng batch 1. |
| F7: STFT lặp trong PIT/auxiliary | Tính STFT gộp speaker, tái sử dụng reference power giữa bốn auxiliary heads và tạo pairwise cost matrix. Một batch bốn head gọi STFT 5 lần thay vì 32; kích thước tensor mỗi lần khác nhau nên không suy ra speedup 6,4 lần. |

Các sửa bổ sung: loss thống kê epoch được tính theo số utterance thay vì số batch; warmup tiếp tục qua epoch tiếp theo nếu chưa đủ số update; plateau scheduler chờ warmup kết thúc. Attention maps chỉ được giữ ở dạng detach khi bật `store_attention=True`, mặc định không giữ tensor graph sau forward. Public model hỗ trợ input một chiều và độ dài không chia hết stride, trả đủ chiều dài gốc.

**Quy ước loss và hiệu năng**

Frame coverage của mỗi câu bằng chạy riêng câu đó: `padded_length=max(frame_length,ceil(length/hop)*hop)`. Số frame là `1+(padded_length-frame_length)//hop`. Đây không phải thay đổi sang STFT center=True hoặc bổ sung mọi cửa sổ giao với đuôi câu. Khi tối ưu scaling của reference, scale bình phương được áp dụng lên STFT power trước khi cộng epsilon magnitude; nhờ đó giữ được giá trị và gradient của phép scale waveform trước STFT, trong sai số số học.

Reference cache chỉ sống trong một batch, không cache theo tên utterance qua optimizer updates. Cách nhóm theo độ dài có thể tạo thêm lần gọi backbone khi batch chứa nhiều độ dài khác nhau; các batch có cùng crop length vẫn đi đường xử lý gộp. Cần đo wall-clock/VRAM trên L40, không suy tốc độ thực từ số lần gọi STFT hoặc số tham số.

**Checkpoint và kết quả cũ**

Không resume run đã train bằng loss/padding cũ dưới tên run mới. Hai mô hình phải bắt đầu lại từ epoch-0 cùng điều kiện. Các file paired epoch-0 có cấu hình chung tương thích vẫn dùng được; chúng không phải checkpoint đã huấn luyện.

Engine yêu cầu `run_contract` khi resume và khi test/infer checkpoint. File cũ thiếu contract bị từ chối thay vì tự bổ sung một metadata có thể sai. Nếu cần dùng trọng số cũ để khởi tạo một thí nghiệm transfer riêng, đặt chúng trong `log/pretrain_weights` và tắt paired initialization; optimizer/epoch được khởi tạo lại. Không gọi thí nghiệm này là baseline/PARR cùng initialization.

Evaluation JSON ghi checkpoint SHA-256, run contract, paired-initialization metadata và cấu hình evaluation hiện tại. Evaluation được phép đổi tùy chọn metric/bootstrap, nhưng không được đổi kiến trúc hoặc source contract. Resume kiểm tra cả full configuration, nên thay đổi batch size, seed, optimizer hoặc giới hạn epoch phải trở thành một run được xác định rõ.

SI-SNR dùng protocol `separation-v2-unit-si-snr`; floor suy biến là −80 dB. Mọi estimate im lặng vẫn có mặt trong trung bình và paired comparison, không bị loại để làm đẹp điểm. CSV có trường `metric_protocol`; công cụ so sánh từ chối ghép hai protocol khác nhau. Điểm tuyệt đối ở vùng epsilon/suy biến có thể khác evaluator cũ, vì vậy phải đánh giá cả hai model cùng phiên bản. BSS-Eval/PESQ/STOI vẫn có failure reporting riêng.

**Kiểm chứng**

```bash
python -m scripts.runtime_checks
python -m scripts.test_parr_fixes
python -m models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr
```

Các regression mới kiểm tra: loss và gradient tương đương oracle tính riêng từng câu (cả mel/non-mel, scale-invariant/scale-dependent); padding chứa dữ liệu rác không ảnh hưởng loss; gradient padding bằng zero; câu ngắn; cache STFT; silence/DC/near-silence; CI bị tắt; mismatch metric protocol; checkpoint round-trip và đổi dilation/dropout/LR; output đúng thứ tự và chiều dài; batching theo true length; mean validation theo số mẫu; warmup qua ranh giới epoch.

Kiểm tra CPU dùng PyTorch 2.1.2+cpu, NumPy 1.26.4 và Python 3.11 portable trong thư mục tạm, tái sử dụng dependency của `.venv-audit`. Không sửa Python hệ thống. Bộ runtime cũ bao gồm hai optimizer updates của cả model, paired initialization/identity và resume với 0/2 workers.

Kết quả cuối: **25/25 regression đạt trong 38,703 giây** (12 test runtime và 13 test sửa lỗi); smoke test PARR đạt; AST của 41 file Python đạt; code dùng chung và cấu hình hai package được đối chiếu nhất quán; `git diff --check` không phát hiện lỗi whitespace. Một lượt gọi gộp qua stdin trước đó không chạy được multiprocessing spawn trên Windows; lượt xác nhận cuối dùng entry point tương thích và test resume hai worker đã đạt.

CUDA/L40, throughput thực và PESQ runtime vẫn cần chạy trong môi trường deployment:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_l40.py --require-l40
```

Các hướng stage conditioning, dilation phụ thuộc stage, gate cố định, stage cuối-only và bỏ output bias vẫn là ablation riêng. Chưa thay đổi chúng trong bản sửa này vì sẽ thay đổi giả thuyết kiến trúc đang được đánh giá.
