# Audit toàn bộ SepReformer-PARR Libri2Mix 16 kHz

Ngày kiểm tra: 2026-09-05

> Cập nhật vận hành ngày 2026-09-07: xem [tái kiểm tra checkpoint, resume,
> preflight và inference](PARR_REAUDIT_2026_09_07.md). Các kết quả runtime chưa
> kiểm chứng trong bản cũ không được coi là đã đạt trên L4.

## 1. Phạm vi

Audit này đối chiếu toàn bộ chuỗi công việc: chẩn đoán khoảng trống, đặc tả
PARR, mã tích hợp, tensor flow, cấu hình Libri2Mix 16 kHz, loss, checkpoint,
khả năng tái lập baseline và các sơ đồ kiến trúc. Đây không chỉ là kiểm tra hai
nâng cấp gần nhất của controller.

## 2. Hợp đồng kiến trúc PARR đã chốt

PARR xử lý đầu ra của từng decoder stage `r = 0,1,2,3`, sau chuỗi Global,
Local và Cross-Speaker Transformer. Không có PARR ở bottleneck. Với
`H_r` có shape `[B*J, 128, T_r]`:

```text
N_r       = ChannelLayerNorm_r(H_r)
Z_r       = ChannelLayerNorm(PReLU(Conv1x1_128_to_64(N_r)))
U_r       = ConvU_theta(Z_r)
V_r       = ConvU_theta(Z_r)
M_r       = DenseDilatedTemporalMemory_theta(V_r), dilation=[1,2,4,8]
R_r       = ChannelLayerNorm(U_r * M_r)
A_r       = sigmoid(Conv1x1_128_to_64(concat(Z_r, R_r)))
Delta_r   = Conv1x1_64_to_128(A_r * R_r)
H'_r      = H_r + gamma_r * Delta_r
```

- `theta` được dùng chung cho bốn stage và mọi speaker branch.
- `ChannelLayerNorm_r` và `gamma_r` riêng theo stage.
- `A_r` là channel-temporal gate `[B*J,64,T_r]`, không broadcast theo channel.
- Controller khởi tạo zero nên gate ban đầu bằng `0.5`.
- `gamma_r=0` nên toàn PARR là exact identity lúc khởi tạo.
- `R_r` là pure correction; không có phép cộng `Z_r + R_r` bên trong core.

Số tham số tăng thêm theo phép đếm runtime hiện tại là 98,186 (khoảng 0.10M),
bao gồm bốn tham số vô hướng `gamma_r`,
trong đó temporal core dùng chung nên không nhân bốn lần theo số stage.

## 3. Ánh xạ đặc tả sang code

| Đặc tả | Vị trí code |
|---|---|
| Channel-wise normalization | `modules/module.py::ChannelLayerNorm` |
| Hai nhánh Conv-U | `modules/module.py::PARRConvU`, `conv_u`, `conv_v` |
| Bộ nhớ giãn cách `[1,2,4,8]` | `DenseDilatedFSMN1D` và `configs.yaml` |
| Pure correction `Norm(U*M)` | `PARRTemporalCore.forward` |
| Channel-temporal controller | `PARRTemporalCore.temporal_controller` |
| Shared core, stage norm/scale riêng | `ProgressiveAdaptiveResidualRefinement` |
| Chèn sau từng decoder stage | `Separator.forward`, ngay sau `dec_stages[idx]` |
| Không chèn ở bottleneck | bottleneck chỉ qua `spk_split_block` |
| Exact identity initialization | `gamma_init: 0.0` và zero-init controller |

Kết quả đối chiếu baseline: `dataset.py`, `model.py`, `main.py`,
`modules/network.py` và package initializers giống byte-for-byte. Khác biệt mô
hình nằm trong PARR, tham số cấu hình PARR và lệnh gọi sau decoder stage. Hai
engine 16 kHz có cùng các sửa lỗi vận hành để không làm lệch phép so sánh.

## 4. Luồng output và loss

- Main output dùng `H'_3` ở độ phân giải đầy đủ.
- Bốn auxiliary inputs là speaker-split bottleneck, `H'_0`, `H'_1`, `H'_2`.
- Loss chính vẫn là time-domain PIT SI-SNR.
- Loss phụ vẫn là STFT-magnitude PIT tại bốn mức.
- Không có Cross-scale Evidence, Stage-consistent PIT, mixture consistency,
  MR-STFT, Local Assignment Loss hay early-exit trong phiên bản này.

`input_sizes` nay được dùng để mask phần padding trong các loss dựa trên
SI-SNR/STFT. Trước sửa đổi, loader pad các utterance ngắn trong batch nhưng loss
chỉ dùng `input_sizes` để đếm batch, khiến phần padding có thể ảnh hưởng metric.
Phép chọn permutation của SI-SNRi cũng đã được sửa thành chọn một permutation
chung cho toàn bộ speaker của từng utterance, thay vì có nguy cơ chọn khác nhau
theo speaker.

## 5. Dữ liệu và cấu hình 16 kHz

- Sampling rate: 16,000 Hz.
- Train crop tối đa: 64,000 samples = 4 giây.
- Audio encoder: kernel 32, stride 8; latent được pad theo `2^4=16` stage.
- Libri2Mix recipe: `wav16k/min`, `train-100`, `dev`, `test`, `mix_clean`.
- Batch size hiện tại: 2; max epoch: 200.
- STFT phụ: frame 1024, hop 256.

Loader ưu tiên thư mục đã giải nén và fallback sang `data/Libri2Mix.zip`.
Đọc trực tiếp ZIP đúng về logic nhưng là nút thắt I/O; nên giải nén trước khi
train dài hạn.

## 6. Checkpoint và tính tái lập

Đã tách rõ hai ngữ nghĩa:

- `log/pretrain_weights`: chỉ nạp tensor model tương thích về tên và shape,
  không nạp optimizer/epoch; bắt đầu run mới từ epoch 1.
- `log/scratch_weights`: resume strict cùng kiến trúc, gồm model, optimizer,
  scheduler và epoch; checkpoint mới luôn lưu vào đây.

Checkpoint legacy không có scheduler state vẫn nạp được nhưng phát cảnh báo.
Vòng lặp epoch đã sửa để `max_epoch: 200` thật sự chạy đến epoch 200.

Checkpoint WSJ0-2Mix 8 kHz của tác giả không phải baseline hợp lệ cho báo cáo
Libri2Mix 16 kHz. Nếu dùng, phải gọi đúng là partial-transfer initialization;
không được mô tả là tự train baseline Libri2Mix từ đầu.

## 7. Ý nghĩa khoa học và giới hạn phát biểu

PARR giải quyết một khoảng trống có lý: SepReformer tái dựng dần ở nhiều scale,
trong khi refinement hậu decoder cuối chỉ sửa biểu diễn sau khi toàn bộ quá
trình tái dựng đã kết thúc. PARR đưa một correction nhẹ, có điều khiển, vào đúng
mỗi bước tái dựng và giữ ổn định bằng shared core cùng exact-identity init.

Tuy nhiên code đúng không tự chứng minh đóng góp thực nghiệm. Trước khi có kết
quả, chỉ nên phát biểu đây là giả thuyết kiến trúc có cơ sở, không khẳng định
chắc chắn tăng SI-SNRi hay là hoàn toàn mới trong toàn bộ literature. Tên
`DenseDilatedFSMN1D` nên được mô tả trong luận văn là **FSMN-style dense dilated
temporal memory**, vì implementation dùng depthwise dilated convolutions chứ
không nên được trình bày như bản sao nguyên dạng của một FSMN chuẩn cụ thể.

Tối thiểu cần báo cáo:

1. SepReformer baseline 16 kHz và PARR dưới cùng pipeline.
2. Số tham số, MACs, peak VRAM và tốc độ inference.
3. SI-SNRi/SDRi trên test, tốt nhất nhiều seed hoặc ít nhất cùng seed và ghi rõ.
4. Một ablation tách đóng góp thích nghi: PARR đầy đủ so với `A_r=1`.
5. Nếu đủ tài nguyên, thêm final-stage-only so với four-stage PARR để chứng minh
   chính placement đa stage tạo lợi ích.

## 8. Trạng thái kiểm tra

- Python syntax/bytecode compilation: đạt.
- YAML parsing và các giá trị cấu hình cốt lõi: đạt.
- XML parsing của bốn SVG: đạt. Ba bản dùng để xuất bản (`overview`,
  `simplified`, `publication_v1`) đã được visual-inspect; SVG giản lược bị thiếu
  trên đĩa đã được phục hồi và render lại thành PNG/PDF. Bản
  `parr_architecture_handdrawn` là sketch minh họa có đường cong, vì vậy không
  nên dùng làm bản chính nếu áp dụng yêu cầu chỉ dùng đường thẳng/vuông góc.
- Kiểm tra source-level baseline/PARR và runtime call path: đạt.
- PARR tensor/gradient smoke test: chưa chạy trong môi trường audit vì Python
  hiện tại không cài `torch`; test đã có sẵn và phải chạy trong môi trường train.
- Full forward/backward trên GPU và benchmark VRAM: chưa thể xác nhận nếu chưa
  chạy trong môi trường CUDA thực tế.

Lệnh xác nhận cuối trong môi trường train:

```bash
python -m models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr
python run.py --model SepReformer_PARR_Libri2Mix_16K --engine-mode train
```

## 9. Bộ đánh giá mở rộng

Pipeline test hiện xuất SI-SNR/i, BSS-Eval SDR/SIR/SAR và các độ cải thiện,
scale-dependent SNR/i, mixture-consistency error, WB-PESQ, STOI, ESTOI,
latency, RTF, peak VRAM, parameter count và MAC estimates.
Một permutation duy nhất được chọn bằng SI-SNR rồi dùng cho mọi quality metric.
Kết quả từng utterance và summary có bootstrap CI 95% nằm trong thư mục
`evaluation/checkpoint_epoch_XXXX`. Quy trình và lệnh paired comparison được mô
tả tại `docs/SEPARATION_EVALUATION_PROTOCOL.md`.

## 10. Trạng thái triển khai NVIDIA L4

Môi trường L4 được cố định bằng `requirements-runtime.txt`, `Dockerfile` và
`scripts/setup_env.sh`. Script `scripts/preflight.py` kiểm tra CUDA, cấu
hình dùng chung baseline/PARR, ba partition Libri2Mix, evaluator, checkpoint
round-trip và một bước forward/loss/backward đầy đủ ở batch 2 × 64.000 samples
cho cả hai mô hình.

Checkpoint đã được tách thành `latest.pth` để resume và `best.pth` để test. Hai
file đều được ghi atomic và có model, optimizer, scheduler, epoch, train/valid
loss cùng best validation loss. DataLoader dùng generator và worker seed riêng,
nhờ đó thứ tự minibatch và chuỗi crop không còn phụ thuộc vào số phép khởi tạo
ngẫu nhiên khác nhau giữa baseline và PARR.

Hướng dẫn vận hành và mount persistent storage nằm tại
`docs/DEPLOYMENT.md`. Kiểm tra tĩnh trên máy phát triển không thay thế
preflight CUDA: chỉ bắt đầu full training khi preflight trên chính L4 kết thúc
với exit code 0.
