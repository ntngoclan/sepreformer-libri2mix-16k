# Quy trình baseline và LTRR trên VN-SpeechMix

Hai package dùng cùng protocol v2: 16 kHz, hai nguồn, crop train/valid 64.000
mẫu, batch train 2, train/valid/test 18.000/3.000/5.000 mixture. Test giữ toàn bộ
độ dài sau căn stride. Checkpoint được chọn bằng validation, không chọn bằng test.
Kiểm tra CPU chỉ xác nhận các đường code đã chạy; cần pilot trên GPU trước khi
chạy chính thức. Không có smoke test nào bảo đảm toàn bộ 200 epoch không lỗi.

## 1. Môi trường và dữ liệu

Môi trường GPU mục tiêu: Linux, Python 3.10, PyTorch/torchaudio 2.1.2 CUDA 12.1.
Dùng `scripts/setup_env.sh` hoặc Dockerfile có bộ biên dịch C++ và libsndfile.
`requirements.txt` là snapshot Linux cũ, không phải lệnh cài đa nền tảng.
Trên Windows, `pesq==0.0.4` cần Microsoft C++ Build Tools nếu không có wheel
phù hợp. Không tắt PESQ trong thí nghiệm chính thức chỉ để vượt qua preflight.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m pip check
python scripts/prepare_vnspeechmix_manifest.py
python scripts/audit_audio_archive.py --dataset vnspeechmix --archive data/VnSpeechMix/rendered.zip --report run_logs/vn_archive.json
```

Chuẩn bị manifest giữ nguyên từng byte của CSV trong ZIP; không sửa đường dẫn
trong bản manifest dùng làm fingerprint. Loader đọc được audio trực tiếp từ ZIP.
Cũng có thể giải nén đầy đủ vào `data/VnSpeechMix/rendered/` nếu đủ dung lượng;
không để một split giải nén dở vì loader ưu tiên thư mục khi đủ tên folder.
Audit archive kiểm tra mọi WAV PCM, CRC, số lượng, file tương ứng và độ dài.
Preflight `--full-audio` còn kiểm tra tổng `s1 + s2` và SoundFile trên mọi mixture.

Nếu dựng lại dữ liệu từ nguồn, toolkit riêng nằm ở
`data/VnSpeechMix/VN-SpeechMix-toolkit/` (không được Git của repo này quản lý):

```bash
python -m pip install -r data/VnSpeechMix/VN-SpeechMix-toolkit/requirements.txt
python -m pip install pytest==8.3.5
python -m pytest data/VnSpeechMix/VN-SpeechMix-toolkit/tests -q
```

Lưu source/config/seed của toolkit cùng bản dữ liệu đã render. Các dependency
denoise nằm trong toolkit, không cần khi dùng `rendered.zip` có sẵn.
Sau khi cài, lưu `python -m pip freeze` cùng kết quả để ghi lại phiên bản thực tế;
file requirements của toolkit khai báo khoảng phiên bản được hỗ trợ.

## 2. Regression và kiểm tra trên GPU thực tế

```bash
python -B -m unittest scripts.test_ltrr scripts.test_run_management scripts.test_libri2mix_protocol scripts.test_parr_fixes scripts.runtime_checks scripts.test_vnspeechmix_readiness
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_Base_VnSpeechMix_16K --device cuda --samples 64000 --steps 2 --workers 12 --full-audio --report run_logs/vn_base_gpu.json
python -B scripts/preflight_vnspeechmix_baseline.py --root . --model SepReformer_LTRR_VnSpeechMix_16K --device cuda --samples 64000 --steps 2 --workers 12 --full-audio --report run_logs/vn_ltrr_gpu.json
python -B scripts/smoke_vnspeechmix_pipeline.py --device cuda --samples 64000 --workers 12 --output run_logs/vn_pipeline_gpu
```

Lệnh pipeline dùng kiến trúc thật, audio thật, Engine thật trên subset nhỏ:
hai epoch → resume checkpoint epoch 1 → test checkpoint best → xuất WAV,
CSV/JSON → so sánh cặp → backup và kiểm SHA-256. Warmup được rút ngắn để
kiểm tra scheduler trong hai epoch. Đây là cấu hình riêng, không phải thí nghiệm
chính thức. CPU có thể chạy thêm `--device cpu --samples 2048 --workers 2`;
nó thay thế đúng các lời gọi dispatch/timing CUDA để kiểm tra vòng lặp Engine.
CPU đòi model/optimizer/scheduler/loss sau resume bằng chính xác run liên tục.
CUDA không được khẳng định bitwise identical.

`--skip-pesq` chỉ dành cho audit thiếu thư viện, trả `partial_smoke_passed`.
`--compact-artifacts` xóa checkpoint smoke dư sau khi kiểm tra resume, giữ best
của run resumed, cùng log/provenance và backup. Không dùng các điểm số từ model
hai bước cập nhật để kết luận baseline hay LTRR tốt hơn.

Sau đó chạy pilot hai epoch **toàn bộ dữ liệu** theo README của từng model,
dùng cùng thư mục initialization pilot. Kiểm tra VRAM, disk, loss hữu hạn,
checkpoint, TensorBoard và thử resume. Cần dành dung lượng cho các milestone:
mỗi checkpoint AdamW đầy đủ khoảng 170 MB, chưa tính TensorBoard, backup và WAV.
Hai model với tám checkpoint mỗi model đã cần khoảng 2,7 GB; giữ thêm khoảng
trống tương ứng cho backup và checkpoint tạm. Test dài nhất trong manifest hiện
tại khoảng 29,565 giây: kiểm tra VRAM ở độ dài này, không chỉ crop train 4 giây.

## 3. Train, test và so sánh chính thức

Đóng băng source/config trước train; thay đổi fingerprint source sẽ bị từ chối
khi resume. Các sửa lỗi audit lần này thay đổi fingerprint; không dùng checkpoint
pilot cũ với source mới. Giữ nguyên cả source cũ nếu cần phục hồi một run cũ.

```bash
python run.py --model SepReformer_Base_VnSpeechMix_16K --seed 0 --run-id vn_base_s0
python run.py --model SepReformer_LTRR_VnSpeechMix_16K --seed 0 --run-id vn_ltrr_s0
python run.py --model SepReformer_Base_VnSpeechMix_16K --seed 0 --run-id vn_base_s0 --engine-mode test
python run.py --model SepReformer_LTRR_VnSpeechMix_16K --seed 0 --run-id vn_ltrr_s0 --engine-mode test
```

Thay `BASE_CSV` và `LTRR_CSV` bằng hai file `metrics_per_utterance.csv` trong
thư mục evaluations của đúng seed/checkpoint. VN-SpeechMix có **5.000** test
mixture; mặc định CLI so sánh là 3.000 cho Libri2Mix nên phải truyền số lượng:

```bash
python -m utils.compare_separation_metrics --baseline BASE_CSV --candidate LTRR_CSV --expected-count 5000 --bootstrap-samples 10000 --seed 0 --output-dir run_logs/vn_comparison_s0
```

Kiểm tra cùng initialization SHA-256, manifest SHA-256, training protocol và
metric protocol trước khi so sánh. CSV phải đủ key, đúng độ dài, không duplicate;
kiểm tra `n_pairs`, `n_excluded` và `metric_failures` trong summary. Báo cáo gồm
SI-SNRi, BSS-SDRi, PESQ, STOI/ESTOI, tham số, latency/RTF và peak VRAM, cùng CI
và kiểm định cặp. Lặp seed 0/1/2 rồi tổng hợp độ biến thiên giữa seed; CI theo
utterance của một seed không thay thế độ biến thiên giữa các lần train.
Published split có hai speaker chung valid/test; train không trùng speaker với
hai split còn lại. Giữ và nêu rõ giới hạn này khi báo cáo.

## 4. Đóng gói và kiểm tra phục hồi

```bash
python scripts/backup_run.py create runs/SepReformer_Base_VnSpeechMix_16K/seed_0000/vn_base_s0 backups/vn_base_s0.zip
python scripts/backup_run.py verify backups/vn_base_s0.zip
python scripts/backup_run.py create runs/SepReformer_LTRR_VnSpeechMix_16K/seed_0000/vn_ltrr_s0 backups/vn_ltrr_s0.zip
python scripts/backup_run.py verify backups/vn_ltrr_s0.zip
```

Backup ở lúc writer đã dừng hoặc biên epoch yên tĩnh. Giữ ZIP **và** `.zip.sha256`,
run cha nếu resume, initialization gốc, manifest, dataset/audit report, cả thư mục
so sánh và môi trường `pip freeze`. Nếu evaluate bằng `--checkpoint` tạo run riêng,
backup cả run evaluation đó. Sau copy sang máy khác chạy `verify` lại rồi mới
giải nén. Không để backup duy nhất trên cùng ổ đĩa với bản làm việc.

## Kết quả audit cục bộ

Xem `run_logs/vnspeechmix_audit/` và báo cáo kết quả audit đi kèm. Kết quả smoke
không phải kết quả hội tụ. Môi trường CPU, thiếu PESQ hoặc GPU pilot chưa chạy
phải được ghi là phần chưa xác nhận, không được đổi thành trạng thái đạt đầy đủ.

Gói audit đã hoàn tất nằm tại `run_logs/vnspeechmix_audit_bundle.zip`.
Kiểm tra tính toàn vẹn của gói đã có:

```bash
python scripts/backup_run.py verify run_logs/vnspeechmix_audit_bundle.zip
```

Gói audit CPU này giữ trạng thái còn thiếu GPU/PESQ; không phải gói kết quả
train chính thức. Giữ cả `.zip.sha256` khi sao chép.
Các script tổng hợp/đóng gói riêng cho đợt audit được lưu trong snapshot của ZIP.
Đóng gói các run mới bằng `scripts/backup_run.py` theo mục 4.
