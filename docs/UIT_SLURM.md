# Chạy SepReformer trên Slurm UIT

Hướng dẫn này dựa trên `HuongDanSuDungSlurm.docx` do người dùng cung cấp.
Chưa đăng nhập hoặc chạy job trên cụm trường. Đối chiếu mẫu hiện hành
`/datastore/templates/job_template.slurm` khi lên server.

## 1. Tài khoản và nơi lưu

- Dùng mạng trường hoặc VPN rồi SSH port 22 đến `slurm.uit.edu.vn`.
- Đặt code, dataset, venv, cache và kết quả dưới `/datastore/<username>`.
- Tài liệu quy định `/home/<username>` tối đa 30 GB; không lưu dataset ở đó.
- Theo tài liệu: tối đa 20 MPS, 5 job đồng thời, 32 vCPU/tài khoản, 72 giờ/job.
  VRAM khai báo phải nhỏ hơn mức GPU: 80 GB A100, 44 GB L40.
- MPS là tài nguyên chia sẻ; `mps:2` không có nghĩa hai GPU độc lập.

Sau khi đăng nhập:

```bash
whoami
mkdir -p "/datastore/$USER/experiments" "/datastore/$USER/.cache/pip"
export PIP_CACHE_DIR="/datastore/$USER/.cache/pip"
export XDG_CACHE_HOME="/datastore/$USER/.cache"
df -h "/datastore/$USER"
module avail
cat /datastore/templates/job_template.slurm
```

## 2. Chuyển bản code mới nhất

Thay đổi local chưa tự xuất hiện trên GitHub. Commit/push bản đã kiểm tra trước khi
clone, hoặc chuyển ZIP source chứa cả file mới chưa track. Không dùng `git archive`
để lấy thay đổi chưa commit. Không chuyển venv Windows lên Linux.

Nếu repository đã cập nhật và tài khoản trường truy cập được:

```bash
cd "/datastore/$USER/experiments"
git clone https://github.com/ntngoclan/sepreformer-libri2mix-16k.git SepReformer
cd SepReformer
mkdir -p run_logs
git log -1 --oneline
ls scripts/submit_job.slurm scripts/setup_env.sh scripts/train.sh
```

Nếu GitHub bị hạn chế đăng nhập, chuyển source bằng SCP/SFTP thay vì sửa chính
sách xác thực của server. Đối với ZIP, giữ đúng một tầng thư mục repo.

## 3. Môi trường Python

Mẫu trường dùng python312 nhưng bộ pin Torch 2.1.2 hiện tại nên dùng Python 3.10
hoặc 3.11. Xem `module avail`, load đúng tên module thật sự có trên cụm rồi kiểm tra:

```bash
python3 --version
```

Không mặc định tên module là python310/python311. Nếu cụm chỉ cung cấp 3.12,
cần chuẩn bị Python 3.10/3.11 riêng hoặc kiểm thử một bộ dependency mới; không
tiếp tục cài nguyên bộ pin cũ bằng Python 3.12.

Sau khi Python đúng phiên bản, tại repo:

```bash
PYTHON_BIN="$(command -v python3)" bash scripts/setup_env.sh
source .venv/bin/activate
python -c "import torch; from torch.utils.tensorboard import SummaryWriter; print(torch.__version__, torch.version.cuda)"
```

Đây là cài dependency; không chạy training trên login node. Nếu trường yêu cầu
cài môi trường trên compute node, thực hiện bước cài trong allocation được cấp.
Nếu Python phụ thuộc environment module, dùng đúng module đó khi submit; có thể
export `PYTHON_MODULE` bằng tên đã xác nhận để mẫu job load lại trên compute node.

## 4. Dữ liệu

Theo [protocol chung](COMMON_16K_PROTOCOL.md), đặt:

```text
data/VnSpeechMix/rendered/
  rendered_metadata.csv
  train/{mix_clean,s1,s2}/     # 18000 bộ ba
  valid/{mix_clean,s1,s2}/     # 3000 bộ ba
  test/{mix_clean,s1,s2}/      # 5000 bộ ba

data/Libri2Mix/wav16k/min/
  train-100/{mix_clean,s1,s2}/ # 13900 bộ ba
  dev/{mix_clean,s1,s2}/       # 3000 bộ ba
  test/{mix_clean,s1,s2}/      # 3000 bộ ba
```

Copy audio đã render/giải nén; Git không chứa audio. Giữ nguyên metadata VN.
Với Libri2Mix, tạo manifest sau khi chuyển dữ liệu:

```bash
python -B scripts/prepare_libri2mix_manifest.py
```

Script đọc/hash mọi WAV; bố trí bước I/O dài này theo quy định của trường,
có thể trong một job CPU. Chưa có Libri2Mix thì bắt đầu riêng VN trước.
Tính dung lượng cho cả ZIP, dữ liệu giải nén, venv và checkpoint của các run.

## 5. Preflight GPU trước

Mẫu `scripts/submit_job.slurm` yêu cầu thử nghiệm ban đầu: 2 MPS, 14 CPU,
32 GB RAM, 24 GiB VRAM trống, tối đa 2 giờ. Đây là yêu cầu khởi điểm,
chưa phải số đo model; điều chỉnh sau báo cáo thực tế. 14 CPU dành chỗ cho
12 DataLoader worker và tiến trình chính. Chạy một job trước để đo tài nguyên.

Luôn submit từ repo root và tạo `run_logs` trước, vì Slurm mở log trước khi
script chạy. Mẫu giữ đúng luồng helper của trường, xử lý requeue code 10,
lỗi code 11, rồi truyền GPU đã chọn cho lệnh Python.

```bash
mkdir -p run_logs
sbatch --job-name=vn_base_check scripts/submit_job.slurm \
  python -B scripts/preflight_vnspeechmix_baseline.py \
  --root . --model SepReformer_Base_VnSpeechMix_16K \
  --device cuda --samples 64000 --steps 2 --workers 12 \
  --report run_logs/vn_base_uit_check.json

squeue -u "$USER"
```

Ghi Job ID được in ra. Dùng `tail -f run_logs/slurm_<JOB_ID>.out` và xem `.err`.
Sau khi kết thúc, xem `run_logs/vn_base_uit_check.json`: yêu cầu
`status: smoke_passed`, loss hữu hạn, optimizer cập nhật, checkpoint roundtrip
qua. Code exit 0 riêng lẻ không đủ: helper có thể vừa requeue.

LTRR VN: lặp lệnh trên với `--model SepReformer_LTRR_VnSpeechMix_16K` và report
khác. Libri2Mix: dùng hai model `SepReformer_Base_Libri2Mix_16K` và
`SepReformer_LTRR_Libri2Mix_16K`, sau khi chuẩn bị manifest.

## 6. Pilot hai epoch cho cặp VN

Tạo config pilot riêng sau preflight; không sửa config chính:

```bash
python - <<'PY'
from pathlib import Path
import yaml
Path('run_logs').mkdir(exist_ok=True)
for variant in ('Base', 'LTRR'):
    model = f'SepReformer_{variant}_VnSpeechMix_16K'
    doc = yaml.safe_load(Path(f'models/{model}/configs.yaml').read_text())
    cfg = doc['config']
    cfg['engine']['max_epoch'] = 2
    cfg['engine']['checkpoint_epochs'] = [1, 2]
    cfg['paired_initialization']['directory'] = 'initializations/vnspeechmix_common_v2_pilot'
    with Path(f'run_logs/vn_{variant.lower()}_pilot.yaml').open('x') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
PY

sbatch --job-name=vn_base_pilot --time=24:00:00 scripts/submit_job.slurm \
  python -u run.py --model SepReformer_Base_VnSpeechMix_16K \
  --config run_logs/vn_base_pilot.yaml --seed 0 --run-id vn_base_uit_pilot_s0
```

Sau khi baseline pilot hoàn tất, submit LTRR với config/run-id tương ứng.
24 giờ là thời hạn yêu cầu, không phải cam kết hoàn tất hai epoch. Đo thời gian
train/valid mỗi epoch, bộ nhớ GPU, lưu checkpoint và kiểm tra resume trước run dài.
Các bài test CPU chạy qua sbatch tương tự: `python -B -m scripts.test_ltrr`,
`python -B -m scripts.test_libri2mix_protocol`, `python -B -m scripts.runtime_checks`.

## 7. Run chính thức và giới hạn 72 giờ

Chỉ bắt đầu sau pilot đạt yêu cầu:

```bash
sbatch --job-name=vn_base --time=72:00:00 scripts/submit_job.slurm \
  python -u run.py --model SepReformer_Base_VnSpeechMix_16K \
  --seed 0 --run-id vn_base_uit_s0
```

Không có cam kết 200 epoch chạy xong trong 72 giờ. Job kết thúc do time limit
không tự resume; code chỉ lưu checkpoint ở cuối epoch. Có thể mất phần epoch
đang chạy. Sau đó submit job mới bằng cùng config/code/seed và run-id mới:

```bash
sbatch --job-name=vn_base_resume --time=72:00:00 scripts/submit_job.slurm \
  python -u run.py --model SepReformer_Base_VnSpeechMix_16K \
  --seed 0 --run-id vn_base_uit_s0_part2 \
  --resume runs/SepReformer_Base_VnSpeechMix_16K/seed_0000/vn_base_uit_s0/checkpoints/latest.pth
```

Chỉ chạy lệnh resume khi file đó tồn tại và job cũ đã dừng. Giữ nguyên toàn bộ
run trước đó. Không resume checkpoint pilot hai epoch vào config 200 epoch.
Checkpoint/config/source từ protocol cũ cũng không dùng để resume v2.
`scancel <JOB_ID>` hủy job; `sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode,MaxRSS`
xem trạng thái sau khi job không còn trong squeue. Đóng SSH không dừng job sbatch;
tmux chỉ hữu ích khi thao tác phiên làm việc, không kéo dài allocation.
