# Sơ đồ LTRR đối chiếu với code hiện tại

## Các file

- `ltrr_architecture_complete.pdf`: đủ 3 trang vector.
- `sepreformer_ltrr_architecture.{pdf,svg}` và `_4k.png`: sơ đồ tổng thể theo bố cục mẫu.
- `ltrr_refinement_detail.{pdf,svg}` và `_4k.png`: residual LTRR và từng phép toán Conv-U.
- `ltrr_dense_memory_detail.{pdf,svg}` và `_4k.png`: bốn tầng dense dilated memory, nối đầy đủ các state.
- `ltrr_shape_audit.json`: kích thước tensor đo bằng forward hooks, số tham số và SHA-256 của source/config.

Nguồn: `models/SepReformer_LTRR_VnSpeechMix_16K/{configs.yaml,model.py,modules/module.py,modules/network.py,modules/ltrr.py}`.
Kiến trúc và cấu hình model cũng khớp với gói LTRR Libri2Mix hiện tại. Đây là protocol mới, không phải config kernel 16 trong thư mục legacy.

## Phạm vi và quy ước

- Bố cục: encoder đi xuống E0 → E1 → E2 → E3 → bottleneck; decoder đi lên D0 → D1 → D2 → D3. Chỉ số là chỉ số `ModuleList` trong Python.
- Các skip lấy **trước DownConv**, qua cùng một `spk_split_block`, rồi đến fusion ở decoder có cùng độ phân giải thời gian. E0 nối D3, E1 nối D2, E2 nối D1, E3 nối D0.
- `×2` và `×3` ở encoder/decoder chỉ các block nối tiếp có tham số riêng; không có nghĩa các block lặp dùng chung trọng số. Các phép toán decoder được dùng chung cho các speaker khi gộp vào chiều BJ.
- `×` trong nhánh LTRR là nhân từng phần tử; vòng tròn `+` là cộng residual. Không thêm speaker gate.
- Hình tổng thể giữ các khối Global/Local/CS ở mức module; cấu tạo tương ứng EGA→GCFN, CLA→GCFN và attention theo speaker→GCFN được ghi dưới hình. Hai trang phóng to triển khai đầy đủ các phép toán của phần đóng góp LTRR.
- Nhánh auxiliary được vẽ riêng ở dưới để không lẫn vào đường suy luận chính. Khi `return_aux=False`, nhánh này không được tính.
- PDF/SVG là vector; PNG rộng 4200 pixel dùng xem nhanh/chèn slide.

## Những điểm đã sửa so với ảnh tham khảo

1. AudioEncoder/AudioDecoder có kernel 32, stride 8; không ghi kernel 16 của thí nghiệm cũ.
2. FeatureProjector có GroupNorm(1,256) trước Conv1d 256→128.
3. Có `bottleneck_G` gồm 2 cặp Global→Local, không giảm mẫu, sau bốn encoder stage.
4. Speaker split: Conv1d 128→1024, GLU theo channels 1024→512, Conv1d 512→256, view thành BJ×128×T, GroupNorm(1,128). Một module được gọi 5 lần, không phải 5 bộ tham số khác nhau.
5. Fusion decoder là concat theo channel rồi **Conv1d 256→128, kernel 1** trong code. Hình hiện tại ghi đúng tên lớp triển khai; cách ghi “Concat. + Linear” ở Hình 2 của bài báo cũng phù hợp về nguyên lý, vì pointwise Conv1d groups=1 thực hiện phép ánh xạ tuyến tính theo channel tại từng frame.
6. Chỉ output cuối D3 qua một LTRR. Không nối output của cả bốn decoder stage vào LTRR.
7. Conv-U có residual nội bộ. Cả hai nhánh u/v có cùng cấu trúc nhưng trọng số khác nhau.
8. Dense memory nhận v, tạo x0, rồi y1/y2/y3/y4 tuần tự. Đầu vào các projection là 64/128/192/256 channel; cuối cùng chỉ concat y1…y4 (256 channel), không concat x0 vào đầu ra cuối.
9. Main OutputLayer: crop về T0, Linear 128→512, GLU 512→256, Linear 256→256. `masking=False`; encoder_output chỉ cung cấp chiều dài T0 cho head chính.
10. Auxiliary lấy đặc trưng **trước** mỗi decoder stage: T=500/1000/2000/4000. Mỗi head có OutputLayer riêng, masking ReLU nhân encoder features, rồi AudioDecoder riêng.

## Kích thước đã đo

Forward thực trên CPU với model chưa train ở eval mode, waveform zero chỉ dùng kiểm tra shape; không phải dữ liệu hay kết quả tách tiếng thực nghiệm. Dropout không hoạt động trong phép đo eval này, nhưng xác suất trong hình lấy từ cấu hình/module khi huấn luyện.

| Vị trí | Shape với B=2, J=2, N=64000 |
|---|---|
| Mixture | `[2,64000]` |
| AudioEncoder | `[2,256,7997]` |
| FeatureProjector | `[2,128,7997]` |
| Separator padding | `[2,128,8000]` |
| Sau các DownConv | T=4000,2000,1000,500; channels=128 |
| Bottleneck sau split | `[4,128,500]` |
| Decoder D0…D3 | `[4,128,1000/2000/4000/8000]` |
| LTRR input/output | `[4,128,8000]` |
| Bottleneck của LTRR | `[4,64,8000]` |
| Main OutputLayer | `[2,2,256,7997]`, thứ tự `[J,B,F0,T0]` |
| Hai waveform cuối | Python list chứa 2 tensor `[2,64000]` |

LTRR có **89.031** tham số. Toàn bộ model gồm cả auxiliary heads có **14.805.191** tham số. Đây là số đếm parameter từ code, không phải FLOPs, tốc độ hay kết quả SI-SNRi.

Shape cụ thể trên hình áp dụng cho crop 4 giây và batch 2. Với N khác, code pad waveform về bội stride (và minimum theo train/eval), rồi pad feature về bội 16; không cố định mọi utterance thành 8000 frame. `input_sizes` cho phép xử lý nhóm độ dài thực riêng. Với B=1 và waveform thông thường N>1, nhánh điều kiện trong `AudioDecoder.forward` chỉ squeeze chiều channel, nên đầu ra vẫn giữ shape `[1,N]`. `Model.forward` cũng thêm chiều batch khi nhận waveform `[N]`; ghi chú trước đây nói trường hợp này trả `[N]` đã được sửa theo code.

## Tạo lại

Trong môi trường có PyTorch/dependency của repo:

```bash
python -B scripts/audit_ltrr_shapes.py
```

Để vẽ, cần thêm `reportlab` và `pymupdf` (nên dùng môi trường vẽ riêng, không đổi dependency của run train đang chạy):

```bash
python scripts/draw_ltrr_architecture.py
```

Generator hỗ trợ `--extra-pythonpath <thư_mục_dependency_riêng>`, kiểm tra SHA-256 source/config, glyph font và giới hạn chữ trong trang. Không thay đổi model, config, dataset hoặc checkpoint. Sau khi source kiến trúc thay đổi, phải kiểm tra lại nội dung sơ đồ, không chỉ chạy lại generator để lấy hình cũ.
