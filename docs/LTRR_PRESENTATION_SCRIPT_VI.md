# Lời thuyết trình kiến trúc SepReformer–LTRR

Tài liệu đi theo ba trang trong `figures/ltrr_architecture_complete.pdf`. Các đoạn trong dấu trích dẫn là lời có thể đọc khi trình bày; phần “Chỉ hình” là gợi ý thao tác. Công thức và bảng dùng để đối chiếu hoặc giải thích khi được hỏi sâu.

**Bản giải thích khái niệm:** [Hiểu kiến trúc SepReformer–LTRR bằng các khái niệm AI](LTRR_THEORY_EXPLANATION_VI.md). Bản này giải thích trực tiếp từng khối theo mạch: khái niệm AI → phép toán → ý nghĩa dễ hiểu → vai trò trong mô hình. Các nội dung gồm representation learning, channel/temporal mixing, attention, normalization, parameter sharing, residual learning, dense connectivity và receptive field. Dùng bản đó để hiểu bản chất; dùng 27 mục dưới đây để theo sơ đồ, thao tác và kích thước tensor.

Trong mỗi mục, phần **Chức năng và mục đích** giải thích vì sao có khối đó, khối làm gì với thông tin và cách hiểu đúng đầu ra. Phần lời nói mẫu tiếp theo mô tả luồng xử lý và kích thước cụ thể. “Mục đích” là cách diễn giải cơ chế thiết kế từ code, không có nghĩa tác dụng ấy đã được chứng minh bằng kết quả thực nghiệm.

Nguồn đối chiếu: gói `models/SepReformer_LTRR_VnSpeechMix_16K`, gồm `configs.yaml`, `model.py`, `modules/module.py`, `modules/network.py`, `modules/ltrr.py`; kết quả đo shape tại `figures/ltrr_shape_audit.json`. Cấu hình model của gói LTRR Libri2Mix hiện tại giống cấu hình model VN-SpeechMix. Thông số dưới đây không áp dụng thay thế cho config lịch sử trong `legacy/`.

## Mở đầu và quy ước

**Chỉ hình:** toàn bộ trang 1, sau đó chỉ phần LTRR màu tím.

> “Em sử dụng SepReformer làm kiến trúc nền cho bài toán tách hai người nói từ một tín hiệu hỗn hợp đơn kênh. Phần bổ sung là LTRR, viết tắt của Lightweight Temporal Residual Refinement. Đây là tên hiện tại của thiết kế NoGate đã được triển khai trong code đồ án. LTRR xử lý đặc trưng sau tầng tái cấu trúc cuối cùng và trước khi chuyển đặc trưng trở lại waveform.
>
> Em sẽ trình bày theo ba mức: đầu tiên là toàn bộ luồng xử lý của mô hình; tiếp theo là các nhánh và residual trong LTRR cùng Conv-U; cuối cùng là cấu trúc dense dilated memory.
>
> Các kích thước cụ thể trong hình tương ứng với batch gồm hai hỗn hợp, mỗi hỗn hợp dài bốn giây, lấy mẫu ở 16 kHz. Mỗi hỗn hợp chứa hai người nói. Vì vậy, một mẫu có 64.000 giá trị waveform, còn batch đầu vào có kích thước [2,64000]. Đây là ví dụ theo cấu hình huấn luyện hiện tại, không phải yêu cầu mọi bản ghi khi suy luận đều phải dài bốn giây.”

| Ký hiệu | Ý nghĩa | Giá trị trong hình |
|---|---|---|
| B | Số hỗn hợp trong batch | 2 |
| J | Số nguồn/người nói cần tách trong mỗi hỗn hợp | 2 |
| N | Số mẫu waveform ban đầu | 64000 |
| Npad | Độ dài waveform sau padding nội bộ | 64000 trong ví dụ này |
| F0 | Số kênh của AudioEncoder | 256 |
| F | Số kênh đặc trưng separator | 128 |
| Cb | Số kênh bottleneck LTRR | 64 |
| T0 | Số frame ngay sau AudioEncoder | 7997 |
| Tp | Số frame sau padding bên trong separator | 8000 |
| BJ | Gộp batch và số nguồn vào một chiều | 4 |

> “Trong hình, các tensor đặc trưng thường được ghi theo thứ tự [batch, channels, time]. Riêng những đoạn dùng Linear hoặc attention, code có đổi trục sang [batch, time, channels] rồi đổi lại. Đổi trục chỉ thay cách sắp xếp tensor để phù hợp với phép toán; không tự thêm hay bớt thông tin.
>
> BJ bằng 4 có nghĩa là hai hỗn hợp, mỗi hỗn hợp có hai nhánh nguồn. Nó không có nghĩa mỗi hỗn hợp chứa bốn người nói. Các kênh 256, 128 và 64 là chiều đặc trưng học được, không phải số người nói và cũng không phải số bin tần số cố định.”

## Trang 1 — Kiến trúc tổng thể

Trước khi đọc từng khối, cần phân biệt ba công việc: **biến đổi biểu diễn** từ waveform sang đặc trưng; **ước lượng đặc trưng của từng nguồn** bằng separator; và **tổng hợp waveform** từ đặc trưng đầu ra. LTRR nằm ở cuối công việc thứ hai. Nó không nhận một file âm thanh riêng, không chạy lại toàn bộ separator và không cần biết danh tính người nói.

### 1. Từ waveform đến AudioEncoder

**Chức năng và mục đích:** Waveform chỉ là một chuỗi giá trị biên độ theo thời gian. AudioEncoder biến từng cửa sổ ngắn của chuỗi này thành một vector 256 chiều để các khối phía sau có biểu diễn học được để xử lý. Mỗi kênh đầu ra ứng với một bộ lọc được cập nhật trong quá trình train; không thể mặc định kênh nào là “giọng nam”, “giọng nữ” hay một tần số cụ thể.

Conv1d tính tổng có trọng số trên 32 mẫu đầu vào cho mỗi bộ lọc rồi dịch 8 mẫu để tính vị trí kế tiếp. Các cửa sổ chồng lấn nhau nên một mẫu waveform có thể tham gia nhiều frame. GELU bổ sung tính phi tuyến sau phép lọc; nó cho phép biểu diễn phụ thuộc vào giá trị đầu vào theo cách vượt ra ngoài một phép biến đổi tuyến tính đơn thuần. GELU không bảo đảm loại bỏ nhiễu hoặc giữ riêng tiếng nói.

Stride làm chuỗi đặc trưng có ít vị trí thời gian hơn waveform, nhưng đồng thời số kênh tăng từ 1 lên 256. Vì vậy không gọi đây là “nén dữ liệu 8 lần” về tổng số phần tử. Padding waveform chỉ phục vụ điều kiện độ dài của kiến trúc; các số 0 được thêm không phải thông tin âm thanh mới. Đầu ra encoder vẫn mô tả **hỗn hợp**, chưa phải hai người nói đã tách.

**Chỉ hình:** cột Waveform front end, từ Mixture đến AudioEncoder.

> “Đầu vào là waveform hỗn hợp x có kích thước [B,N], trong ví dụ là [2,64000]. Mô hình kiểm tra độ dài và có thể thêm các giá trị 0 ở cuối waveform để phù hợp với stride, đồng thời xử lý trường hợp đầu vào quá ngắn. Với đoạn bốn giây ở đây, N đã phù hợp nên Npad vẫn bằng 64.000.
>
> AudioEncoder thêm chiều channel cho tín hiệu mono, chuyển đầu vào thành [2,1,64000]. Sau đó, Conv1d ánh xạ từ một kênh sang 256 kênh với kernel 32, stride 8, không padding và không bias, rồi áp dụng GELU.
>
> Mỗi bộ lọc nhìn vào một cửa sổ gồm 32 mẫu waveform. Sau mỗi lần dịch 8 mẫu, nó tạo đặc trưng cho một vị trí thời gian mới. Ở 16 kHz, 32 mẫu tương ứng 2 mili giây, còn bước dịch 8 mẫu tương ứng 0,5 mili giây. Đây là bộ mã hóa học được, không phải một phép STFT cố định.”

Số frame của Conv1d trong cấu hình này:

```text
T0 = floor((Npad − kernel_size) / stride) + 1
   = (64000 − 32) / 8 + 1
   = 7997

[2,1,64000] → [2,256,7997]
```

> “Vì convolution không padding nên đầu ra thực tế có 7997 frame. Không thể chỉ lấy 64.000 chia 8 rồi nói AudioEncoder tạo ra 8000 frame. Số 8000 xuất hiện ở bước padding đặc trưng phía sau.”

### 2. FeatureProjector và padding đặc trưng

**Chức năng và mục đích:** FeatureProjector chuyển biểu diễn 256 kênh của bộ mã hóa sang độ rộng 128 kênh mà separator sử dụng. Phép chiếu này học cách phối hợp các kênh encoder, đồng thời giới hạn độ rộng tính toán trong các block tiếp theo. Nó có thể làm mất một phần thông tin trong nhánh được chiếu; việc giảm kênh không tự chứng minh chỉ giữ lại “thông tin quan trọng”.

GroupNorm một nhóm tính thống kê trên các chiều channel và time của **từng phần tử batch**, chuẩn hóa rồi áp dụng hệ số affine học được. Mục đích là điều chỉnh thang giá trị đặc trưng trước phép chiếu, không phải chuẩn hóa âm lượng waveform hay trộn các mẫu trong batch. Epsilon tránh phép chia không ổn định khi phương sai rất nhỏ. Conv1d kernel 1 sau đó trộn 256 kênh ở cùng một frame thành 128 kênh; nó không đọc thêm frame lân cận.

**Đối chiếu bài báo:** §3.2, tr.4 mô tả input layer với Linear và LayerNorm theo frame. Ở đây phải giữ đúng tên GroupNorm của code, vì hai cách chuẩn hóa có trục thống kê khác nhau. Riêng Conv1d kernel 1 với groups=1 có thể thực hiện cùng phép ánh xạ channel như Linear ở từng frame; không nên xem khác tên lớp là tự động khác nguyên lý.

Padding từ 7997 lên 8000 giúp chiều dài chia hết cho 16, phù hợp với bốn lần giảm độ phân giải theo hệ số 2. Padding giải quyết sự khớp kích thước giữa encoder, skip và decoder. Ba frame thêm vào ban đầu là 0 nhưng có thể trở thành giá trị khác 0 sau các phép xử lý; cuối đường chính chúng được crop theo T0. Đây không phải cơ chế attention mask bỏ qua mọi ảnh hưởng của padding.

**Chỉ hình:** FeatureProjector, sau đó Separator.pad_signal.

> “Đặc trưng 256 kênh được đưa vào FeatureProjector. Khối này sử dụng GroupNorm với một nhóm trên 256 kênh, sau đó dùng Conv1d kernel 1 để giảm số kênh từ 256 xuống 128.
>
> Convolution kernel 1 phối hợp thông tin giữa các kênh tại cùng vị trí thời gian; nó không giảm số frame. Vì vậy, kích thước chuyển từ [2,256,7997] sang [2,128,7997]. Trong cấu hình này, GroupNorm dùng epsilon bằng 10 mũ trừ 8.
>
> Separator có bốn lần giảm chiều thời gian theo hệ số hai. Vì vậy, code bổ sung padding bên phải để chiều dài đặc trưng chia hết cho 2 mũ 4, tức 16. Từ 7997 frame, mô hình thêm ba frame 0, tạo thành [2,128,8000].”

```text
Tp = 16 × ceil(T0 / 16) = 8000
```

Lưu ý khi giải thích: GroupNorm(1,C) ở đây tính thống kê trên các kênh và vị trí thời gian của từng phần tử batch. Nó khác với ChannelLayerNorm trong LTRR, vốn chuẩn hóa theo kênh ở từng thời điểm.

### 3. Separation Encoder: bốn tầng giảm độ phân giải

**Chức năng và mục đích:** Encoder xây dựng các biểu diễn ở nhiều độ phân giải thời gian. Ở mỗi tầng, Global và Local xử lý thông tin trước khi DownConv giảm số vị trí. Khi chiều dài giảm, các phép xử lý phía sau làm việc trên chuỗi ngắn hơn; mỗi vị trí ở mức thô tương ứng với khoảng thời gian lớn hơn so với mức trước. Không nên diễn giải các tầng thành những mức ngữ nghĩa cố định như “âm vị → từ → câu”, vì code không áp đặt các nhãn đó.

DownConv học bộ lọc trên từng kênh rồi lấy đầu ra với stride 2. Nó không chỉ bỏ mỗi frame thứ hai và cũng không tự trộn giữa các kênh. BatchNorm sau convolution chuẩn hóa từng kênh bằng thống kê theo batch và time khi train; khi eval, mặc định dùng thống kê tích lũy. GELU bổ sung phi tuyến. Giảm độ phân giải có thể làm mất chi tiết; skip giữ biểu diễn **trước khi giảm** để decoder có một đường truy cập lại thông tin ở mức mịn hơn.

Các skip là tensor đặc trưng được giữ lại, không phải một mạng con tự loại nhiễu. Chúng giúp decoder có thêm đầu vào bên cạnh đường bottleneck, nhưng không bảo đảm phục hồi nguyên vẹn mọi chi tiết của tín hiệu gốc.

**Chỉ hình:** E0 ở trên cùng; di chuyển xuống E1, E2, E3.

> “Luồng encoder đi từ trên xuống theo E0, E1, E2 và E3. Cách đánh số này theo trực tiếp chỉ số trong code Python.
>
> Mỗi tầng xử lý theo thứ tự Global Block, Local Block, rồi thêm một Global Block và một Local Block. Ký hiệu nhân hai trong hình biểu thị hai cặp nối tiếp, có các bộ tham số riêng, không phải gọi lặp cùng một block dùng chung trọng số.
>
> Sau các khối Global và Local, tensor được tách thành hai đường. Một đường được giữ làm skip connection ở độ phân giải hiện tại. Đường chính đi tiếp vào DownConv để giảm chiều thời gian một nửa.
>
> DownConv là depthwise convolution với 128 kênh, groups bằng 128, kernel 5, stride 2 và padding 2; sau convolution là BatchNorm và GELU. Vì vậy, nó giảm số frame nhưng vẫn giữ 128 kênh.”

| Tầng | Đầu vào và skip trước DownConv | Đầu ra sau DownConv |
|---|---|---|
| E0 | `[2,128,8000]` | `[2,128,4000]` |
| E1 | `[2,128,4000]` | `[2,128,2000]` |
| E2 | `[2,128,2000]` | `[2,128,1000]` |
| E3 | `[2,128,1000]` | `[2,128,500]` |

> “Điểm cần chú ý là skip được lấy trước DownConv. Nhờ vậy, khi decoder tăng độ phân giải trở lại, nó có thể kết hợp đặc trưng đang tái cấu trúc với đặc trưng encoder ở đúng cùng thang thời gian.”

### 4. Global, Local và GCFN thực sự làm gì?

**Chức năng và mục đích — Global/EGA:** Với chuỗi dài 8000 frame, attention trực tiếp trên toàn bộ vị trí cần ma trận quan hệ theo hai chiều thời gian. EGA đưa chuỗi về 500 vị trí bằng adaptive average pooling trước khi attention. Mỗi giá trị pooled tổng hợp một vùng đầu vào, nên giảm chiều dài cũng làm giảm độ chi tiết temporal mà attention trực tiếp nhìn thấy. Mục đích là khai thác quan hệ xa trên biểu diễn ngắn hơn, sau đó đưa thông tin ấy trở lại độ phân giải của tầng.

Trong attention, LayerNorm chuẩn hóa đặc trưng; ba Linear học ra **Q, K, V**. Có thể hiểu Q là biểu diễn dùng để tìm thông tin phù hợp, K để tính mức phù hợp, và V là nội dung được tổng hợp. Mỗi head tính điểm giữa các vị trí từ tích Q–K, cộng thành phần vị trí tương đối, chia cho căn bậc hai của 16 rồi softmax theo các vị trí được đọc. Các trọng số đó dùng để lấy tổng có trọng số của V. Tám head có các phép chiếu học được khác nhau; không có quy định một head chuyên nhận diện một speaker hay một loại âm.

Relative positional encoding giúp điểm attention phụ thuộc cả vào **khoảng cách và chiều trước/sau giữa hai vị trí**, không chỉ vào độ giống của nội dung. Sau khi gộp head và chiếu đầu ra, LayerScale điều chỉnh độ lớn nhánh attention. Nearest interpolation đưa kết quả trở lại chiều dài ban đầu; nội suy không tự tạo lại chi tiết đã mất ở pooling. Nhánh `LayerNorm → Linear → Sigmoid` tính hệ số theo từng frame/kênh từ đặc trưng gốc, điều chỉnh lượng thông tin attention được cộng vào residual. Đây là gate của EGA, không phải speaker gate đã bỏ trong LTRR.

**Chức năng và mục đích — Local/CLA:** CLA bổ sung phép lọc temporal tại độ phân giải hiện tại, dùng depthwise kernel 65. Khác với attention tạo trọng số quan hệ phụ thuộc nội dung đầu vào, convolution dùng các trọng số kernel đã học và áp dụng cùng cách dọc thời gian. Cửa sổ 65 frame có ý nghĩa thời gian khác nhau ở các tầng có độ phân giải khác nhau; không được gán một số mili giây duy nhất cho mọi Local Block.

LayerNorm chuẩn bị thang giá trị cho nhánh biến đổi. Linear và GLU tạo biểu diễn có điều tiết theo kênh: GLU tách tensor làm hai nửa A, G rồi tính `A × sigmoid(G)`, nên 256 chiều thành 128. Depthwise convolution xử lý lân cận của từng kênh. Các Linear phía sau phối hợp các kênh, BatchNorm điều chỉnh thống kê, GELU bổ sung phi tuyến và Dropout phục vụ regularization. LayerScale và residual cho phép thêm phần biến đổi vào đường đặc trưng đang có thay vì buộc toàn bộ đầu ra chỉ đi qua nhánh mới.

**Chức năng và mục đích — GCFN:** Sau bước attention hoặc CLA, GCFN tiếp tục biến đổi đặc trưng bằng một nhánh feed-forward có convolution và GLU. Linear mở rộng 128 lên 768 để cung cấp không gian trung gian rộng hơn; depthwise kernel 3 cho các đặc trưng mở rộng tương tác với vị trí temporal gần nhau; GLU còn 384 rồi Linear đưa về 128 để cộng residual. Nó kết hợp biến đổi theo kênh và xử lý temporal ngắn trong cùng nhánh, không phải một attention bổ sung. Trong SpkAttention, code đã đưa tensor về dạng chuỗi thời gian của từng nguồn trước khi gọi GCFN; kernel 3 ở đây vẫn chạy theo **time**, không chạy theo hai speaker.

LayerScale của backbone là các hệ số học được theo chiều đặc trưng, khởi tạo nhỏ. Mục đích là kiểm soát mức đóng góp ban đầu của các nhánh biến đổi sâu. Nó khác scalar α duy nhất của LTRR. Các residual cung cấp đường truyền trực tiếp cho giá trị và gradient, nhưng không phải bảo đảm rằng mô hình sẽ hội tụ hay chất lượng luôn tăng.

Phần này giải thích các hộp được gộp trong hình; có thể rút ngắn khi thời gian báo cáo hạn chế.

> “Global Block gồm EGA rồi đến GCFN. Trong EGA, code dùng adaptive average pooling để đưa chuỗi đặc trưng về chiều dài dùng cho attention. Với ví dụ này, chiều dài đó là Tp chia 16, bằng 500, kể cả khi đầu vào của tầng đang có 8000 hay 4000 frame.
>
> Attention dùng tám head, mỗi head có 16 chiều vì tổng chiều đặc trưng là 128. Sau attention, đặc trưng được nội suy nearest trở lại chiều dài của tầng. Một nhánh LayerNorm, Linear và Sigmoid từ đặc trưng gốc điều biến đặc trưng attention trước khi cộng residual. Do đó, việc bỏ speaker gate trong LTRR không có nghĩa SepReformer nền không còn các cơ chế gating khác.
>
> Relative positional encoding cung cấp embedding theo độ lệch vị trí cho nhánh global attention. Cấu hình có bảng embedding 4000 hàng và 16 chiều; tensor pos_k trong ví dụ là [500,500,16]. Đây là encoding vị trí, không phải embedding danh tính người nói.”

> “Local Block gồm CLA rồi đến GCFN. CLA dùng LayerNorm, Linear 128 sang 256 và GLU để trở về 128 kênh. Sau đó là depthwise convolution kernel 65, padding same, nên chiều thời gian được giữ nguyên. Tiếp theo, Linear tăng lên 256 kênh, BatchNorm, GELU, Linear đưa về 128 và Dropout. Nhánh này được nhân LayerScale rồi cộng với đầu vào.
>
> GCFN xuất hiện trong cả Global Block, Local Block và nhánh attention theo speaker. Nó dùng LayerNorm và Linear 128 sang 768, depthwise convolution kernel 3 theo thời gian, rồi GLU giảm từ 768 còn 384. Sau Dropout, Linear đưa từ 384 về 128, thêm Dropout, LayerScale và cộng residual. Do đó, chiều đặc trưng cuối của các block vẫn là 128.
>
> Có thể hiểu Global Block khai thác quan hệ theo thời gian trên chuỗi đã pooling, còn Local Block bổ sung xử lý bằng convolution tại độ phân giải hiện tại. Đây là vai trò theo cấu trúc; mức đóng góp vào chất lượng tách tiếng phải được đánh giá bằng thực nghiệm.”

LayerScale của các block nền được khởi tạo `1e-5`; không nhầm tham số này với scalar `α=0.05` của LTRR. Các phép đổi trục phục vụ Linear, Conv1d và attention được lược bớt khỏi hình tổng thể để dễ đọc.

### 5. Bottleneck của separator

**Chức năng và mục đích:** Đây là điểm xử lý sâu nhất của đường encoder về độ phân giải temporal: chuỗi chỉ còn 500 vị trí. Hai cặp Global–Local tiếp tục biến đổi biểu diễn ở mức này trước khi tách thành nhánh nguồn để đi vào decoder. Với ví dụ hiện tại, độ dài pooling của global attention cũng là 500 nên không rút ngắn thêm chuỗi ở khối này.

Từ “bottleneck” mô tả vị trí có chiều thời gian ngắn trong cấu trúc encoder–decoder, không có nghĩa đặc trưng ở đây đã là waveform sạch. Nó giữ 128 kênh và không giảm thêm số frame; cần phân biệt với bottleneck **giảm kênh** bên trong LTRR.

**Chỉ hình:** `bottleneck_G`, dưới E3.

> “Sau bốn lần giảm mẫu, tensor có kích thước [2,128,500]. Code còn một khối bottleneck_G, gồm hai cặp Global rồi Local giống phần xử lý của encoder stage, nhưng không có DownConv. Vì vậy, đầu ra vẫn là [2,128,500].
>
> Bottleneck này thuộc separator. Nó khác với bottleneck 128 xuống 64 nằm bên trong LTRR: một khối xử lý đặc trưng ở độ phân giải thời gian thấp, còn khối kia giảm số kênh cho nhánh refinement.”

### 6. Speaker Split

**Chức năng và mục đích:** Speaker Split học phép ánh xạ từ một biểu diễn hỗn hợp thành hai nhóm đặc trưng để decoder có thể tạo hai đầu ra khác nhau. Hai Conv1d kernel 1 phối hợp thông tin channel; GLU điều tiết các giá trị trung gian; reshape chỉ sắp xếp hai nhóm kênh thành hai nhánh trong chiều BJ. **Khả năng tạo các nhánh khác nhau nằm ở phép chiếu học được**, không nằm ở thao tác reshape.

Việc gộp speaker vào batch cho phép áp dụng cùng các khối decoder cho mọi nhánh nguồn. Dùng chung một Speaker Split trên các thang thời gian giữ cùng phép ánh xạ tham số, nhưng đầu vào tại mỗi thang khác nhau nên đầu ra không giống nhau. Khối không sao chép một waveform rồi mặc định coi hai bản sao là hai người nói, và không tự gán danh tính cố định cho s1/s2. Ý nghĩa tách nguồn được học thông qua mục tiêu huấn luyện của toàn mạng.

**Đối chiếu bài báo:** Speaker Split thuộc cơ chế early split with a shared decoder (ESSD), §3.2 và Hình 3, tr.4. “Early” là phân nhánh trước reconstruction decoder, không phải khẳng định đã tách sạch tại đây. Bài báo mô tả LN sau split; code hiện tại dùng GroupNorm(1,128). Phụ lục C, tr.18 phân tích lựa chọn dùng chung split giữa các thang.

**Chỉ hình:** cột Speaker split, bắt đầu từ Split bottleneck rồi chỉ các skip.

> “Cho đến trước Speaker Split, đặc trưng vẫn được tổ chức theo từng hỗn hợp. Speaker Split chuyển nó thành các nhánh đặc trưng ứng với hai nguồn cần tách.
>
> Với một tensor [B,128,T], Conv1d kernel 1 tăng số kênh lên 1024. Con số này bằng 4 nhân 128 nhân hai nguồn. GLU chia đôi theo chiều channel, còn 512 kênh. Tiếp theo, Conv1d kernel 1 đưa 512 xuống 256, tương ứng hai nhóm 128 kênh.
>
> Code reshape từ [B,256,T] thành [BJ,128,T], rồi áp dụng GroupNorm với một nhóm trên 128 kênh. Với B bằng hai và J bằng hai, chiều đầu trở thành bốn.
>
> Một Speaker Split module được dùng chung năm lần: bốn lần cho các skip của encoder và một lần cho bottleneck. Không có năm bộ trọng số Speaker Split độc lập.”

```text
[B,128,T]
→ Conv1d: [B,1024,T]
→ GLU:    [B,512,T]
→ Conv1d: [B,256,T]
→ reshape + GroupNorm: [BJ,128,T]
```

> “Các nhánh này là biểu diễn học được để ước lượng hai nguồn. Speaker Split không cần đầu vào là nhãn danh tính hoặc một đoạn giọng tham chiếu. Tên đầu ra s1, s2 cũng không mặc định là một người cụ thể trong toàn bộ dataset.”

### 7. Reconstruction Decoder: tăng độ phân giải và kết hợp skip

**Chức năng và mục đích:** Decoder đưa các nhánh đặc trưng từ 500 về 8000 vị trí, đồng thời kết hợp thông tin đường bottleneck với các skip ở độ phân giải tương ứng. Đây là quá trình tái cấu trúc **đặc trưng**, chưa phải khôi phục trực tiếp mẫu waveform.

Nearest upsampling sao chép/chọn giá trị theo ánh xạ vị trí để khớp chiều dài; bản thân nó không có tham số và không khôi phục được chi tiết đã mất. Concat đặt đặc trưng đang giải mã cạnh đặc trưng skip theo channel, giữ chúng như hai nhóm thông tin riêng ở đầu vào fusion. Conv1d 256→128 học cách phối hợp các nhóm đó. Nếu cộng trực tiếp thay cho concat, các cặp kênh sẽ bị gộp ngay bằng hệ số 1; code hiện tại chọn concat rồi chiếu để cho phép phép phối hợp được học.

Ba bộ Global–Local–speaker attention sau fusion tiếp tục xử lý quan hệ temporal và quan hệ giữa các nhánh nguồn. Decoder không phải phép toán nghịch đảo chính xác của encoder: convolution, activation, pooling và việc giảm kích thước không mặc nhiên có phép đảo duy nhất. Output của D3 là biểu diễn để tạo waveform ước lượng, không phải bằng chứng nguồn đã được tách hoàn hảo.

**Chỉ hình:** từ Split bottleneck đi sang D0 ở dưới, rồi đi lên D1, D2, D3.

> “Decoder bắt đầu từ [4,128,500]. Trong bố cục này, mũi tên decoder đi từ dưới lên. Thứ tự thực thi là D0, D1, D2 rồi D3.
>
> Tại D0, tensor được nội suy nearest từ 500 lên 1000 frame để khớp với skip E3 đã qua Speaker Split. Hai tensor đều có kích thước [4,128,1000]. Concat theo channel tạo [4,256,1000]. Sau đó, Conv1d kernel 1 giảm 256 kênh về 128.
>
> Đặc trưng đã fusion đi qua ba bộ Global Block, Local Block và attention theo speaker nối tiếp. Các bộ này có tham số riêng. Như vậy, D0 trả lại [4,128,1000]. Các tầng tiếp theo thực hiện cùng quy trình ở những độ phân giải cao hơn.”

| Tầng decoder | T đầu vào | T sau upsample | Skip sử dụng | Đầu ra |
|---|---:|---:|---|---|
| D0 | 500 | 1000 | E3 sau split | `[4,128,1000]` |
| D1 | 1000 | 2000 | E2 sau split | `[4,128,2000]` |
| D2 | 2000 | 4000 | E1 sau split | `[4,128,4000]` |
| D3 | 4000 | 8000 | E0 sau split | `[4,128,8000]` |

> “Code nội suy trực tiếp đến chiều dài của skip tương ứng. Trong ví dụ đã pad theo bội 16, điều này đúng bằng tăng hai lần ở mỗi tầng. Nearest interpolation không có trọng số học; việc phối hợp đặc trưng được học ở fusion convolution và các block phía sau.”

### 8. Attention theo speaker

**Chức năng và mục đích:** Sau khi chia thành hai nhánh, xử lý temporal độc lập chưa trực tiếp cho nhánh này sử dụng đặc trưng của nhánh kia. SpkAttention tạo đường trao đổi đó: tại mỗi thời điểm của một hỗn hợp, hai vector nguồn tham gia attention với nhau. Trong mỗi head, ma trận quan hệ theo speaker có kích thước 2×2, gồm cả quan hệ với chính nhánh và với nhánh còn lại.

Mục đích là cho phép cập nhật một biểu diễn nguồn có xét đến biểu diễn nguồn còn lại. Tuy nhiên, softmax attention không ép hai waveform đầu ra phải trực giao, không bảo đảm không lẫn tiếng và không tự áp đặt tổng hai nguồn bằng hỗn hợp. Đây cũng không phải speaker recognition: không có đầu vào danh tính hoặc giọng tham chiếu trong khối này.

**Đối chiếu bài báo:** §3.2, tr.4 giải thích CS giúp các nhánh tham chiếu nhau để hỗ trợ phục hồi thành phần bị sai lệch khi tái cấu trúc. Phụ lục D và Hình 9, tr.18–19 cho một ví dụ mà similarity giảm sau shared Global/Local nhưng có thể tăng sau CS. Vì vậy không nói mọi block đều phải làm hai biểu diễn xa nhau hơn; sự phân biệt nguồn và trao đổi thông tin là hai vai trò bổ sung.

> “Global attention xử lý quan hệ theo thời gian, còn attention theo speaker xử lý quan hệ giữa hai nhánh nguồn trong cùng một hỗn hợp ở mỗi vị trí thời gian.
>
> Cụ thể, code chuyển [BJ,128,T] thành [B,J,128,T], rồi sắp xếp lại thành [BT,J,128]. Khi đó, chiều chuỗi mà attention nhìn vào là J bằng hai. Nó không attention giữa bốn hỗn hợp độc lập và cũng không trộn hai phần tử batch khác nhau.
>
> Sau attention và cộng residual, tensor được đưa về [BJ,128,T], rồi qua GCFN. Đây là nơi các nhánh nguồn trao đổi thông tin trực tiếp trong decoder. Ngược lại, LTRR ở phía sau dùng chung trọng số cho các nhánh, nhưng không thực hiện attention hoặc ghép thông tin giữa hai speaker.”

### 9. Vị trí của LTRR trong toàn mạng

**Chức năng và mục đích:** Tại đây, decoder đã đưa đặc trưng về độ phân giải Tp và tổ chức theo từng nguồn. LTRR bổ sung một phép hiệu chỉnh đặc trưng ngay trước head tạo waveform. Việc đặt ở cuối giúp nó xử lý một biểu diễn đã qua toàn bộ chuỗi encoder–decoder, đồng thời chỉ bổ sung một module trong đường chính.

“Refinement” nghĩa là mô hình học một phần cập nhật cho đặc trưng hiện có. Code không biết sẵn frame nào sai, không tính một bản đồ lỗi so với ground truth bên trong forward, và không bảo đảm phần cập nhật luôn loại được tiếng người còn lại. LTRR cũng không lặp xử lý cho đến khi sạch; một lần forward gọi module này một lần.

**Chỉ hình:** mũi tên từ đầu ra D3 sang Y của LTRR.

> “Sau D3, mô hình đã có đặc trưng riêng cho các nguồn ở độ phân giải đầy đủ Tp, với kích thước [4,128,8000]. Em ký hiệu tensor này là Y.
>
> LTRR nhận Y và trả Y phẩy có cùng kích thước. Như vậy, nó có thể được đặt giữa separator và OutputLayer mà không cần đổi giao diện của hai khối này.
>
> Trong code hiện tại chỉ có một LTRR sau đầu ra cuối của decoder. Không có LTRR riêng tại từng D0, D1, D2 hay trên bốn auxiliary head. Chi tiết cách tính phần hiệu chỉnh được trình bày ở trang thứ hai.”

### 10. OutputLayer chính và AudioDecoder

**Chức năng và mục đích — OutputLayer:** Separator và LTRR làm việc ở 128 kênh, còn AudioDecoder nhận 256 kênh. OutputLayer học phép chuyển giữa hai không gian này. Crop bỏ phần frame padding ở cuối, Linear–GLU–Linear biến đổi các kênh tại từng vị trí, rồi sắp xếp tensor theo speaker và batch. GLU cung cấp sự điều tiết phi tuyến, không phải phép tách người nói theo một ngưỡng cố định.

**Chức năng và mục đích — AudioDecoder:** ConvTranspose1d học cách tổng hợp các đặc trưng 256 kênh thành mẫu waveform. Có thể hình dung mỗi frame tạo đóng góp trên một đoạn đầu ra dài 32 mẫu; các đoạn đặt cách nhau 8 mẫu được cộng ở vùng chồng lấn. Trọng số bộ giải mã được học; code không buộc chúng là nghịch đảo hay bản sao trọng số AudioEncoder.

Head chính tạo trực tiếp đặc trưng dùng để tổng hợp, vì `masking=False`. Không được mô tả bước này là “tạo mask rồi nhân với hỗn hợp” hoặc “biến đổi ngược STFT”. Sau tổng hợp, crop theo N bảo đảm độ dài waveform khớp đầu vào. Hai waveform là hai **ước lượng nguồn**; forward không có một bước riêng ép tổng của chúng đúng bằng waveform hỗn hợp.

**Chỉ hình:** từ đầu ra LTRR theo đường nối bên phải đi lên Main OutputLayer.

> “Sau refinement, OutputLayer cắt chiều thời gian từ Tp bằng 8000 về T0 bằng 7997, tương ứng bỏ phần padding đặc trưng ở cuối. Tensor [4,128,7997] được chuyển trục thành [4,7997,128] để áp dụng các lớp Linear trên chiều đặc trưng.
>
> Linear thứ nhất tăng 128 lên 512. GLU giảm 512 còn 256; nó không giảm chiều thời gian. Linear cuối cùng ánh xạ 256 sang 256. Code đổi trục và tách lại batch với speaker, tạo tensor theo thứ tự [J,B,256,T0], tức [2,2,256,7997].
>
> Ở head chính, masking bằng false. Vì vậy, đầu ra không được nhân với encoder features như một mask. encoder_output được truyền vào head này để lấy chiều dài T0 cho thao tác crop.”

> “Mỗi nguồn sau đó được đưa qua cùng một AudioDecoder. Đây là ConvTranspose1d từ 256 kênh về một kênh, kernel 32, stride 8, không padding và không bias. Độ dài waveform được khôi phục theo công thức (7997 trừ 1) nhân 8 cộng 32, bằng 64.000.
>
> Mô hình cắt waveform về đúng độ dài ban đầu N. Trong ví dụ batch hai mẫu, mỗi nguồn có kích thước [2,64000]. Phần đầu ra chính là danh sách audio gồm hai tensor, tương ứng hai waveform ước lượng. Giá trị trả về đầy đủ của Model.forward là cặp (audio, audio_aux); audio_aux là danh sách đầu ra phụ, hoặc danh sách rỗng khi không yêu cầu nhánh phụ.”

```text
[4,128,8000]
→ crop: [4,128,7997]
→ transpose: [4,7997,128]
→ Linear: [4,7997,512]
→ GLU: [4,7997,256]
→ Linear: [4,7997,256]
→ reshape/transpose: [J,B,256,7997] = [2,2,256,7997]
→ mỗi nguồn [2,256,7997]
→ ConvTranspose1d: [2,1,64000]
→ squeeze/crop: mỗi nguồn [2,64000]
```

### 11. Bốn auxiliary head

**Chức năng và mục đích:** Các head phụ tạo waveform từ biểu diễn trung gian để loss có thể đánh giá thêm những biểu diễn này trong lúc train. Nhờ các đường loss phụ, gradient đi vào phần mạng tạo ra đặc trưng trung gian mà không phải chỉ thông qua head chính ở cuối. Đây là mục đích của giám sát trung gian; việc cải thiện hội tụ cụ thể vẫn cần kết quả thực nghiệm.

Nearest interpolation chỉ đưa các biểu diễn trung gian về độ dài phù hợp để tính head phụ. Head phụ học hệ số qua projection và ReLU, rồi nhân encoder features; vì ReLU không giới hạn trên, đây không phải mask xác suất trong [0,1]. Các head này có decoder riêng, không lấy bốn waveform rồi trung bình để tạo kết quả chính. Chúng cũng không giám sát trực tiếp đầu ra LTRR vì đường đi của chúng rẽ trước LTRR. Khi không yêu cầu auxiliary output, mô hình trả đầu ra chính và danh sách auxiliary rỗng.

**Chỉ hình:** vùng Auxiliary heads phía dưới trang 1.

> “Khi return_aux bằng true, mô hình còn tính bốn đầu ra phụ. Code lấy đặc trưng trước mỗi decoder stage, nên chiều dài tương ứng là 500, 1000, 2000 và 4000; không phải bốn output sau decoder có chiều dài 1000, 2000, 4000 và 8000.
>
> Mỗi đặc trưng được nội suy nearest về T0 bằng 7997, rồi đi qua OutputLayer phụ của riêng nó. Khác với head chính, các head phụ bật masking: đặc trưng sau projection được qua ReLU và nhân từng phần tử với encoder features được lặp theo nguồn.
>
> Sau đó, AudioDecoder riêng của từng head khôi phục waveform. Như vậy có bốn bộ OutputLayer và AudioDecoder phụ với tham số riêng. Những đầu ra này phục vụ loss phụ trong huấn luyện, nhằm cung cấp tín hiệu giám sát cho các biểu diễn trung gian.
>
> Các nhánh phụ không đi qua LTRR. Nếu return_aux bằng false, code bỏ qua việc tính các waveform phụ. Vì vậy, sơ đồ tách riêng nhánh auxiliary khỏi đường tạo đầu ra chính.”

## Trang 2 — Chi tiết LTRR và Conv-U

### 12. Toàn bộ công thức của LTRR

**Chức năng và mục đích:** LTRR học phép cập nhật dạng `Y′ = Y + α × f(Y)`. Đường Y đưa trực tiếp đặc trưng separator đến đầu ra; nhánh f(Y) học phần bổ sung có cùng shape. Với cách tổ chức này, nhánh mới không phải tự tái tạo toàn bộ Y chỉ qua bottleneck 64 kênh. Nếu α bằng 0 thì đầu ra bằng Y, nhưng khởi tạo thực tế là 0,05 nên không có bảo đảm đồng nhất ngay từ đầu.

Trong f(Y), bottleneck giảm chiều kênh; hai Conv-U học hai phép biến đổi của cùng z; dense memory mở rộng tương tác temporal trên nhánh v; tích u×m kết hợp hai biểu diễn; residual z giữ một đường trực tiếp trong không gian 64 kênh; output projection đưa kết quả về 128 để cộng Y. Mọi bước giữ nguyên thứ tự nguồn và chiều thời gian. Không có phép so sánh với waveform đích hay một bộ nhận diện speaker trong khối.

Do cả u và m phụ thuộc vào đầu vào, phép nhân tạo sự điều biến phụ thuộc nội dung của hai nhánh. Đây là lý do cấu trúc có khả năng biểu diễn khác với chỉ thêm một Linear ở cuối separator. Điều đó mô tả năng lực cấu trúc, chưa chứng minh nó tốt hơn mọi lựa chọn khác.

**Chỉ hình:** cột LTRR.forward(Y), theo chiều từ trên xuống.

> “LTRR không trực tiếp xử lý waveform. Nó xử lý tensor Y ở miền đặc trưng, sau khi separator đã tạo các nhánh nguồn. Với ví dụ đang xét, Y có kích thước [4,128,8000].
>
> Thiết kế gồm một đường giữ nguyên Y và một đường học phần hiệu chỉnh. Đường hiệu chỉnh giảm số kênh xuống 64, xử lý tương tác giữa hai nhánh temporal, sau đó chiếu trở lại 128 kênh. Phần hiệu chỉnh được nhân với hệ số học được alpha rồi cộng vào Y.”

```text
z  = PReLU(PWConv_128→64(ChannelLN_128(Y)))
u  = ConvU_u(z)
v  = ConvU_v(z)
m  = DenseDilatedMemory(v)
h  = z + Dropout(u ⊙ m)
ΔY = PWConv_64→128(ChannelLN_64(h))
Y′ = Y + αΔY
```

`PWConv` là Conv1d kernel 1; `⊙` là nhân từng phần tử. Trong hình, phép nhân này được biểu diễn bằng vòng tròn `×`.

### 13. Bottleneck projection: 128 xuống 64 kênh

**Chức năng và mục đích:** ChannelLayerNorm tính trung bình/phương sai trên 128 kênh tại **từng frame của từng nhánh nguồn**, chuẩn hóa rồi áp dụng affine học được. Nó không lấy thống kê dọc toàn bộ thời gian như GroupNorm một nhóm ở FeatureProjector, không trộn các speaker, và không thay đổi số phần tử.

Pointwise convolution 128→64 học 64 tổ hợp tuyến tính của 128 kênh tại cùng frame. Mục đích là dùng một không gian làm việc hẹp hơn cho phần lớn phép biến đổi bổ sung. Đây có thể là nút giới hạn khả năng biểu diễn của nhánh refinement; đường residual Y ngoài cùng giúp đầu ra vẫn nhận trực tiếp biểu diễn 128 kênh ban đầu, nhưng không làm phép chiếu 128→64 trở thành không mất thông tin.

PReLU sau projection giữ nguyên giá trị dương và nhân giá trị âm với một hệ số học được. Khác với ReLU luôn đặt phần âm thành 0, nó cho phép nhánh âm tham gia biểu diễn. Trong code hiện tại, mỗi lời gọi `PReLU()` tạo một hệ số độ dốc dùng chung trong module đó; không phải mỗi phần tử tensor có một hệ số riêng. Độ rộng 64, loại activation và vị trí của chúng là thiết kế được sử dụng, chưa phải kết luận tối ưu từ ablation.

> “Đầu tiên, ChannelLayerNorm chuẩn hóa 128 kênh tại từng vị trí thời gian. Code thực hiện điều này bằng cách đổi từ [BJ,C,T] sang [BJ,T,C], áp dụng LayerNorm trên C, rồi đổi lại. Epsilon bằng 10 mũ trừ 8.
>
> Conv1d kernel 1 giảm 128 xuống 64 kênh, sau đó PReLU tạo phi tuyến. Tensor z có kích thước [4,64,8000]. Chiều thời gian không thay đổi ở bước này.
>
> Mục đích của bottleneck là thực hiện phần lớn phép biến đổi bổ sung trong không gian 64 kênh thay vì 128 kênh. Tuy nhiên, 64 là lựa chọn của thiết kế hiện tại, chưa thể gọi là lựa chọn tối ưu nếu chưa có thí nghiệm so sánh các độ rộng khác.”

### 14. Hai nhánh Conv-U

**Chức năng và mục đích:** Hai nhánh cho phép mô hình học hai biểu diễn khác nhau từ cùng z. Nhánh u cung cấp tensor đi trực tiếp vào phép nhân; nhánh v cung cấp tensor để xây dựng memory m trước khi nhân. Có thể dùng cách nói ngắn “nhánh tương tác trực tiếp” và “nhánh qua ngữ cảnh memory” để nhớ luồng, nhưng không gán sẵn ý nghĩa “u là nội dung, v là giọng” như một sự thật được code bảo đảm.

Hai Conv-U có tham số độc lập nên không bị buộc cho cùng output, dù cùng cấu trúc và cùng input. Tuy nhiên, kiến trúc cũng không có loss riêng ép hai nhánh phải học hai thông tin hoàn toàn khác nhau. Cả u và v đều tồn tại **trong từng nhánh nguồn**; u không phải speaker 1 và v không phải speaker 2.

**Chỉ hình:** từ z rẽ sang Conv-U branch u và Conv-U branch v; sau đó sang cột ConvU.forward(a).

> “Từ cùng tensor z, LTRR tạo hai nhánh u và v bằng hai Conv-U. Hai nhánh có cùng cấu trúc nhưng trọng số độc lập. Tên Conv-U ở đây là tên một loại block; cả nhánh mang tên u lẫn nhánh mang tên v đều dùng cấu trúc đó.
>
> Nhánh u đi đến phép nhân. Nhánh v đi qua dense dilated memory trước, tạo tensor m. Vì các phép toán đều giữ nguyên kích thước [4,64,8000], u và m có thể được nhân trực tiếp theo từng phần tử.”

### 15. Bên trong một Conv-U

**Chức năng và mục đích:** Conv-U kết hợp việc trộn kênh với lọc theo thời gian và một đường residual. Tên khối không có nghĩa nó là một U-Net thu nhỏ: trong Conv-U hiện tại không có downsample, upsample hoặc các tầng encoder–decoder bên trong.

| Phép toán trong nhánh biến đổi | Thực hiện điều gì? | Mục đích và giới hạn |
|---|---|---|
| ChannelLayerNorm | Chuẩn hóa vector 64 kênh của từng frame, có affine học được | Điều chỉnh thang giá trị trước biến đổi; không chuẩn hóa toàn bộ waveform hay tương tác giữa speaker |
| Pointwise Conv1d 64→64, k=1 | Mỗi kênh đầu ra là tổng có trọng số của các kênh cùng frame, cộng bias | Trộn thông tin channel; chưa thêm quan hệ temporal |
| SiLU | Tính `a × sigmoid(a)` trên từng giá trị sau pointwise | Bổ sung phi tuyến trơn; không bảo đảm đầu ra dương hoặc trong [0,1] |
| Depthwise Conv1d, k=3, groups=64 | Mỗi kênh có bộ lọc riêng đọc vị trí t−1, t, t+1 | Bổ sung xử lý temporal gần; không tự trộn các kênh |
| Dropout p=0,05 | Khi train, ngẫu nhiên đặt các phần tử thành 0 và scale phần được giữ; eval thì là phép đồng nhất | Regularization; không xóa frame khỏi tensor, không thay đổi shape |
| Cộng đầu vào a | `output = a + nhánh_biến_đổi(a)` | Giữ đường trực tiếp cho đặc trưng và gradient; không bảo đảm đầu ra luôn tốt hơn a |

Pointwise đứng trước depthwise có nghĩa mỗi bộ lọc temporal xử lý một kênh đã được phối hợp từ các kênh ban đầu. Do đó, toàn Conv-U có cả trộn channel lẫn xử lý time, dù riêng depthwise không trộn channel. Padding 1 giữ chiều dài nhưng dùng thêm 0 ở biên; ở giữa chuỗi, khối nhìn cả vị trí trước và sau nên không causal.

Ví dụ về số tham số, chỉ xét riêng convolution k=3 có bias: depthwise 64 kênh có `64×3 + 64 = 256` tham số, trong khi convolution thường 64→64 có `64×64×3 + 64 = 12.352`. Đây là căn cứ cho việc dùng phép lọc temporal ít tham số. Nó không phải tổng tham số Conv-U và không chứng minh tốc độ thực tế tăng theo đúng tỷ lệ này.

**Chỉ hình:** cột phải trang 2, từ a đi xuống; chỉ đường bypass bên trái khi đến dấu cộng.

> “Xét một Conv-U với đầu vào a có kích thước [4,64,8000]. Đầu vào được giữ lại trên một đường residual.
>
> Đường biến đổi bắt đầu bằng ChannelLayerNorm theo 64 kênh tại từng frame. Sau đó, pointwise convolution kernel 1 ánh xạ 64 sang 64 kênh. Đây là bước phối hợp thông tin giữa các kênh.
>
> Tiếp theo là SiLU, rồi depthwise convolution kernel 3, stride 1, padding 1 và groups bằng 64. Mỗi kênh có bộ lọc thời gian riêng. Depthwise convolution không tự trộn các kênh; việc trộn kênh đã được thực hiện ở pointwise convolution trước đó.
>
> Padding 1 giúp giữ nguyên chiều dài 8000. Sau Dropout với xác suất 0,05, kết quả được cộng với a ban đầu. Vì vậy, đầu ra Conv-U vẫn là [4,64,8000].”

```text
ConvU(a) = a + Dropout(DWConv3(SiLU(PWConv1(ChannelLN(a)))))
```

> “Residual ở đây nằm bên trong từng Conv-U. Nó khác với residual của z trong TemporalInteraction và khác với residual ngoài cùng của Y trong LTRR. Cả ba loại đường cộng này đều tồn tại trong code.”

Trong Conv-U: pointwise convolution và depthwise convolution đều `bias=True`. SiLU có công thức `a × sigmoid(a)`, nhưng nhánh không có một sigmoid riêng để ép tensor memory thành mask trong khoảng 0 đến 1. Dropout tác động khi train và được tắt trong eval; nó không giảm shape tensor.

### 16. Nhân u với memory m, rồi cộng lại z

**Chức năng và mục đích:** Với cùng chỉ số nhánh nguồn, kênh và thời gian, code tính `r[b,c,t] = u[b,c,t] × m[b,c,t]`. Nhờ vậy, giá trị biểu diễn của u được điều chỉnh bởi một tensor đã tổng hợp ngữ cảnh temporal từ v. Nếu dùng phép cộng u+m, hai nhánh chỉ được gộp bằng cộng; phép nhân cho một kiểu tương tác khác, trong đó độ lớn và dấu của một nhánh tác động lên đóng góp của nhánh kia.

Ví dụ minh họa số học, **không phải số đo từ mô hình**: nếu tại một phần tử u=2 thì m=0,1 cho tích 0,2; m=2 cho tích 4; m=−1 cho tích −2. Điều này cho thấy memory có thể làm giảm, tăng hoặc đảo dấu đóng góp của u. Vì m không qua sigmoid ở đầu ra memory, không thể diễn giải nó là “xác suất đặc trưng thuộc speaker này”.

Dropout regularize tích trước khi cộng lại z. Đường z cho phép tensor h vẫn nhận biểu diễn bottleneck trực tiếp, ngay cả khi nhánh tích tại một phần tử bằng 0. Phép nhân cũng làm gradient của một nhánh phụ thuộc vào giá trị nhánh kia: memory gần 0 có thể làm yếu tín hiệu gradient qua u; giá trị lớn có thể làm tăng biên độ. Vì vậy không nên chỉ trình bày phép nhân như một cơ chế chắc chắn ổn định hoặc chắc chắn tăng chất lượng.

**Chỉ hình:** vòng tròn ×, Dropout và dấu + ở giữa LTRR.

> “Sau khi nhánh v được xử lý bởi dense dilated memory, tensor m có cùng kích thước với u. LTRR nhân u và m theo từng batch, channel và frame. Đây là nhân từng phần tử, không phải nhân ma trận và cũng không phải concat.
>
> Có thể hiểu cấu trúc này cho phép biểu diễn ở nhánh u tương tác với thông tin temporal của nhánh memory. Tuy nhiên, m không bị ràng buộc trong khoảng 0 đến 1; nó có thể chứa giá trị âm hoặc lớn hơn một. Vì vậy, không nên mô tả m như một xác suất giữ hoặc bỏ đặc trưng.
>
> Tích này được đưa qua Dropout 0,05 rồi cộng với z, tạo h. Việc cộng hợp lệ vì cả hai nhánh đều có kích thước [4,64,8000]. Dense memory không tự cộng v ở đầu ra của nó; phép cộng đang chỉ trên hình là cộng với z ở cấp TemporalInteraction.”

### 17. Chiếu về 128 kênh và residual ngoài

**Chức năng và mục đích:** ChannelLayerNorm chuẩn hóa h trong không gian 64 kênh; pointwise 64→128 chuyển phần cập nhật sang đúng không gian của Y. Nhờ khớp channel và time, phép cộng residual ngoài có nghĩa từng phần tử. Projection không đơn thuần lặp đôi 64 kênh: mỗi kênh trong 128 đầu ra có bộ trọng số học được riêng trên 64 kênh đầu vào.

Scalar α điều chỉnh mức đóng góp chung của nhánh cập nhật, và optimizer có thể thay đổi nó theo loss. Vì α được dùng trực tiếp chứ không qua sigmoid hay clamp, nó có thể âm, lớn hơn 1 hoặc gần 0; nó không chọn riêng frame hay speaker. α nhỏ chỉ scale nhánh bổ sung, không phải bảo đảm chuẩn của `αΔY` nhỏ hơn chuẩn của Y.

Output projection không có activation sau Conv1d nên không ép ΔY không âm. Phần cập nhật có thể cộng hoặc trừ các giá trị đặc trưng tùy dấu học được. Residual Y giữ biểu diễn gốc như một thành phần của kết quả, nhưng sau cộng thì từng giá trị vẫn có thể thay đổi; không được nói “mọi thông tin gốc được bảo toàn nguyên vẹn trong Y′”.

**Chỉ hình:** Output projection, Scale by α, dấu cộng Y, rồi Y′.

> “Tensor h tiếp tục qua ChannelLayerNorm trên 64 kênh, rồi Conv1d kernel 1 từ 64 lên 128. Khối projection này không có PReLU sau convolution trong code hiện tại. Đầu ra được ký hiệu là delta Y, có kích thước [4,128,8000].
>
> Delta Y được nhân với alpha rồi cộng với Y ban đầu. Alpha là một scalar học được, khởi tạo bằng 0,05; nó không phải vector theo kênh, không phải một hệ số riêng cho từng speaker và không bị cố định trong suốt quá trình train.
>
> Khởi tạo alpha nhỏ làm giảm hệ số của nhánh bổ sung lúc bắt đầu. Tuy vậy, điều đó không chứng minh phần hiệu chỉnh luôn nhỏ so với Y, vì còn phụ thuộc độ lớn delta Y. Alpha khác không cũng có nghĩa đầu ra ban đầu của mô hình LTRR không bắt buộc giống hệt baseline, dù backbone được khởi tạo giống nhau.
>
> Kết quả Y phẩy vẫn có kích thước [4,128,8000], sẵn sàng đưa vào OutputLayer chính.”

### 18. LTRR có tương tác giữa các speaker không?

**Chức năng và mục đích:** Chia sẻ một bộ trọng số LTRR cho BJ nhánh giúp cùng phép refinement được áp dụng cho mọi nguồn, không cần thiết kế một module riêng cho s1 và một module riêng cho s2. Vì các phép toán không giảm hay trộn chiều BJ, giá trị của nhánh này không trực tiếp được đọc bởi nhánh kia trong LTRR. Khi train, gradient từ các nhánh cùng cập nhật bộ trọng số chung; đó là chia sẻ việc học, khác với trao đổi đặc trưng ngay trong forward.

Việc bỏ speaker-guided gate làm nhánh refinement không phụ thuộc một cơ chế điều khiển riêng từ các speaker khác. Đổi lại, LTRR không tự có khả năng dùng trực tiếp đặc trưng của nguồn còn lại để quyết định hiệu chỉnh. Quan hệ giữa các nguồn vẫn được xử lý ở SpkAttention của backbone; cần thực nghiệm mới biết việc thêm hay bỏ gate có lợi hơn trong protocol này.

> “Một LTRR xử lý tensor gộp BJ bằng bốn, với cùng bộ trọng số cho mọi nhánh. Các phép ChannelLayerNorm, convolution và phép nhân ở đây không gộp đặc trưng của hai speaker để tạo quyết định chung. Do đó, LTRR chia sẻ tham số giữa các nhánh nguồn nhưng không trực tiếp trao đổi thông tin giữa chúng.
>
> Việc trao đổi giữa hai nhánh nguồn đã có ở attention theo speaker trong decoder của SepReformer. Đây là điểm cần phân biệt khi giải thích tên NoGate: thiết kế LTRR hiện tại bỏ speaker-guided gate của khối cũ, nhưng vẫn giữ tích u nhân m và giữ các cơ chế GLU hoặc gating thuộc backbone.”

## Trang 3 — Dense dilated memory

### 19. Chuẩn bị x0 từ v

**Chức năng và mục đích:** Khối ffn_in chuẩn bị biểu diễn đầu vào chung cho cả chuỗi memory. ChannelLayerNorm điều chỉnh thang giá trị theo channel; pointwise phối hợp các kênh v; PReLU tạo phi tuyến. Vì kernel pointwise bằng 1 và normalization không chạy dọc time, khối chuẩn bị này không tự mở rộng phạm vi temporal so với v.

x0 là một biểu diễn **đã biến đổi từ v**, không phải bản sao nguyên trạng của v. Nó được cấp lại trực tiếp cho mọi tầng memory để tầng sau có thể dùng biểu diễn chuẩn bị ban đầu bên cạnh các state đã qua convolution. Giữ x0 trong danh sách đầu vào giúp không bắt buộc mọi thông tin chỉ truyền qua toàn bộ chuỗi y1→y2→y3→y4.

**Chỉ hình:** hộp Input v → ffn_in → x0 ở trên cùng.

> “Đầu vào dense memory là v có kích thước [4,64,8000]. Khối ffn_in áp dụng ChannelLayerNorm, pointwise convolution 64 sang 64 và PReLU. Kết quả là x0, vẫn có kích thước [4,64,8000].
>
> Tensor x0 được giữ để cung cấp cho cả bốn tầng memory. Trong hình, đó là đường màu xanh chạy ngang ở phía trên. Những đường nét đứt còn lại đưa các state đã tính đến các tầng phía sau.”

### 20. Cách đọc dense connection

**Chức năng và mục đích:** Dense connection giữ các biểu diễn trung gian như những nhóm kênh riêng và đưa chúng đến các tầng phía sau. Khi một tầng nhận `[x0,y1,y2]`, nó được phép học cách phối hợp cả biểu diễn ban đầu lẫn kết quả hai mức xử lý trước đó. Đây là cơ chế tái sử dụng đặc trưng và tạo nhiều đường truyền trong đồ thị tính toán.

Concat khác cộng residual: concat tăng số kênh và vẫn giữ riêng các giá trị; cộng gộp các tensor cùng shape ngay tại từng phần tử. Sau concat, pointwise projection mới học cách phối hợp và nén số kênh về 64. Việc nén lại giúp các depthwise layer đều làm việc trên 64 kênh, nhưng các phép chiếu ở tầng sau vẫn có đầu vào rộng hơn và các state cần được lưu cho việc tái sử dụng. Vì vậy “dense” không đồng nghĩa với không tốn thêm bộ nhớ.

Các đường nối chỉ đi từ state đã tính sang tầng về sau. Code không đưa y4 quay lại để tính lại y1, không lặp trên thời điểm như RNN, và không có một trạng thái từ utterance trước được truyền vào utterance sau.

> “Dense ở đây không có nghĩa là mạng Fully Connected thông thường. Nó nói đến cách nối: tầng sau nhận x0 cùng tất cả các state trước nó bằng phép concat theo channel.
>
> Bốn tầng được thực hiện tuần tự từ trái sang phải. Tầng thứ hai cần y1; tầng thứ ba cần y1 và y2; tầng thứ tư cần y1, y2 và y3. Các đường vòng lên trên là đường truyền state đến tầng sau, không phải vòng lặp recurrent theo thời gian.”

Với `yi` đánh số từ 1 và chỉ số code bắt đầu từ 0:

```text
ci = Concat_channels(x0, y1, ..., y(i−1))
pi = PWConv_i(ci)
yi = Dropout(PReLU(DWConv_dilation_i(pi)))
```

### 21. Tầng y1: dilation 1

**Chức năng và mục đích:** y1 tạo state đầu tiên bằng cách lọc các vị trí temporal sát nhau của biểu diễn x0 đã được trộn kênh. Với dilation 1, mỗi bộ lọc depthwise đọc t−1, t và t+1. Pointwise chuẩn bị tổ hợp channel cho phép lọc; PReLU biến đổi phi tuyến đầu ra; Dropout regularize state khi train.

State này được cấp trực tiếp cho các tầng sau và cho bộ tổng hợp cuối, nên biểu diễn ở bước xử lý sớm vẫn có đường đến đầu ra memory. Không có quy định y1 chỉ học “âm thanh ngắn” hay tự phát hiện phụ âm; đây là state có phạm vi convolution gần nhất trong chuỗi đang xét.

**Chỉ hình:** cột đầu tiên.

> “Tầng đầu chưa có state trước đó, nên concat chỉ gồm x0. Tensor có 64 kênh. Pointwise convolution 64 sang 64 trộn thông tin kênh.
>
> Sau đó, depthwise convolution dùng kernel 3, dilation 1, padding 1 và stride 1. Tiếp theo là PReLU và Dropout 0,05, tạo y1 có kích thước [4,64,8000].
>
> y1 được chuyển đến các tầng y2, y3, y4 và đồng thời được giữ lại để tổng hợp đầu ra cuối.”

### 22. Tầng y2: dilation 2

**Chức năng và mục đích:** y2 kết hợp x0 với y1 bằng projection 128→64 trước khi lọc temporal. Nhờ vậy, tầng có thể sử dụng thông tin chưa đi qua memory layer và thông tin đã được y1 xử lý. Dilation 2 mở khoảng cách lấy mẫu của riêng kernel thành t−2, t, t+2 mà không tăng số tap từ 3.

Do y1 ở mỗi vị trí đã phụ thuộc lân cận của x0, phạm vi tích lũy của y2 không chỉ là ba điểm cách nhau 2 trên x0. Tầng vẫn giữ 8000 vị trí; dilation không phải lấy mẫu đầu ra thưa đi hay bỏ một nửa frame.

**Chỉ hình:** cột thứ hai, hai mũi tên đi vào concat.

> “Tầng thứ hai ghép x0 và y1 theo channel. Mỗi tensor có 64 kênh nên đầu vào concat là [4,128,8000]. Pointwise convolution đưa 128 kênh trở về 64.
>
> Depthwise convolution tiếp theo dùng kernel 3, dilation 2 và padding 2, rồi PReLU và Dropout. Kết quả y2 vẫn là [4,64,8000]. Dilation thay đổi khoảng cách lấy thông tin theo thời gian, không làm giảm chiều dài đầu ra.”

### 23. Tầng y3: dilation 4

**Chức năng và mục đích:** y3 có thể phối hợp ba biểu diễn x0, y1 và y2 bằng projection 192→64. Nó không bị buộc chỉ dùng output gần nhất là y2. Sau projection, depthwise dilation 4 đọc các vị trí cách 4 frame trên biểu diễn đã phối hợp, bổ sung đường xử lý temporal rộng hơn ở bước này.

Mục đích là để một tầng dùng được cả các đường xử lý ngắn lẫn các đường qua nhiều tầng trước. Sự đa dạng đó đến từ cấu trúc kết nối; chưa thể kết luận các tầng thực tế sẽ học những loại tín hiệu bổ sung cho nhau nếu chưa phân tích hoặc làm ablation.

**Chỉ hình:** cột thứ ba, ba nguồn vào concat.

> “Ở tầng thứ ba, x0, y1 và y2 được ghép thành 192 kênh. Pointwise convolution ánh xạ 192 xuống 64 kênh. Depthwise convolution dùng kernel 3, dilation 4 và padding 4, tiếp theo là PReLU và Dropout để tạo y3.
>
> Nhờ concat các state trước, tầng này có thể học cách phối hợp biểu diễn từ nhiều bước xử lý, thay vì chỉ nhận state ngay trước nó.”

### 24. Tầng y4: dilation 8

**Chức năng và mục đích:** y4 nhận đầy đủ x0 cùng ba state trước, nên projection 256→64 có thể học cách kết hợp mọi mức xử lý đã có. Depthwise dilation 8 là kernel có khoảng cách tap lớn nhất trong chuỗi hiện tại. Nó tiếp tục giữ nguyên channel/time sau projection và không thay thế các state cũ trong phần tổng hợp cuối.

Không chỉ lấy riêng y4 làm memory: y1, y2, y3 đều được giữ và ghép cùng y4 ở cuối. Vì vậy, bộ tổng hợp có quyền sử dụng trực tiếp biểu diễn ở nhiều độ sâu, thay vì buộc thông tin nào cũng đi qua đủ bốn memory layer.

**Chỉ hình:** cột thứ tư.

> “Tầng cuối ghép x0, y1, y2 và y3 thành 256 kênh. Pointwise convolution giảm về 64 kênh. Sau đó là depthwise convolution kernel 3, dilation 8, padding 8, PReLU và Dropout, tạo y4.
>
> Cả bốn tầng đều trả 64 kênh và giữ nguyên Tp. Số kênh chỉ tăng tạm thời ở đầu vào concat; nó không tích lũy thành một tensor đầu ra ngày càng rộng sau mỗi tầng.”

| State | Tensor được concat | Số kênh vào projection | Pointwise | Dilation / padding | Shape state |
|---|---|---:|---|---|---|
| y1 | x0 | 64 | 64→64 | 1 / 1 | `[4,64,8000]` |
| y2 | x0, y1 | 128 | 128→64 | 2 / 2 | `[4,64,8000]` |
| y3 | x0, y1, y2 | 192 | 192→64 | 4 / 4 | `[4,64,8000]` |
| y4 | x0, y1, y2, y3 | 256 | 256→64 | 8 / 8 | `[4,64,8000]` |

### 25. Dilation có ý nghĩa cụ thể thế nào?

**Chức năng và mục đích:** Dilation mở rộng khoảng cách các vị trí mà kernel đọc, giúp một kernel ít tap tổng hợp thông tin trên phạm vi rộng hơn. Nó không thêm trọng số cho các khoảng trống, không đổi số frame đầu ra khi padding phù hợp và không tự trở thành attention. Danh sách 1/2/4/8 là cố định trong cấu hình; trọng số bộ lọc được học nhưng dilation không được chọn động theo âm thanh.

Nếu chỉ xét đường phụ thuộc temporal **bên trong memory, từ x0**, phạm vi tối đa của y1/y2/y3/y4 lần lượt rộng 3/7/15/31 frame: bán kính tích lũy là 1, 1+2, 1+2+4 và 1+2+4+8. Đây là phạm vi có thể ảnh hưởng qua các đường convolution dài nhất; không nói mọi vị trí có trọng số quan trọng như nhau. out_proj kernel 1 ở cuối không mở rộng thêm phạm vi này.

Ở độ phân giải 0,5 ms giữa hai tâm frame, hai đầu của phạm vi 31 frame cách nhau 15 ms. Con số này chỉ mô tả chỉ số temporal của memory đối với x0. Đầu vào đã qua Conv-U và quan trọng hơn là toàn separator với attention, nên tuyệt đối không suy ra toàn mạng chỉ nhìn 15 ms hoặc LTRR chỉ sử dụng thông tin âm thanh trong 15 ms.

> “Với kernel 3 và padding đối xứng, một convolution dilation d lấy thông tin tại các vị trí t trừ d, t và t cộng d của tensor đầu vào của chính tầng đó. Khi d tăng từ 1 lên 2, 4 rồi 8, khoảng cách giữa các vị trí được lấy tăng lên.
>
> Kernel vẫn chỉ có ba vị trí trọng số theo thời gian cho mỗi kênh. Việc tăng dilation không tự làm số trọng số kernel tăng. Ở cùng kernel và stride, padding bằng dilation giúp bảo toàn chiều dài.
>
> Tuy nhiên, y4 không chỉ nhìn ba frame của v ban đầu: đầu vào của nó đã phối hợp x0 với các state trước. Do đó cần phân biệt phạm vi của một convolution riêng lẻ với phạm vi tích lũy của cả chuỗi memory.”

| Dilation d | Offset của một kernel | Bề rộng hiệu dụng của riêng kernel `1+2d` |
|---:|---|---:|
| 1 | t−1, t, t+1 | 3 frame |
| 2 | t−2, t, t+2 | 5 frame |
| 4 | t−4, t, t+4 | 9 frame |
| 8 | t−8, t, t+8 | 17 frame |

Khoảng cách frame tại đầu vào LTRR tương ứng stride 8 mẫu ở 16 kHz, tức 0,5 ms. Vì vậy, các độ lệch d frame tương ứng 0,5/1/2/4 ms. Đây là khoảng cách chỉ số temporal tại độ phân giải này, **không phải receptive field của toàn mô hình**; đặc trưng Y đã qua global attention. Padding đối xứng cho phép dùng cả vị trí phía trước lẫn phía sau; không mô tả phiên bản hiện tại là mô hình causal/streaming.

### 26. Tổng hợp bốn state để tạo m

**Chức năng và mục đích:** Concat cuối giữ bốn nhóm state y1…y4 cạnh nhau, tạo 256 kênh. out_proj 256→64 học cách kết hợp chúng tại từng frame thành tensor memory có đúng shape để nhân với u. Nó không phải trung bình bốn state, không lấy riêng state cuối và không có softmax buộc trọng số tổng hợp cộng thành 1.

Vì projection nhận các state từ nhiều độ sâu, nó có thể học mức phối hợp khác nhau giữa những đường xử lý đó. Tuy nhiên, các trọng số pointwise là tham số chung dọc thời gian; đây không phải một mạng sinh trọng số fusion riêng cho từng frame. Tensor m vẫn phụ thuộc nội dung đầu vào thông qua các giá trị state.

x0 không có mặt trực tiếp trong concat cuối, nhưng đã tham gia đầu vào mọi tầng. Memory trả m mà không cộng v hoặc x0 ở cuối. Giá trị âm và lớn hơn 1 đều được phép ở m; đây là lý do phép nhân u×m phía sau là điều biến đặc trưng, không phải nhân với một mask xác suất.

**Chỉ hình:** các đường đi xuống Concat[y1,y2,y3,y4], rồi out_proj.

> “Sau bốn tầng, code concat y1, y2, y3 và y4 theo channel. Bốn tensor 64 kênh tạo thành [4,256,8000]. Lưu ý x0 không được đưa trực tiếp vào concat cuối này, dù nó đã tham gia đầu vào của cả bốn tầng.
>
> out_proj là pointwise convolution từ 256 xuống 64 kênh, tạo memory m có kích thước [4,64,8000]. Sau out_proj không có activation hay residual bổ sung trong khối memory hiện tại.
>
> Tensor m quay lại trang thứ hai để nhân từng phần tử với u. Kết quả qua Dropout rồi cộng z, chiếu về 128 kênh, nhân alpha và cộng Y, hoàn tất LTRR.”

### 27. Vì sao gọi là memory?

**Chức năng và mục đích:** Tên memory diễn tả việc dùng thông tin từ các vị trí temporal lân cận và tái sử dụng các state trung gian để tạo biểu diễn tại vị trí đang xét. Không có vùng lưu trữ lâu dài chứa giọng của những người đã gặp, không có truy vấn một kho mẫu âm thanh, và không có hidden state phải giữ giữa hai lần inference.

Trong source, lớp có tên `DenseDilatedFSMN1D` và được mô tả là “FSMN-like”; Conv-U và memory có ghi chú lấy cảm hứng từ MossFormer2. Vì vậy khi trình bày đóng góp, nên nói rõ **cách đưa thiết kế refinement này vào vị trí sau decoder của SepReformer, cùng cách kết hợp các nhánh và đánh giá thực nghiệm**, không suy ra từ tên LTRR rằng mọi phép toán thành phần đều được phát minh mới. Việc kế thừa ý tưởng và việc đánh giá một tích hợp kiến trúc là hai nội dung cần trình bày trung thực, riêng biệt.

> “Memory ở đây là tên cho cơ chế tổng hợp ngữ cảnh thời gian bằng convolution có dilation và các dense connection. Nó không lưu một bộ nhớ lâu dài giữa những lần gọi mô hình, không có hidden state recurrent như LSTM và không phải bộ nhớ truy xuất từ cơ sở dữ liệu.
>
> Các state y1 đến y4 được tính trong một lần forward và dùng để tạo đầu ra của lần forward đó. Đây là một thiết kế FSMN-like trong code, không nên đồng nhất với mọi biến thể FSMN trong tài liệu nghiên cứu.”

## Lời kết về đóng góp và giới hạn

**Chỉ hình:** quay về trang 1, chỉ LTRR và vị trí giữa decoder với OutputLayer.

> “Phần backbone SepReformer giữ vai trò mã hóa hỗn hợp, xử lý quan hệ temporal ở nhiều độ phân giải, tạo các nhánh nguồn và tái cấu trúc đặc trưng. Phần bổ sung LTRR tập trung vào refinement ở độ phân giải cuối trước khi tổng hợp waveform.
>
> Đặc điểm của LTRR là xử lý ở bottleneck 64 kênh, kết hợp hai nhánh Conv-U bằng phép nhân với dense dilated memory và dùng các đường residual để học phần hiệu chỉnh. LTRR dùng chung trọng số cho các nhánh nguồn và không có speaker-guided gate.
>
> Theo số đếm parameter từ code, LTRR bổ sung 89.031 tham số. Tổng mô hình gồm cả auxiliary heads là 14.805.191 tham số. Chênh lệch parameter là một căn cứ để mô tả quy mô phần bổ sung, nhưng chưa chứng minh thời gian chạy, FLOPs hay bộ nhớ tăng ít: block vẫn hoạt động trên chuỗi 8000 frame và cần đo trên tài nguyên thực tế.
>
> Thiết kế có thể bổ sung khả năng biến đổi đặc trưng temporal, nhưng chất lượng tách tiếng tốt hơn baseline là giả thuyết cần kiểm chứng. Em sẽ so sánh baseline và LTRR theo cùng cấu hình chung, cùng dữ liệu, quy trình chọn checkpoint và khởi tạo backbone được kiểm soát. Các con số trong sơ đồ là thông số kiến trúc và kích thước tensor, không phải kết quả chứng minh hiệu quả.”

## Bản nói rút gọn để tập trình bày

> “Đầu vào là hai hỗn hợp mono, mỗi hỗn hợp bốn giây ở 16 kHz, nên tensor có kích thước [2,64000]. AudioEncoder dùng Conv1d một sang 256 kênh, kernel 32, stride 8, tạo 7997 frame. FeatureProjector giảm về 128 kênh, sau đó separator pad lên 8000 frame.
>
> Encoder có bốn tầng, mỗi tầng gồm hai cặp Global–Local rồi DownConv. Chiều dài giảm từ 8000 xuống 4000, 2000, 1000 và 500. Skip lấy trước mỗi DownConv. Sau đó còn một bottleneck gồm hai cặp Global–Local nhưng không giảm mẫu.
>
> Speaker Split dùng chung cho bốn skip và bottleneck, chuyển [B,128,T] thành [BJ,128,T]. Với hai hỗn hợp và hai nguồn, BJ bằng bốn. Decoder bắt đầu từ 500 frame, lần lượt upsample, concat skip cùng độ phân giải, chiếu 256 về 128 kênh và qua ba bộ Global–Local–speaker attention. Cuối D3, tensor có kích thước [4,128,8000].
>
> Một LTRR được đặt sau đầu ra cuối này. Nó chuẩn hóa, giảm 128 xuống 64 kênh, rồi tạo hai nhánh Conv-U độc lập. Mỗi Conv-U gồm ChannelLayerNorm, pointwise convolution, SiLU, depthwise convolution kernel 3, Dropout và residual nội bộ.
>
> Nhánh u đi đến phép nhân; nhánh v đi qua dense dilated memory. Memory tạo x0, sau đó bốn state với dilation 1, 2, 4, 8. Mỗi tầng concat x0 cùng tất cả state trước, dùng pointwise convolution giảm về 64 kênh, rồi depthwise convolution, PReLU và Dropout. Cuối cùng, concat bốn state thành 256 kênh và chiếu về 64 để tạo m.
>
> LTRR nhân u với m, qua Dropout rồi cộng z. Kết quả được chiếu lên 128 kênh, nhân alpha khởi tạo 0,05 rồi cộng với Y ban đầu. Kích thước vẫn là [4,128,8000].
>
> Main OutputLayer crop về 7997 frame, dùng Linear–GLU–Linear để tạo đặc trưng 256 kênh cho từng nguồn. AudioDecoder dùng ConvTranspose1d kernel 32, stride 8 để khôi phục hai waveform, mỗi waveform có shape [2,64000]. Bốn auxiliary head được tính từ các đặc trưng trước decoder stage và không đi qua LTRR.
>
> LTRR thêm 89.031 tham số. Hiệu quả của nó phải được kiểm chứng bằng thực nghiệm cùng protocol với baseline, không được suy ra chỉ từ sơ đồ kiến trúc.”

## Những câu trả lời cần giữ chính xác khi được hỏi

| Câu hỏi | Cách trả lời |
|---|---|
| 8000 frame có phải đầu ra AudioEncoder không? | Không. AudioEncoder cho 7997 frame với N=64000; separator pad thành 8000. |
| Tất cả tầng decoder có LTRR không? | Chỉ có một LTRR sau D3, tức đầu ra decoder cuối cùng. |
| BJ=4 có phải tách bốn người nói? | Không. B=2 hỗn hợp, mỗi hỗn hợp tách J=2 nguồn. |
| NoGate có nghĩa không có phép nhân hoặc gating nào? | Không. Bỏ speaker-guided gate của khối cải tiến; vẫn còn u nhân m và GLU/gating trong backbone. |
| Hai Conv-U có dùng chung trọng số? | Không; cùng cấu trúc, hai bộ trọng số riêng. |
| Các speaker có LTRR riêng? | Dùng chung một LTRR trên chiều batch đã gộp BJ. |
| Alpha là hằng số 0,05? | 0,05 là khởi tạo; alpha là scalar học được và code không ép nó nằm trong [0,1]. |
| Có bảo đảm LTRR ban đầu giống baseline? | Không; alpha khởi tạo khác 0. Paired initialization chỉ bảo đảm trọng số backbone tương ứng giống nhau. |
| Dense memory có cộng trực tiếp input v ở cuối? | Không. Nó trả out_proj của concat các state; residual z nằm ở TemporalInteraction. |
| Dense memory có lưu trạng thái giữa các utterance? | Không. State là tensor trung gian trong một lần forward. |
| Head chính có nhân mask với AudioEncoder không? | Không, masking=False; auxiliary heads mới bật masking. |
| Nhánh auxiliary có phải đầu ra sau D0…D3? | Code lấy trước mỗi decoder stage: 500/1000/2000/4000 frame. |
| Đây có phải mô hình chạy causal? | Không. Có padding đối xứng và xử lý ngữ cảnh hai phía; chưa triển khai causal/streaming. |
| Các kernel, dilation và bottleneck đã tối ưu chưa? | Đây là cấu hình được lựa chọn/giữ theo thiết kế; cần ablation để kết luận tối ưu. |

## Ghi chú kỹ thuật dành cho người trình bày

- Bản nói dùng B=2 cho nhất quán với hình. Với input batch B=1 và waveform thông thường N>1, nhánh điều kiện trong `AudioDecoder.forward` chỉ squeeze chiều channel, nên đầu ra vẫn là `[1,N]`. `Model.forward` cũng thêm chiều batch khi nhận đầu vào `[N]`. Ghi chú trước đây nói trường hợp này trả `[N]` đã được sửa theo code.
- Bộ shape audit chạy model chưa train với đầu vào zero ở eval mode. Nó kiểm tra kích thước, không kiểm tra chất lượng separation hoặc hội tụ.
- Khi N thay đổi, các con số T0/Tp/chiều dài các stage thay đổi theo; cấu trúc channel và số stage không đổi.
- Không đọc bảng normalization thành “GroupNorm và LayerNorm giống nhau”: chúng có trục tính thống kê khác nhau trong code này.
- Không nói dilation biến thiên theo từng đầu vào: danh sách 1/2/4/8 là cấu hình cố định; trọng số convolution là thứ được học.
- Không gọi phép chiếu giảm kênh 128→64 là downsampling theo thời gian. Temporal downsampling chỉ nằm ở DownConv của separator trong luồng được mô tả.
- Dùng danh từ “đặc trưng của các nhánh nguồn” trước AudioDecoder; chưa gọi tensor [BJ,128,T] là waveform sạch hoặc kết quả đã tách hoàn hảo.
