# Đánh giá SepReformer-PARR: ý tưởng, triển vọng và vấn đề triển khai

> Báo cáo dưới đây mô tả trạng thái trước sửa. F1–F7 đã được xử lý ở lượt tiếp theo; xem [bản sửa và quy trình kiểm chứng](PARR_FIXES_2026_09_10.md). Script probe đi kèm tái hiện công thức cũ để làm bằng chứng lịch sử, không phải regression cho implementation mới.

**Kết luận.** PARR là một giả thuyết cải tiến hợp lý để tiếp tục thí nghiệm: thêm phép hiệu chỉnh có điều khiển sau từng tầng tái dựng, dùng chung một lõi nhỏ để hạn chế tăng tham số. Tuy nhiên, chưa có bằng chứng thực nghiệm trong workspace chứng minh PARR tốt hơn baseline Libri2Mix 16 kHz. Đóng góp khả dĩ nằm ở cách tích hợp và kiểm soát refinement; không nên trình bày gated convolution, dilated memory, progressive reconstruction hay zero-initialized residual như những nguyên lý mới do PARR phát minh.

Phần triển khai PARR nhìn chung khớp mô tả kiến trúc. Những vấn đề cần ưu tiên hiện nằm ở loss STFT trên batch có padding, kiểm tra cấu hình khi resume, và một số trường hợp biên của evaluator. Nên giải quyết tính đúng đắn của phép so sánh trước khi tăng độ phức tạp kiến trúc.

**1. Phạm vi và mức độ bằng chứng**

Đối tượng là `SepReformer_PARR_Libri2Mix_16K` và baseline cục bộ `SepReformer_Base_Libri2Mix_16K`, với mã nguồn tại workspace ngày 10/09/2026, bao gồm thay đổi chưa commit. Baseline WSJ0 8 kHz chỉ cung cấp bối cảnh kiến trúc; checkpoint đó không thay thế baseline cần huấn luyện trong cùng điều kiện.

| Kiểm tra | Kết quả và giới hạn |
|---|---|
| Đối chiếu hai package | `model.py`, `dataset.py`, `main.py`, `modules/network.py` giống nội dung; `engine.py` chỉ khác một docstring |
| Đối chiếu YAML | Hai phần `config` giống nhau sau khi loại riêng `module_separator.parr` |
| Cú pháp Python | Phân tích AST thành công 39 file trong `models`, `utils`, `scripts` |
| Luồng kiến trúc | Đọc trực tiếp vị trí gọi PARR, nhánh auxiliary, normalization, gate và gamma |
| Phép thử công thức | Đã chạy script thư viện chuẩn Python để kiểm tra STFT, config hash, số tham số và độ dài receptive field |
| PyTorch runtime hiện tại | Chưa chạy: Python mặc định không có `torch`; `.venv-audit` trỏ tới Python 3.11 gốc không còn tồn tại |
| Huấn luyện và đánh giá | Không tìm thấy checkpoint hoặc metrics của hai model Libri2Mix trong phạm vi thư mục model; có checkpoint WSJ0 |
| GPU L4, backward và latency | Chưa kiểm chứng trong lượt đánh giá này |

Các kết quả CPU ghi trong audit ngày 07/09 là bằng chứng lịch sử, không được coi là kết quả vừa chạy lại trên trạng thái hiện tại. Script [parr_review_2026_09_10_probe.py](parr_review_2026_09_10_probe.py) ghi rõ đây là kiểm tra source/công thức, không thay thế forward/backward PyTorch.

**2. PARR thực sự thay đổi gì so với baseline?**

Baseline cục bộ có encoder waveform, feature projector, bốn tầng encoder của separator, bottleneck, speaker split, bốn tầng reconstruction decoder và waveform decoder. Mỗi tầng reconstruction đã chứa global block, local block và cross-speaker attention. Khối CLA dùng kernel 65; GCFN cũng có tích chập theo thời gian. Vì vậy, baseline đã xử lý cả ngữ cảnh gần, ngữ cảnh xa và thông tin giữa người nói.

PARR được gọi sau mỗi tầng reconstruction tại `models/SepReformer_PARR_Libri2Mix_16K/modules/module.py:407`. Với tensor đầu vào `H_r` có kích thước `[B*J,128,T_r]`, phần bổ sung là:

```text
N_r = ChannelLayerNorm_r(H_r)
Z_r = ChannelLayerNorm(PReLU(Conv1x1_128_to_64(N_r)))
U_r = ConvU_u(Z_r)
V_r = ConvU_v(Z_r)
M_r = DenseDilatedMemory(V_r)
R_r = ChannelLayerNorm(U_r * M_r)
A_r = sigmoid(Controller(concat(Z_r, R_r)))
Delta_r = Conv1x1_64_to_128(A_r * R_r)
H'_r = H_r + gamma_r * Delta_r
```

`ConvU_u` và `ConvU_v` là **hai module có trọng số riêng**. Chúng có cùng cấu trúc nhưng không bị ràng buộc bằng nhau. Toàn bộ lõi chứa cả hai module được chia sẻ giữa các stage và speaker; stage normalization và bốn gamma là riêng. Nên dùng ký hiệu `theta_u`, `theta_v` trong luận văn để tránh đọc công thức thành hai lần gọi cùng một phép biến đổi.

| Thuộc tính | Baseline | PARR hiện tại |
|---|---|---|
| Global/local/cross-speaker processing | Có | Giữ nguyên |
| Hiệu chỉnh sau decoder stage | Không có khối PARR riêng | Có ở cả bốn stage |
| Khả năng xử lý theo nhiều độ phân giải | Có sẵn | Dùng cùng lõi refinement tại nhiều độ phân giải |
| Controller bổ sung | Không | Gate theo channel và thời gian, `[B*J,64,T_r]` |
| Residual scale bổ sung | Không | Một scalar học được cho mỗi stage |
| Auxiliary supervision | Có bốn đầu ra | Giữ nguyên vị trí và objective |
| Quan sát mới vào khối refinement | — | Chỉ feature hiện tại; không có waveform error hoặc mixture residual trực tiếp |
| Trạng thái hồi quy theo thời gian | — | Không; đây là mạng feedforward tích chập |

Luồng auxiliary là `bottleneck, H'_0, H'_1, H'_2`; đầu ra chính dùng `H'_3`. Nguyên nhân là `each_stage_outputs.append(x)` chạy trước bước upsample/decoder/PARR trong vòng lặp (`module.py:397`). Đây không phải lỗi lệch index: nó bảo toàn thiết kế baseline. Không nên mô tả thành bốn auxiliary head đều nằm sau bốn PARR, hoặc tự ý chuyển vị trí append để “sửa”.

Hai mô hình dùng cùng loss chính PIT SI-SNR miền thời gian và bốn loss phụ miền STFT magnitude. Trong 100 epoch đầu, trọng số kết hợp là 0,6 cho loss chính và 0,4 cho trung bình loss phụ; sau đó alpha giảm. PARR không thêm loss mới và không ép chất lượng tăng đơn điệu qua từng stage.

**3. Cơ sở nghiên cứu và giới hạn tính mới**

SepReformer là mô hình trong bài *Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation*, NeurIPS 2024. Bài gốc đã có early split, decoder chia sẻ giữa speaker, reconstruction theo nhiều stage và auxiliary losses. Do đó, lập luận cải tiến cần tập trung vào refinement bổ sung trong kiến trúc này. [Bài SepReformer, phần 3](https://arxiv.org/html/2406.05983v2#S3).[^1]

MossFormer2 là công trình gần nhất với lõi PARR: bottleneck, hai Conv-U, dilated FSMN, phép nhân hai nhánh và output projection đều có tiền lệ rõ ràng. PARR hiện dùng dense depthwise **1D** temporal memory, trong khi bài MossFormer2 mô tả thiết kế memory có các khối tích chập **2D**. Vì vậy, đây là một biến thể lấy cảm hứng từ MossFormer2, không phải bản cài đặt nguyên dạng của module đó. [MossFormer2, phần 2.3](https://arxiv.org/html/2312.11825v2#S2.SS3).[^2]

FSMN gốc dùng cấu trúc feedforward có memory blocks để biểu diễn ngữ cảnh mà không cần recurrent feedback. Tên `DenseDilatedFSMN1D` trong code có thể dùng nội bộ, nhưng mô tả học thuật chính xác hơn là “FSMN-inspired dense dilated temporal memory”. [Zhang và cộng sự](https://arxiv.org/abs/1512.08301).[^3]

Khởi tạo gamma bằng zero có tiền lệ trực tiếp trong ReZero. PARR áp dụng nguyên tắc đó cho nhánh bổ sung; điều này không đủ để suy ra toàn bộ SepReformer có tính chất hội tụ hay dynamical isometry giống thí nghiệm của ReZero. [Bachlechner và cộng sự](https://arxiv.org/abs/2003.04887).[^4]

Đến thời điểm đánh giá, nhóm tác giả SepReformer đã công bố preprint SR-CorrNet, đưa chiến lược separation/reconstruction vào backbone miền thời gian–tần số, có stage-wise refinement và correlation-based filtering. Bản arXiv cập nhật 14/05/2026 ghi “submitted”, không phải bằng chứng đã được nhận xuất bản. Công trình này cần có trong related work nếu luận văn thảo luận bối cảnh hiện tại; nó không tự động phủ nhận giá trị của một cải tiến nhỏ trên baseline waveform. [SR-CorrNet](https://arxiv.org/abs/2603.29097).[^5]

**Nhận định:** tổ hợp có triển vọng để tạo đóng góp thực nghiệm, nhưng tính mới của các thành phần riêng lẻ thấp. Tính mới của chính tổ hợp chưa được xác lập chỉ bằng việc không thấy một bài có cùng tên PARR. Cách phát biểu có thể bảo vệ được là: “khảo sát hiệu quả của một khối refinement có điều khiển, chia sẻ giữa các tầng reconstruction của SepReformer”.

**4. Vì sao đáng thử, và vì sao có thể không hiệu quả?**

PARR cho mô hình khả năng điều chỉnh feature ngay trước khi truyền sang tầng tái dựng tiếp theo. Phép nhân `U*M` tạo tương tác phi tuyến giữa nhánh cục bộ và nhánh memory; controller có thể thay đổi mức đóng góp theo channel/time. Nếu phần dư hữu ích thật sự, các tầng sau có thể nhận feature dễ tái dựng hơn. Đây là giả thuyết về cơ chế, chưa phải quan sát thực nghiệm.

Chia sẻ trọng số giúp hạn chế tăng dung lượng. Từ số chiều và từng layer trong code, số tham số bổ sung là **98.186**, gồm 97.158 ở lõi và 1.028 ở normalization/gamma theo stage. Đây là phép đếm giải tích, khớp số đã ghi trong audit cũ; không phải số vừa đo bằng PyTorch.

Tuy nhiên, ít tham số không đồng nghĩa ít chi phí thực thi. Với input 64.000 mẫu, encoder tạo 7.997 frame, separator pad thành 8.000; các PARR chạy ở 1.000, 2.000, 4.000 và 8.000 frame. Tổng tích chập PARR xấp xỉ **2,8608 G MAC cho 4 giây**, batch 1 và hai speaker, tương đương 0,7152 G MAC/giây âm thanh. Con số này chưa gồm normalization, activation, phép nhân, nối tensor hoặc memory traffic; cần đo latency/VRAM thật để quyết định có “nhẹ” hay không.

Đường thời gian dài nhất bên trong PARR có:

```text
1 + (3-1) + (3-1)*(1+2+4+8) = 33 frame
```

Số hạng đầu sau 1 là Conv-V; phần còn lại đi qua cả bốn memory layers. Ở bốn stage, bước frame tương ứng khoảng 4, 2, 1 và 0,5 ms. Khoảng cách từ vị trí đầu tới cuối cửa sổ 33 frame lần lượt là **128, 64, 32 và 16 ms**. Đây chỉ là receptive field gia tăng tính theo đầu vào của PARR; feature đầu vào đã mang ngữ cảnh từ toàn backbone.

Một CLA của baseline đã có kernel 65, lớn hơn 33 frame này. Vì vậy, không có cơ sở gọi PARR là giải pháp mở rộng trí nhớ thời gian so với baseline. Lý do đáng thử là loại tương tác và vị trí hiệu chỉnh khác, không phải độ phủ dài hơn.

| Rủi ro nghiên cứu | Điều cần đo để phân biệt |
|---|---|
| PARR lặp lại chức năng local/GCFN có sẵn | So với residual temporal block đơn giản có dung lượng tương đương |
| Shared core nhận feature khác nhau giữa stage | Đo gradient từng stage; so shared với unshared hoặc adapter nhỏ |
| Cùng dilation nhưng khoảng thời gian vật lý khác nhau | So dilation cố định với dilation phụ thuộc stage |
| Gate chỉ học gần một hằng số | Phân bố gate và ablation gate cố định 0,5 |
| Gamma gần zero khiến nhánh bị bỏ qua | Theo dõi gamma và tỷ lệ norm correction/feature theo stage |
| Residual làm suy giảm tính phân biệt speaker sau CS attention | So lỗi rò nguồn/SIR và ví dụ âm thanh, không chỉ SI-SNRi trung bình |
| Lợi ích chỉ do tăng capacity | Đối chứng cùng dung lượng, cùng training budget |

`gamma=0` không phải lỗi làm PARR chết vĩnh viễn. Với `H'=H+gamma*f_theta(H)`, gradient theo theta bằng zero tại khởi tạo, nhưng gradient theo gamma nhìn chung khác zero. Sau update đầu, gradient có thể đi vào lõi. Bộ `runtime_checks.py` đã có kiểm tra gamma tự mở; smoke test riêng đặt gamma thành 0,1 chỉ kiểm tra một phần khác của vấn đề. Không nên thay gamma_init chỉ để khiến gradient controller khác zero ngay ở backward đầu tiên.

Exact identity của module không đồng nghĩa hai model có mọi output training giống bit-for-bit. PARR chạy dropout và tiêu thụ RNG ngay cả khi gamma bằng zero; điều đó có thể làm đổi mask dropout ở các khối phía sau. Paired initialization và so sánh `eval()` là cách kiểm tra phù hợp. Nhiều seed vẫn cần thiết để đánh giá chênh lệch chất lượng.

**5. Phát hiện triển khai và thứ tự xử lý**

Các mức ưu tiên dưới đây phản ánh ảnh hưởng tới độ tin cậy của nghiên cứu. “Xác nhận bằng source/công thức” không có nghĩa đã tái hiện bằng backend CUDA.

| Mã | Ưu tiên | Phát hiện | Phạm vi |
|---|---|---|---|
| F1 | Cao | Frame STFT của câu ngắn phụ thuộc độ dài padded batch | Loss chung của hai model |
| F2 | Vừa | Shared-config hash không phát hiện thay đổi riêng dilation/dropout PARR khi resume | Kiểm soát thí nghiệm |
| F3 | Vừa | Batch audio quá ngắn có thể ngắn hơn kernel STFT | Robustness loss chung |
| F4 | Vừa | Ước lượng im lặng được evaluator SI-SNR cho 0 dB | Trường hợp suy biến của metric |
| F5 | Thấp | Tắt bootstrap trả CI bằng giá trị mẫu đầu tiên | Nhánh cấu hình không mặc định |
| F6 | Cần đo | Mask ở loss không ngăn padding ảnh hưởng backbone | Training/validation có độ dài khác nhau |
| F7 | Cần tối ưu | STFT tính lại nhiều lần trong PIT và các auxiliary head | Tốc độ huấn luyện |

**F1 — STFT chưa độc lập với padding của batch.** Vị trí: `utils/implements/criterions.py:102`, `:109`, `:185`, `:191`. Loss đã zero-mask theo `input_sizes`, nhưng STFT chọn số frame từ chiều dài tensor chung. Một câu ngắn đứng một mình có ít frame hơn chính câu đó khi ghép với câu dài. Các frame thêm có thể giao với đuôi tín hiệu thật, nên không thể xem chúng là toàn zero rồi bỏ qua.

Phép thử công thức dùng đúng frame 1.024, hop 256, periodic Hann, scale normalization và epsilon của source. Cùng reference/estimate có 1.536 mẫu hợp lệ, lỗi tập trung ở cuối câu: khi batch length là 1.536 thì có 3 frame, loss một cặp là **−16,287956 dB**; khi batch length là 2.560 thì có 7 frame, loss là **−7,799353 dB**. Chênh lệch **8,488603 dB** không đến từ thay đổi tín hiệu hợp lệ. Đây là ví dụ tổng hợp chứng minh cơ chế ở một cặp trước PIT, không phải số liệu SI-SNRi test hoặc mức sai lệch trung bình trên Libri2Mix.

Hướng sửa: định nghĩa frame coverage theo chiều dài thật từng utterance, áp dụng cùng quy ước right-padding và frame mask cho estimate/reference. Bản đúng dễ kiểm chứng nhất là crop từng utterance rồi tính STFT riêng; bản tối ưu có thể vectorize nhưng phải tương đương. Thêm regression so loss của cùng utterance khi đứng một mình và trong batch dài hơn. Chỉ mask waveform trước STFT chưa đủ. Áp dụng cùng thay đổi cho baseline/PARR và ghi nhận phiên bản loss.

**F2 — Resume thiếu fingerprint của cấu hình đầy đủ.** Vị trí: `utils/paired_initialization.py:31`, `:37`, `:231`, `:242`. Việc bỏ `parr` khỏi hash là đúng cho mục đích chia sẻ initialization giữa baseline và candidate. Nhưng resume chỉ kiểm tra các metadata chung này và tensor shapes. Thay `[1,2,4,8]` thành `[1,2,4,16]` hoặc dropout 0,05 thành 0,2 không đổi tên/shape trọng số; hash cũng không đổi. Phép thử gọi chính hai hàm hash trích từ source đã xác nhận điều này.

Hướng sửa: giữ shared hash cho epoch-0, bổ sung `full_training_config_sha256` và fingerprint kiến trúc của từng run vào checkpoint, kiểm tra khi resume. Dilation/dropout đã train phải được đối chiếu riêng. Evaluation nên xuất checkpoint hash, config snapshot và paired-initialization metadata của checkpoint đã nạp để truy nguyên kết quả. Đây là lỗ hổng kiểm soát run, không phải bằng chứng một run hiện tại đã bị resume sai.

**F3 — Audio ngắn hơn cửa sổ STFT có thể lỗi.** Vị trí: `criterions.py:102–109`. Với batch length 512, padding lên bội số hop vẫn là 512, nhỏ hơn kernel 1.024. Valid Conv1d không có frame hợp lệ. Loader hiện chỉ kiểm tra độ dài đạt một stride, không đảm bảo đạt một STFT window. Điều kiện này xảy ra khi cả batch đủ ngắn; chưa xác minh có mẫu như vậy trong archive hiện tại.

Hướng sửa: đảm bảo số mẫu tối thiểu để tạo frame, kết hợp quy ước boundary thống nhất ở F1. Thêm kiểm tra lengths 8, 512, 1.016, 1.024 và batch trộn ngắn/dài. Tín hiệu dưới kernel waveform 32 cũng cần được model/API xử lý hoặc từ chối rõ ràng. Không tự loại mẫu test để làm cho phép chạy thành công.

**F4 — Metric của đầu ra suy biến dễ bị hiểu sai.** Vị trí: `utils/evaluation_metrics.py:86–93`. Nếu estimate hoàn toàn bằng zero và reference có năng lượng, target chiếu và noise đều zero. Công thức cộng epsilon ở cả tử/mẫu cho `10*log10(eps/eps)=0 dB`. Training loss thời gian trong trường hợp tương ứng có quy ước phạt khác, nên cần công bố rõ sự khác nhau.

Hướng sửa: phát hiện reference/estimate suy biến sau zero-mean; chọn chính sách metric rõ ràng, ghi failure hoặc score floor đã định nghĩa, không âm thầm diễn giải 0 dB như một chất lượng tách hợp lệ. Có test cho silence, DC, near-silence, waveform chuẩn và đổi dấu. Vì SI-SNR bất biến theo scale ở trường hợp không suy biến, ngưỡng phát hiện không nên được chọn tùy tiện theo gain tuyệt đối. Chính sách phải giống nhau ở cả hai mô hình, và failure không được lặng lẽ làm biến mất những utterance khó.

**F5 — Nhánh không bootstrap trả CI không đại diện dữ liệu.** Vị trí: `utils/evaluation_metrics.py:135–138`. Khi có nhiều giá trị nhưng `bootstrap_samples<=0`, hàm trả `(values[0],values[0])`. Ví dụ `[1,3]` có mean 2 nhưng interval trả `[1,1]`. Cấu hình hiện là 2.000 nên chưa kích hoạt lỗi. Nếu cho phép tắt bootstrap, nên trả CI không khả dụng; không thể coi một điểm lấy từ mẫu đầu là confidence interval. Đồng thời tên `ci95_*` nên phản ánh trường `confidence` nếu cho phép đặt giá trị khác 0,95.

**F6 — Backbone chưa nhận valid-length mask.** Vị trí: `module.py:23`, `:300`; `network.py:138–149`; model forward không nhận `input_sizes`. `GroupNorm(1,C)` tính thống kê qua các chiều không gian của mỗi mẫu, còn global pooling/attention vẫn xử lý frame padding. Vì thế zero-mask trong loss không đảm bảo output vùng hợp lệ độc lập với độ dài câu cùng batch. Đây là hành vi cần đo, không phải lỗi shape của riêng PARR. [PyTorch GroupNorm](https://docs.pytorch.org/docs/2.1/generated/torch.nn.GroupNorm.html).[^8]

Hướng xử lý ít xáo trộn: đo validation cùng câu ở batch 1 và batch có padding; dùng batch 1 khi cần checkpoint selection gần điều kiện test, hoặc bucket theo độ dài. Nếu sai lệch đáng kể, thiết kế masked normalization/pooling/attention đồng bộ cho cả hai model. Thay GroupNorm bằng ChannelLayerNorm tùy tiện sẽ đổi kiến trúc baseline và cần trở thành thí nghiệm riêng.

**F7 — STFT bị tính lặp trong PIT.** Trong mỗi auxiliary stage, hai permutation × hai speaker × STFT estimate/reference tạo tám lần gọi STFT; bốn stage thành 32 lần. Reference lại được tính ở nhiều permutation và stage. Điều này suy ra từ vòng lặp ở `criterions.py:180–200`, chưa phải benchmark xác nhận nút thắt.

Hướng tối ưu: tính STFT cho mỗi estimate một lần/stage và reference một lần với quy ước frame thống nhất, sau đó xây pairwise cost matrix. Scaling hiện được áp dụng trên waveform trước magnitude có epsilon, nên chuyển scaling ra ngoài không hoàn toàn tương đương nếu xử lý epsilon sai. So cả giá trị loss và gradient trên dữ liệu thường, scale âm/gần zero và padding trước khi thay đường train.

**6. Những điểm nên cải thiện nhưng không nên gọi là bug mặc định**

Output projection của PARR (`module.py:162`) có bias. Nếu controller đóng hoàn toàn về mặt giới hạn, `Delta=W(0)+b=b`, nên gate không tắt được toàn bộ correction. Điều này không phá identity tại gamma zero. Nếu muốn diễn giải gate như “tắt refinement”, nên ablate `bias=False`; nếu giữ code hiện tại, mô tả gate là điều khiển phần input-dependent của correction. Không diễn giải gate là xác suất lỗi hay độ tin cậy đã được hiệu chuẩn.

Các trường hợp khác cần phân biệt mức độ ảnh hưởng:

| Điểm | Đánh giá |
|---|---|
| Norm của `U*M` loại bớt thông tin biên độ | Một tradeoff về ổn định và biểu diễn; cần ablation, không tự động sai |
| Gamma là scalar không bị giới hạn dấu/độ lớn | Phù hợp residual học được; đo tỷ lệ correction trước khi thêm clamp/regularization |
| PIT độc lập giữa các auxiliary head | Kế thừa baseline; khác permutation có thể do head-specific ordering, không tự chứng minh speaker swap vật lý |
| Mean loss theo số batch thay vì số utterance | Nên sửa khi hỗ trợ batch cuối nhỏ; train 13.900/dev 3.000 với batch 2 hiện đều chia hết |
| Warmup chỉ step ở epoch đầu | Recipe hiện có 6.950 update/epoch nên đủ 1.000 warmup steps; subset nhỏ có thể dừng warmup quá sớm |
| Public model với input không chia hết stride | Output có thể thiếu tối đa 7 mẫu; dataset đang trim, inference sample đang pad nên đường mặc định đã né trường hợp này |
| `self.attn` giữ tensor attention sau forward | Có thể tăng thời gian lưu graph/memory giữa các lượt; cần memory trace trước khi kết luận OOM |
| EGA tạo attention theo độ dài bottleneck | Chi phí tăng theo bình phương độ dài bottleneck; train 4 giây không đủ chứng minh xử lý file nhiều phút |
| `deterministic=True` đi cùng `warn_only=True` | Không cam kết mọi CUDA operation deterministic; ghi lại cảnh báo và môi trường |

Không nên thêm STFT loss mới, mixture consistency, stage-consistent PIT và thay dilation đồng thời rồi gán toàn bộ lợi ích cho PARR. Những thay đổi ấy cần tách thành từng yếu tố thí nghiệm.

**7. Kế hoạch thí nghiệm đủ để kết luận triển vọng**

Trước hết khôi phục môi trường đúng recipe và chạy các regression hiện có, cùng test PyTorch cho F1–F4. Xác nhận waveform output, bốn auxiliary head, hai update optimizer thực sự, checkpoint/resume và tất cả gamma mở tự nhiên. Sau đó mới chạy pilot; không lấy hai minibatch thành công làm bằng chứng model sẽ hội tụ sau 200 epoch.

| ID | Cấu hình | Câu hỏi được trả lời |
|---|---|---|
| B | Baseline Libri2Mix 16 kHz | Mốc so sánh trực tiếp |
| P | PARR đầy đủ | Tổ hợp có tạo lợi ích không? |
| G | PARR với gate cố định 0,5 | Adaptive controller có thực sự hữu ích không? |
| F | PARR chỉ tại stage cuối | Chèn nhiều stage có đáng chi phí không? |
| T | Residual temporal block đơn giản, xấp xỉ cùng tham số | Lợi ích có vượt việc thêm capacity không? |
| U | PARR không chia sẻ lõi giữa stage | Sharing tiết kiệm tham số với mức đánh đổi nào? |

Ưu tiên B/P/G/F/T; U là mở rộng nếu ngân sách cho phép. Gate cố định 0,5 giữ cùng biên độ ban đầu với gate neutral. `A=1` vẫn là ablation bổ sung có ích nhưng trộn hiệu ứng thích nghi với thay đổi biên độ input của output projection. Khi loại controller cần báo lại params/MACs; không gán cho hai variant cùng chi phí nếu thực tế đã khác.

Pilot ngắn dùng validation để loại cấu hình lỗi hoặc nhánh không học. Không chọn cấu hình tốt nhất theo test và không giả định thứ hạng ở vài epoch đầu giữ nguyên khi hội tụ. Với ứng viên giữ lại, chạy cùng full schedule, cùng revision loss, cùng data split/crop/batch/optimizer; tối thiểu ba seed nếu tài nguyên cho phép. Mỗi seed của PARR dùng cùng shared epoch-0 với baseline tương ứng.

Thông tin nên log theo từng stage gồm gamma, norm feature, norm `gamma*Delta`, tỷ lệ hai norm, trung bình/độ lệch chuẩn gate, tỷ lệ gate dưới 0,05 hoặc trên 0,95, và gradient norm của shared core. Có thể đo cosine similarity giữa gradient do từng stage đóng góp trên một số minibatch để kiểm tra xung đột khi sharing. Những thống kê đó là chẩn đoán; cần can thiệp ablation mới có cơ sở nói controller gây ra cải thiện.

Đánh giá chính là **paired delta SI-SNRi trên cùng utterance test** và bootstrap CI của delta. CI theo utterance chỉ phản ánh biến thiên trên tập đánh giá của checkpoint đang xét; không thay thế độ lệch giữa seed huấn luyện. Báo thêm SDRi, STOI/PESQ, failure counts, median và phần đuôi phân phối. Nếu các metric có mẫu không hợp lệ, dùng tập cặp hợp lệ chung và công bố số mẫu bị loại.

Đo thời gian và peak VRAM với cùng GPU, dtype, batch size, chế độ inference, độ dài audio và phạm vi auxiliary heads. MAC count từ công cụ khác nhau có thể bỏ qua các phép `matmul`/functional operations khác nhau; không kết luận từ một con số profiler duy nhất. Test cả độ dài gần 4 giây và các utterance dài hơn xuất hiện trong dữ liệu.

Một tiêu chí quyết định có thể đăng ký trước là: delta SI-SNRi dương nhất quán giữa seed, CI paired phù hợp với lợi ích, và mức tăng latency nằm trong ngân sách ứng dụng. Có thể chọn mức cải thiện thực dụng như 0,2 dB nếu phù hợp mục tiêu khóa luận, nhưng đó là ngưỡng do dự án đặt ra, không phải dự đoán kết quả hoặc chuẩn cộng đồng.

**8. Hướng cải tiến kiến trúc nên thử theo thứ tự**

**Ưu tiên đầu tiên: giữ PARR hiện tại làm mốc và làm phép đo đáng tin.** Với trạng thái hiện có, giá trị cao nhất đến từ sửa loss/padding, khóa config run, bổ sung diagnostics và ablation. Chưa có dữ liệu cho thấy cần thay thế toàn bộ module.

**Ưu tiên thứ hai: stage conditioning nhỏ.** Nếu shared core bị xung đột giữa các scale, thêm stage embedding hoặc affine modulation nhỏ vào bottleneck/controller là một cách cho lõi biết nó đang ở mức tái dựng nào. Các stage_norm hiện đã có affine riêng, nên phải so trực tiếp với chúng để chứng minh modulation mới không dư thừa. Khởi tạo modulation neutral để giữ kiểm soát epoch-0.

**Ưu tiên thứ ba: dilation theo thời gian vật lý.** Nếu mục tiêu là cùng khoảng ngữ cảnh tính theo millisecond, cấu hình dilation phụ thuộc độ phân giải có thể phù hợp hơn cố định `[1,2,4,8]`. Đổi dilation không cần đổi số trọng số, nhưng có thể tăng gridding/boundary effects. Chỉ làm sau khi F2 được xử lý để tránh resume nhầm cùng tensor shapes nhưng khác receptive field.

**Ưu tiên thứ tư: gate có thêm bằng chứng về sự lẫn nguồn.** Hiện controller chỉ nhìn feature cùng nhánh. Có thể thử context tổng hợp giữa speaker bằng phép tổng/trung bình đối xứng hoặc mixture-conditioned feature để gate nhận biết thông tin mà nó chưa có trực tiếp. Tuy nhiên, đầu vào PARR đã qua cross-speaker attention; lợi ích bổ sung cần đo và phải giữ tính nhất quán khi hoán đổi speaker branches.

Mixture consistency là hướng riêng: với bài toán đầy đủ `mix_clean=s1+s2`, có thể xét projection làm tổng các ước lượng khớp mixture. Kỹ thuật này đã có tiền lệ; không phải thành phần mới của PARR. Khi dùng SI-SNR, gain/polarity của từng estimate cần được xem xét vì phép cộng waveform phụ thuộc scale. [Wisdom và cộng sự, ICASSP 2019](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/).[^6]

Không suy ra khả năng khái quát sang tiếng Việt, môi trường noisy/reverberant hay số speaker biến đổi chỉ từ kết quả clean Libri2Mix. Dataset cung cấp các biến thể khác nhau về sampling rate, mixture type và min/max; phải ghi đúng biến thể đang đánh giá. [Kho chính thức LibriMix](https://github.com/JorisCos/LibriMix).[^7]

**9. Đánh giá tổng thể**

| Khía cạnh | Nhận định |
|---|---|
| Tính hợp lý của ý tưởng | Có cơ sở; đáng làm pilot và ablation |
| Tính mới của thành phần | Hạn chế; có tiền lệ rõ ở MossFormer2/ReZero |
| Giá trị của tổ hợp | Chưa biết; phụ thuộc ablation và hiệu quả tính toán |
| Mức khớp giữa mô tả PARR và code | Khá tốt, với lưu ý hai Conv-U riêng và vị trí auxiliary |
| Tính đúng đắn pipeline chung | Cần xử lý STFT/padding và các trường hợp biên đã nêu |
| Bằng chứng vượt baseline | Chưa có trong workspace được kiểm tra |
| Hướng phù hợp cho khóa luận | Cải tiến nhỏ có kiểm soát, giải thích được, có phân tích thất bại |

Phát biểu phù hợp trước khi có kết quả: **“PARR được đề xuất để khảo sát hiệu quả của refinement có điều khiển và chia sẻ trọng số trong reconstruction decoder của SepReformer.”** Chỉ đổi thành “cải thiện chất lượng tách” khi thí nghiệm có kiểm soát xác nhận. Nếu full PARR không vượt gate cố định hoặc block đơn giản, kết quả đó vẫn giúp xác định phần thiết kế nào thực sự cần thiết.

**Nguồn tham khảo**

[^1]: Ui-Hyeop Shin, Sangyoun Lee, Taehan Kim, Hyung-Min Park. *Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation*. NeurIPS 2024; arXiv v2, 11/10/2024. [Toàn văn](https://arxiv.org/html/2406.05983v2). Dùng để đối chiếu kiến trúc và multi-loss của baseline.
[^2]: Shengkui Zhao và cộng sự. *MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation*. ICASSP 2024; arXiv v2, 28/11/2024. [Toàn văn](https://arxiv.org/html/2312.11825v2). Dùng để xác định tiền lệ gần nhất của lõi temporal refinement.
[^3]: Shiliang Zhang và cộng sự. *Feedforward Sequential Memory Networks: A New Structure to Learn Long-term Dependency*. arXiv 2015; v2, 05/01/2016. [Bài gốc](https://arxiv.org/abs/1512.08301). Dùng để phân biệt FSMN với recurrent feedback và biến thể tích chập hiện tại.
[^4]: Thomas Bachlechner và cộng sự. *ReZero is All You Need: Fast Convergence at Large Depth*. arXiv 2020. [Bài gốc](https://arxiv.org/abs/2003.04887). Dùng cho tiền lệ zero-initialized residual scale.
[^5]: Ui-Hyeop Shin, Hyung-Min Park. *Asymmetric Encoder-Decoder Based on Time-Frequency Correlation for Speech Separation*. Preprint SR-CorrNet, arXiv v2, 14/05/2026. [Bài gốc](https://arxiv.org/abs/2603.29097). Dùng cho bối cảnh nghiên cứu tiếp nối SepReformer, không thay baseline có kiểm soát.
[^6]: Scott Wisdom và cộng sự. *Differentiable Consistency Constraints for Improved Deep Speech Enhancement*. ICASSP 2019. [Google Research](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/). Dùng cho hướng mở rộng mixture consistency.
[^7]: Joris Cosentino và cộng sự. *LibriMix: An Open-Source Dataset for Generalizable Speech Separation*. 2020. [Kho chính thức](https://github.com/JorisCos/LibriMix), [bài gốc](https://arxiv.org/abs/2005.11262). Dùng cho phạm vi dataset và biến thể.
[^8]: PyTorch. *GroupNorm*, tài liệu API 2.1. [Tài liệu](https://docs.pytorch.org/docs/2.1/generated/torch.nn.GroupNorm.html). Dùng cho phạm vi tính thống kê normalization.

Nguồn cục bộ bổ sung: hai package Libri2Mix, `utils/implements/criterions.py`, `utils/evaluation_metrics.py`, `utils/paired_initialization.py`, `scripts/runtime_checks.py`, `scripts/preflight.py`, `initializations/README.md`, `docs/DATASET_MANIFEST.md` và hai audit PARR trước đó. Số dòng tham chiếu ứng với trạng thái workspace tại ngày đánh giá; báo cáo không thay đổi mã huấn luyện.
