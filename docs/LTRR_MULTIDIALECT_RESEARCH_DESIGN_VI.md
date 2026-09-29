# Thiết kế nghiên cứu: LTRR thích nghi đa thang thời gian trên tiếng Việt đa phương ngữ

Ngày: 2026-09-24. Trạng thái: **đề xuất chưa triển khai, chưa có kết quả thực nghiệm**.

Tài liệu dựa trên `models/SepReformer_LTRR_VnSpeechMix_16K/modules/ltrr.py`, config cùng gói và metadata VN-SpeechMix đang có trong workspace tại commit `00e09ea`. Không thay đổi kiến trúc, config hay kết quả của LTRR gốc. Mọi con số của thiết kế mới bên dưới là thông số đề xuất hoặc phép tính lý thuyết, không phải kết quả đo.

## 1. Quyết định nghiên cứu đề xuất

Ưu tiên một hướng chính: **LTRR-MR — Lightweight Temporal Residual Refinement with Adaptive Multi-Rate Memory**. Đây là tên làm việc, chưa khẳng định tên mới hoặc tính mới so với toàn bộ tài liệu nghiên cứu.

Thay bộ nhớ một tốc độ của LTRR bằng bộ nhớ dùng chung trọng số chạy ở ba tốc độ; học trọng số phối hợp theo đặc trưng của từng nguồn dự đoán. Giữ nguyên vị trí sau reconstruction decoder, kích thước vào/ra và pipeline tách tiếng hiện tại.

Giả thuyết cần kiểm chứng: phối hợp ngữ cảnh cục bộ và ngữ cảnh dài hơn giúp nhánh tinh chỉnh bổ sung sửa lỗi ở nhiều thang thời gian, cải thiện chất lượng tách trên các nhóm phương ngữ mà không cần nhãn phương ngữ khi suy luận.

Không gọi đây là cơ chế đã nhận biết thanh điệu hoặc đã thích nghi phương ngữ. Kiến trúc này có tính tổng quát; mối liên hệ với tiếng Việt phải được kiểm chứng bằng đánh giá phân nhóm và phân tích lỗi.

## 2. LTRR hiện tại: hạn chế nào có bằng chứng?

| Quan sát từ code | Ý nghĩa và giới hạn của nhận xét | Hướng xử lý |
|---|---|---|
| Dense memory dùng `k=3`, dilation `[1,2,4,8]` tại tốc độ 2.000 frame/s | Đường dài nhất trong memory phủ 31 vị trí đầu vào. Ngữ cảnh bổ sung của toán tử này khá cục bộ. Không phải receptive field của toàn SepReformer. | Thêm các tốc độ memory dài hơn, vẫn giữ nhánh gốc. |
| Cùng bộ lọc và quy tắc hợp nhất được dùng cho mọi đầu vào | Các activation vẫn phụ thuộc tín hiệu, nhưng không có bộ chọn thang ngữ cảnh tường minh. | Học trọng số phối hợp các tốc độ theo nguồn dự đoán. |
| `u * m`, với `m` không có activation giới hạn cuối | Có khả năng khuếch đại activation; chưa phải bằng chứng mô hình đang mất ổn định. | Ghi diagnostics trước; thử bounded modulation bằng ablation riêng nếu cần. |
| `alpha` là một scalar học được, khởi tạo 0,05 và không bị chặn | Khởi tạo nhỏ không bảo đảm correction luôn nhỏ trong toàn bộ quá trình train. | Theo dõi alpha, correction ratio, gradient; chưa thay alpha ở thử nghiệm chính. |
| LTRR xử lý tensor `[B*J,C,T]` bằng cùng trọng số | Không có trao đổi thông tin trực tiếp giữa hai nguồn trong LTRR. Tuy nhiên backbone đã có CS-attention. | Chưa thêm cross-speaker attention trùng chức năng nếu chưa chứng minh cần. |
| Bottleneck 128 xuống 64 | Là đánh đổi dung lượng và chi phí; không thể kết luận 64 là sai hoặc làm mất thanh điệu. | Có thể ablate 32/64/128 sau khi xác định lợi ích chính, không quét tất cả ngay. |

Receptive field dài nhất của memory, xét theo đầu vào riêng của memory:

```text
R = 1 + (3 - 1) * (1 + 2 + 4 + 8) = 31 frame
Khoảng cách giữa hai tâm frame xa nhất = (31 - 1) * 0,5 ms = 15 ms
```

Nếu tính thêm Conv-U phía trước, đường dài nhất từ `z` qua nhánh `v` tới memory là 33 frame, khoảng cách tâm 16 ms. Các feature `z` đã chứa ngữ cảnh của backbone. Bộ chọn trọng số theo toàn đoạn trong thiết kế mới cũng tạo phụ thuộc toàn đoạn; không mô tả receptive field của toàn khối mới bằng những số cục bộ này.

## 3. Cơ sở tiếng Việt và điều không nên suy diễn

Tín hiệu thanh điệu liên quan đến đường F0, đồng thời chất giọng/kiểu tạo thanh có vai trò khác nhau giữa các phương ngữ. Nghiên cứu tri giác Bắc–Nam cho thấy khác biệt trong việc sử dụng tín hiệu chất giọng. Vì vậy, chỉ thêm pitch loss không đồng nghĩa đã bảo toàn đầy đủ thanh điệu. [Brunelle, 2009](https://www.sciencedirect.com/science/article/abs/pii/S0095447008000454)

Suy luận thiết kế: nên giữ thông tin biến đổi nhanh đồng thời tạo đường xử lý biến thiên dài hơn. Đây là động cơ nghiên cứu, không phải bằng chứng rằng một rate cụ thể ứng với một thanh hoặc một vùng.

ViMD cung cấp nhãn theo 63 tỉnh/thành trong bộ dữ liệu; VN-SpeechMix sử dụng các nhóm Bắc, Trung, Nam và cặp vùng. Nhãn vùng là mô tả thô, không đồng nhất mọi người nói trong một vùng. [ViMD](https://arxiv.org/abs/2410.03458), [VN-SpeechMix](https://huggingface.co/datasets/pervasiveaidataresearchlab2025/VN-SpeechMix_Dataset)

Không gán cứng ba nhánh memory cho Bắc, Trung, Nam. Một người nói có thể cần đồng thời thông tin ở nhiều thang; hai người cùng vùng vẫn cần được tách. Không chuẩn hóa giọng địa phương thành một giọng được xem là chuẩn: mục tiêu là giữ đúng nội dung và đặc điểm nguồn tham chiếu.

## 4. Kiểm tra metadata cục bộ trước khi thiết kế thí nghiệm

Đã đọc `data/VnSpeechMix/rendered/rendered_metadata.csv`. SHA-256:

```text
dd92aa326f5d18a62f93f77c2682a73c5f742bd3879c4e31bb77d9c5ed5fdfef
```

| Cặp vùng | Train | Valid | Test |
|---|---:|---:|---:|
| North–North | 3000 | 500 | 834 |
| Central–Central | 3000 | 500 | 834 |
| South–South | 3000 | 500 | 833 |
| North–Central | 3000 | 500 | 833 |
| North–South | 3000 | 500 | 833 |
| Central–South | 3000 | 500 | 833 |

Train đã cân bằng sáu cặp vùng. Không lấy việc thêm sampler cân bằng sáu nhóm làm đóng góp chính. Cân bằng số mẫu không bảo đảm độ khó, số người nói, giới tính hoặc chất lượng ghi âm cân bằng.

Metadata có region, province, speakerID và utterance ID theo từng nguồn. Đã quan sát:

- Không giao nhau về chuỗi utterance ID giữa các split.
- Không giao nhau về chuỗi speakerID giữa train–valid và train–test.
- Có **2 speakerID xuất hiện ở cả valid và test**; kiểm tra thêm cặp `(province, speakerID)` vẫn thấy giao nhau, tại QuangNgai và QuangBinh.

Đây là kết quả kiểm tra mã metadata, chưa chứng minh chắc chắn danh tính người nói thực. Cần xác minh quy ước ID và provenance trước khi tuyên bố hoàn toàn speaker-disjoint. Không âm thầm sửa benchmark. Nếu cần protocol strict-disjoint, tạo manifest mới có phiên bản và áp dụng giống nhau cho tất cả mô hình; báo cáo tách biệt với protocol phát hành.

Vị trí source_1/source_2 có tương quan với thứ tự vùng trong metadata. Mọi tác vụ phụ gắn nhãn nguồn phải dùng hoán vị PIT; tuyệt đối không mặc định đầu ra thứ nhất là nguồn thuộc một vùng nhất định.

## 5. Kiến trúc LTRR-MR đề xuất

### 5.1 Phần giữ nguyên

```text
Reconstruction decoder
  -> Y [B*J,128,Tp]
  -> ChannelLN + Conv1x1 128->64 + PReLU
  -> z [B*J,64,Tp]
  -> Conv-U_u và Conv-U_v độc lập
  -> u, v [B*J,64,Tp]
```

Giữ `J=2`, kernel Conv-U bằng 3, dropout 0,05, bottleneck 64. Các con số này lấy từ config hiện tại; chưa có bằng chứng chúng tối ưu.

### 5.2 Memory chạy ở nhiều tốc độ

Gọi `D_theta` là DenseDilatedFSMN1D hiện tại, giữ dilation `[1,2,4,8]`. Đề xuất tập rate `r in {1,4,16}`:

```text
v_r = P_r(v)
m_r = U_r(D_theta(v_r))
```

- `P_1` là identity; `P_4`, `P_16` là average pooling theo thời gian, kernel=stride=r.
- Cùng **một** đối tượng `D_theta` được gọi ba lần, không tạo ba bản tham số độc lập.
- `U_r` nội suy tuyến tính về đúng `Tp`, `align_corners=False`; nhánh r=1 không cần nội suy.
- Dùng mask chiều dài hợp lệ khi pooling và tính thống kê, không coi padding là tín hiệu. Quy định rõ chia cho số frame hợp lệ trong cửa sổ biên và xử lý cửa sổ rỗng.
- Với chiều dài bất kỳ, đệm đến bội số của r và lưu chiều dài trước pooling; nội suy/crop về đúng lưới chung. Kiểm tra biên bằng test riêng.

Ví dụ crop 4 giây, encoder `k=32, stride=8` cho `T0=7997`; separator pad `Tp=8000`:

| Rate r | Đầu vào memory | Bước thời gian ở memory | 31 ô thời gian quy đổi danh nghĩa |
|---|---|---|---|
| 1 | `[B*J,64,8000]` | 0,5 ms | 15,5 ms |
| 4 | `[B*J,64,2000]` | 2 ms | 62 ms |
| 16 | `[B*J,64,500]` | 8 ms | 248 ms |

Cột cuối là `31*r*0,5 ms`, diễn tả độ rộng của 31 ô thời gian. Không phải receptive field chính xác của toàn nhánh có Conv-U/pooling/nội suy, không phải thời lượng cố định của thanh điệu. Rate `[1,4,16]` là điểm khởi đầu để kiểm chứng khoảng ngữ cảnh khác nhau với chi phí vừa phải.

Pooling có nguy cơ làm mất thông tin nhanh và không phải bộ lọc chống alias lý tưởng; nhánh full-rate là đường giữ chi tiết. Chỉ thêm pooling không bảo đảm bảo toàn F0 hay chất giọng. Trọng số dùng chung cũng có thể gây xung đột vì một kernel được dùng ở nhiều đơn vị thời gian; cần theo dõi và có đối chứng nếu mô hình không học tốt.

### 5.3 Bộ chọn trọng số theo nguồn dự đoán

Với mỗi nguồn dự đoán j, tính thống kê theo thời gian từ `z_j`, bỏ padding:

```text
q_j = concat(mean_t(z_j), std_t(z_j))     # [128]
beta_j = softmax(W2 * SiLU(W1*q_j+b1)+b2) # [3]
m_j = sum_r beta_j[r] * m_jr             # [64,Tp]
```

MLP đề xuất `128 -> 32 -> 3`, dùng cùng trọng số cho mọi nguồn. Tính std với epsilon để đoạn hằng không gây gradient không hữu hạn. Bộ chọn dùng một bộ ba trọng số cho cả crop/utterance ở phiên bản đầu, chưa đổi trọng số từng frame.

Khởi tạo layer cuối bằng 0 để ba trọng số ban đầu bằng 1/3. Không cần thêm entropy loss ngay từ đầu; quan sát xem bộ chọn có suy biến về một nhánh hay không trước.

Router này chỉ dùng đặc trưng của mixture đã đi qua separator. Không sử dụng clean reference, nhãn vùng hoặc speakerID khi suy luận. Dùng chung head bảo toàn tính tương đương khi đổi thứ tự các nguồn ở chế độ eval; dropout cần được tắt khi kiểm tra tính chất đó.

MLP có `128*32+32+32*3+3 = 4.227` tham số nếu cả hai Linear có bias. Đây là phần tăng lý thuyết so với LTRR gốc khi pooling/nội suy không có tham số và không thêm normalization mới. Phải xác nhận lại bằng code khi triển khai.

Global statistics dễ học cả người nói, thiết bị và tiếng ồn thay vì phương ngữ. Khác biệt giữa crop 4 giây khi train và utterance dài khi test cũng có thể đổi trọng số. Phải kiểm tra các yếu tố này, không diễn giải router như một bộ nhận diện phương ngữ mặc định.

### 5.4 Phần hợp nhất và đầu ra

```text
h = z + Dropout(u * m)                   # [B*J,64,Tp]
deltaY = Conv1x1_64to128(ChannelLN(h))    # [B*J,128,Tp]
Y_new = Y + alpha * deltaY               # [B*J,128,Tp]
```

Giữ alpha khởi tạo 0,05 như cũ. Sau đó dùng nguyên OutputLayer và AudioDecoder hiện có. Không thêm block nhận dạng văn bản, nhãn thanh hay người nói vào đường suy luận.

Trong bản đầu, chi phí phần memory theo độ dài xấp xỉ `1+1/4+1/16=1,3125` lần memory cũ. Đây không phải tỷ lệ chi phí toàn mô hình hoặc VRAM. Tensor sau nội suy, router, backward và cách cài đặt đều tạo thêm chi phí; cần đo peak VRAM và thời gian trên GPU thực.

### 5.5 Biến thể ổn định hóa, chỉ thử riêng

Nếu diagnostics cho thấy `m` hoặc correction bất thường, thử:

```text
m_bounded = tanh(ChannelLN(m))
h = z + Dropout(u * m_bounded)
```

Giới hạn biên độ bộ điều biến trong [-1,1], không giới hạn toàn bộ u, deltaY hay alpha. Tanh có thể bão hòa, làm mất biên độ hữu ích; không khẳng định tốt hơn nếu chưa ablate. Đây là điều biến nội dung, không khôi phục speaker gate của SGLR cũ.

## 6. Làm thế nào để thực sự gắn với đa phương ngữ?

**Mức 1, bắt buộc:** đánh giá theo sáu cặp vùng, theo vùng của từng nguồn sau PIT, và nhóm kém nhất. Một architecture tổng quát phải chứng minh có ích trên các nhóm này trước khi gọi là cải tiến cho đa phương ngữ.

**Mức 2, tùy chọn sau khi mức 1 có kết quả:** huấn luyện auxiliary head dự đoán vùng từ feature từng nguồn. Đây là một nhánh nghiên cứu riêng, không thêm ngay vào phiên bản chính:

```text
L = L_separation_current + lambda_region * mean_j CE(p_j, region_ref[pi*(j)])
```

`pi*` phải là hoán vị PIT đã xác định từ đầu ra waveform cho sample đó; lambda chọn trên validation. Head chỉ dùng nhãn tham chiếu khi train; bỏ head khi inference. Loss phụ có thể giữ đặc trưng phương ngữ, nhưng cũng có thể học shortcut theo nguồn ghi âm hoặc làm giảm khả năng tách. Không đảm bảo auxiliary accuracy cao đồng nghĩa SI-SNRi cao.

**Mức 3, chỉ khi có chênh lệch nhóm rõ ràng:** cân nhắc mục tiêu nhóm như group DRO. Sáu nhóm có số lượng cân bằng vẫn có thể khác độ khó. Phải đối chiếu với empirical-risk minimization, điều chỉnh bằng validation và theo dõi overfit nhóm khó; group DRO không tự bảo đảm generalization. [Sagawa và cộng sự](https://arxiv.org/abs/1911.08731)

Không triển khai đồng thời mọi mức. Khuyến nghị khóa luận bắt đầu bằng mức 1; mức 2 hoặc 3 được chọn dựa trên kết quả, không cộng tất cả thành một mô hình khó giải thích.

## 7. Thí nghiệm đối chứng tối thiểu

| Mã | Mô hình | Câu hỏi |
|---|---|---|
| B0 | SepReformer baseline | Mốc so sánh không có LTRR. |
| B1 | LTRR gốc | Mức đóng góp của NoGate/LTRR đã có. |
| B2 | LTRR-MR, beta cố định 1/3 | Chỉ thêm nhiều tốc độ có ích không? |
| B3 | LTRR-MR, beta học được | Thích nghi theo đầu vào có ích hơn B2 không? |
| B4, tùy chọn | B3 + bounded modulation | Giới hạn modulator có lợi về độ ổn định/chất lượng không? |
| B5, tùy chọn riêng | B3 + auxiliary region loss | Giám sát vùng có ích hơn học từ separation loss không? |

Giữ chung config huấn luyện hiện tại: 16 kHz, crop 4 giây, encoder k32/s8, 128 feature, loss và lịch auxiliary loss, optimizer/scheduler, effective batch size, số epoch và cách chọn checkpoint. Không sửa riêng STFT loss cho B3 rồi quy toàn bộ cải thiện về module.

Mọi mô hình train từ đầu. Dùng cùng khởi tạo cho các phần backbone chung theo seed; với B1/B2/B3 có thể ghép cả trọng số các thành phần LTRR tương ứng nếu thiết kế tương thích. Router mới dùng RNG riêng hoặc kiểm soát seed để việc thêm tham số không âm thầm đổi shuffle/crop của dataset. Không load checkpoint LTRR đã train làm khởi tạo chỉ cho mô hình đề xuất.

Lộ trình tiết kiệm: smoke test và pilot ngắn để phát hiện lỗi; chạy B1/B2/B3 một seed để kiểm tra khả năng học; chọn thiết kế bằng validation; sau đó ưu tiên ít nhất ba seed cho B0/B1/B3 nếu tài nguyên cho phép. Pilot một vài epoch không đủ để kết luận mô hình tốt hơn. Nếu không đủ seed, công bố giới hạn thay vì coi một lần chạy là bằng chứng chắc chắn.

Nếu cần chứng minh lợi ích không chỉ đến từ tăng tính toán, bổ sung đối chứng LTRR cùng ngân sách tính toán hoặc số tham số. Muốn khẳng định tính chuyên biệt cho tiếng Việt, thử cùng module trên Libri2Mix với protocol thống nhất; lợi ích trên cả hai ngôn ngữ có thể là cải thiện tách tiếng tổng quát.

## 8. Chỉ số và phân tích cần báo cáo

- SI-SNRi và SDRi toàn test, theo đúng một implementation và quy tắc PIT cố định cho mọi mô hình.
- Mean và số mẫu của sáu cặp vùng; macro-average, nhóm thấp nhất và khoảng cách tốt nhất–thấp nhất. Không chỉ dùng khoảng cách nhóm vì làm nhóm tốt kém đi cũng có thể giảm khoảng cách.
- Metric theo vùng của từng nguồn phải gắn nhãn sau PIT; phân biệt với metric theo cặp vùng của mixture.
- So sánh cặp cùng vùng và khác vùng; kiểm soát hoặc ít nhất phân tầng SNR, giới tính và độ dài. Tương quan với vùng không chứng minh nguyên nhân là phương ngữ.
- Params, peak GPU memory, thời gian train/epoch, real-time factor inference; ghi GPU, batch size và độ dài dùng để đo.
- Alpha, tỷ lệ `||alpha*deltaY||/(||Y||+eps)`, thống kê m, gradient norm, tỷ lệ NaN/Inf; mean/entropy của beta theo rate và vùng, kiểm tra collapse.
- Tùy chọn: F0 error trên frame hữu thanh được căn chỉnh, kèm voiced/unvoiced error, lỗi octave và độ tin cậy của bộ trích F0. Không dùng một chỉ số F0 để khẳng định bảo toàn mọi thanh điệu/chất giọng.
- Tùy chọn: CER bằng một ASR cố định, kèm CER trên clean references để thấy sai lệch recognizer theo vùng. Không dùng CER làm bằng chứng duy nhất cho chất lượng tách.

So sánh paired theo cùng mixture khi phân tích chênh lệch. Mixtures có tái sử dụng utterance/người nói; bootstrap từng mixture độc lập có thể đánh giá thấp bất định. Cần xét cluster theo nguồn/người nói, hoặc ghi rõ giới hạn của khoảng tin cậy; độ biến thiên giữa các seed là một loại bất định khác. Quy tắc reuse nằm trong [repository VN-SpeechMix](https://github.com/quocnhut134/VN-SpeechMix).

Nếu muốn đo generalization sang tỉnh/vùng chưa thấy, phải thiết kế split riêng, kiểm soát nguồn/người nói và train lại tất cả đối chứng trên split đó. Test theo nhóm vùng trong split hiện có chỉ chứng minh hiệu quả trên các vùng đã được đại diện trong train. Dataset fully-overlapped cũng chưa đủ để kết luận hiệu quả trên hội thoại thật, overlap một phần hoặc điều kiện nhiễu/reverb mới.

## 9. Kiểm tra bắt buộc khi triển khai

1. Forward/backward với crop 64000 sample, các độ dài lẻ và rất ngắn; shape đầu ra bằng nguồn vào, gradient hữu hạn.
2. Mask padding đúng ở statistics/pooling, nội suy về đúng thời gian; không tính padding vào thống kê router.
3. Chỉ một bộ tham số memory dùng chung; kiểm tra thực tế params, MAC/VRAM và mức sử dụng GPU.
4. Hoán đổi hai feature nguồn ở đầu vào LTRR trong eval phải hoán đổi đầu ra tương ứng.
5. B2 là đường đối chứng trọng số cố định; không để router hoặc loss vùng vô tình hoạt động ở B2.
6. Eval chạy được chỉ với mixture; không đọc nhãn phương ngữ hay reference để tạo feature/router.
7. Paired initialization, checkpoint/resume và run contract nhận diện riêng mô hình mới; giữ nguyên run gốc.
8. Kiểm tra metadata/provenance đã nêu ở mục 4 trước khi phát biểu về split speaker-disjoint.

## 10. Đóng góp có thể bảo vệ và các rủi ro

Cách phát biểu phù hợp trước thực nghiệm: “Đề xuất và đánh giá một cơ chế tinh chỉnh phần dư với bộ nhớ đa tốc độ dùng chung trọng số, phối hợp thích nghi theo từng nguồn, nhằm khảo sát khả năng tách tiếng Việt đa phương ngữ trên VN-SpeechMix.”

Đóng góp dự kiến nằm ở thiết kế tích hợp nhẹ vào LTRR/SepReformer, thực nghiệm kiểm soát từng thành phần, và phân tích đa phương ngữ có sử dụng metadata. Không lấy đổi tên NoGate thành LTRR làm đóng góp mới.

Dense memory/Conv-U đã có nền tảng từ [MossFormer2](https://arxiv.org/abs/2312.11825). Đa thang, trọng số động và chia sẻ trọng số cũng đã xuất hiện trong tách tiếng, ví dụ [FurcaNeXt](https://arxiv.org/abs/1902.04891); convolution đa thang thích nghi còn được nghiên cứu cho [dialect identification](https://arxiv.org/abs/2108.07787). Các nguồn này là tiền lệ cần đối chiếu, không chứng minh thiết kế ở đây sẽ hiệu quả trên tiếng Việt. Chưa đủ cơ sở gọi đây là ý tưởng đầu tiên hoặc một thuật toán hoàn toàn mới.

Rủi ro chính: backbone đã có multi-scale/global attention nên lợi ích bổ sung có thể nhỏ; pooling làm mờ chi tiết; memory chia sẻ gặp xung đột giữa rate; router học shortcut hoặc collapse; thống kê toàn đoạn bị lệch giữa train và test; tăng chi phí mà không tăng chất lượng. Nếu B3 không hơn B2, bỏ router; nếu B2 không hơn B1 trong ngân sách tương đương, không giữ nhánh đa tốc độ chỉ vì kiến trúc trông phức tạp hơn.

Chỉ chốt mô hình mới sau khi kết quả validation, nhiều seed và chi phí thực đo ủng hộ. Không đặt trước mức cải thiện dB hay tuyên bố “tối ưu cho tiếng Việt”.
