# Giải thích kiến trúc SepReformer–LTRR bằng các khái niệm AI

Tài liệu giải thích trực tiếp mô hình trong [bộ sơ đồ ba trang](../figures/ltrr_architecture_complete.pdf). Mạch trình bày là **khái niệm AI → phép toán trong khối → ý nghĩa của phép toán → vai trò trong toàn mô hình**. Thông số và vị trí module theo code hiện tại.

[Bản thuyết trình 27 mục](LTRR_PRESENTATION_SCRIPT_VI.md) dùng để tra cứu chi tiết mũi tên và kích thước tensor. Bản này giúp hiểu bản chất các khối để tự giải thích bằng lời, thay vì chỉ đọc tên lớp và các con số.

## 1. Đặc trưng, không gian tiềm ẩn và việc học của mô hình

**Khái niệm AI: representation learning, latent representation, end-to-end learning.**

Representation learning là học cách biểu diễn dữ liệu để phục vụ nhiệm vụ. Dữ liệu âm thanh ban đầu là waveform: một chuỗi giá trị biên độ theo thời gian. Encoder biến nó thành nhiều kênh đặc trưng. Các kênh này tạo thành latent representation, tức biểu diễn trung gian bên trong mạng, chưa phải âm thanh của từng người nói.

Một đặc trưng là một giá trị mạng tính ra. Một kênh đặc trưng là chuỗi các giá trị như vậy theo thời gian. Thông tin thường được mã hóa bằng sự phối hợp nhiều kênh — distributed representation, hay biểu diễn phân tán. Không có quy định mỗi kênh tương ứng một khái niệm dễ gọi tên như “giọng nam”, “speaker 1” hoặc một tần số cố định.

**Hiểu bằng lời.** Waveform là dữ liệu thô. Đặc trưng là cách mạng tự tổ chức lại dữ liệu để làm nhiệm vụ tách giọng. “128 kênh” nghĩa là tại một frame có 128 giá trị mô tả, không phải 128 người nói.

End-to-end learning nghĩa là loss trên đầu ra truyền gradient về các phần có tham số để học cùng nhau. Gradient mô tả loss biến thiên như thế nào theo một giá trị hoặc trọng số; optimizer dùng nó để cập nhật mạng. Vì được tối ưu cho separation, encoder có thể học biểu diễn phù hợp với các khối xử lý và bộ tổng hợp phía sau.

**Các chiều tensor cần phân biệt:**

| Ký hiệu | Ý nghĩa | Ví dụ trong hình |
|---|---|---|
| B | Số hỗn hợp độc lập trong batch | 2 |
| J | Số nguồn trong mỗi hỗn hợp | 2 |
| N | Số mẫu waveform | 64000, tức 4 giây ở 16 kHz |
| C hoặc F | Số kênh đặc trưng | 256, 128 hoặc 64 tùy vị trí |
| T | Số frame của chuỗi đặc trưng | Thay đổi theo tầng |
| BJ | Gộp batch với chiều nguồn | 4 nhánh từ 2 hỗn hợp |

Với tensor [B,C,T], vector X[b,:,t] chứa C giá trị của mẫu b tại frame t. **Channel mixing** phối hợp các giá trị trong vector này. **Temporal mixing** phối hợp thông tin giữa các frame. Đây là hai công việc khác nhau mà nhiều khối trong mô hình kết hợp.

## 2. AudioEncoder — học các bộ lọc để biểu diễn waveform

**Khái niệm AI: learnable filter bank, convolution, parameter sharing, nonlinear activation.**

Learnable filter bank là một tập bộ lọc có trọng số được học. Conv1d trượt bộ lọc trên waveform; tại mỗi vị trí, mỗi bộ lọc lấy tổng có trọng số của các mẫu trong cửa sổ để tạo một giá trị đầu ra. Cùng trọng số được dùng tại mọi vị trí — parameter sharing theo thời gian.

Trong code có 256 bộ lọc, kernel 32, stride 8, không bias và không padding trong Conv1d. Kernel là độ rộng cửa sổ trực tiếp; stride là bước dịch giữa các lần tính. Sau convolution là GELU, một activation phi tuyến:

    [2,64000] → thêm chiều mono → [2,1,64000]
    T0 = floor((64000 − 32)/8) + 1 = 7997
    Conv1d 1→256 → GELU → [2,256,7997]

**Hiểu bằng lời.** Mạng học 256 cách phân tích các đoạn âm thanh rất ngắn. Các cửa sổ chồng lấn nên cùng một mẫu waveform có thể ảnh hưởng nhiều frame. Kết quả vẫn mô tả hỗn hợp, chưa tách thành hai nguồn.

**Vì sao cần phi tuyến?** Chỉ nối các phép affine mà không có phép phi tuyến ở giữa thì chúng có thể gộp thành một phép affine duy nhất. Activation giúp quan hệ đầu vào–đầu ra phong phú hơn. GELU không phải một ngưỡng cố định dùng để phân loại “tiếng nói” và “nhiễu”.

**Vai trò.** Tạo miền đặc trưng để separator xử lý. Chiều thời gian ngắn hơn waveform nhưng số channel tăng; không thể kết luận tổng số phần tử đã giảm đúng bằng stride. Encoder cũng không phải STFT với các bin tần số cố định.

## 3. FeatureProjector — chuẩn hóa và học phép phối hợp các kênh

**Khái niệm AI: normalization, learned projection, channel mixing.**

FeatureProjector dùng GroupNorm(1,256), sau đó Conv1d kernel 1 chiếu 256 xuống 128 kênh. Normalization điều chỉnh thang giá trị theo thống kê; projection là ánh xạ học được sang một không gian đặc trưng khác.

GroupNorm một nhóm trên [B,C,T] tính trung bình/phương sai trên C và T của từng mẫu b, rồi dùng scale/bias học được. Nó không trộn hai hỗn hợp trong batch để tạo đặc trưng và không phải chỉnh âm lượng waveform.

Conv1d kernel 1 chỉ phối hợp channel tại cùng frame. Với convolution không bias của khối này:

    đặc_trưng_mới[k,t] = tổng_c W[k,c] × đặc_trưng_cũ[c,t]

**Hiểu bằng lời.** Mạng học 128 cách phối hợp từ 256 giá trị tại mỗi frame. Nó không đơn giản bỏ 128 kênh cuối và không chọn cứng “128 thông tin quan trọng nhất”. Projection được học cùng loss, không phải PCA tính sẵn.

**Vai trò.** Đưa dữ liệu về độ rộng 128 của separator và giới hạn độ rộng tính toán. Phép giảm kênh có thể làm mất thông tin trong nhánh được chiếu; không mặc định đó chỉ là thông tin thừa.

Sau projection, separator pad 7997 lên 8000 frame để chia hết cho 16. Padding phục vụ sự khớp kích thước qua bốn lần downsampling, không tạo âm thanh mới. Giá trị ở phần pad có thể thay đổi sau xử lý; cuối đường chính crop lại T0.

## 4. Separation Encoder — học biểu diễn đa độ phân giải

**Khái niệm AI: hierarchical representation, multiscale processing, downsampling, skip connection.**

Hierarchical representation là biểu diễn được biến đổi qua nhiều tầng. Multiscale processing ở đây nghĩa là xử lý chuỗi ở nhiều độ phân giải temporal. Mỗi tầng encoder có hai cặp Global–Local với tham số riêng rồi DownConv; số channel vẫn là 128:

    E0: 8000 → 4000 frame
    E1: 4000 → 2000 frame
    E2: 2000 → 1000 frame
    E3: 1000 →  500 frame

DownConv là depthwise Conv1d k=5, stride=2, padding=2, rồi BatchNorm và GELU. Nó học bộ lọc trên từng kênh và trả chuỗi thưa vị trí hơn. Nó không chỉ bỏ mỗi frame thứ hai và không tự trộn các kênh.

**Hiểu bằng lời.** Mạng lần lượt xử lý tín hiệu ở những mức thời gian thô hơn. Chuỗi ngắn hơn giúp các tầng sâu làm việc với ít vị trí hơn, nhưng có thể mất chi tiết. Vì vậy, mạng giữ đặc trưng trước mỗi DownConv làm skip connection — đường đưa biểu diễn trung gian sang decoder.

Skip không tự sửa âm thanh; nó cấp thêm đầu vào để decoder kết hợp thông tin ở mức tương ứng với thông tin đi qua đường sâu. Không gán cố định các tầng thành “âm vị → từ → câu”, vì code không quy định như vậy.

Sau E3 còn bottleneck_G: hai cặp Global–Local ở 500 frame, không downsample. Đây là temporal bottleneck, khác bottleneck giảm channel 128→64 trong LTRR.

## 5. Global Block/EGA — tổng hợp ngữ cảnh bằng quan hệ phụ thuộc nội dung

**Khái niệm AI: self-attention, query–key–value, contextual representation, positional encoding, gating.**

Self-attention cho một vị trí sử dụng thông tin từ các vị trí khác trong cùng chuỗi bằng các trọng số phụ thuộc dữ liệu. Contextual representation là biểu diễn đã được cập nhật bằng ngữ cảnh, không còn chỉ mô tả riêng một vị trí ban đầu.

EGA dùng adaptive average pooling đưa chuỗi về độ dài bottleneck, bằng 500 trong ví dụ hiện tại. LayerNorm và các phép chiếu học được tạo query Q, key K và value V:

| Thành phần | Vai trò toán học | Cách hiểu |
|---|---|---|
| Query | Đối chiếu với các key để tạo điểm quan hệ | Biểu diễn dùng để tìm thông tin phù hợp |
| Key | Tham gia tính mức liên quan với query | Biểu diễn dùng để quyết định nơi cần đọc |
| Value | Được tổng hợp bằng các trọng số attention | Nội dung được sử dụng để cập nhật |

“Truy vấn” không phải câu hỏi bằng ngôn ngữ. Điểm query–key được kết hợp với thông tin vị trí tương đối, scale và softmax. Softmax tạo trọng số không âm có tổng bằng 1 theo các vị trí được đọc, trước dropout trên attention. Trọng số đó dùng để lấy tổng có trọng số của V.

**Hiểu bằng lời.** Khi xử lý một vị trí, mạng học xem nên sử dụng thông tin ở đâu và với mức đóng góp bao nhiêu. Hai đoạn âm thanh khác nhau có thể tạo quan hệ attention khác nhau dù dùng cùng tham số mô hình. Điều này khác convolution dùng cùng kernel đã học dọc chuỗi.

Multi-head attention có 8 head, mỗi head 16 chiều từ F=128. Các phép chiếu khác nhau cho phép học nhiều kiểu quan hệ, không có quy định mỗi head phụ trách một speaker. Relative positional encoding đưa độ lệch trước/sau vào điểm attention để phép tổng hợp có thông tin về thứ tự temporal.

**Vì sao pooling trước attention?** Attention theo cặp vị trí cần ma trận với hai chiều temporal. Một ma trận 500×500 có 250.000 phần tử, còn 8000×8000 có 64.000.000. Đây là so sánh kích thước ma trận, không phải tuyên bố toàn mô hình nhanh hơn theo cùng tỷ lệ.

Sau attention, projection, dropout và LayerScale, kết quả được nội suy về chiều dài của tầng. Nhánh LayerNorm–Linear–Sigmoid từ đặc trưng gốc tạo gate theo frame/channel; gate nhân vào thông tin attention trước khi cộng residual. Gating là điều tiết mức đóng góp của một biểu diễn bằng biểu diễn khác. Nó kết hợp thông tin global với đặc trưng ở độ phân giải gốc, không đảo ngược chính xác điều pooling đã làm mất.

Sau EGA còn GCFN để tiếp tục biến đổi đặc trưng.

## 6. Local Block/CLA — mô hình hóa lân cận temporal

**Khái niệm AI: local dependency, temporal convolution, inductive bias, receptive field.**

Local dependency là quan hệ giữa các vị trí gần nhau. Inductive bias là kiểu quan hệ mà kiến trúc ưu tiên học. Convolution ưu tiên việc cùng một phép lọc có thể hữu ích ở nhiều vị trí và các giá trị trong cửa sổ có liên hệ với nhau.

CLA thực hiện LayerNorm → Linear 128→256 → GLU còn 128 → depthwise k=65 → Linear 128→256 → BatchNorm → GELU → Linear 256→128 → Dropout → LayerScale → residual. GCFN tiếp tục xử lý output.

**GLU là gì?** Gated Linear Unit chia tensor thành hai phần A và G, rồi tính A ⊙ sigmoid(G). Một phần chứa giá trị, phần còn lại điều tiết chúng. Giảm số kênh một nửa là hệ quả của cách chia này, không phải cắt bỏ tùy ý nửa tensor.

**Hiểu bằng lời.** CLA xử lý đặc trưng tại một vị trí cùng lân cận để học các mẫu biến thiên temporal. Kernel 65 đọc 65 frame của tầng hiện tại, không phải 65 mẫu waveform. Ở các tầng có độ phân giải khác nhau, cùng 65 frame tương ứng khoảng thời gian vật lý khác nhau.

CLA không tạo ma trận QKᵀ và softmax như EGA. Nó thực hiện xử lý local bằng convolution và điều tiết GLU. Global và Local bổ sung hai kiểu xử lý: tổng hợp quan hệ phụ thuộc nội dung trên ngữ cảnh rộng và lọc các mẫu lân cận bằng cấu trúc convolution.

## 7. GCFN — biến đổi channel có xét đến frame gần nhau

**Khái niệm AI: feed-forward network, feature expansion, gated activation, pre-normalization, LayerScale.**

Feed-forward network thực hiện biến đổi theo đồ thị tính toán đi về phía trước, không cần hidden state tuần hoàn như RNN. Trong các block Transformer, FFN giúp phối hợp và biến đổi vector đặc trưng sau attention. GCFN thêm convolution temporal và gating vào kiểu biến đổi này.

    LayerNorm → Linear 128→768
    → depthwise Conv1d k=3 trên 768 kênh
    → GLU 768→384 → Dropout
    → Linear 384→128 → Dropout → LayerScale → cộng input

Feature expansion là mở rộng không gian trung gian để thực hiện các phép biến đổi trước khi quay về độ rộng ban đầu. Linear 128→768 học các tổ hợp channel, không lặp mỗi kênh sáu lần.

**Hiểu bằng lời.** Sau khi nhận thông tin từ ngữ cảnh, mạng còn cần phối hợp lại các giá trị mô tả để tạo biểu diễn phù hợp hơn cho bước tiếp theo. GCFN làm việc đó trong không gian rộng hơn, đồng thời dùng lân cận temporal và GLU trước khi đưa về 128 chiều.

Pre-normalization nghĩa là chuẩn hóa trước nhánh biến đổi. LayerScale có hệ số học được theo chiều đặc trưng, khởi tạo 1e−5, để scale nhánh trước phép cộng. Residual giữ đường input trực tiếp. Những cơ chế này hỗ trợ tổ chức việc cập nhật trong mạng sâu, không phải bảo đảm loss luôn giảm.

Trong SpkAttention, code đã đưa tensor về chuỗi temporal của từng nguồn trước GCFN, nên kernel 3 vẫn chạy theo time, không chạy theo hai speaker.

## 8. Speaker Split — học các biểu diễn theo nguồn

**Khái niệm AI: source-specific representation, early branching, tensor reshape.**

Speaker Split học ánh xạ đặc trưng hỗn hợp thành J nhóm đặc trưng. Với F=128, J=2: Conv1d 128→1024, GLU còn 512, Conv1d 512→256, rồi reshape và GroupNorm đưa [B,256,T] thành [BJ,128,T].

Source-specific representation là biểu diễn được tổ chức theo nhánh để phục vụ việc ước lượng nguồn tương ứng. Vai trò đó được học dưới loss, không phải chứng nhận nhánh đã loại sạch nguồn còn lại.

**Hiểu bằng lời.** Mạng cần chuẩn bị hai đầu vào khác nhau cho hai đầu ra. Các convolution và GLU học cách tạo hai nhóm đó. Reshape chỉ đổi cách tổ chức các giá trị; chính reshape không có năng lực tách giọng.

Một split module được dùng cho bốn skip và bottleneck. Cùng trọng số nhưng đầu vào ở các tầng khác nhau nên giá trị đầu ra vẫn khác nhau. BJ=4 nghĩa là hai hỗn hợp, mỗi hỗn hợp hai nhánh, không phải tách bốn người trong mỗi hỗn hợp.

**Vai trò.** Early split tạo nhánh nguồn trước reconstruction decoder để các tầng sau có thể xử lý chúng. Module không cần biết tên người nói hoặc nhận một đoạn giọng tham chiếu.

## 9. Reconstruction Decoder — tăng độ phân giải và hợp nhất đặc trưng

**Khái niệm AI: upsampling, feature fusion, skip connection, shared weights.**

Từ [BJ,128,500], decoder dùng nearest interpolation khớp chiều dài skip, concat hai tensor theo channel thành 256 rồi dùng Conv1d k=1 về 128. Sau fusion là ba bộ Global–Local–CS. D0…D3 có độ dài output 1000, 2000, 4000, 8000.

Feature fusion là kết hợp thông tin từ nhiều biểu diễn. Concat giữ các nhóm kênh cạnh nhau; projection học cách phối hợp. Cộng trực tiếp sẽ gộp các cặp giá trị với hệ số 1 trước phép xử lý tiếp theo. Đó là hai cách tổ chức thông tin khác nhau.

**Hiểu bằng lời.** Nhánh sâu cung cấp biểu diễn đã qua nhiều bước xử lý; skip cung cấp biểu diễn từ encoder ở độ phân giải tương ứng. Decoder học cách sử dụng cả hai. Nearest interpolation tự nó không học phục hồi chi tiết, chỉ đưa tensor đến chiều dài phù hợp.

Shared weights giữa các nguồn nghĩa là cùng một hàm tham số được áp dụng cho các nhánh. Hai input khác nhau có thể cho hai output khác nhau. Không có nghĩa mọi tầng decoder dùng cùng bộ trọng số; các block nối tiếp có tham số riêng. Chia sẻ trọng số cũng không làm chi phí xử lý hai nhánh bằng chi phí một nhánh.

Decoder đang tái cấu trúc đặc trưng, chưa tổng hợp waveform. Nó không phải phép đảo toán học chính xác của encoder.

## 10. Cross-speaker Attention — cho các nhánh nguồn trao đổi thông tin

**Khái niệm AI: attention axis, cross-source interaction, information exchange.**

Ý nghĩa attention phụ thuộc chiều nào được xem là chuỗi. EGA chạy theo temporal positions; SpkAttention tổ chức [BJ,F,T] thành [BT,J,F], để attention chạy theo J nguồn trong cùng hỗn hợp, tại cùng thời điểm.

**Hiểu bằng lời.** Hai nhánh không chỉ được xử lý riêng. Một nhánh có thể dùng đặc trưng của nhánh kia để cập nhật khi thông tin nguồn còn có thể lẫn nhau. Đây là cơ chế phối hợp, không phải thao tác cứng biết sẵn phải chuyển thành phần nào từ người này sang người kia.

Với J=2, ma trận attention của mỗi head theo nguồn là 2×2. Không có việc trộn các hỗn hợp độc lập trong batch. Sau attention và residual, tensor được đưa về chuỗi temporal của từng nguồn rồi qua GCFN.

CS không phải classifier danh tính, không ép hai đặc trưng trực giao và không yêu cầu mọi cập nhật phải làm hai nhánh khác nhau hơn. Nó bổ sung tương tác trực tiếp giữa nguồn bên cạnh xử lý temporal trong mỗi nguồn.

## 11. LTRR — học phần cập nhật cho biểu diễn cuối separator

**Khái niệm AI: residual learning, latent feature refinement, learnable residual scaling.**

Residual learning biểu diễn output dưới dạng input cộng một nhánh cập nhật học được:

    Y′ = Y + α ΔY(Y)
    Y, Y′: [BJ,128,Tp]

ΔY là hàm của Y được tính bởi các lớp trong LTRR. α là scalar học được, khởi tạo 0,05. Latent feature refinement nghĩa là việc cập nhật diễn ra trên đặc trưng nguồn, trước khi tổng hợp waveform.

**Hiểu bằng lời.** Nhánh mới không phải tái tạo toàn bộ biểu diễn chỉ qua các lớp của nó: Y vẫn có đường đi trực tiếp. Nhánh bổ sung tạo giá trị để cộng vào. Tuy gọi là “hiệu chỉnh”, nó không nhận một tensor lỗi đặc trưng đúng từ bên ngoài; việc cập nhật được học gián tiếp từ loss trên đầu ra âm thanh.

LTRR đặt sau decoder cuối vì tại đây tensor đã có chiều nguồn và đã trở về Tp. Nó bổ sung khả năng biến đổi mà giữ giao diện [BJ,128,Tp] cho OutputLayer. Vị trí này là lựa chọn cấu trúc; muốn kết luận tốt hơn đặt ở vị trí khác cần ablation.

    z = PReLU(PWConv_128→64(ChannelLN(Y)))
    u = ConvU_u(z)
    v = ConvU_v(z)
    m = DenseDilatedMemory(v)
    h = z + Dropout(u ⊙ m)
    ΔY = PWConv_64→128(ChannelLN(h))
    Y′ = Y + α ΔY

Chỉ có một LTRR sau D3. Các head phụ không đi qua nó. Module chia sẻ tham số giữa các nguồn nhưng không trực tiếp trộn chiều BJ.

## 12. Bottleneck channel — tính phần cập nhật trong không gian hẹp hơn

**Khái niệm AI: channel bottleneck, dimensionality reduction, learned projection.**

Bottleneck là một phần đường xử lý có độ rộng biểu diễn nhỏ hơn phần xung quanh. LTRR chiếu F=128 xuống Cb=64 để phần lớn phép biến đổi bổ sung diễn ra trong không gian 64 chiều tại mỗi frame.

Pointwise convolution học các tổ hợp của 128 kênh đầu vào, cộng bias. Nó không chỉ lấy 64 kênh đầu tiên và không bảo đảm chọn đúng “64 thông tin quan trọng nhất”. Đây cũng không phải downsampling: Tp không đổi.

**Hiểu bằng lời.** Nhánh refinement dùng một không gian làm việc hẹp hơn. Mạng học cách phối hợp các giá trị để sử dụng trong nhánh này. Độ rộng thấp hơn giới hạn quy mô các lớp bổ sung, nhưng cũng có thể hạn chế những gì nhánh ấy biểu diễn.

Residual ngoài cho Y 128 kênh đi trực tiếp đến output, nên toàn kết quả không chỉ dựa vào bottleneck. Tuy vậy, nó không làm phép chiếu 128→64 trở thành không mất thông tin. Cb=64 là lựa chọn hiện tại, chưa phải giá trị đã chứng minh tối ưu.

## 13. ChannelLayerNorm và PReLU — chuẩn bị giá trị cho nhánh biến đổi

**Khái niệm AI: feature normalization, affine transformation, nonlinear activation.**

ChannelLayerNorm chuẩn hóa vector channel tại từng frame của từng nhánh nguồn. Code transpose [BJ,C,T] thành [BJ,T,C], LayerNorm trên C rồi đổi lại:

    LN(a) = γ ⊙ (a − μ) / sqrt(σ² + ε) + β

μ, σ² lấy từ các kênh ở frame đang xét. γ, β là scale/bias học được, khác α của LTRR; ε=1e−8 hỗ trợ ổn định phép chia khi phương sai nhỏ. Không có việc lấy thêm frame khác hoặc trộn các speaker để tính thống kê này.

**Hiểu bằng lời.** Khối điều chỉnh thang giá trị của các đặc trưng trước khi xử lý tiếp, đồng thời cho mạng học lại mức scale và độ lệch phù hợp. Sau affine, output không bắt buộc còn trung bình 0 và phương sai 1.

PReLU sau projection giữ phần dương và nhân phần âm với một độ dốc học được. Nó tạo phi tuyến mà không luôn xóa mọi giá trị âm. Mỗi PReLU() hiện có một hệ số độ dốc dùng chung trong module đó. Giá trị âm không mặc định là nhiễu, giá trị dương không mặc định là tiếng nói.

| Normalization | Các phần tử dùng để tính thống kê trong triển khai này |
|---|---|
| ChannelLayerNorm | Channel của một frame, một nhánh nguồn |
| GroupNorm(1,C) trên [B,C,T] | Channel và time của một mẫu |
| BatchNorm1d(C) | Batch và time cho từng channel khi train; eval mặc định dùng thống kê tích lũy |

Ba khái niệm này không thể thay tên cho nhau chỉ vì đều là chuẩn hóa.

## 14. Conv-U — tách việc trộn kênh và lọc thời gian

**Khái niệm AI: pointwise convolution, depthwise convolution, regularization, residual connection.**

    ConvU(a) = a + Dropout(DWConv3(SiLU(PWConv1(ChannelLN(a)))))

Input/output đều [BJ,64,Tp]. Bên trong không có downsampling hay upsampling; tên Conv-U không có nghĩa đây là U-Net thu nhỏ.

**Pointwise convolution:** kernel 1 phối hợp 64 channel tại cùng frame thành 64 channel mới. Đây là channel mixing, chưa đọc thêm vị trí temporal.

**SiLU:** tính a×sigmoid(a) trên từng giá trị sau pointwise. Đây là phi tuyến trơn. Vì còn nhân với a, output không bị ép vào [0,1] như sigmoid thuần.

**Depthwise convolution:** groups=64 nên mỗi channel có bộ lọc temporal riêng. Kernel 3 đọc t−1, t, t+1; padding 1 giữ độ dài. Riêng depthwise không phối hợp channel, nhưng toàn Conv-U đã có khả năng đó nhờ pointwise.

**Hiểu bằng lời.** Trước tiên mạng phối hợp các giá trị mô tả ở cùng thời điểm. Sau đó, từng channel đã phối hợp được xử lý cùng các giá trị lân cận của chính channel ấy. Khối phân công rõ hai việc: kết hợp loại đặc trưng và xử lý biến thiên temporal.

**Vì sao depthwise ít tham số?** Riêng convolution k=3 có bias: depthwise 64 channel có 64×3+64=256 tham số; convolution thường 64→64 có 64×64×3+64=12352. Nó giảm tham số của phép lọc temporal bằng cách hạn chế mỗi channel đọc mọi channel khác. Pointwise đảm nhiệm trộn kênh. Không suy từ tỷ lệ này ra tốc độ toàn mô hình.

**Dropout:** khi train, ngẫu nhiên đặt các phần tử kích hoạt thành 0 với p=0,05 và scale phần được giữ. Đây là regularization, tạo nhiễu huấn luyện nhằm hạn chế quá phụ thuộc một số kích hoạt. Eval thì dropout là phép đồng nhất. Nó không xóa neuron vĩnh viễn, không thay đổi shape và không bảo đảm tự loại overfitting.

**Residual:** cộng a trực tiếp với nhánh biến đổi, giúp output sử dụng cả biểu diễn đang có và phần cập nhật. Residual này nằm trong mỗi Conv-U, khác residual z và residual Y ở các cấp ngoài.

## 15. Hai nhánh Conv-U — hai phép biến đổi học độc lập của cùng input

**Khái niệm AI: parallel feature transformations, independent parameterization.**

Hai Conv-U cùng nhận z nhưng có hai bộ trọng số. Chúng học u=fᵤ(z) và v=fᵥ(z). u đi trực tiếp đến phép nhân; v đi qua memory để tạo m trước khi tương tác với u.

**Hiểu bằng lời.** Mạng tạo hai cách biến đổi từ cùng một biểu diễn. Một cách được dùng trực tiếp; cách còn lại tiếp tục tổng hợp ngữ cảnh rồi tham gia điều chỉnh đóng góp của nhánh thứ nhất. Cùng cấu trúc không có nghĩa output hai nhánh luôn giống nhau.

u và v không phải speaker 1 và speaker 2: mỗi nguồn đều đi qua cả hai nhánh. Không có loss riêng buộc u học “nội dung”, v học “danh tính”, hoặc buộc chúng chứa hai loại thông tin hoàn toàn độc lập. Sự khác nhau được cho phép bởi tham số riêng và được học dưới mục tiêu chung.

## 16. ffn_in của memory — chuẩn bị biểu diễn để tái sử dụng

**Khái niệm AI: feature re-encoding, feed-forward processing, feature reuse.**

    x0 = PReLU(PWConv_64→64(ChannelLN(v)))
    x0: [BJ,64,Tp]

Re-encoding là tạo một biểu diễn mới từ biểu diễn đang có để phục vụ bước tiếp theo. Normalization điều chỉnh thang giá trị; pointwise phối hợp channel; PReLU tạo phi tuyến. Vì normalization theo frame và kernel bằng 1, ffn_in không tự mở rộng phạm vi temporal so với v.

**Hiểu bằng lời.** Memory chuẩn bị x0 rồi giữ nó làm đầu vào chung cho bốn tầng. Tầng sâu vẫn có đường trực tiếp đến biểu diễn chuẩn bị ban đầu. x0 không phải bản sao giữ nguyên v; nó đã được biến đổi.

## 17. Dense connection — tái sử dụng nhiều state

**Khái niệm AI: dense connectivity, feature reuse, concatenation, channel compression.**

Dense connectivity ở đây nghĩa là tầng sau nhận x0 và mọi state đã tính trước đó. Các tensor được concat theo channel:

    c1 = [x0]             :  64 kênh
    c2 = [x0, y1]         : 128 kênh
    c3 = [x0, y1, y2]     : 192 kênh
    c4 = [x0, y1, y2, y3] : 256 kênh

Mỗi cᵢ qua pointwise riêng để còn 64 channel trước memory layer. Projection vừa phối hợp các state vừa giới hạn độ rộng phép lọc temporal tiếp theo.

**Hiểu bằng lời.** Tầng thứ ba không chỉ dựa vào kết quả tầng thứ hai: nó có thể sử dụng trực tiếp cả biểu diễn ban đầu lẫn tầng thứ nhất. Mạng được cung cấp những biểu diễn ở nhiều bước xử lý để học cách phối hợp.

Concat đặt các giá trị cạnh nhau và tăng chiều channel; residual cộng các giá trị cùng vị trí của tensor cùng shape. Dense connection tạo thêm đường cho đặc trưng và gradient, nhưng cần lưu state và các projection có input rộng dần. Projection sau concat vẫn có thể nén thông tin.

Các tầng chạy tuần tự y1→y2→y3→y4. Đường nối đến tầng sau không phải vòng lặp recurrent hoặc đưa y4 quay lại tính y1.

## 18. Dilated convolution — mở rộng receptive field mà giữ số tap

**Khái niệm AI: dilation, receptive field, multiscale context, length preservation.**

Receptive field của một output là tập các vị trí input có thể ảnh hưởng đến nó qua cấu trúc tính toán. Dilation tăng khoảng cách các vị trí kernel đọc. Kernel 3, dilation d đọc t−d, t, t+d của tensor ngay trước nó.

**Hiểu bằng lời.** Bộ lọc vẫn dùng ba giá trị temporal nhưng có thể đọc ba giá trị xa nhau hơn. Dilation không đồng nghĩa bỏ bớt output: stride=1 vẫn tạo giá trị tại mọi vị trí đầu ra. Padding=d giữ chiều dài với kernel 3.

Mỗi tầng thực hiện concat → pointwise → depthwise dilated convolution → PReLU → Dropout:

| Tầng | Projection | Dilation, padding | Vị trí đọc trên input của kernel | Shape output |
|---|---|---|---|---|
| y1 | 64→64 | 1, 1 | t−1, t, t+1 | [BJ,64,Tp] |
| y2 | 128→64 | 2, 2 | t−2, t, t+2 | [BJ,64,Tp] |
| y3 | 192→64 | 4, 4 | t−4, t, t+4 | [BJ,64,Tp] |
| y4 | 256→64 | 8, 8 | t−8, t, t+8 | [BJ,64,Tp] |

**Vì sao nhiều dilation?** Các tầng được tổ chức để sử dụng nhiều khoảng cách temporal, còn dense connection đưa các biểu diễn trước đến tầng sau. Đây là multiscale context trong nhánh memory; không có quy định y1 học âm ngắn, y4 học từ hoặc câu.

**Receptive field riêng và tích lũy khác nhau.** Kernel ở y4 trải từ t−8 đến t+8 trên p4, nhưng p4 đã chứa thông tin qua các state trước. Phạm vi tối đa tính từ x0 của y1/y2/y3/y4 rộng 3/7/15/31 frame. Bán kính tích lũy là 1, 1+2, 1+2+4, 1+2+4+8. Đây là phụ thuộc có thể có theo cấu trúc, không đo mức ảnh hưởng thực tế của mỗi vị trí sau khi train.

Grid frame ở LTRR tương ứng 8 mẫu tại 16 kHz, tức 0,5 ms. Không dùng 31 frame của memory để kết luận phạm vi toàn mô hình: Y đã qua global attention, v đã qua Conv-U. Padding đối xứng cho đọc cả trước và sau, nên thiết kế hiện tại không causal.

## 19. out_proj của memory — tổng hợp các mức xử lý

**Khái niệm AI: multilevel feature aggregation, learned fusion.**

    q = Concat_channel(y1,y2,y3,y4) : [BJ,256,Tp]
    m = PWConv_256→64(q)           : [BJ,64,Tp]

Feature aggregation là tổng hợp nhiều biểu diễn thành một output. Pointwise học cách phối hợp channel từ bốn state tại cùng frame. x0 không được concat trực tiếp ở bước cuối, dù đã là input của mọi tầng.

**Hiểu bằng lời.** Memory không chỉ giữ kết quả tầng cuối; nó cho phép dùng trực tiếp các kết quả ở nhiều bước xử lý. Phép tổng hợp cũng không phải trung bình bốn state: mức phối hợp được học.

Trọng số pointwise được dùng chung dọc time, không có mạng riêng sinh ma trận fusion cho từng frame. m vẫn phụ thuộc nội dung vì các state phụ thuộc input. Sau out_proj không có activation, softmax hay residual cộng v, nên m có thể âm hoặc lớn hơn 1.

“Memory” chỉ việc tổng hợp ngữ cảnh temporal trong forward. Nó không lưu kho giọng nói hay mang hidden state từ utterance trước sang utterance sau. Tên lớp DenseDilatedFSMN1D được chú thích FSMN-like; các phép toán cụ thể ở trên mới xác định hành vi triển khai.

## 20. u ⊙ m — tương tác nhân để điều biến đặc trưng

**Khái niệm AI: Hadamard product, multiplicative interaction, feature modulation.**

Tích Hadamard là nhân từng phần tử: r[b,c,t]=u[b,c,t]×m[b,c,t]. m được tính từ một nhánh của z nên mức đóng góp của u phụ thuộc một biểu diễn khác cũng phụ thuộc input. Đây là multiplicative interaction.

Feature modulation là điều chỉnh giá trị đặc trưng bằng một tín hiệu khác. Trong LTRR, m có thể giảm, tăng hoặc đảo dấu đóng góp từ u; nó không bị ép thành xác suất hay hệ số suy giảm trong [0,1].

**Ví dụ số học minh họa, không phải giá trị đo từ model:**

| u | m | u×m | Ý nghĩa |
|---|---|---|---|
| 2 | 0,1 | 0,2 | Giảm độ lớn |
| 2 | 2 | 4 | Tăng độ lớn |
| 2 | −1 | −2 | Đảo dấu |

Sau tích là Dropout rồi cộng z: h=z+Dropout(u⊙m). Đường z giữ biểu diễn bottleneck trực tiếp; residual này không nằm trong chính memory.

**Ý nghĩa với việc học.** Ở riêng phép nhân, gradient theo u tỷ lệ với m, gradient theo m tỷ lệ với u. Hai nhánh ảnh hưởng nhau cả ở output lẫn đường gradient. Đó không tự động là lợi thế ổn định: giá trị quá nhỏ có thể làm yếu một đường gradient, quá lớn có thể làm tăng biên độ.

Tên NoGate cũ nói đến bỏ speaker-guided gate của thiết kế cải tiến, không xóa GLU, gate EGA hoặc tích u⊙m. LTRR chia sẻ trọng số giữa các nguồn nhưng không thực hiện tương tác trực tiếp giữa chúng; CS trong backbone đảm nhiệm kiểu tương tác đó.

## 21. Output projection và α — khớp chiều và điều chỉnh phần cập nhật

**Khái niệm AI: dimension matching, residual scaling, identity path, gradient flow.**

h:[BJ,64,Tp] qua ChannelLayerNorm và pointwise 64→128 để tạo ΔY. Dimension matching là làm phần cập nhật có cùng shape với Y để cộng từng phần tử. Projection học 128 tổ hợp từ 64 channel, không lặp đôi tensor.

α học được điều chỉnh mức đóng góp chung của nhánh. Không qua sigmoid/clamp nên có thể âm, gần 0 hoặc lớn hơn 1. Khởi tạo 0,05 làm nhỏ hệ số ban đầu, nhưng độ lớn αΔY còn phụ thuộc ΔY.

**Hiểu bằng lời.** Nhánh LTRR chuyển phần bổ sung về đúng “định dạng” 128 channel rồi cộng với đặc trưng separator. Đường Y là identity path, tức truyền input trực tiếp. α điều chỉnh toàn nhánh cập nhật, không chọn riêng từng frame, channel hoặc speaker.

**Vì sao residual hỗ trợ tối ưu?** Với Y′=Y+αf(Y), đạo hàm theo Y có một thành phần trực tiếp là ma trận đơn vị và một thành phần qua nhánh f. Điều này tạo đường gradient không bắt buộc đi qua mọi lớp của f. Nó không bảo đảm gradient luôn ổn định hoặc mô hình chắc chắn hội tụ.

| Cấp residual | Đường giữ trực tiếp | Không gian |
|---|---|---|
| Trong mỗi Conv-U | a | 64 channel |
| TemporalInteraction | z | 64 channel |
| Ngoài LTRR | Y | 128 channel |

α khởi tạo khác 0 nên output ban đầu không bắt buộc giống baseline dù backbone tương ứng được khởi tạo giống nhau. Đường trực tiếp là một thành phần của phép cộng, không bảo đảm mọi thông tin còn nguyên vẹn sau khi hai nhánh cộng lại.

## 22. OutputLayer và AudioDecoder — tổng hợp waveform

**Khái niệm AI: output projection, direct estimation, transposed convolution, learned synthesis.**

OutputLayer crop Tp về T0, đổi trục để Linear xử lý channel, rồi Linear 128→512, GLU còn 256, Linear 256→256. Tensor được tổ chức thành [J,B,256,T0].

Direct estimation ở head chính nghĩa là tạo trực tiếp đặc trưng để tổng hợp waveform. masking=False nên không nhân output với encoder features như mask. encoder_output được truyền vào head chính để lấy chiều dài crop.

AudioDecoder ConvTranspose1d 256→1, k=32, stride=8 thực hiện learned synthesis. Có thể hiểu mỗi frame tạo đóng góp trên một đoạn waveform; các đoạn đặt cách nhau theo stride và cộng ở phần chồng lấn. Bộ tổng hợp được học, không bị buộc là nghịch đảo hay bản sao trọng số encoder.

**Hiểu bằng lời.** Đây là nơi các giá trị mô tả nội bộ được chuyển thành mẫu âm thanh cụ thể. Nó không phải iSTFT vì đặc trưng trước đó không phải hệ số STFT cố định.

Với T0=7997, chiều dài là (7997−1)×8+32=64000 rồi crop N. Phần audio là danh sách hai tensor [B,64000]. Với B=1 và waveform thông thường, chiều batch vẫn giữ. Model.forward trả cặp (audio,audio_aux).

## 23. Auxiliary heads và loss — cung cấp tín hiệu học

**Khái niệm AI: deep supervision, auxiliary objective, permutation-invariant training.**

Deep supervision là đặt mục tiêu ở các biểu diễn trung gian bên cạnh output cuối. Code lấy bốn tensor trước decoder stage, T=500/1000/2000/4000, nội suy về T0 rồi dùng các head phụ riêng để tạo waveform.

Head phụ bật masking: projection qua ReLU rồi nhân encoder features trước audio decoder riêng. Chúng tạo output để tính loss, không được lấy trung bình để tạo kết quả chính và không đi qua LTRR.

**Hiểu bằng lời.** Không chỉ kết quả cuối nhận phản hồi. Các biểu diễn giữa đường cũng được chuyển thành đầu ra để đo mức phù hợp với mục tiêu. Những loss này đưa gradient về các phần tạo ra biểu diễn trung gian; mục đích là hướng dẫn việc học, không bảo đảm cải thiện mọi lần train.

Permutation-invariant training, hay PIT, xử lý việc không có thứ tự tự nhiên cho hai nguồn. Với J=2 có hai cách ghép output với target; loss chọn cách phù hợp nhất thay vì buộc output thứ nhất luôn ứng với một danh tính cố định.

Loss chính là SI-SNR trên miền thời gian; loss phụ của pipeline hiện tại dùng magnitude STFT. SI-SNR đo mức phù hợp với nguồn sau khi xét hệ số scale biên độ. STFT dùng để tính loss không biến đường suy luận thành STFT-mask-iSTFT.

Hệ số phối hợp loss theo lịch huấn luyện và scalar α học được trong LTRR là hai cơ chế khác nhau. Khi giải thích nên gọi riêng α_aux và α_LTRR.

## 24. Kể lại kiến trúc bằng một mạch khái niệm thống nhất

> “Mô hình đầu tiên học cách biểu diễn waveform bằng các kênh đặc trưng. AudioEncoder sử dụng các bộ lọc convolution học được; FeatureProjector phối hợp và đưa các kênh về độ rộng separator. Những tensor này vẫn mô tả hỗn hợp, chưa phải giọng của từng người.
>
> Encoder xây dựng biểu diễn ở nhiều độ phân giải temporal. Global Block tổng hợp ngữ cảnh rộng bằng attention phụ thuộc nội dung; Local Block xử lý lân cận bằng convolution; GCFN tiếp tục phối hợp channel có xét đến frame gần nhau. Downsampling làm chuỗi ngắn hơn, còn skip giữ biểu diễn để decoder dùng lại.
>
> Speaker Split học ánh xạ tạo các nhánh nguồn. Decoder dùng cùng trọng số giữa các nguồn nhưng nhận các biểu diễn khác nhau. Nó tăng độ phân giải, hợp nhất skip và tiếp tục xử lý. Cross-speaker attention cho các nguồn sử dụng thông tin của nhau tại từng thời điểm.
>
> LTRR thực hiện residual learning sau decoder cuối. Y có đường đi trực tiếp, trong khi nhánh bổ sung học phần cập nhật. Nhánh này giảm channel 128 xuống 64 rồi tạo hai biểu diễn bằng hai Conv-U độc lập. Pointwise phối hợp channel, depthwise xử lý temporal, còn activation, dropout và residual tổ chức phép biến đổi và việc học.
>
> Một nhánh đi qua dense memory. Các tầng tái sử dụng biểu diễn trước bằng concat, phối hợp chúng bằng projection và dùng dilation để xử lý các khoảng cách temporal khác nhau. Bốn state được tổng hợp thành memory.
>
> Memory tương tác với nhánh còn lại bằng phép nhân từng phần tử. Nó có thể đổi độ lớn hoặc dấu đóng góp, không phải một mask xác suất. Kết quả cộng z, chiếu về 128, nhân hệ số học được rồi cộng Y. OutputLayer và AudioDecoder sau đó tổng hợp hai waveform.
>
> Các cơ chế này giải thích cách mô hình tổ chức việc biểu diễn và xử lý thông tin. Khả năng biến đổi bổ sung của LTRR là cơ sở thiết kế; chất lượng separation, độ ổn định và chi phí thực tế vẫn phải được đánh giá bằng thực nghiệm.”

## Nguồn kỹ thuật để kiểm tra thông số và phép toán

- [modules/module.py](../models/SepReformer_LTRR_VnSpeechMix_16K/modules/module.py): AudioEncoder, FeatureProjector, Separator, Speaker Split, OutputLayer và AudioDecoder.
- [modules/network.py](../models/SepReformer_LTRR_VnSpeechMix_16K/modules/network.py): EGA, CLA, GCFN, MultiHeadAttention, SpkAttention và LayerScale.
- [modules/ltrr.py](../models/SepReformer_LTRR_VnSpeechMix_16K/modules/ltrr.py): ChannelLayerNorm, ConvU, DenseDilatedFSMN1D, TemporalInteraction và LTRR.
- [model.py](../models/SepReformer_LTRR_VnSpeechMix_16K/model.py), [engine.py](../models/SepReformer_LTRR_VnSpeechMix_16K/engine.py), [criterions.py](../utils/implements/criterions.py): luồng forward, head phụ và loss.
- [configs.yaml](../models/SepReformer_LTRR_VnSpeechMix_16K/configs.yaml), [shape audit](../figures/ltrr_shape_audit.json): thông số và kích thước trong hình. Shape audit không đo chất lượng separation.
- Nguồn kiến trúc backbone: Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation, [PDF được cung cấp](<C:/Users/Ngoc Lan/Downloads/2406.05983v4.pdf>). LTRR là phần bổ sung trong code dự án, không phải module được đề xuất trong bài SepReformer này.
