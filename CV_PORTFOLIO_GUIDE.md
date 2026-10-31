# Hướng Dẫn Trình Bày Dự Án Vào CV & Thuyết Phục Tech Lead

Tài liệu này được thiết kế để **biến dự án Multilingual Video Transcription & Summarization thành điểm sáng lớn nhất trong CV của bạn**, giúp bạn tự tin trả lời phỏng vấn ở vị trí **AI / NLP / Machine Learning Engineer**.

---

## 1. Cách Viết Vào CV (Chuẩn Format Nhà Tuyển Dụng)

### **Tên dự án:**
**VideoInsight: Production-Grade Multilingual Video Transcription & Hierarchical Summarization Platform**

- **Vai trò:** Lead AI / NLP Engineer
- **Tech Stack:** Python, PyTorch, Hugging Face (Transformers, Datasets, Hub API), mT5, ViT5, OpenAI Whisper, yt-dlp, ROUGE (Unicode-aware), BERTScore, Gradio, Kaggle MLOps.

### **Các Bullet Points ghi điểm trong CV:**
* **End-to-End Multimodal Pipeline:** Xây dựng hệ thống hoàn chỉnh từ trích xuất audio trực tiếp qua URL (YouTube, Vimeo bằng `yt-dlp`), chuyển đổi giọng nói thành văn bản kèm timestamp chuẩn xác bằng **Whisper ASR**, đến tóm tắt phân cấp văn bản đa ngôn ngữ.
* **Multilingual Sequence-to-Sequence Modeling:** Huấn luyện mô hình **mT5** (khởi tạo từ `mT5_multilingual_XLSum`, 582M params) hỗ trợ 4 ngôn ngữ (**Vi, En, Fr, Es**) bằng cơ chế định tuyến tiền tố (`summarize <lang>:`), kết hợp dữ liệu hỗn hợp VietNews (20k mẫu), XL-Sum và VT-SSum (video transcripts với hệ số upsampling 2x).
* **Deep Learning Training Stability (Loss NaN Mitigation):** Khắc phục triệt để hiện tượng mất ổn định số học (Loss NaN/Inf do RMSNorm ở FP16) của dòng T5/mT5 bằng việc triển khai kiến trúc **FP32 kết hợp bộ tối ưu hóa Adafactor**, kiểm soát gradient norm và learning rate linear warmup.
* **Fault-Tolerant Multi-Session MLOps:** Thiết kế cơ chế tự động đồng bộ checkpoint bất đồng bộ (Async Hub Sync) lên Hugging Face Hub, cho phép ngắt và tiếp tục huấn luyện (Resume) liền mạch qua nhiều phiên Kaggle độc lập mà không bị gián đoạn hay mất dữ liệu.
* **Hierarchical Map-Reduce for Long Transcripts:** Phát triển thuật toán tóm tắt phân cấp đệ quy (Hierarchical Map-Reduce) giải quyết giới hạn context length của Transformer, cho phép tóm tắt chính xác các video dài 30–90 phút thành Timeline / Key Moments và Executive Summary.
* **Rigorous Evaluation & Benchmark:** Xây dựng bộ đo lường độc lập cho từng ngôn ngữ với **Unicode-aware ROUGE Tokenizer** (bảo toàn 100% dấu tiếng Việt, tránh lỗi chia âm tiết của rouge-score gốc), chứng minh độ vượt trội rõ rệt so với baseline Lead-3 và mô hình Zero-shot ban đầu, kiểm định ngữ nghĩa bằng **Multilingual BERTScore**.

---

## 2. Bộ Câu Hỏi & Câu Trả Lời Sâu Sắc Khi Phỏng Vấn Kỹ Thuật

### Câu 1: "Tại sao em chọn fine-tune mô hình Seq2Seq chuyên biệt (mT5) thay vì gọi API của OpenAI hay Gemini?"
> **Trả lời:**
> "Trong môi trường sản xuất (Production), việc dùng API thương mại tuy triển khai nhanh nhưng gặp 3 rào cản cốt tử:
> 1. **Chi phí vận hành (Inference Cost):** Video 60 phút có thể tạo ra 10,000–15,000 tokens transcript. Nếu gọi API thương mại cho hàng nghìn lượt video mỗi ngày, chi phí token sẽ tăng theo hàm tuyến tính. Mô hình chuyên biệt 580M tham số có thể chạy on-premise hoặc server GPU riêng với chi phí cố định cực rẻ.
> 2. **Bảo mật & Quyền riêng tư (Data Privacy):** Nhiều doanh nghiệp không được phép gửi nội dung cuộc họp hoặc video nội bộ lên máy chủ của bên thứ ba. Mô hình open-weights cho phép deploy hoàn toàn offline trong mạng nội bộ.
> 3. **Kiểm soát độ trễ & Tối ưu hóa (Latency & Quantization):** Với mT5, em có thể lượng hóa INT8/FP16 (vLLM / TensorRT-LLM / ONNX Runtime), kiểm soát chặt chẽ beam search, repetition penalty và tốc độ sinh mà API đóng không can thiệp sâu được."

---

### Câu 2: "Tại sao khi huấn luyện mT5 trên GPU lại dễ bị lỗi Loss = NaN, và em giải quyết như thế nào?"
> **Trả lời:**
> "Kiến trúc T5 và mT5 không sử dụng LayerNorm thông thường mà sử dụng **RMSNorm** và không có bias terms. Khi huấn luyện ở độ chính xác nửa FP16, các giá trị kích hoạt trong self-attention rất dễ bị tràn số (overflow), dẫn đến việc tính toán gradient bị NaN và làm hỏng toàn bộ quá trình train.
> Trong dự án này, em đã xử lý dứt điểm bằng 3 giải pháp:
> 1. Sử dụng **FP32** thay vì FP16 (hoặc BF16 nếu dùng GPU kiến trúc Ampere trở lên như A100).
> 2. Thay thế AdamW bằng **Adafactor Optimizer** với cấu hình chuẩn cho T5 (scale_parameter=False, relative_step=False), vừa tiết kiệm đáng kể bộ nhớ trạng thái optimizer (giúp vừa vặn VRAM 16GB của T4), vừa duy trì cập nhật trọng số cực kỳ ổn định.
> 3. Tích hợp `NanGuardCallback` để theo dõi giá trị loss liên tục và ngắt tiến trình ngay lập tức nếu phát hiện bất thường, tránh lãng phí hạn ngạch GPU."

---

### Câu 3: "Khi đánh giá ROUGE cho tiếng Việt, có vấn đề kỹ thuật gì đặc thù không?"
> **Trả lời:**
> "Thư viện `evaluate` hay `rouge_score` tiêu chuẩn của Python sử dụng regex tách từ tiếng Anh (xóa ký tự non-ASCII hoặc dùng word-boundary `\b`). Khi gặp tiếng Việt có dấu như *'Hà Nội tóm tắt thành phố'*, nó sẽ lọc bỏ toàn bộ nguyên âm có dấu và tách thành từng chữ cái rời rạc (*'h', 'n', 'i', 't', 'm'*), khiến điểm ROUGE bị sai lệch hoàn toàn.
> Em đã giải quyết bằng cách viết một **Custom Unicode Tokenizer** (`UnicodeTokenizer`):
> - Chuẩn hóa toàn bộ văn bản về chuẩn **Unicode NFC**.
> - Loại bỏ dấu câu nhưng giữ nguyên vẹn toàn bộ ký tự chữ cái và dấu thanh tiếng Việt (`[^\w\s]`).
> - Tách từ theo khoảng trắng (syllable-level tokenization), giúp kết quả ROUGE phản ánh chính xác chất lượng bản tóm tắt tiếng Việt."

---

### Câu 4: "Video dài cả tiếng đồng hồ thì mô hình có token limit 512/1024 token tóm tắt kiểu gì?"
> **Trả lời:**
> "Em áp dụng chiến lược **Hierarchical Map-Reduce**:
> - **Tầng Map:** Transcript kèm mốc thời gian từ Whisper được chia nhỏ thành các phân đoạn logic (chunk khoảng 400 tokens) dựa trên dấu kết thúc câu (`.`, `!`, `?`). Mô hình tóm tắt song song từng phân đoạn để tạo ra các gạch đầu dòng theo dòng thời gian (Key Moments / Timeline).
> - **Tầng Reduce:** Toàn bộ các câu tóm tắt cục bộ được ghép lại. Nếu văn bản tổng hợp vẫn dài hơn ngưỡng token, hệ thống tiếp tục áp dụng vòng tóm tắt đệ quy cấp cao hơn (Second-pass summarization) để tạo thành **Executive Summary** cô đọng cho toàn bộ video."

---

### Câu 5: "Làm thế nào để đảm bảo quá trình huấn luyện dài không bị mất khi Kaggle ngắt kết nối ở giới hạn 12 giờ?"
> **Trả lời:**
> "Em thiết kế kiến trúc **Stateful Multi-Session Training**:
> 1. Tích hợp callback `HubSyncCallback` để tự động đẩy checkpoint mới nhất lên repo riêng tư trên Hugging Face Hub qua tiến trình bất đồng bộ (`run_as_future=True`).
> 2. Lưu trữ file `train_config.json` chứa hash cấu hình và siêu tham số để kiểm tra tính toàn vẹn (config validation) khi resume.
> 3. Cài đặt `TimeGuardCallback` để tự động lưu lại và ngắt an toàn tại mốc 10.5 giờ (trước khi Kaggle cưỡng chế tắt ở 12h).
> Khi mở một phiên Kaggle mới, notebook chỉ cần kéo checkpoint từ Hub về và gọi `trainer.train(resume_from_checkpoint=...)` để tiếp tục từ đúng epoch và learning rate đang dở dang."
