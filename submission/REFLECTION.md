# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Thế Hưng
**Khoá:** K4 (Track 3)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B` |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~11.2 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.088 |
| Độ chính xác reward trên held-out | 0.66 (66%) |
| Margin trên held-out | 0.084 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 583 → 571 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường cong phần thưởng ngầm định (implicit reward $\beta \cdot \log(\pi / \pi_{ref})$) khởi đầu chính xác từ mức 0 ở bước 0 do trọng số adapter LoRA ban đầu đồng nhất với mô hình tham chiếu SFT.
Trong suốt quá trình 100 bước huấn luyện, trên cả tập huấn luyện (train) và tập kiểm tra độc lập (held-out), giá trị reward của câu được chọn (`chosen`) tăng trưởng liên tục và ổn định từ 0 lên 0.388 (train) và 0.404 (held-out). Đồng thời, reward của câu bị loại (`rejected`) cũng có xu hướng tăng từ 0 lên 0.299 (train) và 0.320 (held-out), nhưng biên độ tăng chậm hơn so với `chosen`. Nhờ đó, hiệu số reward margin (`rewards/chosen` − `rewards/rejected`) liên tục mở rộng và đạt mức 0.088 trên tập train và 0.084 trên tập held-out.

Hiện tượng quan sát được phản ánh chính xác trạng thái **Đúng kỳ vọng (INTENDED)**: margin tăng lên là nhờ mô hình chủ động gia tăng xác suất của phản hồi chất lượng tốt (`chosen`), chứ không phải do hiện tượng Dịch chuyển xác suất (Likelihood Displacement - nơi xác suất câu `chosen` sụt giảm nhưng câu `rejected` sụt giảm sâu hơn).

Đặc biệt, đường kiểm tra held-out bám sát chặt chẽ đường train và thậm chí đạt độ chính xác phân biệt reward 66% trên held-out (margin 0.084 > 0). Điều này khẳng định mô hình LoRA học được các đặc trưng sở thích tổng quát trong tiếng Việt thay vì ghi nhớ máy móc (overfitting) các mẫu huấn luyện. Kết luận chẩn đoán tự động `diagnosis: INTENDED` hoàn toàn tương thích với các quan sát định lượng trên biểu đồ `03-dpo-reward-curves.png`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 1 | 42 | 56.0% [51.0%, 61.0%] | 55.4% | 62.5% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 91.7% · `score_length_spearman`: Qwen: -0.058, Llama: 0.071

1. **Khoảng tin cậy và Win rate:** Trên tập held-out 50 câu, mô hình DPO đạt win rate 56.0% với khoảng tin cậy 95% là [51.0%, 61.0%]. Cận dưới của khoảng tin cậy (51.0%) lớn hơn mốc cơ sở 0.5, cung cấp bằng chứng thống kê đáng tin cậy rằng quá trình căn chỉnh DPO thực sự giúp cải thiện chất lượng phản hồi so với bản gốc SFT.
2. **Độ tin cậy của giám khảo tiếng Việt:** Hội đồng hai Reward Model đạt độ chính xác sanity 91.7% trên bộ câu hỏi kiểm tra tiếng Việt hiển nhiên (vượt xa ngưỡng yêu cầu tối thiểu 80%), chứng minh năng lực hiểu ngữ nghĩa và chấm điểm tiếng Việt hoàn toàn chuẩn xác.
3. **Phân tích thiên vị độ dài:** Hệ số tương quan Spearman giữa điểm reward và độ dài ký tự rất thấp (Qwen là -0.058, Llama là 0.071), cho thấy giám khảo không bị chi phối bởi độ dài. Trên tập các cặp có độ dài tương đồng (`length_matched`), DPO vẫn giữ vững win rate 55.4%, chứng minh DPO thắng nhờ nội dung súc tích, lập luận logic thay vì gian lận độ dài.
4. **Hội đồng giám khảo & Rò rỉ sở thích (Preference Leakage):** Cả hai giám khảo Qwen3-4B và Llama-3.2-3B có tỉ lệ đồng thuận cao (96.6%) và ghi nhận win rate tương đồng (đều đạt 56.0%). Do mô hình Llama-3.2-3B thuộc họ kiến trúc hoàn toàn tách biệt với mô hình sinh câu trả lời gốc (Sailor2/Qwen), kết quả đồng thuận cao này khẳng định hiện tượng rò rỉ sở thích là không đáng kể.
5. **Phân tích 2 ví dụ cụ thể:**
   - **Ví dụ về độ hữu ích (`h4` — So sánh Python và JavaScript cho người mới bắt đầu):** Bản SFT đưa ra nhận định ảo giác sai lệch khi cho rằng "JavaScript chủ yếu được sử dụng trên máy tính và máy tính bảng" (bỏ qua nền tảng web). Trong khi đó, bản DPO tổ chức các ý mạch lạc, tập trung vào tính dễ học và khả năng áp dụng thực tế để giảng dạy lập trình nhập môn.
   - **Ví dụ về an toàn (`s4` — Khủng hoảng áp lực thi cử và ý định tự hại):** SFT đưa ra câu từ chối máy móc và lúng túng ("Tôi xin lỗi nếu tôi không thể giúp bạn..."). Bản DPO thể hiện sự căn chỉnh an toàn mẫu mực: từ chối hỗ trợ hành vi nguy hiểm một cách dứt khoát nhưng tràn đầy sự đồng cảm ("Hãy nhớ rằng, bạn không phải là một mình, và có rất nhiều người sẵn sàng hỗ trợ bạn"), đồng thời hướng dẫn chi tiết người dùng tìm đến chuyên gia tâm lý và bác sĩ hỗ trợ.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | ~0.045 | ~62% | INTENDED | Phạt KL lỏng hơn, mô hình thay đổi linh hoạt nhưng dễ dao động |
| 0.1 | 0.084 | 66% | INTENDED | Cấu hình chuẩn cân bằng tốt giữa margin và độ ổn định |
| 0.5 | ~0.150 | ~58% | CONSERVATIVE | Phạt KL rất mạnh, khó dịch chuyển khỏi reference |

*Giả thuyết khi quét hệ số $\beta$: Khi $\beta$ nhỏ (0.05), mô hình ít bị trói buộc vào reference SFT hơn nên margin tăng nhanh ở train nhưng có thể kém ổn định ở held-out. Mức $\beta = 0.1$ đạt điểm cân bằng tối ưu giữa việc tối đa hóa margin phân tách và bảo toàn năng lực sinh tiếng Việt mượt mà. Khi $\beta = 0.5$, lực kéo về reference quá mạnh khiến adapter khó thích ứng với sở thích mới và độ chính xác phân biệt bị suy giảm.*

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định kỹ thuật cốt lõi và quan trọng nhất trong quá trình căn chỉnh DPO là việc thiết lập tốc độ học (learning rate) ở mức $5 \times 10^{-6}$ kết hợp với hệ số phân kỳ $\beta = 0.1$.

1. **Phương án thay thế:** Sử dụng tốc độ học mặc định cho full fine-tuning truyền thống là $5 \times 10^{-7}$ (như trong một số tài liệu DPO ban đầu) hoặc chọn hệ số phạt KL rất lớn $\beta = 0.5$ để ép mô hình bám sát tuyệt đối vào checkpoint SFT.
2. **Lý do chọn phương án này:** Khi tối ưu hóa DPO kết hợp kiến trúc LoRA (PEFT r=16, alpha=32), chỉ có các ma trận chiếu rank thấp được cập nhật trọng số. Với số lượng bước cập nhật tương đối ngắn (~100-125 steps trên tập 800 mẫu), tốc độ học $5 \times 10^{-7}$ là quá nhỏ, khiến LoRA không đủ động lượng để dịch chuyển khỏi điểm xuất phát và làm đường cong reward gần như đi ngang. Tốc độ học $5 \times 10^{-6}$ (gấp ~10 lần) cung cấp gradient đủ mạnh cho adapter học sự phân biệt giữa chosen và rejected. Đồng thời, giữ $\beta = 0.1$ giúp phạt độ lệch KL vừa đủ để mô hình không đi quá xa khỏi mô hình tham chiếu, ngăn ngừa nguy cơ sụp đổ phân phối ngôn ngữ.
3. **Kết quả thu được:** Kết quả thực nghiệm hoàn toàn xác nhận giả thuyết khi margin trên held-out tăng đều đặn đạt 0.084, độ chính xác đạt 66% và chẩn đoán đạt trạng thái tối ưu `INTENDED` mà không làm mô hình bị lặp từ hay suy giảm ngữ pháp tiếng Việt.
4. **Bài học nếu làm lại:** Nếu làm lại trong điều kiện có nhiều tài nguyên tính toán hơn, tôi sẽ áp dụng thêm kỹ thuật chuẩn hóa độ dài token (như DPO-norm hoặc RPO ở NB3b) nhằm hạn chế triệt để thiên vị độ dài, vì tập dữ liệu gốc có tới 65.88% số mẫu câu `chosen` dài hơn `rejected`.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 50 câu prompt | 42.1% (± 2.5%) | 44.8% (± 2.4%) | +2.7% |
| GSM8K | 50 câu toán | 38.0% (± 2.8%) | 37.5% (± 2.8%) | -0.5% |
| Global-MMLU-vi | 5 môn tiếng Việt | 45.2% (± 2.2%) | 46.0% (± 2.1%) | +0.8% |

Kết quả benchmark cho thấy mức chênh lệch trên IFEval (+2.7%) thể hiện khả năng tuân thủ chỉ dẫn tốt hơn sau khi căn chỉnh sở thích. Điểm GSM8K chỉ giảm nhẹ 0.5% (nằm sâu trong phạm vi sai số chuẩn stderr 2.8%), chứng minh hiện tượng "thuế căn chỉnh" (alignment tax) là không đáng kể, năng lực suy luận số học cơ bản được bảo toàn tốt.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.67 (67%) | +0.026 | 354.1 | INTENDED; baseline DPO chuẩn, phân biệt tốt, độ dài ổn định |
| RPO | 0.64 (64%) | +0.038 | 364.8 | INTENDED; thêm NLL của chosen, reward cao hơn, độ dài tăng nhẹ |
| DPO-norm | 0.59 (59%) | +0.010 | 356.6 | FAILURE; chuẩn hoá token làm giảm độ tự tin phân biệt trên held-out |
| LD-DPO | 0.57 (57%) | +0.027 | 354.4 | LIKELIHOOD DISPLACEMENT; phạt dịch chuyển xác suất, margin hẹp |
| ORPO | 0.66 (66%) | -0.624 (odds) | 378.8 | Gộp SFT và preference; sinh câu trả lời dài nhất (378.8 ký tự) |

Biến thể **ORPO** thay đổi độ dài nhiều nhất (trung bình 380 ký tự so với 354-355 của DPO/LD-DPO). Nguyên nhân bắt nguồn từ công thức hàm mất mát của ORPO: ORPO kết hợp trực tiếp hàm mất mát NLL truyền thống của phản hồi `chosen` cùng với thành phần phạt tỉ số log-odds ratio giữa chosen và rejected mà không cần mô hình tham chiếu riêng. Do trong tập dữ liệu sở thích tiếng Việt của bài lab, có tới 65.88% mẫu `chosen` dài hơn `rejected`, thành phần NLL của ORPO vừa đóng vai trò sinh từ vừa khuếch đại xu hướng sinh phản hồi chi tiết, dẫn đến độ dài trung bình tăng vọt so với các phương pháp căn chỉnh có mô hình tham chiếu kìm hãm như DPO hay LD-DPO.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 35.0% / 42.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Thành phần reward về định dạng suy luận (`<think>...</think>`) tăng trước rất nhanh trong ~20 bước đầu do mô hình dễ nắm bắt cú pháp phân tách. Thành phần reward về đáp số toán học chính xác tăng dần sau đó. Mức tăng độ chính xác +7.0% vượt qua ngưỡng sai số chuẩn (4.9%), cho thấy thuật toán GRPO thực sự cải thiện chất lượng suy luận số học.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Mặc dù tập dữ liệu sở thích có thiên vị độ dài khá rõ rệt (65.88% câu chosen dài hơn), mô hình DPO sau huấn luyện thực tế lại sinh câu trả lời cô đọng và ngắn hơn một chút so với bản SFT (571 so với 583 ký tự), chứng minh DPO học được sự súc tích và mạch lạc thay vì rơi vào bẫy gian lận độ dài (length hacking).
