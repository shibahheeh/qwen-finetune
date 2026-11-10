# qwen-finetune

Fine-tune **Qwen3-1.7B** bằng **LoRA** cho bài toán **tóm tắt tin tức tiếng Việt** trên bộ dữ liệu [VietNews](https://huggingface.co/datasets/Yuhthe/vietnews), chạy trên Kaggle (1× GPU T4).

## Nội dung

Toàn bộ pipeline nằm trong notebook `finetuning-qwen.ipynb`:

1. **Khảo sát dữ liệu**: độ dài article/abstract theo tokenizer Qwen3, mẫu rỗng, trùng lặp, tỉ lệ abstract chứa số/tên riêng không có trong bài.
2. **Baseline** (`eval_baseline.py`): Lead-1 và Qwen3 zero-shot trên 500 mẫu test cố định (seed 42), chấm bằng ROUGE tiếng Việt (chuẩn hoá NFC, khoảng tin cậy 95% bằng bootstrap).
3. **Fine-tune LoRA** (`train_lora.py`):
   - `--mode overfit`: kiểm tra pipeline trên 32 mẫu.
   - **E1 (`--variant raw`)**: 5.000 mẫu train với abstract gốc.
   - **E2a (`--variant filtered`)**: 5.000 mẫu đã bỏ những abstract chứa số/tên riêng không có trong bài báo.

Baseline và mô hình fine-tune dùng chung prompt, cấu hình sinh (greedy, `max_new_tokens=128`, `repetition_penalty=1.05`) và cách chấm điểm. Chế độ thinking của Qwen3 được tắt.

## Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Mô hình gốc | `Qwen/Qwen3-1.7B` |
| LoRA rank | 16 |
| Target modules | `q, k, v, o, gate, up, down_proj` |
| Tham số huấn luyện | 17,4M (≈1,0%) |
| Learning rate | 2e-4 |
| Batch × grad accum | 2 × 8 (hiệu dụng 16) |
| Epoch | 1 |
| Độ dài article tối đa | 1.536 token |
| Precision | fp16 |
| Thời gian / GPU peak | ~92 phút / 10,6 GB |

Loss chỉ được tính trên phần tóm tắt; phần prompt được gán nhãn `-100`.

## Kết quả

| Hệ thống | n | ROUGE-1 | ROUGE-2 | ROUGE-L | Số từ TB | % không có trong bài* |
|---|---|---|---|---|---|---|
| Lead-1 | 500 | 25,75 | 13,43 | 20,09 | 34,9 | 0,0 |
| Qwen3 zero-shot | 500 | 32,15 | 15,94 | 23,06 | 41,1 | 3,6 |
| **E1** LoRA (raw) | 100 | 32,07 | 16,99 | **24,90** | 33,4 | 5,0 |
| **E2a** LoRA (filtered) | 100 | **32,51** | **17,10** | 24,48 | 33,8 | **1,0** |

\* Tỉ lệ bản tóm tắt chứa số hoặc tên riêng không xuất hiện trong bài báo. Riêng abstract tham chiếu có tỉ lệ này là 23,0%.

**Nhận xét**
- Sau fine-tune, ROUGE-2 và ROUGE-L tăng so với zero-shot, và bản tóm tắt ngắn hơn, gần với độ dài tham chiếu (~34 từ).
- Lọc dữ liệu train (E2a) giảm mạnh tỉ lệ thông tin không có trong bài (5,0% → 1,0%) mà không làm giảm ROUGE.
- Lưu ý: E1/E2a mới được đánh giá trên 100 mẫu, còn baseline trên 500 mẫu, nên cần chạy lại trên đủ 500 mẫu để so sánh chính xác.

## Chạy lại

Trên Kaggle, bật GPU T4:

```bash
pip install -q peft rouge-score
pip uninstall -y torchao          # tránh xung đột với peft

python eval_baseline.py                              # baseline
python train_lora.py --mode overfit                  # kiểm tra pipeline
python train_lora.py --mode train --variant raw      # E1
python train_lora.py --mode train --variant filtered # E2a
```

Phiên bản đã dùng: `torch 2.11.0`, `transformers 5.16.1`, `peft 0.20.0`, `datasets 4.8.5`.

Kết quả baseline được lưu trong `results/`, còn adapter và log huấn luyện được lưu trong `runs/E1_raw/` và `runs/E2a_filtered/`.
