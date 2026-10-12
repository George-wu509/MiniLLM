# MiniLLM 實測驗證

記錄時間：2026-10-12 06:08 Asia/Taipei

硬體：NVIDIA RTX 4000 Ada Generation Laptop GPU，11.99 GB VRAM，63.68 GB RAM。CUDA matmul 已實際執行成功。

## 已完成的訓練

主模型直接執行 MiniMind 原始 Transformer，由隨機初始化開始；沒有載入外部預訓練權重。nanoGPT 使用同一份 train/validation token stream 作架構對照。

| 實驗 | 實際參數量 | 實際 vocabulary | Pretrain steps | SFT steps | DPO steps | LoRA steps |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 500,864 | 835 | 40 | 20 | 10 | 15 |
| laptop-check | 24,030,208 | 835 | 2 | 2 | 2 | 2 |

每個 step 指 optimizer update；laptop-check 使用 batch=2、gradient accumulation=8、pretrain prediction window=256。MiniMind 使用 L+1 個輸入 token 計算 L 個 next-token targets。SFT/DPO/domain 教學配對通常比最大 context 短。

laptop-check 保留 laptop 的 512 hidden / 8 layers 架構，只用 2 steps 驗證各階段。因合成語料的 BPE vocabulary 只有 835，實際是 24.03M 參數；6400 vocabulary 的 laptop 設定約 26.88M。尚未執行 laptop.json 的 2000/500/200/300 steps 長訓練。

## 共同 held-out 評估

| 階段 | Raw validation loss | Perplexity | SFT completion loss | Domain completion loss | Chosen win rate |
|---|---:|---:|---:|---:|---:|
| initialized | 6.7693 | 870.68 | 6.8037 | 6.7083 | 66.7% |
| base | 4.2932 | 73.20 | 6.4739 | 6.4639 | 72.2% |
| sft | 4.3272 | 75.73 | 4.8764 | 6.0428 | 77.8% |
| dpo | 4.3283 | 75.82 | 4.9068 | 6.0253 | 77.8% |
| lora | 4.3388 | 76.62 | 4.9434 | 5.8047 | 77.8% |
| merged | 4.3388 | 76.62 | 4.9434 | 5.8047 | 77.8% |
| nanoGPT (40 steps) | 4.2279 | 68.58 | — | — | — |

評估固定使用相同 raw windows 與全部 held-out pairs，FP32 且停用 TF32。以上不同欄位是不同 objective，不能互相視為同一 loss。Chosen win rate 使用 completion log probability sum，會受回答長度影響；它不是通用能力分數。DPO 的 reward margin 與訓練 loss 可在各 stage metrics 及 TensorBoard 查看。

這批資料只有少量合成教學樣本，模型輸出仍可能是錯誤、重複或無意義文字。測試結果證明 lifecycle 實作與權重更新可運作，沒有證明模型具備實用聊天能力。

## Dataset / tokenizer

Pretrain train tokens：8,388；validation tokens：678。Tokenizer 只從 pretrain train partition 建立。

| 資料 | Raw rows | Empty removed | Duplicate removed | Train | Validation |
|---|---:|---:|---:|---:|---:|
| pretrain | 283 | 1 | 1 | 262 | 19 |
| sft | 120 | 0 | 0 | 102 | 18 |
| dpo | 120 | 0 | 0 | 102 | 18 |
| domain | 80 | 0 | 0 | 75 | 5 |

每個 dataset/tokenizer 的 SHA256 與來源路徑在 data/manifest.json。normalized exact 去重與按 prompt/document 的 hash split 已驗證沒有 partition 交集。TXT、JSONL、Parquet 輸入都已驗證。

## GPU 記憶體實測

| 實驗 | Stage | Peak torch allocated MiB |
|---|---|---:|
| smoke | base | 36.65 |
| smoke | sft | 29.07 |
| smoke | dpo | 35.53 |
| smoke | lora | 26.33 |
| laptop-check | base | 577.23 |
| laptop-check | sft | 487.83 |
| laptop-check | dpo | 585.85 |
| laptop-check | lora | 304.07 |

這是 torch.cuda.max_memory_allocated 的 tensor allocation 峰值，未包含 driver/context/allocator reserved memory，且只涵蓋短跑與實際短配對資料；不是長訓練或所有語料的顯存上限。

## 行為與服務驗證

- 11 項 unittest 全數通過，完整輸出在 reports/tests.txt。
- CPU 上原生預訓練，以及 TRL SFT/DPO/LoRA 從 checkpoint 恢復後，最終所有權重 tensor 與不中斷訓練逐值相等。
- SFT/LoRA 續訓的累積 token 計數也一致；不同回答長度的 gradient accumulation 與完整 batch gradient 一致。
- Causal attention 不會看見未來 token，incremental KV cache logits 與整段 forward 一致。
- 已驗證初始化權重 Normal(0,0.02)、保存/重載 logits，以及 LoRA 只更新 adapter。
- Initialized/Base/SFT/DPO/LoRA/Merged 的 cached/uncached greedy token IDs 均一致。
- LoRA adapter 與 merged 模型在三個固定 prompts 的 greedy token IDs 相同。
- 本機 Uvicorn 已實際啟動，/health、/v1/chat/completions、/metrics HTTP 驗證成功。
- 本機 TensorBoard 已實際啟動，HTTP 與 base loss/gradient/LR scalar tags 驗證成功。
- pip check 通過，所有 Python 模組可編譯。

API/TensorBoard 測試服務已自動停止；需要使用時依 README 啟動。Windows 沙箱會限制 localhost socket，因此 API/TestClient/HTTP 驗證是在允許的沙箱外完成。

HF 的 tied embedding checkpoint 可能提示 lm_head.weight missing；該 tensor 與 embed_tokens.weight 共享權重，續訓逐值一致性測試已驗證正確恢復。

## 版本與重現

- torch: 2.9.1+cu128
- transformers: 4.57.6
- trl: 0.24.0
- peft: 0.17.1
- datasets: 4.3.0
- tokenizers: 0.22.1
- accelerate: 1.11.0

Upstream commits：

- minimind: `f659b55761b754d306bd140573493a6543cafd7f`
- nanogpt: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`

完整版本在 requirements.lock.txt；環境偵測在 reports/environment.json；實際 HTTP 回應与 scalar tags 在 reports/services.json；階段比較與生成樣本在 artifacts/smoke/evaluation。操作命令與資料格式請看 README.md。
