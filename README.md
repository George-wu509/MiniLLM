# MiniLLM Training Laboratory

這套實驗室在 **C:\LLM\MiniLLM** 執行完整 LLM lifecycle：資料清理 → 自訓 tokenizer → 隨機初始化 → Pretraining → SFT → DPO；LoRA 是從 Base/SFT/Aligned Model 分出的微調分支。每個實驗都有獨立資料、模型、配置、checkpoint、metrics 與評估。沒有下載預訓練模型權重。

主模型直接使用 [MiniMind](https://github.com/jingyaogong/minimind) 的 Transformer 原始碼；預訓練迴圈以原生 PyTorch 實作，方便觀察反向傳播與 AdamW。[TRL 0.24](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) 執行 SFT、DPO，PEFT 執行 LoRA。[nanoGPT](https://github.com/karpathy/nanoGPT) 使用同一份 tokenizer/token stream 作另一種 Transformer 的預訓練對照。兩種架構的權重不能直接互換。

## 立即操作

在 PowerShell 執行，使用完整 Python 路徑，不需要先 Activate venv：

```powershell
cd C:\LLM\MiniLLM
.\scripts\lab.ps1 doctor
.\scripts\lab.ps1 infer --run smoke --stage dpo --prompt "Calculate 2 + 3. Answer with a number." --greedy
.\scripts\lab.ps1 evaluate --run smoke
```

`smoke` 是建立時實際跑過的微型實驗。它用合成資料驗證訓練機制，生成文字品質很有限。新的完整實驗請給新的 run 名稱：

```powershell
.\scripts\smoke.ps1 -Run my-first-lab
```

此命令會跑完 MiniMind 主流程、LoRA 分支及 nanoGPT 對照。既有完成階段不會被覆蓋；中斷時請使用各階段的續訓命令。

## 模型與環境

| Profile | Transformer | Vocab 上限 | 訓練 context | 用途 |
|---|---|---:|---:|---|
| configs/smoke.json | hidden 128、2 layers、4 heads | 1024 | 128 | 快速檢查完整流程 |
| configs/laptop.json | hidden 512、8 layers、8 heads、4 KV heads | 6400 | 256 | 約 27M 參數級；12 GB GPU 的起點 |

實際 vocab/參數量取決於資料與 tokenizer；完整數值寫在 manifest/lineage。MiniMind 包含 causal attention、GQA、RoPE、RMSNorm 與 SwiGLU。Laptop profile 是可修改的起點；長時間訓練速度及品質取決於資料量、散熱與實際設定。

本機使用 Python 3.12、PyTorch 2.9.1 CUDA 12.8、Transformers 4.57.6、TRL 0.24.0、PEFT 0.17.1。GPU 與記憶體檢測結果在 `reports/environment.json`。`requirements.lock.txt` 保存安裝後完整版本；`upstream-lock.json` 固定兩個上游 commit。上游原始碼放在 vendor，沒有改動。相容性與初始化設定集中在 `minillm/modeling.py`。

重建環境：

```powershell
.\scripts\setup.ps1
# 或指定自己安裝的 Python 3.12
.\scripts\setup.ps1 -Python C:\Python312\python.exe
# 無 NVIDIA GPU 的環境
.\scripts\setup.ps1 -CpuOnly -Python C:\Python312\python.exe
```

目前 `.venv` 使用 Codex 隨附的 Python 3.12 作為 base interpreter；若其路徑因重新安裝 App 而失效，可指定獨立 Python 3.12 重建。安裝需要網路與數 GB 磁碟；資料生成、訓練和推理使用本地檔案，不需要 Hugging Face 登入。支援 Windows 原生環境；不依賴 bitsandbytes、Triton、torch.compile 或多程序 DataLoader。CUDA 可用且支援時啟用 BF16 autocast，權重與 AdamW state 維持 FP32；CPU 使用 FP32。

## 資料準備

`data/demo` 是本地生成的教學合成資料，含中英文、加法指令、偏好配對與實驗室領域問答。`source.json` 記錄來源與用途。請以足量且可使用的真實資料取代它，才能研究語言品質。原始資料可放 `data/raw`；程式支援單檔或同種類檔案的目錄。

Pretraining 支援 TXT（空白行分文件）、JSONL、Parquet；JSONL/Parquet 需要 `text`：

```json
{"text":"A document used to learn next-token prediction."}
```

SFT/Domain 支援 JSONL、Parquet，接受 instruction/output、prompt/response，或 `messages`/MiniMind 的 `conversations`；多輪資料以最後一個 assistant 回答作 completion，前面訊息作 prompt：

```json
{"instruction":"說明 LoRA。","output":"LoRA 只更新低秩矩陣。"}
{"messages":[{"role":"user","content":"What is SFT?"},{"role":"assistant","content":"Training on instruction-response examples."}]}
```

DPO 支援 JSONL、Parquet，三個欄位都須為字串：

```json
{"prompt":"What is 2 + 3?","chosen":"5","rejected":"9"}
```

清理採 NFKC、換行正規化、移除控制字元、壓縮空白、移除空紀錄、精確去重。這套清理適合自然語言，會移除行首縮排；程式碼語料應先調整 `clean()`。尚未做近似去重或自動個資清理。以 normalized 文件/指令的 hash 決定約 90/10 split，同一 prompt 的不同回答留在同一 partition；同一單輪指令在 SFT/DPO/domain 的 split 也一致。小資料若未產生兩個非空 partition 會明確報錯。

Byte-level BPE tokenizer **只使用 pretraining train partition**，包含完整 byte alphabet 與 chat special tokens。所有階段固定使用相同 tokenizer，避免 token ID/embedding 不一致。manifest 保存來源 SHA256、清理計數、切分與 token 數。配對資料超過 block_size 會報錯，請增加 context 或縮短資料；不會默默截掉回答。

```powershell
.\scripts\lab.ps1 prepare --run real-lab --config configs/laptop.json `
  --raw data/raw/pretrain.jsonl `
  --sft-data data/raw/sft.jsonl `
  --dpo-data data/raw/dpo.jsonl `
  --domain-data data/raw/domain.jsonl
```

若只想研究某一階段，可用 demo 的其他資料欄位補足四種輸入，再只執行所需命令。prepare 和 tokenizer 是不可變的實驗產物；變更資料/模型尺寸請建立新 run。

## 逐階段操作與學習重點

```powershell
.\scripts\lab.ps1 init --run real-lab
.\scripts\lab.ps1 pretrain --run real-lab
.\scripts\lab.ps1 sft --run real-lab
.\scripts\lab.ps1 dpo --run real-lab
.\scripts\lab.ps1 lora --run real-lab --parent sft
.\scripts\lab.ps1 evaluate --run real-lab
.\scripts\lab.ps1 nanogpt --run real-lab
```

1. **初始化**：`initialized/weight_stats.json` 列出每個 tensor 的維度、mean/std 與參數量。Linear/Embedding 使用 Normal(0,0.02)，RMSNorm 使用 1，輸入與輸出 embedding tied。此時沒有語言能力，可用 `infer --stage initialized` 看隨機輸出。
2. **預訓練**：隨機取 token windows，做 shifted next-token cross entropy。觀察 train loss、固定 validation loss/perplexity、clip 前 gradient norm、learning rate、tokens/sec。AdamW、warmup、cosine decay 與 gradient accumulation 都可在 profile 調整。每個 checkpoint 包含 optimizer、scheduler、step、CPU/CUDA RNG；初始與最終模型分開保存。
3. **SFT**：同一 Base Model，chat template 將指令格式化，prompt/padding labels 設 -100，只學習回答和 EOS。輸出完整 SFT weights。
4. **DPO**：預設由 SFT weights 開始，另載一份凍結 SFT reference，使用 chosen/rejected 的相對 log probabilities、beta=0.1 與 sigmoid DPO loss。reward/accuracy/margin/loss 由 TRL 記錄。DPO eval_loss 不能和 pretrain/SFT loss 當成同一指標比較。
5. **LoRA**：預設分支由 SFT 開始，也可 `--parent base` 或 `--parent dpo`。凍結原模型，只更新 attention/MLP projection 的低秩 adapter。`lineage.json` 記錄 trainable fraction、parent 與 parent weights SHA256。推理自動載入精確 parent+adapter；adapter 不是獨立完整模型。LoRA 是更新方法，本實驗用它做 domain SFT。

SFT/DPO/LoRA 每個 checkpoint 由 HF Trainer 保存 optimizer/scheduler/RNG/trainer state；每個完成階段均有 final weights/adapter、tokenizer、配置、metrics、lineage。checkpoint 保留所有 save intervals，較長實驗請監控磁碟。模型的載入使用本實驗室的 `load_model()`；如要以任意外部 `AutoModelForCausalLM` 工具載入，還需註冊 MiniMind 自訂架構。

續訓使用原實驗配置，不要改 model/tokenizer、資料、batch 或步數：

```powershell
.\scripts\lab.ps1 pretrain --run real-lab --resume artifacts/real-lab/base/checkpoint-100
.\scripts\lab.ps1 sft --run real-lab --resume artifacts/real-lab/sft/checkpoint-100
.\scripts\lab.ps1 dpo --run real-lab --resume artifacts/real-lab/dpo/checkpoint-100
.\scripts\lab.ps1 lora --run real-lab --resume artifacts/real-lab/lora/checkpoint-100
```

完成的階段受到覆蓋保護；以上用於中斷而未完成的階段。不要將來源不明的 PyTorch checkpoint 當成實驗室自己產生的續訓檔載入。

## 比較、推理、監控與本機部署

```powershell
.\scripts\lab.ps1 infer --run smoke --stage base --prompt "Arithmetic example:" --raw-prompt --temperature 0.7 --top-p 0.9 --top-k 20
.\scripts\lab.ps1 infer --run smoke --stage sft --prompt "Explain SFT." --greedy
.\scripts\lab.ps1 infer --run smoke --stage lora --prompt "說明 LoRA。" --no-cache
.\scripts\lab.ps1 cache --run smoke --stage dpo
.\scripts\lab.ps1 merge --run smoke
.\scripts\lab.ps1 evaluate --run smoke
.\scripts\monitor.ps1 -Run smoke
```

`--greedy` 取 argmax；sampling 則使用 temperature、top-p、top-k，預設固定 seed。`--raw-prompt` 直接續寫文字，其餘使用同一 chat template。KV cache benchmark 固定生成長度，檢查 cached/uncached greedy token IDs 是否相同並記錄時間；小模型不保證 cache 一定更快。merge 產生獨立的完整 merged weights，同時保留 adapter。

比較報告位於 `artifacts/<run>/evaluation/comparison.md` 與 JSON，含同一份 raw validation loss/perplexity、SFT/domain completion loss、chosen win rate、preference margin、cache 檢查與固定 prompt 生成樣本。評估使用 FP32 且停用 TF32，避免前一個 Trainer 的全域設定影響比較。這些是教學 proxy metrics，不能單靠它們聲稱模型有實用能力。nanoGPT 的原生架構只有 temperature/top-k、沒有 KV cache；它的 raw loss/perplexity 與生成結果保存在 `nanogpt`，作架構對照。

TensorBoard 預設在 http://127.0.0.1:6006 ，可比較各階段 curves。另開 PowerShell 啟動本機 API：

```powershell
.\scripts\lab.ps1 serve --run smoke --stage dpo --port 8000
```

API 文件在 http://127.0.0.1:8000/docs ，提供 `/health`、`/metrics` 與 `/v1/chat/completions`。此教學端點支援單一 user message、非串流，限制 prompt 與 context，預設僅綁定本機；不提供公開服務的 authentication/multiuser scheduling。監控記錄保存 tokens、延遲與 sampling 設定，不寫入 prompt/response。

```powershell
$payload = @{ messages = @(@{role='user';content='Explain LoRA.'}); max_tokens=32; temperature=0.8; top_p=0.9; use_cache=$true } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $payload
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

## 產物與驗證

```text
artifacts/<run>/
  config.json                      # 實驗配置
  data/manifest.json               # dataset/tokenizer hashes 與切分統計
  data/{pretrain,sft,dpo,domain}/   # train/validation JSONL
  data/pretrain/*.bin              # int32 token stream, memory-mapped
  tokenizer/                       # tokenizer.json, config, chat template
  initialized/                     # 隨機 weights + weight statistics
  base/                            # 預訓練 weights + optimizer checkpoints
  sft/                             # TRL SFT weights + Trainer checkpoints
  dpo/                             # TRL DPO weights + Trainer checkpoints
  lora/                            # adapter + parent provenance
  merged/                          # optional merged full weights
  nanogpt/                         # native GPT checkpoint + metrics
  tensorboard/                     # loss/gradient/LR curves
  evaluation/                      # 每個階段與共同比較結果
  api_requests.jsonl               # 本機 API 監控
```

執行行為驗證：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

測試涵蓋精確去重/防洩漏 split、byte-level tokenizer、completion masking、causal attention、cache logits、LoRA 凍結、保存/重載、optimizer/RNG 續訓一致性及本機 API。建立時的實際訓練結果、GPU 峰值與測試摘要在 `reports/validation.md`。

`configs/laptop-check.json` 保留 laptop 結構、batch/context/accumulation，每個訓練階段只跑 2 個 optimizer steps，供硬體適配檢查；它已跑過，但沒有完成 laptop profile 的長訓練。實際 tokenizer vocab 依語料而變，教學資料形成的詞表小於 6400。

API 與 TensorBoard 的真正 HTTP 啟動/讀取測試可用 `.\.venv\Scripts\python.exe scripts/verify_services.py` 重現；此程式會自動關閉自己啟動的測試服務，平常不留背景程序。若在受限沙箱內測試，Windows localhost socket 限制可能使 asyncio/TestClient 卡住，請允許本機 socket 或從一般 PowerShell 執行。

若 CUDA OOM，先降低 batch/context 或模型尺寸，使用 gradient accumulation 維持 effective batch。DPO 會同時保留 policy/reference；不要將 SFT 的記憶體需求當成 DPO 的需求。修改結構/context 後，以新的 run 重新準備資料；真實語料應依來源記錄授權、版本與清理規則。
