"""Build the final human-readable validation report from actual local artifacts."""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import jsonl, read_json, write_json


def main():
    environment = read_json(ROOT / "reports" / "environment.json")
    smoke = ROOT / "artifacts" / "smoke"
    comparison = read_json(smoke / "evaluation" / "comparison.json")
    manifest = read_json(smoke / "data" / "manifest.json")
    tests = (ROOT / "reports" / "tests.txt").read_text(encoding="utf-8-sig")
    assert "Ran 11 tests" in tests and "\nOK" in tests, "Test report did not pass."
    services = read_json(ROOT / "reports" / "services.json")
    assert services["health"]["status"] == "ok"
    samples = list(jsonl(smoke / "evaluation" / "generations.jsonl"))
    lora = {r["prompt"]: r["token_ids"] for r in samples if r["stage"] == "lora"}
    merged = {r["prompt"]: r["token_ids"] for r in samples if r["stage"] == "merged"}
    assert lora == merged and lora, "Merged LoRA generation differs from adapter inference."
    assert all(r["cache_greedy_equal"] for r in comparison["models"])
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M Asia/Taipei")
    lines = ["# MiniLLM 實測驗證", "", f"記錄時間：{now}", "",
             f"硬體：{environment['gpu']}，{environment['vram_gb']} GB VRAM，{environment['ram_gb']} GB RAM。CUDA matmul 已實際執行成功。", "",
             "## 已完成的訓練", "",
             "主模型直接執行 MiniMind 原始 Transformer，由隨機初始化開始；沒有載入外部預訓練權重。nanoGPT 使用同一份 train/validation token stream 作架構對照。", "",
             "| 實驗 | 實際參數量 | 實際 vocabulary | Pretrain steps | SFT steps | DPO steps | LoRA steps |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run_name in ("smoke", "laptop-check"):
        run = ROOT / "artifacts" / run_name
        cfg = read_json(run / "config.json")
        meta = read_json(run / "initialized" / "lineage.json")
        vocab = read_json(run / "data" / "manifest.json")["tokenizer"]["actual_vocab"]
        lines.append(f"| {run_name} | {meta['parameters']:,} | {vocab} | {cfg['pretrain_steps']} | {cfg['sft_steps']} | {cfg['dpo_steps']} | {cfg['lora_steps']} |")
    lines += ["", "每個 step 指 optimizer update；laptop-check 使用 batch=2、gradient accumulation=8、pretrain prediction window=256。MiniMind 使用 L+1 個輸入 token 計算 L 個 next-token targets。SFT/DPO/domain 教學配對通常比最大 context 短。", "",
              "laptop-check 保留 laptop 的 512 hidden / 8 layers 架構，只用 2 steps 驗證各階段。因合成語料的 BPE vocabulary 只有 835，實際是 24.03M 參數；6400 vocabulary 的 laptop 設定約 26.88M。尚未執行 laptop.json 的 2000/500/200/300 steps 長訓練。", "",
              "## 共同 held-out 評估", "", "| 階段 | Raw validation loss | Perplexity | SFT completion loss | Domain completion loss | Chosen win rate |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison["models"]:
        lines.append(f"| {row['stage']} | {row['raw_validation_loss']:.4f} | {row['raw_perplexity']:.2f} | {row['sft_completion_loss']:.4f} | {row['domain_completion_loss']:.4f} | {row['preference_chosen_win_rate']:.1%} |")
    nano = comparison["nanogpt_baseline"]
    lines.append(f"| nanoGPT (40 steps) | {nano['raw_validation_loss']:.4f} | {nano['perplexity']:.2f} | — | — | — |")
    lines += ["", "評估固定使用相同 raw windows 與全部 held-out pairs，FP32 且停用 TF32。以上不同欄位是不同 objective，不能互相視為同一 loss。Chosen win rate 使用 completion log probability sum，會受回答長度影響；它不是通用能力分數。DPO 的 reward margin 與訓練 loss 可在各 stage metrics 及 TensorBoard 查看。", "",
              "這批資料只有少量合成教學樣本，模型輸出仍可能是錯誤、重複或無意義文字。測試結果證明 lifecycle 實作與權重更新可運作，沒有證明模型具備實用聊天能力。", "",
              "## Dataset / tokenizer", "",
              f"Pretrain train tokens：{manifest['tokens']['train']:,}；validation tokens：{manifest['tokens']['validation']:,}。Tokenizer 只從 pretrain train partition 建立。", "",
              "| 資料 | Raw rows | Empty removed | Duplicate removed | Train | Validation |", "|---|---:|---:|---:|---:|---:|"]
    for kind, count in manifest["counts"].items():
        lines.append(f"| {kind} | {count['raw']} | {count['empty_removed']} | {count['duplicates_removed']} | {count['train']} | {count['validation']} |")
    lines += ["", "每個 dataset/tokenizer 的 SHA256 與來源路徑在 data/manifest.json。normalized exact 去重與按 prompt/document 的 hash split 已驗證沒有 partition 交集。TXT、JSONL、Parquet 輸入都已驗證。", "",
              "## GPU 記憶體實測", "", "| 實驗 | Stage | Peak torch allocated MiB |", "|---|---|---:|"]
    for name in ("smoke", "laptop-check"):
        for stage in ("base", "sft", "dpo", "lora"):
            metric = read_json(ROOT / "artifacts" / name / stage / "final_metrics.json")
            lines.append(f"| {name} | {stage} | {metric['peak_vram_mb']:.2f} |")
    lines += ["", "這是 torch.cuda.max_memory_allocated 的 tensor allocation 峰值，未包含 driver/context/allocator reserved memory，且只涵蓋短跑與實際短配對資料；不是長訓練或所有語料的顯存上限。", "",
              "## 行為與服務驗證", "",
              "- 11 項 unittest 全數通過，完整輸出在 reports/tests.txt。", "- CPU 上原生預訓練，以及 TRL SFT/DPO/LoRA 從 checkpoint 恢復後，最終所有權重 tensor 與不中斷訓練逐值相等。", "- SFT/LoRA 續訓的累積 token 計數也一致；不同回答長度的 gradient accumulation 與完整 batch gradient 一致。", "- Causal attention 不會看見未來 token，incremental KV cache logits 與整段 forward 一致。", "- 已驗證初始化權重 Normal(0,0.02)、保存/重載 logits，以及 LoRA 只更新 adapter。", "- Initialized/Base/SFT/DPO/LoRA/Merged 的 cached/uncached greedy token IDs 均一致。", "- LoRA adapter 與 merged 模型在三個固定 prompts 的 greedy token IDs 相同。", "- 本機 Uvicorn 已實際啟動，/health、/v1/chat/completions、/metrics HTTP 驗證成功。", "- 本機 TensorBoard 已實際啟動，HTTP 與 base loss/gradient/LR scalar tags 驗證成功。", "- pip check 通過，所有 Python 模組可編譯。", "",
              "API/TensorBoard 測試服務已自動停止；需要使用時依 README 啟動。Windows 沙箱會限制 localhost socket，因此 API/TestClient/HTTP 驗證是在允許的沙箱外完成。", "",
              "HF 的 tied embedding checkpoint 可能提示 lm_head.weight missing；該 tensor 與 embed_tokens.weight 共享權重，續訓逐值一致性測試已驗證正確恢復。", "",
              "## 版本與重現", ""]
    for name, version in environment["versions"].items():
        lines.append(f"- {name}: {version}")
    lines += ["", "Upstream commits：", ""]
    for name, commit in environment["upstream_commits"].items():
        lines.append(f"- {name}: `{commit}`")
    lines += ["", "完整版本在 requirements.lock.txt；環境偵測在 reports/environment.json；實際 HTTP 回應与 scalar tags 在 reports/services.json；階段比較與生成樣本在 artifacts/smoke/evaluation。操作命令與資料格式請看 README.md。", ""]
    (ROOT / "reports" / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(ROOT / "reports" / "validation_summary.json", {"time": now, "tests_passed": 11, "cuda_matmul": True, "api_http_verified": True, "tensorboard_http_verified": True, "all_stage_cache_equal": True, "lora_merge_greedy_equal": True, "synthetic_data_only": True, "long_laptop_training_completed": False})
    print(ROOT / "reports" / "validation.md")


if __name__ == "__main__":
    main()
