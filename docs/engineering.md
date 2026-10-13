# MiniLLM AI Software Engineering System

本 repo 將 AI Agent 加入 SDLC，同時保存可執行的工程驗證。規範、權限、測試、人工決策各有責任。
目前新增的是本機開發與 GitHub CI、候選模型打包與本機驗證服務；GitHub 保護規則仍需啟用，
也沒有設定任何真正 Production 主機、帳號或自動部署。

## 1. 這次修正與作用

| 檔案／機制 | 変更與原因 |
|---|---|
| tests/fixtures.py、configs/ci.json | 在暫存目錄生成 synthetic data、tokenizer、隨機模型；不依賴 artifacts/smoke |
| tests/test_lab.py | 保留 masking、cache、LoRA、save/load、resume、API 測試；parent fixtures 是隨機模型，測試自己訓練 |
| tests/test_data.py | 不載入 torch 的資料轉換、精確去重、分組與重排穩定性測試 |
| tests/test_engineering.py | context/GQA、hash 竄改、路徑越界、發布 policy 失敗等測試 |
| tests/test_lifecycle.py | 從零完成 CPU pretrain/SFT/DPO/LoRA/nanoGPT/merge/evaluation 與候選包檢查 |
| scripts/verify.py、verify.ps1 | Fast／CPU／GPU 共用入口；非零 exit code 表示失敗；保存當次來源與 logs |
| scripts/gpu_smoke.py | laptop-check 結構、每階段兩 steps、暫存 run；無 CUDA 就失敗，不改用 CPU 假裝通過 |
| pyproject.toml、requirements-dev.txt | 固定 Ruff，先採 E9/F 正確性檢查；不對原始教學程式做大規模排版修改 |
| requirements-cpu.lock.txt | 固定與 CUDA 相同的非 torch 套件；CPU torch 從官方 CPU index 安裝 |
| bootstrap_upstreams.py | 重建固定 vendor commits，遇到既有不同版本／本機変更就停止，不 reset |
| minillm/common.py | provenance 增加 MiniLLM commit、dirty、工作區 source hash、Python/OS、lock hashes |
| minillm/data.py | 新 manifest 用 portable paths，記錄 tokenizer 全部檔案 hash；訓練前重算 prepared hashes |
| minillm/evaluate.py | 每份 stage evaluation 綁定實際 weights、tokenizer、manifest 與評估程式版本 |
| compare_runs.py | 評估單位不相容時回傳非零，避免直接排序不同 tokenizer／validation 的 loss |
| minillm/release.py、scripts/release.py | 產生不可覆寫的 full-model 候選包；檢查檔案、評估綁定與明確 thresholds |
| serve_release.py、api.py | 本機載入前檢查候選包；health/logs 帶 release ID，runtime logs 放包外 |
| AGENTS.md、CLAUDE.md、Skills | 共用開發契約與可重複工作流程；不代替 OS/GitHub 權限 |
| .github/ | CPU CI、Issue／PR 模板、CODEOWNERS；需要 GitHub ruleset 才會強制合併關卡 |

## 2. 本機設定與驗證

使用獨立 Python 3.12，Git 與 PowerShell。現有 .venv 可直接安裝 developer tool，無須重裝 CUDA：

```powershell
Set-Location C:\LLM\MiniLLM
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\verify.ps1 -Profile Fast
.\scripts\verify.ps1 -Profile CPU
.\scripts\verify.ps1 -Profile GPU
```

新的 Windows checkout：

```powershell
.\scripts\setup.ps1 -Python C:\Python312\python.exe -Development
# 無 GPU 或 CPU CI 工作區：加上 -CpuOnly。不要在原 CUDA .venv 上切換環境。
```

Linux／WSL2 的 CPU 環境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-cpu.lock.txt -r requirements-dev.txt
.venv/bin/python scripts/bootstrap_upstreams.py
.venv/bin/python scripts/verify.py --profile cpu
```

驗證內容：

| Profile | 內容 |
|---|---|
| Fast | Python/YAML/JSON/TOML 解析、config dimensions、upstream 版本/clean、CPU/CUDA pins 一致、Ruff、快速測試 |
| CPU | Fast + pip check + 自足行為測試 + 全 lifecycle 與候選包 smoke |
| GPU | CPU + CUDA 必須可用 + laptop-check 兩-step 四階段 + 固定評估/cache |

每次新建 `reports/verification/<profile-id>/summary.json` 和每個 check 的 log，這些輸出被 gitignore。
可指定未存在的 `-Output`／`--output` 路徑。來源包括 tracked/untracked 且未被忽略的檔案內容，
所以未提交変更也有 source hash。驗證期間來源改变會使整個驗證失敗。
本機輸出不是簽章／不可竄改的認證；PR 要搭配 GitHub 當前 commit 的 required checks。

Windows Codex sandbox 若限制 localhost socket，CPU TestClient 可能卡住；允許該次本機測試的
sandbox escalation，或在一般 PowerShell 執行。不要因 sandbox 問題移除 API 測試。

## 3. Branch、worktree、實驗隔離

```powershell
git switch main
git pull --ff-only
git switch -c fix/context-error
# 同時有另一位開發者／Agent 才需要另建 checkout：
git worktree add ..\MiniLLM-context -b fix/context-error-worktree main
```

新 worktree 沒有 .venv、vendor、artifacts：各自 bootstrap；不要複製原 venv。
模型若共用，用唯讀外部來源並核對 hash。每個實驗用新的 run ID；完成的 run 不原地修改。
同一個 checkout 同時只讓一個 Agent 寫入。AI 作者、AI reviewer 可以是不同工具，但都不算人工批准。

## 4. Codex 和 Skills 設定

repo 的 `.agents/skills/` 已加入兩個技能：

- `minillm-change-verification`：選 Fast/CPU/GPU，執行同一 verifier、回報當次證據。
- `minillm-experiment-review`：核對 parent/tokenizer/dataset/protocol，比較可比實驗。

從 repo root 啟動 Codex，查看 skills selector；若新檔案沒有出現，重啟該 session。
CLI/IDE 可輸入 `$minillm-change-verification`，App 可用其技能選擇器，或明確提供 SKILL.md 路徑。
即使某版本沒有自動發現 repo skills，也可以要求 Agent 讀取該檔；實際驗證由 scripts 決定。
技能只會在適用時載入，不授予 merge/deploy 權限。修改 `.agents/` 在 Codex 可能需要指令檔寫入批准。

`configs/codex.example.toml` 是範例，**不是已套用的有效設定**。按 installed client 的設定來源合併：

```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"
approvals_reviewer = "user"
[sandbox_workspace_write]
network_access = false
```

不要將此組配置與 `default_permissions` profile 混用。CLI 的一般用戶配置在 `~/.codex/config.toml`；
App／managed 工作區依實際設定來源與管理政策確認。Reasoning 可依任務從一般升到 high，
數值 loss／checkpoint 修正尤其適合較高 effort，但不取代測試。
`approval_policy=never` 只取消提示，不取消 sandbox；非互動 CI 可在有限權限下使用。
目前 CI 不呼叫付費 AI，也不需要 OpenAI／Anthropic API key。

官方：[Skills](https://learn.chatgpt.com/docs/build-skills)、
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、
[配置](https://learn.chatgpt.com/docs/config-file/config-reference)。

## 5. Claude Code 設定

`CLAUDE.md` 用 `@AGENTS.md` 共用規範。`.claude/skills/` 是薄入口，讀取同一套 `.agents/skills/`。
可輸入 `/minillm-change-verification` 或 `/minillm-experiment-review`；若沒有發現技能，重啟 session。
`.claude/settings.json` 已設定 acceptEdits、有限的驗證命令 allow、push/reset/clean ask 與 secrets Read deny。
執行 `/status` 確認載入來源，`/permissions` 確認實際 rule，`claude doctor` 檢查設定問題。
工具名稱／命令規則需要 installed Claude Code 版本支援；permission 未匹配時仍會要求批准。
個人例外放 `.claude/settings.local.json`，已 gitignore；團隊設定可被個人配置影響。

原生 Windows Claude shell 沒有 Bash OS sandbox。任意 Python subprocess 不受 Read deny 完整保護。
需要強隔離時用受支援的 WSL2 sandbox／VM／獨立帳號，且正式服務資料/credentials 留在 developer
工作區和帳號之外。Skills、MCP 和 file tools 的權限也應各自檢查。
不要將驗證 script 的 allow 解讀成 script 可以安全存取 Production；它仍執行 repo 程式。

官方：[Permissions](https://code.claude.com/docs/en/permissions)、
[Sandbox](https://code.claude.com/docs/en/sandboxing)、[Skills](https://code.claude.com/docs/en/skills)。

## 6. GitHub 上需要你啟用的設定

1. 將此分支經 review 推上 GitHub，建立 PR，讓 CI 首次產生 checks。
2. Settings → Rules → Rulesets：匯入 `configs/github-main.ruleset.json` 或按其內容建立。
3. 選 main、Active，required checks 為 `CPU (ubuntu-latest)` 與 `CPU (windows-latest)`。
   若 UI 能選 expected source，選 GitHub Actions；勾選 up-to-date。
4. Require PR、禁止 force push/deletion、過期 approval 失效、resolve review threads，限制 bypass。
5. CODEOWNERS 預設 repo owner `@George-wu509`；確認實際帳號有 write 權限，加入第二位 human reviewer。
6. 範例 ruleset 要求 1 approval 與最新 push 由他人批准。**PR 作者不能批准自己**。
   單人 repo 若沒有第二位 reviewer，先用 0 approvals、關閉 code-owner/latest-push approval，保留 PR/CI，
   把人工發布決策另行記錄；這不構成獨立雙人 review。不要匯入會讓自己永久無法合併的配置。
7. Rulesets/required reviewers 的可用性與 private/public、GitHub 方案有關，確認 UI 實际能力。

CI 只有 read token，checkout 不保留 credentials、Actions 固定已核對的 SHA，不接觸部署 secrets。
沒有 self-hosted laptop runner／pull_request_target／自動部署 workflow。
GPU 驗證先本機跑；若未來加 self-hosted runner，不允許任意未受信任 PR 在日常 laptop 上執行。

真正有部署目標後，建立 staging/production environments：限定 trusted release refs、人工 reviewers，
禁止自我批准/限制 admin bypass、環境 secrets 只在批准後提供。單人維護與方案限制需另外處理。
只部署同一份已核對的 immutable bundle，保留上一版本。不要在正式主機重新訓練或重解 dependencies。
此處是操作指南，新增檔案本身不會啟用 GitHub 伺服器保護。

官方：[Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)、
[Actions 安全](https://docs.github.com/en/actions/reference/security/secure-use)、
[Environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)。

## 7. 用 AI 增加功能：近似去重的完整例子

這是後續開發任務範例，**這次沒有實作近似去重**。

### 需求／設計階段

給 Codex 或 Claude：

```text
請為 MiniLLM 設計可選的字元 n-gram 近似去重，目前先不修改 code。
目標：不同 Lab question 編號但同題同答可判定冗餘；Calculate 1+2 和 1+3 必須保留。
預設關閉，維持目前 exact dedup 和 prompt group split。
請提出資料格式、threshold、刪除/保留策略、split leakage 防護與可觀察驗收案例。
僅使用 synthetic fixtures，不下載 embedding model。
```

人類確認驗收後，建立 feature/near-dedup 分支，給實作任務：

```text
依已確認的設計實作可選的近似去重，主要範圍 minillm/data.py、設定及資料測試。
驗收：預設輸出與現行 exact dedup 一致；相似但答案不同的數值題不被誤删；
同一相似群組不跨 split；manifest 記錄算法/threshold/移除數量；重排資料仍穩定。
保留 artifacts/smoke，使用新 synthetic run；不改 vendor、不做長訓練。
使用 $minillm-change-verification，完成 CPU 檢查，整理 diff、證據、限制和 PR 草稿。
```

Claude 將 `$...` 改成 `/minillm-change-verification`，或直接要求讀取該 skill。
Agent 應重現案例→實作→行為測試→verification；人類檢查資料刪除策略與測試是否真的符合需求。
必要時再授權新實驗，比較 raw/SFT/domain/task metrics，不能僅用去重數量判斷成功。

### Review／整合階段

可給另一個 Codex/Claude session 唯讀 review 任務：

```text
請 review feature/near-dedup 相對 main 的 diff，先不修改檔案。
重點：是否誤刪數字/否定語意、同群跨 split、threshold 是否記錄、是否削弱測試/CI。
每項問題附具體輸入、預期行為、檔案位置與影響；通過的部分也指出實際證據。
AI review 是建議，人工決定是否合併。
```

修正 review 問題後再次跑必要 checks；PR 使用模板，CI 針對最新 commit 執行。
沒有改模型服務就不需要發布新模型；資料變更通常需要新的 training run。

## 8. 用 AI 修 bug：context 限制的例子

```text
請調查 API 的 prompt+max_tokens 超過 context 時的回應。
先用 temporary random model 重現，確認包含 chat template 的 token 數。
驗收：context 邊界內允許，超過時回傳明確 400；max_tokens 無效回傳 422；
失敗請求不增加成功計數，服務仍健康；不默默截斷 prompt。
若現行程式已符合要求，就補缺少的邊界行為測試並說明，不要為了改 code 而改。
主要範圍 inference.py、api.py、API tests；使用 verification skill 完成 CPU 驗證。
```

這個任務不需要重新 pretrain/SFT；若修改 cache/model forward，才加 GPU profile。
測試應用 tokenizer 算實際長度，不依賴猜測「一個字等於一個 token」。

## 9. SDLC 每個階段誰做甚麼

| 階段 | AI Agent | 人類／系統 |
|---|---|---|
| 需求 | 整理 before/after、案例、未知問題 | 決定目標、資料授權、預算與驗收 |
| 設計 | 找影響檔案、相容性、測試/rollback 方案 | 決定有影響的產品/架構取捨 |
| 開發 | scoped edits、fixtures、行為測試 | 避免共用 checkout 多寫者；保留來源 |
| 驗證 | 呼叫 shared verifier、分析失敗 | CI 按當前 commit 重跑；控制規則不可隨意削弱 |
| Review | 找具體缺陷，提出修正 | Human review 與 merge 決策 |
| 發布 | 整理版本化候選、hash、評估與風險 | 檢查真正品質、批准部署、提供有限部署權限 |
| 維運 | 分析 release ID、錯誤/延遲/品質回饋 | 決定 rollback；新修正回到 Issue→PR |

文件變更可以本機 Fast；runtime 改動 CPU；loss/model/cache 改動加 GPU。
模型訓練是額外 pipeline，程式合併不會自動換掉正式模型。詳細發布命令見 release.md。
