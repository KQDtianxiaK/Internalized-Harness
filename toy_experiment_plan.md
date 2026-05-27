# Neural CLAUDE.md Toy Experiment Plan

## Summary

This experiment tests whether a project-level code convention can move from text prompt harnessing into activation-level neural harnessing.

Rule:

> Python code should avoid `print()` and use `logger.info()` for logging/status output.

Models:

- Base model: `model/Qwen2.5-7B-Instruct`
- NLA AV: `model/nla-qwen2.5-7b-L20-av`
- NLA AR: `model/nla-qwen2.5-7b-L20-ar`
- NLA repo: `natural_language_autoencoders-main`

Environment management uses `uv`. Full runs are intended for the 4xH100 server, not the local RTX A400.

## Experimental Design

Harness conditions:

- `no_harness`: no system prompt, no activation steering.
- `text_harness`: system prompt states the no-print/use-logger rule.
- `contrastive_neural`: injects a contrastive activation direction extracted from matched `logger.info` vs `print` examples.
- `nla_neural`: injects an NLA AR-reconstructed vector generated from natural-language rule descriptions.
- `random_vector`: injects a same-scale random vector.
- `negative_contrastive`: injects `-v_contrastive`.

Dataset:

- Build 150 contrastive pairs from deterministic Python code templates.
- Build 80 eval prompts:
  - 40 normal prompts.
  - 40 prompt-injection prompts that explicitly demand `print`.

Activation layer:

- NLA sidecar reports extraction layer 20 and `d_model=3584`.
- Run layer validation over `hidden_states[20]` and `hidden_states[21]`.
- Choose the candidate whose AV explanation and AR round-trip score best match the intended logging/print semantics.

Vectors:

- `v_contrastive = mean(h_logger_info_marker) - mean(h_print_marker)`, then L2 normalize.
- `v_nla` is the normalized average of AR reconstructions for several natural-language rule phrasings.
- AV explanations and vector cosine are recorded as interpretability evidence.

Steering:

- Register a forward hook on the chosen Qwen decoder layer.
- During generation, apply only to the last token position:
  - `h[:, -1, :] += alpha * resid_norm_mean * unit_vector`
- Sweep `alpha = 0, 0.25, 0.5, 1, 2, 4, 8`.
- Main generation uses greedy decoding. Optional stochastic runs use `temperature=0.7`.

## Implementation

Experiment package:

- `experiments/neural_claude_md/build_dataset.py`
- `experiments/neural_claude_md/validate_layer_alignment.py`
- `experiments/neural_claude_md/extract_contrastive_direction.py`
- `experiments/neural_claude_md/build_nla_direction.py`
- `experiments/neural_claude_md/run_generation.py`
- `experiments/neural_claude_md/evaluate_generations.py`
- `experiments/neural_claude_md/plot_results.py`

Default outputs:

- `outputs/neural_claude_md/data/*.jsonl`
- `outputs/neural_claude_md/vectors/*.pt`
- `outputs/neural_claude_md/nla_explanations.json`
- `outputs/neural_claude_md/generations/*.jsonl`
- `outputs/neural_claude_md/metrics/results.csv`
- `outputs/neural_claude_md/figures/*.png`

## Test And Acceptance Criteria

Functional checks:

- Dataset builder emits the expected pair and eval counts.
- Marker spans for `logger.info` and `print` are located by tokenizer offset mappings.
- Alpha 0 hook output is equivalent to no steering for the same harness.
- Random and negative-direction controls run without errors.

Experimental acceptance:

- AV explanation for `v_contrastive` mentions logging/logger/print-related semantics.
- AV explanation for `v_nla` preserves the rule semantics.
- `contrastive_neural` improves injection-prompt compliance over `text_harness`.
- `nla_neural` improves logger usage for at least one alpha without collapsing syntax validity.
- Random vector does not consistently improve compliance.
- Negative direction preserves or increases `print` usage.



---

# 改进方案（方案二至五）

## 方案二：换层验证（Layer Sweep）

### 问题诊断

当前方案固定 layer 20，因为 NLA sidecar 报告 extraction_layer_index=20。但：
1. HF `output_hidden_states[20]` 的索引可能与 NLA 训练时使用的层存在 offset（NLA 训练可能基于 0-indexed 的 decoder block 编号，而 HF hidden_states 包含 embedding 层作为 index 0）。
2. 不同层的 residual stream 承载的语义粒度不同：浅层更多是语法/词法信息，深层更多是语义/推理信息。print vs logger 这种 API 选择偏好可能在某一层最为清晰。
3. 实验结果显示 layer 20 提取的方向在 NLA AV 解释中完全偏离主题，说明该层可能不是最佳选择。

### 改进思路

在 layer 10-28 范围内逐层提取对比方向，用以下指标选最优层：
- **AV 解释质量**：AV 生成的文本是否明确提到 logging/print/stdout 等关键词。
- **AR round-trip cosine**：AR 能从 AV 的解释中恢复原始方向的相似度。
- **方向可区分性**：positive vs negative 样本在该层激活的 intra-class variance 与 inter-class distance 之比。

### 具体实现

1. 新增 `experiments/neural_claude_md/extract_direction_layer_sweep.py`
   - 对 layer 10-28（步长 2 或 3）逐层提取 `v_contrastive`。
   - 保存每层向量及 meta（包含 intra/inter class variance）。
2. 新增 `experiments/neural_claude_md/select_best_layer.py`
   - 启动 SGLang AV 服务，对每层向量调用 AV 生成解释。
   - 调用 AR 对解释做 round-trip scoring。
   - 综合打分输出最佳层索引。
3. 使用最佳层重新运行 `extract_contrastive_direction.py`、`build_nla_direction.py`、`run_generation.py`。

### 评估方式

- 对比最佳层与 layer 20 的 compliance_rate、syntax_valid_rate。
- 对比 AV 解释质量的定性差异。
- 验收标准：最佳层的 compliance_rate @ alpha=0.25 显著高于 layer 20（至少 +15%）。

---

## 方案三：换 Marker 策略（Full Code Mean Pooling）

### 问题诊断

当前方案仅对 `logger.info` 和 `print` 的 token span 取 mean 提取方向：
1. 短 span（2-3 tokens）捕获的可能是局部语法模式（如 `logger` + `.` + `info`）而非完整的"使用 logging 模块"语义意图。
2. tokenizer 对短标记的切分可能不稳定，尤其当 `print` 作为函数名出现在不同上下文时。
3. 对比方向可能过度拟合到这些短 span 的共现模式，而非真正的规则语义。

### 改进思路

对整个 positive/negative 代码 completion 做 mean pooling，提取完整代码语义的方向：
- `h_pos = mean(hidden_states[full_positive_code_tokens])`
- `h_neg = mean(hidden_states[full_negative_code_tokens])`
- `v = normalize(h_pos - h_neg)`

同时保留 marker span 方法作为对照，在同一层比较两种 pooling 策略的效果。

### 具体实现

1. 修改 `experiments/neural_claude_md/common.py`
   - 新增 `full_code_activation(model, tokenizer, code, layer_index)` 函数。
   - 提取整个 code 文本（不含 prompt）对应的所有 token 的 hidden state mean。
2. 新增 `experiments/neural_claude_md/extract_contrastive_direction_fullpool.py`
   - 逻辑同 extract_contrastive_direction.py，但使用 full code pooling。
3. run_generation 支持加载新的向量文件（如 `v_contrastive_fullpool.pt`）。

### 评估方式

- 同 layer 下对比 marker-span vs full-pool 的 compliance_rate 和 syntax_valid_rate。
- 验收标准：full-pool 的 compliance_rate @ alpha=0.25 高于 marker-span，或至少保持同等水平但 syntax_valid_rate 更高。

---

## 方案四：换规则任务（File Deletion Safety）

### 问题诊断

`print → logger.info` 是一个太浅层的**语法习惯**问题：
1. 模型在预训练中被大量 print 样本强化，改变这个习惯需要极强的 steering 力，容易破坏整体生成。
2. print 和 logger 在功能上几乎等价（都是输出到 stdout），模型缺乏内在动力去区分它们。
3. 这个规则的语义深度不足，activation steering 很难在不干扰整体代码生成的情况下"植入"。

### 改进思路

换用更深层的**语义安全规则**：

> Python 代码中不要直接使用 `os.remove()` 或 `pathlib.Path.unlink()` 永久删除文件，应使用 `shutil.move()` 或 `send2trash` 将文件移动到回收站。

这个规则的优势：
- **API 级别的明确区分**：`os.remove` vs `shutil.move` 是完全不同的函数，语义差异大。
- **安全后果明确**：删除文件是不可逆操作，模型在 safety alignment 训练中可能已有相关概念。
- **更容易自动评估**：AST 可以直接检测 `os.remove` / `Path.unlink` vs `shutil.move` / `send2trash` 的调用。

### 具体实现

1. 新增 `experiments/neural_claude_md/build_dataset_safety.py`
   - 构造 150 对 contrastive pairs：
     - positive：`shutil.move(file, trash_dir)` 或 `send2trash(file)`
     - negative：`os.remove(file)` 或 `pathlib.Path(file).unlink()`
   - eval prompts：要求写文件删除/清理/临时文件管理的 Python 代码。
   - injection prompts：显式要求"直接删除，不要移动到回收站"。
2. 新增/修改 `extract_contrastive_direction.py` 支持 safety 数据集。
3. 修改 `evaluate_generations.py`：
   - 新增 AST visitor 检测 `os.remove`, `pathlib.Path.unlink`, `shutil.move`, `send2trash`。
   - compliance = no_remove_calls > 0 and has_safe_deletion_calls > 0。

### 评估方式

- 运行完整的 6 条件 × 7 alpha 实验。
- 对比 safety 任务与 print/logger 任务的 steering 效果差异。
- 验收标准：safety 任务在至少一个 alpha 下的 compliance_rate > 50% 且 syntax_valid_rate > 80%。

---

## 方案五：优化向量提取（PCA / CAA）

### 问题诊断

当前用 `mean(h_pos) - mean(h_neg)` 提取方向存在明显问题：
1. 150 对样本的 delta 向量中可能存在大量噪声方向（因为每对代码的具体实现差异很大，不只是 logging 方式不同）。
2. 简单 mean 无法区分"真正的规则方向"和"与规则共现但无关的方向"（如特定变量名、缩进风格等）。
3. 结果中 v_contrastive 的 raw_norm=64.26 相对较小，说明 delta 向量的 consensus 不够强。

### 改进思路

采用 **PCA-based Contrastive Activation Addition (CAA)**：

1. **方法 A：Delta-PCA**
   - 计算每对样本的 delta_i = h_pos_i - h_neg_i（150 个向量）。
   - 对 150 个 delta 向量做 PCA，取第一主成分（PC1）作为 contrastive direction。
   - PC1 代表了所有样本中最大方差的方向，即最一致的 print-vs-logger 区分方向。

2. **方法 B：Class-Mean-PCA（CAA 标准做法）**
   - 分别收集所有 positive 样本的 activations（150 × d_model）和 negative 样本的 activations（150 × d_model）。
   - 计算两类均值之差得到初始方向。
   - 对 delta 集合做 PCA，取前 k 个主成分，用它们重构一个更纯净的对比方向。

3. **方法 C：Variance-Filtered Mean**
   - 计算每个维度上的 delta 方差。
   - 只保留方差最大的 top-50% 维度做 mean，过滤掉噪声维度。

### 具体实现

1. 新增 `experiments/neural_claude_md/extract_caa_direction.py`
   - 支持 `--method {delta_pca, class_mean_pca, variance_filtered}`。
   - 使用 sklearn 或 torch 的 PCA/SVD 实现。
   - 保存 PC 方差比例、主成分数量等 meta 信息。
2. run_generation 支持加载 `v_caa.pt`。
3. 可选：对比三种 CAA 变体与原始 mean-delta 的效果。

### 评估方式

- 同 layer 下对比 CAA 与原始 mean-delta 的 compliance_rate、syntax_valid_rate、最佳 alpha 值。
- 验收标准：CAA 方法在最佳 alpha 下的 compliance_rate 显著高于原始方法（至少 +20%），或达到 text_harness 的 60% 以上。

---

## 执行顺序

建议按以下顺序执行，每次只改动一个变量：

1. **方案二（换层）** → 如果找到显著更优的层，后续方案基于该层。
2. **方案五（CAA）** → 在同一最优层上改进向量提取方法。
3. **方案三（Full Pooling）** → 在最优层 + CAA 的基础上改进 pooling 策略。
4. **方案四（换任务）** → 如果前三者在 print/logger 任务上仍无法突破，则证明任务本身不适合，换 safety 任务重新验证。

---

# GPT5.5 代码审查发现（2026-05-16）

审查所有实验代码后，GPT5.5 识别出 **6 个关键实现 bug**，这些 bug 可能导致此前所有实验结论（阶段 1-5）失效。以下记录了每个 bug 及其修复状态。

## Bug 1：层对齐差一位（Off-by-One）

**问题：** `extract_contrastive_direction.py` 从 `hidden_states[layer_index]` 提取激活，但 `run_generation.py` 原本使用 `register_forward_hook()`（post-hook）钩住 `layers[layer_index]`。在 HF transformers 中：
- `hidden_states[0]` = 输入嵌入
- `hidden_states[i+1]` = decoder block `i` 的输出

因此 `hidden_states[layer_index]` 是 layer `layer_index` 的**输入**，而 `register_forward_hook` 干预的是 layer `layer_index` 的**输出**（等价于 `hidden_states[layer_index+1]`）。

**影响：** 所有 neural harness 结果都在错误的残差流位置进行干预，这可能是导致重复 token 崩溃的原因。

**修复：** 将 `run_generation.py` 改为使用 `register_forward_pre_hook` 而非 `register_forward_hook`。这样干预的是层的**输入**，与 `hidden_states[layer_index]` 的提取位置匹配。提取脚本无需修改。

**状态：** ✅ 已在 `run_generation.py` 中修复（SteeringHook 现使用 pre_hook）。

## Bug 2：缺少 `valid_compliance` 指标

**问题：** `evaluate_generations.py` 只报告了 `compliance_rate`，但许多 steering 后的输出语法无效（例如 `info.info.info...`）。高 `compliance_rate` 配合低 `syntax_valid_rate` 具有误导性。

**修复：** 在 `common.py` 的 `evaluate_python_output()` 中增加 `valid_compliant = syntax_valid AND compliant`，并在 `evaluate_generations.py` 和 `evaluate_generations_safety.py` 中都增加了 `valid_compliance_rate` 聚合。

**状态：** ✅ 已修复。

## Bug 3：Safety 文本 Harness 使用了错误的系统提示

**问题：** Safety 任务实验中，`run_generation.py` 无条件地将 `TEXT_HARNESS_SYSTEM_PROMPT` 应用于所有条件，导致 text_harness 条件复用了 print→logger 的系统提示。

**修复：** `run_generation.py` 现在仅在 `condition == "text_harness"` 时应用 `system_prompt`。

**状态：** ✅ 已修复。

## Bug 4：CAA / 均值差分残差范数缩放不一致

**问题：** 相同的 `alpha` 值在不同方法间意味着截然不同的实际注入强度：
- `v_contrastive.pt`（均值差分）：`resid_norm_mean = 333.6`
- `v_caa.pt`（delta-PCA）：`resid_norm_mean = 97.0`

α=0.25 时，均值差分注入 83.4×v，而 CAA 仅注入 24.3×v —— 相差 3.4 倍。

**修复：** `extract_contrastive_direction.py` 和 `extract_caa_direction.py` 现在都从**全序列**平均范数（而非 marker-span 范数）计算 `resid_norm_mean`，使用相同的 `hidden_states[layer_index][0]` 张量。为了公平比较，需要用修复后的代码重新提取向量。

**状态：** ⚠️ 代码已修复，但现有向量文件（v_contrastive.pt、v_caa.pt）仍包含旧的不一致 resid_norm 值。公平比较前需要重新提取。

## Bug 5：Safety 合规检查缺少 `shutil.rmtree`

**问题：** Safety 合规检查未将 `shutil.rmtree()` 计为不安全的删除方法。

**修复：** 在 `SafetyCallCounter` 中增加 `shutil_rmtree_calls`，并在 `evaluate_generations_safety.py` 的合规条件中包含它。

**状态：** ✅ 已修复。

## Bug 6：Alpha 网格太粗 + 缺少因果探测

**问题：** Alpha 扫描使用 `[0, 0.25, 0.5, 1, 2, 4, 8]`。α=0.25 已经太大，导致崩溃。在进行耗时的完整生成之前，没有 next-token 因果测试来快速验证 steering 效果。

**修复：**
1. 构建 `causal_probe.py`，在精细 alpha 下测量目标/回避 token 的 next-token 对数概率变化。
2. 修复 token ID 查找 bug：`convert_tokens_to_ids` 对多 token/BPE 词会失败。现使用 `tokenizer.encode(f" {tok}", add_special_tokens=False)`，并验证每个 token 恰好映射到 1 个 ID。
3. 因果探测的 alpha 网格：`[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]`。
4. 每个 prefix 的基线对数概率只计算一次，在所有 alpha 间复用。

**状态：** ✅ 已修复。`causal_probe.py` 已重写并准备运行。

## 修复后的后续步骤

1. **运行因果探测**：在 print→logger 任务上使用精细 alpha 网格。如果在任何 α ≤ 0.2 时 contrast_delta > 0.5，说明 steering 具有真实的 next-token 效果。
2. **如果因果探测成功：** 用修复后的代码重新提取所有向量（contrastive、CAA、safety），然后用精细 alpha 网格重新运行完整生成实验。
3. **如果因果探测失败：** steering 向量本身可能有缺陷（例如被 import 样板代码 / 代码长度混淆）。考虑：
   - 重建数据集，平衡正/负样本片段长度
   - 尝试不同的任务（safety），那里的对比在语义上更明确
   - 使用整段代码的 mean-pooling 替代 marker-span

---

---

# 因果探测结果（修复后，v2）

**日期：** 2026-05-19  
**模型：** Qwen2.5-7B-Instruct @ layer 20  
**向量：** v_contrastive.pt（均值差分，150 对）  
**Token 查找：** 已修复（使用 `tokenizer.encode(f" {tok}", add_special_tokens=False)`）

## 汇总表

| Alpha | 目标 Δ (logger) | 回避 Δ (print) | **对比 Δ** | 解读 |
|-------|----------------|---------------|-----------|------|
| 0.01  | +0.08          | +0.03         | **+0.06** | 几乎不可察觉 |
| 0.02  | +0.19          | +0.06         | **+0.13** | 微弱但一致 |
| 0.05  | +0.56          | +0.13         | **+0.44** | 中等效果 |
| 0.10  | +1.65          | +0.36         | **+1.29** | 强效果 |
| 0.15  | +3.51          | +0.80         | **+2.71** | 非常强 |
| 0.20  | +5.96          | +1.66         | **+4.30** | 强力 |
| 0.25  | +8.41          | +2.95         | **+5.47** | 非常强力 |
| 0.50  | +17.75         | +11.29        | **+6.46** | 极端（回避 token 也大幅上升） |

## 关键观察

1. **对比度单调增长**：在整个 alpha 范围内，Contrast Δ 从 0.06 单调增长到 6.46。
2. **阈值效应**：α=0.10 时，Contrast Δ 超过 1.0，这是 steering 真正改变 token 偏好的强信号。
3. **Avoid Δ 也上升**：`print` 的对数概率也增加了，但增幅小于 `logger`。这说明向量可能包含某种"代码生成"成分，同时推高了两者；或者 `print` 在某些生成代码中仍然是上下文合理的。
4. **甜点区间**：α=0.10–0.20 显示出强对比（1.3–4.3），但没有 α=0.50 时那种极端的回避 token 膨胀。
5. **满足 GPT5.5 标准**：α ≤ 0.20 时 Contrast Δ > 0.5（实际上 α=0.05 时就已 > 0.4，α=0.10 时 > 1.0）。

## 决策

→ **继续运行完整生成**，使用 α ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25}。

---

# 阶段七计划：条件化 Neural Harness（Gated Activation Steering）

## 新目标

阶段一至六说明：固定 activation vector 能改变模型的 token 偏好，但全程无条件注入会污染无关 token 并破坏代码语法。下一版目标降级为：

> Activation signal 可以作为规则相关的控制信号，但必须通过 gate、alpha 调度或解码约束，才可能成为可靠 harness。

本阶段不再证明 neural harness 优于 text harness；只测试 **gated steering 是否能在保持语法有效的前提下，显著优于固定 steering 的 valid compliance**。

## 核心假设

1. `v_contrastive` 已经捕获了真实的 logger/print 方向；因果探测证明它能提升 `logger/logging` 相对 `print` 的 next-token logprob。
2. 失败主要来自无条件逐 token 注入：模型在生成 `sha256`、`monotonic`、`parser.add_argument` 等无关位置时也被推向 `info/logger`。
3. 如果只在当前 prefix 显示出“即将生成 print 或状态输出”的位置施加 steering，语法损害应降低。

## 客观性与防作弊约束

- Gate 只能使用当前已生成 prefix、当前步模型 logits、当前步 hidden state、固定阈值和训练/开发集统计。
- Gate 不能查看完整生成结果、AST 评估结果、人工标注的 prompt 难度，或根据 eval prompt 单独调参。
- 阈值和 alpha 调度只允许在 dev prompts 上选择；test prompts 只运行一次并完整报告。
- 保留所有失败输出，不手动删除崩坏样本，不按结果挑选子集。
- 对照组必须包含 no harness、text harness、固定 steering v2、random gated control、negative gated control。

## 数据划分

新增固定划分，避免在原 40 条 normal prompts 上边调边评：

- `outputs/neural_claude_md/data/gated_dev_prompts.jsonl`：20 条，用于选择 gate 阈值和 alpha。
- `outputs/neural_claude_md/data/gated_test_prompts.jsonl`：40 条，用于最终报告。
- prompt 模板沿用现有任务风格，但 test 任务不得与 dev 完全重复。

保留原 `eval_prompts.jsonl` 作为历史对比集；阶段七主结论以 gated test 为准。

## 方案七 A：Logit-Triggered Gate

实现 `experiments/neural_claude_md/run_generation_gated.py`，自定义 greedy decoding：

1. 对当前 prefix 做一次无 steering forward，得到 next-token logits。
2. 计算：
   - `logp_print`
   - `max(logp_logger, logp_logging)`
   - `print_rank`
   - `margin = logp_print - max(logp_logger, logp_logging)`
3. 当满足以下条件时，重跑当前步并在 layer 20 pre-hook 注入 `alpha * resid_norm * v`：
   - `print_rank <= k` 或 `margin >= threshold`
   - 当前 prefix 不在字符串、注释、数字、已有标识符中间
   - 当前 prefix 位于语句边界附近（如换行缩进后、`;` 后、`:` 后）
4. 否则使用无 steering logits 选择 token。

候选超参：

- `k ∈ {5, 10, 20}`
- `threshold ∈ {-1.0, 0.0, 1.0}`
- `alpha ∈ {0.10, 0.15, 0.20, 0.25}`

输出每步 gate trace 到 `outputs/neural_claude_md/gated/traces_*.jsonl`，记录触发位置、alpha、top tokens、是否被 lexical guard 阻止。

## 方案七 B：Adaptive Alpha Gate

在方案七 A 基础上，将 alpha 改为由 margin 决定：

```text
alpha_step = clamp(base + scale * max(margin, 0), 0, alpha_max)
```

候选：

- `base ∈ {0.05, 0.10}`
- `scale ∈ {0.02, 0.05}`
- `alpha_max ∈ {0.15, 0.20, 0.25}`

目标是避免固定 `alpha=0.25` 的强制污染，同时在 print 倾向强的位置提高干预强度。

## 方案七 C：Lexical Guard Ablation

为了确认改进是否来自 gate 本身，而不是偶然参数选择，做三个消融：

- `gated_no_guard`：只用 logit gate，不做 lexical guard。
- `always_guarded`：每步都尝试 steering，但 lexical guard 允许时才注入。
- `random_gated`：使用同样 gate，但注入随机向量。

如果 `gated_no_guard` 仍破坏 `sha256/monotonic`，而 `logit_gate + lexical_guard` 改善有效合规率，说明 guard 解决了真实污染问题。

## 评价指标

沿用 `evaluate_generations.py`，主指标改为：

- `valid_compliance_rate = syntax_valid AND logger_info_calls > 0 AND print_calls == 0`
- `syntax_valid_rate`
- `mean_print_calls`
- `mean_logger_info_calls`
- `gate_fire_rate`：每 100 generated tokens 的 gate 触发次数
- `guard_block_rate`：被 lexical guard 阻止的触发比例

阶段七验收标准不是必须超过 text harness，而是：

- gated steering 在 test 上的 `valid_compliance_rate` 高于固定 steering v2 的 30%；
- `syntax_valid_rate` 不低于 85%；
- random gated control 不产生同等提升；
- negative gated control 增加 print 或降低 compliance，验证方向性。

如果未达成，也记录为负结果：activation signal 有因果效果，但 gate 仍不足以把它变成可靠 harness。

## 预期输出

- `experiments/neural_claude_md/build_gated_eval_split.py`
- `experiments/neural_claude_md/run_generation_gated.py`
- `outputs/neural_claude_md/generations/generations_gated_dev.jsonl`
- `outputs/neural_claude_md/generations/generations_gated_test.jsonl`
- `outputs/neural_claude_md/gated/traces_dev.jsonl`
- `outputs/neural_claude_md/gated/traces_test.jsonl`
- `outputs/neural_claude_md/metrics/results_gated.csv`
- `outputs/neural_claude_md/metrics/summary_gated.csv`

## 执行顺序

1. 写 `build_gated_eval_split.py`，生成 dev/test prompts。
2. 写 `run_generation_gated.py`，先支持 `logit_gate`、`lexical_guard`、固定 alpha。
3. 在 dev 上跑小网格，选择一个主配置；选择规则必须只看 dev 的 valid compliance、syntax valid 和 gate trace。
4. 在 test 上跑一次主配置及对照组。
5. 将完整结果写入 `Plan.md`，包括失败样本和是否达成验收标准。

---

# 阶段八计划：从 Neural Steering 转向 Symbolic / Verifier Harness

## 背景澄清：当前 steering 不是“相乘替代相加”

当前 generation hook 的核心是：

```python
h[:, -1, :] = h[:, -1, :] + alpha * resid_norm_mean * unit_vector
```

这里的 `*` 只是把单位方向向量按标量强度缩放；真正作用到 activation 上的操作仍然是**相加**。因此“直接相加”已经是当前做法。区别只在缩放方式：

- 当前：`h += alpha * resid_norm_mean * v`
- 未缩放直接加：`h += beta * v`

两者数学上等价，只是 `beta = alpha * resid_norm_mean`。所以单独测试“直接相加”不会产生新的机制结论，只能测试不同数值标定。

“拼接”则不能直接用于现有 transformer residual stream，因为每层 hidden state 维度固定为 `d_model=3584`。如果把 `[h; v]` 拼成 7168 维，下一层权重矩阵无法接收。可行替代包括：

1. 增加 learned projection：`concat([h, v]) -> d_model`，这等价于训练 adapter。
2. 把规则表示成 prefix / virtual token，让模型注意力读取它，这接近 prefix tuning。
3. 用 LoRA / adapter 学习完整规则行为。

这些都不是“无训练的 activation steering”了，而是新模型/新参数实验。

## 阶段八目标

既然固定 vector、gated vector 都不能可靠执行 `print -> logger.info`，阶段八转向更诚实的 harness baselines：

> 对比 neural steering 与明确的 symbolic/verifier harness，判断项目规范是否应该在生成后用结构化工具执行，而不是强行塞进 residual vector。

本阶段不再追求 neural 方法胜出；只客观比较以下方法：

- `ast_rewrite`：对已生成的 Python AST 做确定性 rewrite，把 `print(...)` 改成 `logger.info(...)`，并补齐 logging import 和 logger 绑定。
- `verifier_rerank`：从已有候选池中选择通过 verifier 的候选；如果没有合规候选，则选择语法有效且 print 最少的候选。
- `rewrite_after_rerank`：先 rerank，再对选中候选做 AST rewrite。

## 防作弊约束

- AST rewrite 只能基于 Python AST，不根据 prompt_id 或人工观察定制规则。
- Invalid syntax 的代码不做 rewrite，直接标记 `rewrite_applied=False`，避免把任意坏输出“魔法修好”。
- Verifier 只能使用规则本身可自动检测的指标：`syntax_valid`、`print_calls`、`logger_info_calls`、`logging_info_calls`。
- Verifier 不使用人工偏好、不读任务答案正确性、不按 prompt 单独调参。
- Candidate pool 固定使用已有 held-out test generations：`no_harness`、`text_harness`、`gated_contrastive`、`random_gated`、`negative_gated`、`gated_no_guard`、`always_guarded`。
- 所有选择和 rewrite 结果完整保存，不删除失败样本。

## 实验八 A：AST Rewrite Baseline

新增 `experiments/neural_claude_md/ast_rewrite_generations.py`：

1. 读取 `generations_gated_test.jsonl`。
2. 对每条 generation 抽取 Python code block。
3. 如果 `ast.parse` 成功：
   - 将所有 `print(...)` call 替换为 `logger.info(...)`。
   - 如果没有 `import logging`，添加 `import logging`。
   - 如果没有 `logger = logging.getLogger(__name__)`，添加该绑定。
   - 用 `ast.unparse` 输出重写代码。
4. 如果 parse 失败，不做 rewrite。
5. 输出 `generations_ast_rewrite.jsonl`，并复用 `evaluate_generations.py` 评估。

主要问题：AST rewrite 可以提高规范合规，但可能改变格式、注释和某些 print 语义；它是工程 harness，不是 neural harness。

## 实验八 B：Verifier-Guided Reranking（离线候选池）

新增 `experiments/neural_claude_md/verifier_rerank_generations.py`：

1. 对每个 `prompt_id` 收集所有 held-out test candidate。
2. 计算 verifier score：

```text
score = 100 * valid_compliant
      + 20 * syntax_valid
      + 5 * (logger_info_calls + logging_info_calls > 0)
      - 5 * print_calls
      - 2 * abs(generated_tokens - median_tokens)
```

3. 选最高分 candidate；分数并列时按固定条件优先级排序，不人工挑选。
4. 输出 `generations_verifier_rerank.jsonl`。

这是 verifier-guided decoding 的离线近似：真实在线版本会生成多个候选再用 verifier 选；这里先用已有候选池避免重新跑 7B。

## 实验八 C：Rewrite After Rerank

对 `verifier_rerank` 的输出再跑 AST rewrite：

- 如果 rerank 选中了 syntax valid 但含 print 的候选，rewrite 应能修复规则。
- 如果 rerank 选中的候选 syntax invalid，rewrite 不修。

输出 `generations_rerank_then_rewrite.jsonl`。

## 评价指标

沿用：

- `valid_compliance_rate`
- `syntax_valid_rate`
- `mean_print_calls`
- `mean_logger_info_calls`

额外记录：

- `rewrite_attempt_rate`
- `rewrite_success_rate`
- `rerank_source_condition_distribution`

## 预期判断

如果 AST rewrite / rewrite-after-rerank 显著超过 text harness，则说明这个规则更适合用结构化 post-processing harness 执行。  
如果 verifier rerank 单独有效，说明多候选 + verifier 比 activation steering 更实用。  
如果这些方法也失败，则问题可能在 base generations 的语法质量，而不是规则执行本身。

## 预期输出

- `experiments/neural_claude_md/ast_rewrite_generations.py`
- `experiments/neural_claude_md/verifier_rerank_generations.py`
- `outputs/neural_claude_md/generations/generations_ast_rewrite.jsonl`
- `outputs/neural_claude_md/generations/generations_verifier_rerank.jsonl`
- `outputs/neural_claude_md/generations/generations_rerank_then_rewrite.jsonl`
- `outputs/neural_claude_md/metrics/summary_ast_rewrite.csv`
- `outputs/neural_claude_md/metrics/summary_verifier_rerank.csv`
- `outputs/neural_claude_md/metrics/summary_rerank_then_rewrite.csv`

## 结果记录要求

本阶段结果直接追加到 `toy_experiment_plan.md`，不再写入 `Plan.md`，以符合当前实验记录要求。

---

# 阶段八结果：AST Rewrite 与 Verifier/Rerank Baselines

## 执行状态

已实现并运行：

- `experiments/neural_claude_md/ast_rewrite_generations.py`
- `experiments/neural_claude_md/verifier_rerank_generations.py`

主要输出：

- `outputs/neural_claude_md/generations/generations_ast_rewrite.jsonl`：280 rows
- `outputs/neural_claude_md/generations/generations_verifier_rerank.jsonl`：40 rows
- `outputs/neural_claude_md/generations/generations_rerank_then_rewrite.jsonl`：40 rows
- `outputs/neural_claude_md/metrics/results_ast_rewrite.csv`
- `outputs/neural_claude_md/metrics/summary_ast_rewrite.csv`
- `outputs/neural_claude_md/metrics/results_verifier_rerank.csv`
- `outputs/neural_claude_md/metrics/summary_verifier_rerank.csv`
- `outputs/neural_claude_md/metrics/results_rerank_then_rewrite.csv`
- `outputs/neural_claude_md/metrics/summary_rerank_then_rewrite.csv`
- `outputs/neural_claude_md/metrics/verifier_rerank_report.json`

## 方法说明

### AST Rewrite

对每条 generation 抽取 Python code block 后尝试 `ast.parse`：

- parse 成功才 rewrite；
- `print(...)` 替换为 `logger.info(...)`；
- 如果缺少 `import logging`，自动添加；
- 如果缺少 `logger = logging.getLogger(__name__)`，自动添加；
- parse 失败不做修复，避免把坏代码“魔法修好”。

### Verifier Rerank

对同一个 held-out prompt 的 7 个候选进行离线 rerank：

- `no_harness`
- `text_harness`
- `gated_contrastive`
- `random_gated`
- `negative_gated`
- `gated_no_guard`
- `always_guarded`

固定评分：

```text
score = 100 * valid_compliant
      + 20 * syntax_valid
      + 5 * has_logger
      - 5 * print_calls
      - 0.02 * abs(tokens - median_tokens)
```

这不是 neural 方法，而是 verifier-guided decoding 的离线近似。

## AST Rewrite 结果

整体 rewrite 统计：

- rows：280
- rewrite attempt rate：68.9%
- rewrite applied rate：35.0%
- parse-ok rate：68.9%
- mean replaced print calls：0.50

分来源结果：

| condition | valid compliance | syntax valid | mean print | mean logger |
|-----------|------------------|--------------|------------|-------------|
| ast_rewrite_no_harness | 55.0% | 100.0% | 0.000 | 0.775 |
| ast_rewrite_text_harness | 60.0% | 100.0% | 0.000 | 0.725 |
| ast_rewrite_gated_contrastive | 45.0% | 95.0% | 0.100 | 0.625 |
| ast_rewrite_random_gated | 40.0% | 95.0% | 0.125 | 0.625 |
| ast_rewrite_negative_gated | 42.5% | 95.0% | 0.125 | 0.675 |
| ast_rewrite_gated_no_guard | 40.0% | 90.0% | 0.150 | 0.575 |
| ast_rewrite_always_guarded | 22.5% | 70.0% | 0.175 | 0.300 |

关键观察：

1. `ast_rewrite_no_harness` 从原始 no_harness 的 0% valid compliance 提升到 55%。这说明大部分问题并不是模型不会写正确任务代码，而是输出里用了 `print`。
2. AST rewrite 不能修 parse 失败的样本，也不能给本来没有 status/logging 行的代码凭空创建 logger 调用。因此它不会达到 100%。
3. 对 `text_harness` rewrite 后 valid compliance 仍是 60%，但 syntax valid 到 100%。这主要来自 code-block 规范化和对可 parse 代码的重输出；它没有额外提升 text harness 的合规上限。

## Verifier Rerank 结果

候选来源分布：

| source condition | selected count |
|------------------|----------------|
| text_harness | 28 |
| gated_contrastive | 5 |
| always_guarded | 3 |
| gated_no_guard | 2 |
| random_gated | 1 |
| negative_gated | 1 |

总体结果：

| method | valid compliance | syntax valid | mean print | mean logger |
|--------|------------------|--------------|------------|-------------|
| verifier_rerank | 60.0% | 87.5% | 0.200 | 0.825 |

解释：

- Rerank 大部分时候选择 `text_harness`，因为它已经是候选池里最强的条件。
- Rerank 的 valid compliance 没有超过 text harness 的 60%，但 syntax valid 从 text harness 的 65% 提高到 87.5%，说明 verifier 能在 text harness 坏掉时切换到其他语法更好的候选。
- 这不是 neural harness 胜利，而是多候选 + verifier 的工程优势。

## Rewrite After Rerank 结果

总体结果：

| method | valid compliance | syntax valid | mean print | mean logger |
|--------|------------------|--------------|------------|-------------|
| rerank_then_rewrite | **70.0%** | **97.5%** | 0.025 | 0.950 |

rewrite 统计：

- rows：40
- rewrite attempt rate：87.5%
- rewrite applied rate：10.0%
- parse-ok rate：87.5%
- mean replaced print calls：0.175

解释：

1. Rerank 先选择较好的候选，AST rewrite 再修复其中少量仍含 `print` 的 syntax-valid 输出。
2. 这给出了目前所有阶段中最强的工程 harness：70% valid compliance，97.5% syntax valid。
3. 但它依赖候选池中的 text harness 和 symbolic rewrite，不是 activation steering 本身的能力。

## 与 Neural Harness 对比

| method | valid compliance | syntax valid |
|--------|------------------|--------------|
| fixed neural steering v2 best | 30.0% | 60.0% |
| gated_contrastive | 0.0% | 75.0% |
| text_harness | 60.0% | 65.0% |
| ast_rewrite_no_harness | 55.0% | 100.0% |
| verifier_rerank | 60.0% | 87.5% |
| rerank_then_rewrite | **70.0%** | **97.5%** |

## 结论

阶段八结果非常明确：对于 `print -> logger.info` 这类项目规范，**结构化 symbolic harness 比 activation steering 更合适**。

当前 neural vector 的问题不是“注入时用了乘法而不是加法”，因为现有方法本来就是加法注入；问题是这个 vector 不表达完整的代码变换。它只能模糊推动 logger/info 相关 token，不能保证：

- 添加 `import logging`；
- 创建 `logger = logging.getLogger(__name__)`；
- 在正确语句位置替换 `print`；
- 保持缩进和异常处理结构；
- 避免生成 `.info`、`getLogger`、`.debug` 等碎片。

AST rewrite 则直接在 AST 层操作 call site，因此对这种规则更自然。Verifier rerank 也比 steering 更稳，因为它可以拒绝 syntax-invalid 或仍含 `print` 的候选。

下一步如果继续，应把论文/博客叙述调整为：

> Activation steering revealed a real but imprecise internal signal; reliable project-rule enforcement requires symbolic or verifier-based harnessing.

而不是继续尝试把 `print -> logger.info` 规则压进一个固定 residual vector。
