# Baseline discussion release: destination-conditioned PPO and Gen4 roadmap

本讨论版共享完整神经网络/PPO 成果、冻结模型和可复算证据。相对协作 main，还包含之前尚未推送的 M3–Gen3 实现及实验记录。

默认采用 **原目的地条件化 DELIVER 头 + Gen3 带近似平局回退的快速推理**。五个 learner 全部保留；演示固定最低编号 seed 1。紧迫性、PICK–DELIVER 协同、注意力、MLP/M4/M5 和规则对照均保留。

## 溯源及共享产物

- 冻结源码/推理基点：`d050c4c100b86bdb6676d57f400a24c9bc80e66b`。
- 默认权重训练 SHA：`705806aba80a06a992286464e31a363f698d726d`。发布整理 commit 由 PR head 标识，不冒充训练 SHA。
- `artifacts/checkpoint_manifest.json`：35 个原始 Dev-selected checkpoint 的 SHA256、各自训练源码、配置指纹、选中步数及 403/26 schema/源码 hash。
- `artifacts/selected_models.tar.gz`：35 个原始选中模型，含全部五个正式默认模型。
- `artifacts/daily_evidence.tar.gz`：288 份既有逐日结果、配置、源码清单、评估和更新日志。三个 Test 保持原名及独立统计。
- `environment.yml` / `requirements-lock.txt`：原精确 CPU 环境，不升级依赖。
- `docs/results_index.md`：主要结论与模型、协议、报告的对应关系。

归档随分支获取，不依赖开发者隐藏文件。中间/last 模型和完整训练 episode 流仍留在原本地目录；此包只共享正式选中模型与对应证据。

## 已有研究结论

同一 Test-v3：目的地 **86.00% ± 0.33pp**，紧迫性 **86.34% ± 0.40pp**，协同 **85.92% ± 0.42pp**。配对 95% 区间为紧迫性 **+0.34pp [−0.21,+0.90]**、协同 **−0.08pp [−0.43,+0.24]**；未证明总体收益，也不宣称严格等价。± 是五个 learner 的样本标准差。

既有孤立单线程 CPU benchmark：目的地 B1 p50/p95 从 **270.435/290.245µs** 到 **212.484/226.709µs**。20 个模型、两条可选路径、4,800 个完整 Dev 日动作/日级结果一致；不宣称浮点 logits 处处逐位相等。口径见 `results/gen3_inference.json`。Test-v3 原 5,400 episodes 无非法动作，它们不是 5,400 次独立训练。

本轮新分析只在 Dev：互斥 load × type × failure_mode 表，以及成功/失败/改善/退化四类共八条配对完整轨迹。High 候选超 12 状态约 6.88%，这批失败订单均曾可见；尚不支持认定 top-K 为主因。先查可见时机和取送时序。详见 `docs/discussion_plan_review.md`。

## 复现

依 README 创建 CPU 环境并安装项目后：

```bash
python scripts/discussion.py verify
python scripts/demo_network.py
python scripts/discussion.py dev
python scripts/discussion.py table
python -m pytest -q
```

默认完整 Dev 日核对带回退快路径、原推理及训练期归档结果。原网络 demo 执行采样、前后向、PPO 更新、保存重载。表格只读取旧记录，不执行 Test 策略；不同本地证据拒绝覆盖。

作者远端新 clone 验收见 `docs/discussion_reproduction.md`：200 项测试通过、固定 Dev 日与两份参考完全匹配、三个历史表已复算。隔离 venv 复用原精确依赖，未声称完成干净系统安装。GitHub Actions 以 PR 实际检查为准，本地测试不称 CI 已通过。另一名组员独立复现仍待完成。

## 研究边界与待讨论决定

R0 先复现发布；R1 先诊断及登记新开发清单；随后有限筛选候选窗口与公开信息短视规划。训练/容量、服务目标、泛化、小实例上界分别立项。28,770,304 transitions 是阶段预算上限，不要求用满。

原总体 Dev 选点与类型取舍报告继续有效，用户没有要求新增服务护栏。约 0.5pp 尚为实践门槛讨论起点。新 Dev-tune/confirm、确认 learner、最终约 200 天 Test 及解封负责人需先查重冻结；本轮没有生成或打开新 Test。

组会需决定优先级、允许的在线时延、独立复现及 Test 管理人。每个假设填写 `docs/EXPERIMENT_SPEC.md`；本 PR 不批准全部组合训练。

**保持 Draft；另一名组员复现并 review 后再打新 annotated tag 和 prerelease。本轮不创建标签、不合并。**

AI 使用：Codex 协助实现、实验、统计、诊断、打包及文档；小组负责审核代码与结论。
