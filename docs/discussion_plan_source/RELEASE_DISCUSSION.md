# [Draft PR] Medical UAV：可复现目的地策略基线与下一阶段研究计划

> 这是一份可放入 PR 的讨论稿。事实来源为项目负责人汇报；推送者需补充本地完整 SHA、真实产物位置及新 clone 验收结果。此文件本身不表示已推送、已创建 PR 或已经通过 GitHub CI。

## 发布目的

先让组员拿到一个能够复现、能够讨论的稳定版本，再在独立研究分支继续探索。该版本为仿真研究原型，不是现实飞行或医疗服务安全认证。

## 建议默认策略

**原目的地条件化 DELIVER 头 + Gen3 带回退的等价快速推理路径。**

保留 MLP、M4/M5、单层注意力、紧迫性增强、PICK–DELIVER 协同及规则基线的配置和结果索引。不要因为部分新结构没有明确收益而删除它们。

## 代码与模型溯源

- 已汇报本地基点：`d050c4c`
- 完整源码 SHA：待推送者填写
- 本 PR 发布 commit：待生成
- 默认模型的训练代码 SHA：待填写（可能不同于快路径代码 SHA）
- 评估/推理代码 SHA：待填写
- 环境、观察、动作 schema/hash：待填写
- checkpoint 与配置的 SHA256：见待补充的 manifest
- 依赖：使用已验证的 CPU 环境文件和完整依赖清单，不临时升级版本

默认演示 seed 预先指定，不以 Test 最好成绩选取。正式评估保留所有五个 learner seeds。

## 已有证据

### Test-v3，同一场景清单

| 方法 | 等权准时率 | 相对原目的地头 |
|---|---:|---:|
| 原目的地头 | 86.00% ± 0.33 pp | — |
| 紧迫性增强 | 86.34% ± 0.40 pp | +0.34 pp；95% CI [-0.21, +0.90] |
| PICK–DELIVER 协同 | 85.92% ± 0.42 pp | -0.08 pp；95% CI [-0.43, +0.24] |

±为五个训练种子间的样本标准差，不是置信区间。两项新结构均未证明总体增益，也不能根据包含零的区间宣称严格等价或完全无价值。

### 工程汇报

- 196 项测试通过（当前本地记录，PR CI 结果另列）。
- 20 个模型、两条路径，共 4,800 个完整 Dev 日动作与日级结果一致。
- p50 推理耗时 270→212 μs；机器、线程、输入和统计口径请在 benchmark 记录中补齐。
- Test-v3 的 5,400 个完整日无非法动作；这些 episode 不是 5,400 次独立训练。

## 本 PR 不作的声明

- 不宣称该神经网络已达到最优。
- 不宣称所有负载、所有订单类型都有提升。
- 不把合法动作掩码提供的可行性称作模型独立学会的安全保证。
- 不跨 Test/Test-v2/Test-v3 拼接分数或比较延迟。
- 不把已使用的三个 Test 继续视作未来方法的未见测试集。

## 组员复现清单

- [ ] 从远端新 clone，不依赖开发者本机未跟踪文件。
- [ ] 使用提交的 CPU 环境安装，记录实际版本。
- [ ] 运行 README 已有且经核实的网络 demo。
- [ ] 运行本地对应的单元/集成测试，附输出和 commit。
- [ ] 加载默认 checkpoint，在固定开发日执行完整 episode。
- [ ] 从提交的逐日数据重算一张报告表。
- [ ] 检查快路径回退行为、动作一致性和 CPU 延迟记录。
- [ ] 确认无秘密、虚拟环境、敏感个人信息或大体积临时产物进入 Git 历史。
- [ ] 确认五种子模型/结果均可定位，不只共享“最好一个”。

## 需要组会讨论的决定

1. 是否采用上述默认策略作为本轮公开给组员的稳定版本？
2. 下一阶段优先“top-K候选瓶颈”与“短视rollout”，还是“服务水平约束”？
3. 哪些类别服务下降需要设置预先明确的容忍度？
4. 为更高服务质量允许多少额外在线计算量？
5. 谁负责独立复现、场景清单和最终 Test 解封？

## 后续 Issue 建议

| Issue 标题 | 优先级 | 完成产物 |
|---|---|---|
| Release: fresh-clone reproducibility and artifact manifest | P0 | 复现收据、完整hash、PR检查 |
| Research: failure taxonomy and candidate visibility audit | P0 | 开发订单级诊断与代表轨迹 |
| Research: K=12 vs K=24 / candidate selection | P1 | 接口版本、同条件多seed对照 |
| Research: public-information lookahead with fallback | P1 | 无泄漏测试、性能/计算开销曲线 |
| Research: training budget and actor–critic alternatives | P1 | 有限搜索、学习曲线、负结果 |
| Research: category service guardrails | P1 | 目标声明、trade-off与可行性 |
| Research: unseen maps and demand shifts | P2 | 零样本与域随机化分开评估 |
| Research: small-instance optimality diagnostics | P2 | 可验证解/有效上界、适用范围 |

本 PR 合并不表示批准所有后续实验组合。每个研究 Issue 需先填写 `EXPERIMENT_SPEC.md`。
