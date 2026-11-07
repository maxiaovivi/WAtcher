# 验证范围

本仓库区分接口测试、真实模型离线推理和机器人闭环验证。接口测试中的mock分数不用于准确率统计。

检查日期：2026-11-08（Asia/Singapore）。当前UR安全头的已有开发集结果见README；本次代码整理不会重新解释为独立安全验证。

| 检查 | 实际结果 |
|---|---|
| 本地安装 | 隔离Python 3.11环境中，editable安装成功 |
| 本地自动测试 | 56 passed、1 skipped；跳过项依赖本机未安装的PyAV，Brev已实际执行原生RGB预处理及WAM推理 |
| 原07安全头CPU等价 | 同一原权重和原归档输入，16个step、post、overall共18个概率最大绝对差0.0 |
| 真实WAM加载 | 匹配的pick A 14k任务、完整130个动作专家参数张量严格恢复；生成有限float32 `[16,7]` |
| 默认strict组合 | 实际夹爪最大值1.0052298307超界，返回POLICY_ERROR，未调用安全头、未返回approved_actions |
| 显式clip组合 | 复用上述唯一一次WAM生成的候选，在相同来源观测上先修正1个夹爪值，再真实调用安全头；6列速度保持逐值不变 |
| clip后模型门 | 18个风险值均有限，整体风险0.0695813894、阈值0.5，返回ALLOW；approved_actions与被评分候选一致 |

真实观测取自已存的同一段原生三视角记录，4帧时间使用原始实测遥测；没有把安全头128像素样例放大充当WAM原生图像。clip检查使用保留的原候选和同来源、相同帧/时间映射重建的观测，没有再次生成WAM动作。

`clip`是显式的动作后处理，会改变超界夹爪值，因此不宣称它与未处理的原始策略输出数值等价。只检查输入接口、模型装载、数值一致性和门控逻辑；这个单例的低风险不证明实际动作安全或任务成功。

CPU等价详情见 [safety_head_cpu_check.json](safety_head_cpu_check.json)。复验工具位于 `tests/check_safety_equivalence.py`、`tests/smoke_wam_record.py`、`tests/rescore_wam_record.py`，需自行提供原权重、原生记录和依赖；读取LeRobot parquet还需 `pyarrow`。这些脚本是手动选择运行的离线检查，不参与默认pytest或自动连接机器人。

本仓库未向机器人发送动作，也未验证停止、撤回、执行延迟或物理安全性能。
