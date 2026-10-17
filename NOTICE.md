# 来源与许可说明

本项目从参赛工程中提取，包含官方模板、比赛期间的算子实现与迭代修改，以及参考官方 SDK 的三角计算序列。原文件的来源与版权声明保留，不添加覆盖上游声明的许可证。

## 官方赛题模板

来源为参赛时取得的 RotaryPosEmb 空工程（模板归档名 `rotaryposemb_problem_1743_template.zip`），提取的模板文件为 main.asc、CMakeLists.txt、data_utils.h、run.sh 和 scripts 下的生成器、NumPy golden、校验器。原文件和注释按快照保留。版本来源及文件指纹见 `evidence/manifest.json`。

赛事仓库参考地址：<https://gitcode.com/cann-competition/2026CANNChallenge_beijing>。这是来源指针，本次未重新核实该页面或重新解释赛事公开发布条款。

## Ascend C SDK 参考

最终和部分候选 kernel 中 `RotaryPackedSdkTrig` 的多项式及 split-pi 序列参考 Huawei Ascend C SDK，固定 commit `1909ae266695b7d9ce40e62bb928c036719d9146`，路径为 `impl/adv_api/detail/math/{sin,cos}/*_common_impl.h`，原版权与来源注释保留。

`LICENSE` 从原工程保存的 `docs/evidence/candidates/sdk-packed-trig/UPSTREAM-LICENSE.txt` 原样复制，为 CANN Open Software License Agreement Version 2.0；适用于相应上游软件及其改编部分。文档没有对整个仓库作额外的统一许可承诺，公开发布仍需遵守实际适用的模板、赛事和上游条款。

## 实现与验证工具

kernel 的算术、布局、调度修改以及 bench 与额外检查脚本来自参赛工作区。未纳入其他队伍的源码；性能对照只保留一个真实公开排名记录的必要数值，并删除其他参赛者账号字段。

新整理的 Markdown、CSV、脱敏证据摘要和离线核验工具用于解释本项目。它们不构成 Huawei 或组委会背书，也不将平台在线训练反馈表述为正式比赛奖项。
