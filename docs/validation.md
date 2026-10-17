# 构建、复现与证据边界

## 已经验证和本次未验证的内容

| 项目 | 状态 |
| --- | --- |
| 最终 kernel 与平台回读 SHA256 | 已核对，b1c5f8ac…881905，376141 字节 |
| v3 与原标签/历史记录 | 已核对，41a4fa4b…06cb30 |
| 历史在线训练 | 509159、721646、同源重交 826799 均 15/15 Pass，结果文件保留 |
| 逐点 CSV、统一参照重算、文件清单 | 可用 `python tools/verify_artifacts.py` 离线核对 |
| 模板生成器/校验器 | 本次进行普通 CPU 冒烟检查，仅验证模板工具链 |
| 额外 NPU runner / benchmark | 从原工程提取，提供入口；本次未执行 |
| 提取后 Ascend 原生编译、NPU 正确性、NPU 性能 | 本次未重跑 |
| 全部合法输入、正式赛果、整模型收益 | 无此验证结论 |

原始训练结果显式标记 `training_only=true`。公开仓库不把它改称正式比赛通过或获奖证据。本次为材料整理，没有启动云实例或新增平台提交。

## 环境要求

实际运行需要 Linux、匹配的 CANN/Ascend C（ASC CMake 包、ACL 运行库）、NPU 驱动和可访问的昇腾设备。模板默认目标架构是 `dav-2201`；本地补充验证历史中使用过 A3/910C，判题设备为群答复所述 910B4，两者结果不能互换。模板页历史显示 CANN 9.0.0；不能由此确认判题机每个底层库的精确构建版本。

Python 数据工具需要 NumPy，`requirements.txt` 固定为本次 CPU 检查所用的 2.4.6。不要因为有 Python golden 就认为 NPU 计算已被复现；它只是独立生成输入和参考输出。

## 普通电脑：核对导出材料

```bash
python tools/verify_artifacts.py
```

不需要 NumPy、CANN 或网络。检查文件清单/大小/SHA、基线与最终源码绑定、15 点状态与原始显示单位、CSV、163 条历史记录的统一参照分数及最终选版关系。

## NPU：模板 case 0

先在自己的工具链安装目录加载 `set_env.sh`，确认 `ASCEND_HOME_PATH` 有效。然后在仓库根目录：

```bash
python3 -m pip install -r requirements.txt
cd final
bash run.sh
```

该脚本会删除并重建当前工程自己的 `build/`，生成 `[1,7,1,16]` FP32 输入、执行 main，再按原模板 rtol/atol 校验结果。它仅覆盖 case 0，不运行平台隐藏 15 点。基线工程可进入 `baselines/v3` 执行同样入口，独立保存日志。

需要显式指定目标架构时，可手动构建：

```bash
cmake -S final -B final/build -DNPU_ARCH=dav-2201
cmake --build final/build -j4
```

运行程序还需要在其工作目录准备 `input/x.bin`、`input/positions.bin`；最省步骤的完整示例仍是 `run.sh`。此处命令是原工程提取后的复现入口，本次没有在 NPU 环境重新执行确认。

## NPU：补充边界与数值检查

从仓库根目录运行：

```bash
cmake -S final/bench -B final/bench/build
cmake --build final/bench/build -j4
python3 final/scripts/check_npu_cases.py --runner final/bench/build/rotary_pos_emb_validate
```

runner 接收 B/S/H/D、dtype、theta 和二进制输入/输出路径；RoPE 全部仍由 `kernel.asc` 完成。现有脚本针对 theta=10000，覆盖两种 dtype、多个非对齐维度/头数、长位置、零位置和 NaN/Inf 等；它不是覆盖全部 theta 的通用验证器。发生超时或设备错误后停止继续启动，以便检查现场。

脚本分别报告题面表格口径、答题页严格绝对/相对口径、模板 isclose 口径；历史两套文字说明存在差异，所以不要只读退出码。返回 1 也可能是某个严格本地口径未满足，须查看 report.json，不能直接归为平台失败或降低阈值掩盖差异。

`final/bench/benchmark.asc` 也予以保留，参数用法见源文件；普通 CPU 的执行时间、CPU API 替身、A3 的数据均不能作为判题 910B4 的提速结论。

## 再次实验时应记录什么

记录源码哈希、设备与软件版本、完整参数、输入种子、计时范围、预热/重复次数、每轮原始结果与正确性。比较小幅变化时使用同环境交错控制并报告波动。涉及数学顺序变化时按原参考与原容差验证，不能要求所有改动与旧版位同，也不能放宽规则来通过。
