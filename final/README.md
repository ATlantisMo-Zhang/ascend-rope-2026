# 最终保存工程

`kernel.asc` 是记录 680947 / 686549 / 721646 / 826799 对应的同一份源码，376141 字节，SHA256：

```text
b1c5f8ac1ecad92a872dfaa921688f6e8892a60c581348d78e7a42aac0881905
```

`main.asc`、构建文件、模板输入生成与验证脚本均按原文件复制。`bench/` 和 `scripts/check_npu_cases.py` 是比赛期间已有的额外验证/计时工具；本次没有改动其实现。`main.asc` 仅运行模板 case 0。

常规入口：配置 CANN 环境后 `bash run.sh`。它会重建当前目录的 `build/`；环境前提、额外验证命令及精度口径见[复现说明](../docs/validation.md)。最终核函数已存在于历史平台 Pass 记录中，本次提取的独立工程尚未重新进行 NPU 编译或执行。
