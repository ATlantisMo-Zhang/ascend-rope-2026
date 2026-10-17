# v3：首个 15 点全通过版本

该完整工程从原比赛仓库标签 `rotaryposemb-baseline-v3` 提取，kernel SHA256：

```text
41a4fa4bcad2f0ee3efc092c86aaa5068f28f85d04f32ebee96abcfed306cb30
```

历史在线训练记录为 509159。v3 使用两分量 FP32 算术处理频率精度，随后逐 token、逐 head 搬运并旋转，适合先理解完整计算过程。配套模板、构建脚本和 golden 同样从该标签提取。

在匹配 CANN/NPU 环境配置后于此目录运行 `bash run.sh`；会重建本目录的 `build/`。本次只核对文件与历史证据，未重新编译执行。完整逐点耗时见[记录](../../evidence/platform/509159.json)。
