# solo-convnet

从零实现的卷积神经网络库，仅用 Python 标准库、不联网。

- 入口：`python convnet.py <子命令>`
- 检查点续训：`python convnet.py resumenorm INPUT EPOCHS OUTPUT`（INPUT 为 `-` 时从 fitnorm 初值开始）。
- 数据绑定检查点续训：`python convnet.py resumedata DATA INPUT EPOCHS OUTPUT`（DATA 沿用 fitdata 契约，身份为原始字节 SHA-256；INPUT 为 `-` 时从 fitnorm 初值开始）。
- 末轮逐类统计报告：`python convnet.py valreport STATS VAL OUTPUT`（STATS 为 epoch≥1 的 version 3 resumevalstats 产物，VAL 沿用 fitdata 契约且字节 SHA-256 须匹配 val_sha；重算末轮混淆矩阵并与历史末项核对后输出 precision/recall/f1 等指标）。
- 训练必须完全确定：相同种子、相同数据与超参数产生逐字节相同的权重与指标。
- 数据集与权重全部来自仓库内的本地文件，不下载任何外部资源。

## 测试

    python -m unittest discover
