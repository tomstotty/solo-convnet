# solo-convnet

从零实现的卷积神经网络库，仅用 Python 标准库、不联网。

- 入口：`python convnet.py <子命令>`
- 检查点续训：`python convnet.py resumenorm INPUT EPOCHS OUTPUT`（INPUT 为 `-` 时从 fitnorm 初值开始）。
- 数据绑定检查点续训：`python convnet.py resumedata DATA INPUT EPOCHS OUTPUT`（DATA 沿用 fitdata 契约，身份为原始字节 SHA-256；INPUT 为 `-` 时从 fitnorm 初值开始）。
- 多报告闸门趋势回归统计：`python convnet.py trendstats MANIFEST OUTPUT`。
- 趋势汇总可配置质量门禁：`python convnet.py trendgate STATS CONFIG OUTPUT`（STATS 为 trendstats 严格产物；全通过退出 0、未通过退出 3、参数错退出 2、其他输入/计算/I-O 错退出 1 且不改 OUTPUT）。
- 训练必须完全确定：相同种子、相同数据与超参数产生逐字节相同的权重与指标。
- 数据集与权重全部来自仓库内的本地文件，不下载任何外部资源。

## 测试

    python -m unittest discover
