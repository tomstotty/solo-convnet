# solo-convnet

从零实现的卷积神经网络库，仅用 Python 标准库、不联网。

- 入口：`python convnet.py <子命令>`
- 检查点续训：`python convnet.py resumenorm INPUT EPOCHS OUTPUT`（INPUT 为 `-` 时从 fitnorm 初值开始）。
- 数据驱动检查点续训：`python convnet.py resumedata DATA INPUT EPOCHS OUTPUT`（DATA 为 fitdata 契约数据、以其字节 SHA-256 为身份，INPUT 为 `-` 时从 fitnorm 初值开始）。
- 训练必须完全确定：相同种子、相同数据与超参数产生逐字节相同的权重与指标。
- 数据集与权重全部来自仓库内的本地文件，不下载任何外部资源。

## 测试

    python -m unittest discover
