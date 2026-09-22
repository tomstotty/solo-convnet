# solo-convnet

从零实现的卷积神经网络库，仅用 Python 标准库、不联网。

- 入口：`python convnet.py <子命令>`
  - 训练/评估：`train`、`evaluate`、`fitcnn`、`evalcnn`、`fitnorm`、
    `evalnorm`、`fitdata`、`evaldata`、`predictdata`
  - 检查点续训：`resumenorm INPUT EPOCHS OUTPUT`（`INPUT` 为 `-` 时从
    fitnorm 初值、start=0 开始；`EPOCHS` 为 `0` 或无前导零 ASCII 正整数；
    成功退出 0、参数数目错退出 2、其余失败退出 1）
  - 基准/校验：`benchmark`、`benchmark_batches`、`benchmark_data`、
    `gradcheck`、`convcheck`
- 训练必须完全确定：相同种子、相同数据与超参数产生逐字节相同的权重与指标。
- 数据集与权重全部来自仓库内的本地文件，不下载任何外部资源。

## 测试

    python -m unittest discover
