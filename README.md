# solo-gradebook

课程成绩统计与报表命令行工具，数据保存在工作目录下的本地文件中。

- 仅使用 Python 标准库，不联网。
- 入口：`python gradebook.py <子命令>`
- 分数为 0–100 的整数；加权汇总按项目权重计算，权重之和必须为 1。

## 测试

    python -m unittest discover
