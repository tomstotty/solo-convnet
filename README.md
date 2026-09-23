# solo-convnet

从零实现的卷积神经网络库，仅用 Python 标准库、不联网。

- 入口：`python convnet.py <子命令>`
- 检查点续训：`python convnet.py resumenorm INPUT EPOCHS OUTPUT`（INPUT 为 `-` 时从 fitnorm 初值开始）。
- 数据绑定检查点续训：`python convnet.py resumedata DATA INPUT EPOCHS OUTPUT`（DATA 沿用 fitdata 契约，身份为原始字节 SHA-256；INPUT 为 `-` 时从 fitnorm 初值开始）。
- 多报告闸门趋势回归统计：`python convnet.py trendstats MANIFEST OUTPUT`。
- 趋势汇总可配置质量门禁：`python convnet.py trendgate STATS CONFIG OUTPUT`（STATS 为 trendstats 严格产物；全通过退出 0、未通过退出 3、参数错退出 2、其他输入/计算/I-O 错退出 1 且不改 OUTPUT）。
- 趋势审计（内存串联 trendstats + trendgate）：`python convnet.py trendaudit MANIFEST CONFIG OUTPUT`（MANIFEST 沿用 trendstats 输入契约、CONFIG 沿用 trendgate 配置契约，相对 CONFIG/OUTPUT 以 MANIFEST 目录解析；全程不落中间文件；全通过退出 0、未通过退出 3、参数错退出 2、其他输入/契约/路径/计算/I-O 错退出 1 且不改 OUTPUT）。
- 批量趋势审计：`python convnet.py trendauditbatch MANIFEST OUTPUT`（批清单唯一键 `audits`（≥2 项，项键序 name/manifest/config，三值非空 str，name 唯一）；manifest/OUTPUT 相对批清单目录解析，config 相对对应 manifest 目录解析，批清单、OUTPUT、全部 manifest/config 及子趋势文件两两互异；逐项内存执行 trendaudit 不落中间文件；OUTPUT 键序 audit_count/passed_count/results/pass，results 项键序 name/stats/gate/pass，stats/gate 为各项单独 trendaudit 同名对象的逐层等值副本；全通过退出 0、未全过退出 3、参数错退出 2、其他输入/契约/路径/计算/I-O 错退出 1 且不改 OUTPUT，合法未全过仍写盘）。
- 批量趋势审计回归对比：`python convnet.py trendauditbatchdiff BASELINE CURRENT OUTPUT`（BASELINE/CURRENT 均为 trendauditbatch 严格产物，audit_count 与外层 name 顺序须一致，对应项 stats 的 gate_count 及 results 内 name 顺序须一致；每项聚合 stats 内 ba/f1：regressions 求和、worst_delta 取最小值，regressions_delta/worst_delta_delta 均为当前汇总减基线汇总（按未舍入值计算），项 pass 仅当前者 ≤0 且后者 ≥0；OUTPUT 键序 audit_count/results/pass，results 项键序 name/regressions_delta/worst_delta_delta/pass，顶层 pass 为各项 pass 之与；三路径绝对化后两两互异；无回归退出 0、有回归退出 3、参数错退出 2、其他输入/契约/路径/计算/I-O 错退出 1 且不改 OUTPUT，合法有回归仍原子写盘）。
- 训练必须完全确定：相同种子、相同数据与超参数产生逐字节相同的权重与指标。
- 数据集与权重全部来自仓库内的本地文件，不下载任何外部资源。

## 测试

    python -m unittest discover
