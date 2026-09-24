"""convnet.py — 从零实现的卷积神经网络组件（仅 Python 标准库）。

当前提供：
- Conv2D 层：NCHW 嵌套 list、互相关（不翻转核）；stride 可为
  正 int 或 (SH, SW) tuple，padding 可为非负 int 或 (PT,PB,PL,PR) tuple，
  dilation 可为正 int 或 (DH, DW) tuple，groups 为正 int 分组数
  （默认 1，O 与 C 均须被其整除，weights 形状 [O][C/G][KH][KW]），
  padding_mode 为 "zeros"/"replicate"/"circular"/"reflect" 之一
  （默认 "zeros"，越界采样坐标的映射规则见 Conv2D 文档）。
- ConvTranspose2D 层：NCHW 嵌套 list 的转置卷积，weights
  [C][O/G][KH][KW]、bias [O]；stride 可为正 int 或 (SH, SW) tuple，
  padding 可为非负 int、(PT,PB,PL,PR) tuple 或 "same"/"valid"
  （数值规则同 Conv2D 的 stride/padding；"valid" 四边 0，"same" 按
  T=E+OP-S 解析四边、输出恰为输入乘步幅，详见 ConvTranspose2D 文档），
  dilation 可为正 int 或 (DH, DW) tuple（规则同 Conv2D 的 dilation），
  output_padding 可为非负 int 或 (OPH, OPW) tuple（OPH<SH、OPW<SW），
  groups 为正 int 分组数（默认 1，C 与 O 均须被其整除，输入通道 c
  仅连接同组输出 (c//(C/G))*(O/G)+oi）；forward 另接受 output_size=
  (OH, OW) 按本次输出尺寸反推输出补边（0≤OH-BH<SH、0≤OW-BW<SW），
  覆盖仅作用于本次调用；数值补边输出形状
  [(H-1)*SH-PT-PB+(KH-1)*DH+1+OPH]
  [(W-1)*SW-PL-PR+(KW-1)*DW+1+OPW]，"valid" 同此式（四边 0），
  "same" 输出形状 [N][O][H*SH][W*SW]。
- MaxPool2D 层：NCHW 嵌套 list、逐通道最大池化、补边位置不参与比较，
  dilation 支持膨胀采样（正 int 或 (DH, DW) tuple，默认 1），
  ceil_mode 可选末轴窗口向上取整（仅 bool，默认 False）。
- AvgPool2D 层：NCHW 嵌套 list、逐通道平均池化，dilation 支持膨胀
  采样（正 int 或 (DH, DW) tuple，默认 1），有效核长 EH/EW 决定窗口
  布局，count_include_pad 控制除数是否计入声明补边矩形内采样点，
  ceil_mode 同 MaxPool2D。
- AdaptiveAvgPool2D 层：NCHW 嵌套 list、逐通道自适应平均池化，输出尺寸
  为正 int 或 (OH, OW) tuple，分箱区间 [floor(oh*H/OH), ceil((oh+1)*H/OH))，
  OH/OW 可大于 H/W。
- AdaptiveMaxPool2D 层：NCHW 嵌套 list、逐通道自适应最大池化，输出尺寸
  为正 int 或 (OH, OW) tuple，分箱区间同 AdaptiveAvgPool2D，箱内按
  ih→iw 扫描取首个最大坐标，OH/OW 可大于 H/W。
- Flatten 层：NCHW 嵌套 list 展平为 [N][C*H*W]（按 c→h→w 顺序）。
- Linear 层：全连接，weights [O][I]、bias [O]，输入 [N][I] 输出 [N][O]。
- ReLU 层：逐元素 max(0, v)，限二维 [N][D]。
- Dropout 层：NCHW 嵌套 list，训练态按概率 p 置零并放大保留项，推理态原样复制。
- Dropout2D 层：NCHW 嵌套 list 的整通道 dropout，训练态按 n→c 每通道一次
  LCG 抽样，u<p 时整通道置 0、否则整通道除以 (1-p)，缓存 [N][C] 的
  0 或 1/(1-p) 通道掩码；推理态原样复制、不推进随机状态、缓存全 1。
- BatchNorm2D 层：NCHW 嵌套 list，逐通道批归一化；训练态按批次统计并更新
  running_mean/running_var，推理态使用运行统计仿射。
- SoftmaxCrossEntropy 层：二维 logits [N][K] 与 labels [N] 的加权
  softmax 交叉熵损失（可选 label_smoothing、class_weights、
  ignore_index），反向返回对 logits 的梯度（被忽略标签行为全 0）。

- 公开推理 API（仅标准库）：
- load_model(path)：严格校验 train 产物后返回键序为 values、bias 的新 dict。
- load_benchmark(path)：严格校验 benchmark 产物后返回键序为 model、metrics
  的新 dict（model 为全 float 契约，metrics 含 predictions）。
- predict_batch(model, x)：对 [N][1][1][4] 输入逐样本计算 logit，返回预测类别。
- predict_norm_batch(model, x)：对 [N][1][2][2] 输入按标准化七层网络
  （Conv2D→BatchNorm2D 保存统计→Dropout 推理恒等→MaxPool2D(2,2,0)
  →Flatten→Linear）逐样本推理，返回新元组 (predictions, logits)，
  依次为 int[N] 与 float[N][2]。

公开训练 API（仅标准库）：
- train_norm_step(layers, x, labels, lr=0.1)：对
  Conv2D→BatchNorm2D→Dropout→(MaxPool2D 或
  AdaptiveAvgPool2D)→Flatten→Linear→SoftmaxCrossEntropy
  七层按序前向、自损失层起逆序反传，做一次同步 SGD 更新，返回 float
  批均损失；任何异常都把七层恢复到入口状态。
- train_norm(layers, x, labels, epochs=20, lr=0.1)：连续训练 epochs 轮，
  返回各轮更新前批均损失的新 list[float]；末轮更新后仅以更新后的
  Conv2D 输出做一次训练态 BN 前向刷新运行统计；任何异常都把七层
  整体恢复到函数入口状态。
- train_norm_batches(layers, x, labels, batch_size=1, epochs=1, lr=0.1,
  seed=0, shuffle=True)：每轮按 [0,…,N-1]（shuffle 为真时以 seed
  起始、跨轮延续的 32 位 LCG 做 Fisher–Yates 洗牌）切分为大小
  batch_size 的批（末批可短），逐批调用 train_norm_step，返回按轮、
  批顺序排列、长度 epochs*ceil(N/batch_size) 的各批更新前批均损失
  新 list[float]；任何异常都把七层整体恢复到函数入口状态。
- train_deep_step(layers, x, labels, lr=0.1)：对
  Conv2D→BatchNorm2D→(Dropout 或 Dropout2D)→(MaxPool2D、
  AdaptiveAvgPool2D 或
  AdaptiveMaxPool2D)→Flatten→Linear→ReLU→Linear→SoftmaxCrossEntropy
  九层（双层分类头；索引 2 接受 Dropout 或整通道 Dropout2D，其余类型
  抛 TypeError）按序前向、自损失层起逆序反传，以新 list 同步
  更新 conv 权重/偏置、BN gamma/beta 及两个 Linear 权重/偏置（损失
  梯度已批均，不再除 N），返回更新前 float 批均损失；容器/成员类型
  错抛 TypeError，长度错或 BN/Dropout/Dropout2D 非训练态抛
  ValueError；成功
  保留更新与状态推进，任何异常都把九层（含参数引用）恢复到入口状态。
- train_deep_batches(layers, x, labels, batch_size=1, epochs=1, lr=0.1,
  seed=0, shuffle=True, clip=None, state=None, max_batches=None)：每轮按
  [0,…,N-1]（shuffle 为真时以 seed 起始、跨轮延续的 32 位 LCG 做
  Fisher–Yates 洗牌）切分为大小 batch_size 的批（末批可短），逐批按
  train_deep_step 次序前反向（九层结构同 train_deep_step，索引 2 接受
  Dropout 或 Dropout2D，第 4 层接受
  MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D，后者反向按首个
  最大坐标传梯度并累加重叠分箱），并在同步 SGD 更新前依次展平 conv
  权重/偏置、BN gamma/beta、两个 Linear 权重/偏置八组梯度，以
  sqrt(fsum(g*g)) 求裁剪前全局范数；clip 非 None 且范数大于 clip 时八组
  梯度同乘 clip/norm，更新不额外除批量。state 与 max_batches 均为 None
  时返回 (losses, grad_norms)，均为按轮、批顺序排列、长度
  epochs*ceil(N/batch_size) 的各批更新前批均损失与裁剪前范数新
  list[float]；任一非 None 时返回 (losses, grad_norms, state)，前两项仅
  记录本次实际训练的批，state 为 (epoch, order, cursor, rng) 进度四元组，
  支持批边界暂停/续训（max_batches 为 None 完成剩余批，否则为非负 int，
  0 不训练；state 为 None 等价 (0,[],0,seed)，仅在轮界洗牌），任意切分
  拼接与一次训练完全相同；任何异常都把九层（含参数引用）整体恢复到入口
  状态。
- train_deep_momentum_batches(layers, x, labels, batch_size=1, epochs=1,
  lr=0.1, seed=0, shuffle=True, clip=None, momentum=0.9, velocity=None,
  state=None, max_batches=None)：批次、洗牌、裁剪、暂停/续训与回滚均
  沿用 train_deep_batches，区别在每批先裁剪梯度 g，再逐叶执行
  v=momentum*v+g、p=p-lr*v，momentum 为 [0,1) 有限 int/float（拒绝
  bool），velocity 为 None（八组全零）或与 conv 权重/偏置、BN
  gamma/beta、两个 Linear 权重/偏置同序同形的 8 项 tuple；返回
  (losses, grad_norms, velocity, state)，grad_norms 仍记裁剪前范数，
  velocity 为推进后的 8 项 tuple，state 恒为沿用 train_deep_batches 语义
  的进度四元组（全部完成时为 (epochs, [], 0, rng)）；任何异常都把九层
  （含参数引用）与入参 velocity/state 原样恢复，任意批边界切分的拼接与
  一次训练完全相同。
- train_deep_adam_batches(layers, x, labels, batch_size=1, epochs=1,
  lr=0.001, seed=0, shuffle=True, clip=None, beta1=0.9, beta2=0.999,
  eps=1e-8, m=None, v=None, step=0, state=None, max_batches=None)：
  批次、洗牌、裁剪、暂停/续训、指标与回滚均沿用
  train_deep_momentum_batches，区别在每批先裁剪梯度 g，令 t=step+1，再
  逐叶执行一阶/二阶矩与偏差校正更新
  m=β1·m+(1-β1)·g、v=β2·v+(1-β2)·g²、
  p=p-lr·(m/(1-β1ᵗ))/(√(v/(1-β2ᵗ))+eps)；beta1/beta2 为 [0,1) 有限
  int/float、eps 为正有限 int/float、step 为非负 int（均拒绝 bool），
  类型错抛 TypeError、值错抛 ValueError；m/v 须同为 None（八组全零且
  step=0）或同为沿用 velocity 八项契约的 tuple（与 conv 权重/偏置、BN
  gamma/beta、两个 Linear 权重/偏置同序同形），配对或与 step 关系错抛
  ValueError；返回 (losses, grad_norms, m, v, step, state)，m/v 为推进
  后的 8 项新 tuple、step 为累计已更新批数的 int、state 恒为进度四元组
  （全部完成时为 (epochs, [], 0, rng)）；任何异常都把九层（含参数引用）
  与入参 m/v/state 原样恢复，任意批边界切分的两列表拼接、八组参数、两
  组矩、step 与层状态均与一次训练完成完全相同。
- train_deep_accum_batches(layers, x, labels, microbatch_size=1,
  accum_steps=2, epochs=1, lr=0.1, seed=0, shuffle=True, clip=None,
  state=None, max_updates=None, max_microbatches=None)：
  九层结构、顺序与 x、labels 异常沿用 train_deep_step（索引 2 接受
  Dropout 或 Dropout2D，第 4 层接受
  MaxPool2D、AdaptiveAvgPool2D、AdaptiveMaxPool2D，AdaptiveMaxPool2D
  反向按首个最大坐标传梯度并累加重叠分箱）；其余参数、异常与 LCG 洗牌
  沿用 train_deep_batches（microbatch_size、accum_steps 为正 int 且
  拒绝 bool，类型错 TypeError、非正 ValueError）。每轮切大小
  microbatch_size 的微批（末批可短），逐微批按 train_deep_step 次序
  前反向但不更新，把微批均值损失与八组梯度各乘微批样本数累加；满
  accum_steps 个微批或轮末时除以组内累计样本数得样本加权均值，按既有
  顺序展平八组均值梯度求裁剪前范数、可选裁剪并同步 SGD；余组不跨轮。
  state、max_updates、max_microbatches 均为 None 时返回
  (losses, grad_norms)，均为按更新记录的新 list[float]（长度
  epochs*ceil(ceil(N/microbatch_size)/accum_steps)），loss 为样本加权
  更新前均值、范数为裁剪前值；任一非 None 时启用暂停/续训，返回
  (losses, grad_norms, state)，前两项仅含本次完成的更新。max_updates、
  max_microbatches 各为 None 或非负 int（拒绝 bool），类型错 TypeError、
  负值 ValueError；0 不洗牌、不前向；两预算并用时任一耗尽即停（前者按
  更新计数、后者按前向微批计数）。state 四元组
  (epoch, order, cursor, rng) 表示更新边界（轮中 cursor 为小于 N 的
  microbatch_size*accum_steps 正倍数）；八元组在其后追加
  pending_samples, pending_microbatches, pending_loss, pending_grads，
  表示任意微批边界：cursor 为小于 N 的 microbatch_size 正倍数，
  pending_samples 为组内已累计样本正 int、pending_microbatches 为
  [1,accum_steps) 内 int、pending_loss 为有限 float（样本加权和）、
  pending_grads 为与八组参数同序同形的 8 项有限数 tuple（样本加权和）；
  容器或叶类型错抛 TypeError，长度、形状、排列、范围、非有限或字段不
  一致抛 ValueError；累计量不跨轮，轮末短组必结算。无累计量返回四元
  状态，否则返回八元状态。不修改入参 state，任意微批/更新边界分段的两
  列表拼接、八组参数、BN 统计、Dropout/Dropout2D 状态与终态均与一次
  训练完成完全相同。累计、求均值、范数或新参数非有限抛 ValueError；
  失败恢复入口
  状态与参数引用且不改 x、labels，成功保留 BN 统计与 Dropout/Dropout2D
  推进，相同入口确定。
- check_deep_gradients(layers, x, labels, eps=1e-6, atol=1e-6,
  rtol=1e-4)：以中心差分依次检验 x 与上述八组参数的数值梯度（索引 2
  接受 Dropout 或 Dropout2D，每次前向前恢复入口随机状态以重放同一
  掩码），损失、
  误差、容差判定与 (ok, max_e, max_r) 返回沿用 check_train_gradients；
  pool 为 AdaptiveMaxPool2D 时任一次前向中任一分箱并列最大抛
  ValueError；结束时（含异常路径）九层入口状态与参数引用整体恢复，
  结果确定。

命令行子命令（仅标准库）：
- `python convnet.py train OUTPUT`：在 data/tiny.csv 上训练“展平 + Linear”，
  将权重与指标以紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py evaluate WEIGHTS OUTPUT`：读取 train 产物 WEIGHTS，
  在 data/tiny.csv 上逐样本预测，将样本数、预测与准确率以紧凑 JSON
  原子写入 OUTPUT。
- `python convnet.py fitcnn OUTPUT`：在 data/tiny.csv 上训练
  “Conv2D(1×1,2 核) → MaxPool2D(2,2,0) → Flatten → Linear”，
  四特征重排为 [N][1][2][2]，将权重与指标以紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py evalcnn WEIGHTS OUTPUT`：读取 fitcnn 产物 WEIGHTS，
  以同一网络在 data/tiny.csv 上逐样本预测，将样本数、预测与准确率以
  紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py fitnorm OUTPUT`：在 data/tiny.csv 上训练
  “Conv2D(1×1,2 核) → BatchNorm2D(γ=1,β=0,eps=1e-5,momentum=1)
  → Dropout(0.25, seed=7) → MaxPool2D(2,2,0) → Flatten → Linear”，
  四特征重排为 [N][1][2][2]，将权重与指标以紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py evalnorm WEIGHTS OUTPUT`：读取 fitnorm 产物 WEIGHTS，
  以保存的 BN 运行统计在推理态网络上逐样本预测，将样本数、预测与
  准确率以紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py benchmark OUTPUT`：训练沿用 data/tiny.csv，另以与其
  逐字节相同的 data/tiny-val.csv 为验证集（各自独立加载、绝不混用）；
  以 fitnorm 初值、七层配置（Dropout seed=7）调用 train_norm 训练
  20 轮、lr=0.1，要求更新前批均 loss 末项小于首项、六组参数至少一组
  改变；model 在写出前与序列化重载后均须通过严格契约校验——沿用
  fitnorm 逐层键序与形状，且 conv、batchnorm、linear 的每个叶值都是
  有限 float（拒绝 int/bool），随后以 BN 保存统计、Dropout 推理态在
  验证集预测（最大 logit 平局取小类），预测须为 [0,1]、accuracy 为
  1.0。OUTPUT 顶层键依次为 model、metrics；model 沿用 fitnorm 逐层
  键序与形状、全叶值有限 float；metrics 键依次为 epochs、lr、seed、
  loss、predictions、accuracy，取值依次为 int 20、float 0.1、int 7、
  有限 float[20]、恰为 [0,1] 的 int 列表、float 1.0。紧凑 UTF-8 原子
  写盘，float 固定 12 位、负零归零、禁非有限、末尾 LF，相同基线逐字节
  一致；任一数据、计算、阈值、重载、路径冲突或 I/O 失败退出 1 且不改
  OUTPUT。
- `python convnet.py benchmark_batches OUTPUT`：训练沿用 data/tiny.csv，
  另以与其逐字节相同的 data/tiny-val.csv 为验证集（各自独立加载、绝不
  混用）；以 fitnorm 初值、七层配置（Dropout seed=7）沿七层分批训练 API
  原样调用 train_norm_batches(layers, x, labels, batch_size=1,
  epochs=20, lr=0.1, seed=7, shuffle=True)，共 20 轮 × 2 批 = 40 项
  更新前监控 loss（每批任何前向/更新前，以当前参数与 BN 运行统计在完整
  训练集上做推理态批均 softmax 交叉熵；batch_size=1 时批内 BN 方差恒 0，
  逐批训练损失不可用于衡量收敛）。要求 40 项 loss 末项严格小于首项、六组
  参数至少一组改变；末批更新后按 fitnorm 同法以完整训练集训练态 BN 前向
  刷新最终运行统计。model 写出前与序列化重载后均须通过严格契约校验，随后
  以重载权重、BN 保存统计、Dropout 推理态在验证集预测（最大 logit 平局取
  小类），预测须为 [0,1]、accuracy 为 1.0。OUTPUT 顶层键依次为 model、
  metrics；model 沿用 fitnorm 逐层键序与形状、全叶值有限 float；metrics
  键依次为 epochs、lr、seed、batch_size、shuffle、loss、predictions、
  accuracy，取值依次为 int 20、float 0.1、int 7、int 1、bool 真、有限
  float[40]、恰为 [0,1] 的 int 列表、float 1.0。紧凑 UTF-8 原子写盘，
  float 固定 12 位、负零归零、禁非有限、末尾 LF，相同基线逐字节一致；任一
  数据、计算、阈值、重载、路径冲突或 I/O 失败退出 1 且不改 OUTPUT。
- `python convnet.py benchmarkdeep OUTPUT`：训练沿用 data/tiny.csv，另以
  与其逐字节相同的 data/tiny-val.csv 为验证集（各自独立加载、绝不混用，
  沿用 benchmark 数据契约）；九层网络前五层（Conv2D→BatchNorm2D→
  Dropout→MaxPool2D→Flatten）沿用 fitnorm 初值（Dropout seed=7），后接
  Linear（2×2 单位权重、零偏置）→ReLU→Linear（沿用 _CNN_LINEAR_INIT、
  零偏置）→SoftmaxCrossEntropy，原样调用
  train_deep_momentum_batches(batch_size=2, epochs=20, lr=0.1, seed=7,
  shuffle=True, clip=1.0, momentum=0.9)，共 20 项更新前批均 loss。要求
  20 项 loss 末项严格小于首项，卷积权重及两个 Linear 权重三组均须改变；
  末批更新后按 fitnorm 同法以完整训练集训练态 BN 前向刷新最终运行统计，
  随后以 BN 保存统计、Dropout 推理态在验证集预测（最大 logit 平局取小
  类），预测须恰为 [0,1]、accuracy 为 1.0。OUTPUT 键依次为 loss、
  predictions、accuracy，取值依次为有限 float[20]、恰为 [0,1] 的 int
  列表、float 1.0。紧凑 UTF-8 原子写盘，float 固定 12 位、负零归零、
  末尾 LF，重复运行逐字节一致；任一数据、计算、阈值、路径冲突或 I/O
  失败退出 1 且不改 OUTPUT。成功退出 0、参数数目错退出 2。
- load_benchmark(path)：读取完整 benchmark 产物，返回键序为 model、
  metrics 的新 dict（所有嵌套 dict/list 均深拷贝）；model 严格沿用
  benchmark 修复后的逐层键序、形状与全 float 契约，metrics 严格校验
  上述键序、类型、形状/长度与取值。path 或 JSON 容器/标量类型错抛
  TypeError；文件不可读抛 OSError；非法 UTF-8 抛 UnicodeDecodeError；
  JSON 语法、重复/缺失/额外/错序键、形状、长度、取值或非有限错抛
  ValueError。
- load_benchmark_batches(path)：读取完整 benchmark_batches 产物，返回
  键序为 model、metrics 的新 dict（所有嵌套 dict/list 均深拷贝）；model
  严格沿用 benchmark 的逐层键序、形状与全 float 契约，metrics 严格校验
  键序、类型、形状/长度与取值（epochs、lr、seed、batch_size、shuffle、
  loss[40]、predictions、accuracy）。异常类型同 load_benchmark。
- `python convnet.py fitdata DATA OUTPUT`：读取本地 UTF-8 JSON DATA
  （键依次为 x、labels，x 为有限数的规则 list[N][1][2][2]、N≥2，
  labels 为等长 list、元素为 int 0/1），以 fitnorm 初值与七层配置
  （Dropout seed=7）调用 train_norm(epochs=20, lr=0.1)，要求末项 loss
  小于首项、六组参数至少一组改变、推理态重建网络 accuracy 为 1.0，
  将与 fitnorm 同构的权重与指标（loss 换成本次 20 项）以紧凑 JSON
  原子写入 OUTPUT。
- `python convnet.py evaldata WEIGHTS DATA OUTPUT`：读取 fitnorm 或
  fitdata 产物 WEIGHTS（校验同 evalnorm，metrics 取值不参与推理），
  以保存参数与 BN 运行统计重建推理态网络（BN/Dropout 切推理态），
  对 fitdata 契约的 DATA 逐样本取最大 logit 预测（并列取小类），
  将样本数、预测与准确率以紧凑 JSON 原子写入 OUTPUT。
- `python convnet.py predictdata WEIGHTS DATA OUTPUT`：读取 fitnorm 或
  fitdata 产物 WEIGHTS（校验同 evalnorm，metrics 合法值不参与推理），
  以保存参数与 BN 运行统计重建推理态网络（BN/Dropout 切推理态），对仅含
  顺序键 x（有限 int/float、拒绝 bool 的规则 list[N][1][2][2]、N≥1）的
  DATA 逐样本取最大 logit 预测（并列取小类），将 sample_count、
  predictions（int[N]）与 logits（float[N][2]）以紧凑 JSON 原子写入
  OUTPUT；成功退出 0、参数数目错退出 2、其余失败退出 1 且不改 OUTPUT。
- `python convnet.py gradcheck CONFIG OUTPUT`：读取 UTF-8 JSON CONFIG
  （键依次为 x、labels、eps、atol、rtol，x 为有限数的规则
  list[N][1][2][2]、N≥1），新建与 fitnorm 初始参数、层配置相同的七层
  训练链并以 SoftmaxCrossEntropy 为损失层调用 check_train_gradients，
  将 ok、max_e、max_r 以紧凑 JSON 原子写入 OUTPUT；ok 真退出 0、
  假退出 1，其余失败退出 1 且不改 OUTPUT。
- `python convnet.py convcheck CONFIG OUTPUT`：读取 UTF-8 JSON CONFIG
  （无重复键，键依次为 weights、bias、stride、padding、x、dy、eps、
  atol、rtol；stride 为恰含两项正 int 且至少一项大于 1 的 list，
  padding 为恰含四项非负 int 且至少一项大于 0 的 list，成员拒绝
  bool；四张量与三容差沿用 Conv2D、check_gradients 契约），stride、
  padding 转 tuple 后构造零补边、dilation=groups=1 的 Conv2D，取
  forward(x) 与以同形 dy 的 backward(dy)，再以同参新层执行
  check_gradients(layer, x, dy, eps, atol, rtol)；将 forward、
  backward{dx,dweights,dbias}、check{ok,max_e,max_r}（张量叶值均为
  有限 float）以紧凑 JSON 原子写入 OUTPUT；ok 真退出 0、假退出 1
  （仍写盘），CONFIG/OUTPUT 同路径或其余失败退出 1 且不改 OUTPUT。
- `python convnet.py resumenorm INPUT EPOCHS OUTPUT`：从检查点续训
  fitnorm 同构的七层网络。INPUT 为 "-" 时以 fitnorm 初值新建训练态
  网络、start_epoch=0；否则读取其全部字节交 load_norm_checkpoint 加载，
  start_epoch 取检查点 epoch。EPOCHS 须为 "0" 或无前导零 ASCII 正整数；
  沿用 fitnorm 对 data/tiny.csv 的严格校验，以 lr=0.1 调用 train_norm
  追加 EPOCHS 轮（0 轮不训练、不刷新统计，状态原样保持）。OUTPUT 写出
  dump_norm_checkpoint(layers, start_epoch+EPOCHS) 的字节，原子替换；
  成功时标准输出为紧凑 JSON 加 LF，键依次为 start_epoch、added_epochs、
  loss，前两者为 int，loss 为本段逐轮更新前批均损失的 float 列表
  （固定 12 位小数、负零归零）。分段续训拼接的 loss 与最终检查点分别
  与总轮数连续训练逐项、逐字节相同。输入不可读、契约或轮数非法、非有限
  计算、INPUT/OUTPUT 路径冲突或 I/O 失败均退出 1，标准输出为空且不改
  OUTPUT；参数数目错退出 2。
- `python convnet.py resumedata DATA INPUT EPOCHS OUTPUT`：DATA 沿用
  fitdata 契约，其身份为文件原始字节的 SHA-256 小写 hex。INPUT 为 "-"
  时以 fitnorm 初值新建训练态网络、start_epoch=0；否则加载 resumedata
  检查点（顶层键依次为 version、data_sha256、epoch、model、
  dropout_state；version 为 int 1；data_sha256 须与 DATA 摘要逐字符
  相同；model/dropout_state 沿用 dump_norm_checkpoint 契约），
  start_epoch 取检查点 epoch。EPOCHS 须为 "0" 或无前导零 ASCII 正整数；
  以 lr=0.1 在该 DATA 上调用 train_norm 追加 EPOCHS 轮（0 轮不训练、不
  刷新统计，状态原样保持）。OUTPUT 写出 dump_data_checkpoint(layers,
  start_epoch+EPOCHS, data_sha256) 的字节，原子替换；成功时标准输出为
  紧凑 JSON 加 LF，键依次为 start_epoch、added_epochs、loss，前两者为
  int，loss 为本段逐轮更新前批均损失的 float 列表（固定 12 位小数、负零
  归零）。同一 DATA 分段续训拼接的 loss 与连续训练逐项相同，最终检查点
  逐字节相同；检查点摘要不符（含 DATA 换用其他数据）一律拒绝。DATA/
  INPUT 不可读、契约/轮数/摘要非法、非有限计算、DATA/INPUT/OUTPUT 路径
  冲突或 I/O 失败均退出 1，标准输出为空且不改 OUTPUT；参数数目错退出 2。
- dump_norm_checkpoint(layers, epoch)：序列化 fitnorm 训练态七层网络与
  epoch 为紧凑 JSON bytes（末尾 LF）；load_norm_checkpoint(data) 从其
  bytes 重建无缓存训练态七层网络与 epoch，拒绝 JSON 常量
  NaN/Infinity/-Infinity，hex 叶值仅接受 dump_norm_checkpoint 的规范
  小写串（负零仅 "0x0.0p+0"），其余契约/类型/形状错误分别抛
  ValueError/TypeError/UnicodeDecodeError。
- dump_data_checkpoint(layers, epoch, data_sha256) 与
  load_data_checkpoint(data, data_sha256)：resumedata 使用的数据绑定
  检查点，model/dropout_state 契约同 norm 检查点，顶层键依次为
  version（int 1）、data_sha256（64 位小写 hex）、epoch、model、
  dropout_state；加载时摘要须逐字符相符，否则抛 ValueError。
"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile


def _is_number(value):
    """合法元素：有限 int/float，拒绝 bool。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_element(value, name):
    if isinstance(value, list):
        raise ValueError("%s 的层级过深：标量位置出现了 list" % name)
    if not _is_number(value):
        raise TypeError(
            "%s 的元素必须是 int/float（拒绝 bool），得到 %s"
            % (name, type(value).__name__)
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s 含有非有限值（NaN/inf）" % name)


def _shape_of(node, depth, name):
    """校验嵌套 list 张量并返回形状 tuple。

    规则：各维非空、形状规则（矩形）、元素为有限 int/float。
    """
    if depth == 0:
        _check_element(node, name)
        return ()
    if not isinstance(node, list):
        raise ValueError("%s 的层级不足：期望 %d 层嵌套 list" % (name, depth))
    if len(node) == 0:
        raise ValueError("%s 存在空维度" % name)
    sub = _shape_of(node[0], depth - 1, name)
    for child in node[1:]:
        if _shape_of(child, depth - 1, name) != sub:
            raise ValueError("%s 的形状不规则（各子列表长度必须一致）" % name)
    return (len(node),) + sub


def _shape_of_strict(node, depth, name):
    """校验嵌套 list 张量并返回形状 tuple（容器类型错误一律 TypeError）。

    与 _shape_of 同样要求各维非空、形状规则（矩形）、叶值为有限
    int/float（拒绝 bool）；区别在于任一层级期望 list 却得到其他容器或
    标量时抛 TypeError（而非 ValueError）：层级不足、空维、形状不规则、
    标量位置出现 list（层级过深）或非有限值仍抛 ValueError，标量类型错
    （如 str/bool）抛 TypeError。
    """
    if depth == 0:
        _check_element(node, name)
        return ()
    if not isinstance(node, list):
        raise TypeError(
            "%s 必须是嵌套 list，得到 %s" % (name, type(node).__name__)
        )
    if len(node) == 0:
        raise ValueError("%s 存在空维度" % name)
    sub = _shape_of_strict(node[0], depth - 1, name)
    for child in node[1:]:
        if _shape_of_strict(child, depth - 1, name) != sub:
            raise ValueError("%s 的形状不规则（各子列表长度必须一致）" % name)
    return (len(node),) + sub


def _require_list(value, name):
    if not isinstance(value, list):
        raise TypeError(
            "%s 必须是嵌套 list，得到 %s" % (name, type(value).__name__)
        )


def _check_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s 必须是 int，得到 %s" % (name, type(value).__name__))
    return value


def _check_kernel2d(value, name):
    """校验二维核尺寸：正 int 或恰含 (KH, KW) 的正 int tuple（拒绝 bool）。

    int 展开为 (K, K)；整体类型错抛 TypeError，tuple 长度错或成员非正
    抛 ValueError，成员类型错（含 bool）抛 TypeError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "%s 必须是 int 或 tuple，得到 %s" % (name, type(value).__name__)
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("%s 必须为正整数" % name)
        return (value, value)
    if len(value) != 2:
        raise ValueError("%s tuple 必须恰含 (KH, KW) 两个元素" % name)
    kh_, kw_ = value
    for member_name, member in (("KH", kh_), ("KW", kw_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "%s 的 %s 必须是 int，得到 %s"
                % (name, member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError("%s 的 %s 必须为正整数" % (name, member_name))
    return (kh_, kw_)


def _check_output_size2d(value, name):
    """校验自适应池化输出尺寸：正 int 或恰含 (OH, OW) 的正 int tuple（拒绝 bool）。

    int 展开为 (O, O)；整体类型错抛 TypeError，tuple 长度错或成员非正
    抛 ValueError，成员类型错（含 bool）抛 TypeError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "%s 必须是 int 或 tuple，得到 %s" % (name, type(value).__name__)
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("%s 必须为正整数" % name)
        return (value, value)
    if len(value) != 2:
        raise ValueError("%s tuple 必须恰含 (OH, OW) 两个元素" % name)
    oh_, ow_ = value
    for member_name, member in (("OH", oh_), ("OW", ow_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "%s 的 %s 必须是 int，得到 %s"
                % (name, member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError("%s 的 %s 必须为正整数" % (name, member_name))
    return (oh_, ow_)


def _check_stride2d(value):
    """校验 Conv2D 步长：正 int 或恰含 (SH, SW) 的正 int tuple（拒绝 bool）。

    int 展开为 (S, S)；整体类型错抛 TypeError，tuple 长度错或成员非正
    抛 ValueError，成员类型错（含 bool）抛 TypeError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "stride 必须是 int 或 tuple，得到 %s" % type(value).__name__
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("stride 必须为正整数")
        return (value, value)
    if len(value) != 2:
        raise ValueError("stride tuple 必须恰含 (SH, SW) 两个元素")
    sh_, sw_ = value
    for member_name, member in (("SH", sh_), ("SW", sw_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "stride 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError("stride 的 %s 必须为正整数" % member_name)
    return (sh_, sw_)


def _check_padding2d(value):
    """校验 Conv2D 补边：非负 int 或恰含 (PT, PB, PL, PR) 的非负 int tuple。

    int 展开为 (P, P, P, P)；拒绝 bool。整体类型错抛 TypeError，tuple
    长度错抛 ValueError，成员类型错（含 bool）抛 TypeError，成员为负
    抛 ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "padding 必须是 int 或 tuple，得到 %s" % type(value).__name__
        )
    if isinstance(value, int):
        if value < 0:
            raise ValueError("padding 必须为非负整数")
        return (value, value, value, value)
    if len(value) != 4:
        raise ValueError(
            "padding tuple 必须恰含 (PT, PB, PL, PR) 四个元素"
        )
    pt_, pb_, pl_, pr_ = value
    for member_name, member in (
        ("PT", pt_), ("PB", pb_), ("PL", pl_), ("PR", pr_),
    ):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "padding 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member < 0:
            raise ValueError("padding 的 %s 必须为非负整数" % member_name)
    return (pt_, pb_, pl_, pr_)


def _check_conv_padding2d(value):
    """校验 Conv2D 补边：非负 int/四元非负 int tuple，或 "same"/"valid"。

    int/tuple 的展开与校验完全沿用 _check_padding2d（拒绝 bool），返回
    四元 tuple；字符串 "same"/"valid" 原样返回，实际四边补边在每次
    forward 按输入形状解析。其他 str 抛 ValueError，其他类型（含 bool）
    抛 TypeError。
    """
    if isinstance(value, str):
        if value not in ("same", "valid"):
            raise ValueError(
                "padding 字符串必须是 'same' 或 'valid'，得到 %r" % value
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "padding 必须是 int、tuple 或 str，得到 %s"
            % type(value).__name__
        )
    return _check_padding2d(value)


def _resolve_conv_padding2d(spec, h_, w_, sh_, sw_, ekh_, ekw_):
    """按本次输入轴长把 Conv2D 的 padding 配置解析为 (PT, PB, PL, PR)。

    四元 tuple 原样返回；"valid" 四边取 0；"same" 按轴令 Q=ceil(L/S)、
    总补边 P=max((Q-1)*S+E-L, 0)，前侧 P//2、后侧 P-P//2，高宽两轴
    分别计算（E 为该轴有效核长）。
    """
    if isinstance(spec, tuple):
        return spec
    if spec == "valid":
        return (0, 0, 0, 0)
    qh_ = (h_ + sh_ - 1) // sh_
    ph_ = max((qh_ - 1) * sh_ + ekh_ - h_, 0)
    qw_ = (w_ + sw_ - 1) // sw_
    pw_ = max((qw_ - 1) * sw_ + ekw_ - w_, 0)
    return (ph_ // 2, ph_ - ph_ // 2, pw_ // 2, pw_ - pw_ // 2)


def _check_convtranspose_padding2d(value):
    """校验 ConvTranspose2D 补边：非负 int/四元非负 int tuple，或 "same"/"valid"。

    int/tuple 的展开与校验完全沿用 _check_padding2d（拒绝 bool），返回
    四元 tuple；字符串 "same"/"valid" 原样返回，实际四边补边在每次
    forward 按输入形状解析（"same" 的可行性还依赖 stride/dilation/
    output_padding，见 _resolve_convtranspose_padding2d）。其他 str
    抛 ValueError，其他类型（含 bool）抛 TypeError。
    """
    if isinstance(value, str):
        if value not in ("same", "valid"):
            raise ValueError(
                "padding 字符串必须是 'same' 或 'valid'，得到 %r" % value
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "padding 必须是 int、tuple 或 str，得到 %s"
            % type(value).__name__
        )
    return _check_padding2d(value)


def _resolve_convtranspose_padding2d(spec, h_, w_, sh_, sw_, ekh_, ekw_,
                                     oph_, opw_):
    """把 ConvTranspose2D 的 padding 配置解析为本次 forward 的 (PT, PB, PL, PR)。

    四元 tuple 原样返回；"valid" 四边取 0；"same" 按轴令
    T=E+OP-S（E=(K-1)*D+1 为该轴有效核长、S 为步幅、OP 为输出补边），
    前侧 T//2、后侧 T-T//2，高宽两轴分别计算。此补边使输出长度恰为
    输入长度乘 S：(L-1)*S-T+E+OP = L*S。T<0 抛 ValueError（该步幅/
    膨胀/输出补边组合无法做 same）。
    """
    if isinstance(spec, tuple):
        return spec
    if spec == "valid":
        return (0, 0, 0, 0)
    th_ = ekh_ + oph_ - sh_
    tw_ = ekw_ + opw_ - sw_
    if th_ < 0 or tw_ < 0:
        raise ValueError(
            "padding='same' 要求有效核长+输出补边不小于步幅："
            "EKH=%d、EKW=%d、OPH=%d、OPW=%d、SH=%d、SW=%d"
            % (ekh_, ekw_, oph_, opw_, sh_, sw_)
        )
    return (
        th_ // 2, th_ - th_ // 2,
        tw_ // 2, tw_ - tw_ // 2,
    )


def _check_dilation2d(value):
    """校验 Conv2D 膨胀：正 int 或恰含 (DH, DW) 的正 int tuple（拒绝 bool）。

    int 展开为 (D, D)；整体类型错抛 TypeError，tuple 长度错或成员非正
    抛 ValueError，成员类型错（含 bool）抛 TypeError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "dilation 必须是 int 或 tuple，得到 %s" % type(value).__name__
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("dilation 必须为正整数")
        return (value, value)
    if len(value) != 2:
        raise ValueError("dilation tuple 必须恰含 (DH, DW) 两个元素")
    dh_, dw_ = value
    for member_name, member in (("DH", dh_), ("DW", dw_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "dilation 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError("dilation 的 %s 必须为正整数" % member_name)
    return (dh_, dw_)


def _check_groups(value):
    """校验 Conv2D 分组数：正 int（拒绝 bool）。

    类型错（含 bool）抛 TypeError，非正抛 ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "groups 必须是 int，得到 %s" % type(value).__name__
        )
    if value <= 0:
        raise ValueError("groups 必须为正整数")
    return value


def _check_forward_output_size2d(value):
    """校验 ConvTranspose2D.forward 的 output_size：None 或恰含 (OH, OW) 的
    正 int tuple（拒绝 bool）。

    None 原样返回，表示沿用构造时的 output_padding；整体类型错（含 bool、
    int 等非 tuple）抛 TypeError，tuple 长度错或成员非正抛 ValueError，
    成员类型错（含 bool）抛 TypeError。
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, tuple):
        raise TypeError(
            "output_size 必须是 None 或 tuple，得到 %s"
            % type(value).__name__
        )
    if len(value) != 2:
        raise ValueError("output_size tuple 必须恰含 (OH, OW) 两个元素")
    oh_, ow_ = value
    for member_name, member in (("OH", oh_), ("OW", ow_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "output_size 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError(
                "output_size 的 %s 必须为正整数" % member_name
            )
    return (oh_, ow_)


def _check_output_padding2d(value, sh_, sw_):
    """校验转置卷积输出补边：非负 int 或恰含 (OPH, OPW) 的非负 int tuple。

    int 展开为 (OPH, OPH)；拒绝 bool。整体类型错抛 TypeError，tuple
    长度错抛 ValueError，成员类型错（含 bool）抛 TypeError，成员为负、
    OPH≥SH 或 OPW≥SW 抛 ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, tuple)):
        raise TypeError(
            "output_padding 必须是 int 或 tuple，得到 %s"
            % type(value).__name__
        )
    if isinstance(value, int):
        if value < 0:
            raise ValueError("output_padding 必须为非负整数")
        oph_, opw_ = value, value
    else:
        if len(value) != 2:
            raise ValueError(
                "output_padding tuple 必须恰含 (OPH, OPW) 两个元素"
            )
        oph_, opw_ = value
        for member_name, member in (("OPH", oph_), ("OPW", opw_)):
            if isinstance(member, bool) or not isinstance(member, int):
                raise TypeError(
                    "output_padding 的 %s 必须是 int，得到 %s"
                    % (member_name, type(member).__name__)
                )
    if oph_ < 0 or opw_ < 0:
        raise ValueError("output_padding 必须为非负整数")
    if oph_ >= sh_ or opw_ >= sw_:
        raise ValueError(
            "output_padding 必须小于对应步长：OPH=%d、SH=%d、OPW=%d、SW=%d"
            % (oph_, sh_, opw_, sw_)
        )
    return (oph_, opw_)


_PADDING_MODES = ("zeros", "replicate", "circular", "reflect")


def _check_padding_mode(value):
    """校验 Conv2D 补边模式：取值为 _PADDING_MODES 之一的 str。

    非 str（含 bool）抛 TypeError，str 但取值非法抛 ValueError。
    """
    if not isinstance(value, str):
        raise TypeError(
            "padding_mode 必须是 str，得到 %s" % type(value).__name__
        )
    if value not in _PADDING_MODES:
        raise ValueError(
            "padding_mode 必须是 'zeros'、'replicate'、'circular' 或 "
            "'reflect'，得到 %r" % value
        )
    return value


def _map_pad_index(mode, q, length):
    """按补边模式把采样坐标 q 映射为 [0, length) 内的下标。

    zeros：越界坐标返回 None（不累加）；replicate：夹取到
    min(max(q, 0), length-1)；circular：取 q % length；
    reflect：令 T = 2*length-2、r = q % T，r < length 时取 r，否则取
    T - r。界内坐标在四种模式下都映射为其自身。
    """
    if 0 <= q < length:
        return q
    if mode == "zeros":
        return None
    if mode == "replicate":
        return 0 if q < 0 else length - 1
    if mode == "circular":
        return q % length
    # reflect：调用方保证有补边的轴 length >= 2（否则 forward 已抛错）。
    t = 2 * length - 2
    r = q % t
    return r if r < length else t - r


def _zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class Conv2D:
    """二维互相关层（NCHW，嵌套 list，可配置补边模式）。

    weights: [O][C/G][KH][KW]，bias: [O]，输入 x: [N][C][H][W]。
    stride: 正 int（展开为 (S, S)）或恰含 (SH, SW) 的正 int tuple。
    padding: 非负 int（展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边；
    也接受字符串 "same"/"valid"。"valid" 四边补边恒为 0；"same"
    的实际四边补边在每次 forward 按各轴输入长度 L、步长 S、有效核
    E=(K-1)*D+1 解析：Q=ceil(L/S)、总补边 P=max((Q-1)*S+E-L,0)，
    前侧 P//2、后侧 P-P//2，高宽两轴分别计算。其他 str 抛 ValueError，
    其他类型（含 bool）抛 TypeError。
    dilation: 正 int（展开为 (D, D)）或恰含 (DH, DW) 的正 int tuple。
    groups: 正 int 分组数 G（拒绝 bool），默认 1。O 与 C 均须被 G
    整除；输出通道 o 属于组 g=o//(O/G)，仅与输入通道
    [g*C/G, (g+1)*C/G) 做互相关，weights[o][ci] 对应组内第 ci 通道，
    组间无连接。构造时 O 不能整除 G 抛 ValueError；forward 时 C 不能
    整除 G 或 C/G 不等于 weights 第二维抛 ValueError。
    padding_mode: str，默认 "zeros"，取值 "zeros"/"replicate"/
    "circular"/"reflect"，决定越界采样坐标 q（轴长 L）的映射：
    zeros 越界不累加；replicate 映射到 min(max(q,0),L-1)；circular
    映射到 q%L；reflect 令 T=2L-2、r=q%T，r<L 时取 r 否则取 T-r。
    非 str（含 bool）抛 TypeError，str 但取值非法抛 ValueError。
    reflect 模式下某轴有补边且该轴长度 L<2 时 forward 抛 ValueError。
    令有效核高宽 EKH=(KH-1)*DH+1、EKW=(KW-1)*DW+1，输出:
    [N][O][floor((H+PT+PB-EKH)/SH)+1][floor((W+PL+PR-EKW)/SW)+1]；
    有效核大于补边后输入抛 ValueError，不能整除时舍弃底部或右侧余量。
    "same" 时输出高宽恰为 ceil(H/SH)、ceil(W/SW)；"valid" 时四边 0，
    有效核大于输入任一轴即抛 ValueError。"same" 的四边补边按每次
    forward 的输入形状解析：backward 使用最近一次成功 forward 解析的
    (PT, PB, PL, PR)，未成功 forward 前调用 backward 抛 ValueError；
    forward 失败时旧的解析补边与缓存原样保留，成功时不改变实参。
    """

    def __init__(self, weights, bias, stride=1, padding=0, dilation=1,
                 groups=1, padding_mode="zeros"):
        sh_, sw_ = _check_stride2d(stride)
        padding_spec = _check_conv_padding2d(padding)
        dh_, dw_ = _check_dilation2d(dilation)
        g_ = _check_groups(groups)
        mode_ = _check_padding_mode(padding_mode)

        _require_list(weights, "weights")
        _require_list(bias, "bias")
        w_shape = _shape_of(weights, 4, "weights")
        b_shape = _shape_of(bias, 1, "bias")
        if b_shape[0] != w_shape[0]:
            raise ValueError(
                "bias 长度 %d 与 weights 输出通道数 %d 不符"
                % (b_shape[0], w_shape[0])
            )
        if w_shape[0] % g_ != 0:
            raise ValueError(
                "weights 输出通道数 %d 不能被 groups %d 整除"
                % (w_shape[0], g_)
            )

        self._weights = weights
        self._bias = bias
        self._stride = (sh_, sw_)
        # _padding_spec 为构造实参（四元 tuple 或 "same"/"valid"）；
        # _padding 为最近一次成功 forward 解析出的 (PT,PB,PL,PR)，
        # "same" 在首次成功 forward 前先置 (0,0,0,0)（此时 backward
        # 必因尚无缓存而抛错）。
        self._padding_spec = padding_spec
        if isinstance(padding_spec, tuple):
            self._padding = padding_spec
        else:
            self._padding = (0, 0, 0, 0)
        self._dilation = (dh_, dw_)
        self._groups = g_
        self._padding_mode = mode_
        self._w_shape = w_shape  # (O, C/G, KH, KW)

        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][C][H][W] 按补边模式做互相关，返回嵌套 list 并缓存输入。

        越界采样坐标按 padding_mode 映射（见类文档）；reflect 模式下
        有补边的轴长度小于 2 时抛 ValueError。padding 为 "same"/"valid"
        时按本次输入形状解析四边补边（见类文档）。校验或计算失败不改变
        实参、旧缓存与旧的解析补边；成功后缓存输入、输出形状与本次解析
        的 (PT, PB, PL, PR)，供 backward 使用。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        o_ch, w_c, kh_, kw_ = self._w_shape
        g_ = self._groups
        if c_ % g_ != 0:
            raise ValueError(
                "输入通道数 %d 不能被 groups %d 整除" % (c_, g_)
            )
        if c_ // g_ != w_c:
            raise ValueError(
                "输入通道数 %d 与 weights 通道数 %d 不符" % (c_, w_c)
            )
        sh_, sw_ = self._stride
        dh_, dw_ = self._dilation
        mode = self._padding_mode
        ekh_ = (kh_ - 1) * dh_ + 1
        ekw_ = (kw_ - 1) * dw_ + 1
        # 本次 forward 实际四边补边：tuple 原样、valid 全 0、same 按
        # 当前 H/W 解析（仅写入局部变量，成功末尾才提交到 self._padding）。
        pt_, pb_, pl_, pr_ = _resolve_conv_padding2d(
            self._padding_spec, h_, w_, sh_, sw_, ekh_, ekw_
        )
        if mode == "reflect":
            if (pt_ + pb_ > 0 and h_ < 2) or (pl_ + pr_ > 0 and w_ < 2):
                raise ValueError(
                    "reflect 补边模式下，有补边的轴长度必须至少为 2"
                )
        if ekh_ > h_ + pt_ + pb_ or ekw_ > w_ + pl_ + pr_:
            raise ValueError("核在补边后仍越界：核尺寸大于补边后的输入")
        oh_ = (h_ + pt_ + pb_ - ekh_) // sh_ + 1
        ow_ = (w_ + pl_ + pr_ - ekw_) // sw_ + 1

        weights = self._weights
        bias = self._bias
        o_per_g = o_ch // g_
        out = []
        for n in range(n_):
            out_n = []
            for o in range(o_ch):
                c_base = (o // o_per_g) * w_c
                out_o = []
                for oh in range(oh_):
                    row = []
                    base_h = oh * sh_ - pt_
                    for ow in range(ow_):
                        base_w = ow * sw_ - pl_
                        acc = bias[o]
                        for ci in range(w_c):
                            x_c = x[n][c_base + ci]
                            w_c_o = weights[o][ci]
                            for kh in range(kh_):
                                ih = _map_pad_index(mode, base_h + kh * dh_, h_)
                                if ih is None:
                                    continue
                                x_row = x_c[ih]
                                w_row = w_c_o[kh]
                                for kw in range(kw_):
                                    iw = _map_pad_index(
                                        mode, base_w + kw * dw_, w_
                                    )
                                    if iw is not None:
                                        acc += x_row[iw] * w_row[kw]
                        row.append(acc)
                    out_o.append(row)
                out_n.append(out_o)
            out.append(out_n)

        self._x = x
        self._out_shape = (n_, o_ch, oh_, ow_)
        # 仅在输出完整构建后提交本次解析的补边，失败路径保留旧值与旧缓存。
        self._padding = (pt_, pb_, pl_, pr_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回 (dx, dweights, dbias)。

        dy 的形状必须等于最近一次成功 forward 的输出形状。
        未成功 forward 前调用一律抛 ValueError。补边使用最近一次成功
        forward 解析的 (PT, PB, PL, PR)（"same" 为按该次输入解析的
        四边补边）。越界采样坐标按 padding_mode 映射：dweights 按映射
        坐标取 x 累加，dx 累加至映射坐标（多个核位置映射到同一输入坐标
        时梯度在此累加）；zeros 补边位置不产生 dx，多个输出位置对同一
        输入坐标的梯度在此累加。
        """
        if self._x is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        x = self._x
        weights = self._weights
        n_, o_ch, oh_, ow_ = self._out_shape
        g_ = self._groups
        o_per_g = o_ch // g_
        c_per_g = self._w_shape[1]
        kh_ = self._w_shape[2]
        kw_ = self._w_shape[3]
        c_ = len(x[0])
        h_ = len(x[0][0])
        w_ = len(x[0][0][0])
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        dh_, dw_ = self._dilation
        mode = self._padding_mode

        dx = _zeros((n_, c_, h_, w_))
        dw = _zeros(self._w_shape)
        db = _zeros((o_ch,))

        for n in range(n_):
            for o in range(o_ch):
                c_base = (o // o_per_g) * c_per_g
                for oh in range(oh_):
                    base_h = oh * sh_ - pt_
                    for ow in range(ow_):
                        g = dy[n][o][oh][ow]
                        db[o] += g
                        base_w = ow * sw_ - pl_
                        for ci in range(c_per_g):
                            c = c_base + ci
                            x_c = x[n][c]
                            dx_c = dx[n][c]
                            w_c_o = weights[o][ci]
                            dw_c_o = dw[o][ci]
                            for kh in range(kh_):
                                ih = _map_pad_index(mode, base_h + kh * dh_, h_)
                                if ih is None:
                                    continue
                                x_row = x_c[ih]
                                dx_row = dx_c[ih]
                                w_row = w_c_o[kh]
                                dw_row = dw_c_o[kh]
                                for kw in range(kw_):
                                    iw = _map_pad_index(
                                        mode, base_w + kw * dw_, w_
                                    )
                                    if iw is not None:
                                        dw_row[kw] += g * x_row[iw]
                                        dx_row[iw] += g * w_row[kw]
        return dx, dw, db


class ConvTranspose2D:
    """二维转置卷积层（NCHW，嵌套 list，分组、非对称步幅、四边补边、膨胀核）。

    令 G=groups。weights: [C][O/G][KH][KW]（第一维输入通道 C、第二维为
    每组输出通道数 O/G），bias: [O]，输入 x: [N][C][H][W]。输入通道 c
    仅连接同组输出：组 g=c//(C/G)，weights[c][oi] 对应输出通道
    g*(O/G)+oi，组间无连接。
    groups: 正 int（拒绝 bool），默认 1；类型错（含 bool）抛 TypeError，
    非正抛 ValueError。C、O 均须被 G 整除，否则抛 ValueError。
    stride: 正 int（展开为 (S, S)）或恰含 (SH, SW) 的正 int tuple，
    整体与成员均须为 int 且拒绝 bool；类型错抛 TypeError，长度错或
    成员非正抛 ValueError。
    padding: 非负 int（展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边；亦可取
    字符串 "valid"（四边 0）或 "same"。整体与成员均须为 int 且拒绝
    bool；其他 str 抛 ValueError，其他类型（含 bool）抛 TypeError，
    tuple 长度错或成员为负抛 ValueError。
    记各轴有效核 EH=(KH-1)*DH+1、EW=(KW-1)*DW+1，步幅 SH/SW，输出
    补边 OPH/OPW。"same" 令 TH=EH+OPH-SH、TW=EW+OPW-SW，TH<0 或
    TW<0 在构造时抛 ValueError；否则前侧（上/左）补 T//2、后侧
    （下/右）补 T-T//2，使输出长度恰为输入长度乘步幅。
    dilation: 正 int（展开为 (D, D)）或恰含 (DH, DW) 的正 int tuple，
    整体与成员均须为 int 且拒绝 bool；类型错抛 TypeError，长度错或
    成员非正抛 ValueError。
    output_padding: 非负 int（展开为 (OPH, OPH)）或恰含 (OPH, OPW)
    的非负 int tuple；整体与成员均须为 int 且拒绝 bool；类型错（含
    bool）抛 TypeError，长度错、成员为负或 OPH≥SH、OPW≥SW 抛
    ValueError。
    forward(x, output_size=None) 的 output_size 仅可为 None 或恰含
    (OH, OW) 的正 int tuple；成员须为正 int 且拒绝 bool，整体或成员
    类型错抛 TypeError，长度错或成员非正抛 ValueError。每次先按既有
    规则解析四边补边，令
    BH=(H-1)*SH-PT-PB+(KH-1)*DH+1、BW 同理（即不含输出补边的基础窗口）：
    None 沿用构造时 OPH/OPW，输出为 BH+OPH、BW+OPW；tuple 令本次
    OPH=OH-BH、OPW=OW-BW，任一不满足 0≤OPH<SH、0≤OPW<SW 抛
    ValueError，输出为 OH、OW。该覆盖只作用于本次调用，不修改构造配置；
    失败保留此前成功缓存及解析补边，不改实参。
    输出形状:
    数值补边（int/tuple）与 "valid" 为
    [N][O][(H-1)*SH-PT-PB+(KH-1)*DH+1+OPH]
          [(W-1)*SW-PL-PR+(KW-1)*DW+1+OPW]（"valid" 时四边皆 0），
    "same" 为 [N][O][H*SH][W*SW]；此处 OPH/OPW 为本次调用实际生效的
    输出补边（output_size=None 即构造值，否则按 OH-BH、OW-BW 覆盖）；
    输出高或宽非正抛 ValueError。每个
    输出从 bias[o] 起按组内 c→kh→kw 累加；仅当 (oh+PT-kh*DH) 可被
    SH 整除、(ow+PL-kw*DW) 可被 SW 整除，且其商 ih、iw 分别落在
    [0,H)、[0,W) 内时，加入
    x[n][c][ih][iw]*weights[c][oi][kh][kw]（o=(c//(C/G))*(O/G)+oi）。
    "same"/"valid" 的四边补边按每次 forward 解析（"same" 与输入形状
    无关，每次相同），backward 使用最近一次成功 forward 解析的
    (PT, PB, PL, PR)。
    backward(dy) 返回 (dx, dweights, dbias)，形状依次同 x、weights、
    bias，均为全新嵌套 list 且不修改任何实参，组间不串梯度。dx 各元素
    按组内 o→kh→kw、dweights 各元素按 n→ih→iw、dbias 各元素按
    n→oh→ow 累加，累加坐标与前向完全一致。未成功 forward 前调用
    backward、dy 与最近一次成功 forward 的输出不同形，或反向计算产生
    非有限值，均抛 ValueError。forward 失败时旧缓存与旧的解析补边原样
    保留，仅成功 forward 才更新缓存。
    """

    def __init__(self, weights, bias, stride=1, padding=0, dilation=1,
                 output_padding=0, groups=1):
        sh_, sw_ = _check_stride2d(stride)
        padding_spec = _check_convtranspose_padding2d(padding)
        dh_, dw_ = _check_dilation2d(dilation)
        g_ = _check_groups(groups)
        oph_, opw_ = _check_output_padding2d(output_padding, sh_, sw_)

        _require_list(weights, "weights")
        _require_list(bias, "bias")
        w_shape = _shape_of(weights, 4, "weights")
        b_shape = _shape_of(bias, 1, "bias")
        c_in_, o_per_g, kh_, kw_ = w_shape
        o_ch_ = b_shape[0]
        if c_in_ % g_ != 0:
            raise ValueError(
                "weights 输入通道数 C=%d 不能被 groups G=%d 整除"
                % (c_in_, g_)
            )
        if o_ch_ % g_ != 0:
            raise ValueError(
                "输出通道数 O=%d 不能被 groups G=%d 整除"
                % (o_ch_, g_)
            )
        if o_per_g != o_ch_ // g_:
            raise ValueError(
                "weights 第二维 %d 不等于 O/G=%d（O=%d、G=%d）"
                % (o_per_g, o_ch_ // g_, o_ch_, g_)
            )
        # "same" 可行性只依赖 stride/dilation/output_padding 与核长，
        # 与输入形状无关，构造时即校验 T=E+OP-S 非负。
        if padding_spec == "same":
            ekh_ = (kh_ - 1) * dh_ + 1
            ekw_ = (kw_ - 1) * dw_ + 1
            if ekh_ + oph_ - sh_ < 0 or ekw_ + opw_ - sw_ < 0:
                raise ValueError(
                    "padding='same' 要求有效核长+输出补边不小于步幅："
                    "EKH=%d、EKW=%d、OPH=%d、OPW=%d、SH=%d、SW=%d"
                    % (ekh_, ekw_, oph_, opw_, sh_, sw_)
                )

        self._weights = weights
        self._bias = bias
        self._stride = (sh_, sw_)
        # _padding_spec 为构造实参（四元 tuple 或 "same"/"valid"）；
        # _padding 为最近一次成功 forward 解析出的 (PT,PB,PL,PR)，
        # 字符串模式在首次成功 forward 前先置 (0,0,0,0)（此时 backward
        # 必因尚无缓存而抛错）。
        self._padding_spec = padding_spec
        if isinstance(padding_spec, tuple):
            self._padding = padding_spec
        else:
            self._padding = (0, 0, 0, 0)
        self._dilation = (dh_, dw_)
        self._output_padding = (oph_, opw_)
        self._groups = g_
        self._w_shape = w_shape  # (C, O/G, KH, KW)
        self._o_channels = o_ch_

        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x, output_size=None):
        """按分组转置卷积规则计算输出并缓存输入，返回全新嵌套 list。

        output_size 仅可为 None 或恰含 (OH, OW) 的正 int tuple（拒绝
        bool）：整体或成员类型错抛 TypeError，长度错或成员非正抛
        ValueError。每次先按既有规则解析本次四边补边 (PT, PB, PL, PR)，
        再令基础窗口 BH=(H-1)*SH-PT-PB+(KH-1)*DH+1、
        BW=(W-1)*SW-PL-PR+(KW-1)*DW+1（不含输出补边）：
        output_size=None 时沿用构造时 OPH/OPW，输出为 BH+OPH、BW+OPW
        （"same" 恰为 H*SH、W*SW）；output_size=(OH,OW) 时令本次
        OPH=OH-BH、OPW=OW-BW，任一不满足 0≤OPH<SH、0≤OPW<SW 抛
        ValueError，输出即为 OH、OW。覆盖只作用于本次调用，不修改构造
        配置（self._output_padding 不变）。每个输出从 bias[o] 起按组内
        c→kh→kw 累加，仅当 (oh+PT-kh*DH)、(ow+PL-kw*DW) 分别可被 SH、
        SW 整除且商 ih、iw 有效时加入乘积（是否命中完全由上述坐标条件
        决定，输出补边仅扩大输出窗口）。None 路径输出高/宽非正或计算
        出现非有限值抛 ValueError。padding 为 "same"/"valid" 时按本次
        forward 解析四边补边。校验或计算失败不改变实参、旧缓存与旧的
        解析补边；成功后才缓存输入、输出形状与本次解析的
        (PT, PB, PL, PR)。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        c_in_, o_per_g, kh_, kw_ = self._w_shape
        g_ = self._groups
        if c_ != c_in_:
            raise ValueError(
                "输入通道数 %d 与 weights 输入通道数 %d 不符"
                % (c_, c_in_)
            )
        o_ch_ = self._o_channels
        sh_, sw_ = self._stride
        dh_, dw_ = self._dilation
        oph_, opw_ = self._output_padding
        ekh_ = (kh_ - 1) * dh_ + 1
        ekw_ = (kw_ - 1) * dw_ + 1
        # 本次 forward 实际四边补边：tuple 原样、valid 全 0、same 按
        # T=E+OP-S 解析（仅写入局部变量，成功末尾才提交到 self._padding）。
        pt_, pb_, pl_, pr_ = _resolve_convtranspose_padding2d(
            self._padding_spec, h_, w_, sh_, sw_,
            ekh_, ekw_, oph_, opw_,
        )
        # 不含输出补边的基础窗口；"same" 补边用构造 OP 解析，故
        # BH+构造OPH=H*SH、BW+构造OPW=W*SW。
        bh_ = (h_ - 1) * sh_ - pt_ - pb_ + ekh_
        bw_ = (w_ - 1) * sw_ - pl_ - pr_ + ekw_
        requested = _check_forward_output_size2d(output_size)
        if requested is None:
            eff_oph_, eff_opw_ = oph_, opw_
            oh_ = bh_ + eff_oph_
            ow_ = bw_ + eff_opw_
            if oh_ <= 0 or ow_ <= 0:
                raise ValueError(
                    "转置卷积输出尺寸非正：OH=%d、OW=%d" % (oh_, ow_)
                )
        else:
            req_oh_, req_ow_ = requested
            eff_oph_ = req_oh_ - bh_
            eff_opw_ = req_ow_ - bw_
            if not (0 <= eff_oph_ < sh_ and 0 <= eff_opw_ < sw_):
                raise ValueError(
                    "output_size 推得的输出补边越界：OPH=%d（须满足"
                    " 0≤OPH<SH=%d）、OPW=%d（须满足 0≤OPW<SW=%d），"
                    "BH=%d、BW=%d、OH=%d、OW=%d"
                    % (eff_oph_, sh_, eff_opw_, sw_,
                       bh_, bw_, req_oh_, req_ow_)
                )
            oh_, ow_ = req_oh_, req_ow_

        weights = self._weights
        bias = self._bias
        c_per_g = c_in_ // g_
        out = []
        for n in range(n_):
            out_n = []
            x_n = x[n]
            for o in range(o_ch_):
                g_o = o // o_per_g
                c_base = g_o * c_per_g
                out_o = []
                for oh in range(oh_):
                    row = []
                    for ow in range(ow_):
                        acc = bias[o]
                        for ci in range(c_per_g):
                            c = c_base + ci
                            w_c_o = weights[c][o - g_o * o_per_g]
                            x_c = x_n[c]
                            for kh in range(kh_):
                                num_h = oh + pt_ - kh * dh_
                                if num_h % sh_ != 0:
                                    continue
                                ih = num_h // sh_
                                if ih < 0 or ih >= h_:
                                    continue
                                x_row = x_c[ih]
                                w_row = w_c_o[kh]
                                for kw in range(kw_):
                                    num_w = ow + pl_ - kw * dw_
                                    if num_w % sw_ != 0:
                                        continue
                                    iw = num_w // sw_
                                    if iw < 0 or iw >= w_:
                                        continue
                                    acc += x_row[iw] * w_row[kw]
                        if isinstance(acc, float) and not math.isfinite(acc):
                            raise ValueError(
                                "转置卷积前向计算产生非有限值（NaN/inf）"
                            )
                        row.append(acc)
                    out_o.append(row)
                out_n.append(out_o)
            out.append(out_n)

        self._x = x
        self._out_shape = (n_, o_ch_, oh_, ow_)
        # 仅在输出完整构建后提交本次解析的补边，失败路径保留旧值与旧缓存。
        self._padding = (pt_, pb_, pl_, pr_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回 (dx, dweights, dbias)。

        dy 的形状必须等于最近一次成功 forward 的输出形状。未成功
        forward 前调用一律抛 ValueError。补边使用最近一次成功 forward
        解析的 (PT, PB, PL, PR)（"same"/"valid" 为该次解析的四边
        补边）。dx 形状同 x、dweights 形状同 weights、dbias 形状同
        bias；梯度不跨组：dx 各元素仅按组内
        o→kh→kw、dweights 各元素按 n→ih→iw、dbias 各元素按
        n→oh→ow 累加，累加坐标 oh=ih*SH-PT+kh*DH、
        ow=iw*SW-PL+kw*DW 与前向完全一致（是否落在输出窗口内仅由坐标
        条件判定，output_padding 只扩大窗口）；结果均为全新嵌套 list，
        不修改实参。计算产生非有限值抛 ValueError。
        """
        if self._x is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        x = self._x
        weights = self._weights
        n_, o_ch_, oh_, ow_ = self._out_shape
        c_in_, o_per_g, kh_, kw_ = self._w_shape
        g_ = self._groups
        c_per_g = c_in_ // g_
        h_ = len(x[0][0])
        w_ = len(x[0][0][0])
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        dh_, dw_ = self._dilation

        dx = _zeros((n_, c_in_, h_, w_))
        dw = _zeros(self._w_shape)
        db = _zeros((o_ch_,))

        # dx：每个元素按组内 o→kh→kw 累加；oh=ih*SH-PT+kh*DH、
        # ow=iw*SW-PL+kw*DW 落在输出范围内的 (kh,kw) 才有贡献。
        for n in range(n_):
            for c in range(c_in_):
                g_c = c // c_per_g
                o_base = g_c * o_per_g
                for ih in range(h_):
                    dx_row = dx[n][c][ih]
                    for iw in range(w_):
                        cell = 0
                        for oi in range(o_per_g):
                            w_c_o = weights[c][oi]
                            dy_o = dy[n][o_base + oi]
                            for kh in range(kh_):
                                coh = ih * sh_ - pt_ + kh * dh_
                                if coh < 0 or coh >= oh_:
                                    continue
                                dy_row = dy_o[coh]
                                w_row = w_c_o[kh]
                                for kw in range(kw_):
                                    cow = iw * sw_ - pl_ + kw * dw_
                                    if 0 <= cow < ow_:
                                        cell += dy_row[cow] * w_row[kw]
                        if isinstance(cell, float) and not math.isfinite(cell):
                            raise ValueError(
                                "转置卷积反向 dx 计算产生非有限值（NaN/inf）"
                            )
                        dx_row[iw] = cell

        # dweights：每个元素按 n→ih→iw 累加；weights[c][oi] 的输出通道
        # 为 (c//(C/G))*(O/G)+oi，不跨组。
        for c in range(c_in_):
            o_base = (c // c_per_g) * o_per_g
            for oi in range(o_per_g):
                o = o_base + oi
                for kh in range(kh_):
                    for kw in range(kw_):
                        cell = 0
                        for n in range(n_):
                            for ih in range(h_):
                                coh = ih * sh_ - pt_ + kh * dh_
                                if coh < 0 or coh >= oh_:
                                    continue
                                for iw in range(w_):
                                    cow = iw * sw_ - pl_ + kw * dw_
                                    if 0 <= cow < ow_:
                                        cell += (
                                            dy[n][o][coh][cow]
                                            * x[n][c][ih][iw]
                                        )
                        if isinstance(cell, float) and not math.isfinite(cell):
                            raise ValueError(
                                "转置卷积反向 dweights 计算产生非有限值"
                                "（NaN/inf）"
                            )
                        dw[c][oi][kh][kw] = cell

        # dbias：每个元素按 n→oh→ow 累加。
        for o in range(o_ch_):
            cell = 0
            for n in range(n_):
                for coh in range(oh_):
                    dy_row = dy[n][o][coh]
                    for cow in range(ow_):
                        cell += dy_row[cow]
            if isinstance(cell, float) and not math.isfinite(cell):
                raise ValueError(
                    "转置卷积反向 dbias 计算产生非有限值（NaN/inf）"
                )
            db[o] = cell

        return dx, dw, db


class MaxPool2D:
    """二维最大池化层（NCHW，嵌套 list，逐通道池化，支持膨胀采样）。

    kernel_size: 正 int（双轴同值，展开为 (K, K)）或恰含 (KH, KW) 的
    正 int tuple。
    stride: 正 int（双轴同值）或恰含 (SH, SW) 的正 int tuple；为 None
    时取 (KH, KW)。
    padding: 非负 int（四边同值，展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边；且
    PT、PB < KH，PL、PR < KW。补边位置不参与比较。
    dilation: 正 int（双轴同值，展开为 (D, D)）或恰含 (DH, DW) 的
    正 int tuple。
    ceil_mode: 只能为 bool（默认 False），否则抛 TypeError。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]。记有效核长
    EH=(KH-1)*DH+1、EW=(KW-1)*DW+1，A=H+PT+PB-EH、
    B=W+PL+PR-EW：ceil_mode 为假时
    OH = A // SH + 1，
    OW = B // SW + 1；
    为真时
    OH = (A + SH - 1) // SH + 1，
    OW = (B + SW - 1) // SW + 1，
    但若末窗口起点满足 (OH-1)*SH >= H+PT（宽轴为
    (OW-1)*SW >= W+PL）则相应轴减 1。
    有效核长大于补边后输入抛 ValueError。ceil_mode 为假不能整除时
    舍弃底部或右侧余量。窗口采样坐标为 oh*SH-PT+kh*DH、ow*SW-PL+kw*DW（kh、kw 从
    0 起），越界补边位置不参与比较；某窗口无真实采样点（膨胀可使
    采样点全部落到补边区域）抛 ValueError。窗口内按 kh→kw 扫描，
    并列最大只取首个真实坐标。
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False):
        kh_, kw_ = _check_kernel2d(kernel_size, "kernel_size")
        if stride is None:
            sh_, sw_ = kh_, kw_
        else:
            sh_, sw_ = _check_stride2d(stride)
        pt_, pb_, pl_, pr_ = _check_padding2d(padding)
        if pt_ >= kh_ or pb_ >= kh_:
            raise ValueError("padding 的 PT、PB 必须小于 KH")
        if pl_ >= kw_ or pr_ >= kw_:
            raise ValueError("padding 的 PL、PR 必须小于 KW")
        dh_, dw_ = _check_dilation2d(dilation)
        if not isinstance(ceil_mode, bool):
            raise TypeError(
                "ceil_mode 必须是 bool，得到 %s"
                % type(ceil_mode).__name__
            )

        self._kernel_size = (kh_, kw_)
        self._stride = (sh_, sw_)
        self._padding = (pt_, pb_, pl_, pr_)
        self._dilation = (dh_, dw_)
        self._ceil_mode = ceil_mode

        self._x_shape = None   # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状
        self._winners = None   # 每个输出位置的最大值来源 (ih, iw)

    def forward(self, x):
        """对 x: [N][C][H][W] 做最大池化，返回新 list 并缓存获胜位置。

        仅在全部窗口采样完成后才更新缓存：任何失败都保留旧缓存，且不
        修改实参 x。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        kh_, kw_ = self._kernel_size
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        dh_, dw_ = self._dilation
        eh_ = (kh_ - 1) * dh_ + 1
        ew_ = (kw_ - 1) * dw_ + 1
        if eh_ > h_ + pt_ + pb_ or ew_ > w_ + pl_ + pr_:
            raise ValueError("池化窗口在补边后仍越界：有效核大于补边后的输入")
        oh_ = _avgpool_out_size(
            h_, pt_, pb_, eh_, sh_, self._ceil_mode
        )
        ow_ = _avgpool_out_size(
            w_, pl_, pr_, ew_, sw_, self._ceil_mode
        )

        out = []
        winners = []
        for n in range(n_):
            out_n = []
            win_n = []
            for c in range(c_):
                x_c = x[n][c]
                out_c = []
                win_c = []
                for oh in range(oh_):
                    base_h = oh * sh_ - pt_
                    row = []
                    win_row = []
                    for ow in range(ow_):
                        base_w = ow * sw_ - pl_
                        best = None
                        best_pos = None
                        for kh in range(kh_):
                            ih = base_h + kh * dh_
                            if ih < 0 or ih >= h_:
                                continue
                            x_row = x_c[ih]
                            for kw in range(kw_):
                                iw = base_w + kw * dw_
                                if 0 <= iw < w_:
                                    v = x_row[iw]
                                    if best is None or v > best:
                                        best = v
                                        best_pos = (ih, iw)
                        if best_pos is None:
                            raise ValueError(
                                "池化窗口 (%d, %d) 无真实采样点："
                                "膨胀/补边使窗口完全落在补边区域"
                                % (oh, ow)
                            )
                        row.append(best)
                        win_row.append(best_pos)
                    out_c.append(row)
                    win_c.append(win_row)
                out_n.append(out_c)
                win_n.append(win_c)
            out.append(out_n)
            winners.append(win_n)

        self._x_shape = (n_, c_, h_, w_)
        self._out_shape = (n_, c_, oh_, ow_)
        self._winners = winners
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与输入同形状的新 list dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状；梯度按
        n→c→oh→ow 顺序累加到 forward 记录的获胜位置。未成功 forward
        前调用、dy 形状错或累加产生非有限值一律抛 ValueError。
        """
        if self._winners is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, oh_, ow_ = self._out_shape
        dx = _zeros(self._x_shape)
        winners = self._winners
        for n in range(n_):
            for c in range(c_):
                dx_c = dx[n][c]
                dy_c = dy[n][c]
                win_c = winners[n][c]
                for oh in range(oh_):
                    dy_row = dy_c[oh]
                    win_row = win_c[oh]
                    for ow in range(ow_):
                        ih, iw = win_row[ow]
                        cell = dx_c[ih][iw] + dy_row[ow]
                        if (
                            isinstance(cell, float)
                            and not math.isfinite(cell)
                        ):
                            raise ValueError(
                                "池化反向梯度计算产生非有限值（NaN/inf）"
                            )
                        dx_c[ih][iw] = cell
        return dx


def _avgpool_out_size(length, pad_lo, pad_hi, kernel, stride, ceil_mode):
    """计算池化单轴输出长度。

    记 A=L+pad_lo+pad_hi-kernel。floor 模式取 A//stride+1；ceil 模式取
    (A+stride-1)//stride+1，但若最后窗口起点越过声明补边矩形的真实输入
    起点（(O-1)*stride >= L+pad_lo）则丢弃该末窗口、轴长减 1。
    """
    a = length + pad_lo + pad_hi - kernel
    if not ceil_mode:
        return a // stride + 1
    out = (a + stride - 1) // stride + 1
    if (out - 1) * stride >= length + pad_lo:
        out -= 1
    return out


def _avgpool_window_count(base_h, base_w, h_, w_, kh_, kw_,
                          pt_, pb_, pl_, pr_, include_pad,
                          dh_=1, dw_=1):
    """返回窗口除数：真实坐标数，或声明补边矩形内的坐标数。

    采样坐标为 (base_h+kh*DH, base_w+kw*DW)（kh∈[0,KH)、
    kw∈[0,KW)），即按膨胀间距落在有效矩形 [base, base+(K-1)*D+1)
    上的离散点。真实坐标为采样点中位于 [0,H)×[0,W) 的点数；
    include_pad 为真时改取落在声明补边矩形 [-PT,H+PB)×[-PL,W+PR)
    内的点数（含补边零值位置），ceil 产生的矩形外位置两种情形均
    不计入。
    """
    eh_ = (kh_ - 1) * dh_ + 1
    ew_ = (kw_ - 1) * dw_ + 1
    real_lo_h = max(base_h, 0)
    real_hi_h = min(base_h + eh_, h_)
    real_lo_w = max(base_w, 0)
    real_hi_w = min(base_w + ew_, w_)
    if not include_pad:
        cnt_h = _dilated_span_count(base_h, real_lo_h, real_hi_h, dh_)
        cnt_w = _dilated_span_count(base_w, real_lo_w, real_hi_w, dw_)
        return cnt_h * cnt_w
    decl_lo_h = max(base_h, -pt_)
    decl_hi_h = min(base_h + eh_, h_ + pb_)
    decl_lo_w = max(base_w, -pl_)
    decl_hi_w = min(base_w + ew_, w_ + pr_)
    cnt_h = _dilated_span_count(base_h, decl_lo_h, decl_hi_h, dh_)
    cnt_w = _dilated_span_count(base_w, decl_lo_w, decl_hi_w, dw_)
    return cnt_h * cnt_w


def _dilated_span_count(base, lo, hi, d):
    """统计采样点 base+k*D（k 为非负 int）落在半开区间 [lo, hi) 的个数。

    调用方保证 hi <= base+(K-1)*D+1，故上界自动不超过 K-1。
    """
    if hi <= lo:
        return 0
    k_lo = -((-(lo - base)) // d)  # ceil((lo-base)/D)
    if k_lo < 0:
        k_lo = 0
    k_hi = (hi - 1 - base) // d    # floor((hi-1-base)/D)
    if k_hi < k_lo:
        return 0
    return k_hi - k_lo + 1


class AvgPool2D:
    """二维平均池化层（NCHW，嵌套 list，逐通道池化，支持膨胀采样）。

    参数展开与 TypeError/ValueError 分类与 MaxPool2D 完全一致：
    kernel_size: 正 int（双轴同值，展开为 (K, K)）或恰含 (KH, KW) 的
    正 int tuple。
    stride: 正 int（双轴同值）或恰含 (SH, SW) 的正 int tuple；为 None
    时取 (KH, KW)。
    padding: 非负 int（四边同值，展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边；且
    PT、PB < KH，PL、PR < KW。
    count_include_pad、ceil_mode 只能为 bool（默认均为 False），否则
    抛 TypeError。
    dilation: 正 int（双轴同值，展开为 (D, D)）或恰含 (DH, DW) 的
    正 int tuple（默认 1）。
    记有效核长 EH=(KH-1)*DH+1、EW=(KW-1)*DW+1，
    A=H+PT+PB-EH、B=W+PL+PR-EW：ceil_mode 为假时
    OH=A//SH+1、OW=B//SW+1；为真时 OH=(A+SH-1)//SH+1、
    OW=(B+SW-1)//SW+1，但若最后窗口起点满足 (OH-1)*SH>=H+PT（宽轴为
    (OW-1)*SW>=W+PL）则相应轴减 1。有效核长大于补边后输入抛
    ValueError。ceil_mode 为假不能整除时舍弃底部或右侧余量。
    窗口采样坐标为 oh*SH-PT+kh*DH、ow*SW-PL+kw*DW（kh、kw 从
    0 起，按 kh→kw 扫描），仅真实坐标 [0,H)×[0,W) 参与求和，补边与
    ceil 产生的矩形外位置忽略。
    count_include_pad 为假时除数为真实采样点数；为真时除数还计入
    声明补边矩形 [-PT,H+PB)×[-PL,W+PR) 内的采样点（含补边零值位置），
    但不计 ceil 产生的矩形外点；除数为 0 抛 ValueError。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]。
    """

    def __init__(self, kernel_size, stride=None, padding=0,
                 count_include_pad=False, ceil_mode=False, dilation=1):
        kh_, kw_ = _check_kernel2d(kernel_size, "kernel_size")
        if stride is None:
            sh_, sw_ = kh_, kw_
        else:
            sh_, sw_ = _check_stride2d(stride)
        pt_, pb_, pl_, pr_ = _check_padding2d(padding)
        if pt_ >= kh_ or pb_ >= kh_:
            raise ValueError("padding 的 PT、PB 必须小于 KH")
        if pl_ >= kw_ or pr_ >= kw_:
            raise ValueError("padding 的 PL、PR 必须小于 KW")
        if not isinstance(count_include_pad, bool):
            raise TypeError(
                "count_include_pad 必须是 bool，得到 %s"
                % type(count_include_pad).__name__
            )
        if not isinstance(ceil_mode, bool):
            raise TypeError(
                "ceil_mode 必须是 bool，得到 %s"
                % type(ceil_mode).__name__
            )
        dh_, dw_ = _check_dilation2d(dilation)

        self._kernel_size = (kh_, kw_)
        self._stride = (sh_, sw_)
        self._padding = (pt_, pb_, pl_, pr_)
        self._count_include_pad = count_include_pad
        self._ceil_mode = ceil_mode
        self._dilation = (dh_, dw_)

        self._x_shape = None    # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][C][H][W] 做平均池化，返回新 float list 并缓存形状。

        仅在全部窗口计算完成后才更新缓存：任何失败都保留旧缓存，且不
        修改实参 x。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        kh_, kw_ = self._kernel_size
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        dh_, dw_ = self._dilation
        eh_ = (kh_ - 1) * dh_ + 1
        ew_ = (kw_ - 1) * dw_ + 1
        if eh_ > h_ + pt_ + pb_ or ew_ > w_ + pl_ + pr_:
            raise ValueError("池化窗口在补边后仍越界：有效核大于补边后的输入")
        oh_ = _avgpool_out_size(
            h_, pt_, pb_, eh_, sh_, self._ceil_mode
        )
        ow_ = _avgpool_out_size(
            w_, pl_, pr_, ew_, sw_, self._ceil_mode
        )

        out = []
        for n in range(n_):
            out_n = []
            for c in range(c_):
                x_c = x[n][c]
                out_c = []
                for oh in range(oh_):
                    base_h = oh * sh_ - pt_
                    row = []
                    for ow in range(ow_):
                        base_w = ow * sw_ - pl_
                        total = 0.0
                        for kh in range(kh_):
                            ih = base_h + kh * dh_
                            if ih < 0 or ih >= h_:
                                continue
                            x_row = x_c[ih]
                            for kw in range(kw_):
                                iw = base_w + kw * dw_
                                if 0 <= iw < w_:
                                    total += x_row[iw]
                        count = _avgpool_window_count(
                            base_h, base_w, h_, w_, kh_, kw_,
                            pt_, pb_, pl_, pr_, self._count_include_pad,
                            dh_, dw_,
                        )
                        if count == 0:
                            raise ValueError(
                                "平均池化窗口 (%d, %d) 除数为 0："
                                "膨胀采样点全部落在补边矩形之外"
                                % (oh, ow)
                            )
                        avg = total / count
                        if not math.isfinite(avg):
                            raise ValueError("平均池化计算产生非有限值（NaN/inf）")
                        row.append(avg)
                    out_c.append(row)
                out_n.append(out_c)
            out.append(out_n)

        self._x_shape = (n_, c_, h_, w_)
        self._out_shape = (n_, c_, oh_, ow_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与输入同形状的新 list dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状；每个窗口把
        dy / 该窗口除数（forward 同一除数：默认仅真实采样点数，
        count_include_pad 时含声明补边矩形内的采样位置）按
        n→c→oh→ow、kh→kw 的膨胀采样坐标
        oh*SH-PT+kh*DH、ow*SW-PL+kw*DW 仅累加至各真实输入坐标
        （补边位置不分摊）。未成功 forward 前调用、dy 形状错、除数为 0
        或累加产生非有限值一律抛 ValueError；失败时不改动缓存与实参。
        """
        if self._out_shape is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, oh_, ow_ = self._out_shape
        _, _, h_, w_ = self._x_shape
        kh_, kw_ = self._kernel_size
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        dh_, dw_ = self._dilation

        dx = _zeros(self._x_shape)
        for n in range(n_):
            dx_n = dx[n]
            dy_n = dy[n]
            for c in range(c_):
                dx_c = dx_n[c]
                dy_c = dy_n[c]
                for oh in range(oh_):
                    base_h = oh * sh_ - pt_
                    dy_row = dy_c[oh]
                    for ow in range(ow_):
                        base_w = ow * sw_ - pl_
                        count = _avgpool_window_count(
                            base_h, base_w, h_, w_, kh_, kw_,
                            pt_, pb_, pl_, pr_, self._count_include_pad,
                            dh_, dw_,
                        )
                        if count == 0:
                            raise ValueError(
                                "平均池化窗口 (%d, %d) 除数为 0："
                                "膨胀采样点全部落在补边矩形之外"
                                % (oh, ow)
                            )
                        share = dy_row[ow] / count
                        if not math.isfinite(share):
                            raise ValueError("平均池化反向计算产生非有限值（NaN/inf）")
                        for kh in range(kh_):
                            ih = base_h + kh * dh_
                            if ih < 0 or ih >= h_:
                                continue
                            dx_row = dx_c[ih]
                            for kw in range(kw_):
                                iw = base_w + kw * dw_
                                if 0 <= iw < w_:
                                    dx_row[iw] += share
                                    if not math.isfinite(dx_row[iw]):
                                        raise ValueError(
                                            "平均池化反向累加产生非有限值（NaN/inf）"
                                        )
        return dx


class AdaptiveAvgPool2D:
    """二维自适应平均池化层（NCHW，嵌套 list，逐通道池化）。

    output_size: 正 int（双轴同值，展开为 (O, O)）或恰含 (OH, OW) 的
    正 int tuple（拒绝 bool）。整体类型错抛 TypeError，tuple 长度错或
    成员非正抛 ValueError，成员类型错（含 bool）抛 TypeError。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]。输出位置 oh 的输入
    区间为 [floor(oh*H/OH), ceil((oh+1)*H/OH))，ow 同理；每个区间至少
    含一个元素，故 OH/OW 可以大于 H/W（此时部分区间长度为 1，且同一
    输入坐标可被相邻区间重复覆盖）。各输出以 0.0 起按 n→c→oh→ow、
    ih→iw 顺序累加区间内输入值，再除以区间元素数。
    """

    def __init__(self, output_size):
        oh_, ow_ = _check_output_size2d(output_size, "output_size")
        self._output_size = (oh_, ow_)

        self._x_shape = None    # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状

    def _bins(self, length, out_len):
        """输出轴每个位置覆盖的输入 [起, 止) 区间。"""
        return [
            (
                (i * length) // out_len,
                -((-(i + 1) * length) // out_len),
            )
            for i in range(out_len)
        ]

    def forward(self, x):
        """对 x: [N][C][H][W] 做自适应平均池化，返回新 float list 并缓存形状。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        oh_, ow_ = self._output_size
        h_bins = self._bins(h_, oh_)
        w_bins = self._bins(w_, ow_)

        out = []
        for n in range(n_):
            out_n = []
            for c in range(c_):
                x_c = x[n][c]
                out_c = []
                for ih0, ih1 in h_bins:
                    row = []
                    for iw0, iw1 in w_bins:
                        total = 0.0
                        for ih in range(ih0, ih1):
                            x_row = x_c[ih]
                            for iw in range(iw0, iw1):
                                total += x_row[iw]
                        count = (ih1 - ih0) * (iw1 - iw0)
                        avg = total / count
                        if not math.isfinite(avg):
                            raise ValueError(
                                "自适应平均池化计算产生非有限值（NaN/inf）"
                            )
                        row.append(avg)
                    out_c.append(row)
                out_n.append(out_c)
            out.append(out_n)

        self._x_shape = (n_, c_, h_, w_)
        self._out_shape = (n_, c_, oh_, ow_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与输入同形状的 dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状；按相同的
        n→c→oh→ow、ih→iw 分箱顺序，把 dy[n][c][oh][ow] 除以该区间
        元素数后累加至区间覆盖的每个输入坐标（区间可重叠，同一输入
        坐标可收到多份梯度）。未成功 forward 前调用一律抛 ValueError。
        """
        if self._out_shape is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, oh_, ow_ = self._out_shape
        _, _, h_, w_ = self._x_shape
        h_bins = self._bins(h_, oh_)
        w_bins = self._bins(w_, ow_)

        dx = _zeros(self._x_shape)
        for n in range(n_):
            dx_n = dx[n]
            dy_n = dy[n]
            for c in range(c_):
                dx_c = dx_n[c]
                dy_c = dy_n[c]
                for oh, (ih0, ih1) in enumerate(h_bins):
                    dy_row = dy_c[oh]
                    for ow, (iw0, iw1) in enumerate(w_bins):
                        count = (ih1 - ih0) * (iw1 - iw0)
                        share = dy_row[ow] / count
                        if not math.isfinite(share):
                            raise ValueError(
                                "自适应平均池化反向计算产生非有限值（NaN/inf）"
                            )
                        for ih in range(ih0, ih1):
                            dx_row = dx_c[ih]
                            for iw in range(iw0, iw1):
                                dx_row[iw] += share
                                if not math.isfinite(dx_row[iw]):
                                    raise ValueError(
                                        "自适应平均池化反向累加产生非有限值（NaN/inf）"
                                    )
        return dx


class AdaptiveMaxPool2D:
    """二维自适应最大池化层（NCHW，嵌套 list，逐通道池化）。

    output_size: 正 int（双轴同值，展开为 (O, O)）或恰含 (OH, OW) 的
    正 int tuple（拒绝 bool）。整体类型错抛 TypeError，tuple 长度错或
    成员非正抛 ValueError，成员类型错（含 bool）抛 TypeError。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]。输出位置 oh 的输入
    区间为 [floor(oh*H/OH), ceil((oh+1)*H/OH))，ow 同理；每个区间至少
    含一个元素，故 OH/OW 可以大于 H/W（此时部分区间长度为 1，且同一
    输入坐标可被相邻区间重复覆盖）。区间内按 ih→iw 扫描取最大值，
    并列最大只取首个坐标；forward 成功后缓存输入/输出形状与每个输出
    位置的获胜坐标。
    """

    def __init__(self, output_size):
        oh_, ow_ = _check_output_size2d(output_size, "output_size")
        self._output_size = (oh_, ow_)

        self._x_shape = None    # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状
        self._winners = None    # 每个输出位置的最大值来源 (ih, iw)

    def _bins(self, length, out_len):
        """输出轴每个位置覆盖的输入 [起, 止) 区间。"""
        return [
            (
                (i * length) // out_len,
                -((-(i + 1) * length) // out_len),
            )
            for i in range(out_len)
        ]

    def forward(self, x):
        """对 x: [N][C][H][W] 做自适应最大池化，返回新 list 并缓存获胜位置。

        仅在全部分箱扫描完成后才更新缓存：任何失败都保留旧缓存，且不
        修改实参 x。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        oh_, ow_ = self._output_size
        h_bins = self._bins(h_, oh_)
        w_bins = self._bins(w_, ow_)

        out = []
        winners = []
        for n in range(n_):
            out_n = []
            win_n = []
            for c in range(c_):
                x_c = x[n][c]
                out_c = []
                win_c = []
                for ih0, ih1 in h_bins:
                    row = []
                    win_row = []
                    for iw0, iw1 in w_bins:
                        best = None
                        best_pos = None
                        for ih in range(ih0, ih1):
                            x_row = x_c[ih]
                            for iw in range(iw0, iw1):
                                v = x_row[iw]
                                if best is None or v > best:
                                    best = v
                                    best_pos = (ih, iw)
                        row.append(best)
                        win_row.append(best_pos)
                    out_c.append(row)
                    win_c.append(win_row)
                out_n.append(out_c)
                win_n.append(win_c)
            out.append(out_n)
            winners.append(win_n)

        self._x_shape = (n_, c_, h_, w_)
        self._out_shape = (n_, c_, oh_, ow_)
        self._winners = winners
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与输入同形状的新 list dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状；梯度按
        n→c→oh→ow 顺序累加到 forward 记录的获胜位置（分箱可重叠，
        同一输入坐标可收到多份梯度）。未成功 forward 前调用、dy 形状
        错或累加产生非有限值一律抛 ValueError。
        """
        if self._winners is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, oh_, ow_ = self._out_shape
        dx = _zeros(self._x_shape)
        winners = self._winners
        for n in range(n_):
            for c in range(c_):
                dx_c = dx[n][c]
                dy_c = dy[n][c]
                win_c = winners[n][c]
                for oh in range(oh_):
                    dy_row = dy_c[oh]
                    win_row = win_c[oh]
                    for ow in range(ow_):
                        ih, iw = win_row[ow]
                        cell = dx_c[ih][iw] + dy_row[ow]
                        if (
                            isinstance(cell, float)
                            and not math.isfinite(cell)
                        ):
                            raise ValueError(
                                "自适应最大池化反向梯度计算产生非有限值（NaN/inf）"
                            )
                        dx_c[ih][iw] = cell
        return dx


class Flatten:
    """展平层：将 [N][C][H][W] 按 c→h→w 顺序展平为 [N][C*H*W]。

    backward 将 [N][C*H*W] 的梯度还原为最近一次输入的 [N][C][H][W] 形状。
    """

    def __init__(self):
        self._x_shape = None   # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """把 x: [N][C][H][W] 展平为 [N][C*H*W]，返回新 list 并缓存形状。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")

        out = []
        for n in range(n_):
            row = []
            x_n = x[n]
            for c in range(c_):
                x_c = x_n[c]
                for h in range(h_):
                    x_row = x_c[h]
                    for w in range(w_):
                        row.append(x_row[w])
            out.append(row)

        self._x_shape = (n_, c_, h_, w_)
        self._out_shape = (n_, c_ * h_ * w_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与最近输入同形状 [N][C][H][W] 的 dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状 [N][C*H*W]。
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._x_shape is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 2, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, h_, w_ = self._x_shape
        dx = _zeros(self._x_shape)
        for n in range(n_):
            dy_row = dy[n]
            dx_n = dx[n]
            idx = 0
            for c in range(c_):
                dx_c = dx_n[c]
                for h in range(h_):
                    dx_row = dx_c[h]
                    for w in range(w_):
                        dx_row[w] = dy_row[idx]
                        idx += 1
        return dx


class Linear:
    """全连接层：y[n][o] = bias[o] + sum_i x[n][i] * weights[o][i]。

    weights: [O][I]，bias: [O]，输入 x: [N][I]，输出: [N][O]。
    """

    def __init__(self, weights, bias):
        _require_list(weights, "weights")
        _require_list(bias, "bias")
        w_shape = _shape_of(weights, 2, "weights")
        b_shape = _shape_of(bias, 1, "bias")
        if b_shape[0] != w_shape[0]:
            raise ValueError(
                "bias 长度 %d 与 weights 输出维度 %d 不符"
                % (b_shape[0], w_shape[0])
            )

        self._weights = weights
        self._bias = bias
        self._w_shape = w_shape  # (O, I)

        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][I] 做线性变换，返回 [N][O] 新 list 并缓存输入。"""
        _require_list(x, "x")
        n_, i_ = _shape_of(x, 2, "x")
        o_, w_i = self._w_shape
        if i_ != w_i:
            raise ValueError(
                "输入维度 %d 与 weights 输入维度 %d 不符" % (i_, w_i)
            )

        weights = self._weights
        bias = self._bias
        out = []
        for n in range(n_):
            x_row = x[n]
            row = []
            for o in range(o_):
                acc = bias[o]
                w_row = weights[o]
                for i in range(i_):
                    acc += x_row[i] * w_row[i]
                row.append(acc)
            out.append(row)

        self._x = x
        self._out_shape = (n_, o_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回 (dx, dweights, dbias)。

        形状依次同 x、weights、bias；dx 对 o 求和，dweights/dbias 对 n 求和。
        dy 的形状必须等于最近一次成功 forward 的输出形状。
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._x is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 2, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        x = self._x
        weights = self._weights
        n_, o_ = self._out_shape
        i_ = self._w_shape[1]

        dx = _zeros((n_, i_))
        dw = _zeros(self._w_shape)
        db = _zeros((o_,))

        for n in range(n_):
            dy_row = dy[n]
            x_row = x[n]
            dx_row = dx[n]
            for o in range(o_):
                g = dy_row[o]
                db[o] += g
                w_row = weights[o]
                dw_row = dw[o]
                for i in range(i_):
                    dw_row[i] += g * x_row[i]
                    dx_row[i] += g * w_row[i]
        return dx, dw, db


class ReLU:
    """逐元素 ReLU 层（限二维 [N][D]）：v > 0 时输出 v，否则输出 0。

    反向仅在最近输入 > 0 的位置传递 dy，零点梯度为 0。
    """

    def __init__(self):
        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][D] 逐元素取 max(0, v)，返回新 list 并缓存输入。"""
        _require_list(x, "x")
        n_, d_ = _shape_of(x, 2, "x")

        out = []
        for n in range(n_):
            x_row = x[n]
            row = []
            for d in range(d_):
                v = x_row[d]
                row.append(v if v > 0 else 0)
            out.append(row)

        self._x = x
        self._out_shape = (n_, d_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与输入同形状的 dx。

        仅在最近一次成功 forward 的输入 > 0 的位置传递 dy，其余为 0。
        dy 的形状必须等于最近一次成功 forward 的输出形状。
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._x is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 2, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        x = self._x
        n_, d_ = self._out_shape
        dx = _zeros(self._out_shape)
        for n in range(n_):
            x_row = x[n]
            dy_row = dy[n]
            dx_row = dx[n]
            for d in range(d_):
                if x_row[d] > 0:
                    dx_row[d] = dy_row[d]
        return dx


class Dropout:
    """Dropout 层（NCHW，嵌套 list）：训练态按概率 p 置零并放大保留项。

    p: 丢弃概率，[0, 1) 的有限 int/float（拒绝 bool）。
    seed: 随机种子，[0, 2^32-1] 的 int（拒绝 bool）；随机状态 s 初始为 seed。
    默认训练态；train(mode) 切换模式，mode 仅接收 bool。

    训练前向按 n→c→h→w 顺序逐元素推进线性同余发生器：
    s = (1664525*s + 1013904223) % 2^32，u = s / 2^32；
    u < p 时输出 0，否则输出 x/(1-p)，并缓存同形的 0 或 1/(1-p) 掩码。
    推理前向返回 x 的新副本，不推进 s，缓存全 1 掩码。
    backward 返回 dy 与最近缓存掩码逐元素相乘的新 list。
    不读写全局 random 状态；相同 seed、输入与调用序列结果完全相同。
    """

    def __init__(self, p=0.5, seed=0):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(
                "p 必须是 int/float（拒绝 bool），得到 %s" % type(p).__name__
            )
        if not math.isfinite(p):
            raise ValueError("p 必须是有限值（拒绝 NaN/inf）")
        if p < 0 or p >= 1:
            raise ValueError("p 必须满足 0 <= p < 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
            )
        if seed < 0 or seed > 0xFFFFFFFF:
            raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")

        self._p = p
        self._seed = seed
        self._s = seed          # 当前随机状态
        self._training = True   # 默认训练态

        self._mask = None       # 最近一次成功 forward 的掩码
        self._out_shape = None  # 最近一次成功 forward 的输出形状

    def train(self, mode=True):
        """切换训练/推理模式并返回 None；mode 仅接收 bool，否则抛 TypeError。"""
        if not isinstance(mode, bool):
            raise TypeError(
                "mode 必须是 bool，得到 %s" % type(mode).__name__
            )
        self._training = mode
        return None

    def forward(self, x):
        """对 x: [N][C][H][W] 施加 dropout，返回新 list 并缓存掩码。

        训练态按 n→c→h→w 顺序逐元素推进内部随机状态；推理态返回 x 的
        新副本且不推进随机状态。校验失败不改变实参、随机状态与旧缓存。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")

        if self._training:
            p = self._p
            scale = 1 / (1 - p)
            s = self._s
            out = []
            mask = []
            for n in range(n_):
                out_n = []
                mask_n = []
                for c in range(c_):
                    x_c = x[n][c]
                    out_c = []
                    mask_c = []
                    for hh in range(h_):
                        x_row = x_c[hh]
                        out_row = []
                        mask_row = []
                        for w in range(w_):
                            s = (1664525 * s + 1013904223) % 4294967296
                            u = s / 4294967296
                            if u < p:
                                out_row.append(0)
                                mask_row.append(0)
                            else:
                                out_row.append(x_row[w] / (1 - p))
                                mask_row.append(scale)
                        out_c.append(out_row)
                        mask_c.append(mask_row)
                    out_n.append(out_c)
                    mask_n.append(mask_c)
                out.append(out_n)
                mask.append(mask_n)
            self._s = s
        else:
            out = _deep_copy(x)
            mask = []
            for n in range(n_):
                mask_n = []
                for c in range(c_):
                    mask_c = []
                    for hh in range(h_):
                        mask_c.append([1] * w_)
                    mask_n.append(mask_c)
                mask.append(mask_n)

        self._mask = mask
        self._out_shape = (n_, c_, h_, w_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与掩码逐元素相乘的新 list。

        dy 的形状必须等于最近一次成功 forward 的输出形状；
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._mask is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, h_, w_ = self._out_shape
        mask = self._mask
        dx = []
        for n in range(n_):
            dx_n = []
            for c in range(c_):
                dx_c = []
                for hh in range(h_):
                    dy_row = dy[n][c][hh]
                    mask_row = mask[n][c][hh]
                    dx_row = []
                    for w in range(w_):
                        dx_row.append(dy_row[w] * mask_row[w])
                    dx_c.append(dx_row)
                dx_n.append(dx_c)
            dx.append(dx_n)
        return dx


class Dropout2D:
    """二维通道 Dropout 层（NCHW，嵌套 list）：训练态整通道置零并放大保留通道。

    p: 丢弃概率，[0, 1) 的有限 int/float（拒绝 bool）。
    seed: 随机种子，[0, 2^32-1] 的 int（拒绝 bool）；随机状态 s 初始为 seed。
    默认训练态；train(mode) 切换模式，mode 仅接收 bool 并返回 None。

    训练前向按 n→c 顺序【每个通道一次】推进线性同余发生器：
    s = (1664525*s + 1013904223) % 2^32，u = s / 2^32；
    u < p 时该 (n, c) 整通道全部置 0，否则整通道每个元素除以 (1-p)；
    同时缓存形状 [N][C] 的逐通道掩码，取 0 或 1/(1-p)。
    推理前向返回 x 的新副本，不推进 s，缓存 [N][C] 的全 1 掩码。
    backward 要求 dy 与最近一次成功 forward 的输出同形（并沿用 x 的校验
    规则），返回 dy 与按通道广播的掩码逐元素相乘的新 list。
    未成功执行 forward 前调用 backward 一律抛 ValueError。
    不读写全局 random 状态；相同 seed、输入与调用序列结果完全相同。
    校验或计算失败不改变实参、随机状态 s 与旧缓存。
    """

    def __init__(self, p=0.5, seed=0):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(
                "p 必须是 int/float（拒绝 bool），得到 %s" % type(p).__name__
            )
        # 先按类型分支：超大整数 math.isfinite 会抛 OverflowError，而范围
        # 比较为精确整数运算，故 int 先查范围（[0,1) 内仅 0，必有限），
        # float 先查有限性再查范围，统一不外泄 OverflowError。
        if isinstance(p, int):
            if p < 0 or p >= 1:
                raise ValueError("p 必须满足 0 <= p < 1")
        else:
            if not math.isfinite(p):
                raise ValueError("p 必须是有限值（拒绝 NaN/inf）")
            if p < 0 or p >= 1:
                raise ValueError("p 必须满足 0 <= p < 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
            )
        if seed < 0 or seed > 0xFFFFFFFF:
            raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")

        self._p = p
        self._seed = seed
        self._s = seed          # 当前随机状态
        self._training = True   # 默认训练态

        self._mask = None       # 最近一次成功 forward 的 [N][C] 通道掩码
        self._out_shape = None  # 最近一次成功 forward 的输出形状

    def train(self, mode=True):
        """切换训练/推理模式并返回 None；mode 仅接收 bool，否则抛 TypeError。"""
        if not isinstance(mode, bool):
            raise TypeError(
                "mode 必须是 bool，得到 %s" % type(mode).__name__
            )
        self._training = mode
        return None

    def forward(self, x):
        """对 x: [N][C][H][W] 逐通道施加 dropout，返回新 list 并缓存掩码。

        训练态按 n→c 逐通道推进内部随机状态；推理态返回 x 的新副本且不
        推进随机状态。校验或计算失败不改变实参、随机状态与旧缓存。
        """
        n_, c_, h_, w_ = _shape_of_strict(x, 4, "x")

        p = self._p
        if self._training:
            scale = 1 / (1 - p)
            s = self._s
            out = []
            mask = []
            try:
                for n in range(n_):
                    out_n = []
                    mask_n = []
                    for c in range(c_):
                        s = (1664525 * s + 1013904223) % 4294967296
                        u = s / 4294967296
                        if u < p:
                            m = 0
                        else:
                            m = scale
                        mask_n.append(m)
                        x_c = x[n][c]
                        out_c = []
                        for hh in range(h_):
                            x_row = x_c[hh]
                            if m == 0:
                                out_c.append([0] * w_)
                            else:
                                out_c.append(
                                    [x_row[w] / (1 - p) for w in range(w_)]
                                )
                        out_n.append(out_c)
                    out.append(out_n)
                    mask.append(mask_n)
            except OverflowError:
                # 超大整数除以浮点 (1-p) 会抛 OverflowError，统一按非有限
                # 计算以 ValueError 拒绝，不向上泄漏 OverflowError。
                raise ValueError("前向计算产生非有限值（超大整数）")
            for n in range(n_):
                for c in range(c_):
                    if not math.isfinite(mask[n][c]):
                        raise ValueError("前向计算产生非有限值（NaN/inf）")
                    for hh in range(h_):
                        for w in range(w_):
                            if not math.isfinite(out[n][c][hh][w]):
                                raise ValueError(
                                    "前向计算产生非有限值（NaN/inf）"
                                )
            # 全部校验与计算成功后才提交随机状态与缓存。
            self._s = s
        else:
            out = _deep_copy(x)
            mask = [[1] * c_ for _ in range(n_)]

        self._mask = mask
        self._out_shape = (n_, c_, h_, w_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回与按通道广播的掩码逐元素相乘的新 list。

        dy 必须为与最近一次成功 forward 的输出同形的非空规则嵌套
        list[N][C][H][W]（容器/标量类型、层级、空维、不规则与非有限校验
        与 forward 的 x 完全一致）；未成功 forward 前调用或形状不符一律
        抛 ValueError。
        """
        if self._mask is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        n_, c_, h_, w_ = _shape_of_strict(dy, 4, "dy")
        if (n_, c_, h_, w_) != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % ((n_, c_, h_, w_), self._out_shape)
            )

        mask = self._mask
        dx = []
        try:
            for n in range(n_):
                dx_n = []
                for c in range(c_):
                    m = mask[n][c]
                    dy_c = dy[n][c]
                    dx_c = []
                    for hh in range(h_):
                        dy_row = dy_c[hh]
                        dx_row = []
                        for w in range(w_):
                            val = dy_row[w] * m
                            if not math.isfinite(val):
                                raise ValueError(
                                    "反向计算产生非有限值（NaN/inf）"
                                )
                            dx_row.append(val)
                        dx_c.append(dx_row)
                    dx_n.append(dx_c)
                dx.append(dx_n)
        except OverflowError:
            # 超大整数乘浮点掩码或经 math.isfinite 转换会抛 OverflowError，
            # 统一按非有限计算以 ValueError 拒绝，不向上泄漏 OverflowError。
            raise ValueError("反向计算产生非有限值（超大整数）")
        return dx


class BatchNorm2D:
    """二维批归一化层（NCHW，嵌套 list，训练/推理两态）。

    gamma、beta: 等长非空一维 list，元素为有限 int/float（拒绝 bool）；
    长度 C 即通道数。eps: 正的有限 int/float（拒绝 bool）。
    momentum: [0, 1] 的有限 int/float（拒绝 bool），默认 0.1。

    默认训练态；train(mode) 仅接收 bool 用于切换模式并返回 None。
    running_mean、running_var 为长度 C 的公开 list，初值分别全 0.0、1.0，
    未训练即可推理。

    训练态每次 forward 按当前批次统计：
    M = N*H*W，μ = Σx/M，v = Σ(x-μ)^2/M，z = (x-μ)/sqrt(v+eps)，
    y = gamma*z + beta；成功后令 m = momentum，逐通道更新
    running_mean = (1-m)*running_mean + m*μ，
    running_var  = (1-m)*running_var  + m*v。
    推理态不更新统计量与缓存，逐项输出
    y = gamma*(x-running_mean)/sqrt(running_var+eps) + beta。
    backward 返回 (dx, dgamma, dbeta)，使用最近一次成功训练 forward 的缓存：
    dbeta = Σdy、dgamma = Σ(dy*z)，
    dx = gamma/sqrt(v+eps) * (dy - (dbeta + z*dgamma)/M)，
    求和均按 n→h→w 逐通道进行。模式切换不清缓存。
    """

    def __init__(self, gamma, beta, eps=1e-5, momentum=0.1):
        _require_list(gamma, "gamma")
        _require_list(beta, "beta")
        g_shape = _shape_of(gamma, 1, "gamma")
        b_shape = _shape_of(beta, 1, "beta")
        if b_shape[0] != g_shape[0]:
            raise ValueError(
                "beta 长度 %d 与 gamma 长度 %d 不符"
                % (b_shape[0], g_shape[0])
            )
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(
                "eps 必须是 int/float（拒绝 bool），得到 %s"
                % type(eps).__name__
            )
        if not math.isfinite(eps):
            raise ValueError("eps 必须是有限值（拒绝 NaN/inf）")
        if eps <= 0:
            raise ValueError("eps 必须为正数")
        if isinstance(momentum, bool) or not isinstance(
            momentum, (int, float)
        ):
            raise TypeError(
                "momentum 必须是 int/float（拒绝 bool），得到 %s"
                % type(momentum).__name__
            )
        if not math.isfinite(momentum):
            raise ValueError("momentum 必须是有限值（拒绝 NaN/inf）")
        if momentum < 0 or momentum > 1:
            raise ValueError("momentum 必须满足 0 <= momentum <= 1")

        c_ = g_shape[0]
        self._gamma = gamma
        self._beta = beta
        self._eps = eps
        self._momentum = momentum
        self._training = True   # 默认训练态

        self.running_mean = [0.0] * c_
        self.running_var = [1.0] * c_

        self._z = None           # 最近一次成功【训练】forward 缓存的归一化值 z
        self._var = None         # 最近一次成功【训练】forward 缓存的每通道方差 v
        self._out_shape = None   # 最近一次成功【训练】forward 的输出形状

    def train(self, mode=True):
        """切换训练/推理模式并返回 None；mode 仅接收 bool，否则抛 TypeError。

        仅切换模式，不清空 running 统计量与训练 forward 缓存。
        """
        if not isinstance(mode, bool):
            raise TypeError(
                "mode 必须是 bool，得到 %s" % type(mode).__name__
            )
        self._training = mode
        return None

    def forward(self, x):
        """对 x: [N][C][H][W] 逐通道归一化，返回同形新 list。

        训练态按批次统计、缓存 z/v/形状并更新 running_mean/running_var；
        推理态使用运行统计仿射，不更新统计量与训练缓存。
        校验或计算失败不改变实参、统计量与旧缓存。
        """
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        if c_ != len(self._gamma):
            raise ValueError(
                "输入通道数 %d 与 gamma/beta 长度 %d 不符"
                % (c_, len(self._gamma))
            )

        gamma = self._gamma
        beta = self._beta
        eps = self._eps

        if not self._training:
            out = []
            for n in range(n_):
                out_n = []
                for c in range(c_):
                    inv = 1.0 / math.sqrt(self.running_var[c] + eps)
                    gc = gamma[c]
                    bc = beta[c]
                    rm = self.running_mean[c]
                    x_c = x[n][c]
                    out_c = []
                    for hh in range(h_):
                        x_row = x_c[hh]
                        out_row = []
                        for ww in range(w_):
                            zval = (x_row[ww] - rm) * inv
                            out_row.append(gc * zval + bc)
                        out_c.append(out_row)
                    out_n.append(out_c)
                out.append(out_n)

            for n in range(n_):
                for c in range(c_):
                    for hh in range(h_):
                        for ww in range(w_):
                            if not math.isfinite(out[n][c][hh][ww]):
                                raise ValueError(
                                    "前向计算产生非有限值（NaN/inf）"
                                )
            return out

        m_ = n_ * h_ * w_
        means = []
        variances = []
        for c in range(c_):
            acc = 0.0
            for n in range(n_):
                x_c = x[n][c]
                for hh in range(h_):
                    x_row = x_c[hh]
                    for ww in range(w_):
                        acc += x_row[ww]
            mu = acc / m_
            sq = 0.0
            for n in range(n_):
                x_c = x[n][c]
                for hh in range(h_):
                    x_row = x_c[hh]
                    for ww in range(w_):
                        d = x_row[ww] - mu
                        sq += d * d
            v = sq / m_
            means.append(mu)
            variances.append(v)

        out = []
        z_cache = []
        for n in range(n_):
            out_n = []
            z_n = []
            for c in range(c_):
                inv = 1.0 / math.sqrt(variances[c] + eps)
                gc = gamma[c]
                bc = beta[c]
                mu = means[c]
                x_c = x[n][c]
                out_c = []
                z_c = []
                for hh in range(h_):
                    x_row = x_c[hh]
                    out_row = []
                    z_row = []
                    for ww in range(w_):
                        zval = (x_row[ww] - mu) * inv
                        yval = gc * zval + bc
                        z_row.append(zval)
                        out_row.append(yval)
                    z_c.append(z_row)
                    out_c.append(out_row)
                z_n.append(z_c)
                out_n.append(out_c)
            out.append(out_n)
            z_cache.append(z_n)

        for c in range(c_):
            if not math.isfinite(means[c]) or not math.isfinite(variances[c]):
                raise ValueError("前向计算产生非有限值（NaN/inf）")
        for n in range(n_):
            for c in range(c_):
                for hh in range(h_):
                    for ww in range(w_):
                        yval = out[n][c][hh][ww]
                        zval = z_cache[n][c][hh][ww]
                        if not (math.isfinite(yval) and math.isfinite(zval)):
                            raise ValueError("前向计算产生非有限值（NaN/inf）")

        # 先在临时 list 中更新运行统计，全部有限后再提交，保证失败不改旧值。
        mom = self._momentum
        new_mean = [0.0] * c_
        new_var = [0.0] * c_
        for c in range(c_):
            nm = (1 - mom) * self.running_mean[c] + mom * means[c]
            nv = (1 - mom) * self.running_var[c] + mom * variances[c]
            if not (math.isfinite(nm) and math.isfinite(nv)):
                raise ValueError("前向计算产生非有限值（NaN/inf）")
            new_mean[c] = nm
            new_var[c] = nv

        self.running_mean = new_mean
        self.running_var = new_var
        self._z = z_cache
        self._var = variances
        self._out_shape = (n_, c_, h_, w_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回 (dx, dgamma, dbeta)。

        形状依次同 x、gamma、beta；使用最近一次成功【训练】forward 的缓存，
        dy 的形状必须等于该次 forward 的输出形状。模式切换不清缓存；
        尚无成功训练 forward 缓存时调用一律抛 ValueError。
        """
        if self._z is None:
            raise ValueError("尚未成功执行 forward，无法 backward")
        _require_list(dy, "dy")
        dy_shape = _shape_of(dy, 4, "dy")
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        n_, c_, h_, w_ = self._out_shape
        m_ = n_ * h_ * w_
        gamma = self._gamma
        eps = self._eps
        z_cache = self._z
        variances = self._var

        dgamma = [0.0] * c_
        dbeta = [0.0] * c_
        for c in range(c_):
            sg = 0.0
            sb = 0.0
            for n in range(n_):
                dy_c = dy[n][c]
                z_c = z_cache[n][c]
                for hh in range(h_):
                    dy_row = dy_c[hh]
                    z_row = z_c[hh]
                    for ww in range(w_):
                        g = dy_row[ww]
                        sb += g
                        sg += g * z_row[ww]
            dgamma[c] = sg
            dbeta[c] = sb

        dx = []
        inv_m = 1.0 / m_
        for n in range(n_):
            dx_n = []
            for c in range(c_):
                scale = gamma[c] / math.sqrt(variances[c] + eps)
                dy_c = dy[n][c]
                z_c = z_cache[n][c]
                dx_c = []
                for hh in range(h_):
                    dy_row = dy_c[hh]
                    z_row = z_c[hh]
                    dx_row = []
                    for ww in range(w_):
                        zval = z_row[ww]
                        corr = (dbeta[c] + zval * dgamma[c]) * inv_m
                        dx_row.append(scale * (dy_row[ww] - corr))
                    dx_c.append(dx_row)
                dx_n.append(dx_c)
            dx.append(dx_n)

        for c in range(c_):
            if not (math.isfinite(dgamma[c]) and math.isfinite(dbeta[c])):
                raise ValueError("反向计算产生非有限值（NaN/inf）")
        for n in range(n_):
            for c in range(c_):
                for hh in range(h_):
                    for ww in range(w_):
                        if not math.isfinite(dx[n][c][hh][ww]):
                            raise ValueError("反向计算产生非有限值（NaN/inf）")
        return dx, dgamma, dbeta


class SoftmaxCrossEntropy:
    """Softmax + 交叉熵损失层（限二维 logits [N][K]，labels [N]）。

    SoftmaxCrossEntropy(label_smoothing=0.0, class_weights=None,
    ignore_index=None)：
    - label_smoothing 须为 [0, 1) 内的有限 int/float（拒绝 bool）；类型错
      抛 TypeError，非有限、越界或超大整数（转 float 溢出，不泄漏
      OverflowError）抛 ValueError；默认 0.0 等价于普通 one-hot 交叉熵。
    - class_weights 为 None 或非空 list：元素为有限非负 int/float（拒绝
      bool，超大整数按非有限拒绝），且至少一项严格大于 0；类型错抛
      TypeError，空表、负值或非有限权重抛 ValueError。
    - ignore_index 为 None 或 int（拒绝 bool），否则抛 TypeError。

    forward(logits, labels)：logits 为非空规则嵌套 list[N][K]（N、K ≥ 1），
    元素为有限 int/float（拒绝 bool）；labels 为长度 N 的 list，元素为
    int（拒绝 bool）。逐行取 m = max(row)，按 k 递增求
    e[k] = exp(row[k] - m)、z = Σe、p[k] = e[k]/z；令 s = label_smoothing、
    q[k] = (1 - s) * I(k == label) + s / K。标签等于 ignore_index 的样本
    整体忽略，其余标签须在 [0, K) 内；class_weights 非 None 时其长度须
    等于 K，有效样本权重 w 取该标签对应权重，否则 w 恒为 1.0。返回
    Σ_n(w_n * Σ_k(-q[k] * log(p[k]))) / Σ_n w_n（float）；无有效样本、
    分母非正或任一计算非有限一律抛 ValueError。仅成功时以新 list 缓存
    p、标签副本、逐样本权重与分母并覆盖旧缓存，失败不修改实参与旧缓存。

    backward()：返回新 list[N][K]，被忽略样本的行全为 0.0，其余元素为
    w_n * (p[n][k] - q[n][k]) / Σ_n w_n，均为 float。
    未成功 forward 前调用一律抛 ValueError；重复调用返回等值独立列表。
    两个方法均不修改实参。默认参数时逐位等价于批均 one-hot 交叉熵。
    """

    def __init__(self, label_smoothing=0.0, class_weights=None,
                 ignore_index=None):
        if isinstance(label_smoothing, bool) or not isinstance(
            label_smoothing, (int, float)
        ):
            raise TypeError(
                "label_smoothing 必须是 int/float（拒绝 bool），得到 %s"
                % type(label_smoothing).__name__
            )
        # 超大整数（如 10**400）转 float 抛 OverflowError；统一按非有限
        # 值以 ValueError 拒绝，不向上泄漏 OverflowError。
        try:
            smooth = float(label_smoothing)
        except OverflowError:
            raise ValueError(
                "label_smoothing 必须是有限值（拒绝超大整数/NaN/inf）"
            )
        if not math.isfinite(smooth):
            raise ValueError("label_smoothing 必须是有限值（拒绝 NaN/inf）")
        if smooth < 0.0 or smooth >= 1.0:
            raise ValueError(
                "label_smoothing 必须在 [0, 1) 内，得到 %r"
                % (label_smoothing,)
            )
        self._label_smoothing = smooth

        if class_weights is None:
            self._class_weights = None
        else:
            _require_list(class_weights, "class_weights")
            if len(class_weights) == 0:
                raise ValueError("class_weights 不能为空 list")
            weights = []
            for i, w in enumerate(class_weights):
                if isinstance(w, bool) or not isinstance(w, (int, float)):
                    raise TypeError(
                        "class_weights 的元素必须是 int/float（拒绝 bool），"
                        "得到 %s" % type(w).__name__
                    )
                try:
                    wf = float(w)
                except OverflowError:
                    raise ValueError(
                        "class_weights[%d] 为非有限值（超大整数）" % i
                    )
                if not math.isfinite(wf):
                    raise ValueError(
                        "class_weights[%d] 为非有限值（NaN/inf）" % i
                    )
                if wf < 0.0:
                    raise ValueError(
                        "class_weights[%d] = %r 不能为负" % (i, w)
                    )
                weights.append(wf)
            if not any(w > 0.0 for w in weights):
                raise ValueError("class_weights 至少须有一项严格大于 0")
            self._class_weights = weights

        if ignore_index is not None:
            if isinstance(ignore_index, bool) or not isinstance(
                ignore_index, int
            ):
                raise TypeError(
                    "ignore_index 必须是 int 或 None（拒绝 bool），得到 %s"
                    % type(ignore_index).__name__
                )
        self._ignore_index = ignore_index

        self._probs = None      # 最近一次成功 forward 的 softmax 概率
        self._labels = None     # 最近一次成功 forward 的标签副本
        self._weights = None    # 最近一次成功 forward 的逐样本权重副本
        self._den = None        # 最近一次成功 forward 的有效权重和 Σ_n w_n
        self._out_shape = None  # 最近一次成功 forward 的 (N, K)

    def forward(self, logits, labels):
        """计算（可加权/可忽略标签的）softmax 交叉熵，返回 float 并缓存。"""
        _require_list(logits, "logits")
        _require_list(labels, "labels")
        n_, k_ = _shape_of(logits, 2, "logits")
        if len(labels) != n_:
            raise ValueError(
                "labels 长度 %d 与 logits 样本数 %d 不符"
                % (len(labels), n_)
            )
        if self._class_weights is not None and len(self._class_weights) != k_:
            raise ValueError(
                "class_weights 长度 %d 与类别数 %d 不符"
                % (len(self._class_weights), k_)
            )

        ignore = self._ignore_index
        valid_count = 0
        for n in range(n_):
            label = labels[n]
            if isinstance(label, bool) or not isinstance(label, int):
                raise TypeError(
                    "labels 的元素必须是 int（拒绝 bool），得到 %s"
                    % type(label).__name__
                )
            # 等于 ignore_index 的标签整体忽略（不做取值范围检查）。
            if ignore is not None and label == ignore:
                continue
            if label < 0 or label >= k_:
                raise ValueError(
                    "labels[%d] = %d 超出 [0, %d) 范围" % (n, label, k_)
                )
            valid_count += 1

        smooth = self._label_smoothing
        class_weights = self._class_weights
        probs = []
        sample_weights = [0.0] * n_
        loss_sum = 0.0
        den = 0.0
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for k in range(1, k_):
                if row[k] > m:
                    m = row[k]
            exps = []
            z = 0.0
            for k in range(k_):
                e = math.exp(row[k] - m)
                exps.append(e)
                z += e
            p = [e / z for e in exps]
            probs.append(p)
            label = labels[n]
            if ignore is not None and label == ignore:
                continue
            w = 1.0 if class_weights is None else class_weights[label]
            sample_weights[n] = w
            den += w
            log_z = math.log(z)
            for k in range(k_):
                q = (1.0 - smooth) * (1.0 if k == label else 0.0) \
                    + smooth / k_
                loss_sum += w * (q * (m + log_z - row[k]))

        if valid_count == 0:
            raise ValueError("没有有效样本（全部标签被 ignore_index 忽略）")
        if not math.isfinite(den):
            raise ValueError("前向计算产生非有限值（权重和 NaN/inf）")
        if den <= 0.0:
            raise ValueError("有效样本权重之和非正（分母必须严格大于 0）")

        for n in range(n_):
            for k in range(k_):
                if not math.isfinite(probs[n][k]):
                    raise ValueError("前向计算产生非有限值（NaN/inf）")
        loss = loss_sum / den
        if not math.isfinite(loss):
            raise ValueError("前向计算产生非有限值（NaN/inf）")

        self._probs = probs
        self._labels = list(labels)
        self._weights = sample_weights
        self._den = den
        self._out_shape = (n_, k_)
        return loss

    def backward(self):
        """返回 logits 的梯度新 list[N][K]，元素均为 float。

        被忽略样本的行全为 0.0；其余元素为
        w_n * (p[n][k] - q[n][k]) / Σ_n w_n，其中
        q[n][k] = (1 - s) * I(k == label[n]) + s / K，s 为
        label_smoothing；默认参数下逐位等价于 (p - q) / N。
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._probs is None:
            raise ValueError("尚未成功执行 forward，无法 backward")

        n_, k_ = self._out_shape
        probs = self._probs
        labels = self._labels
        weights = self._weights
        den = self._den
        smooth = self._label_smoothing
        ignore = self._ignore_index
        dx = []
        for n in range(n_):
            label = labels[n]
            if ignore is not None and label == ignore:
                dx.append([0.0] * k_)
                continue
            p_row = probs[n]
            w = weights[n]
            row = []
            for k in range(k_):
                q = (1.0 - smooth) * (1.0 if k == label else 0.0) \
                    + smooth / k_
                row.append(w * (p_row[k] - q) / den)
            dx.append(row)

        for n in range(n_):
            for k in range(k_):
                if not math.isfinite(dx[n][k]):
                    raise ValueError("反向计算产生非有限值（NaN/inf）")
        return dx


def _deep_copy(t):
    """递归复制嵌套 list；标量原样返回。"""
    if isinstance(t, list):
        return [_deep_copy(v) for v in t]
    return t


def _leaf_slots(t):
    """按从外到内的嵌套顺序产出每个标量所在的 (容器, 下标)。"""
    for i in range(len(t)):
        v = t[i]
        if isinstance(v, list):
            for slot in _leaf_slots(v):
                yield slot
        else:
            yield (t, i)


def _flatten_into(t, out):
    """按从外到内的嵌套顺序把全部标量追加到 out。"""
    for v in t:
        if isinstance(v, list):
            _flatten_into(v, out)
        else:
            out.append(v)


def check_gradients(layer, x, dy, eps=1e-6, atol=1e-6, rtol=1e-4):
    """用中心差分数值梯度检验层的前向/反向实现。

    layer 限 Conv2D/ConvTranspose2D/MaxPool2D/AvgPool2D/AdaptiveAvgPool2D/
    AdaptiveMaxPool2D/Flatten/Linear/ReLU/Dropout/Dropout2D/BatchNorm2D
    实例，其余抛 TypeError。解析梯度 a 取自原值 forward(x) 后 backward(dy)
    的对应返回：Conv2D/ConvTranspose2D/Linear 还包含 dweights、dbias，按
    x、weights、bias 顺序检查 dx、dweights、dbias；BatchNorm2D 按 x、
    gamma、beta 顺序检查 dx、dgamma、dbeta；MaxPool2D/AvgPool2D/
    AdaptiveAvgPool2D/AdaptiveMaxPool2D/Flatten/ReLU/Dropout/Dropout2D
    只检查 x。
    AdaptiveMaxPool2D 的任一次前向中任一分箱并列最大一律抛 ValueError
    ——max 在并列点梯度无定义（解析梯度前向与每次正、负扰动前向均检测）。
    对每个标量 v，定义标量损失 L：acc=0.0，按输出嵌套索引从外到内递增
    执行 acc += y*dy（y 为前向输出），数值梯度
    n = (L(v+eps) - L(v-eps)) / (2*eps)，各目标内部标量按嵌套序遍历。

    BatchNorm2D 仅在训练态检查，推理态一律抛 ValueError；每次数值前向都
    重新按当前批次统计（gamma/beta 扰动不影响归一化值 z）。
    Dropout/Dropout2D 训练态以入口随机状态 _s 为基准：解析梯度前向及每次
    正、负扰动前向之前都把 _s 恢复为入口值，使各次前向重放同一掩码
    （Dropout2D 的通道掩码按 n→c 推进），故同一入口状态结果确定；推理态
    不推进随机状态，按恒等映射检查。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    校验顺序固定为 layer、eps/atol/rtol、BatchNorm2D 模式、forward(x)、
    backward(dy)：layer 或数值参数类型错抛 TypeError；BatchNorm2D 推理态、
    eps 非正、容差为负或任一参数非有限抛 ValueError；x、dy 的校验及异常
    完全沿用对应层的 forward/backward（容器/标量类型错抛 TypeError，
    形状、层级、空维、不规则或非有限错误抛 ValueError）；
    计算产生非有限值抛 ValueError。x、dy、参数、训练/推理模式、Dropout/
    Dropout2D 随机状态与掩码、BatchNorm2D 运行统计与旧缓存在所有成功或
    异常路径均原样恢复，实参内容不变。
    """
    if not isinstance(
        layer,
        (Conv2D, ConvTranspose2D, MaxPool2D, AvgPool2D, AdaptiveAvgPool2D,
         AdaptiveMaxPool2D, Flatten, Linear, ReLU, Dropout, Dropout2D,
         BatchNorm2D),
    ):
        raise TypeError(
            "layer 必须是 Conv2D/ConvTranspose2D/MaxPool2D/AvgPool2D/"
            "AdaptiveAvgPool2D/AdaptiveMaxPool2D/Flatten/Linear/ReLU/"
            "Dropout/Dropout2D/BatchNorm2D 实例，得到 %s"
            % type(layer).__name__
        )
    for name, val in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                "%s 必须是 int/float（拒绝 bool），得到 %s"
                % (name, type(val).__name__)
            )
        if not math.isfinite(val):
            raise ValueError("%s 必须是有限值（拒绝 NaN/inf）" % name)
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if atol < 0:
        raise ValueError("atol 必须为非负数")
    if rtol < 0:
        raise ValueError("rtol 必须为非负数")

    is_batchnorm = isinstance(layer, BatchNorm2D)
    is_dropout = isinstance(layer, (Dropout, Dropout2D))
    is_adaptive_maxpool = isinstance(layer, AdaptiveMaxPool2D)
    if is_batchnorm and not layer._training:
        raise ValueError("BatchNorm2D 仅在训练态支持梯度检查")

    saved_state = dict(layer.__dict__)
    dropout_entry_s = layer._s if is_dropout else None
    try:
        # Dropout/Dropout2D 训练态：解析梯度前向先回到入口随机状态，
        # 掩码随后可重放。
        if is_dropout and layer._training:
            layer._s = dropout_entry_s
        layer.forward(x)  # x 的校验沿用该层 forward
        if is_adaptive_maxpool:
            _check_adaptive_maxpool_ties(layer, x)
        grad = layer.backward(dy)  # dy 的校验沿用该层 backward
        if isinstance(layer, (Conv2D, ConvTranspose2D, Linear)):
            dx, dw, db = grad
            targets = (
                ("x", x, dx),
                ("weights", layer._weights, dw),
                ("bias", layer._bias, db),
            )
        elif is_batchnorm:
            dx, dgamma, dbeta = grad
            targets = (
                ("x", x, dx),
                ("gamma", layer._gamma, dgamma),
                ("beta", layer._beta, dbeta),
            )
        else:
            # MaxPool2D/AvgPool2D/AdaptiveAvgPool2D/AdaptiveMaxPool2D/
            # Flatten/ReLU/Dropout/Dropout2D（训练态与推理态）只检查 x。
            targets = (("x", x, grad),)

        def loss(x_arg):
            y = layer.forward(x_arg)
            # AdaptiveMaxPool2D 并列最大在每次（含扰动）前向一并检出。
            if is_adaptive_maxpool:
                _check_adaptive_maxpool_ties(layer, x_arg)
            acc = 0.0

            def rec(a, b):
                nonlocal acc
                if isinstance(a, list):
                    for i in range(len(a)):
                        rec(a[i], b[i])
                else:
                    acc += a * b

            rec(y, dy)
            return acc

        def replay_dropout_mask():
            """正/负扰动前恢复入口随机状态，使训练前向重放同一掩码。"""
            if is_dropout and layer._training:
                layer._s = dropout_entry_s

        ok = True
        max_e = 0.0
        max_r = 0.0
        for name, original, analytic in targets:
            analytic_flat = []
            _flatten_into(analytic, analytic_flat)
            work = _deep_copy(original)  # 只扰动副本，原张量不被修改
            if name == "weights":
                layer._weights = work
            elif name == "bias":
                layer._bias = work
            elif name == "gamma":
                layer._gamma = work
            elif name == "beta":
                layer._beta = work
            idx = 0
            for container, i in _leaf_slots(work):
                v = container[i]
                replay_dropout_mask()
                container[i] = v + eps
                lp = loss(work if name == "x" else x)
                replay_dropout_mask()
                container[i] = v - eps
                lm = loss(work if name == "x" else x)
                container[i] = v
                if not (math.isfinite(lp) and math.isfinite(lm)):
                    raise ValueError("数值梯度计算产生非有限值（NaN/inf）")
                n = (lp - lm) / (2 * eps)
                a = analytic_flat[idx]
                idx += 1
                if not (math.isfinite(a) and math.isfinite(n)):
                    raise ValueError("梯度计算产生非有限值（NaN/inf）")
                aa = abs(a)
                an = abs(n)
                e = abs(a - n)
                r = e / max(aa, an, 1e-12)
                if e > atol + rtol * max(aa, an):
                    ok = False
                if e > max_e:
                    max_e = e
                if r > max_r:
                    max_r = r
    finally:
        layer.__dict__.clear()
        layer.__dict__.update(saved_state)

    return (bool(ok), float(max_e), float(max_r))


def _check_pool_window_ties(pool, inp):
    """扫描 MaxPool2D 对 inp 的全部有效窗口，任一窗口并列最大即抛 ValueError。

    窗口/步长/补边/膨胀/ceil_mode 规则与 MaxPool2D.forward 完全一致
    （采样坐标 oh*SH-PT+kh*DH、ow*SW-PL+kw*DW，补边位置不参与比较，
    窗口数按 ceil_mode 调整）：窗口内同一最大值出现两次及以上即视为
    并列。inp 须为池化层合法四维输入。
    """
    kh_, kw_ = pool._kernel_size
    sh_, sw_ = pool._stride
    pt_, pb_, pl_, pr_ = pool._padding
    dh_, dw_ = pool._dilation
    eh_ = (kh_ - 1) * dh_ + 1
    ew_ = (kw_ - 1) * dw_ + 1
    n_ = len(inp)
    c_ = len(inp[0])
    h_ = len(inp[0][0])
    w_ = len(inp[0][0][0])
    oh_ = _avgpool_out_size(h_, pt_, pb_, eh_, sh_, pool._ceil_mode)
    ow_ = _avgpool_out_size(w_, pl_, pr_, ew_, sw_, pool._ceil_mode)
    for n in range(n_):
        x_n = inp[n]
        for c in range(c_):
            x_c = x_n[c]
            for oh in range(oh_):
                base_h = oh * sh_ - pt_
                for ow in range(ow_):
                    base_w = ow * sw_ - pl_
                    best = None
                    tied = False
                    for kh in range(kh_):
                        ih = base_h + kh * dh_
                        if ih < 0 or ih >= h_:
                            continue
                        x_row = x_c[ih]
                        for kw in range(kw_):
                            iw = base_w + kw * dw_
                            if 0 <= iw < w_:
                                v = x_row[iw]
                                if best is None or v > best:
                                    best = v
                                    tied = False
                                elif v == best:
                                    tied = True
                    if tied:
                        raise ValueError(
                            "池化有效窗口存在并列最大值，max 梯度在该点无定义"
                        )


def _check_adaptive_maxpool_ties(pool, inp):
    """扫描 AdaptiveMaxPool2D 对 inp 的全部分箱，任一分箱并列最大即抛 ValueError。

    分箱规则与 AdaptiveMaxPool2D.forward 完全一致：输出位置 oh 的输入
    区间为 [floor(oh*H/OH), ceil((oh+1)*H/OH))，宽轴同理；分箱内同一
    最大值出现两次及以上即视为并列。inp 须为该层合法四维输入。
    """
    oh_, ow_ = pool._output_size
    n_ = len(inp)
    c_ = len(inp[0])
    h_ = len(inp[0][0])
    w_ = len(inp[0][0][0])
    h_bins = pool._bins(h_, oh_)
    w_bins = pool._bins(w_, ow_)
    for n in range(n_):
        for c in range(c_):
            x_c = inp[n][c]
            for ih0, ih1 in h_bins:
                for iw0, iw1 in w_bins:
                    best = None
                    tied = False
                    for ih in range(ih0, ih1):
                        x_row = x_c[ih]
                        for iw in range(iw0, iw1):
                            v = x_row[iw]
                            if best is None or v > best:
                                best = v
                                tied = False
                            elif v == best:
                                tied = True
                    if tied:
                        raise ValueError(
                            "自适应最大池化分箱存在并列最大值，max 梯度在该点无定义"
                        )


def check_cnn_gradients(
    conv, pool, flatten, linear, x, dy, eps=1e-6, atol=1e-6, rtol=1e-4
):
    """用中心差分数值梯度检验 Conv2D→Pool2D→Flatten→Linear 整链。

    conv/pool/flatten/linear 须依次为 Conv2D/MaxPool2D、AvgPool2D、
    AdaptiveAvgPool2D 或 AdaptiveMaxPool2D/Flatten/Linear 实例，其余抛
    TypeError。前向按
    conv→pool→flatten→linear 执行，解析梯度按
    linear→flatten→pool→conv 逆序取各层 backward 结果。
    标量损失 L：acc=0.0，按线性层输出的嵌套索引从外到内递增执行
    acc += y*dy（y 为整链前向输出）。

    数值梯度依次扰动 x、conv 的 weights/bias、linear 的 weights/bias
    （各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    pool 为 MaxPool2D 时，任一次前向中任一池化有效窗口并列最大（补边
    位置不参与比较）一律抛 ValueError——max 在并列点梯度无定义；
    pool 为 AdaptiveMaxPool2D 时，任一次前向中任一自适应分箱并列最大
    一律抛 ValueError；pool 为 AvgPool2D 或 AdaptiveAvgPool2D 时不做
    并列检测。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    eps/atol/rtol 须为有限 int/float（拒绝 bool）：类型错抛 TypeError；
    eps 非正、容差为负或任一非有限抛 ValueError。x、dy 的校验及异常
    完全沿用各层 forward/backward；计算产生非有限值抛 ValueError。
    x、dy、参数及四层实例状态（含缓存）在所有成功或异常路径均原样恢复。
    """
    if not isinstance(conv, Conv2D):
        raise TypeError(
            "conv 必须是 Conv2D 实例，得到 %s" % type(conv).__name__
        )
    if not isinstance(
        pool, (MaxPool2D, AvgPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D)
    ):
        raise TypeError(
            "pool 必须是 MaxPool2D、AvgPool2D、AdaptiveAvgPool2D 或"
            " AdaptiveMaxPool2D 实例，得到 %s" % type(pool).__name__
        )
    if not isinstance(flatten, Flatten):
        raise TypeError(
            "flatten 必须是 Flatten 实例，得到 %s" % type(flatten).__name__
        )
    if not isinstance(linear, Linear):
        raise TypeError(
            "linear 必须是 Linear 实例，得到 %s" % type(linear).__name__
        )
    for name, val in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                "%s 必须是 int/float（拒绝 bool），得到 %s"
                % (name, type(val).__name__)
            )
        if not math.isfinite(val):
            raise ValueError("%s 必须是有限值（拒绝 NaN/inf）" % name)
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if atol < 0:
        raise ValueError("atol 必须为非负数")
    if rtol < 0:
        raise ValueError("rtol 必须为非负数")

    saved_state = (
        dict(conv.__dict__),
        dict(pool.__dict__),
        dict(flatten.__dict__),
        dict(linear.__dict__),
    )
    try:
        def chain_forward(x_arg):
            conv_out = conv.forward(x_arg)
            if isinstance(pool, MaxPool2D):
                _check_pool_window_ties(pool, conv_out)
            elif isinstance(pool, AdaptiveMaxPool2D):
                _check_adaptive_maxpool_ties(pool, conv_out)
            pool_out = pool.forward(conv_out)
            flat = flatten.forward(pool_out)
            return linear.forward(flat)

        def loss(x_arg):
            y = chain_forward(x_arg)
            acc = 0.0

            def rec(a, b):
                nonlocal acc
                if isinstance(a, list):
                    for i in range(len(a)):
                        rec(a[i], b[i])
                else:
                    acc += a * b

            rec(y, dy)
            return acc

        # 前向按 conv→pool→flatten→linear；x 的校验沿各层 forward，
        # MaxPool2D/AdaptiveMaxPool2D 并列最大在此一并检出。
        chain_forward(x)

        # 解析梯度按 linear→flatten→pool→conv 逆序；dy 校验沿 linear.backward。
        dx_flat, dlw, dlb = linear.backward(dy)
        dx_pool = flatten.backward(dx_flat)
        dx_conv_out = pool.backward(dx_pool)
        dx, dcw, dcb = conv.backward(dx_conv_out)

        targets = (
            ("x", x, dx),
            ("conv_weights", conv._weights, dcw),
            ("conv_bias", conv._bias, dcb),
            ("linear_weights", linear._weights, dlw),
            ("linear_bias", linear._bias, dlb),
        )

        ok = True
        max_e = 0.0
        max_r = 0.0
        for name, original, analytic in targets:
            analytic_flat = []
            _flatten_into(analytic, analytic_flat)
            work = _deep_copy(original)  # 只扰动副本，原张量不被修改
            if name == "conv_weights":
                conv._weights = work
            elif name == "conv_bias":
                conv._bias = work
            elif name == "linear_weights":
                linear._weights = work
            elif name == "linear_bias":
                linear._bias = work
            idx = 0
            for container, i in _leaf_slots(work):
                v = container[i]
                container[i] = v + eps
                lp = loss(work if name == "x" else x)
                container[i] = v - eps
                lm = loss(work if name == "x" else x)
                container[i] = v
                if not (math.isfinite(lp) and math.isfinite(lm)):
                    raise ValueError("数值梯度计算产生非有限值（NaN/inf）")
                n = (lp - lm) / (2 * eps)
                a = analytic_flat[idx]
                idx += 1
                if not (math.isfinite(a) and math.isfinite(n)):
                    raise ValueError("梯度计算产生非有限值（NaN/inf）")
                aa = abs(a)
                an = abs(n)
                e = abs(a - n)
                r = e / max(aa, an, 1e-12)
                if e > atol + rtol * max(aa, an):
                    ok = False
                if e > max_e:
                    max_e = e
                if r > max_r:
                    max_r = r
    finally:
        for layer, state in zip(
            (conv, pool, flatten, linear), saved_state
        ):
            layer.__dict__.clear()
            layer.__dict__.update(state)

    return (bool(ok), float(max_e), float(max_r))


def check_norm_gradients(
    conv, bn, dropout, pool, flatten, linear, x, dy,
    eps=1e-6, atol=1e-6, rtol=1e-4,
):
    """用中心差分数值梯度检验 Conv2D→BatchNorm2D→Dropout→Pool2D→Flatten→Linear 训练链。

    conv/bn/dropout/pool/flatten/linear 须依次为 Conv2D/BatchNorm2D/Dropout/
    MaxPool2D、AvgPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D/Flatten/
    Linear 实例，其余抛
    TypeError；bn 或 dropout 非训练态抛 ValueError。前向按
    conv→bn→dropout→pool→flatten→linear 执行，解析梯度按
    linear→flatten→pool→dropout→bn→conv 逆序取各层 backward 结果。标量损失 L：acc=0.0，按线性层输出的嵌套索引从外
    到内递增执行 acc += y*dy（y 为整链前向输出）。

    数值梯度依次扰动 x、conv 的 weights/bias、bn 的 gamma/beta、linear 的
    weights/bias（各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    每次前向（含解析梯度前向与每次正、负扰动前向）之前都把 dropout 的随机
    状态 _s 恢复为入口值，使各次前向重放同一掩码，故同一入口状态结果确定。
    BatchNorm2D 每次数值前向都重新按当前批次统计。pool 为 MaxPool2D 时，
    任一次前向中任一池化有效窗口并列最大（补边位置不参与比较）一律抛
    ValueError——max 在并列点梯度无定义；pool 为 AdaptiveMaxPool2D 时，
    任一次前向中任一自适应分箱并列最大一律抛 ValueError；pool 为
    AvgPool2D 或 AdaptiveAvgPool2D 时不做并列检测。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    eps/atol/rtol 须为有限 int/float（拒绝 bool）：类型错抛 TypeError；
    eps 非正、容差为负或任一非有限抛 ValueError。x、dy 的校验及异常
    完全沿用各层 forward/backward；计算产生非有限值抛 ValueError。
    x、dy、参数及六层实例状态（训练/推理模式、缓存、Dropout 随机状态与
    掩码、BatchNorm2D 运行统计）在所有成功或异常路径均原样恢复。
    """
    if not isinstance(conv, Conv2D):
        raise TypeError(
            "conv 必须是 Conv2D 实例，得到 %s" % type(conv).__name__
        )
    if not isinstance(bn, BatchNorm2D):
        raise TypeError(
            "bn 必须是 BatchNorm2D 实例，得到 %s" % type(bn).__name__
        )
    if not isinstance(dropout, Dropout):
        raise TypeError(
            "dropout 必须是 Dropout 实例，得到 %s" % type(dropout).__name__
        )
    if not isinstance(
        pool, (MaxPool2D, AvgPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D)
    ):
        raise TypeError(
            "pool 必须是 MaxPool2D、AvgPool2D、AdaptiveAvgPool2D 或"
            " AdaptiveMaxPool2D 实例，得到 %s" % type(pool).__name__
        )
    if not isinstance(flatten, Flatten):
        raise TypeError(
            "flatten 必须是 Flatten 实例，得到 %s" % type(flatten).__name__
        )
    if not isinstance(linear, Linear):
        raise TypeError(
            "linear 必须是 Linear 实例，得到 %s" % type(linear).__name__
        )
    for name, val in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                "%s 必须是 int/float（拒绝 bool），得到 %s"
                % (name, type(val).__name__)
            )
        if not math.isfinite(val):
            raise ValueError("%s 必须是有限值（拒绝 NaN/inf）" % name)
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if atol < 0:
        raise ValueError("atol 必须为非负数")
    if rtol < 0:
        raise ValueError("rtol 必须为非负数")
    if not bn._training:
        raise ValueError("BatchNorm2D 仅在训练态支持梯度检查")
    if not dropout._training:
        raise ValueError("Dropout 仅在训练态支持梯度检查")

    saved_state = (
        dict(conv.__dict__),
        dict(bn.__dict__),
        dict(dropout.__dict__),
        dict(pool.__dict__),
        dict(flatten.__dict__),
        dict(linear.__dict__),
    )
    dropout_entry_s = dropout._s
    try:
        def chain_forward(x_arg):
            # 每次前向前恢复入口随机状态，使训练前向重放同一掩码。
            dropout._s = dropout_entry_s
            conv_out = conv.forward(x_arg)
            bn_out = bn.forward(conv_out)
            drop_out = dropout.forward(bn_out)
            if isinstance(pool, MaxPool2D):
                _check_pool_window_ties(pool, drop_out)
            elif isinstance(pool, AdaptiveMaxPool2D):
                _check_adaptive_maxpool_ties(pool, drop_out)
            pool_out = pool.forward(drop_out)
            flat = flatten.forward(pool_out)
            return linear.forward(flat)

        def loss(x_arg):
            y = chain_forward(x_arg)
            acc = 0.0

            def rec(a, b):
                nonlocal acc
                if isinstance(a, list):
                    for i in range(len(a)):
                        rec(a[i], b[i])
                else:
                    acc += a * b

            rec(y, dy)
            return acc

        # 前向按 conv→bn→dropout→pool→flatten→linear；x 的校验沿各层
        # forward，MaxPool2D/AdaptiveMaxPool2D 并列最大在此一并检出。
        chain_forward(x)

        # 解析梯度按 linear→flatten→pool→dropout→bn→conv 逆序；
        # dy 校验沿 linear.backward。
        dx_flat, dlw, dlb = linear.backward(dy)
        dx_pool = flatten.backward(dx_flat)
        dx_drop = pool.backward(dx_pool)
        dx_bn = dropout.backward(dx_drop)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        dx, dcw, dcb = conv.backward(dx_conv)

        targets = (
            ("x", x, dx),
            ("conv_weights", conv._weights, dcw),
            ("conv_bias", conv._bias, dcb),
            ("bn_gamma", bn._gamma, dgamma),
            ("bn_beta", bn._beta, dbeta),
            ("linear_weights", linear._weights, dlw),
            ("linear_bias", linear._bias, dlb),
        )

        ok = True
        max_e = 0.0
        max_r = 0.0
        for name, original, analytic in targets:
            analytic_flat = []
            _flatten_into(analytic, analytic_flat)
            work = _deep_copy(original)  # 只扰动副本，原张量不被修改
            if name == "conv_weights":
                conv._weights = work
            elif name == "conv_bias":
                conv._bias = work
            elif name == "bn_gamma":
                bn._gamma = work
            elif name == "bn_beta":
                bn._beta = work
            elif name == "linear_weights":
                linear._weights = work
            elif name == "linear_bias":
                linear._bias = work
            idx = 0
            for container, i in _leaf_slots(work):
                v = container[i]
                container[i] = v + eps
                lp = loss(work if name == "x" else x)
                container[i] = v - eps
                lm = loss(work if name == "x" else x)
                container[i] = v
                if not (math.isfinite(lp) and math.isfinite(lm)):
                    raise ValueError("数值梯度计算产生非有限值（NaN/inf）")
                n = (lp - lm) / (2 * eps)
                a = analytic_flat[idx]
                idx += 1
                if not (math.isfinite(a) and math.isfinite(n)):
                    raise ValueError("梯度计算产生非有限值（NaN/inf）")
                aa = abs(a)
                an = abs(n)
                e = abs(a - n)
                r = e / max(aa, an, 1e-12)
                if e > atol + rtol * max(aa, an):
                    ok = False
                if e > max_e:
                    max_e = e
                if r > max_r:
                    max_r = r
    finally:
        for layer, state in zip(
            (conv, bn, dropout, pool, flatten, linear), saved_state
        ):
            layer.__dict__.clear()
            layer.__dict__.update(state)

    return (bool(ok), float(max_e), float(max_r))


def check_train_gradients(
    conv, bn, dropout, pool, flatten, linear, loss, x, labels,
    eps=1e-6, atol=1e-6, rtol=1e-4,
):
    """用中心差分数值梯度检验 Conv2D→BatchNorm2D→Dropout→Pool2D→Flatten→Linear→SoftmaxCrossEntropy 训练链。

    conv/bn/dropout/pool/flatten/linear 须依次为 Conv2D/BatchNorm2D/Dropout/
    MaxPool2D、AvgPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D/Flatten/
    Linear 实例，loss 须为
    SoftmaxCrossEntropy 实例，其余抛 TypeError；bn 或 dropout 非训练态抛
    ValueError。前向按 conv→bn→dropout→pool→flatten→linear 执行得
    logits，标量损失 L = loss.forward(logits, labels) 返回的 float 批均
    损失；解析梯度自 loss.backward() 起按 linear→flatten→pool→dropout→
    bn→conv 逆序取各层 backward 结果。

    数值梯度依次扰动 x、conv 的 weights/bias、bn 的 gamma/beta、linear 的
    weights/bias（各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    每次前向（含解析梯度前向与每次正、负扰动前向）之前都把 dropout 的随机
    状态 _s 恢复为入口值，使各次前向重放同一掩码，故同一入口状态结果确定。
    BatchNorm2D 每次数值前向都重新按当前批次统计。pool 为 MaxPool2D 时，
    任一次前向中任一池化有效窗口并列最大（补边位置不参与比较）一律抛
    ValueError——max 在并列点梯度无定义；pool 为 AdaptiveMaxPool2D 时，
    任一次前向中任一自适应分箱并列最大一律抛 ValueError；pool 为
    AvgPool2D 或 AdaptiveAvgPool2D 时不做并列检测。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    eps/atol/rtol 须为有限 int/float（拒绝 bool）：类型错抛 TypeError；
    eps 非正、容差为负或任一非有限抛 ValueError。x 的校验沿各层 forward，
    labels 的校验沿 loss.forward；计算产生非有限值抛 ValueError。
    x、labels、参数及七层实例状态（训练/推理模式、缓存、Dropout 随机状态
    与掩码、BatchNorm2D 运行统计、SoftmaxCrossEntropy 缓存）在所有成功
    或异常路径均原样恢复（含参数引用），重复调用结果一致。
    """
    if not isinstance(conv, Conv2D):
        raise TypeError(
            "conv 必须是 Conv2D 实例，得到 %s" % type(conv).__name__
        )
    if not isinstance(bn, BatchNorm2D):
        raise TypeError(
            "bn 必须是 BatchNorm2D 实例，得到 %s" % type(bn).__name__
        )
    if not isinstance(dropout, Dropout):
        raise TypeError(
            "dropout 必须是 Dropout 实例，得到 %s" % type(dropout).__name__
        )
    if not isinstance(
        pool, (MaxPool2D, AvgPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D)
    ):
        raise TypeError(
            "pool 必须是 MaxPool2D、AvgPool2D、AdaptiveAvgPool2D 或"
            " AdaptiveMaxPool2D 实例，得到 %s" % type(pool).__name__
        )
    if not isinstance(flatten, Flatten):
        raise TypeError(
            "flatten 必须是 Flatten 实例，得到 %s" % type(flatten).__name__
        )
    if not isinstance(linear, Linear):
        raise TypeError(
            "linear 必须是 Linear 实例，得到 %s" % type(linear).__name__
        )
    if not isinstance(loss, SoftmaxCrossEntropy):
        raise TypeError(
            "loss 必须是 SoftmaxCrossEntropy 实例，得到 %s"
            % type(loss).__name__
        )
    for name, val in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                "%s 必须是 int/float（拒绝 bool），得到 %s"
                % (name, type(val).__name__)
            )
        if not math.isfinite(val):
            raise ValueError("%s 必须是有限值（拒绝 NaN/inf）" % name)
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if atol < 0:
        raise ValueError("atol 必须为非负数")
    if rtol < 0:
        raise ValueError("rtol 必须为非负数")
    if not bn._training:
        raise ValueError("BatchNorm2D 仅在训练态支持梯度检查")
    if not dropout._training:
        raise ValueError("Dropout 仅在训练态支持梯度检查")

    saved_state = (
        dict(conv.__dict__),
        dict(bn.__dict__),
        dict(dropout.__dict__),
        dict(pool.__dict__),
        dict(flatten.__dict__),
        dict(linear.__dict__),
        dict(loss.__dict__),
    )
    dropout_entry_s = dropout._s
    try:
        def chain_forward(x_arg):
            # 每次前向前恢复入口随机状态，使训练前向重放同一掩码。
            dropout._s = dropout_entry_s
            conv_out = conv.forward(x_arg)
            bn_out = bn.forward(conv_out)
            drop_out = dropout.forward(bn_out)
            if isinstance(pool, MaxPool2D):
                _check_pool_window_ties(pool, drop_out)
            elif isinstance(pool, AdaptiveMaxPool2D):
                _check_adaptive_maxpool_ties(pool, drop_out)
            pool_out = pool.forward(drop_out)
            flat = flatten.forward(pool_out)
            return linear.forward(flat)

        def loss_value(x_arg):
            return loss.forward(chain_forward(x_arg), labels)

        # 前向按 conv→bn→dropout→pool→flatten→linear→loss；x 的校验沿
        # 各层 forward，labels 的校验沿 loss.forward，MaxPool2D 并列最大
        # 在此一并检出。
        loss_value(x)

        # 解析梯度自 loss.backward() 起按 linear→flatten→pool→dropout→
        # bn→conv 逆序。
        dlogits = loss.backward()
        dx_flat, dlw, dlb = linear.backward(dlogits)
        dx_pool = flatten.backward(dx_flat)
        dx_drop = pool.backward(dx_pool)
        dx_bn = dropout.backward(dx_drop)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        dx, dcw, dcb = conv.backward(dx_conv)

        targets = (
            ("x", x, dx),
            ("conv_weights", conv._weights, dcw),
            ("conv_bias", conv._bias, dcb),
            ("bn_gamma", bn._gamma, dgamma),
            ("bn_beta", bn._beta, dbeta),
            ("linear_weights", linear._weights, dlw),
            ("linear_bias", linear._bias, dlb),
        )

        ok = True
        max_e = 0.0
        max_r = 0.0
        for name, original, analytic in targets:
            analytic_flat = []
            _flatten_into(analytic, analytic_flat)
            work = _deep_copy(original)  # 只扰动副本，原张量不被修改
            if name == "conv_weights":
                conv._weights = work
            elif name == "conv_bias":
                conv._bias = work
            elif name == "bn_gamma":
                bn._gamma = work
            elif name == "bn_beta":
                bn._beta = work
            elif name == "linear_weights":
                linear._weights = work
            elif name == "linear_bias":
                linear._bias = work
            idx = 0
            for container, i in _leaf_slots(work):
                v = container[i]
                container[i] = v + eps
                lp = loss_value(work if name == "x" else x)
                container[i] = v - eps
                lm = loss_value(work if name == "x" else x)
                container[i] = v
                if not (math.isfinite(lp) and math.isfinite(lm)):
                    raise ValueError("数值梯度计算产生非有限值（NaN/inf）")
                n = (lp - lm) / (2 * eps)
                a = analytic_flat[idx]
                idx += 1
                if not (math.isfinite(a) and math.isfinite(n)):
                    raise ValueError("梯度计算产生非有限值（NaN/inf）")
                aa = abs(a)
                an = abs(n)
                e = abs(a - n)
                r = e / max(aa, an, 1e-12)
                if e > atol + rtol * max(aa, an):
                    ok = False
                if e > max_e:
                    max_e = e
                if r > max_r:
                    max_r = r
    finally:
        for layer, state in zip(
            (conv, bn, dropout, pool, flatten, linear, loss), saved_state
        ):
            layer.__dict__.clear()
            layer.__dict__.update(state)

    return (bool(ok), float(max_e), float(max_r))


# ---------------------------------------------------------------------------
# 命令行训练：python convnet.py train OUTPUT
# ---------------------------------------------------------------------------

_TRAIN_DATA_PATH = os.path.join("data", "tiny.csv")
_TRAIN_EXPECTED_BYTES = b"0,0,0,0,0\n1,1,1,1,1\n"
_TRAIN_NUM_CLASSES = 2
_TRAIN_NUM_FEATURES = 4
_TRAIN_EPOCHS = 50
_TRAIN_LR = 0.5


class _TrainDataError(Exception):
    """训练数据缺失或字节内容不符。"""


def _load_train_samples():
    """读取 data/tiny.csv，严格校验字节后解析为 (images, labels)。

    images: [N][1][1][4]（NCHW，c→h→w 各为 1）；labels: [N]，取值 0/1。
    文件缺失、不可读或字节不等于约定内容一律抛 _TrainDataError。
    """
    try:
        with open(_TRAIN_DATA_PATH, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise _TrainDataError("无法读取 %s：%s" % (_TRAIN_DATA_PATH, exc))
    if raw != _TRAIN_EXPECTED_BYTES:
        raise _TrainDataError(
            "%s 的字节内容与预期不符" % _TRAIN_DATA_PATH
        )

    images = []
    labels = []
    for line in raw.decode("utf-8").split("\n"):
        if line == "":
            continue
        fields = line.split(",")
        if len(fields) != _TRAIN_NUM_FEATURES + 1:
            raise _TrainDataError("%s 存在字段数不为 5 的行" % _TRAIN_DATA_PATH)
        values = []
        for field in fields:
            if field not in ("0", "1"):
                raise _TrainDataError(
                    "%s 含非 0/1 字段：%r" % (_TRAIN_DATA_PATH, field)
                )
            values.append(int(field))
        images.append([[values[:_TRAIN_NUM_FEATURES]]])
        labels.append(values[_TRAIN_NUM_FEATURES])
    if not images:
        raise _TrainDataError("%s 不含任何样本" % _TRAIN_DATA_PATH)
    return images, labels


def _train_run():
    """执行确定性训练，返回 (weights_values, bias, losses, accuracy)。"""
    images, labels = _load_train_samples()
    n_ = len(images)

    flatten = Flatten()
    weights = [[0.0] * _TRAIN_NUM_FEATURES for _ in range(_TRAIN_NUM_CLASSES)]
    bias = [0.0] * _TRAIN_NUM_CLASSES
    linear = Linear(weights, bias)

    losses = []
    lr = _TRAIN_LR
    for _ in range(_TRAIN_EPOCHS):
        flat = flatten.forward(images)
        logits = linear.forward(flat)

        # 交叉熵前向：softmax 先减去每行最大 logit；记录更新前的批均损失。
        probs = []
        loss_sum = 0.0
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for o in range(1, _TRAIN_NUM_CLASSES):
                if row[o] > m:
                    m = row[o]
            exps = []
            denom = 0.0
            for o in range(_TRAIN_NUM_CLASSES):
                e = math.exp(row[o] - m)
                exps.append(e)
                denom += e
            p = [e / denom for e in exps]
            probs.append(p)
            loss_sum += -math.log(p[labels[n]])
        loss = loss_sum / n_
        if not math.isfinite(loss):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
        losses.append(loss)

        # 反向：softmax-交叉熵对 logits 的梯度为 p - onehot。
        grad_logits = [
            [
                probs[n][o] - (1.0 if labels[n] == o else 0.0)
                for o in range(_TRAIN_NUM_CLASSES)
            ]
            for n in range(n_)
        ]
        _, dw, db = linear.backward(grad_logits)

        # 除 N 后同步 SGD 更新（先全部算出新值再提交）。
        inv_n = 1.0 / n_
        new_weights = [
            [
                weights[o][i] - lr * dw[o][i] * inv_n
                for i in range(_TRAIN_NUM_FEATURES)
            ]
            for o in range(_TRAIN_NUM_CLASSES)
        ]
        new_bias = [
            bias[o] - lr * db[o] * inv_n
            for o in range(_TRAIN_NUM_CLASSES)
        ]
        for o in range(_TRAIN_NUM_CLASSES):
            if not math.isfinite(new_bias[o]):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            for i in range(_TRAIN_NUM_FEATURES):
                if not math.isfinite(new_weights[o][i]):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
        weights = new_weights
        bias = new_bias
        linear = Linear(weights, bias)

    # 损失须逐轮严格下降。
    for e in range(1, _TRAIN_EPOCHS):
        if not losses[e] < losses[e - 1]:
            raise ValueError("训练损失未逐轮严格下降")

    # 末次更新后以最大 logit 预测，并列取小类。
    final_logits = linear.forward(flatten.forward(images))
    correct = 0
    for n in range(n_):
        row = final_logits[n]
        pred = 0
        for o in range(1, _TRAIN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        if pred == labels[n]:
            correct += 1
    accuracy = correct / n_
    if accuracy != 1.0:
        raise ValueError("训练后 accuracy 不为 1.0")
    return weights, bias, losses, accuracy


def _fmt_float(v):
    """格式化为固定 1 位整数 + 12 位小数；负零规范化为 0.000000000000。"""
    if not math.isfinite(v):
        raise ValueError("出现非有限值（NaN/inf），禁止写入产物")
    s = "%.12f" % v
    if s == "-0.000000000000":
        s = "0.000000000000"
    return s


def _dump_compact(value):
    """手工生成紧凑 JSON：int 原样、float 固定 12 位小数；键序由 dict 保证。"""
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=False) + ":" + _dump_compact(v)
            for k, v in value.items()
        ) + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _fmt_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_dump_compact(v) for v in value) + "]"
    raise TypeError("产物含不支持的类型：%s" % type(value).__name__)


def _atomic_write_output(output_path, payload):
    """先写同目录临时文件，再原子替换；任何失败都不改动既有 output_path。"""
    directory = os.path.dirname(os.path.abspath(output_path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-train-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _cmd_train(output_path):
    """train 子命令主体；数据/计算失败返回 1。"""
    try:
        weights, bias, losses, accuracy = _train_run()
        artifact = {
            "weights": {"values": weights, "bias": bias},
            "metrics": {
                "epochs": _TRAIN_EPOCHS,
                "lr": float(_TRAIN_LR),
                "loss": losses,
                "accuracy": accuracy,
            },
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行 CNN 训练：python convnet.py fitcnn OUTPUT
# ---------------------------------------------------------------------------

_CNN_NUM_CLASSES = 2
_CNN_NUM_FEATURES = 4
_CNN_EPOCHS = 20
_CNN_LR = 0.1
_CNN_CONV_INIT = [[[[1.0]]], [[[-1.0]]]]   # [2][1][1][1]：两枚 1×1 卷积核
_CNN_LINEAR_INIT = [[-1.0, 1.0], [1.0, -1.0]]  # [2][2]


def _load_cnn_samples():
    """沿用 train 的数据校验，将四特征按行优先重排为 [N][1][2][2]。"""
    flat_images, labels = _load_train_samples()
    images = []
    for n in range(len(flat_images)):
        row = flat_images[n][0][0]
        images.append([[[row[0], row[1]], [row[2], row[3]]]])
    return images, labels


def _cnn_forward(conv_w, conv_b, lin_w, lin_b, images):
    """Conv2D(1×1) → MaxPool2D(2,2,0) → Flatten → Linear 一次前向。"""
    conv = Conv2D(conv_w, conv_b)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear = Linear(lin_w, lin_b)
    conv_out = conv.forward(images)
    pool_out = pool.forward(conv_out)
    flat = flatten.forward(pool_out)
    return linear.forward(flat)


def _fitcnn_run():
    """执行确定性 CNN 训练，返回 (conv_w, conv_b, lin_w, lin_b, losses, accuracy)。"""
    images, labels = _load_cnn_samples()
    n_ = len(images)

    conv_w = _deep_copy(_CNN_CONV_INIT)
    conv_b = [0.0] * _CNN_NUM_CLASSES
    lin_w = _deep_copy(_CNN_LINEAR_INIT)
    lin_b = [0.0] * _CNN_NUM_CLASSES

    losses = []
    lr = _CNN_LR
    for _ in range(_CNN_EPOCHS):
        conv = Conv2D(conv_w, conv_b)
        pool = MaxPool2D(2, 2, 0)
        flatten = Flatten()
        linear = Linear(lin_w, lin_b)

        conv_out = conv.forward(images)
        pool_out = pool.forward(conv_out)
        flat = flatten.forward(pool_out)
        logits = linear.forward(flat)

        # 交叉熵前向：softmax 先减去每行最大 logit；记录更新前的批均损失。
        probs = []
        loss_sum = 0.0
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > m:
                    m = row[o]
            exps = []
            denom = 0.0
            for o in range(_CNN_NUM_CLASSES):
                e = math.exp(row[o] - m)
                exps.append(e)
                denom += e
            p = [e / denom for e in exps]
            probs.append(p)
            loss_sum += -math.log(p[labels[n]])
        loss = loss_sum / n_
        if not math.isfinite(loss):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
        losses.append(loss)

        # 反向：softmax-交叉熵对 logits 的梯度为 p - onehot，沿链逐层回传。
        grad_logits = [
            [
                probs[n][o] - (1.0 if labels[n] == o else 0.0)
                for o in range(_CNN_NUM_CLASSES)
            ]
            for n in range(n_)
        ]
        dx_flat, dlw, dlb = linear.backward(grad_logits)
        dx_pool = flatten.backward(dx_flat)
        dx_conv = pool.backward(dx_pool)
        _, dcw, dcb = conv.backward(dx_conv)

        # 除 N 后同步 SGD 更新（先全部算出新值再提交）。
        inv_n = 1.0 / n_
        new_conv_w = [
            [
                [
                    [
                        conv_w[o][c][kh][kw] - lr * dcw[o][c][kh][kw] * inv_n
                        for kw in range(1)
                    ]
                    for kh in range(1)
                ]
                for c in range(1)
            ]
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_conv_b = [
            conv_b[o] - lr * dcb[o] * inv_n
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_lin_w = [
            [
                lin_w[o][i] - lr * dlw[o][i] * inv_n
                for i in range(_CNN_NUM_CLASSES)
            ]
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_lin_b = [
            lin_b[o] - lr * dlb[o] * inv_n
            for o in range(_CNN_NUM_CLASSES)
        ]
        for new_tensor in (new_conv_w, new_conv_b, new_lin_w, new_lin_b):
            flat_vals = []
            _flatten_into(new_tensor, flat_vals)
            for v in flat_vals:
                if not math.isfinite(v):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
        conv_w, conv_b = new_conv_w, new_conv_b
        lin_w, lin_b = new_lin_w, new_lin_b

    # 末次更新后以同一网络前向，最大 logit 预测，并列取小类。
    final_logits = _cnn_forward(conv_w, conv_b, lin_w, lin_b, images)
    correct = 0
    for n in range(n_):
        row = final_logits[n]
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        if pred == labels[n]:
            correct += 1
    accuracy = correct / n_
    if accuracy != 1.0:
        raise ValueError("训练后 accuracy 不为 1.0")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")
    conv_changed = any(
        conv_w[o][0][0][0] != _CNN_CONV_INIT[o][0][0][0]
        for o in range(_CNN_NUM_CLASSES)
    )
    if not conv_changed:
        raise ValueError("训练后卷积权重未发生改变")
    return conv_w, conv_b, lin_w, lin_b, losses, accuracy


def _cmd_fitcnn(output_path):
    """fitcnn 子命令主体；数据/计算失败返回 1。"""
    try:
        conv_w, conv_b, lin_w, lin_b, losses, accuracy = _fitcnn_run()
        artifact = {
            "model": {
                "conv": {"values": conv_w, "bias": conv_b},
                "linear": {"values": lin_w, "bias": lin_b},
            },
            "metrics": {
                "epochs": _CNN_EPOCHS,
                "lr": float(_CNN_LR),
                "loss": losses,
                "accuracy": accuracy,
            },
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行 BatchNorm CNN 训练：python convnet.py fitnorm OUTPUT
# ---------------------------------------------------------------------------

_NORM_EPOCHS = 20
_NORM_LR = 0.1
_NORM_EPS = 1e-5
_NORM_MOMENTUM = 1.0
_NORM_DROPOUT_P = 0.25
_NORM_DROPOUT_SEED = 7
_NORM_GAMMA_INIT = [1.0, 1.0]
_NORM_BETA_INIT = [0.0, 0.0]


def _norm_forward(
    conv_w, conv_b, gamma, beta, running_mean, running_var,
    lin_w, lin_b, images, training,
):
    """Conv2D → BatchNorm2D → Dropout(0.25,7) → MaxPool → Flatten → Linear。

    training=True 时 BN 处于训练态并使用内部随机状态推进的训练态 Dropout；
    training=False 时 BN 用传入的 running_mean/running_var 推理，Dropout 为
    推理态。Dropout 层在调用方创建并持有，其随机状态跨前向延续。
    """
    conv = Conv2D(conv_w, conv_b)
    bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
    bn.running_mean = _deep_copy(running_mean)
    bn.running_var = _deep_copy(running_var)
    bn.train(training)
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
    dropout.train(training)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear = Linear(lin_w, lin_b)
    conv_out = conv.forward(images)
    bn_out = bn.forward(conv_out)
    drop_out = dropout.forward(bn_out)
    pool_out = pool.forward(drop_out)
    flat = flatten.forward(pool_out)
    logits = linear.forward(flat)
    return logits, bn.running_mean, bn.running_var


def _fitnorm_run():
    """执行确定性 BatchNorm CNN 训练。

    返回 (conv_w, conv_b, gamma, beta, running_mean, running_var,
    lin_w, lin_b, losses, accuracy)。Dropout 随机状态跨轮延续；每轮重建
    BN 层（momentum=1，running 统计每轮被当批统计覆盖）。末次参数更新后
    用最终 conv 权重对全批做仅一次训练态 BN 前向来刷新最终统计，随后切换
    BN/Dropout 为推理态做最终预测。
    """
    images, labels = _load_cnn_samples()
    n_ = len(images)

    conv_w = _deep_copy(_CNN_CONV_INIT)
    conv_b = [0.0] * _CNN_NUM_CLASSES
    gamma = _deep_copy(_NORM_GAMMA_INIT)
    beta = _deep_copy(_NORM_BETA_INIT)
    lin_w = _deep_copy(_CNN_LINEAR_INIT)
    lin_b = [0.0] * _CNN_NUM_CLASSES

    # Dropout 层在整个训练期只创建一次，随机状态 s 跨轮延续。
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)

    losses = []
    lr = _NORM_LR
    for _ in range(_NORM_EPOCHS):
        conv = Conv2D(conv_w, conv_b)
        bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
        pool = MaxPool2D(2, 2, 0)
        flatten = Flatten()
        linear = Linear(lin_w, lin_b)

        conv_out = conv.forward(images)
        bn_out = bn.forward(conv_out)
        drop_out = dropout.forward(bn_out)
        pool_out = pool.forward(drop_out)
        flat = flatten.forward(pool_out)
        logits = linear.forward(flat)

        # 交叉熵前向：softmax 先减去每行最大 logit；记录更新前的批均损失。
        probs = []
        loss_sum = 0.0
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > m:
                    m = row[o]
            exps = []
            denom = 0.0
            for o in range(_CNN_NUM_CLASSES):
                e = math.exp(row[o] - m)
                exps.append(e)
                denom += e
            p = [e / denom for e in exps]
            probs.append(p)
            loss_sum += -math.log(p[labels[n]])
        loss = loss_sum / n_
        if not math.isfinite(loss):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
        losses.append(loss)

        # 反向：softmax-交叉熵对 logits 的梯度为 p - onehot，沿链逐层回传。
        grad_logits = [
            [
                probs[n][o] - (1.0 if labels[n] == o else 0.0)
                for o in range(_CNN_NUM_CLASSES)
            ]
            for n in range(n_)
        ]
        dx_flat, dlw, dlb = linear.backward(grad_logits)
        dx_pool = flatten.backward(dx_flat)
        dx_drop = pool.backward(dx_pool)
        dx_bn = dropout.backward(dx_drop)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        _, dcw, dcb = conv.backward(dx_conv)

        # 除 N 后同步 SGD 更新（先全部算出新值再提交）。
        inv_n = 1.0 / n_
        new_conv_w = [
            [
                [
                    [
                        conv_w[o][c][kh][kw] - lr * dcw[o][c][kh][kw] * inv_n
                        for kw in range(1)
                    ]
                    for kh in range(1)
                ]
                for c in range(1)
            ]
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_conv_b = [
            conv_b[o] - lr * dcb[o] * inv_n
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_gamma = [
            gamma[c] - lr * dgamma[c] * inv_n
            for c in range(_CNN_NUM_CLASSES)
        ]
        new_beta = [
            beta[c] - lr * dbeta[c] * inv_n
            for c in range(_CNN_NUM_CLASSES)
        ]
        new_lin_w = [
            [
                lin_w[o][i] - lr * dlw[o][i] * inv_n
                for i in range(_CNN_NUM_CLASSES)
            ]
            for o in range(_CNN_NUM_CLASSES)
        ]
        new_lin_b = [
            lin_b[o] - lr * dlb[o] * inv_n
            for o in range(_CNN_NUM_CLASSES)
        ]
        for new_tensor in (
            new_conv_w, new_conv_b, new_gamma, new_beta,
            new_lin_w, new_lin_b,
        ):
            flat_vals = []
            _flatten_into(new_tensor, flat_vals)
            for v in flat_vals:
                if not math.isfinite(v):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
        conv_w, conv_b = new_conv_w, new_conv_b
        gamma, beta = new_gamma, new_beta
        lin_w, lin_b = new_lin_w, new_lin_b

    # 末次更新后用最终 conv 权重对全批仅做一次训练态 BN 前向，刷新最终
    # 运行统计（momentum=1，running_* 即当批均值/方差）；不经过 Dropout，
    # 也不更新任何参数。
    final_conv = Conv2D(conv_w, conv_b)
    final_bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
    final_conv_out = final_conv.forward(images)
    final_bn.forward(final_conv_out)
    running_mean = final_bn.running_mean
    running_var = final_bn.running_var

    # 切 BN/Dropout 为推理态，以保存的运行统计做最终预测。
    final_bn.train(False)
    dropout.train(False)
    final_pool = MaxPool2D(2, 2, 0)
    final_flatten = Flatten()
    final_linear = Linear(lin_w, lin_b)
    infer_bn_out = final_bn.forward(final_conv_out)
    infer_drop_out = dropout.forward(infer_bn_out)
    infer_pool_out = final_pool.forward(infer_drop_out)
    infer_flat = final_flatten.forward(infer_pool_out)
    final_logits = final_linear.forward(infer_flat)
    correct = 0
    for n in range(n_):
        row = final_logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        if pred == labels[n]:
            correct += 1
    accuracy = correct / n_
    if accuracy != 1.0:
        raise ValueError("训练后 accuracy 不为 1.0")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")
    return (
        conv_w, conv_b, gamma, beta, running_mean, running_var,
        lin_w, lin_b, losses, accuracy,
    )


def _cmd_fitnorm(output_path):
    """fitnorm 子命令主体；数据/计算失败返回 1。"""
    try:
        (
            conv_w, conv_b, gamma, beta, running_mean, running_var,
            lin_w, lin_b, losses, accuracy,
        ) = _fitnorm_run()
        artifact = {
            "model": {
                "conv": {"values": conv_w, "bias": conv_b},
                "batchnorm": {
                    "gamma": gamma,
                    "beta": beta,
                    "running_mean": running_mean,
                    "running_var": running_var,
                },
                "linear": {"values": lin_w, "bias": lin_b},
            },
            "metrics": {
                "epochs": _NORM_EPOCHS,
                "lr": float(_NORM_LR),
                "seed": _NORM_DROPOUT_SEED,
                "loss": losses,
                "accuracy": accuracy,
            },
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行评估：python convnet.py evaluate WEIGHTS OUTPUT
# ---------------------------------------------------------------------------

_EVAL_TOP_KEYS = ["weights", "metrics"]
_EVAL_WEIGHTS_KEYS = ["values", "bias"]
_EVAL_METRICS_KEYS = ["epochs", "lr", "loss", "accuracy"]


def _reject_duplicate_keys(pairs):
    """object_pairs_hook：保序构造 dict，发现重复键即抛 ValueError。"""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("JSON 对象含重复键：%r" % key)
        out[key] = value
    return out


def _check_metrics_float(value, name):
    """metrics 的浮点字段：限有限 float（拒绝 bool 与 int）。"""
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(
            "%s 必须是 float，得到 %s" % (name, type(value).__name__)
        )
    if not math.isfinite(value):
        raise ValueError("%s 含有非有限值（NaN/inf）" % name)


def _load_weights_artifact(weights_path):
    """读取并严格校验 train 产物，返回 (values, bias)。

    顶层及 weights/metrics 两层的键名、键序、类型与长度均须与 train
    写出的公开契约一致；另拒绝重复键，values 须为 [2][4]、bias 须为 [2]，
    数值限有限 int/float 且拒绝 bool。文件不可读、UTF-8/JSON 非法或
    结构/数值不符时抛 OSError/ValueError/TypeError。
    """
    with open(weights_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("权重产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _EVAL_TOP_KEYS:
        raise ValueError("权重产物顶层键必须依次为 weights、metrics")

    weights_obj = doc["weights"]
    if not isinstance(weights_obj, dict) or (
        list(weights_obj.keys()) != _EVAL_WEIGHTS_KEYS
    ):
        raise ValueError("weights 的键必须依次为 values、bias")
    metrics_obj = doc["metrics"]
    if not isinstance(metrics_obj, dict) or (
        list(metrics_obj.keys()) != _EVAL_METRICS_KEYS
    ):
        raise ValueError("metrics 的键必须依次为 epochs、lr、loss、accuracy")

    values = weights_obj["values"]
    bias = weights_obj["bias"]
    if _shape_of(values, 2, "values") != (
        _TRAIN_NUM_CLASSES,
        _TRAIN_NUM_FEATURES,
    ):
        raise ValueError("values 的形状必须为 [2][4]")
    if _shape_of(bias, 1, "bias") != (_TRAIN_NUM_CLASSES,):
        raise ValueError("bias 的形状必须为 [2]")

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    _check_metrics_float(metrics_obj["lr"], "lr")
    _check_metrics_float(metrics_obj["accuracy"], "accuracy")
    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _TRAIN_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符" % (len(loss), _TRAIN_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")
    return values, bias


def _cmd_evaluate(weights_path, output_path):
    """evaluate 子命令主体；权重/数据/计算/写出失败返回 1。"""
    try:
        values, bias = _load_weights_artifact(weights_path)
        images, labels = _load_train_samples()
        n_ = len(images)

        # 按 n→o→i 计算 logit = bias[o] + Σ x[i]*values[o][i]。
        predictions = []
        correct = 0
        for n in range(n_):
            x_row = images[n][0][0]
            logits = []
            for o in range(_TRAIN_NUM_CLASSES):
                acc = bias[o]
                w_row = values[o]
                for i in range(_TRAIN_NUM_FEATURES):
                    acc += x_row[i] * w_row[i]
                if not math.isfinite(acc):
                    raise ValueError("评估计算产生非有限值（NaN/inf）")
                logits.append(acc)
            # 取最大 logit，并列取较小类别。
            pred = 0
            for o in range(1, _TRAIN_NUM_CLASSES):
                if logits[o] > logits[pred]:
                    pred = o
            predictions.append(pred)
            if pred == labels[n]:
                correct += 1

        artifact = {
            "sample_count": n_,
            "predictions": predictions,
            "accuracy": correct / n_,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行 CNN 评估：python convnet.py evalcnn WEIGHTS OUTPUT
# ---------------------------------------------------------------------------

_CNN_EVAL_TOP_KEYS = ["model", "metrics"]
_CNN_MODEL_KEYS = ["conv", "linear"]
_CNN_LAYER_KEYS = ["values", "bias"]
_CNN_METRICS_KEYS = ["epochs", "lr", "loss", "accuracy"]


def _load_cnn_artifact(weights_path):
    """读取并严格校验 fitcnn 产物，返回 (conv_values, conv_bias, lin_values, lin_bias)。

    顶层及 model/conv/linear/metrics 各层的键名、键序、类型与形状均须与
    fitcnn 写出的公开契约一致；拒绝重复键，conv values 须为 [2][1][1][1]、
    conv bias 为 [2]，linear values 为 [2][2]、linear bias 为 [2]，
    数值限有限 int/float 且拒绝 bool。文件不可读、UTF-8/JSON 非法或
    结构/数值不符时抛 OSError/ValueError/TypeError。
    """
    with open(weights_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("CNN 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _CNN_EVAL_TOP_KEYS:
        raise ValueError("CNN 产物顶层键必须依次为 model、metrics")

    model_obj = doc["model"]
    if not isinstance(model_obj, dict) or (
        list(model_obj.keys()) != _CNN_MODEL_KEYS
    ):
        raise ValueError("model 的键必须依次为 conv、linear")
    metrics_obj = doc["metrics"]
    if not isinstance(metrics_obj, dict) or (
        list(metrics_obj.keys()) != _CNN_METRICS_KEYS
    ):
        raise ValueError("metrics 的键必须依次为 epochs、lr、loss、accuracy")

    conv_obj = model_obj["conv"]
    if not isinstance(conv_obj, dict) or (
        list(conv_obj.keys()) != _CNN_LAYER_KEYS
    ):
        raise ValueError("conv 的键必须依次为 values、bias")
    linear_obj = model_obj["linear"]
    if not isinstance(linear_obj, dict) or (
        list(linear_obj.keys()) != _CNN_LAYER_KEYS
    ):
        raise ValueError("linear 的键必须依次为 values、bias")

    conv_values = conv_obj["values"]
    conv_bias = conv_obj["bias"]
    lin_values = linear_obj["values"]
    lin_bias = linear_obj["bias"]
    if _shape_of(conv_values, 4, "conv values") != (2, 1, 1, 1):
        raise ValueError("conv values 的形状必须为 [2][1][1][1]")
    if _shape_of(conv_bias, 1, "conv bias") != (2,):
        raise ValueError("conv bias 的形状必须为 [2]")
    if _shape_of(lin_values, 2, "linear values") != (2, 2):
        raise ValueError("linear values 的形状必须为 [2][2]")
    if _shape_of(lin_bias, 1, "linear bias") != (2,):
        raise ValueError("linear bias 的形状必须为 [2]")

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    _check_metrics_float(metrics_obj["lr"], "lr")
    _check_metrics_float(metrics_obj["accuracy"], "accuracy")
    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _CNN_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符" % (len(loss), _CNN_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")
    return conv_values, conv_bias, lin_values, lin_bias


def _cmd_evalcnn(weights_path, output_path):
    """evalcnn 子命令主体；权重/数据/计算/写出失败返回 1。"""
    try:
        conv_values, conv_bias, lin_values, lin_bias = _load_cnn_artifact(
            weights_path
        )
        images, labels = _load_cnn_samples()
        n_ = len(images)

        logits = _cnn_forward(
            conv_values, conv_bias, lin_values, lin_bias, images
        )
        predictions = []
        correct = 0
        for n in range(n_):
            row = logits[n]
            for o in range(_CNN_NUM_CLASSES):
                if not math.isfinite(row[o]):
                    raise ValueError("评估计算产生非有限值（NaN/inf）")
            # 取最大 logit，并列取较小类别。
            pred = 0
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > row[pred]:
                    pred = o
            predictions.append(pred)
            if pred == labels[n]:
                correct += 1

        artifact = {
            "sample_count": n_,
            "predictions": predictions,
            "accuracy": correct / n_,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行 BatchNorm CNN 评估：python convnet.py evalnorm WEIGHTS OUTPUT
# ---------------------------------------------------------------------------

_NORM_EVAL_TOP_KEYS = ["model", "metrics"]
_NORM_MODEL_KEYS = ["conv", "batchnorm", "linear"]
_NORM_LAYER_KEYS = ["values", "bias"]
_NORM_BN_KEYS = ["gamma", "beta", "running_mean", "running_var"]
_NORM_METRICS_KEYS = ["epochs", "lr", "seed", "loss", "accuracy"]


def _load_norm_artifact(weights_path):
    """读取并严格校验 fitnorm 产物。

    返回 (conv_values, conv_bias, gamma, beta, running_mean, running_var,
    lin_values, lin_bias)。顶层及 model/conv/batchnorm/linear/metrics
    各层的键名、键序、类型与形状均须与 fitnorm 写出的公开契约一致；
    拒绝重复键。conv values 须为 [2][1][1][1]、conv bias 为 [2]；
    batchnorm 的 gamma/beta/running_mean/running_var 均须为长度 2 的
    有限 float 一维 list（拒绝 bool 与 int）；linear values 为 [2][2]、
    linear bias 为 [2]；metrics 的 epochs/seed 须为 int、lr/accuracy
    须为有限 float、loss 须为长度 20 的有限 float list。推理只使用
    保存的权重与 BN 运行统计，metrics 的具体取值不参与推理，故仅校验
    类型/形状/有限性而不校验数值（与 evalcnn 一致）。文件不可读、
    UTF-8/JSON 非法或结构/数值不符时抛 OSError/ValueError/TypeError。
    """
    with open(weights_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("BatchNorm CNN 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _NORM_EVAL_TOP_KEYS:
        raise ValueError("BatchNorm CNN 产物顶层键必须依次为 model、metrics")

    model_obj = doc["model"]
    if not isinstance(model_obj, dict) or (
        list(model_obj.keys()) != _NORM_MODEL_KEYS
    ):
        raise ValueError("model 的键必须依次为 conv、batchnorm、linear")
    metrics_obj = doc["metrics"]
    if not isinstance(metrics_obj, dict) or (
        list(metrics_obj.keys()) != _NORM_METRICS_KEYS
    ):
        raise ValueError(
            "metrics 的键必须依次为 epochs、lr、seed、loss、accuracy"
        )

    conv_obj = model_obj["conv"]
    if not isinstance(conv_obj, dict) or (
        list(conv_obj.keys()) != _NORM_LAYER_KEYS
    ):
        raise ValueError("conv 的键必须依次为 values、bias")
    bn_obj = model_obj["batchnorm"]
    if not isinstance(bn_obj, dict) or (
        list(bn_obj.keys()) != _NORM_BN_KEYS
    ):
        raise ValueError(
            "batchnorm 的键必须依次为 gamma、beta、running_mean、running_var"
        )
    linear_obj = model_obj["linear"]
    if not isinstance(linear_obj, dict) or (
        list(linear_obj.keys()) != _NORM_LAYER_KEYS
    ):
        raise ValueError("linear 的键必须依次为 values、bias")

    conv_values = conv_obj["values"]
    conv_bias = conv_obj["bias"]
    lin_values = linear_obj["values"]
    lin_bias = linear_obj["bias"]
    if _shape_of(conv_values, 4, "conv values") != (2, 1, 1, 1):
        raise ValueError("conv values 的形状必须为 [2][1][1][1]")
    if _shape_of(conv_bias, 1, "conv bias") != (2,):
        raise ValueError("conv bias 的形状必须为 [2]")

    # batchnorm 的四个参数均须为长度 2 的有限 float[2]（拒绝 bool 与 int）。
    bn_vectors = {}
    for bn_key in _NORM_BN_KEYS:
        vector = bn_obj[bn_key]
        if not isinstance(vector, list):
            raise TypeError(
                "%s 必须是 list，得到 %s" % (bn_key, type(vector).__name__)
            )
        if len(vector) != _CNN_NUM_CLASSES:
            raise ValueError("%s 的长度必须为 2" % bn_key)
        for entry in vector:
            _check_metrics_float(entry, bn_key)
        bn_vectors[bn_key] = vector

    if _shape_of(lin_values, 2, "linear values") != (2, 2):
        raise ValueError("linear values 的形状必须为 [2][2]")
    if _shape_of(lin_bias, 1, "linear bias") != (2,):
        raise ValueError("linear bias 的形状必须为 [2]")

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    _check_metrics_float(metrics_obj["lr"], "lr")
    seed = metrics_obj["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int，得到 %s" % type(seed).__name__
        )
    _check_metrics_float(metrics_obj["accuracy"], "accuracy")
    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _NORM_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符" % (len(loss), _NORM_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")
    return (
        conv_values, conv_bias,
        bn_vectors["gamma"], bn_vectors["beta"],
        bn_vectors["running_mean"], bn_vectors["running_var"],
        lin_values, lin_bias,
    )


def _cmd_evalnorm(weights_path, output_path):
    """evalnorm 子命令主体；权重/数据/计算/写出失败返回 1。"""
    try:
        (
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
        ) = _load_norm_artifact(weights_path)
        images, labels = _load_cnn_samples()
        n_ = len(images)

        # 推理态：BN 用保存的运行统计仿射，Dropout 为恒等映射。
        logits, _, _ = _norm_forward(
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
            images, False,
        )
        predictions = []
        correct = 0
        for n in range(n_):
            row = logits[n]
            for o in range(_CNN_NUM_CLASSES):
                if not math.isfinite(row[o]):
                    raise ValueError("评估计算产生非有限值（NaN/inf）")
            # 取最大 logit，并列取较小类别。
            pred = 0
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > row[pred]:
                    pred = o
            predictions.append(pred)
            if pred == labels[n]:
                correct += 1

        artifact = {
            "sample_count": n_,
            "predictions": predictions,
            "accuracy": correct / n_,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行带验证集训练：python convnet.py benchmark OUTPUT
# ---------------------------------------------------------------------------

# 训练沿用 data/tiny.csv；data/tiny-val.csv 必须与前者逐字节相同且绝不混用
# （训练只加载训练集，预测只加载验证集，两者各自独立读取校验）。
_BENCH_TRAIN_PATH = os.path.join("data", "tiny.csv")
_BENCH_VAL_PATH = os.path.join("data", "tiny-val.csv")

# benchmark 产物的严格契约：顶层键依次为 model、metrics；model 沿用 fitnorm
# 逐层键序，conv/linear 的 values/bias 与 batchnorm 四个向量的每个叶值都
# 必须是有限 float（写出前与重载后均拒绝 int/bool）；metrics 依次为
# epochs=20(int)、lr=0.1(float)、seed=7(int)、loss 有限 float[20]、
# predictions int 列表恰为 [0,1]、accuracy=1.0(float)。
_BENCH_TOP_KEYS = ["model", "metrics"]
_BENCH_METRICS_KEYS = [
    "epochs", "lr", "seed", "loss", "predictions", "accuracy",
]
_BENCH_EXPECTED_EPOCHS = 20
_BENCH_EXPECTED_LR = 0.1
_BENCH_EXPECTED_SEED = 7
_BENCH_EXPECTED_PREDICTIONS = [0, 1]
_BENCH_EXPECTED_ACCURACY = 1.0


def _check_float_shape(node, depth, shape, name):
    """校验嵌套 list 的形状恰为 shape，且每个叶值为有限 float。

    与 _shape_of 同套规则（各维非空、矩形），但标量位置只接受 float：
    int 与 bool 一律拒绝（bool 是 int 子类），非有限 float 拒绝。层级不足、
    空维、形状不规则或不符抛 ValueError；容器或标量类型错抛 TypeError。
    """
    if depth == 0:
        if isinstance(node, list):
            raise ValueError("%s 的层级过深：标量位置出现了 list" % name)
        if isinstance(node, bool) or not isinstance(node, float):
            raise TypeError(
                "%s 的叶值必须是 float（拒绝 int/bool），得到 %s"
                % (name, type(node).__name__)
            )
        if not math.isfinite(node):
            raise ValueError("%s 含有非有限值（NaN/inf）" % name)
        return
    if not isinstance(node, list):
        raise TypeError(
            "%s 必须是嵌套 list，得到 %s" % (name, type(node).__name__)
        )
    if len(node) != shape[0]:
        raise ValueError(
            "%s 的形状必须为 %s" % (name, "".join("[%d]" % d for d in shape))
        )
    for child in node:
        _check_float_shape(child, depth - 1, shape[1:], name)


def _parse_benchmark_model(model_obj):
    """严格校验 benchmark/fitnorm 同构 model，返回深拷贝的全新 model dict。

    逐层沿用修复后的键序 model(conv,batchnorm,linear)、conv/linear 的
    (values,bias) 与 batchnorm 的 (gamma,beta,running_mean,running_var)；
    每个叶值均须为有限 float（拒绝 int/bool），形状固定为 conv values
    [2][1][1][1]、conv bias [2]、batchnorm 四向量各 [2]、linear values
    [2][2]、linear bias [2]。容器类型错抛 TypeError；重复/缺失/额外/错序
    键、形状或取值错抛 ValueError；非有限值抛 ValueError。
    """
    if not isinstance(model_obj, dict):
        raise TypeError(
            "model 必须是 JSON 对象，得到 %s" % type(model_obj).__name__
        )
    if list(model_obj.keys()) != _NORM_MODEL_KEYS:
        raise ValueError("model 的键必须依次为 conv、batchnorm、linear")
    conv_obj = model_obj["conv"]
    bn_obj = model_obj["batchnorm"]
    linear_obj = model_obj["linear"]
    if not isinstance(conv_obj, dict):
        raise TypeError(
            "conv 必须是 JSON 对象，得到 %s" % type(conv_obj).__name__
        )
    if not isinstance(bn_obj, dict):
        raise TypeError(
            "batchnorm 必须是 JSON 对象，得到 %s" % type(bn_obj).__name__
        )
    if not isinstance(linear_obj, dict):
        raise TypeError(
            "linear 必须是 JSON 对象，得到 %s" % type(linear_obj).__name__
        )
    if list(conv_obj.keys()) != _NORM_LAYER_KEYS:
        raise ValueError("conv 的键必须依次为 values、bias")
    if list(bn_obj.keys()) != _NORM_BN_KEYS:
        raise ValueError(
            "batchnorm 的键必须依次为 gamma、beta、running_mean、running_var"
        )
    if list(linear_obj.keys()) != _NORM_LAYER_KEYS:
        raise ValueError("linear 的键必须依次为 values、bias")

    conv_values = conv_obj["values"]
    conv_bias = conv_obj["bias"]
    gamma = bn_obj["gamma"]
    beta = bn_obj["beta"]
    running_mean = bn_obj["running_mean"]
    running_var = bn_obj["running_var"]
    lin_values = linear_obj["values"]
    lin_bias = linear_obj["bias"]

    _check_float_shape(conv_values, 4, (2, 1, 1, 1), "conv values")
    _check_float_shape(conv_bias, 1, (2,), "conv bias")
    _check_float_shape(gamma, 1, (2,), "gamma")
    _check_float_shape(beta, 1, (2,), "beta")
    _check_float_shape(running_mean, 1, (2,), "running_mean")
    _check_float_shape(running_var, 1, (2,), "running_var")
    _check_float_shape(lin_values, 2, (2, 2), "linear values")
    _check_float_shape(lin_bias, 1, (2,), "linear bias")

    return {
        "conv": {
            "values": _deep_copy(conv_values),
            "bias": _deep_copy(conv_bias),
        },
        "batchnorm": {
            "gamma": _deep_copy(gamma),
            "beta": _deep_copy(beta),
            "running_mean": _deep_copy(running_mean),
            "running_var": _deep_copy(running_var),
        },
        "linear": {
            "values": _deep_copy(lin_values),
            "bias": _deep_copy(lin_bias),
        },
    }


def _parse_benchmark_metrics(metrics_obj):
    """严格校验 benchmark metrics，返回深拷贝的全新 metrics dict。

    键依次为 epochs、lr、seed、loss、predictions、accuracy；取值依次为
    int 20（拒绝 bool）、float 0.1、int 7（拒绝 bool）、长度 20 的有限
    float list、恰为 [0,1] 的 int 列表（元素拒绝 bool）、float 1.0。
    容器类型错抛 TypeError；错序/缺失/额外键、长度、取值或非有限错抛
    ValueError。
    """
    if not isinstance(metrics_obj, dict):
        raise TypeError(
            "metrics 必须是 JSON 对象，得到 %s"
            % type(metrics_obj).__name__
        )
    if list(metrics_obj.keys()) != _BENCH_METRICS_KEYS:
        raise ValueError(
            "metrics 的键必须依次为 epochs、lr、seed、loss、"
            "predictions、accuracy"
        )

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    if epochs != _BENCH_EXPECTED_EPOCHS:
        raise ValueError("epochs 必须为 %d" % _BENCH_EXPECTED_EPOCHS)

    lr = metrics_obj["lr"]
    _check_metrics_float(lr, "lr")
    if lr != _BENCH_EXPECTED_LR:
        raise ValueError("lr 必须为 %r" % _BENCH_EXPECTED_LR)

    seed = metrics_obj["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int，得到 %s" % type(seed).__name__
        )
    if seed != _BENCH_EXPECTED_SEED:
        raise ValueError("seed 必须为 %d" % _BENCH_EXPECTED_SEED)

    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _BENCH_EXPECTED_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符"
            % (len(loss), _BENCH_EXPECTED_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")

    predictions = metrics_obj["predictions"]
    if not isinstance(predictions, list):
        raise TypeError(
            "predictions 必须是 list，得到 %s"
            % type(predictions).__name__
        )
    # 先逐个校验标量类型（int 且拒绝 bool；注意 True == 1，不能只靠
    # 等值比较），类型错归 TypeError；随后校验取值/长度，归 ValueError。
    for idx, pred in enumerate(predictions):
        if isinstance(pred, bool) or not isinstance(pred, int):
            raise TypeError(
                "predictions[%d] 必须是 int（拒绝 bool），得到 %s"
                % (idx, type(pred).__name__)
            )
    if predictions != _BENCH_EXPECTED_PREDICTIONS:
        raise ValueError("predictions 必须恰为 [0, 1]")

    accuracy = metrics_obj["accuracy"]
    _check_metrics_float(accuracy, "accuracy")
    if accuracy != _BENCH_EXPECTED_ACCURACY:
        raise ValueError("accuracy 必须为 1.0")

    return {
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "loss": _deep_copy(loss),
        "predictions": _deep_copy(predictions),
        "accuracy": accuracy,
    }


def _load_benchmark_doc(path):
    """读取 benchmark 产物字节并解析为去重保序的 JSON 文档。

    path 必须是 str，否则抛 TypeError；文件不可读抛 OSError；内容不是
    合法 UTF-8 抛 UnicodeDecodeError；JSON 语法错或含重复键抛 ValueError。
    """
    if not isinstance(path, str):
        raise TypeError(
            "path 必须是 str，得到 %s" % type(path).__name__
        )
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


class _BenchDataError(Exception):
    """benchmark 训练/验证数据缺失或字节内容不符。"""


def _load_bench_samples(data_path):
    """按 fitnorm 规则严格加载一个 CSV 数据文件，返回 (images, labels, raw)。

    与 _load_train_samples/_load_cnn_samples 同一解析规则：文件须可读且
    字节恰为 data/tiny.csv 的约定内容，每行 5 个 0/1 字段，四特征按行优先
    重排为 [N][1][2][2]，标签为末字段；空字节、空样本、字段数错或含
    非 0/1 字段均抛 _BenchDataError。raw 为读入的原始字节，供两份数据
    逐字节比对。该函数只打开给定路径、不触碰另一份数据文件，从而保证
    训练集与验证集只经各自路径独立加载、互不混用。
    """
    try:
        with open(data_path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise _BenchDataError("无法读取 %s：%s" % (data_path, exc))
    if raw != _TRAIN_EXPECTED_BYTES:
        raise _BenchDataError("%s 的字节内容与预期不符" % data_path)

    images = []
    labels = []
    for line in raw.decode("utf-8").split("\n"):
        if line == "":
            continue
        fields = line.split(",")
        if len(fields) != _CNN_NUM_FEATURES + 1:
            raise _BenchDataError("%s 存在字段数不为 5 的行" % data_path)
        row = []
        for field in fields:
            if field not in ("0", "1"):
                raise _BenchDataError(
                    "%s 含非 0/1 字段：%r" % (data_path, field)
                )
            row.append(int(field))
        images.append([[[row[0], row[1]], [row[2], row[3]]]])
        labels.append(row[_CNN_NUM_FEATURES])
    if not images:
        raise _BenchDataError("%s 不含任何样本" % data_path)
    return images, labels, raw


def _bench_run():
    """执行带验证集的确定性训练并经序列化/重载后推理。

    训练集与验证集分别经独立路径加载（先各自校验，再确认逐字节相同）。
    以 fitnorm 初值、七层配置（Dropout seed=7）调用 train_norm 训练
    20 轮、lr=0.1；校验 20 项更新前批均 loss 末项严格小于首项，且
    conv 权重/偏置、BN gamma/beta、linear 权重/偏置六组参数至少一组
    相对初值改变。随后把模型经修复后的严格契约（逐层键序、固定形状、
    每个叶值均为有限 float，拒绝 int/bool）校验并深拷贝，再按同一
    转储与严格校验序列化为文本并重新加载，用重载得到的权重与 BN 保存
    统计、Dropout 推理态在验证集上预测，逐样本取最大 logit、并列取较小
    类别，accuracy 必须为 1.0 且预测恰为 [0, 1]，任一 logit 非有限即失败。
    返回严格契约下的 model dict、loss 列表、预测与 accuracy。
    """
    x, labels, train_bytes = _load_bench_samples(_BENCH_TRAIN_PATH)
    x_val, val_labels, val_bytes = _load_bench_samples(_BENCH_VAL_PATH)

    # 两份数据须逐字节相同；用各自独立读入的字节比对，杜绝训练/验证数据
    # 混用或漂移（训练只使用 x/labels，验证只使用 x_val/val_labels）。
    if val_bytes != train_bytes:
        raise _BenchDataError("训练集与验证集字节内容不一致")

    layers = _build_norm_layers()
    conv, bn = layers[0], layers[1]

    init_conv_w = _deep_copy(_CNN_CONV_INIT)
    init_conv_b = [0.0] * _CNN_NUM_CLASSES
    init_gamma = _deep_copy(_NORM_GAMMA_INIT)
    init_beta = _deep_copy(_NORM_BETA_INIT)
    init_lin_w = _deep_copy(_CNN_LINEAR_INIT)
    init_lin_b = [0.0] * _CNN_NUM_CLASSES

    losses = train_norm(
        layers, x, labels, epochs=_NORM_EPOCHS, lr=_NORM_LR
    )
    losses = [float(v) for v in losses]
    for v in losses:
        if not math.isfinite(v):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
    if len(losses) != _NORM_EPOCHS:
        raise ValueError("loss 项数与训练轮数不符")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")

    conv_w = conv._weights
    conv_b = conv._bias
    gamma = bn._gamma
    beta = bn._beta
    lin_w = layers[5]._weights
    lin_b = layers[5]._bias
    changed = (
        conv_w != init_conv_w or conv_b != init_conv_b
        or gamma != init_gamma or beta != init_beta
        or lin_w != init_lin_w or lin_b != init_lin_b
    )
    if not changed:
        raise ValueError("训练后六组参数均未改变")

    running_mean = _deep_copy(bn.running_mean)
    running_var = _deep_copy(bn.running_var)

    # 写出前严格校验 model：修复后的逐层键序、形状，且 conv/batchnorm/
    # linear 的每个叶值都是有限 float（拒绝 int/bool）；返回深拷贝作为
    # 唯一参与序列化与返回的模型，与层内状态完全脱钩。
    model = _parse_benchmark_model({
        "conv": {"values": conv_w, "bias": conv_b},
        "batchnorm": {
            "gamma": gamma,
            "beta": beta,
            "running_mean": running_mean,
            "running_var": running_var,
        },
        "linear": {"values": lin_w, "bias": lin_b},
    })

    # 按修复后的同一转储/严格校验路径序列化再重载，确保参与验证集推理的
    # 是重载后的同构模型：逐键序、形状校验且每个叶值重载后仍为有限 float。
    model_text = _dump_compact(model)
    reloaded = _parse_benchmark_model(
        json.loads(model_text, object_pairs_hook=_reject_duplicate_keys)
    )
    r_conv_w = reloaded["conv"]["values"]
    r_conv_b = reloaded["conv"]["bias"]
    r_gamma = reloaded["batchnorm"]["gamma"]
    r_beta = reloaded["batchnorm"]["beta"]
    r_mean = reloaded["batchnorm"]["running_mean"]
    r_var = reloaded["batchnorm"]["running_var"]
    r_lin_w = reloaded["linear"]["values"]
    r_lin_b = reloaded["linear"]["bias"]

    # 仅在验证集上以 BN 保存统计、Dropout 推理态预测（不接触训练集）。
    logits, _, _ = _norm_forward(
        r_conv_w, r_conv_b, r_gamma, r_beta,
        r_mean, r_var, r_lin_w, r_lin_b,
        x_val, False,
    )
    n_ = len(x_val)
    predictions = []
    correct = 0
    for n in range(n_):
        row = logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("验证计算产生非有限值（NaN/inf）")
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        predictions.append(pred)
        if pred == val_labels[n]:
            correct += 1
    accuracy = correct / n_
    if predictions != [0, 1]:
        raise ValueError("验证集预测必须为 [0, 1]")
    if accuracy != 1.0:
        raise ValueError("验证集 accuracy 不为 1.0")

    return model, losses, predictions, accuracy


def _cmd_benchmark(output_path):
    """benchmark 子命令主体；数据/计算/重载/写出失败返回 1 且不改 OUTPUT。"""
    try:
        # OUTPUT 不得与任一数据文件同路径：避免原子写出破坏训练/验证数据。
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(_BENCH_TRAIN_PATH):
            raise ValueError("OUTPUT 与训练集不能是同一路径")
        if out_abs == os.path.abspath(_BENCH_VAL_PATH):
            raise ValueError("OUTPUT 与验证集不能是同一路径")
        model, losses, predictions, accuracy = _bench_run()
        # 写出前对 metrics 做与 load_benchmark 同一的严格校验（键序、类型、
        # 长度、取值），并以其深拷贝作为载荷；model 已在 _bench_run 内
        # 经同一严格契约校验。
        metrics = _parse_benchmark_metrics({
            "epochs": _NORM_EPOCHS,
            "lr": float(_NORM_LR),
            "seed": _NORM_DROPOUT_SEED,
            "loss": losses,
            "predictions": predictions,
            "accuracy": accuracy,
        })
        artifact = {
            "model": model,
            "metrics": metrics,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_BenchDataError, ValueError, TypeError, OSError):
        return 1
    return 0


def load_benchmark(path):
    """读取完整 benchmark 产物，返回键序为 model、metrics 的新 dict。

    顶层键须依次为 model、metrics（重复/缺失/额外/错序一律非法）。model
    严格沿用修复后的逐层键序、形状与全 float 契约：model 的键依次为
    conv、batchnorm、linear；conv/linear 的键依次为 values、bias；
    batchnorm 的键依次为 gamma、beta、running_mean、running_var；
    conv values 形状 [2][1][1][1]、conv bias [2]，batchnorm 四向量各 [2]，
    linear values [2][2]、linear bias [2]，每个叶值都必须是有限 float
    （拒绝 int/bool 与非有限值）。metrics 的键依次为 epochs、lr、seed、
    loss、predictions、accuracy，取值依次须为 int 20、float 0.1、int 7、
    长度 20 的有限 float list、恰为 [0,1] 的 int 列表、float 1.0。

    成功返回的 dict 与全部嵌套 dict/list 均为新建深拷贝，不与解析结果
    共享任何可变结构。path 或 JSON 容器/标量类型错抛 TypeError；文件
    不可读抛 OSError；非法 UTF-8 抛 UnicodeDecodeError；JSON 语法错、
    重复/缺失/额外/错序键、形状、长度、取值或非有限值错抛 ValueError。
    """
    doc = _load_benchmark_doc(path)
    if not isinstance(doc, dict):
        raise TypeError("benchmark 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _BENCH_TOP_KEYS:
        raise ValueError("benchmark 产物顶层键必须依次为 model、metrics")
    model = _parse_benchmark_model(doc["model"])
    metrics = _parse_benchmark_metrics(doc["metrics"])
    return {"model": model, "metrics": metrics}


# ---------------------------------------------------------------------------
# 命令行分批训练：python convnet.py benchmark_batches OUTPUT
# ---------------------------------------------------------------------------

# benchmark_batches 产物的严格契约：顶层键依次为 model、metrics；model 严格
# 沿用 benchmark/fitnorm 逐层键序、形状与全 float 契约；metrics 依次为
# epochs=20(int)、lr=0.1(float)、seed=7(int)、batch_size=1(int)、
# shuffle=true(bool)、loss 有限 float[40]、predictions 恰为 [0,1] 的 int
# 列表、accuracy=1.0(float)。
_BB_TOP_KEYS = ["model", "metrics"]
_BB_METRICS_KEYS = [
    "epochs", "lr", "seed", "batch_size", "shuffle",
    "loss", "predictions", "accuracy",
]
_BB_BATCH_SIZE = 1
_BB_LOSS_COUNT = _NORM_EPOCHS * 2   # 20 轮 × 每轮 2 批（N=2、batch_size=1）


class _BatchMonitorConv2D(Conv2D):
    """在每批前向前记录全集更新前监控 loss 的 Conv2D 子类。

    batch_size=1 时批内 BatchNorm 方差恒为 0，train_norm_step 返回的逐批
    损失不含可比较的收敛信息；故在七层链的第一层（Conv2D）前向入口、即
    每批任何前向/更新发生之前，以当前参数与当前 BN 运行统计重建推理态网络
    （BN 用 running_mean/running_var、Dropout 推理态），在【完整训练集】上
    计算一次批均 softmax 交叉熵作为该批的更新前监控损失。该探测只读取当前
    参数、另建临时层，绝不推进七层自身的任何缓存、BN 统计或 Dropout 随机
    状态；训练本身完全由未经改动的 train_norm_batches 驱动。本类是真正的
    Conv2D 子类，isinstance 校验与逐层 forward/backward 行为均不变。
    """

    def __init__(self, weights, bias, context):
        super().__init__(weights, bias)
        self._monitor_context = context

    def forward(self, x):
        self._monitor_context.record_full_loss()
        return super().forward(x)


class _BatchMonitorContext:
    """持有七层链与完整训练集，供 _BatchMonitorConv2D 逐批记录监控 loss。"""

    def __init__(self, layers, images, labels):
        self._layers = layers
        self._images = images
        self._labels = labels
        self.records = []

    def record_full_loss(self):
        layers = self._layers
        conv, bn = layers[0], layers[1]
        linear = layers[5]
        # 仅读取当前参数与 running 统计，经推理态网络（BN 保存统计、Dropout
        # 关闭）在完整训练集上前向；_norm_forward 内部自建全部临时层，与七
        # 层链状态完全隔离，不留下任何缓存或随机推进。
        logits, _, _ = _norm_forward(
            conv._weights, conv._bias, bn._gamma, bn._beta,
            bn.running_mean, bn.running_var,
            linear._weights, linear._bias, self._images, False,
        )
        total = 0.0
        n_ = len(self._images)
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > m:
                    m = row[o]
            exps = []
            denom = 0.0
            for o in range(_CNN_NUM_CLASSES):
                e = math.exp(row[o] - m)
                exps.append(e)
                denom += e
            total += -math.log(exps[self._labels[n]] / denom)
        self.records.append(total / n_)


def _bench_batches_run():
    """执行分批确定性训练并经序列化/重载后在验证集推理。

    训练集与验证集分别经独立路径加载（先各自校验，再确认逐字节相同）。
    七层链以 fitnorm 初值构建（Conv2D 换为逐批记录全集更新前监控 loss 的
    _BatchMonitorConv2D 子类，Dropout seed=7），随后原样调用
    train_norm_batches(layers, x, labels, batch_size=1, epochs=20, lr=0.1,
    seed=7, shuffle=True)：20 轮、每轮 2 个大小 1 的批，共 40 项监控 loss。
    训练后校验 40 项均有限且末项严格小于首项、六组参数至少一组相对初值
    改变。batch_size=1 训练态 BN 每批方差为 0，running 统计退化为常数，故
    末批更新后以全新 Conv2D 与训练态 BN 对完整训练集仅做一次前向，按
    fitnorm 同法刷新最终 running_mean/running_var（momentum=1 即当批统计，
    不经过 Dropout、不更新参数、不触发监控钩子）。随后 model 经严格契约
    校验并深拷贝，再按同一转储/严格校验序列化为文本并重新加载，用重载得到
    的权重与 BN 保存统计、Dropout 推理态在验证集上预测，逐样本取最大
    logit、并列取较小类别，accuracy 必须为 1.0 且预测恰为 [0, 1]。
    返回严格契约下的 model dict、40 项监控 loss、预测与 accuracy。
    """
    x, labels, train_bytes = _load_bench_samples(_BENCH_TRAIN_PATH)
    x_val, val_labels, val_bytes = _load_bench_samples(_BENCH_VAL_PATH)
    if val_bytes != train_bytes:
        raise _BenchDataError("训练集与验证集字节内容不一致")

    layers = _build_norm_layers()
    conv, bn = layers[0], layers[1]

    init_conv_w = _deep_copy(_CNN_CONV_INIT)
    init_conv_b = [0.0] * _CNN_NUM_CLASSES
    init_gamma = _deep_copy(_NORM_GAMMA_INIT)
    init_beta = _deep_copy(_NORM_BETA_INIT)
    init_lin_w = _deep_copy(_CNN_LINEAR_INIT)
    init_lin_b = [0.0] * _CNN_NUM_CLASSES

    # 第一层换为监控子类：真正的 Conv2D（七层 isinstance 校验不变），仅在
    # 每批前向入口多读一次全集推理态监控 loss，不改变训练计算。
    context = _BatchMonitorContext(layers, x, labels)
    layers[0] = _BatchMonitorConv2D(conv._weights, conv._bias, context)
    conv = layers[0]

    # 沿七层分批训练 API 原样驱动全部更新（参数按题面逐字给出）。
    train_norm_batches(
        layers, x, labels,
        batch_size=_BB_BATCH_SIZE, epochs=_NORM_EPOCHS, lr=_NORM_LR,
        seed=_NORM_DROPOUT_SEED, shuffle=True,
    )

    losses = [float(v) for v in context.records]
    for v in losses:
        if not math.isfinite(v):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
    if len(losses) != _BB_LOSS_COUNT:
        raise ValueError(
            "监控 loss 项数 %d 与预期 %d 不符"
            % (len(losses), _BB_LOSS_COUNT)
        )
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")

    conv_w = conv._weights
    conv_b = conv._bias
    gamma = bn._gamma
    beta = bn._beta
    lin_w = layers[5]._weights
    lin_b = layers[5]._bias
    changed = (
        conv_w != init_conv_w or conv_b != init_conv_b
        or gamma != init_gamma or beta != init_beta
        or lin_w != init_lin_w or lin_b != init_lin_b
    )
    if not changed:
        raise ValueError("训练后六组参数均未改变")

    # 末批更新后以全新 Conv2D（非监控子类，不会再记录）与训练态 BN 对完整
    # 训练集仅做一次前向，按 fitnorm 同法刷新最终 running 统计；不经过
    # Dropout、不更新任何参数。batch_size=1 期间 running 方差恒为 0，必须
    # 以全集统计替换，推理才有意义。
    final_conv = Conv2D(conv_w, conv_b)
    final_bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
    final_bn.forward(final_conv.forward(x))
    running_mean = _deep_copy(final_bn.running_mean)
    running_var = _deep_copy(final_bn.running_var)

    # 写出前严格校验 model：逐层键序、形状，且每个叶值都是有限 float（拒绝
    # int/bool）；返回深拷贝作为唯一参与序列化与返回的模型。
    model = _parse_benchmark_model({
        "conv": {"values": conv_w, "bias": conv_b},
        "batchnorm": {
            "gamma": gamma,
            "beta": beta,
            "running_mean": running_mean,
            "running_var": running_var,
        },
        "linear": {"values": lin_w, "bias": lin_b},
    })

    # 按同一转储/严格校验路径序列化再重载，确保参与验证集推理的是重载后的
    # 同构模型：逐键序、形状校验且每个叶值重载后仍为有限 float。
    model_text = _dump_compact(model)
    reloaded = _parse_benchmark_model(
        json.loads(model_text, object_pairs_hook=_reject_duplicate_keys)
    )
    r_conv_w = reloaded["conv"]["values"]
    r_conv_b = reloaded["conv"]["bias"]
    r_gamma = reloaded["batchnorm"]["gamma"]
    r_beta = reloaded["batchnorm"]["beta"]
    r_mean = reloaded["batchnorm"]["running_mean"]
    r_var = reloaded["batchnorm"]["running_var"]
    r_lin_w = reloaded["linear"]["values"]
    r_lin_b = reloaded["linear"]["bias"]

    # 仅在验证集上以 BN 保存统计、Dropout 推理态预测（不接触训练集）。
    logits, _, _ = _norm_forward(
        r_conv_w, r_conv_b, r_gamma, r_beta,
        r_mean, r_var, r_lin_w, r_lin_b,
        x_val, False,
    )
    n_ = len(x_val)
    predictions = []
    correct = 0
    for n in range(n_):
        row = logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("验证计算产生非有限值（NaN/inf）")
        # 并列取较小类别：自小类向大类扫描，仅严格更大才更换。
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        predictions.append(pred)
        if pred == val_labels[n]:
            correct += 1
    accuracy = correct / n_
    if predictions != [0, 1]:
        raise ValueError("验证集预测必须为 [0, 1]")
    if accuracy != 1.0:
        raise ValueError("验证集 accuracy 不为 1.0")

    return model, losses, predictions, accuracy


def _parse_bb_metrics(metrics_obj):
    """严格校验 benchmark_batches metrics，返回深拷贝的全新 metrics dict。

    键依次为 epochs、lr、seed、batch_size、shuffle、loss、predictions、
    accuracy；取值依次为 int 20、float 0.1、int 7、int 1、bool 真、长度
    40 的有限 float list、恰为 [0,1] 的 int 列表（元素拒绝 bool）、
    float 1.0。容器类型错抛 TypeError；错序/缺失/额外键、长度、取值或
    非有限错抛 ValueError。
    """
    if not isinstance(metrics_obj, dict):
        raise TypeError(
            "metrics 必须是 JSON 对象，得到 %s"
            % type(metrics_obj).__name__
        )
    if list(metrics_obj.keys()) != _BB_METRICS_KEYS:
        raise ValueError(
            "metrics 的键必须依次为 epochs、lr、seed、batch_size、"
            "shuffle、loss、predictions、accuracy"
        )

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    if epochs != _BENCH_EXPECTED_EPOCHS:
        raise ValueError("epochs 必须为 %d" % _BENCH_EXPECTED_EPOCHS)

    lr = metrics_obj["lr"]
    _check_metrics_float(lr, "lr")
    if lr != _BENCH_EXPECTED_LR:
        raise ValueError("lr 必须为 %r" % _BENCH_EXPECTED_LR)

    seed = metrics_obj["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int，得到 %s" % type(seed).__name__
        )
    if seed != _BENCH_EXPECTED_SEED:
        raise ValueError("seed 必须为 %d" % _BENCH_EXPECTED_SEED)

    batch_size = metrics_obj["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size 必须是 int，得到 %s" % type(batch_size).__name__
        )
    if batch_size != _BB_BATCH_SIZE:
        raise ValueError("batch_size 必须为 %d" % _BB_BATCH_SIZE)

    shuffle = metrics_obj["shuffle"]
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    if shuffle is not True:
        raise ValueError("shuffle 必须为 true")

    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _BB_LOSS_COUNT:
        raise ValueError(
            "loss 长度 %d 与预期 %d 不符" % (len(loss), _BB_LOSS_COUNT)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")

    predictions = metrics_obj["predictions"]
    if not isinstance(predictions, list):
        raise TypeError(
            "predictions 必须是 list，得到 %s"
            % type(predictions).__name__
        )
    for idx, pred in enumerate(predictions):
        if isinstance(pred, bool) or not isinstance(pred, int):
            raise TypeError(
                "predictions[%d] 必须是 int（拒绝 bool），得到 %s"
                % (idx, type(pred).__name__)
            )
    if predictions != _BENCH_EXPECTED_PREDICTIONS:
        raise ValueError("predictions 必须恰为 [0, 1]")

    accuracy = metrics_obj["accuracy"]
    _check_metrics_float(accuracy, "accuracy")
    if accuracy != _BENCH_EXPECTED_ACCURACY:
        raise ValueError("accuracy 必须为 1.0")

    return {
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "loss": _deep_copy(loss),
        "predictions": _deep_copy(predictions),
        "accuracy": accuracy,
    }


def _load_bb_doc(path):
    """读取 benchmark_batches 产物字节并解析为去重保序的 JSON 文档。

    规则同 _load_benchmark_doc：path 非 str 抛 TypeError；文件不可读抛
    OSError；非法 UTF-8 抛 UnicodeDecodeError；JSON 语法错或含重复键抛
    ValueError。
    """
    if not isinstance(path, str):
        raise TypeError(
            "path 必须是 str，得到 %s" % type(path).__name__
        )
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


def load_benchmark_batches(path):
    """读取完整 benchmark_batches 产物，返回键序为 model、metrics 的新 dict。

    顶层键须依次为 model、metrics（重复/缺失/额外/错序一律非法）。model
    严格沿用 benchmark/fitnorm 的逐层键序、形状与全 float 契约。metrics
    的键依次为 epochs、lr、seed、batch_size、shuffle、loss、predictions、
    accuracy，取值依次须为 int 20、float 0.1、int 7、int 1、bool 真、
    长度 40 的有限 float list、恰为 [0,1] 的 int 列表、float 1.0。

    成功返回的 dict 与全部嵌套 dict/list 均为新建深拷贝。path 或 JSON
    容器/标量类型错抛 TypeError；文件不可读抛 OSError；非法 UTF-8 抛
    UnicodeDecodeError；JSON 语法错、重复/缺失/额外/错序键、形状、长度、
    取值或非有限值错抛 ValueError。
    """
    doc = _load_bb_doc(path)
    if not isinstance(doc, dict):
        raise TypeError("benchmark_batches 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _BB_TOP_KEYS:
        raise ValueError(
            "benchmark_batches 产物顶层键必须依次为 model、metrics"
        )
    model = _parse_benchmark_model(doc["model"])
    metrics = _parse_bb_metrics(doc["metrics"])
    return {"model": model, "metrics": metrics}


def _cmd_benchmark_batches(output_path):
    """benchmark_batches 子命令主体；失败返回 1 且不改 OUTPUT。"""
    try:
        # OUTPUT 不得与任一数据文件同路径：避免原子写出破坏训练/验证数据。
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(_BENCH_TRAIN_PATH):
            raise ValueError("OUTPUT 与训练集不能是同一路径")
        if out_abs == os.path.abspath(_BENCH_VAL_PATH):
            raise ValueError("OUTPUT 与验证集不能是同一路径")
        model, losses, predictions, accuracy = _bench_batches_run()
        # 写出前对 metrics 做与 load_benchmark_batches 同一的严格校验。
        metrics = _parse_bb_metrics({
            "epochs": _NORM_EPOCHS,
            "lr": float(_NORM_LR),
            "seed": _NORM_DROPOUT_SEED,
            "batch_size": _BB_BATCH_SIZE,
            "shuffle": True,
            "loss": losses,
            "predictions": predictions,
            "accuracy": accuracy,
        })
        artifact = {
            "model": model,
            "metrics": metrics,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_BenchDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行双层头带动量训练：python convnet.py benchmarkdeep OUTPUT
# ---------------------------------------------------------------------------

# benchmarkdeep 固定超参数（题面逐项固定，不接受命令行覆盖）。
_BD_BATCH_SIZE = 2
_BD_EPOCHS = 20
_BD_LR = 0.1
_BD_SEED = 7
_BD_SHUFFLE = True
_BD_CLIP = 1.0
_BD_MOMENTUM = 0.9
# 首个 Linear 为 2×2 单位权重、零偏置；末 Linear 沿用 _CNN_LINEAR_INIT、
# 零偏置。
_BD_LINEAR1_INIT = [[1.0, 0.0], [0.0, 1.0]]  # [2][2]
_BD_TOP_KEYS = ["loss", "predictions", "accuracy"]


def _build_deep_bench_layers():
    """以 fitnorm 前五层初值与双层分类头新建训练态九层链。

    Conv2D/BatchNorm2D/Dropout/MaxPool2D/Flatten 沿用 fitnorm 初值
    （Dropout seed=7）；首个 Linear 为 2×2 单位权重、零偏置，后接 ReLU，
    末 Linear 沿用 _CNN_LINEAR_INIT、零偏置，SoftmaxCrossEntropy 收尾。
    """
    conv = Conv2D(_deep_copy(_CNN_CONV_INIT), [0.0] * _CNN_NUM_CLASSES)
    bn = BatchNorm2D(
        _deep_copy(_NORM_GAMMA_INIT),
        _deep_copy(_NORM_BETA_INIT),
        _NORM_EPS,
        _NORM_MOMENTUM,
    )
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear1 = Linear(
        _deep_copy(_BD_LINEAR1_INIT), [0.0] * _CNN_NUM_CLASSES
    )
    relu = ReLU()
    linear2 = Linear(
        _deep_copy(_CNN_LINEAR_INIT), [0.0] * _CNN_NUM_CLASSES
    )
    loss = SoftmaxCrossEntropy()
    return [
        conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss,
    ]


def _deep_forward_infer(
    conv_w, conv_b, gamma, beta, running_mean, running_var,
    lin1_w, lin1_b, lin2_w, lin2_b, images,
):
    """推理态九层前向，返回 logits。

    Conv2D→BatchNorm2D（保存统计）→Dropout（推理恒等）→MaxPool→Flatten
    →Linear→ReLU→Linear；BN 切推理态使用传入 running_mean/running_var
    仿射，Dropout 切推理态原样复制，末 Linear 给出两类 logits。
    """
    conv = Conv2D(conv_w, conv_b)
    bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
    bn.running_mean = _deep_copy(running_mean)
    bn.running_var = _deep_copy(running_var)
    bn.train(False)
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
    dropout.train(False)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear1 = Linear(lin1_w, lin1_b)
    relu = ReLU()
    linear2 = Linear(lin2_w, lin2_b)
    conv_out = conv.forward(images)
    bn_out = bn.forward(conv_out)
    drop_out = dropout.forward(bn_out)
    pool_out = pool.forward(drop_out)
    flat = flatten.forward(pool_out)
    hidden = relu.forward(linear1.forward(flat))
    logits = linear2.forward(hidden)
    return logits


def _benchmarkdeep_run():
    """执行双层头带动量批训练并在验证集以推理态预测。

    训练集与验证集分别经独立路径加载（沿用 benchmark 数据契约：字节须
    恰为 data/tiny.csv 约定内容，两份逐字节相同，训练只用训练集、预测
    只用验证集）。九层链以 _build_deep_bench_layers 构建，原样调用
    train_deep_momentum_batches(batch_size=2, epochs=20, lr=0.1, seed=7,
    shuffle=True, clip=1.0, momentum=0.9)：整批 2 样本 × 20 轮，共 20 项
    更新前批均 loss。要求 20 项均有限且末项严格小于首项；卷积权重及两个
    Linear 权重三组均须相对各自初值改变。末批更新后按 fitnorm 同法以全新
    Conv2D 与训练态 BN 对完整训练集仅做一次前向刷新最终运行统计
    （momentum=1 即当批统计），不经过 Dropout、不更新参数。随后以 BN
    保存统计、Dropout 推理态（九层含 ReLU 双层头）在验证集逐样本预测，
    最大 logit 并列取较小类别，任一 logit 非有限即失败；预测须恰为
    [0, 1]、accuracy 为 1.0。返回 (losses, predictions, accuracy)。
    """
    x, labels, train_bytes = _load_bench_samples(_BENCH_TRAIN_PATH)
    x_val, val_labels, val_bytes = _load_bench_samples(_BENCH_VAL_PATH)
    if val_bytes != train_bytes:
        raise _BenchDataError("训练集与验证集字节内容不一致")

    layers = _build_deep_bench_layers()
    conv, bn = layers[0], layers[1]
    linear1, linear2 = layers[5], layers[7]

    init_conv_w = _deep_copy(_CNN_CONV_INIT)
    init_lin1_w = _deep_copy(_BD_LINEAR1_INIT)
    init_lin2_w = _deep_copy(_CNN_LINEAR_INIT)

    losses, _grad_norms, _velocity, _state = train_deep_momentum_batches(
        layers, x, labels,
        batch_size=_BD_BATCH_SIZE, epochs=_BD_EPOCHS, lr=_BD_LR,
        seed=_BD_SEED, shuffle=_BD_SHUFFLE, clip=_BD_CLIP,
        momentum=_BD_MOMENTUM,
    )
    losses = [float(v) for v in losses]
    for v in losses:
        if not math.isfinite(v):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
    if len(losses) != _BD_EPOCHS:
        raise ValueError("loss 项数与训练轮数不符")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")

    conv_w = conv._weights
    conv_b = conv._bias
    lin1_w = linear1._weights
    lin2_w = linear2._weights
    # 题面阈值：卷积及两个 Linear 的权重三组都必须改变。
    if conv_w == init_conv_w:
        raise ValueError("训练后卷积权重未改变")
    if lin1_w == init_lin1_w:
        raise ValueError("训练后首个 Linear 权重未改变")
    if lin2_w == init_lin2_w:
        raise ValueError("训练后末 Linear 权重未改变")

    # 末批更新后以全新 Conv2D 与训练态 BN 对完整训练集仅做一次前向，按
    # fitnorm 同法刷新最终 running 统计；不经过 Dropout、不更新参数。
    gamma = bn._gamma
    beta = bn._beta
    final_bn = BatchNorm2D(gamma, beta, _NORM_EPS, _NORM_MOMENTUM)
    final_bn.forward(Conv2D(conv_w, conv_b).forward(x))
    running_mean = _deep_copy(final_bn.running_mean)
    running_var = _deep_copy(final_bn.running_var)

    # 仅在验证集上以 BN 保存统计、Dropout 推理态预测（不接触训练集）。
    logits = _deep_forward_infer(
        conv_w, conv_b, gamma, beta, running_mean, running_var,
        lin1_w, linear1._bias, lin2_w, linear2._bias, x_val,
    )
    n_ = len(x_val)
    predictions = []
    correct = 0
    for n in range(n_):
        row = logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("验证计算产生非有限值（NaN/inf）")
        # 并列取较小类别：自小类向大类扫描，仅严格更大才更换。
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        predictions.append(pred)
        if pred == val_labels[n]:
            correct += 1
    accuracy = correct / n_
    if predictions != [0, 1]:
        raise ValueError("验证集预测必须为 [0, 1]")
    if accuracy != 1.0:
        raise ValueError("验证集 accuracy 不为 1.0")
    return losses, predictions, accuracy


def _parse_bd_metrics(metrics_obj):
    """严格校验 benchmarkdeep 产物对象，返回深拷贝的全新 dict。

    键依次为 loss、predictions、accuracy；取值依次为长度 20 的有限
    float list（拒绝 int/bool）、恰为 [0,1] 的 int 列表（元素拒绝
    bool）、float 1.0。容器类型错抛 TypeError；错序/缺失/额外键、长度、
    取值或非有限错抛 ValueError。
    """
    if not isinstance(metrics_obj, dict):
        raise TypeError(
            "产物必须是 JSON 对象，得到 %s"
            % type(metrics_obj).__name__
        )
    if list(metrics_obj.keys()) != _BD_TOP_KEYS:
        raise ValueError("产物键必须依次为 loss、predictions、accuracy")

    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _BD_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符" % (len(loss), _BD_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")

    predictions = metrics_obj["predictions"]
    if not isinstance(predictions, list):
        raise TypeError(
            "predictions 必须是 list，得到 %s"
            % type(predictions).__name__
        )
    # 先逐个校验标量类型（int 且拒绝 bool；注意 True == 1，不能只靠
    # 等值比较），类型错归 TypeError；随后校验取值，归 ValueError。
    for idx, pred in enumerate(predictions):
        if isinstance(pred, bool) or not isinstance(pred, int):
            raise TypeError(
                "predictions[%d] 必须是 int（拒绝 bool），得到 %s"
                % (idx, type(pred).__name__)
            )
    if predictions != _BENCH_EXPECTED_PREDICTIONS:
        raise ValueError("predictions 必须恰为 [0, 1]")

    accuracy = metrics_obj["accuracy"]
    _check_metrics_float(accuracy, "accuracy")
    if accuracy != _BENCH_EXPECTED_ACCURACY:
        raise ValueError("accuracy 必须为 1.0")

    return {
        "loss": _deep_copy(loss),
        "predictions": _deep_copy(predictions),
        "accuracy": accuracy,
    }


def _cmd_benchmarkdeep(output_path):
    """benchmarkdeep 子命令主体；失败返回 1 且不改 OUTPUT。"""
    try:
        # OUTPUT 不得与任一数据文件同路径：避免原子写出破坏训练/验证数据。
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(_BENCH_TRAIN_PATH):
            raise ValueError("OUTPUT 与训练集不能是同一路径")
        if out_abs == os.path.abspath(_BENCH_VAL_PATH):
            raise ValueError("OUTPUT 与验证集不能是同一路径")
        losses, predictions, accuracy = _benchmarkdeep_run()
        # 写出前以严格契约校验产物（键序、类型、长度、取值），并以其深拷贝
        # 作为唯一载荷。
        artifact = _parse_bd_metrics({
            "loss": losses,
            "predictions": predictions,
            "accuracy": accuracy,
        })
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (_BenchDataError, ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行数据驱动训练：python convnet.py fitdata DATA OUTPUT
# ---------------------------------------------------------------------------

_FITDATA_KEYS = ["x", "labels"]


def _parse_fitdata(raw):
    """按 fitdata 契约从 DATA 原始字节解析并严格校验，返回 (x, labels)。

    raw 须为 UTF-8 JSON bytes：非法 UTF-8 抛 UnicodeDecodeError；JSON
    语法错抛 ValueError（json 接受的 NaN/Infinity 常量会在后续有限性/
    类型校验中被拒绝）。结构契约同 _load_fitdata：顶层键仅依次为 x、
    labels，重复、缺失、额外或错序键一律非法；x 须为有限 int/float
    （拒绝 bool）的规则 list[N][1][2][2] 且 N>=2；labels 须为与 x 等长
    的 list，元素为 int 0 或 1（拒绝 bool）。
    """
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("DATA 顶层必须是 JSON 对象")
    if list(doc.keys()) != _FITDATA_KEYS:
        raise ValueError("DATA 键必须依次为 x、labels")

    x = doc["x"]
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    if x_shape[1:] != (1, 2, 2):
        raise ValueError("x 的形状必须为 [N][1][2][2]")
    if x_shape[0] < 2:
        raise ValueError("x 的样本数 N 必须 >= 2")

    labels = doc["labels"]
    _require_list(labels, "labels")
    if len(labels) != x_shape[0]:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符"
            % (len(labels), x_shape[0])
        )
    for n, label in enumerate(labels):
        if isinstance(label, list):
            raise ValueError("labels 的层级过深：标量位置出现了 list")
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError(
                "labels 的元素必须是 int（拒绝 bool），得到 %s"
                % type(label).__name__
            )
        if label not in (0, 1):
            raise ValueError("labels 的元素必须为 0 或 1")
    return x, labels


def _load_fitdata(data_path):
    """读取并严格校验 fitdata 的 DATA，返回 (x, labels)。

    文件不可读抛 OSError；字节内容的 UTF-8/JSON 与结构/类型/取值契约见
    _parse_fitdata。返回的列表不与原文共享标量之外的可变结构。
    """
    with open(data_path, "rb") as f:
        raw = f.read()
    return _parse_fitdata(raw)


def _build_norm_layers():
    """以 fitnorm 初值与七层配置新建训练态七层链（Dropout seed=7）。"""
    conv = Conv2D(_deep_copy(_CNN_CONV_INIT), [0.0] * _CNN_NUM_CLASSES)
    bn = BatchNorm2D(
        _deep_copy(_NORM_GAMMA_INIT),
        _deep_copy(_NORM_BETA_INIT),
        _NORM_EPS,
        _NORM_MOMENTUM,
    )
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear = Linear(
        _deep_copy(_CNN_LINEAR_INIT), [0.0] * _CNN_NUM_CLASSES
    )
    loss = SoftmaxCrossEntropy()
    return [conv, bn, dropout, pool, flatten, linear, loss]


def _fitdata_run(x, labels):
    """按 fitdata 契约在 DATA 上训练并重建推理态网络。

    以 fitnorm 初值与七层配置（Dropout seed=7）调用
    train_norm(layers, x, labels, epochs=20, lr=0.1)；训练后校验
    20 项更新前批均 loss 末项严格小于首项，且 conv 权重/偏置、
    BN gamma/beta、linear 权重/偏置六组参数至少一组相对初值改变。
    随后以最终参数与 BN 运行统计经 _norm_forward 重建推理态网络
    （BN 用保存统计、Dropout 关闭），逐样本取最大 logit、并列取较小
    类别为预测，accuracy 必须为 1.0，任一预测 logit 含非有限值即失败。
    返回 (conv_w, conv_b, gamma, beta, running_mean, running_var,
    lin_w, lin_b, losses, accuracy)。
    """
    layers = _build_norm_layers()
    conv, bn = layers[0], layers[1]

    init_conv_w = _deep_copy(_CNN_CONV_INIT)
    init_conv_b = [0.0] * _CNN_NUM_CLASSES
    init_gamma = _deep_copy(_NORM_GAMMA_INIT)
    init_beta = _deep_copy(_NORM_BETA_INIT)
    init_lin_w = _deep_copy(_CNN_LINEAR_INIT)
    init_lin_b = [0.0] * _CNN_NUM_CLASSES

    losses = train_norm(
        layers, x, labels, epochs=_NORM_EPOCHS, lr=_NORM_LR
    )
    losses = [float(v) for v in losses]
    for v in losses:
        if not math.isfinite(v):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")

    conv_w = conv._weights
    conv_b = conv._bias
    gamma = bn._gamma
    beta = bn._beta
    lin_w = layers[5]._weights
    lin_b = layers[5]._bias
    changed = (
        conv_w != init_conv_w or conv_b != init_conv_b
        or gamma != init_gamma or beta != init_beta
        or lin_w != init_lin_w or lin_b != init_lin_b
    )
    if not changed:
        raise ValueError("训练后六组参数均未改变")

    # train_norm 末轮已用最终 conv 输出做训练态 BN 前向刷新统计；
    # running_mean/running_var 即本批统计（momentum=1）。
    running_mean = _deep_copy(bn.running_mean)
    running_var = _deep_copy(bn.running_var)

    # 以最终参数与 BN 统计重建推理态网络：Dropout 关闭，逐样本预测。
    logits, _, _ = _norm_forward(
        conv_w, conv_b, gamma, beta,
        running_mean, running_var, lin_w, lin_b,
        x, False,
    )
    n_ = len(x)
    correct = 0
    for n in range(n_):
        row = logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("评估计算产生非有限值（NaN/inf）")
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        if pred == labels[n]:
            correct += 1
    accuracy = correct / n_
    if accuracy != 1.0:
        raise ValueError("训练后 accuracy 不为 1.0")

    return (
        conv_w, conv_b, gamma, beta, running_mean, running_var,
        lin_w, lin_b, losses, accuracy,
    )


def _cmd_fitdata(data_path, output_path):
    """fitdata 子命令主体；数据/训练/写出失败返回 1 且不改 OUTPUT。"""
    try:
        if os.path.abspath(data_path) == os.path.abspath(output_path):
            raise ValueError("DATA 与 OUTPUT 不能是同一路径")
        x, labels = _load_fitdata(data_path)
        (
            conv_w, conv_b, gamma, beta, running_mean, running_var,
            lin_w, lin_b, losses, accuracy,
        ) = _fitdata_run(x, labels)
        artifact = {
            "model": {
                "conv": {"values": conv_w, "bias": conv_b},
                "batchnorm": {
                    "gamma": gamma,
                    "beta": beta,
                    "running_mean": running_mean,
                    "running_var": running_var,
                },
                "linear": {"values": lin_w, "bias": lin_b},
            },
            "metrics": {
                "epochs": _NORM_EPOCHS,
                "lr": float(_NORM_LR),
                "seed": _NORM_DROPOUT_SEED,
                "loss": losses,
                "accuracy": accuracy,
            },
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行数据驱动评估：python convnet.py evaldata WEIGHTS DATA OUTPUT
# ---------------------------------------------------------------------------


def _cmd_evaldata(weights_path, data_path, output_path):
    """evaldata 子命令主体；权重/数据/计算/写出失败返回 1 且不改 OUTPUT。"""
    try:
        if os.path.abspath(output_path) == os.path.abspath(weights_path):
            raise ValueError("OUTPUT 与 WEIGHTS 不能是同一路径")
        if os.path.abspath(output_path) == os.path.abspath(data_path):
            raise ValueError("OUTPUT 与 DATA 不能是同一路径")
        (
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
        ) = _load_norm_artifact(weights_path)
        x, labels = _load_fitdata(data_path)
        n_ = len(x)

        # 推理态：BN 用保存的运行统计仿射，Dropout 为恒等映射。
        logits, _, _ = _norm_forward(
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
            x, False,
        )
        predictions = []
        correct = 0
        for n in range(n_):
            row = logits[n]
            for o in range(_CNN_NUM_CLASSES):
                if not math.isfinite(row[o]):
                    raise ValueError("评估计算产生非有限值（NaN/inf）")
            # 取最大 logit，并列取较小类别。
            pred = 0
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > row[pred]:
                    pred = o
            predictions.append(pred)
            if pred == labels[n]:
                correct += 1

        artifact = {
            "sample_count": n_,
            "predictions": predictions,
            "accuracy": correct / n_,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行数据驱动取 logit 推理：python convnet.py predictdata WEIGHTS DATA OUTPUT
# ---------------------------------------------------------------------------

_PREDICTDATA_KEYS = ["x"]


def _load_predictdata(data_path):
    """读取并严格校验 predictdata 的 DATA，返回 x。

    DATA 须为 UTF-8 JSON 对象，键仅为 x 且顺序固定，重复、缺失、额外或
    错序键一律非法；x 须为有限 int/float（拒绝 bool）的规则
    list[N][1][2][2] 且 N>=1。文件不可读、UTF-8/JSON 非法或结构/类型/
    取值不符时抛 OSError/ValueError/TypeError。
    """
    with open(data_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("DATA 顶层必须是 JSON 对象")
    if list(doc.keys()) != _PREDICTDATA_KEYS:
        raise ValueError("DATA 键必须仅为 x")

    x = doc["x"]
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    if x_shape[1:] != (1, 2, 2):
        raise ValueError("x 的形状必须为 [N][1][2][2]")
    if x_shape[0] < 1:
        raise ValueError("x 的样本数 N 必须 >= 1")
    return x


def _cmd_predictdata(weights_path, data_path, output_path):
    """predictdata 子命令主体；权重/数据/计算/写出失败返回 1 且不改 OUTPUT。"""
    try:
        # OUTPUT 不得与任一输入同路径：避免原子写出覆盖权重或数据。
        if os.path.abspath(output_path) == os.path.abspath(weights_path):
            raise ValueError("OUTPUT 与 WEIGHTS 不能是同一路径")
        if os.path.abspath(output_path) == os.path.abspath(data_path):
            raise ValueError("OUTPUT 与 DATA 不能是同一路径")
        (
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
        ) = _load_norm_artifact(weights_path)
        x = _load_predictdata(data_path)
        n_ = len(x)

        # 推理态：BN 用保存的运行统计仿射，Dropout 为恒等映射。
        logits, _, _ = _norm_forward(
            conv_values, conv_bias, gamma, beta,
            running_mean, running_var, lin_values, lin_bias,
            x, False,
        )
        predictions = []
        out_logits = []
        for n in range(n_):
            row = logits[n]
            for o in range(_CNN_NUM_CLASSES):
                if not math.isfinite(row[o]):
                    raise ValueError("推理计算产生非有限值（NaN/inf）")
            # 取最大 logit，并列取较小类别。
            pred = 0
            for o in range(1, _CNN_NUM_CLASSES):
                if row[o] > row[pred]:
                    pred = o
            predictions.append(pred)
            out_logits.append([float(row[0]), float(row[1])])

        artifact = {
            "sample_count": n_,
            "predictions": predictions,
            "logits": out_logits,
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行数据驱动训练+验证：
# python convnet.py benchmark_data TRAIN VAL OUTPUT
# ---------------------------------------------------------------------------


def _benchmarkdata_run(train_path, val_path):
    """TRAIN/VAL 独立加载后训练并在验证集上推理。

    训练集与验证集分别经独立路径按同一契约（_load_fitdata）各加载一次，
    解析结果互不共享、绝不混用：训练只使用训练集的 x/labels，验证只使用
    验证集的 x_val/val_labels。以 fitnorm 初值、七层配置（Dropout seed=7）
    调用 train_norm(epochs=20, lr=0.1)，记录 20 项更新前批均 loss；校验
    末项严格小于首项、conv 权重/偏置、BN gamma/beta、linear 权重/偏置六组
    参数至少一组相对初值改变。train_norm 末轮已用训练集 Conv 输出做训练态
    BN 前向刷新统计（momentum=1，running_* 即当批统计）；随后以该统计、
    关闭 Dropout 在 VAL 上逐样本推理，取最大 logit、并列取较小类别，
    accuracy 必须为 1.0，任一 logit 非有限即失败。model 在返回前经
    _parse_benchmark_model 严格校验（逐层键序、固定形状、叶值全为有限
    float）并深拷贝，推理也使用该与写出完全同构的副本。
    返回 (model, losses, predictions, accuracy)。
    """
    x, labels = _load_fitdata(train_path)
    x_val, val_labels = _load_fitdata(val_path)

    layers = _build_norm_layers()
    conv, bn = layers[0], layers[1]

    init_conv_w = _deep_copy(_CNN_CONV_INIT)
    init_conv_b = [0.0] * _CNN_NUM_CLASSES
    init_gamma = _deep_copy(_NORM_GAMMA_INIT)
    init_beta = _deep_copy(_NORM_BETA_INIT)
    init_lin_w = _deep_copy(_CNN_LINEAR_INIT)
    init_lin_b = [0.0] * _CNN_NUM_CLASSES

    losses = train_norm(
        layers, x, labels, epochs=_NORM_EPOCHS, lr=_NORM_LR
    )
    losses = [float(v) for v in losses]
    for v in losses:
        if not math.isfinite(v):
            raise ValueError("训练计算产生非有限值（NaN/inf）")
    if len(losses) != _NORM_EPOCHS:
        raise ValueError("loss 项数与训练轮数不符")
    if not losses[-1] < losses[0]:
        raise ValueError("末次 loss 未小于首次 loss")

    conv_w = conv._weights
    conv_b = conv._bias
    gamma = bn._gamma
    beta = bn._beta
    lin_w = layers[5]._weights
    lin_b = layers[5]._bias
    changed = (
        conv_w != init_conv_w or conv_b != init_conv_b
        or gamma != init_gamma or beta != init_beta
        or lin_w != init_lin_w or lin_b != init_lin_b
    )
    if not changed:
        raise ValueError("训练后六组参数均未改变")

    running_mean = _deep_copy(bn.running_mean)
    running_var = _deep_copy(bn.running_var)

    # 写出前严格校验 model：逐层键序、固定形状，叶值全为有限 float
    # （拒绝 int/bool 与非有限值）；返回的深拷贝同时用于验证集推理，
    # 保证推理所用模型与 OUTPUT 中写出的模型逐值一致。
    model = _parse_benchmark_model({
        "conv": {"values": conv_w, "bias": conv_b},
        "batchnorm": {
            "gamma": gamma,
            "beta": beta,
            "running_mean": running_mean,
            "running_var": running_var,
        },
        "linear": {"values": lin_w, "bias": lin_b},
    })

    # 仅在验证集上以 BN 保存统计、Dropout 推理态预测（不接触训练集）。
    logits, _, _ = _norm_forward(
        model["conv"]["values"], model["conv"]["bias"],
        model["batchnorm"]["gamma"], model["batchnorm"]["beta"],
        model["batchnorm"]["running_mean"],
        model["batchnorm"]["running_var"],
        model["linear"]["values"], model["linear"]["bias"],
        x_val, False,
    )
    n_val = len(x_val)
    predictions = []
    correct = 0
    for n in range(n_val):
        row = logits[n]
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(row[o]):
                raise ValueError("验证计算产生非有限值（NaN/inf）")
        # 取最大 logit，并列取较小类别。
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        predictions.append(pred)
        if pred == val_labels[n]:
            correct += 1
    accuracy = correct / n_val
    if accuracy != 1.0:
        raise ValueError("验证集 accuracy 不为 1.0")

    return model, losses, predictions, accuracy


def _cmd_benchmarkdata(train_path, val_path, output_path):
    """benchmark_data 子命令主体；契约/训练/未达标/写出失败返回 1 且不改 OUTPUT。"""
    try:
        # OUTPUT 不得与任一输入同路径：避免原子写出破坏训练/验证数据。
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(train_path):
            raise ValueError("OUTPUT 与 TRAIN 不能是同一路径")
        if out_abs == os.path.abspath(val_path):
            raise ValueError("OUTPUT 与 VAL 不能是同一路径")
        model, losses, predictions, accuracy = _benchmarkdata_run(
            train_path, val_path
        )
        artifact = {
            "model": model,
            "metrics": {
                "epochs": _NORM_EPOCHS,
                "lr": float(_NORM_LR),
                "seed": _NORM_DROPOUT_SEED,
                "loss": losses,
                "predictions": predictions,
                "accuracy": accuracy,
            },
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行梯度检查：python convnet.py gradcheck CONFIG OUTPUT
# ---------------------------------------------------------------------------

_GRADCHECK_CONFIG_KEYS = ["x", "labels", "eps", "atol", "rtol"]


def _load_gradcheck_config(config_path):
    """读取并严格校验 gradcheck 的 CONFIG，返回 (x, labels, eps, atol, rtol)。

    CONFIG 须为 UTF-8 JSON 对象，键仅依次为 x、labels、eps、atol、rtol，
    重复、缺失、额外或错序键一律非法；x 须为有限 int/float（拒绝 bool）
    的规则 list[N][1][2][2] 且 N≥1。labels 与 eps/atol/rtol 的校验交给
    SoftmaxCrossEntropy.forward 与 check_train_gradients 完成。
    文件不可读、UTF-8/JSON 非法或结构/数值不符时抛 OSError/ValueError/TypeError。
    """
    with open(config_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("CONFIG 顶层必须是 JSON 对象")
    if list(doc.keys()) != _GRADCHECK_CONFIG_KEYS:
        raise ValueError(
            "CONFIG 键必须依次为 x、labels、eps、atol、rtol"
        )

    x = doc["x"]
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    if x_shape[1:] != (1, 2, 2):
        raise ValueError("x 的形状必须为 [N][1][2][2]")
    if x_shape[0] < 1:
        raise ValueError("x 的样本数 N 必须 >= 1")

    labels = doc["labels"]
    eps = doc["eps"]
    atol = doc["atol"]
    rtol = doc["rtol"]
    return x, labels, eps, atol, rtol


def _cmd_gradcheck(config_path, output_path):
    """gradcheck 子命令主体；配置/检查/写出失败返回 1，ok 假也返回 1。"""
    try:
        if os.path.abspath(config_path) == os.path.abspath(output_path):
            raise ValueError("CONFIG 与 OUTPUT 不能是同一路径")
        x, labels, eps, atol, rtol = _load_gradcheck_config(config_path)

        # 每次新建与 fitnorm 初始参数、层配置相同的七层训练链。
        conv = Conv2D(_deep_copy(_CNN_CONV_INIT), [0.0] * _CNN_NUM_CLASSES)
        bn = BatchNorm2D(
            _deep_copy(_NORM_GAMMA_INIT),
            _deep_copy(_NORM_BETA_INIT),
            _NORM_EPS,
            _NORM_MOMENTUM,
        )
        dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
        pool = MaxPool2D(2, 2, 0)
        flatten = Flatten()
        linear = Linear(
            _deep_copy(_CNN_LINEAR_INIT), [0.0] * _CNN_NUM_CLASSES
        )
        loss = SoftmaxCrossEntropy()

        ok, max_e, max_r = check_train_gradients(
            conv, bn, dropout, pool, flatten, linear, loss,
            x, labels, eps=eps, atol=atol, rtol=rtol,
        )
        if not (
            isinstance(ok, bool)
            and isinstance(max_e, float)
            and isinstance(max_r, float)
            and math.isfinite(max_e)
            and math.isfinite(max_r)
        ):
            raise ValueError("梯度检查结果非法或含非有限值（NaN/inf）")

        artifact = {"ok": ok, "max_e": max_e, "max_r": max_r}
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 命令行卷积梯度检查：python convnet.py convcheck CONFIG OUTPUT
# ---------------------------------------------------------------------------

_CONVCHECK_CONFIG_KEYS = [
    "weights", "bias", "stride", "padding", "x", "dy", "eps", "atol", "rtol",
]


def _check_convcheck_stride(value):
    """校验 convcheck 的 stride：恰含两项正 int 的 list（拒绝 bool），且至少一项 > 1。

    整体非 list 或成员类型错（含 bool）抛 TypeError；长度不为 2、成员
    非正或两项均不大于 1 抛 ValueError。返回 (SH, SW) tuple。
    """
    if not isinstance(value, list):
        raise TypeError(
            "stride 必须是 list，得到 %s" % type(value).__name__
        )
    if len(value) != 2:
        raise ValueError("stride 必须恰含 (SH, SW) 两个元素")
    sh_, sw_ = value
    for member_name, member in (("SH", sh_), ("SW", sw_)):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "stride 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member <= 0:
            raise ValueError("stride 的 %s 必须为正整数" % member_name)
    if sh_ <= 1 and sw_ <= 1:
        raise ValueError("stride 至少一项必须大于 1")
    return (sh_, sw_)


def _check_convcheck_padding(value):
    """校验 convcheck 的 padding：恰含四项非负 int 的 list（拒绝 bool），且至少一项 > 0。

    整体非 list 或成员类型错（含 bool）抛 TypeError；长度不为 4、成员
    为负或四项均为 0 抛 ValueError。返回 (PT, PB, PL, PR) tuple。
    """
    if not isinstance(value, list):
        raise TypeError(
            "padding 必须是 list，得到 %s" % type(value).__name__
        )
    if len(value) != 4:
        raise ValueError("padding 必须恰含 (PT, PB, PL, PR) 四个元素")
    pt_, pb_, pl_, pr_ = value
    for member_name, member in (
        ("PT", pt_), ("PB", pb_), ("PL", pl_), ("PR", pr_),
    ):
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError(
                "padding 的 %s 必须是 int，得到 %s"
                % (member_name, type(member).__name__)
            )
        if member < 0:
            raise ValueError("padding 的 %s 必须为非负整数" % member_name)
    if pt_ == 0 and pb_ == 0 and pl_ == 0 and pr_ == 0:
        raise ValueError("padding 至少一项必须大于 0")
    return (pt_, pb_, pl_, pr_)


def _load_convcheck_config(config_path):
    """读取并严格校验 convcheck 的 CONFIG。

    返回 (weights, bias, stride, padding, x, dy, eps, atol, rtol)，其中
    stride、padding 已转为 tuple。CONFIG 须为无重复键的 UTF-8 JSON 对象，
    键仅依次为 weights、bias、stride、padding、x、dy、eps、atol、rtol，
    重复、缺失、额外或错序键一律非法；weights、bias、x、dy 与 eps、atol、
    rtol 的校验分别交给 Conv2D 构造、forward/backward 与 check_gradients
    完成。文件不可读、UTF-8/JSON 非法或结构/数值不符时抛
    OSError/ValueError/TypeError。
    """
    with open(config_path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("CONFIG 顶层必须是 JSON 对象")
    if list(doc.keys()) != _CONVCHECK_CONFIG_KEYS:
        raise ValueError(
            "CONFIG 键必须依次为 weights、bias、stride、padding、x、dy、"
            "eps、atol、rtol"
        )
    stride = _check_convcheck_stride(doc["stride"])
    padding = _check_convcheck_padding(doc["padding"])
    return (
        doc["weights"], doc["bias"], stride, padding,
        doc["x"], doc["dy"], doc["eps"], doc["atol"], doc["rtol"],
    )


def _float_tensor(t):
    """把嵌套 list 的叶值逐个转为 float，返回同形新 list；非有限叶抛 ValueError。"""
    if isinstance(t, list):
        return [_float_tensor(v) for v in t]
    try:
        v = float(t)
    except OverflowError:
        raise ValueError("计算结果含非有限值（NaN/inf）")
    if not math.isfinite(v):
        raise ValueError("计算结果含非有限值（NaN/inf）")
    return v


def _cmd_convcheck(config_path, output_path):
    """convcheck 子命令主体；配置/计算/写出失败返回 1，ok 假也返回 1。"""
    try:
        if os.path.abspath(config_path) == os.path.abspath(output_path):
            raise ValueError("CONFIG 与 OUTPUT 不能是同一路径")
        (weights, bias, stride, padding,
         x, dy, eps, atol, rtol) = _load_convcheck_config(config_path)

        # 零补边、dilation=groups=1 的 Conv2D：先取前向与同形 dy 的反向。
        layer = Conv2D(weights, bias, stride=stride, padding=padding)
        forward = layer.forward(x)
        dx, dweights, dbias = layer.backward(dy)

        # 再以同参新层做中心差分梯度检查。
        check_layer = Conv2D(weights, bias, stride=stride, padding=padding)
        ok, max_e, max_r = check_gradients(
            check_layer, x, dy, eps=eps, atol=atol, rtol=rtol
        )
        if not (
            isinstance(ok, bool)
            and isinstance(max_e, float)
            and isinstance(max_r, float)
            and math.isfinite(max_e)
            and math.isfinite(max_r)
        ):
            raise ValueError("梯度检查结果非法或含非有限值（NaN/inf）")

        artifact = {
            "forward": _float_tensor(forward),
            "backward": {
                "dx": _float_tensor(dx),
                "dweights": _float_tensor(dweights),
                "dbias": _float_tensor(dbias),
            },
            "check": {"ok": ok, "max_e": max_e, "max_r": max_r},
        }
        payload = (_dump_compact(artifact) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 公开推理 API：load_model(path)、predict_batch(model, x)
# ---------------------------------------------------------------------------

_MODEL_KEYS = ["values", "bias"]


def load_model(path):
    """读取并严格校验 train 产物，返回 {"values": ..., "bias": ...} 新 dict。

    完整复用 evaluate 对 train 产物的键名/键序、类型、长度、重复键、有限值
    及 values[2][4]、bias[2] 校验；返回的 dict 与两层嵌套 list 均为新建副本。

    path 必须是 str，否则抛 TypeError；文件不可读抛 OSError；内容不是合法
    UTF-8 抛 UnicodeDecodeError；JSON 语法错、重复键、键序/长度/形状/非有限
    值抛 ValueError；顶层对象或字段类型错抛 TypeError。
    """
    if not isinstance(path, str):
        raise TypeError(
            "path 必须是 str，得到 %s" % type(path).__name__
        )
    with open(path, "rb") as f:
        raw = f.read()
    doc = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("权重产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _EVAL_TOP_KEYS:
        raise ValueError("权重产物顶层键必须依次为 weights、metrics")

    weights_obj = doc["weights"]
    if not isinstance(weights_obj, dict):
        raise TypeError(
            "weights 必须是 JSON 对象，得到 %s"
            % type(weights_obj).__name__
        )
    if list(weights_obj.keys()) != _EVAL_WEIGHTS_KEYS:
        raise ValueError("weights 的键必须依次为 values、bias")
    metrics_obj = doc["metrics"]
    if not isinstance(metrics_obj, dict):
        raise TypeError(
            "metrics 必须是 JSON 对象，得到 %s"
            % type(metrics_obj).__name__
        )
    if list(metrics_obj.keys()) != _EVAL_METRICS_KEYS:
        raise ValueError("metrics 的键必须依次为 epochs、lr、loss、accuracy")

    values = weights_obj["values"]
    bias = weights_obj["bias"]
    if not isinstance(values, list):
        raise TypeError(
            "values 必须是嵌套 list，得到 %s" % type(values).__name__
        )
    if not isinstance(bias, list):
        raise TypeError(
            "bias 必须是 list，得到 %s" % type(bias).__name__
        )
    if _shape_of(values, 2, "values") != (
        _TRAIN_NUM_CLASSES,
        _TRAIN_NUM_FEATURES,
    ):
        raise ValueError("values 的形状必须为 [2][4]")
    if _shape_of(bias, 1, "bias") != (_TRAIN_NUM_CLASSES,):
        raise ValueError("bias 的形状必须为 [2]")

    epochs = metrics_obj["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int，得到 %s" % type(epochs).__name__
        )
    _check_metrics_float(metrics_obj["lr"], "lr")
    _check_metrics_float(metrics_obj["accuracy"], "accuracy")
    loss = metrics_obj["loss"]
    if not isinstance(loss, list):
        raise TypeError(
            "loss 必须是 list，得到 %s" % type(loss).__name__
        )
    if len(loss) != _TRAIN_EPOCHS:
        raise ValueError(
            "loss 长度 %d 与训练轮数 %d 不符" % (len(loss), _TRAIN_EPOCHS)
        )
    for entry in loss:
        _check_metrics_float(entry, "loss")

    return {"values": _deep_copy(values), "bias": _deep_copy(bias)}


def predict_batch(model, x):
    """对批量输入做线性分类，返回长度 N 的新 int 列表（类别下标）。

    model 必须严格为 load_model 返回的 dict：键依次为 values、bias，
    values 为 [2][4]、bias 为 [2] 的嵌套 list；x 必须为非空规则
    list[N][1][1][4]。model 与 x 的全部标量须为有限 int/float（拒绝 bool）。

    按 n→o→i 顺序计算 bias[o] + Σ x[n][0][0][i] * values[o][i]，运算产生
    非有限值抛 ValueError；取最大 logit，并列时取较小类别。不修改任何实参。

    容器或标量类型错抛 TypeError；键序、嵌套层级、空维、不规则、形状不符或
    非有限值抛 ValueError。
    """
    if not isinstance(model, dict):
        raise TypeError(
            "model 必须是 dict，得到 %s" % type(model).__name__
        )
    if list(model.keys()) != _MODEL_KEYS:
        raise ValueError("model 的键必须依次为 values、bias")
    values = model["values"]
    bias = model["bias"]
    if not isinstance(values, list):
        raise TypeError(
            "values 必须是嵌套 list，得到 %s" % type(values).__name__
        )
    if not isinstance(bias, list):
        raise TypeError(
            "bias 必须是 list，得到 %s" % type(bias).__name__
        )
    if _shape_of(values, 2, "values") != (
        _TRAIN_NUM_CLASSES,
        _TRAIN_NUM_FEATURES,
    ):
        raise ValueError("values 的形状必须为 [2][4]")
    if _shape_of(bias, 1, "bias") != (_TRAIN_NUM_CLASSES,):
        raise ValueError("bias 的形状必须为 [2]")

    if not isinstance(x, list):
        raise TypeError("x 必须是嵌套 list，得到 %s" % type(x).__name__)
    if _shape_of(x, 4, "x") != (
        len(x),
        1,
        1,
        _TRAIN_NUM_FEATURES,
    ):
        raise ValueError("x 的形状必须为 [N][1][1][4]（N >= 1）")

    n_ = len(x)
    predictions = []
    for n in range(n_):
        x_row = x[n][0][0]
        logits = []
        for o in range(_TRAIN_NUM_CLASSES):
            acc = bias[o]
            w_row = values[o]
            for i in range(_TRAIN_NUM_FEATURES):
                acc += x_row[i] * w_row[i]
            if not math.isfinite(acc):
                raise ValueError("预测计算产生非有限值（NaN/inf）")
            logits.append(acc)
        # 取最大 logit，并列取较小类别。
        pred = 0
        for o in range(1, _TRAIN_NUM_CLASSES):
            if logits[o] > logits[pred]:
                pred = o
        predictions.append(pred)
    return predictions


def predict_norm_batch(model, x):
    """对批量 [N][1][2][2] 输入做标准化七层网络推理，返回新元组
    (predictions, logits)：predictions 为长度 N 的新 int 列表，logits 为
    float[N][2] 的新嵌套 list。

    model 必须严格为 load_benchmark 或 load_benchmark_batches 返回的 model
    dict：键依次为 conv、batchnorm、linear；conv/linear 的键依次为
    values、bias，batchnorm 的键依次为 gamma、beta、running_mean、
    running_var；形状固定为 conv values [2][1][1][1]、conv bias [2]、
    batchnorm 四向量各 [2]、linear values [2][2]、linear bias [2]，每个
    叶值都必须是有限 float（拒绝 int/bool 与非有限值）。x 必须为非空
    规则 list[N][1][2][2]，元素为有限 int/float（拒绝 bool）。

    逐样本按 Conv2D → BatchNorm2D（推理态，使用保存的 running_mean/
    running_var，不更新统计）→ Dropout（推理态恒等）→
    MaxPool2D(2,2,0) → Flatten → Linear 前向；每样本取最大 logit，
    并列取较小类别。任何中间值或最终 logit 非有限抛 ValueError。

    model 或 x 的容器/标量类型错抛 TypeError；键序、层级、空维、不规则、
    形状不符或非有限值抛 ValueError。model 经严格校验并深拷贝后才参与
    计算，x 仅被读取：不修改任一实参，不使用随机数，重复调用结果相同。
    """
    # 与 load_benchmark/load_benchmark_batches 同一严格契约校验，并得到
    # 全新深拷贝；后续层只接触该副本，绝不修改实参 model。
    m = _parse_benchmark_model(model)

    if not isinstance(x, list):
        raise TypeError("x 必须是嵌套 list，得到 %s" % type(x).__name__)
    if _shape_of(x, 4, "x") != (len(x), 1, 2, 2):
        raise ValueError("x 的形状必须为 [N][1][2][2]（N >= 1）")

    conv_w = m["conv"]["values"]
    conv_b = m["conv"]["bias"]
    gamma = m["batchnorm"]["gamma"]
    beta = m["batchnorm"]["beta"]
    running_mean = m["batchnorm"]["running_mean"]
    running_var = m["batchnorm"]["running_var"]
    lin_w = m["linear"]["values"]
    lin_b = m["linear"]["bias"]

    predictions = []
    logits_out = []
    for n in range(len(x)):
        # 逐样本前向：BN 用保存的运行统计推理、Dropout 推理恒等。
        # _norm_forward 内部自建全部临时层并深拷贝 running 统计，不留
        # 缓存、不推进任何随机状态，与实参完全隔离。
        try:
            logits, _, _ = _norm_forward(
                conv_w, conv_b, gamma, beta, running_mean, running_var,
                lin_w, lin_b, [x[n]], False,
            )
        except OverflowError:
            # 极大有限 int 与 float 权值相乘在转 float 时溢出，按非有限
            # 计算统一抛 ValueError。
            raise ValueError("预测计算产生非有限值（NaN/inf）")
        row = logits[0]
        vals = []
        for o in range(_CNN_NUM_CLASSES):
            v = float(row[o])
            if not math.isfinite(v):
                raise ValueError("预测计算产生非有限值（NaN/inf）")
            vals.append(v)
        # 取最大 logit，并列取较小类别。
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if vals[o] > vals[pred]:
                pred = o
        predictions.append(int(pred))
        logits_out.append(vals)
    return predictions, logits_out


# ---------------------------------------------------------------------------
# 公开训练 API：train_norm_step(layers, x, labels, lr=0.1)、
# train_norm(layers, x, labels, epochs=20, lr=0.1) 与
# train_norm_batches(layers, x, labels, batch_size=1, epochs=1, lr=0.1,
#                    seed=0, shuffle=True)
# ---------------------------------------------------------------------------

# 七层每个实例的全部可变状态：缓存（每次成功 forward 覆盖）、模式无关的
# 随机/运行统计，以及被更新的参数 list 本体。
def _snapshot_layers(layers):
    conv, bn, dropout, pool, flatten, linear, loss = layers
    # MaxPool2D 另缓存获胜位置；AdaptiveAvgPool2D 仅有形状缓存。
    if isinstance(pool, MaxPool2D):
        pool_state = {
            "x_shape": pool._x_shape, "out_shape": pool._out_shape,
            "winners": pool._winners,
        }
    else:
        pool_state = {
            "x_shape": pool._x_shape, "out_shape": pool._out_shape,
            "winners": None,
        }
    return {
        "conv": {
            "weights": conv._weights, "bias": conv._bias,
            "x": conv._x, "out_shape": conv._out_shape,
        },
        "bn": {
            "gamma": bn._gamma, "beta": bn._beta,
            "training": bn._training,
            "running_mean": bn.running_mean, "running_var": bn.running_var,
            "z": bn._z, "var": bn._var, "out_shape": bn._out_shape,
        },
        "dropout": {
            "training": dropout._training, "s": dropout._s,
            "mask": dropout._mask, "out_shape": dropout._out_shape,
        },
        "pool": pool_state,
        "flatten": {
            "x_shape": flatten._x_shape, "out_shape": flatten._out_shape,
        },
        "linear": {
            "weights": linear._weights, "bias": linear._bias,
            "x": linear._x, "out_shape": linear._out_shape,
        },
        "loss": {
            "probs": loss._probs, "labels": loss._labels,
            "weights": loss._weights, "den": loss._den,
            "out_shape": loss._out_shape,
        },
    }


def _restore_layers(layers, snap):
    conv, bn, dropout, pool, flatten, linear, loss = layers
    conv._weights, conv._bias = snap["conv"]["weights"], snap["conv"]["bias"]
    conv._x, conv._out_shape = (
        snap["conv"]["x"], snap["conv"]["out_shape"]
    )
    bn._gamma, bn._beta = snap["bn"]["gamma"], snap["bn"]["beta"]
    bn._training = snap["bn"]["training"]
    bn.running_mean, bn.running_var = (
        snap["bn"]["running_mean"], snap["bn"]["running_var"]
    )
    bn._z, bn._var, bn._out_shape = (
        snap["bn"]["z"], snap["bn"]["var"], snap["bn"]["out_shape"]
    )
    dropout._training = snap["dropout"]["training"]
    dropout._s = snap["dropout"]["s"]
    dropout._mask, dropout._out_shape = (
        snap["dropout"]["mask"], snap["dropout"]["out_shape"]
    )
    pool._x_shape, pool._out_shape = (
        snap["pool"]["x_shape"], snap["pool"]["out_shape"]
    )
    # 仅 MaxPool2D 持有获胜位置；AdaptiveAvgPool2D 无该缓存。
    if isinstance(pool, MaxPool2D):
        pool._winners = snap["pool"]["winners"]
    flatten._x_shape, flatten._out_shape = (
        snap["flatten"]["x_shape"], snap["flatten"]["out_shape"]
    )
    linear._weights, linear._bias = (
        snap["linear"]["weights"], snap["linear"]["bias"]
    )
    linear._x, linear._out_shape = (
        snap["linear"]["x"], snap["linear"]["out_shape"]
    )
    loss._probs, loss._labels, loss._weights, loss._den, loss._out_shape = (
        snap["loss"]["probs"], snap["loss"]["labels"],
        snap["loss"]["weights"], snap["loss"]["den"],
        snap["loss"]["out_shape"],
    )


def _validate_norm_layers(layers):
    """严格七层结构校验（train_norm_step 契约的前四项）。

    layers 必须是恰含 Conv2D/BatchNorm2D/Dropout/(MaxPool2D 或
    AdaptiveAvgPool2D)/Flatten/Linear/SoftmaxCrossEntropy 七层实例
    （类型与顺序均固定，第 4 层（索引 3）接受 MaxPool2D 或
    AdaptiveAvgPool2D）的 list：容器或成员类型错抛 TypeError，长度错抛
    ValueError。另要求 BatchNorm2D 与 Dropout 均处于训练态，否则抛
    ValueError。
    """
    if not isinstance(layers, list):
        raise TypeError(
            "layers 必须是 list，得到 %s" % type(layers).__name__
        )
    if len(layers) != 7:
        raise ValueError(
            "layers 必须恰含 7 层，得到 %d 层" % len(layers)
        )
    expected = (
        Conv2D, BatchNorm2D, Dropout, (MaxPool2D, AdaptiveAvgPool2D),
        Flatten, Linear, SoftmaxCrossEntropy,
    )
    names = (
        "Conv2D", "BatchNorm2D", "Dropout",
        "MaxPool2D 或 AdaptiveAvgPool2D",
        "Flatten", "Linear", "SoftmaxCrossEntropy",
    )
    for idx, (layer, cls, name) in enumerate(
        zip(layers, expected, names)
    ):
        if not isinstance(layer, cls):
            raise TypeError(
                "layers[%d] 必须是 %s 实例，得到 %s"
                % (idx, name, type(layer).__name__)
            )
    if not layers[1]._training:
        raise ValueError("BatchNorm2D 必须处于训练态")
    if not layers[2]._training:
        raise ValueError("Dropout 必须处于训练态")


def _validate_train_norm_args(layers, lr):
    """train_norm_step 与 train_norm 共用的 layers/lr 校验。

    layers 的结构（七项契约的长度/模式/类型）校验见
    _validate_norm_layers；lr 必须是正的有限 int/float（拒绝 bool）：
    类型错抛 TypeError，非有限或非正抛 ValueError。
    """
    _validate_norm_layers(layers)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError(
            "lr 必须是 int/float（拒绝 bool），得到 %s" % type(lr).__name__
        )
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr 必须是正的有限值")


def _require_finite_grads(grad):
    """递归检查梯度树的全部标量，任一非有限值抛 ValueError。"""
    flat_vals = []
    _flatten_into(grad, flat_vals)
    for g in flat_vals:
        if not math.isfinite(g):
            raise ValueError("训练计算产生非有限值（NaN/inf）")


def train_norm_step(layers, x, labels, lr=0.1):
    """七层网络（Conv2D/BN/Dropout/(MaxPool 或 AdaptiveAvgPool)/Flatten/
    Linear/SoftmaxCE）的一步标准化训练：按列表顺序前向，自损失层 backward() 起逆序反传，
    以新 list 同步把 conv 的 weights/bias、BN 的 gamma/beta、linear 的
    weights/bias 各减去 lr 乘对应梯度，返回 float 批均损失。

    layers 与 lr 的校验见 _validate_train_norm_args。损失层梯度已按
    批均（含 1/N），参数更新时不再除 N。被调层方法的输入校验错误
    （类型/形状/取值）原样传播。损失、梯度或更新后的参数含非有限值抛
    ValueError：损失层及每个反向层返回后，递归检查全部传播梯度、参数
    梯度与最终输入梯度，任一非有限值即抛错。

    成功时替换层内参数，前向产生的正常缓存、BN 运行统计与 Dropout 随机
    推进均保留；任何路径都不修改 x、labels 及构造参数所用的原 list；
    一旦出错（含校验与非有限值），七层全部恢复到入口状态（含随机状态、
    运行统计与旧缓存），如同本次调用从未发生。
    """
    _validate_train_norm_args(layers, lr)

    conv, bn, dropout, pool, flatten, linear, loss = layers
    snapshot = _snapshot_layers(layers)
    try:
        # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
        conv_out = conv.forward(x)
        bn_out = bn.forward(conv_out)
        drop_out = dropout.forward(bn_out)
        pool_out = pool.forward(drop_out)
        flat = flatten.forward(pool_out)
        logits = linear.forward(flat)
        loss_value = loss.forward(logits, labels)
        if not math.isfinite(loss_value):
            raise ValueError("训练计算产生非有限值（NaN/inf）")

        # 自损失层起逆序反传；损失梯度已批均，不再除 N。每层返回后
        # 立即递归检查其全部输出梯度（含参数梯度与最终输入梯度）。
        grad_logits = loss.backward()
        _require_finite_grads(grad_logits)
        dx_flat, dlw, dlb = linear.backward(grad_logits)
        _require_finite_grads(dx_flat)
        _require_finite_grads(dlw)
        _require_finite_grads(dlb)
        dx_pool = flatten.backward(dx_flat)
        _require_finite_grads(dx_pool)
        dx_drop = pool.backward(dx_pool)
        _require_finite_grads(dx_drop)
        dx_bn = dropout.backward(dx_drop)
        _require_finite_grads(dx_bn)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        _require_finite_grads(dx_conv)
        _require_finite_grads(dgamma)
        _require_finite_grads(dbeta)
        dx_input, dcw, dcb = conv.backward(dx_conv)
        _require_finite_grads(dx_input)
        _require_finite_grads(dcw)
        _require_finite_grads(dcb)

        # 全部新参数先在独立新 list 中算出并校验，再同步提交，保证
        # 失败时层内参数与构造参数原 list 均不被改动。
        def _step(param, grad):
            if isinstance(param, list):
                return [_step(v, g) for v, g in zip(param, grad)]
            new_value = param - lr * grad
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_value

        new_conv_w = _step(conv._weights, dcw)
        new_conv_b = _step(conv._bias, dcb)
        new_gamma = _step(bn._gamma, dgamma)
        new_beta = _step(bn._beta, dbeta)
        new_lin_w = _step(linear._weights, dlw)
        new_lin_b = _step(linear._bias, dlb)

        # 同步替换层内参数；其余缓存、BN 统计、Dropout 随机状态保留。
        conv._weights = new_conv_w
        conv._bias = new_conv_b
        bn._gamma = new_gamma
        bn._beta = new_beta
        linear._weights = new_lin_w
        linear._bias = new_lin_b
        return float(loss_value)
    except BaseException:
        _restore_layers(layers, snapshot)
        raise


# ---------------------------------------------------------------------------
# 双层分类头：Conv2D→BN→Dropout→Pool→Flatten→Linear→ReLU→Linear→SoftmaxCE
# ---------------------------------------------------------------------------

_DEEP_LAYER_TYPES = (
    Conv2D, BatchNorm2D, Dropout,
    (MaxPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D),
    Flatten, Linear, ReLU, Linear, SoftmaxCrossEntropy,
)
_DEEP_LAYER_NAMES = (
    "Conv2D", "BatchNorm2D", "Dropout",
    "MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D",
    "Flatten", "Linear", "ReLU", "Linear", "SoftmaxCrossEntropy",
)

# 三个 deep 批训练接口（train_deep_batches/train_deep_momentum_batches/
# train_deep_adam_batches）的第 4 层与一步训练同集：MaxPool2D、
# AdaptiveAvgPool2D 或 AdaptiveMaxPool2D。
_DEEP_BATCH_LAYER_TYPES = (
    Conv2D, BatchNorm2D, Dropout,
    (MaxPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D),
    Flatten, Linear, ReLU, Linear, SoftmaxCrossEntropy,
)
_DEEP_BATCH_LAYER_NAMES = (
    "Conv2D", "BatchNorm2D", "Dropout",
    "MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D",
    "Flatten", "Linear", "ReLU", "Linear", "SoftmaxCrossEntropy",
)

# train_deep_step/check_deep_gradients/train_deep_batches/
# train_deep_accum_batches 四个接口的第 3 层（索引 2）接受 Dropout 或
# Dropout2D（整通道 dropout）；其余 deep 训练接口仍只接受 Dropout。
_DEEP2_LAYER_TYPES = (
    Conv2D, BatchNorm2D, (Dropout, Dropout2D),
    (MaxPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D),
    Flatten, Linear, ReLU, Linear, SoftmaxCrossEntropy,
)
_DEEP2_LAYER_NAMES = (
    "Conv2D", "BatchNorm2D", "Dropout 或 Dropout2D",
    "MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D",
    "Flatten", "Linear", "ReLU", "Linear", "SoftmaxCrossEntropy",
)


def _check_deep_layer_types(
    layers, expected=_DEEP_LAYER_TYPES, names=_DEEP_LAYER_NAMES
):
    """九层结构的容器/长度/类型校验（不含训练态检查）。

    容器或成员类型错抛 TypeError，长度错抛 ValueError；校验次序沿用
    check_train_gradients：先容器与长度，再逐层类型。expected/names
    为逐层期望类型与显示名，默认取一步训练/梯度检查契约（第 4 层接受
    MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D）。
    """
    if not isinstance(layers, list):
        raise TypeError(
            "layers 必须是 list，得到 %s" % type(layers).__name__
        )
    if len(layers) != 9:
        raise ValueError(
            "layers 必须恰含 9 层，得到 %d 层" % len(layers)
        )
    for idx, (layer, cls, name) in enumerate(zip(layers, expected, names)):
        if not isinstance(layer, cls):
            raise TypeError(
                "layers[%d] 必须是 %s 实例，得到 %s"
                % (idx, name, type(layer).__name__)
            )


def _validate_deep_layers(layers):
    """严格九层结构校验（双层分类头契约）。

    layers 必须是恰含 Conv2D/BatchNorm2D/Dropout/(MaxPool2D、
    AdaptiveAvgPool2D 或 AdaptiveMaxPool2D)/Flatten/Linear/ReLU/Linear/
    SoftmaxCrossEntropy 九层实例（类型与顺序均固定）的 list：容器或成员
    类型错抛 TypeError，长度错抛 ValueError。另要求 BatchNorm2D 与
    Dropout 均处于训练态，否则抛 ValueError。
    """
    _check_deep_layer_types(layers)
    if not layers[1]._training:
        raise ValueError("BatchNorm2D 必须处于训练态")
    if not layers[2]._training:
        raise ValueError("Dropout 必须处于训练态")


def _validate_deep_batch_layers(layers):
    """严格九层结构校验（三个 deep 批训练接口契约）。

    与 _validate_deep_layers 相同：layers 必须是恰含
    Conv2D/BatchNorm2D/Dropout/(MaxPool2D、AdaptiveAvgPool2D 或
    AdaptiveMaxPool2D)/Flatten/Linear/ReLU/Linear/SoftmaxCrossEntropy
    九层实例（类型与顺序均固定）的 list：容器或成员类型错抛 TypeError，
    长度错或 BN/Dropout 非训练态抛 ValueError。
    """
    _check_deep_layer_types(
        layers, _DEEP_BATCH_LAYER_TYPES, _DEEP_BATCH_LAYER_NAMES
    )
    if not layers[1]._training:
        raise ValueError("BatchNorm2D 必须处于训练态")
    if not layers[2]._training:
        raise ValueError("Dropout 必须处于训练态")


def _validate_deep2_layers(layers):
    """严格九层结构校验（索引 2 接受 Dropout2D 的四个 deep 接口契约）。

    与 _validate_deep_layers 相同，但 layers[2] 接受 Dropout 或
    Dropout2D 实例：容器或成员类型错抛 TypeError，长度错或
    BN/Dropout/Dropout2D 非训练态抛 ValueError。
    """
    _check_deep_layer_types(layers, _DEEP2_LAYER_TYPES, _DEEP2_LAYER_NAMES)
    if not layers[1]._training:
        raise ValueError("BatchNorm2D 必须处于训练态")
    if not layers[2]._training:
        raise ValueError(
            "%s 必须处于训练态" % type(layers[2]).__name__
        )


def _snapshot_deep_layers(layers):
    """快照九层的全部实例 __dict__（含参数引用、缓存与随机状态）。"""
    return tuple(dict(layer.__dict__) for layer in layers)


def _restore_deep_layers(layers, saved_state):
    """把九层实例 __dict__ 整体还原到快照（恢复参数引用与全部状态）。"""
    for layer, state in zip(layers, saved_state):
        layer.__dict__.clear()
        layer.__dict__.update(state)


def _check_deep_scalar(value, name):
    """eps/atol/rtol/lr 标量契约：有限 int/float（拒绝 bool）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s 必须是 int/float（拒绝 bool），得到 %s"
            % (name, type(value).__name__)
        )
    if not math.isfinite(value):
        raise ValueError("%s 必须是有限值（拒绝 NaN/inf）" % name)


def train_deep_step(layers, x, labels, lr=0.1):
    """九层网络（Conv2D/BN/(Dropout 或 Dropout2D)/(MaxPool、
    AdaptiveAvgPool 或 AdaptiveMaxPool)/Flatten/Linear/ReLU/Linear/
    SoftmaxCE，双层分类头）
    的一步训练：按列表顺序前向，自损失层 backward() 起逆序反传，以新
    list 同步把 conv 的 weights/bias、BN 的 gamma/beta、两个 Linear 的
    weights/bias 各减去 lr 乘对应梯度，返回更新前 float 批均损失。

    layers 必须是恰含上述九层实例的 list（索引 2 接受 Dropout 或
    Dropout2D，其余类型抛 TypeError；Dropout2D 按 n→c 逐通道推进随机
    状态并广播整通道掩码完成前反向）：容器/成员类型错抛 TypeError，
    长度错或 BN/Dropout/Dropout2D 非训练态抛 ValueError。lr 必须是正的有限
    int/float（拒绝 bool）：类型错抛 TypeError，非有限或非正抛
    ValueError。x 的校验沿各层 forward，labels 的校验沿损失层 forward；
    维度或标签不匹配等被调层错误原样传播（含展平维度与首个 Linear
    输入维度不符，抛 ValueError）。pool 为 AdaptiveMaxPool2D 时反向把
    各分箱梯度累加到 forward 记录的获胜坐标（分箱可重叠，同一输入坐标
    可收到多份梯度）。损失梯度已批均（含 1/N），参数更新时不再除 N；
    损失、任一传播梯度/参数梯度或更新后的参数含非有限值均抛
    ValueError。

    成功时以新 list 同步替换层内参数，前向缓存、BN 运行统计与 Dropout/
    Dropout2D 随机推进均保留；任何路径都不修改 x、labels 及构造参数所用
    的原 list；一旦出错（含校验与非有限值），九层全部恢复到入口状态
    （含参数引用、随机状态、运行统计与旧缓存），如同本次调用从未发生。
    相同入口状态结果完全确定。
    """
    _validate_deep2_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    saved_state = _snapshot_deep_layers(layers)
    try:
        # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
        conv_out = conv.forward(x)
        bn_out = bn.forward(conv_out)
        drop_out = dropout.forward(bn_out)
        pool_out = pool.forward(drop_out)
        flat = flatten.forward(pool_out)
        hidden = linear1.forward(flat)
        relu_out = relu.forward(hidden)
        logits = linear2.forward(relu_out)
        loss_value = loss.forward(logits, labels)
        if not math.isfinite(loss_value):
            raise ValueError("训练计算产生非有限值（NaN/inf）")

        # 自损失层起逆序反传；损失梯度已批均，不再除 N。每层返回后
        # 立即递归检查其全部输出梯度（含参数梯度与最终输入梯度）。
        grad_logits = loss.backward()
        _require_finite_grads(grad_logits)
        dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
        _require_finite_grads(dx_relu)
        _require_finite_grads(dl2w)
        _require_finite_grads(dl2b)
        dx_hidden = relu.backward(dx_relu)
        _require_finite_grads(dx_hidden)
        dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
        _require_finite_grads(dx_flat)
        _require_finite_grads(dl1w)
        _require_finite_grads(dl1b)
        dx_pool = flatten.backward(dx_flat)
        _require_finite_grads(dx_pool)
        dx_drop = pool.backward(dx_pool)
        _require_finite_grads(dx_drop)
        dx_bn = dropout.backward(dx_drop)
        _require_finite_grads(dx_bn)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        _require_finite_grads(dx_conv)
        _require_finite_grads(dgamma)
        _require_finite_grads(dbeta)
        dx_input, dcw, dcb = conv.backward(dx_conv)
        _require_finite_grads(dx_input)
        _require_finite_grads(dcw)
        _require_finite_grads(dcb)

        # 全部新参数先在独立新 list 中算出并校验，再同步提交，保证
        # 失败时层内参数与构造参数原 list 均不被改动。
        def _step(param, grad):
            if isinstance(param, list):
                return [_step(v, g) for v, g in zip(param, grad)]
            new_value = param - lr * grad
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_value

        new_conv_w = _step(conv._weights, dcw)
        new_conv_b = _step(conv._bias, dcb)
        new_gamma = _step(bn._gamma, dgamma)
        new_beta = _step(bn._beta, dbeta)
        new_l1_w = _step(linear1._weights, dl1w)
        new_l1_b = _step(linear1._bias, dl1b)
        new_l2_w = _step(linear2._weights, dl2w)
        new_l2_b = _step(linear2._bias, dl2b)

        # 同步替换层内参数；其余缓存、BN 统计、Dropout 随机状态保留。
        conv._weights = new_conv_w
        conv._bias = new_conv_b
        bn._gamma = new_gamma
        bn._beta = new_beta
        linear1._weights = new_l1_w
        linear1._bias = new_l1_b
        linear2._weights = new_l2_w
        linear2._bias = new_l2_b
        return float(loss_value)
    except BaseException:
        _restore_deep_layers(layers, saved_state)
        raise


def check_deep_gradients(layers, x, labels, eps=1e-6, atol=1e-6, rtol=1e-4):
    """用中心差分数值梯度检验九层双层分类头训练链。

    layers 须为恰含 Conv2D/BatchNorm2D/(Dropout 或 Dropout2D)/(MaxPool2D、
    AdaptiveAvgPool2D 或 AdaptiveMaxPool2D)/Flatten/Linear/ReLU/Linear/
    SoftmaxCrossEntropy 九层实例的 list（索引 2 为其他类型抛
    TypeError）：容器/成员类型错抛 TypeError，
    长度错或 BN/Dropout/Dropout2D 非训练态抛 ValueError。前向按 conv→bn→dropout→
    pool→flatten→linear1→relu→linear2 执行得 logits，标量损失 L =
    loss.forward(logits, labels) 为 float 批均损失；解析梯度自
    loss.backward() 起按 linear2→relu→linear1→flatten→pool→dropout→
    bn→conv 逆序取各层 backward 结果。pool 为 AdaptiveMaxPool2D 时
    反向按 forward 记录的获胜坐标累加，重叠分箱的梯度正确累加。

    数值梯度依次扰动 x、conv 的 weights/bias、bn 的 gamma/beta、第一个
    Linear 的 weights/bias、第二个 Linear 的 weights/bias（各张量内部按
    嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。每次前向（含解析
    梯度前向与每次正、负扰动前向）之前都把 dropout 的随机状态 _s 恢复
    为入口值，使各次前向重放同一掩码（Dropout2D 重放同一 [N][C] 通道
    掩码），故同一入口状态结果确定。
    BatchNorm2D 每次数值前向都重新按当前批次统计。pool 为 MaxPool2D
    时，任一次前向中任一池化有效窗口并列最大（补边位置不参与比较）
    一律抛 ValueError；pool 为 AdaptiveMaxPool2D 时，任一次前向中任一
    自适应分箱并列最大一律抛 ValueError——max 在并列点梯度无定义；
    pool 为 AdaptiveAvgPool2D 时不做并列检测。

    损失、误差、容差判定与返回 (ok, max_e, max_r) 的类型、顺序均沿用
    check_train_gradients：令 e = abs(a - n)、r = e /
    max(abs(a), abs(n), 1e-12)，返回 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    eps/atol/rtol 须为有限 int/float（拒绝 bool）：类型错抛 TypeError；
    eps 非正、容差为负或任一非有限抛 ValueError。x 的校验沿各层
    forward，labels 的校验沿 loss.forward；维度或标签不匹配、计算或
    梯度非有限均抛 ValueError。x、labels、参数及九层实例状态（训练/
    推理模式、缓存、Dropout/Dropout2D 随机状态与掩码、BatchNorm2D 运行
    统计、SoftmaxCrossEntropy 缓存）在所有成功或异常路径均原样恢复
    （含参数引用），重复调用结果一致。
    """
    _check_deep_layer_types(layers, _DEEP2_LAYER_TYPES, _DEEP2_LAYER_NAMES)
    for name, val in (("eps", eps), ("atol", atol), ("rtol", rtol)):
        _check_deep_scalar(val, name)
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if atol < 0:
        raise ValueError("atol 必须为非负数")
    if rtol < 0:
        raise ValueError("rtol 必须为非负数")
    if not layers[1]._training:
        raise ValueError("BatchNorm2D 仅在训练态支持梯度检查")
    if not layers[2]._training:
        raise ValueError(
            "%s 仅在训练态支持梯度检查" % type(layers[2]).__name__
        )

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    saved_state = _snapshot_deep_layers(layers)
    dropout_entry_s = dropout._s
    try:
        def chain_forward(x_arg):
            # 每次前向前恢复入口随机状态，使训练前向重放同一掩码。
            dropout._s = dropout_entry_s
            conv_out = conv.forward(x_arg)
            bn_out = bn.forward(conv_out)
            drop_out = dropout.forward(bn_out)
            if isinstance(pool, MaxPool2D):
                _check_pool_window_ties(pool, drop_out)
            elif isinstance(pool, AdaptiveMaxPool2D):
                _check_adaptive_maxpool_ties(pool, drop_out)
            pool_out = pool.forward(drop_out)
            flat = flatten.forward(pool_out)
            hidden = linear1.forward(flat)
            relu_out = relu.forward(hidden)
            return linear2.forward(relu_out)

        def loss_value(x_arg):
            return loss.forward(chain_forward(x_arg), labels)

        # 前向按 conv→…→linear2→loss；x 的校验沿各层 forward，labels
        # 的校验沿 loss.forward，MaxPool2D/AdaptiveMaxPool2D 并列最大
        # 在此一并检出。
        loss_value(x)

        # 解析梯度自 loss.backward() 起按 linear2→relu→linear1→
        # flatten→pool→dropout→bn→conv 逆序。
        dlogits = loss.backward()
        dx_relu, dl2w, dl2b = linear2.backward(dlogits)
        dx_hidden = relu.backward(dx_relu)
        dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
        dx_pool = flatten.backward(dx_flat)
        dx_drop = pool.backward(dx_pool)
        dx_bn = dropout.backward(dx_drop)
        dx_conv, dgamma, dbeta = bn.backward(dx_bn)
        dx, dcw, dcb = conv.backward(dx_conv)

        # (层对象, 被替换参数的属性名)；x 不替换任何层属性。
        targets = (
            ("x", None, None, x, dx),
            ("conv_weights", conv, "_weights", conv._weights, dcw),
            ("conv_bias", conv, "_bias", conv._bias, dcb),
            ("bn_gamma", bn, "_gamma", bn._gamma, dgamma),
            ("bn_beta", bn, "_beta", bn._beta, dbeta),
            ("linear1_weights", linear1, "_weights", linear1._weights, dl1w),
            ("linear1_bias", linear1, "_bias", linear1._bias, dl1b),
            ("linear2_weights", linear2, "_weights", linear2._weights, dl2w),
            ("linear2_bias", linear2, "_bias", linear2._bias, dl2b),
        )

        ok = True
        max_e = 0.0
        max_r = 0.0
        for name, owner, attr, original, analytic in targets:
            analytic_flat = []
            _flatten_into(analytic, analytic_flat)
            work = _deep_copy(original)  # 只扰动副本，原张量不被修改
            if owner is not None:
                setattr(owner, attr, work)
            idx = 0
            for container, i in _leaf_slots(work):
                v = container[i]
                container[i] = v + eps
                lp = loss_value(work if name == "x" else x)
                container[i] = v - eps
                lm = loss_value(work if name == "x" else x)
                container[i] = v
                if not (math.isfinite(lp) and math.isfinite(lm)):
                    raise ValueError("数值梯度计算产生非有限值（NaN/inf）")
                n = (lp - lm) / (2 * eps)
                a = analytic_flat[idx]
                idx += 1
                if not (math.isfinite(a) and math.isfinite(n)):
                    raise ValueError("梯度计算产生非有限值（NaN/inf）")
                aa = abs(a)
                an = abs(n)
                e = abs(a - n)
                r = e / max(aa, an, 1e-12)
                if e > atol + rtol * max(aa, an):
                    ok = False
                if e > max_e:
                    max_e = e
                if r > max_r:
                    max_r = r
    finally:
        _restore_deep_layers(layers, saved_state)

    return (bool(ok), float(max_e), float(max_r))


def train_norm(layers, x, labels, epochs=20, lr=0.1):
    """七层网络（Conv2D/BN/Dropout/(MaxPool 或 AdaptiveAvgPool)/Flatten/
    Linear/SoftmaxCE）的
    多轮标准化训练：连续执行 epochs 轮 train_norm_step，返回各轮更新前
    批均损失组成的新 list[float]。

    layers、x、labels、lr 的校验以及前反向次序、同步 SGD 更新均沿用
    train_norm_step；损失梯度已批均，不额外除批量。epochs 必须是正
    int（拒绝 bool）：类型错抛 TypeError，非正抛 ValueError。

    末轮更新后，以更新后的 Conv2D 输出仅执行一次训练态 BN 前向，刷新
    running_mean/running_var；不经过后续层，也不更新任何参数。

    任一失败（含校验与非有限值）都把七层的参数引用、模式、缓存、BN
    运行统计、Dropout 随机状态与掩码整体恢复到函数入口状态，且不修改
    x、labels 及构造参数所用的原 list；成功时保留全部参数更新、BN
    统计刷新结果与 Dropout 随机推进，BN 与 Dropout 仍处于训练态。
    相同入口状态结果完全确定。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    _validate_train_norm_args(layers, lr)

    conv, bn = layers[0], layers[1]
    snapshot = _snapshot_layers(layers)
    try:
        losses = []
        for _ in range(epochs):
            losses.append(train_norm_step(layers, x, labels, lr))
        # 末轮更新后，仅用更新后的 Conv2D 输出做一次训练态 BN 前向，
        # 刷新运行统计；不经过后续层，也不更新参数。
        bn.forward(conv.forward(x))
        return losses
    except BaseException:
        _restore_layers(layers, snapshot)
        raise


def train_norm_batches(
    layers, x, labels, batch_size=1, epochs=1, lr=0.1, seed=0, shuffle=True
):
    """七层网络（Conv2D/BN/Dropout/(MaxPool 或 AdaptiveAvgPool)/Flatten/
    Linear/SoftmaxCE）的
    分轮分批标准化训练：每轮按顺序 [0,…,N-1]（shuffle 为真时先做
    Fisher–Yates 洗牌）切分若干批，逐批以批内样本调用 train_norm_step，
    返回按轮、批顺序排列的各批更新前批均损失组成的新 list[float]，长度
    为 epochs*ceil(N/batch_size)。

    layers、lr、x、labels 的校验沿用 train_norm_step（各批仅切取 x、
    labels 的新子 list 传入，不复制样本），另要求 labels 与 x 样本数
    相等，否则抛 ValueError。batch_size、epochs 必须是正 int，seed
    必须是 [0, 2^32-1] 内的 int，三者均拒绝 bool：类型错抛 TypeError，
    范围错抛 ValueError；batch_size 大于 N 时每轮仅一个含全部样本的
    短批。shuffle 必须是 bool，否则抛 TypeError。

    洗牌使用与 Dropout 相同的 32 位线性同余发生器
    s=(1664525*s+1013904223) mod 2^32：每轮自 i=N-1 降至 1，先推进 s
    再令 j=s%(i+1) 并交换 order[i]、order[j]；s 自 seed 起跨轮延续，
    shuffle 为假时整轮不推进 s（seed 仍须合法）。该发生器独立于七层
    自身状态。

    任一失败（含参数校验、x/labels 不匹配与各批 train_norm_step 的
    计算错误）都把七层的参数引用、模式、缓存、BN 运行统计、Dropout
    随机状态与掩码整体恢复到函数入口状态，且不修改 x、labels 及构造
    参数所用的原 list；成功时保留全部参数更新与各步带来的 BN 统计、
    Dropout 随机推进。相同入口状态结果完全确定。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size 必须是 int（拒绝 bool），得到 %s"
            % type(batch_size).__name__
        )
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_train_norm_args(layers, lr)
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    snapshot = _snapshot_layers(layers)
    try:
        losses = []
        s = seed
        for _ in range(epochs):
            # 每轮 order 都从 [0,…,N-1] 重新开始；洗牌状态 s 跨轮延续。
            order = list(range(n_))
            if shuffle and n_ > 1:
                # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                # 再以 j=s%(i+1) 交换；s 跨轮延续。
                for i in range(n_ - 1, 0, -1):
                    s = (1664525 * s + 1013904223) % 4294967296
                    j = s % (i + 1)
                    order[i], order[j] = order[j], order[i]
            for start in range(0, n_, batch_size):
                idx = order[start:start + batch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]
                losses.append(
                    train_norm_step(layers, batch_x, batch_labels, lr)
                )
        return losses
    except BaseException:
        _restore_layers(layers, snapshot)
        raise


def train_deep_batches(
    layers, x, labels, batch_size=1, epochs=1, lr=0.1, seed=0,
    shuffle=True, clip=None, state=None, max_batches=None,
):
    """九层网络（Conv2D/BN/(Dropout 或 Dropout2D)/(MaxPool、
    AdaptiveAvgPool 或 AdaptiveMaxPool)/Flatten/Linear/ReLU/Linear/
    SoftmaxCE，双层分类头；
    第 3 层（索引 2）接受 Dropout 或 Dropout2D，第 4 层接受 MaxPool2D、
    AdaptiveAvgPool2D 或 AdaptiveMaxPool2D，其他类型抛 TypeError）
    的分轮分批训练：每轮按顺序 [0,…,N-1]（shuffle 为真时先做 Fisher–
    Yates 洗牌）切分若干批，逐批按 train_deep_step 的次序前向、自损失层
    起逆序反传，并在同步 SGD 更新前对全部八组参数梯度做可选全局范数裁剪；
    state 与 max_batches 均为 None（默认）时返回 (losses, grad_norms)，
    二者均为按轮、批顺序排列的新 list[float]，长度均为
    epochs*ceil(N/batch_size)，分别记录各批更新前批均损失与
    裁剪前全局梯度范数。

    state 或 max_batches 非 None 时启用批边界暂停/续训，返回
    (losses, grad_norms, state)：前两项仅记录本次调用实际训练的批
    （新 list[float]），state 为新的 (epoch, order, cursor, rng) 四元组，
    描述本次结束（暂停或完成）后的训练进度，不与入参别名。
    max_batches 为 None 时完成剩余全部批；否则必须是非负 int（拒绝
    bool）：类型错抛 TypeError，负值抛 ValueError，0 表示不训练任何批
    （直接返回当前进度状态）。state 为 None 时等价于
    (0, [], 0, seed)，即从第 0 轮起点开始；否则必须是长度恰为 4 的
    tuple，依次为 epoch(int)、order(list[int])、cursor(int)、
    rng(int)，四者均拒绝 bool：容器或成员类型错抛 TypeError，长度、
    范围或相互关系错抛 ValueError。epoch 必须属于 [0, epochs]；
    order 为空时 cursor 必须为 0（轮起点，尚未洗牌）；order 非空时
    必须恰是 0..N-1 的一个全排列，且 cursor 必须属于 [0, N)，为该轮
    下一批的起点（按 batch_size 切分时与各轮各批起点对齐，轮内最后一批
    可短）；rng 必须属于 [0, 2^32-1]。epoch == epochs 时训练已完成、不得
    再训练任何批：max_batches 为正整数即抛 ValueError；max_batches 为 0
    或 None（剩余批为空）时不训练、原样返回完成态。

    续训仅在轮界洗牌：order 为空（轮起点）时，先按既有 32 位 LCG 自
    rng 起生成该轮 order 并把推进后的 rng 记入状态，再从 cursor 切批；
    order 非空（轮中暂停）时沿用既有 order/rng，不重复洗牌。每训练一批
    后 cursor 前进一个批长（最后一批可为短批）；cursor 到达 N 即轮毕：
    epoch 加一、order 清空、cursor 归 0（rng 保持轮界洗牌后的值）。
    入参 state 不会被修改。任意批边界切分后多次调用的两列表拼接、八组
    参数、BN 运行统计、Dropout/Dropout2D 随机状态（含掩码）及终态，
    均与原函数一次训练完成完全相同。

    layers 的九层类型/顺序与 BN/Dropout/Dropout2D 训练态校验、lr 校验
    以及各批 x、
    labels 的校验均沿用 train_deep_step（各批仅切取 x、labels 的新子
    list 传入，不复制样本；pool 为 AdaptiveMaxPool2D 时反向把各分箱
    梯度累加到 forward 记录的首个最大坐标，重叠分箱的梯度累加）；另要求
    labels 与 x 样本数相等，否则抛 ValueError。batch_size、epochs 必须
    是正 int，seed 必须是
    [0, 2^32-1] 内的 int，三者均拒绝 bool：类型错抛 TypeError，范围错抛
    ValueError；batch_size 大于 N 时每轮仅一个含全部样本的短批。shuffle
    必须是 bool，否则抛 TypeError。

    洗牌使用与 Dropout 相同的 32 位线性同余发生器
    s=(1664525*s+1013904223) mod 2^32：每轮自 i=N-1 降至 1，先推进 s
    再令 j=s%(i+1) 并交换 order[i]、order[j]；s 自 seed 起跨轮延续，
    shuffle 为假时整轮不推进 s（seed 仍须合法）。该发生器独立于九层
    自身状态。

    clip 必须为 None 或正的有限 int/float（拒绝 bool）：类型错抛
    TypeError，非有限或非正抛 ValueError。每批反传结束后、参数更新前，
    依次展平 conv 权重/偏置、BN gamma/beta、两个 Linear 权重/偏置共八组
    梯度，以 sqrt(math.fsum(g*g)) 计算裁剪前全局范数；任一梯度或该范数
    非有限抛 ValueError。clip 非 None 且范数大于 clip 时，八组梯度同乘
    clip/norm（以新嵌套 list 承载），随后与未裁剪时一样以新 list 同步
    SGD：各参数减去 lr 乘（裁剪后）梯度，损失梯度已批均（含 1/N），
    更新时不额外除批量。

    任一失败（含参数校验、x/labels 不匹配与各批前反向、非有限值错误）都把
    九层的参数引用、模式、缓存、BN 运行统计、Dropout/Dropout2D 随机状态
    与掩码整体恢复到函数入口状态，且不修改 x、labels 及构造参数所用的原
    list；成功时保留全部参数更新与各批带来的 BN 统计、Dropout/Dropout2D
    随机推进。相同入口状态结果完全确定。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size 必须是 int（拒绝 bool），得到 %s"
            % type(batch_size).__name__
        )
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep2_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    # ---- 批边界暂停/续训参数（state、max_batches）校验 ----
    resume_mode = state is not None or max_batches is not None
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise TypeError(
                "max_batches 必须是 None 或 int（拒绝 bool），得到 %s"
                % type(max_batches).__name__
            )
        if max_batches < 0:
            raise ValueError("max_batches 必须是非负整数")
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None 或 (epoch, order, cursor, rng) 四元组"
                "（tuple），得到 %s" % type(state).__name__
            )
        if len(state) != 4:
            raise ValueError("state 必须恰含 (epoch, order, cursor, rng) 四项")
        cur_epoch, cur_order, cur_cursor, cur_s = state
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for v in cur_order:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(v).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if cur_cursor < 0 or cur_cursor >= n_:
                raise ValueError(
                    "state 的 cursor 必须满足 0 <= cursor < N（%d）" % n_
                )
            # cursor 必须与按 batch_size 切分的各批起点对齐，且对应批
            # 仍有样本（cursor==0 只可能是空 order 的轮起点）。
            if cur_cursor % batch_size != 0:
                raise ValueError(
                    "state 的 cursor %d 不是 batch_size=%d 的批起点"
                    % (cur_cursor, batch_size)
                )
    # epoch == epochs：训练已完成，不得再训练任何批。max_batches 为 0 或
    # None（剩余批为空，自然无事可做）时原样返回完成态；正整数请求训练则
    # 抛 ValueError。
    if (
        cur_epoch == epochs
        and resume_mode
        and max_batches is not None
        and max_batches > 0
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        # max_batches 为 None 时训练剩余全部批。
        remaining = max_batches
        done = False

        def _scaled(tree, factor):
            # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
            if isinstance(tree, list):
                return [_scaled(v, factor) for v in tree]
            return tree * factor

        # 全部新参数先在独立新 list 中算出并校验，再同步提交，保证
        # 失败时层内参数与构造参数原 list 均不被改动。
        def _step(param, grad):
            if isinstance(param, list):
                return [_step(v, g) for v, g in zip(param, grad)]
            new_value = param - lr * grad
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_value

        # max_batches=0：不训练任何批、不在轮界洗牌，直接回传当前进度。
        if max_batches == 0:
            done = True

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            while start < n_:
                if remaining is not None:
                    remaining -= 1
                idx = order[start:start + batch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除 N。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 更新前依次展平八组梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                new_conv_w = _step(conv._weights, dcw)
                new_conv_b = _step(conv._bias, dcb)
                new_gamma = _step(bn._gamma, dgamma)
                new_beta = _step(bn._beta, dbeta)
                new_l1_w = _step(linear1._weights, dl1w)
                new_l1_b = _step(linear1._bias, dl1b)
                new_l2_w = _step(linear2._weights, dl2w)
                new_l2_b = _step(linear2._bias, dl2b)

                # 同步替换层内参数；其余缓存、BN 统计、Dropout 随机状态保留。
                conv._weights = new_conv_w
                conv._bias = new_conv_b
                bn._gamma = new_gamma
                bn._beta = new_beta
                linear1._weights = new_l1_w
                linear1._bias = new_l1_b
                linear2._weights = new_l2_w
                linear2._bias = new_l2_b

                losses.append(float(loss_value))
                grad_norms.append(float(grad_norm))

                # 批后推进游标（idx 长度即本批实际样本数，末批可短）。
                start += len(idx)
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                if remaining == 0:
                    # 恰在批边界暂停：游标已是下一批起点。
                    done = True
                    break
        if resume_mode:
            out_state = (epoch_idx, order, start, s)
            return losses, grad_norms, out_state
        return losses, grad_norms
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


def _check_deep_momentum(value):
    """动量系数契约：[0,1) 内的有限 int/float（拒绝 bool）。

    类型错（含 bool 或非 int/float）抛 TypeError，非有限或越界抛
    ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "momentum 必须是 int/float（拒绝 bool），得到 %s"
            % type(value).__name__
        )
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise ValueError("momentum 必须满足 0 <= momentum < 1")


def _zeros_like_tree(tree):
    """按嵌套 list 结构返回同形的全 0.0 树（新 list）。"""
    return [
        _zeros_like_tree(v) if isinstance(v, list) else 0.0 for v in tree
    ]


def _check_velocity_tree(value, ref, group_name):
    """校验单组速度：与对应参数同形的嵌套 list，叶为有限 int/float。

    容器或叶类型错（含 bool）抛 TypeError，形状错或叶非有限抛
    ValueError。
    """
    if not isinstance(value, list):
        raise TypeError(
            "velocity 的 %s 必须是嵌套 list，得到 %s"
            % (group_name, type(value).__name__)
        )
    if len(value) != len(ref):
        raise ValueError(
            "velocity 的 %s 长度 %d 与参数形状 %d 不符"
            % (group_name, len(value), len(ref))
        )
    for i, (v_child, p_child) in enumerate(zip(value, ref)):
        if isinstance(p_child, list):
            if not isinstance(v_child, list):
                raise TypeError(
                    "velocity 的 %s[%d] 必须是 list，得到 %s"
                    % (group_name, i, type(v_child).__name__)
                )
            _check_velocity_tree(
                v_child, p_child, "%s[%d]" % (group_name, i)
            )
        else:
            if isinstance(v_child, list):
                raise ValueError(
                    "velocity 的 %s[%d] 层级过深：标量位置出现了 list"
                    % (group_name, i)
                )
            if isinstance(v_child, bool) or not isinstance(
                v_child, (int, float)
            ):
                raise TypeError(
                    "velocity 的 %s[%d] 元素必须是 int/float（拒绝 bool）"
                    "，得到 %s"
                    % (group_name, i, type(v_child).__name__)
                )
            if not math.isfinite(v_child):
                raise ValueError(
                    "velocity 的 %s[%d] 含有非有限值（NaN/inf）"
                    % (group_name, i)
                )


def train_deep_momentum_batches(
    layers, x, labels, batch_size=1, epochs=1, lr=0.1, seed=0,
    shuffle=True, clip=None, momentum=0.9, velocity=None, state=None,
    max_batches=None,
):
    """九层网络（Conv2D/BN/Dropout/(MaxPool、AdaptiveAvgPool 或
    AdaptiveMaxPool)/Flatten/Linear/ReLU/Linear/SoftmaxCE，双层分类头；
    第 4 层接受 MaxPool2D、AdaptiveAvgPool2D 或 AdaptiveMaxPool2D，
    其他类型抛 TypeError）
    的分轮分批带动量训练：批次切分、Fisher–Yates 洗牌、前反向次序、
    裁剪前全局梯度范数、可选全局范数裁剪、批边界暂停/续训及失败回滚均
    沿用 train_deep_batches，唯一区别在每批参数更新规则——先对八组梯度
    g 做裁剪，再逐叶以带动量 SGD 更新：

        v = momentum * v + g
        p = p - lr * v

    其中 v 为跨批延续的速度，初值由 velocity 指定。恒返回
    (losses, grad_norms, velocity, state)：losses、grad_norms 为按轮、
    批顺序排列的新 list[float]，未启用暂停时长度均为
    epochs*ceil(N/batch_size)，grad_norms 仍记各批裁剪前的全局梯度范数；
    state 沿用 train_deep_batches 的 (epoch, order, cursor, rng) 进度
    四元组（不与入参别名），全部训练完成时为 (epochs, [], 0, rng)；state
    或 max_batches 非 None 时启用暂停/续训，前两项仅记本次实际训练的
    批，语义与 train_deep_batches 完全相同。velocity 返回项恒为 8 项
    tuple，顺序与八组参数一致：conv weights/bias、BN gamma/beta、第一个
    Linear weights/bias、第二个 Linear weights/bias，每项为与对应参数
    同形的嵌套 list，不与入参 velocity 别名。

    momentum 必须是 [0,1) 内的有限 int/float（拒绝 bool）：类型错抛
    TypeError，非有限或越界抛 ValueError。velocity 必须为 None 或恰含
    8 项的 tuple，依次与上述八组参数同序；每项必须是与对应参数同形的
    嵌套 list，叶值必须是有限 int/float（拒绝 bool）：容器或叶类型错抛
    TypeError，tuple 长度、张量形状不符或叶非有限抛 ValueError；None
    视为八组全零速度。

    其余参数（layers、x、labels、batch_size、epochs、lr、seed、shuffle、
    clip、state、max_batches）的校验、洗牌 LCG、裁剪因子与暂停/续训规则
    全部沿用 train_deep_batches。任一失败（含参数校验、x/labels 不匹配
    与各批前反向、非有限值错误）都把九层的参数引用、模式、缓存、BN 运行
    统计、Dropout 随机状态与掩码整体恢复到函数入口状态，且不修改 x、
    labels、velocity、state 及构造参数所用的原 list；成功时保留全部参数
    更新、速度推进与各批带来的 BN 统计、Dropout 随机推进。相同入口状态
    结果完全确定；任意批边界切分后多次调用的两列表拼接、八组参数、速度、
    BN 运行统计、Dropout 随机状态（含掩码）及终态，均与一次训练完成完全
    相同。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size 必须是 int（拒绝 bool），得到 %s"
            % type(batch_size).__name__
        )
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep_batch_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _check_deep_momentum(momentum)
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    # 八组参数/速度的同序配对：conv 权重/偏置、BN gamma/beta、两个
    # Linear 权重/偏置。
    param_attrs = (
        (conv, "_weights"), (conv, "_bias"),
        (bn, "_gamma"), (bn, "_beta"),
        (linear1, "_weights"), (linear1, "_bias"),
        (linear2, "_weights"), (linear2, "_bias"),
    )
    velocity_names = (
        "conv_weights", "conv_bias",
        "bn_gamma", "bn_beta",
        "linear1_weights", "linear1_bias",
        "linear2_weights", "linear2_bias",
    )

    # velocity 校验：None 视为八组全零；否则须为恰含 8 项的 tuple，逐项
    # 与对应参数同形、叶值有限。校验阶段即复制，绝不修改入参 velocity。
    if velocity is None:
        vel = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
    else:
        if not isinstance(velocity, tuple):
            raise TypeError(
                "velocity 必须是 None 或 8 项 tuple，得到 %s"
                % type(velocity).__name__
            )
        if len(velocity) != 8:
            raise ValueError(
                "velocity 必须恰含 8 项，得到 %d 项" % len(velocity)
            )
        vel = []
        for v_tree, (owner, attr), vname in zip(
            velocity, param_attrs, velocity_names
        ):
            _check_velocity_tree(v_tree, getattr(owner, attr), vname)
            vel.append(_deep_copy(v_tree))

    # ---- 批边界暂停/续训参数（state、max_batches）校验 ----
    resume_mode = state is not None or max_batches is not None
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise TypeError(
                "max_batches 必须是 None 或 int（拒绝 bool），得到 %s"
                % type(max_batches).__name__
            )
        if max_batches < 0:
            raise ValueError("max_batches 必须是非负整数")
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None 或 (epoch, order, cursor, rng) 四元组"
                "（tuple），得到 %s" % type(state).__name__
            )
        if len(state) != 4:
            raise ValueError("state 必须恰含 (epoch, order, cursor, rng) 四项")
        cur_epoch, cur_order, cur_cursor, cur_s = state
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for v in cur_order:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(v).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if cur_cursor < 0 or cur_cursor >= n_:
                raise ValueError(
                    "state 的 cursor 必须满足 0 <= cursor < N（%d）" % n_
                )
            # cursor 必须与按 batch_size 切分的各批起点对齐，且对应批
            # 仍有样本（cursor==0 只可能是空 order 的轮起点）。
            if cur_cursor % batch_size != 0:
                raise ValueError(
                    "state 的 cursor %d 不是 batch_size=%d 的批起点"
                    % (cur_cursor, batch_size)
                )
    # epoch == epochs：训练已完成，不得再训练任何批。max_batches 为 0 或
    # None（剩余批为空，自然无事可做）时原样返回完成态；正整数请求训练则
    # 抛 ValueError。
    if (
        cur_epoch == epochs
        and resume_mode
        and max_batches is not None
        and max_batches > 0
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        # max_batches 为 None 时训练剩余全部批。
        remaining = max_batches
        done = False

        def _scaled(tree, factor):
            # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
            if isinstance(tree, list):
                return [_scaled(v, factor) for v in tree]
            return tree * factor

        def _momentum_step(v_tree, grad, param):
            # 逐叶 v=momentum*v+g、p=p-lr*v；新速度、新参数在独立新 list
            # 中算出并以 (速度树, 参数树) 对返回，再由调用方同步提交。
            if isinstance(param, list):
                v_children, p_children = [], []
                for vv, g, pp in zip(v_tree, grad, param):
                    v_sub, p_sub = _momentum_step(vv, g, pp)
                    v_children.append(v_sub)
                    p_children.append(p_sub)
                return v_children, p_children
            new_v = momentum * v_tree + grad
            if not math.isfinite(new_v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_value = param - lr * new_v
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_v, new_value

        # max_batches=0：不训练任何批、不在轮界洗牌，直接回传当前进度。
        if max_batches == 0:
            done = True

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            while start < n_:
                if remaining is not None:
                    remaining -= 1
                idx = order[start:start + batch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除 N。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 更新前依次展平八组梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                # 先裁剪梯度，再逐叶 v=momentum*v+g、p=p-lr*v；八组新
                # 速度、新参数均先在独立新 list 中算出并校验。裁剪可能
                # 重绑上述变量，故此处重新组装（裁剪后的）八组梯度。
                clipped_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                new_vel = []
                new_params = []
                for v_tree, g_tree, (owner, attr) in zip(
                    vel, clipped_groups, param_attrs
                ):
                    v_new, p_new = _momentum_step(
                        v_tree, g_tree, getattr(owner, attr)
                    )
                    new_vel.append(v_new)
                    new_params.append(p_new)

                # 同步替换八组层内参数与八组速度；其余缓存、BN 统计、
                # Dropout 随机状态保留。
                for (owner, attr), p_new, v_new in zip(
                    param_attrs, new_params, new_vel
                ):
                    setattr(owner, attr, p_new)
                vel = new_vel

                losses.append(float(loss_value))
                grad_norms.append(float(grad_norm))

                # 批后推进游标（idx 长度即本批实际样本数，末批可短）。
                start += len(idx)
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                if remaining == 0:
                    # 恰在批边界暂停：游标已是下一批起点。
                    done = True
                    break
        out_velocity = tuple(vel)
        out_state = (epoch_idx, order, start, s)
        return losses, grad_norms, out_velocity, out_state
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


def _check_deep_adam_beta(value, name):
    """Adam 衰减系数契约：[0,1) 内的有限 int/float（拒绝 bool）。

    类型错（含 bool 或非 int/float）抛 TypeError，非有限或越界抛
    ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s 必须是 int/float（拒绝 bool），得到 %s"
            % (name, type(value).__name__)
        )
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise ValueError("%s 必须满足 0 <= %s < 1" % (name, name))


def train_deep_adam_batches(
    layers, x, labels, batch_size=1, epochs=1, lr=0.001, seed=0,
    shuffle=True, clip=None, beta1=0.9, beta2=0.999, eps=1e-8,
    m=None, v=None, step=0, state=None, max_batches=None,
):
    """九层网络（结构同 train_deep_momentum_batches）的分轮分批 Adam
    训练：批次切分、Fisher–Yates 洗牌、前反向次序、裁剪前全局梯度范数、
    可选全局范数裁剪、批边界暂停/续训及失败回滚均沿用
    train_deep_momentum_batches，唯一区别在每批参数更新规则——先对八组
    梯度 g 做裁剪，令 t=step+1，再逐叶以 Adam 更新：

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g*g
        p = p - lr * (m/(1-beta1**t)) / (sqrt(v/(1-beta2**t)) + eps)

    其中 m、v 为跨批延续的一阶、二阶矩，初值由 m、v 指定。恒返回
    (losses, grad_norms, m, v, step, state)：losses、grad_norms 语义与
    train_deep_momentum_batches 完全相同（grad_norms 仍记各批裁剪前的
    全局梯度范数）；m、v 为推进后的 8 项 tuple，顺序与八组参数一致
    （conv weights/bias、BN gamma/beta、第一个 Linear weights/bias、
    第二个 Linear weights/bias），每项为与对应参数同形的嵌套 list，不与
    入参别名；step 为累计已完成的更新批数（非负 int）；state 沿用进度
    四元组语义，全部训练完成时为 (epochs, [], 0, rng)。state 或
    max_batches 非 None 时启用暂停/续训，前两项仅记本次实际训练的批。

    beta1、beta2 必须是 [0,1) 内的有限 int/float（拒绝 bool）；eps 必须
    是正的有限 int/float（拒绝 bool）；step 必须是非负 int（拒绝 bool）：
    类型错抛 TypeError，非有限/越界/非正抛 ValueError。m、v 必须同为
    None（八组矩全零，此时 step 必须为 0）或同为恰含 8 项的 tuple，逐项
    沿用 train_deep_momentum_batches 的 velocity 八项契约（与对应参数
    同形、叶值有限）：恰有一个为 None 的配对错，或同为 None 而 step 非
    0 的关系错抛 ValueError；tuple 容器/叶类型错抛 TypeError，长度、
    形状不符或叶非有限抛 ValueError。

    其余参数（layers、x、labels、batch_size、epochs、lr、seed、shuffle、
    clip、state、max_batches）的校验与暂停/续训规则全部沿用
    train_deep_momentum_batches。任一失败（含参数校验、x/labels 不匹配、
    各批前反向与 Adam 计算的非有限值错误）都把九层整体恢复到函数入口
    状态，且不修改 x、labels、m、v、state 及构造参数所用的原 list；成功
    时保留全部参数更新、两矩推进与各批带来的 BN 统计、Dropout 随机推进。
    相同入口状态结果完全确定；任意批边界切分后多次调用的两列表拼接、
    八组参数、两矩、step、BN 运行统计、Dropout 随机状态（含掩码）及
    终态，均与一次训练完成完全相同。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size 必须是 int（拒绝 bool），得到 %s"
            % type(batch_size).__name__
        )
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep_batch_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _check_deep_adam_beta(beta1, "beta1")
    _check_deep_adam_beta(beta2, "beta2")
    _check_deep_scalar(eps, "eps")
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(
            "step 必须是 int（拒绝 bool），得到 %s" % type(step).__name__
        )
    if step < 0:
        raise ValueError("step 必须是非负整数")
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    # 八组参数/矩的同序配对：conv 权重/偏置、BN gamma/beta、两个
    # Linear 权重/偏置。
    param_attrs = (
        (conv, "_weights"), (conv, "_bias"),
        (bn, "_gamma"), (bn, "_beta"),
        (linear1, "_weights"), (linear1, "_bias"),
        (linear2, "_weights"), (linear2, "_bias"),
    )
    moment_names = (
        "conv_weights", "conv_bias",
        "bn_gamma", "bn_beta",
        "linear1_weights", "linear1_bias",
        "linear2_weights", "linear2_bias",
    )

    # m/v 配对契约：须同为 None（八组全零，且 step==0）或同为 8 项 tuple
    # （逐项沿用 velocity 八项同形/有限契约）。恰有一个 None 为配对错，
    # 同为 None 而 step 非 0 为关系错。校验阶段即复制，绝不修改入参。
    m_none, v_none = m is None, v is None
    if m_none != v_none:
        raise ValueError("m 与 v 必须同为 None 或同为 8 项 tuple")
    if m_none:
        if step != 0:
            raise ValueError("m、v 同为 None（初始态）时 step 必须为 0")
        m_trees = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
        v_trees = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
    else:
        if not isinstance(m, tuple) or not isinstance(v, tuple):
            raise TypeError("m、v 必须是 None 或各含 8 项的 tuple")
        if len(m) != 8 or len(v) != 8:
            raise ValueError("m、v 必须各恰含 8 项")
        m_trees = []
        v_trees = []
        for m_tree, v_tree, (owner, attr), mname in zip(
            m, v, param_attrs, moment_names
        ):
            _check_velocity_tree(m_tree, getattr(owner, attr), mname)
            _check_velocity_tree(v_tree, getattr(owner, attr), mname)
            m_trees.append(_deep_copy(m_tree))
            v_trees.append(_deep_copy(v_tree))

    # ---- 批边界暂停/续训参数（state、max_batches）校验 ----
    resume_mode = state is not None or max_batches is not None
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise TypeError(
                "max_batches 必须是 None 或 int（拒绝 bool），得到 %s"
                % type(max_batches).__name__
            )
        if max_batches < 0:
            raise ValueError("max_batches 必须是非负整数")
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None 或 (epoch, order, cursor, rng) 四元组"
                "（tuple），得到 %s" % type(state).__name__
            )
        if len(state) != 4:
            raise ValueError("state 必须恰含 (epoch, order, cursor, rng) 四项")
        cur_epoch, cur_order, cur_cursor, cur_s = state
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for val in cur_order:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(val).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if cur_cursor < 0 or cur_cursor >= n_:
                raise ValueError(
                    "state 的 cursor 必须满足 0 <= cursor < N（%d）" % n_
                )
            # cursor 必须与按 batch_size 切分的各批起点对齐，且对应批
            # 仍有样本（cursor==0 只可能是空 order 的轮起点）。
            if cur_cursor % batch_size != 0:
                raise ValueError(
                    "state 的 cursor %d 不是 batch_size=%d 的批起点"
                    % (cur_cursor, batch_size)
                )
    # epoch == epochs：训练已完成，不得再训练任何批。max_batches 为 0 或
    # None（剩余批为空，自然无事可做）时原样返回完成态；正整数请求训练则
    # 抛 ValueError。
    if (
        cur_epoch == epochs
        and resume_mode
        and max_batches is not None
        and max_batches > 0
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        cur_step = step
        # max_batches 为 None 时训练剩余全部批。
        remaining = max_batches
        done = False

        def _scaled(tree, factor):
            # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
            if isinstance(tree, list):
                return [_scaled(child, factor) for child in tree]
            return tree * factor

        def _adam_step(m_tree, v_tree, grad, param, t, bias1, bias2):
            # 逐叶推进两矩并做偏差校正后更新参数；新矩、新参数均在独立
            # 新 list 中算出并校验，以 (m 树, v 树, 参数树) 三元组返回，
            # 再由调用方同步提交。
            if isinstance(param, list):
                m_children, v_children, p_children = [], [], []
                for mm, vv, g, pp in zip(m_tree, v_tree, grad, param):
                    m_sub, v_sub, p_sub = _adam_step(
                        mm, vv, g, pp, t, bias1, bias2
                    )
                    m_children.append(m_sub)
                    v_children.append(v_sub)
                    p_children.append(p_sub)
                return m_children, v_children, p_children
            new_m = beta1 * m_tree + (1.0 - beta1) * grad
            if not math.isfinite(new_m):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_v = beta2 * v_tree + (1.0 - beta2) * grad * grad
            if not math.isfinite(new_v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            m_hat = new_m / bias1
            v_hat = new_v / bias2
            denom = math.sqrt(v_hat) + eps
            if not math.isfinite(m_hat) or not math.isfinite(v_hat):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            if not math.isfinite(denom):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_value = param - lr * m_hat / denom
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_m, new_v, new_value

        # max_batches=0：不训练任何批、不在轮界洗牌，直接回传当前进度。
        if max_batches == 0:
            done = True

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            while start < n_:
                if remaining is not None:
                    remaining -= 1
                idx = order[start:start + batch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除 N。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 更新前依次展平八组梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                # 每批裁剪后 t=step+1；bias1/bias2 为偏差校正分母
                # 1-beta**t（beta∈[0,1)、t>=1，恒为正）。先裁剪梯度再
                # 逐叶 Adam 更新；八组新矩、新参数均先在独立新 list 中
                # 算出并校验。裁剪可能重绑上述变量，故此处重新组装（裁剪
                # 后的）八组梯度。
                t = cur_step + 1
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                clipped_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                new_m_all = []
                new_v_all = []
                new_params = []
                for m_tree, v_tree, g_tree, (owner, attr) in zip(
                    m_trees, v_trees, clipped_groups, param_attrs
                ):
                    m_new, v_new, p_new = _adam_step(
                        m_tree, v_tree, g_tree, getattr(owner, attr),
                        t, bias1, bias2,
                    )
                    new_m_all.append(m_new)
                    new_v_all.append(v_new)
                    new_params.append(p_new)

                # 同步替换八组层内参数与八组两矩；其余缓存、BN 统计、
                # Dropout 随机状态保留。
                for (owner, attr), p_new in zip(param_attrs, new_params):
                    setattr(owner, attr, p_new)
                m_trees = new_m_all
                v_trees = new_v_all
                cur_step = t

                losses.append(float(loss_value))
                grad_norms.append(float(grad_norm))

                # 批后推进游标（idx 长度即本批实际样本数，末批可短）。
                start += len(idx)
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                if remaining == 0:
                    # 恰在批边界暂停：游标已是下一批起点。
                    done = True
                    break
        out_m = tuple(m_trees)
        out_v = tuple(v_trees)
        out_state = (epoch_idx, order, start, s)
        return losses, grad_norms, out_m, out_v, cur_step, out_state
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


def _check_pending_grad_tree(value, ref, group_name):
    """校验微批边界累计量中的单组梯度：与对应参数同形的嵌套 list，叶为
    有限 int/float（拒绝 bool）。

    累计量以样本加权和承载，允许整数值，故 int/float 均合法。容器或叶
    类型错（含 bool）抛 TypeError，长度/形状不符、层级错位或叶非有限抛
    ValueError。
    """
    if not isinstance(value, list):
        raise TypeError(
            "state 的 pending_grads 的 %s 必须是嵌套 list，得到 %s"
            % (group_name, type(value).__name__)
        )
    if len(value) != len(ref):
        raise ValueError(
            "state 的 pending_grads 的 %s 长度 %d 与参数形状 %d 不符"
            % (group_name, len(value), len(ref))
        )
    for i, (v_child, p_child) in enumerate(zip(value, ref)):
        if isinstance(p_child, list):
            if not isinstance(v_child, list):
                raise TypeError(
                    "state 的 pending_grads 的 %s[%d] 必须是 list，得到 %s"
                    % (group_name, i, type(v_child).__name__)
                )
            _check_pending_grad_tree(
                v_child, p_child, "%s[%d]" % (group_name, i)
            )
        else:
            if isinstance(v_child, list):
                raise ValueError(
                    "state 的 pending_grads 的 %s[%d] 层级过深：标量位置"
                    "出现了 list" % (group_name, i)
                )
            if isinstance(v_child, bool) or not isinstance(
                v_child, (int, float)
            ):
                raise TypeError(
                    "state 的 pending_grads 的 %s[%d] 元素必须是 "
                    "int/float（拒绝 bool），得到 %s"
                    % (group_name, i, type(v_child).__name__)
                )
            if not math.isfinite(v_child):
                raise ValueError(
                    "state 的 pending_grads 的 %s[%d] 含有非有限值"
                    "（NaN/inf）" % (group_name, i)
                )


def train_deep_accum_batches(
    layers, x, labels, microbatch_size=1, accum_steps=2, epochs=1, lr=0.1,
    seed=0, shuffle=True, clip=None, state=None, max_updates=None,
    max_microbatches=None,
):
    """九层网络（结构同 train_deep_step，索引 2 接受 Dropout 或
    Dropout2D，第 4 层接受 MaxPool2D、
    AdaptiveAvgPool2D 或 AdaptiveMaxPool2D）的微批梯度累积分轮训练：
    每轮按 [0,…,N-1]（shuffle 为真时以 seed 起始、跨轮延续的 32 位
    LCG 做 Fisher–Yates 洗牌）切分为大小 microbatch_size 的微批（末批
    可短），逐微批按 train_deep_step 的次序前向、自损失层起逆序反传，
    但不立即更新参数；将微批均值损失与八组梯度各乘微批样本数累加，
    满 accum_steps 个微批或轮末（余组不跨轮）时除以组内累计样本数得
    样本加权均值，随后按 train_deep_batches 的既有顺序展平八组均值
    梯度、以 sqrt(math.fsum(g*g)) 求裁剪前全局范数、可选裁剪并同步
    SGD 更新。返回 (losses, grad_norms)，均为按更新顺序记录的新
    list[float]，长度均为 epochs*ceil(ceil(N/microbatch_size)/
    accum_steps)：losses 为各次更新前的样本加权均值损失，grad_norms
    为裁剪前范数。

    state、max_updates、max_microbatches 均为 None 时返回
    (losses, grad_norms)；任一非 None 时启用暂停/续训，返回
    (losses, grad_norms, state)，前两项仅记本次实际完成的更新。state 为
    None 等价 (0, [], 0, seed)；否则须为四元 tuple
    (epoch, order, cursor, rng)（更新边界）或八元 tuple
    (epoch, order, cursor, rng, pending_samples, pending_microbatches,
    pending_loss, pending_grads)（微批边界）：epoch、cursor、rng 为 int，
    order 为 int 列表（均拒绝 bool），类型错抛 TypeError；长度、范围、
    排列或关系错抛 ValueError。N 为样本数：epoch 须满足
    0 <= epoch <= epochs，rng 须满足 0 <= rng <= 2^32-1；order 为空时
    cursor 必须为 0，非空时须为 0..N-1 的全排列（此时 epoch 必小于
    epochs）。四元态的 cursor 为小于 N 的 microbatch_size*accum_steps
    正倍数（更新边界）。八元态额外要求：cursor 为小于 N 的
    microbatch_size 正倍数（微批边界）；pending_samples 为正 int（拒绝
    bool），且不超过 cursor 与上一更新边界之间的样本数；
    pending_microbatches 为 [1, accum_steps) 内 int；pending_loss 为有限
    float（拒绝 bool）；pending_grads 为恰含 8 项的 tuple（容器类型错抛
    TypeError），八项依次与 conv 权重/偏置、BN gamma/beta、第一个 Linear
    权重/偏置、第二个 Linear 权重/偏置同序同形的嵌套 list，叶为有限
    int/float（拒绝 bool），长度、形状、排列、范围、非有限或字段不一致
    抛 ValueError。四元态轮中 cursor 必须处于更新边界；八元态 cursor 必须
    处于微批边界；累计量不跨轮，轮末短组必结算。
    max_updates 为 None 表示完成剩余全部更新；否则必须是非负 int（拒绝
    bool），类型错抛 TypeError，负值抛 ValueError。max_microbatches 为
    None 表示不以微批计数暂停；否则必须是非负 int（拒绝 bool），类型错
    抛 TypeError，负值抛 ValueError；0 表示不洗牌、不前向任何微批，原样
    回传当前进度。两预算并用时任一耗尽即停：max_updates 按完成的更新
    计数，max_microbatches 按实际前向的微批计数；更新预算耗尽时停在更新
    边界（回传四元态），微批预算先耗尽时停在微批边界（回传携带累计量的
    八元态）。轮末短组结算后 epoch 加一并清空 order，累计量绝不跨轮。
    完成态（epoch == epochs）仅对正数预算（max_updates 或
    max_microbatches 为正）抛 ValueError，预算均为 None 或 0 时原样
    返回。不修改入参 state；任意微批/更新边界分段后多次调用的
    losses/grad_norms 拼接、八组参数、BN 运行统计、Dropout/Dropout2D
    随机状态与终态，均与一次训练完成完全相同。

    layers 的九层类型/顺序与 BN/Dropout/Dropout2D 训练态校验、lr 校验
    以及各微批
    x、labels 的校验均沿用 train_deep_step（各微批仅切取 x、labels 的
    新子 list 传入，不复制样本），另要求 labels 与 x 样本数相等，否则
    抛 ValueError。pool 为 AdaptiveMaxPool2D 时反向把各分箱梯度累加到
    forward 记录的首个最大坐标（分箱可重叠，同一输入坐标可收到多份
    梯度）。epochs、seed、shuffle、clip 的校验沿用 train_deep_batches。
    microbatch_size、accum_steps 必须是正 int（拒绝 bool）：类型错抛
    TypeError，非正抛 ValueError；microbatch_size 大于 N 时每轮仅一个
    含全部样本的短微批。

    洗牌使用与 Dropout 相同的 32 位线性同余发生器
    s=(1664525*s+1013904223) mod 2^32：每轮自 i=N-1 降至 1，先推进 s
    再令 j=s%(i+1) 并交换 order[i]、order[j]；s 自 seed 起跨轮延续，
    shuffle 为假时整轮不推进 s（seed 仍须合法）。该发生器独立于九层
    自身状态。

    累计、求均值、范数或更新后的参数含非有限值均抛 ValueError。任一
    失败（含参数校验、x/labels 不匹配与各微批前反向、非有限值错误）
    都把九层的参数引用、模式、缓存、BN 运行统计、Dropout/Dropout2D
    随机状态与掩码整体恢复到函数入口状态，且不修改 x、labels 及构造参数
    所用的原 list；成功时保留全部参数更新与各微批带来的 BN 统计、
    Dropout/Dropout2D 随机推进。相同入口状态结果完全确定。
    """
    if isinstance(microbatch_size, bool) or not isinstance(
        microbatch_size, int
    ):
        raise TypeError(
            "microbatch_size 必须是 int（拒绝 bool），得到 %s"
            % type(microbatch_size).__name__
        )
    if microbatch_size <= 0:
        raise ValueError("microbatch_size 必须为正整数")
    if isinstance(accum_steps, bool) or not isinstance(accum_steps, int):
        raise TypeError(
            "accum_steps 必须是 int（拒绝 bool），得到 %s"
            % type(accum_steps).__name__
        )
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep2_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    # ---- 暂停/续训参数（state、max_updates、max_microbatches）校验 ----
    resume_mode = (
        state is not None
        or max_updates is not None
        or max_microbatches is not None
    )
    for _budget_name, _budget in (
        ("max_updates", max_updates),
        ("max_microbatches", max_microbatches),
    ):
        if _budget is not None:
            if isinstance(_budget, bool) or not isinstance(_budget, int):
                raise TypeError(
                    "%s 必须是 None 或 int（拒绝 bool），得到 %s"
                    % (_budget_name, type(_budget).__name__)
                )
            if _budget < 0:
                raise ValueError("%s 必须是非负整数" % _budget_name)
    # 八元态累计量的局部承载：四元态/轮界态恒为“无累计量”。
    cur_pending_samples = 0
    cur_pending_micro = 0
    cur_pending_loss = 0.0
    cur_pending_grads = None
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None、四元组 (epoch, order, cursor, rng)"
                "（更新边界）或八元组 (epoch, order, cursor, rng, "
                "pending_samples, pending_microbatches, pending_loss, "
                "pending_grads)（微批边界）（tuple），得到 %s"
                % type(state).__name__
            )
        if len(state) not in (4, 8):
            raise ValueError(
                "state 必须恰含 4 项（更新边界）或 8 项（微批边界），"
                "得到 %d 项" % len(state)
            )
        cur_epoch, cur_order, cur_cursor, cur_s = state[:4]
        group_span = microbatch_size * accum_steps
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for val in cur_order:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(val).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if len(state) == 4:
                # 四元态只停在更新边界：轮中游标必为整组（accum_steps 个
                # 微批）后的下一组起点，即小于 N 的
                # microbatch_size*accum_steps 正倍数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % group_span != 0
                ):
                    raise ValueError(
                        "四元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size*accum_steps（%d）正倍数（更新边界）"
                        % (cur_cursor, n_, group_span)
                    )
            else:
                # 八元态可停在任意微批边界：轮中游标为小于 N 的
                # microbatch_size 正倍数（末个短微批必随轮末结算，不会
                # 暂停），cursor%group_span 即组内已累计样本数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % microbatch_size != 0
                ):
                    raise ValueError(
                        "八元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size（%d）正倍数（微批边界）"
                        % (cur_cursor, n_, microbatch_size)
                    )
        if len(state) == 8:
            (
                cur_pending_samples,
                cur_pending_micro,
                cur_pending_loss,
                cur_pending_grads,
            ) = state[4:]
            if isinstance(cur_pending_samples, bool) or not isinstance(
                cur_pending_samples, int
            ):
                raise TypeError(
                    "state 的 pending_samples 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_samples).__name__
                )
            if cur_pending_samples <= 0:
                raise ValueError("state 的 pending_samples 必须是正整数")
            if isinstance(cur_pending_micro, bool) or not isinstance(
                cur_pending_micro, int
            ):
                raise TypeError(
                    "state 的 pending_microbatches 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_micro).__name__
                )
            if not (1 <= cur_pending_micro < accum_steps):
                raise ValueError(
                    "state 的 pending_microbatches 必须满足 1 <= "
                    "pending_microbatches < accum_steps（%d）"
                    % accum_steps
                )
            if isinstance(cur_pending_loss, bool) or not isinstance(
                cur_pending_loss, float
            ):
                raise TypeError(
                    "state 的 pending_loss 必须是 float（拒绝 bool），"
                    "得到 %s" % type(cur_pending_loss).__name__
                )
            if not math.isfinite(cur_pending_loss):
                raise ValueError(
                    "state 的 pending_loss 必须是有限值（拒绝 NaN/inf）"
                )
            if not isinstance(cur_pending_grads, tuple):
                raise TypeError(
                    "state 的 pending_grads 必须是恰含 8 项的 tuple，"
                    "得到 %s" % type(cur_pending_grads).__name__
                )
            if len(cur_pending_grads) != 8:
                raise ValueError(
                    "state 的 pending_grads 必须恰含 8 项（与八组参数同序）"
                    "，得到 %d 项" % len(cur_pending_grads)
                )
            # 八项依次与 conv 权重/偏置、BN gamma/beta、第一个 Linear
            # 权重/偏置、第二个 Linear 权重/偏置同序同形；层校验已保证八
            # 组参数存在且为有限嵌套 list。
            _conv_l, _bn_l, _l1_l, _l2_l = (
                layers[0], layers[1], layers[5], layers[7]
            )
            _pending_refs = (
                (_conv_l._weights, "conv weights"),
                (_conv_l._bias, "conv bias"),
                (_bn_l._gamma, "batchnorm gamma"),
                (_bn_l._beta, "batchnorm beta"),
                (_l1_l._weights, "linear1 weights"),
                (_l1_l._bias, "linear1 bias"),
                (_l2_l._weights, "linear2 weights"),
                (_l2_l._bias, "linear2 bias"),
            )
            for _g, (_ref, _gname) in zip(cur_pending_grads, _pending_refs):
                _check_pending_grad_tree(_g, _ref, _gname)
            # 字段一致性：轮界（order 空）无累计量；轮中组内累计样本数必
            # 恰为游标越过最近更新边界的样本量，且等于微批数×微批大小。
            _pending_span = cur_cursor % group_span if cur_order else 0
            if cur_pending_samples != _pending_span:
                raise ValueError(
                    "state 的 pending_samples（%d）与 cursor（%d）相对最近"
                    "更新边界的组内样本数（%d）不一致"
                    % (cur_pending_samples, cur_cursor, _pending_span)
                )
            if cur_pending_micro * microbatch_size != cur_pending_samples:
                raise ValueError(
                    "state 的 pending_microbatches（%d）× microbatch_size"
                    "（%d）与 pending_samples（%d）不一致"
                    % (cur_pending_micro, microbatch_size,
                       cur_pending_samples)
                )
    # epoch == epochs：训练已完成，不得再前向任何微批。两预算均为 0 或
    # None（剩余工作为空，自然无事可做）时原样返回完成态；任一预算为正
    # 整数则抛 ValueError。
    if (
        cur_epoch == epochs
        and resume_mode
        and (
            (max_updates is not None and max_updates > 0)
            or (max_microbatches is not None and max_microbatches > 0)
        )
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量；
        # 八元态携带的累计量同样取深拷贝，回传 state 不与入参别名。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        # 两预算独立计数：None 表示不限；任一降至 0 即停（更新预算在结算
        # 后扣减，微批预算在每次前向后扣减）。
        remaining_updates = max_updates
        remaining_micro = max_microbatches
        # 任一预算为 0：不洗牌、不前向任何微批，原样回传当前进度（含八元
        # 态已携带的累计量）。None == 0 恒为 False。
        done = max_updates == 0 or max_microbatches == 0
        # 组内累计器初值取自八元态（四元态/轮界态恒为空）：余组不跨轮，
        # 仅在轮中续训时携带。
        acc_count = cur_pending_samples   # 组内累计样本数
        acc_micro = cur_pending_micro     # 组内已累计微批数
        acc_loss = cur_pending_loss       # Σ 微批均值损失 × 微批样本数
        acc_grads = (                     # 八组 Σ 微批梯度 × 微批样本数
            [_deep_copy(g) for g in cur_pending_grads]
            if cur_pending_grads is not None else None
        )

        # 把 grad 树乘 factor 累加进 acc 树（返回新树，不改原结构）。
        def _add_scaled(acc, grad, factor):
            if isinstance(grad, list):
                return [
                    _add_scaled(a, g, factor) for a, g in zip(acc, grad)
                ]
            value = acc + grad * factor
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 累计树除以组内样本数得样本加权均值（返回新树）。
        def _div_count(tree, count):
            if isinstance(tree, list):
                return [_div_count(v, count) for v in tree]
            value = tree / count
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
        def _scaled(tree, factor):
            if isinstance(tree, list):
                return [_scaled(v, factor) for v in tree]
            return tree * factor

        # 全部新参数先在独立新 list 中算出并校验，再同步提交，保证
        # 失败时层内参数与构造参数原 list 均不被改动。
        def _step(param, grad):
            if isinstance(param, list):
                return [_step(v, g) for v, g in zip(param, grad)]
            new_value = param - lr * grad
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_value

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 某一预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            # 组内累计器：八元态轮中续训时携带暂停前的累计量；轮界进入
            # 新一轮时恒为空。余组不跨轮，轮末必结算清空。
            while start < n_:
                idx = order[start:start + microbatch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除批量。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 微批均值损失与八组梯度各乘微批样本数累加（不更新参数）。
                m_ = len(idx)
                acc_loss += loss_value * m_
                if not math.isfinite(acc_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                micro_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                if acc_grads is None:
                    acc_grads = [
                        _zeros_like_tree(g) for g in micro_groups
                    ]
                acc_grads = [
                    _add_scaled(a, g, m_)
                    for a, g in zip(acc_grads, micro_groups)
                ]
                acc_count += m_
                acc_micro += 1
                start += m_

                # 本次前向完成，扣减微批预算（None 表示不限，不扣减）。
                if remaining_micro is not None:
                    remaining_micro -= 1

                # 满 accum_steps 个微批（更新边界）或轮末（短组必结算，
                # 累计量不跨轮）时结算；否则处于未结算的微批边界。
                settled = acc_micro >= accum_steps or start >= n_
                if not settled:
                    # 微批预算恰在微批边界耗尽：携带组内累计量以八元态
                    # 暂停；预算未尽则继续前向下一微批。
                    if remaining_micro == 0:
                        done = True
                        break
                    continue

                # 除以组内累计样本数得样本加权均值损失与八组均值梯度。
                mean_loss = acc_loss / acc_count
                if not math.isfinite(mean_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                dcw, dcb, dgamma, dbeta, dl1w, dl1b, dl2w, dl2b = (
                    _div_count(a, acc_count) for a in acc_grads
                )

                # 更新前按既有顺序展平八组均值梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                new_conv_w = _step(conv._weights, dcw)
                new_conv_b = _step(conv._bias, dcb)
                new_gamma = _step(bn._gamma, dgamma)
                new_beta = _step(bn._beta, dbeta)
                new_l1_w = _step(linear1._weights, dl1w)
                new_l1_b = _step(linear1._bias, dl1b)
                new_l2_w = _step(linear2._weights, dl2w)
                new_l2_b = _step(linear2._bias, dl2b)

                # 同步替换层内参数；其余缓存、BN 统计、Dropout 随机状态
                # 保留。
                conv._weights = new_conv_w
                conv._bias = new_conv_b
                bn._gamma = new_gamma
                bn._beta = new_beta
                linear1._weights = new_l1_w
                linear1._bias = new_l1_b
                linear2._weights = new_l2_w
                linear2._bias = new_l2_b

                losses.append(float(mean_loss))
                grad_norms.append(float(grad_norm))

                # 结算后清空组内累计器；余组不跨轮。
                acc_count = 0
                acc_loss = 0.0
                acc_grads = None
                acc_micro = 0

                # 本次更新完成，扣减更新预算（None 表示不限，不扣减）。
                if remaining_updates is not None:
                    remaining_updates -= 1
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。累计量已结算，不跨轮。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                # 更新边界（累计器已空）：任一预算耗尽即在此暂停，游标已是
                # 下一组起点（必为 microbatch_size*accum_steps 正倍数）；
                # 两预算皆有余量时继续下一组。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    break
        if resume_mode:
            # 无未结算累计量回传四元更新边界态；否则回传携带样本加权累计
            # 量的八元微批边界态（累计量取深拷贝，不与内部/入参别名）。
            if acc_count > 0:
                out_state = (
                    epoch_idx, order, start, s,
                    acc_count, acc_micro, float(acc_loss),
                    tuple(_deep_copy(g) for g in acc_grads),
                )
            else:
                out_state = (epoch_idx, order, start, s)
            return losses, grad_norms, out_state
        return losses, grad_norms
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


def train_deep_adam_accum_batches(
    layers, x, labels, microbatch_size=1, accum_steps=2, epochs=1,
    lr=0.001, seed=0, shuffle=True, clip=None, beta1=0.9, beta2=0.999,
    eps=1e-8, m=None, v=None, step=0, state=None, max_updates=None,
    max_microbatches=None,
):
    """九层网络（结构同 train_deep_accum_batches）的微批梯度累积 Adam
    分轮训练：微批切分、Fisher–Yates 洗牌、前反向次序、按样本数累计
    微批损失与八组梯度、满 accum_steps 或轮末结算求样本加权均值、
    裁剪前全局梯度范数、可选全局范数裁剪、微批/更新边界暂停续训及
    失败回滚均沿用 train_deep_accum_batches，唯一区别在每次结算后的
    参数更新规则——先对八组均值梯度 g 做裁剪，令 t=step+1，再逐叶以
    Adam 更新（同 train_deep_adam_batches）：

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g*g
        p = p - lr * (m/(1-beta1**t)) / (sqrt(v/(1-beta2**t)) + eps)

    其中 m、v 为跨更新延续的一阶、二阶矩，初值由 m、v 指定。恒返回
    (losses, grad_norms, m, v, step, state)：losses、grad_norms 语义
    与 train_deep_accum_batches 完全相同（均为按更新顺序记录的新
    list[float]，仅记本次实际完成的更新；losses 为各次更新前的样本
    加权均值损失，grad_norms 为裁剪前范数）；m、v 为推进后的 8 项
    tuple，顺序与八组参数一致（conv weights/bias、BN gamma/beta、
    第一个 Linear weights/bias、第二个 Linear weights/bias），每项
    为与对应参数同形的嵌套 list，不与入参别名（m、v 同为 None 时
    展开为八组全零矩）；step 为累计已完成的更新次数（非负 int）；
    state 沿用 train_deep_accum_batches 的进度语义：无未结算累计量
    时为 (epoch, order, cursor, rng) 四元组，否则为携带累计量的
    八元组，全部训练完成时为 (epochs, [], 0, rng)。

    beta1、beta2、eps、step 及 m、v 的校验全部沿用
    train_deep_adam_batches：beta1、beta2 必须是 [0,1) 内的有限
    int/float（拒绝 bool）；eps 必须是正的有限 int/float（拒绝
    bool）；step 必须是非负 int（拒绝 bool）：类型错抛 TypeError，
    非有限/越界/非正抛 ValueError。m、v 必须同为 None（八组矩全零，
    此时 step 必须为 0）或同为恰含 8 项的 tuple（逐项沿用 velocity
    八项同形/有限契约）：恰有一个为 None 的配对错，或同为 None 而
    step 非 0 的关系错抛 ValueError；tuple 容器/叶类型错抛
    TypeError，长度、形状不符或叶非有限抛 ValueError。

    其余参数（layers、x、labels、microbatch_size、accum_steps、
    epochs、lr、seed、shuffle、clip、state、max_updates、
    max_microbatches）的校验与暂停/续训规则全部沿用
    train_deep_accum_batches：state 为 None 等价 (0, [], 0, seed)；
    否则须为四元 tuple（更新边界）或八元 tuple（微批边界，携带
    pending_samples、pending_microbatches、pending_loss、
    pending_grads）。max_updates、max_microbatches 均为 None 时完成
    剩余全部更新；为 0 时不洗牌、不前向任何微批，原样回传当前进度
    （含八元态已携带的累计量）。两预算并用时任一耗尽即停：更新预算
    耗尽停在更新边界（四元态），微批预算先耗尽停在微批边界（八元
    态）。完成态（epoch == epochs）仅对正数预算抛 ValueError。

    累计、求均值、范数、两矩或更新后的参数含非有限值均抛
    ValueError。任一失败（含参数校验、x/labels 不匹配与各微批前
    反向、非有限值错误）都把九层的参数引用、模式、缓存、BN 运行
    统计、Dropout 随机状态与掩码整体恢复到函数入口状态，且不修改
    x、labels、m、v、state 及构造参数所用的原 list；成功时保留全部
    参数更新、两矩推进与各微批带来的 BN 统计、Dropout 随机推进。
    相同入口状态结果完全确定；任意微批/更新边界分段后多次调用的
    losses/grad_norms 拼接、末次调用返回的 m、v、step、state、八组
    参数、BN 运行统计与 Dropout 随机状态，均与一次训练完成完全
    相同。
    """
    if isinstance(microbatch_size, bool) or not isinstance(
        microbatch_size, int
    ):
        raise TypeError(
            "microbatch_size 必须是 int（拒绝 bool），得到 %s"
            % type(microbatch_size).__name__
        )
    if microbatch_size <= 0:
        raise ValueError("microbatch_size 必须为正整数")
    if isinstance(accum_steps, bool) or not isinstance(accum_steps, int):
        raise TypeError(
            "accum_steps 必须是 int（拒绝 bool），得到 %s"
            % type(accum_steps).__name__
        )
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _check_deep_adam_beta(beta1, "beta1")
    _check_deep_adam_beta(beta2, "beta2")
    _check_deep_scalar(eps, "eps")
    if eps <= 0:
        raise ValueError("eps 必须为正数")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(
            "step 必须是 int（拒绝 bool），得到 %s" % type(step).__name__
        )
    if step < 0:
        raise ValueError("step 必须是非负整数")
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    # ---- 暂停/续训预算（max_updates、max_microbatches）校验 ----
    for _budget_name, _budget in (
        ("max_updates", max_updates),
        ("max_microbatches", max_microbatches),
    ):
        if _budget is not None:
            if isinstance(_budget, bool) or not isinstance(_budget, int):
                raise TypeError(
                    "%s 必须是 None 或 int（拒绝 bool），得到 %s"
                    % (_budget_name, type(_budget).__name__)
                )
            if _budget < 0:
                raise ValueError("%s 必须是非负整数" % _budget_name)

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    # 八组参数/矩的同序配对：conv 权重/偏置、BN gamma/beta、两个
    # Linear 权重/偏置。
    param_attrs = (
        (conv, "_weights"), (conv, "_bias"),
        (bn, "_gamma"), (bn, "_beta"),
        (linear1, "_weights"), (linear1, "_bias"),
        (linear2, "_weights"), (linear2, "_bias"),
    )
    moment_names = (
        "conv_weights", "conv_bias",
        "bn_gamma", "bn_beta",
        "linear1_weights", "linear1_bias",
        "linear2_weights", "linear2_bias",
    )

    # m/v 配对契约：须同为 None（八组全零，且 step==0）或同为 8 项 tuple
    # （逐项沿用 velocity 八项同形/有限契约）。恰有一个 None 为配对错，
    # 同为 None 而 step 非 0 为关系错。校验阶段即复制，绝不修改入参。
    m_none, v_none = m is None, v is None
    if m_none != v_none:
        raise ValueError("m 与 v 必须同为 None 或同为 8 项 tuple")
    if m_none:
        if step != 0:
            raise ValueError("m、v 同为 None（初始态）时 step 必须为 0")
        m_trees = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
        v_trees = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
    else:
        if not isinstance(m, tuple) or not isinstance(v, tuple):
            raise TypeError("m、v 必须是 None 或各含 8 项的 tuple")
        if len(m) != 8 or len(v) != 8:
            raise ValueError("m、v 必须各恰含 8 项")
        m_trees = []
        v_trees = []
        for m_tree, v_tree, (owner, attr), mname in zip(
            m, v, param_attrs, moment_names
        ):
            _check_velocity_tree(m_tree, getattr(owner, attr), mname)
            _check_velocity_tree(v_tree, getattr(owner, attr), mname)
            m_trees.append(_deep_copy(m_tree))
            v_trees.append(_deep_copy(v_tree))

    # ---- 暂停/续训进度（state）校验：四元更新边界态或八元微批边界态 ----
    # 八元态累计量的局部承载：四元态/轮界态恒为“无累计量”。
    cur_pending_samples = 0
    cur_pending_micro = 0
    cur_pending_loss = 0.0
    cur_pending_grads = None
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None、四元组 (epoch, order, cursor, rng)"
                "（更新边界）或八元组 (epoch, order, cursor, rng, "
                "pending_samples, pending_microbatches, pending_loss, "
                "pending_grads)（微批边界）（tuple），得到 %s"
                % type(state).__name__
            )
        if len(state) not in (4, 8):
            raise ValueError(
                "state 必须恰含 4 项（更新边界）或 8 项（微批边界），"
                "得到 %d 项" % len(state)
            )
        cur_epoch, cur_order, cur_cursor, cur_s = state[:4]
        group_span = microbatch_size * accum_steps
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for val in cur_order:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(val).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if len(state) == 4:
                # 四元态只停在更新边界：轮中游标必为整组（accum_steps 个
                # 微批）后的下一组起点，即小于 N 的
                # microbatch_size*accum_steps 正倍数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % group_span != 0
                ):
                    raise ValueError(
                        "四元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size*accum_steps（%d）正倍数（更新边界）"
                        % (cur_cursor, n_, group_span)
                    )
            else:
                # 八元态可停在任意微批边界：轮中游标为小于 N 的
                # microbatch_size 正倍数（末个短微批必随轮末结算，不会
                # 暂停），cursor%group_span 即组内已累计样本数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % microbatch_size != 0
                ):
                    raise ValueError(
                        "八元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size（%d）正倍数（微批边界）"
                        % (cur_cursor, n_, microbatch_size)
                    )
        if len(state) == 8:
            (
                cur_pending_samples,
                cur_pending_micro,
                cur_pending_loss,
                cur_pending_grads,
            ) = state[4:]
            if isinstance(cur_pending_samples, bool) or not isinstance(
                cur_pending_samples, int
            ):
                raise TypeError(
                    "state 的 pending_samples 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_samples).__name__
                )
            if cur_pending_samples <= 0:
                raise ValueError("state 的 pending_samples 必须是正整数")
            if isinstance(cur_pending_micro, bool) or not isinstance(
                cur_pending_micro, int
            ):
                raise TypeError(
                    "state 的 pending_microbatches 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_micro).__name__
                )
            if not (1 <= cur_pending_micro < accum_steps):
                raise ValueError(
                    "state 的 pending_microbatches 必须满足 1 <= "
                    "pending_microbatches < accum_steps（%d）"
                    % accum_steps
                )
            if isinstance(cur_pending_loss, bool) or not isinstance(
                cur_pending_loss, float
            ):
                raise TypeError(
                    "state 的 pending_loss 必须是 float（拒绝 bool），"
                    "得到 %s" % type(cur_pending_loss).__name__
                )
            if not math.isfinite(cur_pending_loss):
                raise ValueError(
                    "state 的 pending_loss 必须是有限值（拒绝 NaN/inf）"
                )
            if not isinstance(cur_pending_grads, tuple):
                raise TypeError(
                    "state 的 pending_grads 必须是恰含 8 项的 tuple，"
                    "得到 %s" % type(cur_pending_grads).__name__
                )
            if len(cur_pending_grads) != 8:
                raise ValueError(
                    "state 的 pending_grads 必须恰含 8 项（与八组参数同序）"
                    "，得到 %d 项" % len(cur_pending_grads)
                )
            # 八项依次与 conv 权重/偏置、BN gamma/beta、第一个 Linear
            # 权重/偏置、第二个 Linear 权重/偏置同序同形；层校验已保证八
            # 组参数存在且为有限嵌套 list。
            _pending_refs = (
                (conv._weights, "conv weights"),
                (conv._bias, "conv bias"),
                (bn._gamma, "batchnorm gamma"),
                (bn._beta, "batchnorm beta"),
                (linear1._weights, "linear1 weights"),
                (linear1._bias, "linear1 bias"),
                (linear2._weights, "linear2 weights"),
                (linear2._bias, "linear2 bias"),
            )
            for _g, (_ref, _gname) in zip(cur_pending_grads, _pending_refs):
                _check_pending_grad_tree(_g, _ref, _gname)
            # 字段一致性：轮界（order 空）无累计量；轮中组内累计样本数必
            # 恰为游标越过最近更新边界的样本量，且等于微批数×微批大小。
            _pending_span = cur_cursor % group_span if cur_order else 0
            if cur_pending_samples != _pending_span:
                raise ValueError(
                    "state 的 pending_samples（%d）与 cursor（%d）相对最近"
                    "更新边界的组内样本数（%d）不一致"
                    % (cur_pending_samples, cur_cursor, _pending_span)
                )
            if cur_pending_micro * microbatch_size != cur_pending_samples:
                raise ValueError(
                    "state 的 pending_microbatches（%d）× microbatch_size"
                    "（%d）与 pending_samples（%d）不一致"
                    % (cur_pending_micro, microbatch_size,
                       cur_pending_samples)
                )
    # epoch == epochs：训练已完成，不得再前向任何微批。两预算均为 0 或
    # None（剩余工作为空，自然无事可做）时原样返回完成态；任一预算为正
    # 整数则抛 ValueError。
    if (
        cur_epoch == epochs
        and (
            (max_updates is not None and max_updates > 0)
            or (max_microbatches is not None and max_microbatches > 0)
        )
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量；
        # 八元态携带的累计量同样取深拷贝，回传 state 不与入参别名。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        cur_step = step
        # 两预算独立计数：None 表示不限；任一降至 0 即停（更新预算在结算
        # 后扣减，微批预算在每次前向后扣减）。
        remaining_updates = max_updates
        remaining_micro = max_microbatches
        # 任一预算为 0：不洗牌、不前向任何微批，原样回传当前进度（含八元
        # 态已携带的累计量）。None == 0 恒为 False。
        done = max_updates == 0 or max_microbatches == 0
        # 组内累计器初值取自八元态（四元态/轮界态恒为空）：余组不跨轮，
        # 仅在轮中续训时携带。
        acc_count = cur_pending_samples   # 组内累计样本数
        acc_micro = cur_pending_micro     # 组内已累计微批数
        acc_loss = cur_pending_loss       # Σ 微批均值损失 × 微批样本数
        acc_grads = (                     # 八组 Σ 微批梯度 × 微批样本数
            [_deep_copy(g) for g in cur_pending_grads]
            if cur_pending_grads is not None else None
        )

        # 把 grad 树乘 factor 累加进 acc 树（返回新树，不改原结构）。
        def _add_scaled(acc, grad, factor):
            if isinstance(grad, list):
                return [
                    _add_scaled(a, g, factor) for a, g in zip(acc, grad)
                ]
            value = acc + grad * factor
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 累计树除以组内样本数得样本加权均值（返回新树）。
        def _div_count(tree, count):
            if isinstance(tree, list):
                return [_div_count(v, count) for v in tree]
            value = tree / count
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
        def _scaled(tree, factor):
            if isinstance(tree, list):
                return [_scaled(v, factor) for v in tree]
            return tree * factor

        def _adam_step(m_tree, v_tree, grad, param, t, bias1, bias2):
            # 逐叶推进两矩并做偏差校正后更新参数；新矩、新参数均在独立
            # 新 list 中算出并校验，以 (m 树, v 树, 参数树) 三元组返回，
            # 再由调用方同步提交。
            if isinstance(param, list):
                m_children, v_children, p_children = [], [], []
                for mm, vv, g, pp in zip(m_tree, v_tree, grad, param):
                    m_sub, v_sub, p_sub = _adam_step(
                        mm, vv, g, pp, t, bias1, bias2
                    )
                    m_children.append(m_sub)
                    v_children.append(v_sub)
                    p_children.append(p_sub)
                return m_children, v_children, p_children
            new_m = beta1 * m_tree + (1.0 - beta1) * grad
            if not math.isfinite(new_m):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_v = beta2 * v_tree + (1.0 - beta2) * grad * grad
            if not math.isfinite(new_v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            m_hat = new_m / bias1
            v_hat = new_v / bias2
            denom = math.sqrt(v_hat) + eps
            if not math.isfinite(m_hat) or not math.isfinite(v_hat):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            if not math.isfinite(denom):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_value = param - lr * m_hat / denom
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_m, new_v, new_value

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 某一预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            # 组内累计器：八元态轮中续训时携带暂停前的累计量；轮界进入
            # 新一轮时恒为空。余组不跨轮，轮末必结算清空。
            while start < n_:
                idx = order[start:start + microbatch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除批量。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 微批均值损失与八组梯度各乘微批样本数累加（不更新参数）。
                m_ = len(idx)
                acc_loss += loss_value * m_
                if not math.isfinite(acc_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                micro_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                if acc_grads is None:
                    acc_grads = [
                        _zeros_like_tree(g) for g in micro_groups
                    ]
                acc_grads = [
                    _add_scaled(a, g, m_)
                    for a, g in zip(acc_grads, micro_groups)
                ]
                acc_count += m_
                acc_micro += 1
                start += m_

                # 本次前向完成，扣减微批预算（None 表示不限，不扣减）。
                if remaining_micro is not None:
                    remaining_micro -= 1

                # 满 accum_steps 个微批（更新边界）或轮末（短组必结算，
                # 累计量不跨轮）时结算；否则处于未结算的微批边界。
                settled = acc_micro >= accum_steps or start >= n_
                if not settled:
                    # 微批预算恰在微批边界耗尽：携带组内累计量以八元态
                    # 暂停；预算未尽则继续前向下一微批。
                    if remaining_micro == 0:
                        done = True
                        break
                    continue

                # 除以组内累计样本数得样本加权均值损失与八组均值梯度。
                mean_loss = acc_loss / acc_count
                if not math.isfinite(mean_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                dcw, dcb, dgamma, dbeta, dl1w, dl1b, dl2w, dl2b = (
                    _div_count(a, acc_count) for a in acc_grads
                )

                # 更新前按既有顺序展平八组均值梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                # 每次结算后 t=step+1；bias1/bias2 为偏差校正分母
                # 1-beta**t（beta∈[0,1)、t>=1，恒为正）。先裁剪梯度再
                # 逐叶 Adam 更新；八组新矩、新参数均先在独立新 list 中
                # 算出并校验。裁剪可能重绑上述变量，故此处重新组装（裁剪
                # 后的）八组梯度。
                t = cur_step + 1
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                clipped_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                new_m_all = []
                new_v_all = []
                new_params = []
                for m_tree, v_tree, g_tree, (owner, attr) in zip(
                    m_trees, v_trees, clipped_groups, param_attrs
                ):
                    m_new, v_new, p_new = _adam_step(
                        m_tree, v_tree, g_tree, getattr(owner, attr),
                        t, bias1, bias2,
                    )
                    new_m_all.append(m_new)
                    new_v_all.append(v_new)
                    new_params.append(p_new)

                # 同步替换八组层内参数与八组两矩；其余缓存、BN 统计、
                # Dropout 随机状态保留。
                for (owner, attr), p_new in zip(param_attrs, new_params):
                    setattr(owner, attr, p_new)
                m_trees = new_m_all
                v_trees = new_v_all
                cur_step = t

                losses.append(float(mean_loss))
                grad_norms.append(float(grad_norm))

                # 结算后清空组内累计器；余组不跨轮。
                acc_count = 0
                acc_loss = 0.0
                acc_grads = None
                acc_micro = 0

                # 本次更新完成，扣减更新预算（None 表示不限，不扣减）。
                if remaining_updates is not None:
                    remaining_updates -= 1
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。累计量已结算，不跨轮。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                # 更新边界（累计器已空）：任一预算耗尽即在此暂停，游标已是
                # 下一组起点（必为 microbatch_size*accum_steps 正倍数）；
                # 两预算皆有余量时继续下一组。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    break
        out_m = tuple(m_trees)
        out_v = tuple(v_trees)
        # 无未结算累计量回传四元更新边界态；否则回传携带样本加权累计
        # 量的八元微批边界态（累计量取深拷贝，不与内部/入参别名）。
        if acc_count > 0:
            out_state = (
                epoch_idx, order, start, s,
                acc_count, acc_micro, float(acc_loss),
                tuple(_deep_copy(g) for g in acc_grads),
            )
        else:
            out_state = (epoch_idx, order, start, s)
        return losses, grad_norms, out_m, out_v, cur_step, out_state
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


def train_deep_momentum_accum_batches(
    layers, x, labels, microbatch_size=1, accum_steps=2, epochs=1,
    lr=0.1, seed=0, shuffle=True, clip=None, momentum=0.9, velocity=None,
    state=None, max_updates=None, max_microbatches=None,
):
    """九层网络（结构同 train_deep_accum_batches）的微批梯度累积带动量
    分轮训练：微批切分、Fisher–Yates 洗牌、前反向次序、按样本数累计
    微批损失与八组梯度、满 accum_steps 或轮末结算求样本加权均值、
    裁剪前全局梯度范数、可选全局范数裁剪、四元/八元 state 暂停续训、
    双预算及失败回滚均沿用 train_deep_accum_batches，唯一区别在每次
    结算后的参数更新规则——先对八组均值梯度 g 做裁剪，再逐叶以带动量
    SGD 更新（同 train_deep_momentum_batches）：

        v = momentum * v + g
        p = p - lr * v

    其中 v 为跨更新延续的速度，初值由 velocity 指定；仅结算时推进，
    未结算的微批不影响 v。恒返回 (losses, grad_norms, velocity,
    state)：losses、grad_norms 语义与 train_deep_accum_batches 完全
    相同（均为按更新顺序记录的新 list[float]，仅记本次实际完成的
    更新；losses 为各次更新前的样本加权均值损失，grad_norms 为裁剪
    前范数）；velocity 为推进后的 8 项 tuple，顺序与八组参数一致
    （conv weights/bias、BN gamma/beta、第一个 Linear weights/bias、
    第二个 Linear weights/bias），每项为与对应参数同形的嵌套 list，
    不与入参 velocity 别名（velocity 为 None 时视为八组全零速度）；
    state 沿用 train_deep_accum_batches 的进度语义：无未结算累计量
    时为 (epoch, order, cursor, rng) 四元组，否则为携带累计量的
    八元组，全部训练完成时为 (epochs, [], 0, rng)。

    momentum 必须是 [0,1) 内的有限 int/float（拒绝 bool）：类型错抛
    TypeError，非有限或越界抛 ValueError。velocity 必须为 None 或恰
    含 8 项的 tuple，依次与上述八组参数同序；每项必须是与对应参数
    同形的嵌套 list，叶值必须是有限 int/float（拒绝 bool）：容器或
    叶类型错抛 TypeError，tuple 长度、张量形状不符或叶非有限抛
    ValueError；None 视为八组全零速度。

    其余参数（layers、x、labels、microbatch_size、accum_steps、
    epochs、lr、seed、shuffle、clip、state、max_updates、
    max_microbatches）的校验与暂停/续训规则全部沿用
    train_deep_accum_batches：state 为 None 等价 (0, [], 0, seed)；
    否则须为四元 tuple（更新边界）或八元 tuple（微批边界，携带
    pending_samples、pending_microbatches、pending_loss、
    pending_grads）。max_updates、max_microbatches 均为 None 时完成
    剩余全部更新；为 0 时不洗牌、不前向任何微批，原样回传当前进度
    （含八元态已携带的累计量）。两预算并用时任一耗尽即停：更新预算
    耗尽停在更新边界（四元态），微批预算先耗尽停在微批边界（八元
    态）。完成态（epoch == epochs）仅对正数预算抛 ValueError。

    累计、求均值、范数、速度或更新后的参数含非有限值均抛
    ValueError。任一失败（含参数校验、x/labels 不匹配与各微批前
    反向、非有限值错误）都把九层的参数引用、模式、缓存、BN 运行
    统计、Dropout 随机状态与掩码整体恢复到函数入口状态，且不修改
    x、labels、velocity、state 及构造参数所用的原 list；成功时保留
    全部参数更新、速度推进与各微批带来的 BN 统计、Dropout 随机
    推进。相同入口状态结果完全确定；任意微批/更新边界分段后多次
    调用的 losses/grad_norms 拼接、末次调用返回的 velocity、
    state、八组参数、BN 运行统计与 Dropout 随机状态，均与一次训练
    完成完全相同。
    """
    if isinstance(microbatch_size, bool) or not isinstance(
        microbatch_size, int
    ):
        raise TypeError(
            "microbatch_size 必须是 int（拒绝 bool），得到 %s"
            % type(microbatch_size).__name__
        )
    if microbatch_size <= 0:
        raise ValueError("microbatch_size 必须为正整数")
    if isinstance(accum_steps, bool) or not isinstance(accum_steps, int):
        raise TypeError(
            "accum_steps 必须是 int（拒绝 bool），得到 %s"
            % type(accum_steps).__name__
        )
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正整数")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(
            "epochs 必须是 int（拒绝 bool），得到 %s"
            % type(epochs).__name__
        )
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "seed 必须是 int（拒绝 bool），得到 %s" % type(seed).__name__
        )
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("seed 必须满足 0 <= seed <= 2^32-1")
    if not isinstance(shuffle, bool):
        raise TypeError(
            "shuffle 必须是 bool，得到 %s" % type(shuffle).__name__
        )
    _validate_deep_layers(layers)
    _check_deep_scalar(lr, "lr")
    if lr <= 0:
        raise ValueError("lr 必须为正数")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)):
            raise TypeError(
                "clip 必须是 None 或 int/float（拒绝 bool），得到 %s"
                % type(clip).__name__
            )
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip 必须为正的有限值")
    _check_deep_momentum(momentum)
    _require_list(x, "x")
    x_shape = _shape_of(x, 4, "x")
    n_ = x_shape[0]
    _require_list(labels, "labels")
    if len(labels) != n_:
        raise ValueError(
            "labels 长度 %d 与 x 样本数 %d 不符" % (len(labels), n_)
        )

    # ---- 暂停/续训预算（max_updates、max_microbatches）校验 ----
    for _budget_name, _budget in (
        ("max_updates", max_updates),
        ("max_microbatches", max_microbatches),
    ):
        if _budget is not None:
            if isinstance(_budget, bool) or not isinstance(_budget, int):
                raise TypeError(
                    "%s 必须是 None 或 int（拒绝 bool），得到 %s"
                    % (_budget_name, type(_budget).__name__)
                )
            if _budget < 0:
                raise ValueError("%s 必须是非负整数" % _budget_name)

    conv, bn, dropout, pool, flatten, linear1, relu, linear2, loss = layers
    # 八组参数/速度的同序配对：conv 权重/偏置、BN gamma/beta、两个
    # Linear 权重/偏置。
    param_attrs = (
        (conv, "_weights"), (conv, "_bias"),
        (bn, "_gamma"), (bn, "_beta"),
        (linear1, "_weights"), (linear1, "_bias"),
        (linear2, "_weights"), (linear2, "_bias"),
    )
    velocity_names = (
        "conv_weights", "conv_bias",
        "bn_gamma", "bn_beta",
        "linear1_weights", "linear1_bias",
        "linear2_weights", "linear2_bias",
    )

    # velocity 校验：None 视为八组全零；否则须为恰含 8 项的 tuple，逐项
    # 与对应参数同形、叶值有限。校验阶段即复制，绝不修改入参 velocity。
    if velocity is None:
        vel = [
            _zeros_like_tree(getattr(owner, attr))
            for owner, attr in param_attrs
        ]
    else:
        if not isinstance(velocity, tuple):
            raise TypeError(
                "velocity 必须是 None 或 8 项 tuple，得到 %s"
                % type(velocity).__name__
            )
        if len(velocity) != 8:
            raise ValueError(
                "velocity 必须恰含 8 项，得到 %d 项" % len(velocity)
            )
        vel = []
        for v_tree, (owner, attr), vname in zip(
            velocity, param_attrs, velocity_names
        ):
            _check_velocity_tree(v_tree, getattr(owner, attr), vname)
            vel.append(_deep_copy(v_tree))

    # ---- 暂停/续训进度（state）校验：四元更新边界态或八元微批边界态 ----
    # 八元态累计量的局部承载：四元态/轮界态恒为“无累计量”。
    cur_pending_samples = 0
    cur_pending_micro = 0
    cur_pending_loss = 0.0
    cur_pending_grads = None
    if state is None:
        cur_epoch, cur_order, cur_cursor, cur_s = 0, [], 0, seed
    else:
        if not isinstance(state, tuple):
            raise TypeError(
                "state 必须是 None、四元组 (epoch, order, cursor, rng)"
                "（更新边界）或八元组 (epoch, order, cursor, rng, "
                "pending_samples, pending_microbatches, pending_loss, "
                "pending_grads)（微批边界）（tuple），得到 %s"
                % type(state).__name__
            )
        if len(state) not in (4, 8):
            raise ValueError(
                "state 必须恰含 4 项（更新边界）或 8 项（微批边界），"
                "得到 %d 项" % len(state)
            )
        cur_epoch, cur_order, cur_cursor, cur_s = state[:4]
        group_span = microbatch_size * accum_steps
        if isinstance(cur_epoch, bool) or not isinstance(cur_epoch, int):
            raise TypeError(
                "state 的 epoch 必须是 int（拒绝 bool），得到 %s"
                % type(cur_epoch).__name__
            )
        if cur_epoch < 0 or cur_epoch > epochs:
            raise ValueError(
                "state 的 epoch 必须满足 0 <= epoch <= epochs（%d）"
                % epochs
            )
        if not isinstance(cur_order, list):
            raise TypeError(
                "state 的 order 必须是 list，得到 %s"
                % type(cur_order).__name__
            )
        for val in cur_order:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    "state 的 order 成员必须是 int（拒绝 bool），得到 %s"
                    % type(val).__name__
                )
        if isinstance(cur_cursor, bool) or not isinstance(cur_cursor, int):
            raise TypeError(
                "state 的 cursor 必须是 int（拒绝 bool），得到 %s"
                % type(cur_cursor).__name__
            )
        if isinstance(cur_s, bool) or not isinstance(cur_s, int):
            raise TypeError(
                "state 的 rng 必须是 int（拒绝 bool），得到 %s"
                % type(cur_s).__name__
            )
        if cur_s < 0 or cur_s > 0xFFFFFFFF:
            raise ValueError("state 的 rng 必须满足 0 <= rng <= 2^32-1")
        if len(cur_order) == 0:
            if cur_cursor != 0:
                raise ValueError("state 的 order 为空时 cursor 必须为 0")
        else:
            if cur_epoch == epochs:
                # epoch==epochs 表示训练已完成（轮毕必清空 order），
                # 不可能同时停在某一轮中途。
                raise ValueError(
                    "state 的 epoch 已等于 epochs（%d），order 必须为空"
                    % epochs
                )
            if len(cur_order) != n_:
                raise ValueError(
                    "state 的 order 长度 %d 必须等于样本数 %d"
                    % (len(cur_order), n_)
                )
            if sorted(cur_order) != list(range(n_)):
                raise ValueError(
                    "state 的 order 必须恰是 0..%d 的一个全排列" % (n_ - 1)
                )
            if len(state) == 4:
                # 四元态只停在更新边界：轮中游标必为整组（accum_steps 个
                # 微批）后的下一组起点，即小于 N 的
                # microbatch_size*accum_steps 正倍数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % group_span != 0
                ):
                    raise ValueError(
                        "四元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size*accum_steps（%d）正倍数（更新边界）"
                        % (cur_cursor, n_, group_span)
                    )
            else:
                # 八元态可停在任意微批边界：轮中游标为小于 N 的
                # microbatch_size 正倍数（末个短微批必随轮末结算，不会
                # 暂停），cursor%group_span 即组内已累计样本数。
                if (
                    cur_cursor <= 0
                    or cur_cursor >= n_
                    or cur_cursor % microbatch_size != 0
                ):
                    raise ValueError(
                        "八元 state 的 cursor %d 必须是小于 N（%d）的 "
                        "microbatch_size（%d）正倍数（微批边界）"
                        % (cur_cursor, n_, microbatch_size)
                    )
        if len(state) == 8:
            (
                cur_pending_samples,
                cur_pending_micro,
                cur_pending_loss,
                cur_pending_grads,
            ) = state[4:]
            if isinstance(cur_pending_samples, bool) or not isinstance(
                cur_pending_samples, int
            ):
                raise TypeError(
                    "state 的 pending_samples 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_samples).__name__
                )
            if cur_pending_samples <= 0:
                raise ValueError("state 的 pending_samples 必须是正整数")
            if isinstance(cur_pending_micro, bool) or not isinstance(
                cur_pending_micro, int
            ):
                raise TypeError(
                    "state 的 pending_microbatches 必须是 int（拒绝 bool），"
                    "得到 %s" % type(cur_pending_micro).__name__
                )
            if not (1 <= cur_pending_micro < accum_steps):
                raise ValueError(
                    "state 的 pending_microbatches 必须满足 1 <= "
                    "pending_microbatches < accum_steps（%d）"
                    % accum_steps
                )
            if isinstance(cur_pending_loss, bool) or not isinstance(
                cur_pending_loss, float
            ):
                raise TypeError(
                    "state 的 pending_loss 必须是 float（拒绝 bool），"
                    "得到 %s" % type(cur_pending_loss).__name__
                )
            if not math.isfinite(cur_pending_loss):
                raise ValueError(
                    "state 的 pending_loss 必须是有限值（拒绝 NaN/inf）"
                )
            if not isinstance(cur_pending_grads, tuple):
                raise TypeError(
                    "state 的 pending_grads 必须是恰含 8 项的 tuple，"
                    "得到 %s" % type(cur_pending_grads).__name__
                )
            if len(cur_pending_grads) != 8:
                raise ValueError(
                    "state 的 pending_grads 必须恰含 8 项（与八组参数同序）"
                    "，得到 %d 项" % len(cur_pending_grads)
                )
            # 八项依次与 conv 权重/偏置、BN gamma/beta、第一个 Linear
            # 权重/偏置、第二个 Linear 权重/偏置同序同形；层校验已保证八
            # 组参数存在且为有限嵌套 list。
            _pending_refs = (
                (conv._weights, "conv weights"),
                (conv._bias, "conv bias"),
                (bn._gamma, "batchnorm gamma"),
                (bn._beta, "batchnorm beta"),
                (linear1._weights, "linear1 weights"),
                (linear1._bias, "linear1 bias"),
                (linear2._weights, "linear2 weights"),
                (linear2._bias, "linear2 bias"),
            )
            for _g, (_ref, _gname) in zip(cur_pending_grads, _pending_refs):
                _check_pending_grad_tree(_g, _ref, _gname)
            # 字段一致性：轮界（order 空）无累计量；轮中组内累计样本数必
            # 恰为游标越过最近更新边界的样本量，且等于微批数×微批大小。
            _pending_span = cur_cursor % group_span if cur_order else 0
            if cur_pending_samples != _pending_span:
                raise ValueError(
                    "state 的 pending_samples（%d）与 cursor（%d）相对最近"
                    "更新边界的组内样本数（%d）不一致"
                    % (cur_pending_samples, cur_cursor, _pending_span)
                )
            if cur_pending_micro * microbatch_size != cur_pending_samples:
                raise ValueError(
                    "state 的 pending_microbatches（%d）× microbatch_size"
                    "（%d）与 pending_samples（%d）不一致"
                    % (cur_pending_micro, microbatch_size,
                       cur_pending_samples)
                )
    # epoch == epochs：训练已完成，不得再前向任何微批。两预算均为 0 或
    # None（剩余工作为空，自然无事可做）时原样返回完成态；任一预算为正
    # 整数则抛 ValueError。
    if (
        cur_epoch == epochs
        and (
            (max_updates is not None and max_updates > 0)
            or (max_microbatches is not None and max_microbatches > 0)
        )
    ):
        raise ValueError("训练已完成（epoch == epochs），不得继续训练")

    snapshot = _snapshot_deep_layers(layers)
    try:
        losses = []
        grad_norms = []
        # 续训入参 state 绝不修改：order 取副本，游标与 LCG 状态用局部量；
        # 八元态携带的累计量同样取深拷贝，回传 state 不与入参别名。
        epoch_idx = cur_epoch
        order = list(cur_order)
        start = cur_cursor
        s = cur_s
        # 两预算独立计数：None 表示不限；任一降至 0 即停（更新预算在结算
        # 后扣减，微批预算在每次前向后扣减）。
        remaining_updates = max_updates
        remaining_micro = max_microbatches
        # 任一预算为 0：不洗牌、不前向任何微批，原样回传当前进度（含八元
        # 态已携带的累计量）。None == 0 恒为 False。
        done = max_updates == 0 or max_microbatches == 0
        # 组内累计器初值取自八元态（四元态/轮界态恒为空）：余组不跨轮，
        # 仅在轮中续训时携带。
        acc_count = cur_pending_samples   # 组内累计样本数
        acc_micro = cur_pending_micro     # 组内已累计微批数
        acc_loss = cur_pending_loss       # Σ 微批均值损失 × 微批样本数
        acc_grads = (                     # 八组 Σ 微批梯度 × 微批样本数
            [_deep_copy(g) for g in cur_pending_grads]
            if cur_pending_grads is not None else None
        )

        # 把 grad 树乘 factor 累加进 acc 树（返回新树，不改原结构）。
        def _add_scaled(acc, grad, factor):
            if isinstance(grad, list):
                return [
                    _add_scaled(a, g, factor) for a, g in zip(acc, grad)
                ]
            value = acc + grad * factor
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 累计树除以组内样本数得样本加权均值（返回新树）。
        def _div_count(tree, count):
            if isinstance(tree, list):
                return [_div_count(v, count) for v in tree]
            value = tree / count
            if not math.isfinite(value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return value

        # 以新嵌套 list 承载裁剪后梯度，不改 backward 返回的原结构。
        def _scaled(tree, factor):
            if isinstance(tree, list):
                return [_scaled(v, factor) for v in tree]
            return tree * factor

        def _momentum_step(v_tree, grad, param):
            # 逐叶 v=momentum*v+g、p=p-lr*v；新速度、新参数均在独立新
            # list 中算出并校验，以 (速度树, 参数树) 对返回，再由调用方
            # 同步提交。
            if isinstance(param, list):
                v_children, p_children = [], []
                for vv, g, pp in zip(v_tree, grad, param):
                    v_sub, p_sub = _momentum_step(vv, g, pp)
                    v_children.append(v_sub)
                    p_children.append(p_sub)
                return v_children, p_children
            new_v = momentum * v_tree + grad
            if not math.isfinite(new_v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            new_value = param - lr * new_v
            if not math.isfinite(new_value):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            return new_v, new_value

        while not done and epoch_idx < epochs:
            # order 为空表示停在轮起点：仅此刻在轮界按既有 LCG 洗牌；
            # 轮中暂停（order 非空）沿用暂停时的 order 与 rng，不重洗。
            if not order:
                # 某一预算恰在上一轮用尽时，不得为下一轮提前洗牌，直接以
                # 轮界完成态 (epoch, [], 0, rng) 暂停。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    continue
                start = 0
                order = list(range(n_))
                if shuffle and n_ > 1:
                    # Fisher–Yates 洗牌：自 N-1 降至 1，先推进 LCG，
                    # 再以 j=s%(i+1) 交换；s 跨轮延续。
                    for i in range(n_ - 1, 0, -1):
                        s = (1664525 * s + 1013904223) % 4294967296
                        j = s % (i + 1)
                        order[i], order[j] = order[j], order[i]
            # 组内累计器：八元态轮中续训时携带暂停前的累计量；轮界进入
            # 新一轮时恒为空。余组不跨轮，轮末必结算清空。
            while start < n_:
                idx = order[start:start + microbatch_size]
                batch_x = [x[k] for k in idx]
                batch_labels = [labels[k] for k in idx]

                # 按列表顺序前向；各层输入错误由其 forward 原样抛出。
                conv_out = conv.forward(batch_x)
                bn_out = bn.forward(conv_out)
                drop_out = dropout.forward(bn_out)
                pool_out = pool.forward(drop_out)
                flat = flatten.forward(pool_out)
                hidden = linear1.forward(flat)
                relu_out = relu.forward(hidden)
                logits = linear2.forward(relu_out)
                loss_value = loss.forward(logits, batch_labels)
                if not math.isfinite(loss_value):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 自损失层起逆序反传；损失梯度已批均，不再除批量。每层
                # 返回后立即递归检查其全部输出梯度（含参数梯度与最终
                # 输入梯度）。
                grad_logits = loss.backward()
                _require_finite_grads(grad_logits)
                dx_relu, dl2w, dl2b = linear2.backward(grad_logits)
                _require_finite_grads(dx_relu)
                _require_finite_grads(dl2w)
                _require_finite_grads(dl2b)
                dx_hidden = relu.backward(dx_relu)
                _require_finite_grads(dx_hidden)
                dx_flat, dl1w, dl1b = linear1.backward(dx_hidden)
                _require_finite_grads(dx_flat)
                _require_finite_grads(dl1w)
                _require_finite_grads(dl1b)
                dx_pool = flatten.backward(dx_flat)
                _require_finite_grads(dx_pool)
                dx_drop = pool.backward(dx_pool)
                _require_finite_grads(dx_drop)
                dx_bn = dropout.backward(dx_drop)
                _require_finite_grads(dx_bn)
                dx_conv, dgamma, dbeta = bn.backward(dx_bn)
                _require_finite_grads(dx_conv)
                _require_finite_grads(dgamma)
                _require_finite_grads(dbeta)
                dx_input, dcw, dcb = conv.backward(dx_conv)
                _require_finite_grads(dx_input)
                _require_finite_grads(dcw)
                _require_finite_grads(dcb)

                # 微批均值损失与八组梯度各乘微批样本数累加（不更新参数，
                # 不推进速度）。
                m_ = len(idx)
                acc_loss += loss_value * m_
                if not math.isfinite(acc_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                micro_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                if acc_grads is None:
                    acc_grads = [
                        _zeros_like_tree(g) for g in micro_groups
                    ]
                acc_grads = [
                    _add_scaled(a, g, m_)
                    for a, g in zip(acc_grads, micro_groups)
                ]
                acc_count += m_
                acc_micro += 1
                start += m_

                # 本次前向完成，扣减微批预算（None 表示不限，不扣减）。
                if remaining_micro is not None:
                    remaining_micro -= 1

                # 满 accum_steps 个微批（更新边界）或轮末（短组必结算，
                # 累计量不跨轮）时结算；否则处于未结算的微批边界。
                settled = acc_micro >= accum_steps or start >= n_
                if not settled:
                    # 微批预算恰在微批边界耗尽：携带组内累计量以八元态
                    # 暂停；预算未尽则继续前向下一微批。
                    if remaining_micro == 0:
                        done = True
                        break
                    continue

                # 除以组内累计样本数得样本加权均值损失与八组均值梯度。
                mean_loss = acc_loss / acc_count
                if not math.isfinite(mean_loss):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")
                dcw, dcb, dgamma, dbeta, dl1w, dl1b, dl2w, dl2b = (
                    _div_count(a, acc_count) for a in acc_grads
                )

                # 更新前按既有顺序展平八组均值梯度，求裁剪前全局范数。
                grad_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                flat_grads = []
                for group in grad_groups:
                    _flatten_into(group, flat_grads)
                grad_norm = math.sqrt(math.fsum(g * g for g in flat_grads))
                if not math.isfinite(grad_norm):
                    raise ValueError("训练计算产生非有限值（NaN/inf）")

                # 可选全局范数裁剪：八组梯度同乘 clip/norm；范数不大于
                # clip 或 clip 为 None 时保持原梯度。
                if clip is not None and grad_norm > clip:
                    factor = clip / grad_norm
                    dcw, dcb, dgamma, dbeta = (
                        _scaled(dcw, factor), _scaled(dcb, factor),
                        _scaled(dgamma, factor), _scaled(dbeta, factor),
                    )
                    dl1w, dl1b, dl2w, dl2b = (
                        _scaled(dl1w, factor), _scaled(dl1b, factor),
                        _scaled(dl2w, factor), _scaled(dl2b, factor),
                    )

                # 先裁剪梯度，再逐叶 v=momentum*v+g、p=p-lr*v；八组新
                # 速度、新参数均先在独立新 list 中算出并校验。裁剪可能
                # 重绑上述变量，故此处重新组装（裁剪后的）八组梯度。
                clipped_groups = (
                    dcw, dcb, dgamma, dbeta,
                    dl1w, dl1b, dl2w, dl2b,
                )
                new_vel = []
                new_params = []
                for v_tree, g_tree, (owner, attr) in zip(
                    vel, clipped_groups, param_attrs
                ):
                    v_new, p_new = _momentum_step(
                        v_tree, g_tree, getattr(owner, attr)
                    )
                    new_vel.append(v_new)
                    new_params.append(p_new)

                # 同步替换八组层内参数与八组速度；其余缓存、BN 统计、
                # Dropout 随机状态保留。
                for (owner, attr), p_new in zip(param_attrs, new_params):
                    setattr(owner, attr, p_new)
                vel = new_vel

                losses.append(float(mean_loss))
                grad_norms.append(float(grad_norm))

                # 结算后清空组内累计器；余组不跨轮。
                acc_count = 0
                acc_loss = 0.0
                acc_grads = None
                acc_micro = 0

                # 本次更新完成，扣减更新预算（None 表示不限，不扣减）。
                if remaining_updates is not None:
                    remaining_updates -= 1
                if start >= n_:
                    # 轮毕：epoch 加一、清空 order、游标归 0；rng 保持
                    # 本轮轮界洗牌后的值，跨轮延续。累计量已结算，不跨轮。
                    epoch_idx += 1
                    order = []
                    start = 0
                    break
                # 更新边界（累计器已空）：任一预算耗尽即在此暂停，游标已是
                # 下一组起点（必为 microbatch_size*accum_steps 正倍数）；
                # 两预算皆有余量时继续下一组。
                if remaining_updates == 0 or remaining_micro == 0:
                    done = True
                    break
        out_velocity = tuple(vel)
        # 无未结算累计量回传四元更新边界态；否则回传携带样本加权累计
        # 量的八元微批边界态（累计量取深拷贝，不与内部/入参别名）。
        if acc_count > 0:
            out_state = (
                epoch_idx, order, start, s,
                acc_count, acc_micro, float(acc_loss),
                tuple(_deep_copy(g) for g in acc_grads),
            )
        else:
            out_state = (epoch_idx, order, start, s)
        return losses, grad_norms, out_velocity, out_state
    except BaseException:
        _restore_deep_layers(layers, snapshot)
        raise


# ---------------------------------------------------------------------------
# 训练检查点：dump_norm_checkpoint(layers, epoch)、
# load_norm_checkpoint(data)
# ---------------------------------------------------------------------------

_CKPT_TOP_KEYS = ["epoch", "model", "dropout_state"]
# resumedata 检查点的顶层键序：在 epoch 前增加 version、data_sha256。
_CKPT_DATA_TOP_KEYS = [
    "version", "data_sha256", "epoch", "model", "dropout_state",
]
# resumedata 检查点版本：当前固定为 1。
_CKPT_DATA_VERSION = 1
# 原始字节 SHA-256 小写 hex：恰好 64 个十六进制小写字符。
_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z")
# model 完整沿用 fitnorm 产物的逐层键序（见 _cmd_fitnorm）。
_CKPT_MODEL_KEYS = ["conv", "batchnorm", "linear"]
_CKPT_LAYER_KEYS = ["values", "bias"]
_CKPT_BN_KEYS = ["gamma", "beta", "running_mean", "running_var"]
_CKPT_CONV_W_SHAPE = (2, 1, 1, 1)
_CKPT_BIAS_SHAPE = (2,)
_CKPT_LINEAR_W_SHAPE = (2, 2)
# float.hex() 产出的严格语法：可选负号、0x、十六进制尾数（整数部分 1 位，
# 小数部分含小数点与至少 1 位）、p 与带符号十进制指数；无空白、不接受
# 十进制写法（"1.0"）、inf/nan 或下划线。float.fromhex 比该语法更宽松，
# 故先以此正则把关再交 fromhex 取值。
_HEX_FLOAT_RE = re.compile(r"\A-?0x[0-9a-f]\.[0-9a-f]+p[+-][0-9]+\Z")
# resumenorm 的 EPOCHS：仅 "0" 或无前导零 ASCII 正整数（拒绝 +1、01、
# 负数、空白、全角数字等任何其他拼写）。
_RESUME_EPOCHS_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")
# 标准 json 以 NaN/Infinity/-Infinity 为合法标量（非标准扩展），检查点
# 一律拒绝；逐字节扫描时跳过字符串字面量，避免键名/取值中的字符误判。
_JSON_TOKEN_RE = re.compile(
    rb'"(?:[^"\\]|\\.)*"|(-?Infinity|NaN)|.',
    re.DOTALL,
)


def _reject_json_constants(data):
    """拒绝 JSON 文本中的 NaN/Infinity/-Infinity 常量，跳过字符串字面量。"""
    for match in _JSON_TOKEN_RE.finditer(data):
        if match.group(1) is not None:
            raise ValueError("JSON 不允许 NaN/Infinity/-Infinity 常量")


# 紧凑 JSON 不得含结构性空白；字符串字面量内部的空格属于数据（紧凑序列化
# 不会转义空格），逐字节扫描时跳过字符串，仅在字面量之外命中空白才拒绝。
_JSON_WS_TOKEN_RE = re.compile(rb'"(?:[^"\\]|\\.)*"|([ \t\n\r])', re.DOTALL)


def _reject_json_whitespace(data):
    """拒绝字符串字面量之外的空格/制表/换行/回车（即要求紧凑序列化）。"""
    for match in _JSON_WS_TOKEN_RE.finditer(data):
        if match.group(1) is not None:
            raise ValueError("不得含结构性空白，须为紧凑 JSON")


def _validate_norm_checkpoint_layers(layers):
    """检查点的七层合法性：结构契约 + fitnorm 配置/形状/有限性。

    先经 _validate_norm_layers 校验七层长度、类型与顺序、BN/Dropout
    训练态；再要求七层的配置与 fitnorm 一致（_build_norm_layers）：
    Conv2D 为 1×1 单通道 stride=1、无补边/空洞/分组；BatchNorm2D 的
    eps/momentum 为 fitnorm 取值；Dropout 的 p/seed 为 fitnorm 取值；
    池化恰为 MaxPool2D(2,2,0)；Linear 输入维 2；各参数张量形状固定为
    conv weights [2][1][1][1]、conv bias [2]、BN 四向量 [2]、
    linear weights [2][2]、linear bias [2]，全部叶值有限且为 int/float
    （拒绝 bool）。长度/模式/配置/形状/非有限错抛 ValueError，
    容器/元素类型错抛 TypeError。
    """
    _validate_norm_layers(layers)
    conv, bn, dropout, pool, _flatten, linear, _loss = layers

    if conv._w_shape != _CKPT_CONV_W_SHAPE:
        raise ValueError(
            "Conv2D weights 形状必须为 [2][1][1][1]，得到 %s"
            % (conv._w_shape,)
        )
    if conv._stride != (1, 1):
        raise ValueError("Conv2D stride 必须与 fitnorm 一致（1, 1）")
    if conv._padding != (0, 0, 0, 0):
        raise ValueError("Conv2D padding 必须与 fitnorm 一致（0）")
    if conv._dilation != (1, 1):
        raise ValueError("Conv2D dilation 必须与 fitnorm 一致（1）")
    if conv._groups != 1:
        raise ValueError("Conv2D groups 必须与 fitnorm 一致（1）")
    if conv._padding_mode != "zeros":
        raise ValueError(
            "Conv2D padding_mode 必须与 fitnorm 一致（zeros），得到 %r"
            % conv._padding_mode
        )

    if bn._eps != _NORM_EPS:
        raise ValueError("BatchNorm2D eps 必须与 fitnorm 一致（%r）" % _NORM_EPS)
    if bn._momentum != _NORM_MOMENTUM:
        raise ValueError(
            "BatchNorm2D momentum 必须与 fitnorm 一致（%r）" % _NORM_MOMENTUM
        )

    if dropout._p != _NORM_DROPOUT_P:
        raise ValueError(
            "Dropout p 必须与 fitnorm 一致（%r）" % _NORM_DROPOUT_P
        )
    if dropout._seed != _NORM_DROPOUT_SEED:
        raise ValueError(
            "Dropout seed 必须与 fitnorm 一致（%d）" % _NORM_DROPOUT_SEED
        )

    # fitnorm 的第四层恰为 MaxPool2D(2, 2, 0)，不接受自适应池化。
    if not isinstance(pool, MaxPool2D):
        raise ValueError("第 4 层必须是 fitnorm 的 MaxPool2D，得到 AdaptiveAvgPool2D")
    if pool._kernel_size != (2, 2):
        raise ValueError("MaxPool2D kernel_size 必须与 fitnorm 一致（2, 2）")
    if pool._stride != (2, 2):
        raise ValueError("MaxPool2D stride 必须与 fitnorm 一致（2, 2）")
    if pool._padding != (0, 0, 0, 0):
        raise ValueError("MaxPool2D padding 必须与 fitnorm 一致（0）")

    if linear._w_shape != _CKPT_LINEAR_W_SHAPE:
        raise ValueError(
            "Linear weights 形状必须为 [2][2]，得到 %s" % (linear._w_shape,)
        )

    # 形状校验同时保证全部叶值为有限 int/float（拒绝 bool）。
    if _shape_of(conv._weights, 4, "conv weights") != _CKPT_CONV_W_SHAPE:
        raise ValueError("conv weights 的形状必须为 [2][1][1][1]")
    if _shape_of(conv._bias, 1, "conv bias") != _CKPT_BIAS_SHAPE:
        raise ValueError("conv bias 的形状必须为 [2]")
    for name, vector in (
        ("gamma", bn._gamma),
        ("beta", bn._beta),
        ("running_mean", bn.running_mean),
        ("running_var", bn.running_var),
    ):
        if _shape_of(vector, 1, name) != _CKPT_BIAS_SHAPE:
            raise ValueError("%s 的形状必须为 [2]" % name)
    if _shape_of(linear._weights, 2, "linear weights") != _CKPT_LINEAR_W_SHAPE:
        raise ValueError("linear weights 的形状必须为 [2][2]")
    if _shape_of(linear._bias, 1, "linear bias") != _CKPT_BIAS_SHAPE:
        raise ValueError("linear bias 的形状必须为 [2]")

    # Dropout 随机状态始终由 LCG 约束在 [0, 2^32-1]，此处仅做防御性校验。
    if isinstance(dropout._s, bool) or not isinstance(dropout._s, int):
        raise TypeError(
            "Dropout 随机状态必须是 int，得到 %s"
            % type(dropout._s).__name__
        )
    if dropout._s < 0 or dropout._s > 0xFFFFFFFF:
        raise ValueError("Dropout 随机状态必须在 [0, 2^32-1] 内")


def _dump_hex_tensor(node, depth, name):
    """把深度恰为 depth 的嵌套 list 张量序列化为 hex 字符串嵌套 list。"""
    if depth == 0:
        if isinstance(node, list):
            raise ValueError("%s 的层级过深：标量位置出现了 list" % name)
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise TypeError(
                "%s 的元素必须是 int/float（拒绝 bool），得到 %s"
                % (name, type(node).__name__)
            )
        try:
            value = float(node)
        except OverflowError:
            # 超出 float 范围的 int 按非有限值处理。
            raise ValueError("%s 的数值超出 float 范围" % name)
        if not math.isfinite(value):
            raise ValueError("%s 含有非有限值（NaN/inf）" % name)
        # float.hex() 对负零产出 "-0x0.0p+0"，统一规范为 "0x0.0p+0"。
        text = value.hex()
        if text == "-0x0.0p+0":
            text = "0x0.0p+0"
        return json.dumps(text)
    if not isinstance(node, list):
        raise TypeError(
            "%s 必须是嵌套 list，得到 %s" % (name, type(node).__name__)
        )
    return "[" + ",".join(
        _dump_hex_tensor(child, depth - 1, name) for child in node
    ) + "]"


def dump_norm_checkpoint(layers, epoch):
    """序列化 fitnorm 训练态七层网络与 epoch，返回紧凑 JSON bytes（末尾 LF）。

    layers 须符合 train_norm_step 的七项契约且配置与 fitnorm 一致，
    BatchNorm2D 与 Dropout 必须处于训练态（校验见
    _validate_norm_checkpoint_layers）；epoch 为非负 int（拒绝 bool）。
    类型错抛 TypeError；长度、模式、配置、形状或非有限值错抛
    ValueError。返回值为 UTF-8 编码的紧凑 JSON 加单个换行（LF），
    顶层键依次为 epoch、model、dropout_state；model 完整沿用 fitnorm
    产物的逐层键序（conv.values/conv.bias、batchnorm 的 gamma/beta/
    running_mean/running_var、linear.values/linear.bias）与数组形状；
    dropout_state 为 Dropout 当前 LCG 随机状态，[0, 2^32-1] 内的 int。
    model 的全部数值叶写为 Python float.hex() 的小写十六进制字符串，
    负零统一写为 "0x0.0p+0"。只读不修改 layers；同一状态重复调用产出
    逐字节相同的 bytes。
    """
    _validate_norm_checkpoint_layers(layers)
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(
            "epoch 必须是 int（拒绝 bool），得到 %s" % type(epoch).__name__
        )
    if epoch < 0:
        raise ValueError("epoch 必须为非负整数")
    conv, bn, dropout, _pool, _flatten, linear, _loss = layers

    parts = []
    parts.append('"epoch":' + str(int(epoch)))
    parts.append(_dump_checkpoint_model_text(conv, bn, linear))
    parts.append('"dropout_state":' + str(int(dropout._s)))
    text = "{" + ",".join(parts) + "}\n"
    return text.encode("utf-8")


def _dump_checkpoint_model_text(conv, bn, linear):
    """序列化检查点 model 字段（完整沿用 fitnorm 逐层键序与形状）。"""
    conv_values = _dump_hex_tensor(conv._weights, 4, "conv values")
    conv_bias = _dump_hex_tensor(conv._bias, 1, "conv bias")
    gamma = _dump_hex_tensor(bn._gamma, 1, "gamma")
    beta = _dump_hex_tensor(bn._beta, 1, "beta")
    running_mean = _dump_hex_tensor(bn.running_mean, 1, "running_mean")
    running_var = _dump_hex_tensor(bn.running_var, 1, "running_var")
    linear_values = _dump_hex_tensor(linear._weights, 2, "linear values")
    linear_bias = _dump_hex_tensor(linear._bias, 1, "linear bias")
    return (
        '"model":{"conv":{"values":' + conv_values
        + ',"bias":' + conv_bias + '},"batchnorm":{"gamma":' + gamma
        + ',"beta":' + beta + ',"running_mean":' + running_mean
        + ',"running_var":' + running_var + '},"linear":{"values":'
        + linear_values + ',"bias":' + linear_bias + "}}"
    )


def _parse_hex_tensor(node, depth, shape, name):
    """解析并校验 depth 层嵌套 list，叶为 hex 字符串，返回新 float 张量。

    容器类型错抛 TypeError；形状不符（空维/不规则/尺寸不符）抛
    ValueError；叶非 str、非 float.hex() 严格语法、非 dump_norm_checkpoint
    规范小写串（负零仅 "0x0.0p+0"）、非法 hex 或非有限值（含指数越界）
    抛 ValueError。
    """
    if depth == 0:
        if isinstance(node, list):
            raise ValueError("%s 的层级过深：标量位置出现了 list" % name)
        if not isinstance(node, str):
            raise TypeError(
                "%s 的叶值必须是 JSON 字符串，得到 %s"
                % (name, type(node).__name__)
            )
        if _HEX_FLOAT_RE.match(node) is None:
            raise ValueError("%s 含非法十六进制浮点：%r" % (name, node))
        try:
            value = float.fromhex(node)
        except ValueError:
            raise ValueError("%s 含非法十六进制浮点：%r" % (name, node))
        except OverflowError:
            # 指数越界（如 0x1.0p+99999）按非有限值处理。
            raise ValueError("%s 的十六进制浮点越界：%r" % (name, node))
        if not math.isfinite(value):
            raise ValueError("%s 含有非有限值（NaN/inf）" % name)
        # 仅接受 dump_norm_checkpoint 产出的规范串：与 float.hex() 逐字符
        # 相同（唯一小写、指数符号齐全、去前导零），负零唯一写法为
        # 0x0.0p+0（float.hex() 的 -0x0.0p+0 同样被规范化）。
        canonical = value.hex()
        if canonical == "-0x0.0p+0":
            canonical = "0x0.0p+0"
        if canonical != node:
            raise ValueError(
                "%s 含非规范十六进制浮点（只接受 dump_norm_checkpoint 的"
                " 规范小写串）：%r" % (name, node)
            )
        return value
    if not isinstance(node, list):
        raise TypeError(
            "%s 必须是嵌套 list，得到 %s" % (name, type(node).__name__)
        )
    if len(node) != shape[0]:
        raise ValueError(
            "%s 的形状必须为 %s" % (name, "".join("[%d]" % d for d in shape))
        )
    return [
        _parse_hex_tensor(child, depth - 1, shape[1:], name)
        for child in node
    ]


def load_norm_checkpoint(data):
    """从 dump_norm_checkpoint 的 bytes 重建无缓存训练态七层网络。

    data 必须是 bytes（bytearray 等其他类型一律 TypeError）。其内容须为
    UTF-8 编码的检查点 JSON：非法 UTF-8 抛 UnicodeDecodeError；JSON
    语法错、JSON 常量 NaN/Infinity/-Infinity、重复/缺失/额外/错序键、
    非法或非规范 hex（仅接受 dump_norm_checkpoint 的规范小写串，负零仅
    "0x0.0p+0"）、形状/范围错或非有限值抛 ValueError，容器或字段类型
    错抛 TypeError。

    返回 (layers, epoch)：layers 为七层新 list
    （Conv2D/BatchNorm2D/Dropout/MaxPool2D/Flatten/Linear/
    SoftmaxCrossEntropy），配置同 fitnorm（Conv 1×1 stride 1、
    BN eps=1e-5 momentum=1、Dropout p=0.25 seed=7、MaxPool(2,2,0)），
    BN 与 Dropout 均为训练态、不含任何前向缓存；BN 的 running_mean/
    running_var 取自检查点，Dropout 的 LCG 随机状态取 dropout_state。
    epoch 为非负 int。加载后续训与不中断训练得到的参数、BN 统计及
    Dropout 状态完全相同；不修改实参 data。
    """
    if not isinstance(data, bytes):
        raise TypeError(
            "data 必须是 bytes，得到 %s" % type(data).__name__
        )
    # 非法 UTF-8 原样抛 UnicodeDecodeError（strict 为默认行为）。
    text = data.decode("utf-8")
    # json 接受非标准的 NaN/Infinity 常量，先逐字节拒绝（跳过字符串）。
    _reject_json_constants(data)
    doc = json.loads(
        text, object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("检查点顶层必须是 JSON 对象")
    if list(doc.keys()) != _CKPT_TOP_KEYS:
        raise ValueError(
            "检查点顶层键必须依次为 epoch、model、dropout_state"
        )

    state = _parse_norm_checkpoint_state(doc)
    return _layers_from_checkpoint_state(state)


def _parse_norm_checkpoint_state(doc):
    """校验并取出检查点文档中 norm 检查点共有的字段。

    顶层须含（依次）epoch、model、dropout_state（resumedata 检查点在其前
    另有 version、data_sha256，键序由调用方先行校验）。返回一个状态 dict：
    epoch（非负 int）、model 解析出的八组张量与 dropout_state
    （[0,2^32-1] 内 int）。重复/缺失/额外/错序键、类型/形状/取值或非规范
    hex 错误分别抛 ValueError/TypeError。
    """
    epoch = doc["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(
            "epoch 必须是 int，得到 %s" % type(epoch).__name__
        )
    if epoch < 0:
        raise ValueError("epoch 必须为非负整数")

    model = doc["model"]
    if not isinstance(model, dict):
        raise TypeError("model 必须是 JSON 对象，得到 %s" % type(model).__name__)
    if list(model.keys()) != _CKPT_MODEL_KEYS:
        raise ValueError("model 的键必须依次为 conv、batchnorm、linear")
    dropout_state = doc["dropout_state"]
    if isinstance(dropout_state, bool) or not isinstance(dropout_state, int):
        raise TypeError(
            "dropout_state 必须是 int，得到 %s"
            % type(dropout_state).__name__
        )
    if dropout_state < 0 or dropout_state > 0xFFFFFFFF:
        raise ValueError("dropout_state 必须在 [0, 2^32-1] 内")

    conv_obj = model["conv"]
    bn_obj = model["batchnorm"]
    linear_obj = model["linear"]
    for obj_name, obj in (
        ("conv", conv_obj),
        ("batchnorm", bn_obj),
        ("linear", linear_obj),
    ):
        if not isinstance(obj, dict):
            raise TypeError(
                "%s 必须是 JSON 对象，得到 %s"
                % (obj_name, type(obj).__name__)
            )
    if list(conv_obj.keys()) != _CKPT_LAYER_KEYS:
        raise ValueError("conv 的键必须依次为 values、bias")
    if list(bn_obj.keys()) != _CKPT_BN_KEYS:
        raise ValueError(
            "batchnorm 的键必须依次为 gamma、beta、running_mean、running_var"
        )
    if list(linear_obj.keys()) != _CKPT_LAYER_KEYS:
        raise ValueError("linear 的键必须依次为 values、bias")

    conv_w = _parse_hex_tensor(
        conv_obj["values"], 4, _CKPT_CONV_W_SHAPE, "conv values"
    )
    conv_b = _parse_hex_tensor(
        conv_obj["bias"], 1, _CKPT_BIAS_SHAPE, "conv bias"
    )
    gamma = _parse_hex_tensor(bn_obj["gamma"], 1, _CKPT_BIAS_SHAPE, "gamma")
    beta = _parse_hex_tensor(bn_obj["beta"], 1, _CKPT_BIAS_SHAPE, "beta")
    running_mean = _parse_hex_tensor(
        bn_obj["running_mean"], 1, _CKPT_BIAS_SHAPE, "running_mean"
    )
    running_var = _parse_hex_tensor(
        bn_obj["running_var"], 1, _CKPT_BIAS_SHAPE, "running_var"
    )
    lin_w = _parse_hex_tensor(
        linear_obj["values"], 2, _CKPT_LINEAR_W_SHAPE, "linear values"
    )
    lin_b = _parse_hex_tensor(
        linear_obj["bias"], 1, _CKPT_BIAS_SHAPE, "linear bias"
    )
    return {
        "epoch": int(epoch),
        "conv_w": conv_w,
        "conv_b": conv_b,
        "gamma": gamma,
        "beta": beta,
        "running_mean": running_mean,
        "running_var": running_var,
        "lin_w": lin_w,
        "lin_b": lin_b,
        "dropout_state": dropout_state,
    }


def _layers_from_checkpoint_state(state):
    """以解析出的检查点状态新建无缓存训练态七层网络并返回 (layers, epoch)。"""
    conv = Conv2D(state["conv_w"], state["conv_b"])
    bn = BatchNorm2D(
        state["gamma"], state["beta"], _NORM_EPS, _NORM_MOMENTUM
    )
    dropout = Dropout(_NORM_DROPOUT_P, _NORM_DROPOUT_SEED)
    pool = MaxPool2D(2, 2, 0)
    flatten = Flatten()
    linear = Linear(state["lin_w"], state["lin_b"])
    loss = SoftmaxCrossEntropy()

    # 构造后 running_mean/var 为默认 0/1：以检查点统计覆盖；
    # Dropout 的 LCG 状态恢复为 dropout_state（默认即训练态）。
    bn.running_mean = state["running_mean"]
    bn.running_var = state["running_var"]
    dropout._s = state["dropout_state"]

    return [conv, bn, dropout, pool, flatten, linear, loss], state["epoch"]


def dump_data_checkpoint(layers, epoch, data_sha256):
    """序列化 resumedata 检查点：绑定 DATA 身份（原始字节 SHA-256）。

    与 dump_norm_checkpoint 的 model/dropout_state 契约完全一致，顶层键
    依次为 version、data_sha256、epoch、model、dropout_state；version
    固定为 int 1，data_sha256 须为 64 位小写十六进制 str（DATA 原始字节
    的 SHA-256 摘要）。layers/epoch 的校验与异常同 dump_norm_checkpoint；
    data_sha256 类型错抛 TypeError，摘要格式非法抛 ValueError。返回紧凑
    UTF-8 JSON bytes（末尾 LF）；同一状态重复调用逐字节相同。
    """
    _validate_norm_checkpoint_layers(layers)
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(
            "epoch 必须是 int（拒绝 bool），得到 %s" % type(epoch).__name__
        )
    if epoch < 0:
        raise ValueError("epoch 必须为非负整数")
    if not isinstance(data_sha256, str):
        raise TypeError(
            "data_sha256 必须是 str，得到 %s" % type(data_sha256).__name__
        )
    if _SHA256_HEX_RE.match(data_sha256) is None:
        raise ValueError(
            "data_sha256 必须是 64 位小写十六进制 SHA-256 摘要：%r"
            % data_sha256
        )
    conv, bn, dropout, _pool, _flatten, linear, _loss = layers

    parts = []
    parts.append('"version":' + str(int(_CKPT_DATA_VERSION)))
    parts.append('"data_sha256":' + json.dumps(data_sha256))
    parts.append('"epoch":' + str(int(epoch)))
    parts.append(_dump_checkpoint_model_text(conv, bn, linear))
    parts.append('"dropout_state":' + str(int(dropout._s)))
    text = "{" + ",".join(parts) + "}\n"
    return text.encode("utf-8")


def load_data_checkpoint(data, data_sha256):
    """从 dump_data_checkpoint 的 bytes 重建无缓存训练态七层网络。

    校验与 load_norm_checkpoint 相同（UTF-8、JSON 常量、重复/错序键、
    规范 hex、形状与范围），另要求顶层键依次为 version、data_sha256、
    epoch、model、dropout_state：version 恰为 int 1（拒绝 bool），
    data_sha256 为 64 位小写十六进制 str 且与实参 data_sha256 逐字符
    相同，否则抛 ValueError（摘要不符）。实参 data_sha256 非 str 抛
    TypeError。返回 (layers, epoch)，结构与 load_norm_checkpoint 一致。
    """
    if not isinstance(data, bytes):
        raise TypeError(
            "data 必须是 bytes，得到 %s" % type(data).__name__
        )
    if not isinstance(data_sha256, str):
        raise TypeError(
            "data_sha256 必须是 str，得到 %s" % type(data_sha256).__name__
        )
    text = data.decode("utf-8")
    _reject_json_constants(data)
    doc = json.loads(
        text, object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("检查点顶层必须是 JSON 对象")
    if list(doc.keys()) != _CKPT_DATA_TOP_KEYS:
        raise ValueError(
            "检查点顶层键必须依次为 version、data_sha256、epoch、"
            "model、dropout_state"
        )

    version = doc["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(
            "version 必须是 int，得到 %s" % type(version).__name__
        )
    if version != _CKPT_DATA_VERSION:
        raise ValueError(
            "version 必须为 %d，得到 %r" % (_CKPT_DATA_VERSION, version)
        )

    digest = doc["data_sha256"]
    if not isinstance(digest, str):
        raise TypeError(
            "data_sha256 必须是 JSON 字符串，得到 %s"
            % type(digest).__name__
        )
    if _SHA256_HEX_RE.match(digest) is None:
        raise ValueError(
            "data_sha256 必须是 64 位小写十六进制 SHA-256 摘要：%r"
            % digest
        )
    if digest != data_sha256:
        raise ValueError("检查点 data_sha256 与 DATA 摘要不符")

    state = _parse_norm_checkpoint_state(doc)
    return _layers_from_checkpoint_state(state)


def _parse_resume_epochs(text):
    """解析 resumenorm 的 EPOCHS：仅接受 "0" 或无前导零 ASCII 正整数。"""
    if _RESUME_EPOCHS_RE.match(text) is None:
        raise ValueError("EPOCHS 必须为 0 或无前导零的正整数：%r" % text)
    return int(text)


def _cmd_resumenorm(input_path, epochs_text, output_path):
    """resumenorm 子命令主体；成功 0、契约/轮数非法等失败返回 1。

    INPUT 为 "-" 时以 fitnorm 初值新建训练态七层、start=0；否则读取其
    全部字节交 load_norm_checkpoint 加载并以检查点 epoch 为 start。数据
    沿用 fitnorm 对 data/tiny.csv 的严格校验，以 lr=0.1 追加 EPOCHS 轮
    train_norm；EPOCHS=0 时不训练、不刷新统计，状态原样保持。OUTPUT
    写出 dump_norm_checkpoint(layers, start+EPOCHS) 的字节，原子替换；
    成功后标准输出为紧凑 JSON 加 LF，键序 start_epoch、added_epochs、
    loss（前两者 int，loss 为本段逐轮更新前批均损失，固定 12 位小数、
    负零归零）。任一失败标准输出为空且不改 OUTPUT。
    """
    try:
        if input_path != "-" and os.path.abspath(input_path) == os.path.abspath(
            output_path
        ):
            raise ValueError("INPUT 与 OUTPUT 不能是同一路径")
        epochs = _parse_resume_epochs(epochs_text)

        if input_path == "-":
            layers = _build_norm_layers()
            start = 0
        else:
            with open(input_path, "rb") as f:
                data = f.read()
            layers, start = load_norm_checkpoint(data)

        # 数据契约与 fitnorm 完全一致（逐字节校验 data/tiny.csv）。
        x, labels = _load_cnn_samples()

        if epochs == 0:
            # 0 轮：不训练、不做末次 BN 刷新，状态与检查点/初值逐位一致。
            losses = []
        else:
            losses = train_norm(
                layers, x, labels, epochs=epochs, lr=_NORM_LR
            )
        losses = [float(v) for v in losses]
        for v in losses:
            if not math.isfinite(v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")

        payload = dump_norm_checkpoint(layers, start + epochs)
        report = {
            "start_epoch": start,
            "added_epochs": epochs,
            "loss": losses,
        }
        report_text = (_dump_compact(report) + "\n").encode("utf-8")

        # 原子替换成功后才向标准输出写报告，保证失败时 stdout 为空。
        _atomic_write_output(output_path, payload)
        sys.stdout.buffer.write(report_text)
        sys.stdout.buffer.flush()
    except (_TrainDataError, ValueError, TypeError, OSError):
        return 1
    return 0


def _cmd_resumedata(data_path, input_path, epochs_text, output_path):
    """resumedata 子命令主体；成功 0、契约/轮数/摘要非法等失败返回 1。

    DATA 沿用 fitdata 契约，其身份为文件原始字节的 SHA-256 小写 hex。
    INPUT 为 "-" 时以 fitnorm 初值新建训练态七层、start=0；否则读取其
    全部字节交 load_data_checkpoint 加载（检查点 data_sha256 须与 DATA
    摘要逐字符相同），以检查点 epoch 为 start。以 lr=0.1 在该 DATA 上
    追加 EPOCHS 轮 train_norm；EPOCHS=0 时不训练、不刷新统计，状态原样
    保持。OUTPUT 写出 dump_data_checkpoint(layers, start+EPOCHS, digest)
    的字节，原子替换；成功后标准输出为紧凑 JSON 加 LF，键序
    start_epoch、added_epochs、loss（int、int、本段逐轮更新前批均损失
    float 列表，固定 12 位小数、负零归零）。任一失败标准输出为空且不改
    OUTPUT。
    """
    try:
        if os.path.abspath(data_path) == os.path.abspath(output_path):
            raise ValueError("DATA 与 OUTPUT 不能是同一路径")
        if input_path != "-" and os.path.abspath(input_path) == os.path.abspath(
            output_path
        ):
            raise ValueError("INPUT 与 OUTPUT 不能是同一路径")
        epochs = _parse_resume_epochs(epochs_text)

        with open(data_path, "rb") as f:
            data_raw = f.read()
        digest = hashlib.sha256(data_raw).hexdigest()
        x, labels = _parse_fitdata(data_raw)

        if input_path == "-":
            layers = _build_norm_layers()
            start = 0
        else:
            with open(input_path, "rb") as f:
                checkpoint_raw = f.read()
            layers, start = load_data_checkpoint(checkpoint_raw, digest)

        if epochs == 0:
            # 0 轮：不训练、不做末次 BN 刷新，状态与检查点/初值逐位一致。
            losses = []
        else:
            losses = train_norm(
                layers, x, labels, epochs=epochs, lr=_NORM_LR
            )
        losses = [float(v) for v in losses]
        for v in losses:
            if not math.isfinite(v):
                raise ValueError("训练计算产生非有限值（NaN/inf）")

        payload = dump_data_checkpoint(layers, start + epochs, digest)
        report = {
            "start_epoch": start,
            "added_epochs": epochs,
            "loss": losses,
        }
        report_text = (_dump_compact(report) + "\n").encode("utf-8")

        # 原子替换成功后才向标准输出写报告，保证失败时 stdout 为空。
        _atomic_write_output(output_path, payload)
        sys.stdout.buffer.write(report_text)
        sys.stdout.buffer.flush()
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行双数据绑定验证续训：
# python convnet.py resumeval TRAIN VAL INPUT EPOCHS OUTPUT
# ---------------------------------------------------------------------------

_RESUMEVAL_VERSION = 2
_RESUMEVAL_TOP_KEYS = [
    "version", "train_sha", "val_sha", "epoch", "model", "rng", "metrics",
]
_RESUMEVAL_METRICS_KEYS = ["loss", "val_loss", "accuracy"]

# resumevalstats 为 resumeval 的逐类统计版本：顶层契约不变，version 固定为
# 3；metrics 增加 class_correct、class_total、confusion 三个逐轮整数序列。
_RESUMEVALSTATS_VERSION = 3
_RESUMEVALSTATS_TOP_KEYS = _RESUMEVAL_TOP_KEYS
_RESUMEVAL_FLOAT_KEYS = ["loss", "val_loss", "accuracy"]
_RESUMEVALSTATS_METRICS_KEYS = _RESUMEVAL_FLOAT_KEYS + [
    "class_correct", "class_total", "confusion",
]


def _eval_norm_stats(layers, x_val, val_labels):
    """不改训练状态地评估当前七层网络，返回
    (val_loss, accuracy, class_correct, class_total, confusion)。

    取当前 conv/BN/linear 参数与 BN 运行统计，经 _norm_forward 以
    training=False（BN 用保存统计、Dropout 推理恒等）在 VAL 全批上做一次
    前向：val_loss 为 SoftmaxCrossEntropy 的批均交叉熵；逐样本取最大 logit
    （并列取较小类别 0）得预测类。类别序固定 0、1：class_total/class_correct
    为长度 2 的各类样本数/正确数 list；confusion 为 [2][2]，行真实类、列
    预测类；accuracy 为对角和/样本数。layers 的参数、模式、缓存、BN 统计
    与 Dropout 随机状态均不被读取外的任何修改——_norm_forward 由参数深拷贝
    另建七层，调用后 layers 仍为训练态、Dropout 状态不推进。
    """
    conv, bn, _dropout, _pool, _flatten, linear, _loss = layers
    logits, _, _ = _norm_forward(
        conv._weights, conv._bias, bn._gamma, bn._beta,
        bn.running_mean, bn.running_var,
        linear._weights, linear._bias, x_val, False,
    )
    n_val = len(x_val)
    for n in range(n_val):
        for o in range(_CNN_NUM_CLASSES):
            if not math.isfinite(logits[n][o]):
                raise ValueError("验证计算产生非有限值（NaN/inf）")
    val_loss = SoftmaxCrossEntropy().forward(logits, val_labels)
    if not math.isfinite(val_loss):
        raise ValueError("验证计算产生非有限值（NaN/inf）")
    class_correct = [0, 0]
    class_total = [0, 0]
    confusion = [[0, 0], [0, 0]]
    for n in range(n_val):
        true_label = val_labels[n]
        class_total[true_label] += 1
        row = logits[n]
        pred = 0
        for o in range(1, _CNN_NUM_CLASSES):
            if row[o] > row[pred]:
                pred = o
        confusion[true_label][pred] += 1
        if pred == true_label:
            class_correct[true_label] += 1
    correct = class_correct[0] + class_correct[1]
    accuracy = correct / n_val
    return (
        float(val_loss), accuracy,
        class_correct, class_total, confusion,
    )


def _eval_norm_state(layers, x_val, val_labels):
    """_eval_norm_stats 的二元 (val_loss, accuracy) 视图，供 resumeval 复用；
    逐类统计与混淆矩阵的语义见 _eval_norm_stats。"""
    val_loss, accuracy, _, _, _ = _eval_norm_stats(
        layers, x_val, val_labels
    )
    return val_loss, accuracy


def _load_resumeval_checkpoint(data, train_digest, val_digest):
    """加载并严格校验 resumeval 产物，返回状态 dict。

    data 必须是 bytes；其 UTF-8 JSON 顶层键须依次为 version、train_sha、
    val_sha、epoch、model、rng、metrics：version 恰为 int 2；train_sha/
    val_sha 均为 64 位小写十六进制 str 且分别与实参摘要逐字符相同；
    epoch 为非负 int；model 结构与 dump_data_checkpoint 完全一致（解析见
    _parse_norm_checkpoint_state）；rng 为 [0,2^32-1] 内 int；metrics 键
    依次为 loss、val_loss、accuracy，均为长度等于 epoch 的 list，元素全为
    有限 float（拒绝 bool/int 与 NaN/inf）。重复/缺失/额外/错序键、
    JSON 常量、非规范 hex、摘要不符或类型/形状/范围错分别抛
    UnicodeDecodeError/ValueError/TypeError。
    """
    if not isinstance(data, bytes):
        raise TypeError(
            "data 必须是 bytes，得到 %s" % type(data).__name__
        )
    if not isinstance(train_digest, str) or not isinstance(val_digest, str):
        raise TypeError("数据摘要必须是 str")
    text = data.decode("utf-8")
    _reject_json_constants(data)
    doc = json.loads(
        text, object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _RESUMEVAL_TOP_KEYS:
        raise ValueError(
            "产物顶层键必须依次为 version、train_sha、val_sha、epoch、"
            "model、rng、metrics"
        )

    version = doc["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(
            "version 必须是 int，得到 %s" % type(version).__name__
        )
    if version != _RESUMEVAL_VERSION:
        raise ValueError(
            "version 必须为 %d，得到 %r"
            % (_RESUMEVAL_VERSION, version)
        )

    train_sha = doc["train_sha"]
    val_sha = doc["val_sha"]
    for name, digest in (
        ("train_sha", train_sha), ("val_sha", val_sha)
    ):
        if not isinstance(digest, str):
            raise TypeError(
                "%s 必须是 JSON 字符串，得到 %s"
                % (name, type(digest).__name__)
            )
        if _SHA256_HEX_RE.match(digest) is None:
            raise ValueError(
                "%s 必须是 64 位小写十六进制 SHA-256 摘要：%r"
                % (name, digest)
            )
    if train_sha != train_digest:
        raise ValueError("产物 train_sha 与 TRAIN 摘要不符")
    if val_sha != val_digest:
        raise ValueError("产物 val_sha 与 VAL 摘要不符")

    # epoch/model 的校验与 norm 检查点共有字段完全一致；本产物的 rng
    # 即检查点的 dropout_state，以同构子文档复用同一套严格解析。
    rng = doc["rng"]
    if isinstance(rng, bool) or not isinstance(rng, int):
        raise TypeError(
            "rng 必须是 int，得到 %s" % type(rng).__name__
        )
    if rng < 0 or rng > 0xFFFFFFFF:
        raise ValueError("rng 必须在 [0, 2^32-1] 内")
    sub_doc = {
        "epoch": doc["epoch"],
        "model": doc["model"],
        "dropout_state": rng,
    }
    state = _parse_norm_checkpoint_state(sub_doc)

    metrics = doc["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError(
            "metrics 必须是 JSON 对象，得到 %s"
            % type(metrics).__name__
        )
    if list(metrics.keys()) != _RESUMEVAL_METRICS_KEYS:
        raise ValueError(
            "metrics 的键必须依次为 loss、val_loss、accuracy"
        )
    total = state["epoch"]
    for name in _RESUMEVAL_METRICS_KEYS:
        series = metrics[name]
        _require_list(series, name)
        if len(series) != total:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符" % (name, len(series), total)
            )
        for v in series:
            _check_metrics_float(v, name)
    state["losses"] = [float(v) for v in metrics["loss"]]
    state["val_losses"] = [float(v) for v in metrics["val_loss"]]
    state["accuracies"] = [float(v) for v in metrics["accuracy"]]
    return state


def _dump_resumeval_checkpoint(
    layers, epoch, train_digest, val_digest, losses, val_losses, accuracies
):
    """序列化 resumeval 产物，返回紧凑 UTF-8 JSON bytes（末尾 LF）。

    顶层键依次为 version（int 2）、train_sha/val_sha（64 位小写 hex）、
    epoch（非负 int）、model（与 dump_data_checkpoint 同构的 hex 张量，
    校验同 _validate_norm_checkpoint_layers）、rng（Dropout 当前 LCG
    状态，[0,2^32-1] 内 int）、metrics（键依次 loss、val_loss、accuracy，
    均为长度 epoch 的有限 float list，固定 12 位小数、负零归零）。同一
    状态重复调用逐字节相同。
    """
    _validate_norm_checkpoint_layers(layers)
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(
            "epoch 必须是 int（拒绝 bool），得到 %s"
            % type(epoch).__name__
        )
    if epoch < 0:
        raise ValueError("epoch 必须为非负整数")
    for name, digest in (
        ("train_sha256", train_digest), ("val_sha256", val_digest)
    ):
        if not isinstance(digest, str):
            raise TypeError(
                "%s 必须是 str，得到 %s" % (name, type(digest).__name__)
            )
        if _SHA256_HEX_RE.match(digest) is None:
            raise ValueError(
                "%s 必须是 64 位小写十六进制 SHA-256 摘要：%r"
                % (name, digest)
            )
    for series_name, series in (
        ("loss", losses), ("val_loss", val_losses),
        ("accuracy", accuracies),
    ):
        _require_list(series, series_name)
        if len(series) != epoch:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符"
                % (series_name, len(series), epoch)
            )
        for v in series:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    "%s 的元素必须为 float，得到 %s"
                    % (series_name, type(v).__name__)
                )
            if not math.isfinite(v):
                raise ValueError(
                    "%s 含有非有限值（NaN/inf）" % series_name
                )

    conv, bn, dropout, _pool, _flatten, linear, _loss = layers
    parts = []
    parts.append('"version":' + str(int(_RESUMEVAL_VERSION)))
    parts.append('"train_sha":' + json.dumps(train_digest))
    parts.append('"val_sha":' + json.dumps(val_digest))
    parts.append('"epoch":' + str(int(epoch)))
    parts.append(_dump_checkpoint_model_text(conv, bn, linear))
    parts.append('"rng":' + str(int(dropout._s)))
    metrics = {
        "loss": [float(v) for v in losses],
        "val_loss": [float(v) for v in val_losses],
        "accuracy": [float(v) for v in accuracies],
    }
    parts.append('"metrics":' + _dump_compact(metrics))
    text = "{" + ",".join(parts) + "}\n"
    return text.encode("utf-8")


def _cmd_resumeval(train_path, val_path, input_path, epochs_text, output_path):
    """resumeval 子命令主体；成功 0、契约/阈值/路径/I-O 失败 1 且不改 OUTPUT。

    TRAIN、VAL 分别按 fitdata 契约（_parse_fitdata）独立解析，身份为各自
    原始字节的 SHA-256 小写 hex；TRAIN、VAL、INPUT、OUTPUT 的文件路径必须
    互异（INPUT 为 "-" 时不参与比较，OUTPUT 不得与 TRAIN/VAL 相同）。
    INPUT 为 "-" 时以 fitnorm 初值新建训练态七层、start=0、历史为空；否则
    读取其全部字节交 _load_resumeval_checkpoint 加载，校验两份摘要一致，
    以产物 epoch 为 start 并继承其 loss/val_loss/accuracy 历史与 Dropout
    随机状态。

    EPOCHS 沿用 resumedata 的 "0"/无前导零正整数契约。追加的每一轮：先在
    训练态记录该轮更新前的全批 SoftmaxCrossEntropy 批均 loss
    （train_norm epochs=1 返回值），更新后再以 BN 保存统计、Dropout 推理
    态在 VAL 全批上算批均 val_loss 与 accuracy（正确数/N，最大 logit、
    并列取较小类别）；评估经 _norm_forward 另建网络，不修改训练状态。
    lr=0.1。当总轮数 >=20 时要求历史 loss 末项严格小于首项且 accuracy 末项
    恰为 1.0，否则失败。EPOCHS=0 时状态与历史逐位不变。

    OUTPUT 原子写出 _dump_resumeval_checkpoint 的字节；失败时标准输出为空
    且 OUTPUT 原样保留。
    """
    try:
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(train_path):
            raise ValueError("OUTPUT 与 TRAIN 不能是同一路径")
        if out_abs == os.path.abspath(val_path):
            raise ValueError("OUTPUT 与 VAL 不能是同一路径")
        if os.path.abspath(train_path) == os.path.abspath(val_path):
            raise ValueError("TRAIN 与 VAL 不能是同一路径")
        if input_path != "-":
            if os.path.abspath(input_path) == os.path.abspath(train_path):
                raise ValueError("INPUT 与 TRAIN 不能是同一路径")
            if os.path.abspath(input_path) == os.path.abspath(val_path):
                raise ValueError("INPUT 与 VAL 不能是同一路径")
            if out_abs == os.path.abspath(input_path):
                raise ValueError("INPUT 与 OUTPUT 不能是同一路径")
        epochs = _parse_resume_epochs(epochs_text)

        with open(train_path, "rb") as f:
            train_raw = f.read()
        train_digest = hashlib.sha256(train_raw).hexdigest()
        x, labels = _parse_fitdata(train_raw)
        with open(val_path, "rb") as f:
            val_raw = f.read()
        val_digest = hashlib.sha256(val_raw).hexdigest()
        x_val, val_labels = _parse_fitdata(val_raw)

        if input_path == "-":
            layers = _build_norm_layers()
            start = 0
            losses = []
            val_losses = []
            accuracies = []
        else:
            with open(input_path, "rb") as f:
                checkpoint_raw = f.read()
            state = _load_resumeval_checkpoint(
                checkpoint_raw, train_digest, val_digest
            )
            layers, start = _layers_from_checkpoint_state(state)
            losses = state["losses"]
            val_losses = state["val_losses"]
            accuracies = state["accuracies"]

        # 逐轮追加：每轮 train_norm(epochs=1) 与连续 train_norm(epochs=E)
        # 逐位等价（末轮 BN 统计刷新与 Dropout 推进完全一致）；评估在另建
        # 的推理态网络上进行，训练态参数、缓存、统计与随机状态均不受影响。
        for _ in range(epochs):
            (step_loss,) = train_norm(
                layers, x, labels, epochs=1, lr=_NORM_LR
            )
            step_loss = float(step_loss)
            if not math.isfinite(step_loss):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            val_loss, accuracy = _eval_norm_state(layers, x_val, val_labels)
            losses.append(step_loss)
            val_losses.append(val_loss)
            accuracies.append(accuracy)

        total = start + epochs
        if total >= 20:
            if not losses[-1] < losses[0]:
                raise ValueError("末次 loss 未小于首次 loss")
            if accuracies[-1] != 1.0:
                raise ValueError("末次 validation accuracy 不为 1.0")

        payload = _dump_resumeval_checkpoint(
            layers, total, train_digest, val_digest,
            losses, val_losses, accuracies,
        )
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


def _resumevalstats_int_pair(value, name):
    """校验长度恰为 2 的非负 int list（拒绝 bool/嵌套与负值），原样返回。"""
    _require_list(value, name)
    if len(value) != 2:
        raise ValueError("%s 必须是长度 2 的 list" % name)
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                "%s[%d] 必须是非负 int（拒绝 bool），得到 %s"
                % (name, i, type(v).__name__)
            )
        if v < 0:
            raise ValueError("%s[%d] 必须非负" % (name, i))
    return value


def _load_resumevalstats_checkpoint(data, train_digest, val_digest, n_val):
    """加载并严格校验 resumevalstats 产物，返回状态 dict。

    顶层契约同 resumeval（键依次 version、train_sha、val_sha、epoch、
    model、rng、metrics），唯 version 恰为 int 3、两份摘要分别与实参相符。
    metrics 键须依次为 loss、val_loss、accuracy、class_correct、
    class_total、confusion：前三者为长度等于 epoch 的有限 float list
    （拒绝 bool/int 与 NaN/inf）；class_correct、class_total 每轮为非负
    int[2]，confusion 每轮为非负 int[2][2]，全部 int 拒绝 bool。逐轮
    关系校验：class_total[c] 等于混淆矩阵第 c 行之和、class_correct[c]
    等于对角项 confusion[c][c]、矩阵总和等于 VAL 样本数 n_val、accuracy
    恰等于对角和/n_val（空类 total、correct 均为 0）。重复/缺失/额外/错序
    键、JSON 常量、非规范 hex、摘要不符、历史长度或关系错、类别越界（负值）
    或类型/形状/范围错分别抛 UnicodeDecodeError/ValueError/TypeError。
    """
    if not isinstance(data, bytes):
        raise TypeError(
            "data 必须是 bytes，得到 %s" % type(data).__name__
        )
    if not isinstance(train_digest, str) or not isinstance(val_digest, str):
        raise TypeError("数据摘要必须是 str")
    text = data.decode("utf-8")
    _reject_json_constants(data)
    doc = json.loads(
        text, object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(doc, dict):
        raise TypeError("产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _RESUMEVALSTATS_TOP_KEYS:
        raise ValueError(
            "产物顶层键必须依次为 version、train_sha、val_sha、epoch、"
            "model、rng、metrics"
        )

    version = doc["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(
            "version 必须是 int，得到 %s" % type(version).__name__
        )
    if version != _RESUMEVALSTATS_VERSION:
        raise ValueError(
            "version 必须为 %d，得到 %r"
            % (_RESUMEVALSTATS_VERSION, version)
        )

    train_sha = doc["train_sha"]
    val_sha = doc["val_sha"]
    for name, digest in (
        ("train_sha", train_sha), ("val_sha", val_sha)
    ):
        if not isinstance(digest, str):
            raise TypeError(
                "%s 必须是 JSON 字符串，得到 %s"
                % (name, type(digest).__name__)
            )
        if _SHA256_HEX_RE.match(digest) is None:
            raise ValueError(
                "%s 必须是 64 位小写十六进制 SHA-256 摘要：%r"
                % (name, digest)
            )
    if train_sha != train_digest:
        raise ValueError("产物 train_sha 与 TRAIN 摘要不符")
    if val_sha != val_digest:
        raise ValueError("产物 val_sha 与 VAL 摘要不符")

    # epoch/model 的校验与 norm 检查点共有字段完全一致；本产物的 rng
    # 即检查点的 dropout_state，以同构子文档复用同一套严格解析。
    rng = doc["rng"]
    if isinstance(rng, bool) or not isinstance(rng, int):
        raise TypeError(
            "rng 必须是 int，得到 %s" % type(rng).__name__
        )
    if rng < 0 or rng > 0xFFFFFFFF:
        raise ValueError("rng 必须在 [0, 2^32-1] 内")
    sub_doc = {
        "epoch": doc["epoch"],
        "model": doc["model"],
        "dropout_state": rng,
    }
    state = _parse_norm_checkpoint_state(sub_doc)

    metrics = doc["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError(
            "metrics 必须是 JSON 对象，得到 %s"
            % type(metrics).__name__
        )
    if list(metrics.keys()) != _RESUMEVALSTATS_METRICS_KEYS:
        raise ValueError(
            "metrics 的键必须依次为 loss、val_loss、accuracy、"
            "class_correct、class_total、confusion"
        )
    total_epochs = state["epoch"]
    for name in _RESUMEVAL_FLOAT_KEYS:
        series = metrics[name]
        _require_list(series, name)
        if len(series) != total_epochs:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符"
                % (name, len(series), total_epochs)
            )
        for v in series:
            _check_metrics_float(v, name)

    correct_hist = []
    total_hist = []
    confusion_hist = []
    for name in (
        "class_correct", "class_total", "confusion",
    ):
        series = metrics[name]
        _require_list(series, name)
        if len(series) != total_epochs:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符"
                % (name, len(series), total_epochs)
            )
    for e in range(total_epochs):
        correct = _resumevalstats_int_pair(
            metrics["class_correct"][e], "class_correct"
        )
        per_total = _resumevalstats_int_pair(
            metrics["class_total"][e], "class_total"
        )
        matrix_node = metrics["confusion"][e]
        _require_list(matrix_node, "confusion")
        if len(matrix_node) != 2:
            raise ValueError("confusion 每轮必须是 [2][2] 的 list")
        matrix = [
            _resumevalstats_int_pair(matrix_node[c], "confusion[%d]" % c)
            for c in range(2)
        ]
        # 行真实类、列预测类：total 为行和，correct 为对角。
        for c in range(2):
            row_sum = matrix[c][0] + matrix[c][1]
            if per_total[c] != row_sum:
                raise ValueError(
                    "第 %d 轮 class_total[%d]=%d 与混淆矩阵行和 %d 不符"
                    % (e, c, per_total[c], row_sum)
                )
            if correct[c] != matrix[c][c]:
                raise ValueError(
                    "第 %d 轮 class_correct[%d]=%d 与对角项 %d 不符"
                    % (e, c, correct[c], matrix[c][c])
                )
        matrix_sum = sum(sum(row) for row in matrix)
        if matrix_sum != n_val:
            raise ValueError(
                "第 %d 轮混淆矩阵总和 %d 与 VAL 样本数 %d 不符"
                % (e, matrix_sum, n_val)
            )
        diag = matrix[0][0] + matrix[1][1]
        # accuracy 以固定 12 位小数序列化，关系校验须在同一精度下比较：
        # 载入值与对角和/样本数各自的 12 位渲染必须一致，否则 1/3 这类
        # 自产物反而无法被续训载入。
        expected_accuracy = float(metrics["accuracy"][e])
        if _fmt_float(expected_accuracy) != _fmt_float(diag / n_val):
            raise ValueError(
                "第 %d 轮 accuracy %r 与对角和/样本数 %r 不符"
                % (e, expected_accuracy, diag / n_val)
            )
        correct_hist.append([correct[0], correct[1]])
        total_hist.append([per_total[0], per_total[1]])
        confusion_hist.append([
            [matrix[0][0], matrix[0][1]],
            [matrix[1][0], matrix[1][1]],
        ])

    state["losses"] = [float(v) for v in metrics["loss"]]
    state["val_losses"] = [float(v) for v in metrics["val_loss"]]
    state["accuracies"] = [float(v) for v in metrics["accuracy"]]
    state["class_correct"] = correct_hist
    state["class_total"] = total_hist
    state["confusion"] = confusion_hist
    return state


def _dump_resumevalstats_checkpoint(
    layers, epoch, train_digest, val_digest,
    losses, val_losses, accuracies,
    class_correct_hist, class_total_hist, confusion_hist,
):
    """序列化 resumevalstats 产物，返回紧凑 UTF-8 JSON bytes（末尾 LF）。

    顶层键依次为 version（int 3）、train_sha/val_sha（64 位小写 hex）、
    epoch（非负 int）、model（与 dump_data_checkpoint 同构的 hex 张量，
    校验同 _validate_norm_checkpoint_layers）、rng（Dropout 当前 LCG
    状态，[0,2^32-1] 内 int）、metrics（键依次 loss、val_loss、accuracy、
    class_correct、class_total、confusion）。前三者为长度 epoch 的有限
    float list（固定 12 位小数、负零归零）；后三者每轮分别为非负 int[2]、
    int[2]、int[2][2]（拒绝 bool）。同一状态重复调用逐字节相同。
    """
    _validate_norm_checkpoint_layers(layers)
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(
            "epoch 必须是 int（拒绝 bool），得到 %s"
            % type(epoch).__name__
        )
    if epoch < 0:
        raise ValueError("epoch 必须为非负整数")
    for name, digest in (
        ("train_sha256", train_digest), ("val_sha256", val_digest)
    ):
        if not isinstance(digest, str):
            raise TypeError(
                "%s 必须是 str，得到 %s" % (name, type(digest).__name__)
            )
        if _SHA256_HEX_RE.match(digest) is None:
            raise ValueError(
                "%s 必须是 64 位小写十六进制 SHA-256 摘要：%r"
                % (name, digest)
            )
    for series_name, series in (
        ("loss", losses), ("val_loss", val_losses),
        ("accuracy", accuracies),
    ):
        _require_list(series, series_name)
        if len(series) != epoch:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符"
                % (series_name, len(series), epoch)
            )
        for v in series:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    "%s 的元素必须为 float，得到 %s"
                    % (series_name, type(v).__name__)
                )
            if not math.isfinite(v):
                raise ValueError(
                    "%s 含有非有限值（NaN/inf）" % series_name
                )
    for series_name, series in (
        ("class_correct", class_correct_hist),
        ("class_total", class_total_hist),
        ("confusion", confusion_hist),
    ):
        _require_list(series, series_name)
        if len(series) != epoch:
            raise ValueError(
                "%s 长度 %d 与 epoch %d 不符"
                % (series_name, len(series), epoch)
            )
        for e, node in enumerate(series):
            if series_name == "confusion":
                _require_list(node, "%s[%d]" % (series_name, e))
                if len(node) != 2:
                    raise ValueError(
                        "confusion[%d] 必须是 [2][2] 的 list" % e
                    )
                for c in range(2):
                    _resumevalstats_int_pair(
                        node[c], "%s[%d][%d]" % (series_name, e, c)
                    )
            else:
                _resumevalstats_int_pair(
                    node, "%s[%d]" % (series_name, e)
                )

    conv, bn, dropout, _pool, _flatten, linear, _loss = layers
    parts = []
    parts.append('"version":' + str(int(_RESUMEVALSTATS_VERSION)))
    parts.append('"train_sha":' + json.dumps(train_digest))
    parts.append('"val_sha":' + json.dumps(val_digest))
    parts.append('"epoch":' + str(int(epoch)))
    parts.append(_dump_checkpoint_model_text(conv, bn, linear))
    parts.append('"rng":' + str(int(dropout._s)))
    metrics = {
        "loss": [float(v) for v in losses],
        "val_loss": [float(v) for v in val_losses],
        "accuracy": [float(v) for v in accuracies],
        "class_correct": class_correct_hist,
        "class_total": class_total_hist,
        "confusion": confusion_hist,
    }
    parts.append('"metrics":' + _dump_compact(metrics))
    text = "{" + ",".join(parts) + "}\n"
    return text.encode("utf-8")


def _cmd_resumevalstats(
    train_path, val_path, input_path, epochs_text, output_path
):
    """resumevalstats 子命令主体；成功 0、契约/阈值/路径/I-O 失败 1 且不改
    OUTPUT。

    训练、校验、阈值、路径互异与原子写契约完全沿用 resumeval
    （_cmd_resumeval），区别仅在：INPUT 为 "-" 时从初值与空历史开始，否则
    载入同类产物（version 为 int 3 的 resumevalstats 产物，拒绝 resumeval
    的 version 2 等异质产物）；每轮校验另记录 class_correct/class_total/
    confusion（语义见 _eval_norm_stats 与
    _load_resumevalstats_checkpoint），OUTPUT 为
    _dump_resumevalstats_checkpoint 的字节。失败时标准输出为空且 OUTPUT
    原样保留。
    """
    try:
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(train_path):
            raise ValueError("OUTPUT 与 TRAIN 不能是同一路径")
        if out_abs == os.path.abspath(val_path):
            raise ValueError("OUTPUT 与 VAL 不能是同一路径")
        if os.path.abspath(train_path) == os.path.abspath(val_path):
            raise ValueError("TRAIN 与 VAL 不能是同一路径")
        if input_path != "-":
            if os.path.abspath(input_path) == os.path.abspath(train_path):
                raise ValueError("INPUT 与 TRAIN 不能是同一路径")
            if os.path.abspath(input_path) == os.path.abspath(val_path):
                raise ValueError("INPUT 与 VAL 不能是同一路径")
            if out_abs == os.path.abspath(input_path):
                raise ValueError("INPUT 与 OUTPUT 不能是同一路径")
        epochs = _parse_resume_epochs(epochs_text)

        with open(train_path, "rb") as f:
            train_raw = f.read()
        train_digest = hashlib.sha256(train_raw).hexdigest()
        x, labels = _parse_fitdata(train_raw)
        with open(val_path, "rb") as f:
            val_raw = f.read()
        val_digest = hashlib.sha256(val_raw).hexdigest()
        x_val, val_labels = _parse_fitdata(val_raw)
        n_val = len(val_labels)

        if input_path == "-":
            layers = _build_norm_layers()
            start = 0
            losses = []
            val_losses = []
            accuracies = []
            class_correct_hist = []
            class_total_hist = []
            confusion_hist = []
        else:
            with open(input_path, "rb") as f:
                checkpoint_raw = f.read()
            state = _load_resumevalstats_checkpoint(
                checkpoint_raw, train_digest, val_digest, n_val
            )
            layers, start = _layers_from_checkpoint_state(state)
            losses = state["losses"]
            val_losses = state["val_losses"]
            accuracies = state["accuracies"]
            class_correct_hist = state["class_correct"]
            class_total_hist = state["class_total"]
            confusion_hist = state["confusion"]

        # 逐轮追加：每轮 train_norm(epochs=1) 与连续 train_norm(epochs=E)
        # 逐位等价（末轮 BN 统计刷新与 Dropout 推进完全一致）；评估在另建
        # 的推理态网络上进行，训练态参数、缓存、统计与随机状态均不受影响。
        for _ in range(epochs):
            (step_loss,) = train_norm(
                layers, x, labels, epochs=1, lr=_NORM_LR
            )
            step_loss = float(step_loss)
            if not math.isfinite(step_loss):
                raise ValueError("训练计算产生非有限值（NaN/inf）")
            (
                val_loss, accuracy,
                class_correct, class_total, confusion,
            ) = _eval_norm_stats(layers, x_val, val_labels)
            losses.append(step_loss)
            val_losses.append(val_loss)
            accuracies.append(accuracy)
            class_correct_hist.append(class_correct)
            class_total_hist.append(class_total)
            confusion_hist.append(confusion)

        total = start + epochs
        if total >= 20:
            if not losses[-1] < losses[0]:
                raise ValueError("末次 loss 未小于首次 loss")
            if accuracies[-1] != 1.0:
                raise ValueError("末次 validation accuracy 不为 1.0")

        payload = _dump_resumevalstats_checkpoint(
            layers, total, train_digest, val_digest,
            losses, val_losses, accuracies,
            class_correct_hist, class_total_hist, confusion_hist,
        )
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行末轮逐类报告：
# python convnet.py valreport STATS VAL OUTPUT
# ---------------------------------------------------------------------------


def _cmd_valreport(stats_path, val_path, output_path):
    """valreport 子命令主体；成功 0、契约/路径/I-O 失败 1 且不改 OUTPUT。

    STATS 须为 epoch>=1 的 resumevalstats 产物（version 3），沿用
    _load_resumevalstats_checkpoint 的严格加载契约（train_sha 仅校验格式，
    无 TRAIN 可对照）；VAL 沿用 fitdata 契约（_parse_fitdata），其原始字节
    的 SHA-256 小写 hex 须与产物 val_sha 逐字符相同。STATS、VAL、OUTPUT
    三路径必须两两不同。

    以产物保存的模型与 BN 统计、关闭 Dropout（_eval_norm_stats 经
    _norm_forward 另建推理态七层，不修改任何状态）在 VAL 全批上重算末轮
    混淆矩阵 M（逐样本取最大有限 logit、仅相等取类别 0，行真实类、列
    预测类），M 须与历史末项一致，否则失败。类别序固定 0、1：类 c 的
    precision=M[c][c]/第 c 列和、recall=M[c][c]/第 c 行和，零分母取
    0.0；F1 在 precision+recall 为 0 时取 0.0，否则取 2pr/(p+r)；
    balanced_accuracy 为两类 recall 的均值，macro_f1 为两类 F1 的均值。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），键依次为 epoch（int）、
    sample_count（int）、confusion（非负 int[2][2]）、precision/recall/f1
    （各为有限 float[2]）、balanced_accuracy/macro_f1（有限 float）；
    float 固定 12 位小数、负零归零。输入非法、摘要或统计不符、推理非
    有限或 I/O 失败均返回 1，标准输出为空且 OUTPUT 原样保留；同一输入
    重复运行产物逐字节相同。
    """
    try:
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(stats_path):
            raise ValueError("OUTPUT 与 STATS 不能是同一路径")
        if out_abs == os.path.abspath(val_path):
            raise ValueError("OUTPUT 与 VAL 不能是同一路径")
        if os.path.abspath(stats_path) == os.path.abspath(val_path):
            raise ValueError("STATS 与 VAL 不能是同一路径")

        with open(val_path, "rb") as f:
            val_raw = f.read()
        val_digest = hashlib.sha256(val_raw).hexdigest()
        x_val, val_labels = _parse_fitdata(val_raw)
        n_val = len(val_labels)

        with open(stats_path, "rb") as f:
            stats_raw = f.read()
        # valreport 不读 TRAIN：以产物自身的 train_sha 为期望摘要（仅其
        # 格式受严格校验），val_sha 则与 VAL 实际摘要逐字符比对。
        pre_doc = json.loads(stats_raw.decode("utf-8"))
        train_sha = (
            pre_doc.get("train_sha") if isinstance(pre_doc, dict) else None
        )
        state = _load_resumevalstats_checkpoint(
            stats_raw, train_sha, val_digest, n_val
        )
        epoch = state["epoch"]
        if epoch < 1:
            raise ValueError("STATS 产物 epoch 必须 >= 1")

        layers, _start = _layers_from_checkpoint_state(state)
        (
            _val_loss, _accuracy,
            _class_correct, _class_total, confusion,
        ) = _eval_norm_stats(layers, x_val, val_labels)
        if confusion != state["confusion"][-1]:
            raise ValueError("重算混淆矩阵与产物历史末项不符")

        precision = []
        recall = []
        f1 = []
        for c in range(2):
            col_sum = confusion[0][c] + confusion[1][c]
            row_sum = confusion[c][0] + confusion[c][1]
            p = confusion[c][c] / col_sum if col_sum else 0.0
            r = confusion[c][c] / row_sum if row_sum else 0.0
            precision.append(p)
            recall.append(r)
            f1.append(0.0 if p + r == 0.0 else 2.0 * p * r / (p + r))
        balanced_accuracy = (recall[0] + recall[1]) / 2
        macro_f1 = (f1[0] + f1[1]) / 2

        report = {
            "epoch": epoch,
            "sample_count": n_val,
            "confusion": [
                [confusion[0][0], confusion[0][1]],
                [confusion[1][0], confusion[1][1]],
            ],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        }
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行末轮阈值闸门：
# python convnet.py valgate STATS VAL CONFIG OUTPUT
# ---------------------------------------------------------------------------

_VALGATE_CONFIG_KEYS = ["balanced_accuracy", "macro_f1"]


def _load_valgate_config(raw):
    """按 valgate 契约从 CONFIG 原始字节解析并严格校验，返回阈值 dict。

    raw 不得含 UTF-8 BOM 或任何 JSON 空白（空格、制表、换行、回车），
    须为紧凑 UTF-8 JSON 对象，键仅依次为 balanced_accuracy、macro_f1，
    重复/缺失/额外/错序键一律非法；值须为 [0,1] 内有限 float（拒绝
    bool/int 与 NaN/Infinity 常量）。非法 UTF-8/JSON 或契约不符分别抛
    UnicodeDecodeError/ValueError/TypeError。
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("CONFIG 不得含 UTF-8 BOM")
    text = raw.decode("utf-8")
    if any(ch in text for ch in " \t\n\r"):
        raise ValueError("CONFIG 不得含空白，须为紧凑 JSON")
    _reject_json_constants(raw)
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("CONFIG 顶层必须是 JSON 对象")
    if list(doc.keys()) != _VALGATE_CONFIG_KEYS:
        raise ValueError(
            "CONFIG 键必须依次为 balanced_accuracy、macro_f1"
        )
    thresholds = {}
    for name in _VALGATE_CONFIG_KEYS:
        value = doc[name]
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(
                "%s 必须是 float（拒绝 bool/int），得到 %s"
                % (name, type(value).__name__)
            )
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError("%s 必须是 [0,1] 内的有限 float" % name)
        thresholds[name] = value
    return thresholds


def _valgate_compute(stats_path, val_path, config_path):
    """按 valgate 契约读取 STATS/VAL/CONFIG 并重算末轮闸门指标。

    STATS/VAL 的加载、摘要绑定、epoch>=1 校验、以保存模型与 BN 统计
    关闭 Dropout 重算末轮混淆矩阵并与历史末项核对，全部沿用
    _cmd_valreport；CONFIG 的严格契约见 _load_valgate_config。由重算
    混淆矩阵按 valreport 公式取得未舍入 balanced_accuracy、macro_f1，
    逐项以实测值 >= 阈值判定，pass 为两项之与。

    返回 dict（键依次为 epoch、sample_count、balanced_accuracy、
    macro_f1、min_balanced_accuracy、min_macro_f1、
    balanced_accuracy_pass、macro_f1_pass、pass）：前两项 int、
    中间四项未舍入 float、末三项 bool。路径不存在/不可读、UTF-8/JSON
    非法、契约或摘要不符、推理非有限分别抛 OSError/UnicodeDecodeError/
    ValueError/TypeError。
    """
    with open(val_path, "rb") as f:
        val_raw = f.read()
    val_digest = hashlib.sha256(val_raw).hexdigest()
    x_val, val_labels = _parse_fitdata(val_raw)
    n_val = len(val_labels)

    with open(stats_path, "rb") as f:
        stats_raw = f.read()
    # 同 valreport：不读 TRAIN，以产物自身的 train_sha 为期望摘要
    # （仅格式受严格校验），val_sha 与 VAL 实际摘要逐字符比对。
    pre_doc = json.loads(stats_raw.decode("utf-8"))
    train_sha = (
        pre_doc.get("train_sha") if isinstance(pre_doc, dict) else None
    )
    state = _load_resumevalstats_checkpoint(
        stats_raw, train_sha, val_digest, n_val
    )
    epoch = state["epoch"]
    if epoch < 1:
        raise ValueError("STATS 产物 epoch 必须 >= 1")

    with open(config_path, "rb") as f:
        config_raw = f.read()
    thresholds = _load_valgate_config(config_raw)

    layers, _start = _layers_from_checkpoint_state(state)
    (
        _val_loss, _accuracy,
        _class_correct, _class_total, confusion,
    ) = _eval_norm_stats(layers, x_val, val_labels)
    if confusion != state["confusion"][-1]:
        raise ValueError("重算混淆矩阵与产物历史末项不符")

    # 公式同 _cmd_valreport：使用未舍入实测值，仅写出时格式化为 12 位。
    recall = []
    f1 = []
    for c in range(2):
        col_sum = confusion[0][c] + confusion[1][c]
        row_sum = confusion[c][0] + confusion[c][1]
        p = confusion[c][c] / col_sum if col_sum else 0.0
        r = confusion[c][c] / row_sum if row_sum else 0.0
        recall.append(r)
        f1.append(0.0 if p + r == 0.0 else 2.0 * p * r / (p + r))
    balanced_accuracy = (recall[0] + recall[1]) / 2
    macro_f1 = (f1[0] + f1[1]) / 2

    ba_pass = balanced_accuracy >= thresholds["balanced_accuracy"]
    f1_pass = macro_f1 >= thresholds["macro_f1"]
    return {
        "epoch": epoch,
        "sample_count": n_val,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "min_balanced_accuracy": thresholds["balanced_accuracy"],
        "min_macro_f1": thresholds["macro_f1"],
        "balanced_accuracy_pass": bool(ba_pass),
        "macro_f1_pass": bool(f1_pass),
        "pass": bool(ba_pass and f1_pass),
    }


def _cmd_valgate(stats_path, val_path, config_path, output_path):
    """valgate 子命令主体；达标退出 0、合法未达标退出 3、契约/路径/I-O
    失败退出 1 且不改 OUTPUT。

    STATS、VAL、CONFIG、OUTPUT 四路径必须两两不同。校验、重算与未舍入
    比较契约见 _valgate_compute。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），键依次为 epoch（int）、
    sample_count（int）、balanced_accuracy/macro_f1/min_balanced_accuracy/
    min_macro_f1（有限 float，固定 12 位小数、负零归零）、
    balanced_accuracy_pass/macro_f1_pass/pass（bool）；同一输入重复运行
    逐字节相同。输入非法、摘要或统计不符、推理非有限或 I/O 失败均返回
    1，标准输出为空且 OUTPUT 原样保留。
    """
    overall = False
    try:
        out_abs = os.path.abspath(output_path)
        if out_abs == os.path.abspath(stats_path):
            raise ValueError("OUTPUT 与 STATS 不能是同一路径")
        if out_abs == os.path.abspath(val_path):
            raise ValueError("OUTPUT 与 VAL 不能是同一路径")
        if out_abs == os.path.abspath(config_path):
            raise ValueError("OUTPUT 与 CONFIG 不能是同一路径")
        if os.path.abspath(stats_path) == os.path.abspath(val_path):
            raise ValueError("STATS 与 VAL 不能是同一路径")
        if os.path.abspath(stats_path) == os.path.abspath(config_path):
            raise ValueError("STATS 与 CONFIG 不能是同一路径")
        if os.path.abspath(val_path) == os.path.abspath(config_path):
            raise ValueError("VAL 与 CONFIG 不能是同一路径")

        report = _valgate_compute(stats_path, val_path, config_path)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(output_path, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行批量末轮阈值闸门：
# python convnet.py valgatebatch MANIFEST OUTPUT
# ---------------------------------------------------------------------------

_VALGATEBATCH_MANIFEST_KEYS = ["gates"]
_VALGATEBATCH_ITEM_KEYS = ["name", "stats", "val", "config"]


def _load_valgatebatch_manifest(raw, manifest_dir):
    """按 valgatebatch 契约从 MANIFEST 原始字节解析并严格校验。

    raw 不得含 UTF-8 BOM 或字符串字面量之外的任何 JSON 空白（空格、
    制表、换行、回车），须为紧凑 UTF-8 JSON 对象；唯一键 gates 为长度
    >= 2 的 list，重复/缺失/额外/错序键（含各闸门对象内部）一律拒绝
    （object_pairs_hook 逐对象查重）并拒绝 NaN/Infinity 常量。各项须为
    JSON 对象，键仅依次
    为 name、stats、val、config；四值均为非空 str（拒绝其他任何 JSON
    类型），且 name 全局唯一。stats/val/config 相对清单目录解析
    （绝对路径原样使用）为 abspath，跨全部闸门两两不同，否则 ValueError。
    返回按清单顺序的 (name, stats_path, val_path, config_path) 列表；
    非法 UTF-8/JSON 或契约不符分别抛 UnicodeDecodeError/ValueError/
    TypeError。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "MANIFEST 必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("MANIFEST 不得含 UTF-8 BOM")
    text = raw.decode("utf-8")
    _reject_json_whitespace(raw)
    _reject_json_constants(raw)
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("MANIFEST 顶层必须是 JSON 对象")
    if list(doc.keys()) != _VALGATEBATCH_MANIFEST_KEYS:
        raise ValueError("MANIFEST 唯一键必须为 gates")
    gates = doc["gates"]
    if not isinstance(gates, list):
        raise TypeError("gates 必须是 list，得到 %s" % type(gates).__name__)
    if len(gates) < 2:
        raise ValueError("gates 长度必须 >= 2，得到 %d" % len(gates))

    groups = []
    names = set()
    seen_paths = set()
    for idx, item in enumerate(gates):
        if not isinstance(item, dict):
            raise TypeError("gates[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _VALGATEBATCH_ITEM_KEYS:
            raise ValueError(
                "gates[%d] 的键必须依次为 name、stats、val、config" % idx
            )
        values = {}
        for key in _VALGATEBATCH_ITEM_KEYS:
            value = item[key]
            if not isinstance(value, str):
                raise TypeError(
                    "gates[%d].%s 必须是非空 str，得到 %s"
                    % (idx, key, type(value).__name__)
                )
            if len(value) == 0:
                raise ValueError("gates[%d].%s 不得为空字符串" % (idx, key))
            values[key] = value
        name = values["name"]
        if name in names:
            raise ValueError("gates 的 name 必须唯一，重复：%r" % name)
        names.add(name)

        resolved = {}
        for key in ("stats", "val", "config"):
            path = os.path.abspath(os.path.join(manifest_dir, values[key]))
            if path in seen_paths:
                raise ValueError(
                    "gates 的 stats/val/config 路径必须两两不同，"
                    "%s 重复指向 %s" % (key, path)
                )
            seen_paths.add(path)
            resolved[key] = path
        groups.append(
            (name, resolved["stats"], resolved["val"], resolved["config"])
        )
    return groups


def _cmd_valgatebatch(manifest_path, output_path):
    """valgatebatch 子命令主体；全组达标退出 0、合法但未全过退出 3、
    契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    MANIFEST 的严格契约见 _load_valgatebatch_manifest：闸门数 >= 2，
    name 唯一，清单内 stats/val/config 相对清单目录解析且跨全部闸门
    两两不同。相对 OUTPUT 以 MANIFEST 所在目录解析（绝对 OUTPUT 原样
    使用）；绝对化后 OUTPUT 另须与 MANIFEST 及所有闸门输入路径两两
    不同，任一冲突即失败退出 1 且不触碰 OUTPUT。每组完全复用
    _valgate_compute（即 valgate 对 STATS、VAL、CONFIG 的校验、混淆
    矩阵重算比对与未舍入阈值比较契约）；任一组失败即整体失败退出 1，
    OUTPUT 原样保留（所有组重算完成后才一次性原子写盘）。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    gate_count（int）、passed_count（int）、results（list）、pass
    （bool，各项 pass 之与）；results 按 gates 顺序，每项键依次为
    name（str）、epoch、sample_count（int）、balanced_accuracy、
    macro_f1、min_balanced_accuracy、min_macro_f1（有限 float，固定
    12 位小数、负零归零）、balanced_accuracy_pass、macro_f1_pass、
    pass（bool）；同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        manifest_abs = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_abs)
        # 相对 OUTPUT 以 MANIFEST 所在目录解析（绝对路径原样使用）。
        out_abs = os.path.abspath(os.path.join(manifest_dir, output_path))
        if out_abs == manifest_abs:
            raise ValueError("OUTPUT 与 MANIFEST 不能是同一路径")

        with open(manifest_path, "rb") as f:
            manifest_raw = f.read()
        groups = _load_valgatebatch_manifest(manifest_raw, manifest_dir)

        # 清单内路径两两不同已由加载器保证；绝对化后 MANIFEST、OUTPUT
        # 与全部闸门输入路径亦须两两不同。
        for _name, stats_path, val_path, config_path in groups:
            for path in (stats_path, val_path, config_path):
                if path == out_abs:
                    raise ValueError(
                        "OUTPUT 与闸门输入路径不能是同一路径：%s" % path
                    )
                if path == manifest_abs:
                    raise ValueError(
                        "MANIFEST 与闸门输入路径不能是同一路径：%s" % path
                    )

        results = []
        passed_count = 0
        for name, stats_path, val_path, config_path in groups:
            metrics = _valgate_compute(stats_path, val_path, config_path)
            if metrics["pass"]:
                passed_count += 1
            results.append({
                "name": name,
                "epoch": metrics["epoch"],
                "sample_count": metrics["sample_count"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "min_balanced_accuracy": metrics["min_balanced_accuracy"],
                "min_macro_f1": metrics["min_macro_f1"],
                "balanced_accuracy_pass": metrics["balanced_accuracy_pass"],
                "macro_f1_pass": metrics["macro_f1_pass"],
                "pass": metrics["pass"],
            })

        overall = passed_count == len(groups)
        report = {
            "gate_count": len(groups),
            "passed_count": passed_count,
            "results": results,
            "pass": bool(overall),
        }
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行批量闸门逐组回归对比：
# python convnet.py valgatebatchdiff BASELINE CURRENT OUTPUT
# ---------------------------------------------------------------------------

_VALGATEBATCHDIFF_TOP_KEYS = ["gate_count", "results", "pass"]
_VALGATEBATCHDIFF_ITEM_KEYS = ["name", "ba", "f1", "pass"]
_VALGATEBATCHDIFF_METRIC_KEYS = ["baseline", "current", "delta", "pass"]
# valgatebatch 产物的公开契约（键序即写出顺序）。
_VALGATEBATCH_TOP_KEYS = ["gate_count", "passed_count", "results", "pass"]
_VALGATEBATCH_RESULT_KEYS = [
    "name",
    "epoch",
    "sample_count",
    "balanced_accuracy",
    "macro_f1",
    "min_balanced_accuracy",
    "min_macro_f1",
    "balanced_accuracy_pass",
    "macro_f1_pass",
    "pass",
]


def _load_valgatebatch_output(raw):
    """按 valgatebatch 产物契约从原始字节严格解析一份对比输入。

    与 valgatebatch 写出的逐字节产物一致：不得含 UTF-8 BOM，须以恰好
    一个 LF（b"\\n"）结尾（拒绝 CR 与多余空行），其前正文不得含字符串
    字面量之外的任何 JSON 空白（紧凑 JSON），拒绝 NaN/Infinity 常量与
    重复键。正文顶层键须依次为 gate_count、
    passed_count、results、pass：gate_count 为 >= 2 的 int（拒绝 bool），
    passed_count 为 [0, gate_count] 内 int，pass 为 bool，results 长度恰为
    gate_count；每项键须依次为 name/epoch/sample_count/balanced_accuracy/
    macro_f1/min_balanced_accuracy/min_macro_f1/balanced_accuracy_pass/
    macro_f1_pass/pass：name 为非空 str 且在本份产物内唯一、顺序保留；
    epoch/sample_count 为 int，四个指标为有限 float，三个 pass 为 bool；
    顶层 pass 须恰为各项 pass 之与，passed_count 须恰为通过项数，
    各项 pass 须恰为 balanced_accuracy_pass 与 macro_f1_pass 之与，
    且 min_* 阈值与实测通过判定自洽（实测通过则必 >= 阈值）。非法
    UTF-8/JSON 或契约不符分别抛 UnicodeDecodeError/ValueError/TypeError。
    返回 (gate_count, [(name, balanced_accuracy, macro_f1), ...])。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "valgatebatch 产物必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("valgatebatch 产物不得含 UTF-8 BOM")
    if b"\r" in raw:
        raise ValueError("valgatebatch 产物不得含回车（CR）")
    # 生产者保证紧凑正文 + 恰好一个末尾 LF。
    if not raw.endswith(b"\n"):
        raise ValueError("valgatebatch 产物必须以恰好一个 LF 结尾")
    body = raw[:-1]
    if body.endswith(b"\n"):
        raise ValueError("valgatebatch 产物末尾仅可有一个 LF")
    _reject_json_whitespace(body)
    _reject_json_constants(body)
    text = body.decode("utf-8")
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("valgatebatch 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _VALGATEBATCH_TOP_KEYS:
        raise ValueError(
            "valgatebatch 产物顶层键必须依次为 "
            "gate_count、passed_count、results、pass"
        )

    gate_count = doc["gate_count"]
    if isinstance(gate_count, bool) or not isinstance(gate_count, int):
        raise TypeError(
            "gate_count 必须是 int，得到 %s" % type(gate_count).__name__
        )
    if gate_count < 2:
        raise ValueError("gate_count 必须 >= 2，得到 %d" % gate_count)

    passed_count = doc["passed_count"]
    if isinstance(passed_count, bool) or not isinstance(passed_count, int):
        raise TypeError(
            "passed_count 必须是 int，得到 %s"
            % type(passed_count).__name__
        )
    if not (0 <= passed_count <= gate_count):
        raise ValueError(
            "passed_count 必须在 [0, gate_count] 内，得到 %d" % passed_count
        )

    overall_pass = doc["pass"]
    if not isinstance(overall_pass, bool):
        raise TypeError(
            "顶层 pass 必须是 bool，得到 %s"
            % type(overall_pass).__name__
        )

    results = doc["results"]
    if not isinstance(results, list):
        raise TypeError(
            "results 必须是 list，得到 %s" % type(results).__name__
        )
    if len(results) != gate_count:
        raise ValueError(
            "results 长度必须等于 gate_count（%d），得到 %d"
            % (gate_count, len(results))
        )

    items = []
    names = set()
    counted_pass = 0
    all_pass = True
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError("results[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _VALGATEBATCH_RESULT_KEYS:
            raise ValueError(
                "results[%d] 的键必须依次为 name、epoch、sample_count、"
                "balanced_accuracy、macro_f1、min_balanced_accuracy、"
                "min_macro_f1、balanced_accuracy_pass、macro_f1_pass、"
                "pass" % idx
            )
        name = item["name"]
        if not isinstance(name, str) or len(name) == 0:
            raise TypeError(
                "results[%d].name 必须是非空 str，得到 %s"
                % (idx, type(name).__name__)
            )
        if name in names:
            raise ValueError(
                "results 的 name 必须唯一，重复：%r" % name
            )
        names.add(name)

        for key in ("epoch", "sample_count"):
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "results[%d].%s 必须是 int，得到 %s"
                    % (idx, key, type(value).__name__)
                )
        if item["epoch"] < 1:
            raise ValueError("results[%d].epoch 必须 >= 1" % idx)
        if item["sample_count"] < 1:
            raise ValueError(
                "results[%d].sample_count 必须 >= 1" % idx
            )

        metrics = {}
        for key in (
            "balanced_accuracy",
            "macro_f1",
            "min_balanced_accuracy",
            "min_macro_f1",
        ):
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError(
                    "results[%d].%s 必须是 float，得到 %s"
                    % (idx, key, type(value).__name__)
                )
            if not math.isfinite(value):
                raise ValueError(
                    "results[%d].%s 含有非有限值（NaN/inf）" % (idx, key)
                )
            metrics[key] = value
        if not (0.0 <= metrics["balanced_accuracy"] <= 1.0):
            raise ValueError(
                "results[%d].balanced_accuracy 必须在 [0,1] 内" % idx
            )
        if not (0.0 <= metrics["macro_f1"] <= 1.0):
            raise ValueError(
                "results[%d].macro_f1 必须在 [0,1] 内" % idx
            )
        if not (0.0 <= metrics["min_balanced_accuracy"] <= 1.0):
            raise ValueError(
                "results[%d].min_balanced_accuracy 必须在 [0,1] 内" % idx
            )
        if not (0.0 <= metrics["min_macro_f1"] <= 1.0):
            raise ValueError(
                "results[%d].min_macro_f1 必须在 [0,1] 内" % idx
            )

        item_passes = {}
        for key in ("balanced_accuracy_pass", "macro_f1_pass", "pass"):
            value = item[key]
            if not isinstance(value, bool):
                raise TypeError(
                    "results[%d].%s 必须是 bool，得到 %s"
                    % (idx, key, type(value).__name__)
                )
            item_passes[key] = value
        expected_item_pass = (
            item_passes["balanced_accuracy_pass"]
            and item_passes["macro_f1_pass"]
        )
        if item_passes["pass"] != expected_item_pass:
            raise ValueError(
                "results[%d].pass 必须为两个指标 pass 之与" % idx
            )
        if item_passes["balanced_accuracy_pass"] != (
            metrics["balanced_accuracy"]
            >= metrics["min_balanced_accuracy"]
        ):
            raise ValueError(
                "results[%d].balanced_accuracy_pass 与实测值/阈值不自洽"
                % idx
            )
        if item_passes["macro_f1_pass"] != (
            metrics["macro_f1"] >= metrics["min_macro_f1"]
        ):
            raise ValueError(
                "results[%d].macro_f1_pass 与实测值/阈值不自洽" % idx
            )

        if item_passes["pass"]:
            counted_pass += 1
        else:
            all_pass = False
        items.append(
            (name, metrics["balanced_accuracy"], metrics["macro_f1"])
        )

    if passed_count != counted_pass:
        raise ValueError(
            "passed_count 必须等于 results 中 pass 为真的项数"
        )
    if overall_pass != all_pass:
        raise ValueError("顶层 pass 必须为各项 pass 之与")
    return gate_count, items


def _valgatebatchdiff_compute(baseline_path, current_path):
    """读取并逐组对比两份 valgatebatch 产物，返回 diff 报告 dict。

    两份输入均按 _load_valgatebatch_output 的 valgatebatch 产物契约
    严格校验；gate_count 必须相等，且各 results 项的 name 顺序必须逐项
    一致（数量相同但顺序/名称不同一律拒绝）。每组 ba、f1 分别取
    balanced_accuracy、macro_f1；差值 delta 为当前值减基线值（未舍入
    float 相减），delta >= 0（非负）该指标才通过。任何 delta 非有限即
    失败（契约上不会发生，防御性校验）。

    返回 dict（键依次为 gate_count、results、pass）：results 每项键依次
    为 name、ba、f1、pass；ba/f1 键依次为 baseline、current、delta、
    pass，前三项 float、末项 bool；项 pass 为 ba/f1 pass 之与，顶层
    pass 为各项 pass 之与。输入不可读或契约不符抛
    OSError/UnicodeDecodeError/ValueError/TypeError。
    """
    with open(baseline_path, "rb") as f:
        baseline_raw = f.read()
    with open(current_path, "rb") as f:
        current_raw = f.read()

    base_count, base_items = _load_valgatebatch_output(baseline_raw)
    cur_count, cur_items = _load_valgatebatch_output(current_raw)
    if base_count != cur_count:
        raise ValueError(
            "两份产物 gate_count 不一致：%d != %d"
            % (base_count, cur_count)
        )
    for idx, (base_item, cur_item) in enumerate(zip(base_items, cur_items)):
        if base_item[0] != cur_item[0]:
            raise ValueError(
                "results[%d] 的 name 顺序不一致：%r != %r"
                % (idx, base_item[0], cur_item[0])
            )

    results = []
    overall = True
    for idx, (
        (name, base_ba, base_f1),
        (_cur_name, cur_ba, cur_f1),
    ) in enumerate(zip(base_items, cur_items)):
        metric_blocks = []
        item_pass = True
        for metric_name, base_value, cur_value in (
            ("ba", base_ba, cur_ba),
            ("f1", base_f1, cur_f1),
        ):
            delta = cur_value - base_value
            if not math.isfinite(delta):
                raise ValueError(
                    "results[%d].%s 的差值非有限" % (idx, metric_name)
                )
            metric_pass = delta >= 0.0
            # 负零归零（-0.0 >= 0 为真，但写出时 _fmt_float 亦会归一）。
            if delta == 0.0:
                delta = 0.0
            if not metric_pass:
                item_pass = False
            metric_blocks.append(
                (
                    metric_name,
                    {
                        "baseline": base_value,
                        "current": cur_value,
                        "delta": delta,
                        "pass": bool(metric_pass),
                    },
                )
            )
        if not item_pass:
            overall = False
        results.append(
            {
                "name": name,
                "ba": metric_blocks[0][1],
                "f1": metric_blocks[1][1],
                "pass": bool(item_pass),
            }
        )

    return {
        "gate_count": base_count,
        "results": results,
        "pass": bool(overall),
    }


def _cmd_valgatebatchdiff(baseline_path, current_path, output_path):
    """valgatebatchdiff 子命令主体；全组非回归退出 0、合法但存在回归
    退出 3、参数契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    BASELINE、CURRENT 为两份 valgatebatch 产物，契约见
    _load_valgatebatch_output：二者 gate_count 与各 results 项 name 顺序
    必须一致；每组 ba、f1 的 delta 为当前值减基线值，非负才通过。
    BASELINE、CURRENT、OUTPUT 三路径绝对化后须两两不同。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    gate_count（int）、results（list）、pass（bool）；results 每项键
    依次为 name（str）、ba、f1、pass（bool）；ba/f1 键依次为
    baseline、current、delta（有限 float，固定 12 位小数、负零归零）、
    pass（bool）；同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        base_abs = os.path.abspath(baseline_path)
        cur_abs = os.path.abspath(current_path)
        out_abs = os.path.abspath(output_path)
        if base_abs == cur_abs:
            raise ValueError("BASELINE 与 CURRENT 不能是同一路径")
        if base_abs == out_abs:
            raise ValueError("BASELINE 与 OUTPUT 不能是同一路径")
        if cur_abs == out_abs:
            raise ValueError("CURRENT 与 OUTPUT 不能是同一路径")

        report = _valgatebatchdiff_compute(baseline_path, current_path)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行多报告闸门趋势对比：
# python convnet.py valgatebatchtrend MANIFEST OUTPUT
# ---------------------------------------------------------------------------

_VALGATEBATCHTREND_MANIFEST_KEYS = ["reports"]
_VALGATEBATCHTREND_TOP_KEYS = [
    "report_count",
    "gate_count",
    "results",
    "pass",
]
_VALGATEBATCHTREND_ITEM_KEYS = ["name", "ba", "f1", "pass"]
_VALGATEBATCHTREND_METRIC_KEYS = [
    "worst_delta",
    "from_index",
    "to_index",
    "pass",
]


def _load_valgatebatchtrend_manifest(raw, manifest_dir):
    """按 valgatebatchtrend 契约从 MANIFEST 原始字节解析并严格校验。

    raw 不得含 UTF-8 BOM 或字符串字面量之外的任何 JSON 空白（空格、
    制表、换行、回车），须为紧凑 UTF-8 JSON 对象；唯一键 reports 为长度
    >= 2 的 list，重复/缺失/额外/错序键一律拒绝
    （object_pairs_hook 逐对象查重）并拒绝 NaN/Infinity 常量。各项须为
    非空 str（拒绝其他任何 JSON 类型），相对清单目录解析（绝对路径原样
    使用）为 abspath，绝对化后各项两两不同，否则 ValueError。返回按清单
    顺序的报告 abspath 列表；非法 UTF-8/JSON 或契约不符分别抛
    UnicodeDecodeError/ValueError/TypeError。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "MANIFEST 必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("MANIFEST 不得含 UTF-8 BOM")
    text = raw.decode("utf-8")
    _reject_json_whitespace(raw)
    _reject_json_constants(raw)
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("MANIFEST 顶层必须是 JSON 对象")
    if list(doc.keys()) != _VALGATEBATCHTREND_MANIFEST_KEYS:
        raise ValueError("MANIFEST 唯一键必须为 reports")
    reports = doc["reports"]
    if not isinstance(reports, list):
        raise TypeError(
            "reports 必须是 list，得到 %s" % type(reports).__name__
        )
    if len(reports) < 2:
        raise ValueError("reports 长度必须 >= 2，得到 %d" % len(reports))

    paths = []
    seen_paths = set()
    for idx, value in enumerate(reports):
        if not isinstance(value, str):
            raise TypeError(
                "reports[%d] 必须是非空 str，得到 %s"
                % (idx, type(value).__name__)
            )
        if len(value) == 0:
            raise ValueError("reports[%d] 不得为空字符串" % idx)
        path = os.path.abspath(os.path.join(manifest_dir, value))
        if path in seen_paths:
            raise ValueError(
                "reports 绝对化后路径必须两两不同，重复指向 %s" % path
            )
        seen_paths.add(path)
        paths.append(path)
    return paths


def _valgatebatchtrend_compute(report_paths):
    """读取多份 valgatebatch 产物并逐组计算相邻报告的最差趋势。

    每份报告均按 _load_valgatebatch_output 的 valgatebatch 产物契约
    严格校验；全部报告 gate_count 必须相等，且各 results 项的 name 顺序
    必须逐份逐项一致（数量相同但顺序/名称不同一律拒绝）。对相邻报告
    （i, i+1）计算每组 balanced_accuracy、macro_f1 的后值减前值（未舍入
    float 相减）；各指标取所有相邻对中的最小差值，平局取最早一对；
    worst_delta >= 0（非负）该指标才通过。任何差值非有限即失败
    （契约上不会发生，防御性校验）。

    返回 dict（键依次为 report_count、gate_count、results、pass）：
    results 每项键依次为 name、ba、f1、pass；ba/f1 键依次为
    worst_delta（float）、from_index、to_index（int，reports 零基下标）、
    pass（bool）；项 pass 为 ba/f1 pass 之与，顶层 pass 为各项 pass 之与。
    输入不可读或契约不符抛
    OSError/UnicodeDecodeError/ValueError/TypeError。
    """
    loaded = []
    gate_count = None
    for report_idx, path in enumerate(report_paths):
        with open(path, "rb") as f:
            raw = f.read()
        count, items = _load_valgatebatch_output(raw)
        if gate_count is None:
            gate_count = count
        elif count != gate_count:
            raise ValueError(
                "reports[%d] 的 gate_count 与首份报告不一致：%d != %d"
                % (report_idx, count, gate_count)
            )
        loaded.append(items)

    first_names = [name for name, _ba, _f1 in loaded[0]]
    for report_idx in range(1, len(loaded)):
        names = [name for name, _ba, _f1 in loaded[report_idx]]
        if names != first_names:
            raise ValueError(
                "reports[%d] 与首份报告的 results name 顺序不一致"
                % report_idx
            )

    results = []
    overall = True
    for gate_idx, name in enumerate(first_names):
        metric_blocks = []
        item_pass = True
        for metric_name, value_idx in (("ba", 1), ("f1", 2)):
            worst_delta = None
            worst_pair = None
            metric_pass = True
            for pair_idx in range(len(loaded) - 1):
                delta = (
                    loaded[pair_idx + 1][gate_idx][value_idx]
                    - loaded[pair_idx][gate_idx][value_idx]
                )
                if not math.isfinite(delta):
                    raise ValueError(
                        "results[%d].%s 的相邻差值非有限"
                        % (gate_idx, metric_name)
                    )
                if delta < 0.0:
                    metric_pass = False
                # 严格小于才替换：平局保留最早一对。
                if worst_delta is None or delta < worst_delta:
                    worst_delta = delta
                    worst_pair = pair_idx
            # 负零归零（-0.0 >= 0 为真，写出时 _fmt_float 亦会归一）。
            if worst_delta == 0.0:
                worst_delta = 0.0
            if not metric_pass:
                item_pass = False
            metric_blocks.append(
                (
                    metric_name,
                    {
                        "worst_delta": worst_delta,
                        "from_index": worst_pair,
                        "to_index": worst_pair + 1,
                        "pass": bool(metric_pass),
                    },
                )
            )
        if not item_pass:
            overall = False
        results.append(
            {
                "name": name,
                "ba": metric_blocks[0][1],
                "f1": metric_blocks[1][1],
                "pass": bool(item_pass),
            }
        )

    return {
        "report_count": len(report_paths),
        "gate_count": gate_count,
        "results": results,
        "pass": bool(overall),
    }


def _cmd_valgatebatchtrend(manifest_path, output_path):
    """valgatebatchtrend 子命令主体；全部趋势非回归退出 0、合法但存在
    回归退出 3、参数契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    MANIFEST 的严格契约见 _load_valgatebatchtrend_manifest：reports 数
    >= 2，均为非空 str，相对清单目录解析（绝对路径原样使用）且绝对化后
    两两不同。相对 OUTPUT 以 MANIFEST 所在目录解析（绝对 OUTPUT 原样
    使用）；绝对化后 MANIFEST、OUTPUT 与各报告路径须两两不同，任一冲突
    即失败退出 1 且不触碰 OUTPUT。每份报告完全复用
    _load_valgatebatch_output 的 valgatebatch 产物契约，gate_count 与
    results 的 name 顺序须跨报告一致；任一份报告读取/校验失败即整体失败
    退出 1，OUTPUT 原样保留（全部读取与计算完成后才一次性原子写盘）。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    report_count（int）、gate_count（int）、results（list）、pass
    （bool，各项 pass 之与）；results 按 name 顺序，每项键依次为
    name（str）、ba、f1、pass（bool）；ba/f1 键依次为 worst_delta
    （有限 float，固定 12 位小数、负零归零）、from_index、to_index
    （int，reports 零基下标）、pass（bool）；同一输入重复运行逐字节
    相同，标准输出为空。
    """
    overall = False
    try:
        manifest_abs = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_abs)
        # 相对 OUTPUT 以 MANIFEST 所在目录解析（绝对路径原样使用）。
        out_abs = os.path.abspath(os.path.join(manifest_dir, output_path))
        if out_abs == manifest_abs:
            raise ValueError("OUTPUT 与 MANIFEST 不能是同一路径")

        with open(manifest_path, "rb") as f:
            manifest_raw = f.read()
        report_paths = _load_valgatebatchtrend_manifest(
            manifest_raw, manifest_dir
        )

        # 清单内报告路径两两不同已由加载器保证；绝对化后 MANIFEST、
        # OUTPUT 与各报告路径亦须两两不同。
        for path in report_paths:
            if path == out_abs:
                raise ValueError(
                    "OUTPUT 与报告路径不能是同一路径：%s" % path
                )
            if path == manifest_abs:
                raise ValueError(
                    "MANIFEST 与报告路径不能是同一路径：%s" % path
                )

        report = _valgatebatchtrend_compute(report_paths)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行多报告闸门趋势回归统计：
# python convnet.py trendstats MANIFEST OUTPUT
# ---------------------------------------------------------------------------

_TRENDSTATS_MANIFEST_KEYS = ["trends"]
_TRENDSTATS_OUTPUT_KEYS = ["trend_count", "gate_count", "results", "pass"]
_TRENDSTATS_OUTPUT_ITEM_KEYS = ["name", "ba", "f1", "pass"]
_TRENDSTATS_OUTPUT_METRIC_KEYS = [
    "regressions",
    "worst_delta",
    "trend",
    "from",
    "to",
    "pass",
]


def _load_trendstats_manifest(raw, manifest_dir):
    """按 trendstats 契约从 MANIFEST 原始字节解析并严格校验。

    raw 不得含 UTF-8 BOM 或字符串字面量之外的任何 JSON 空白（空格、
    制表、换行、回车），须为紧凑 UTF-8 JSON 对象；唯一键 trends 为长度
    >= 2 的 list，重复/缺失/额外/错序键一律拒绝
    （object_pairs_hook 逐对象查重）并拒绝 NaN/Infinity 常量。各项须为
    非空 str（拒绝其他任何 JSON 类型），相对清单目录解析（绝对路径原样
    使用）为 abspath，绝对化后各项两两不同，否则 ValueError。返回按清单
    顺序的 valgatebatchtrend 产物 abspath 列表；非法 UTF-8/JSON 或契约
    不符分别抛 UnicodeDecodeError/ValueError/TypeError。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "MANIFEST 必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("MANIFEST 不得含 UTF-8 BOM")
    text = raw.decode("utf-8")
    _reject_json_whitespace(raw)
    _reject_json_constants(raw)
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("MANIFEST 顶层必须是 JSON 对象")
    if list(doc.keys()) != _TRENDSTATS_MANIFEST_KEYS:
        raise ValueError("MANIFEST 唯一键必须为 trends")
    trends = doc["trends"]
    if not isinstance(trends, list):
        raise TypeError(
            "trends 必须是 list，得到 %s" % type(trends).__name__
        )
    if len(trends) < 2:
        raise ValueError("trends 长度必须 >= 2，得到 %d" % len(trends))

    paths = []
    seen_paths = set()
    for idx, value in enumerate(trends):
        if not isinstance(value, str):
            raise TypeError(
                "trends[%d] 必须是非空 str，得到 %s"
                % (idx, type(value).__name__)
            )
        if len(value) == 0:
            raise ValueError("trends[%d] 不得为空字符串" % idx)
        path = os.path.abspath(os.path.join(manifest_dir, value))
        if path in seen_paths:
            raise ValueError(
                "trends 绝对化后路径必须两两不同，重复指向 %s" % path
            )
        seen_paths.add(path)
        paths.append(path)
    return paths


def _load_valgatebatchtrend_output(raw):
    """按 valgatebatchtrend 产物契约从原始字节严格解析一份趋势产物。

    与 valgatebatchtrend 写出的逐字节产物一致：不得含 UTF-8 BOM，须以
    恰好一个 LF 结尾（拒绝 CR 与多余空行），其前正文不得含字符串字面量
    之外的任何 JSON 空白（紧凑 JSON），拒绝 NaN/Infinity 常量与重复键。
    正文顶层键须依次为 report_count、gate_count、results、pass：
    report_count 为 >= 2 的 int（拒绝 bool），gate_count 为 >= 2 的 int，
    pass 为 bool，results 长度恰为 gate_count；每项键须依次为
    name/ba/f1/pass：name 为非空 str 且在本份产物内唯一，pass 为 bool；
    ba/f1 键须依次为 worst_delta/from_index/to_index/pass：worst_delta
    为有限 float、from_index/to_index 为满足
    0 <= from_index < to_index < report_count 的 int，pass 为 bool 且须与
    worst_delta >= 0 自洽；各项 pass 须恰为 ba/f1 pass 之与，顶层 pass
    须恰为各项 pass 之与。
    非法 UTF-8/JSON 或契约不符分别抛
    UnicodeDecodeError/ValueError/TypeError。返回
    (gate_count, [(name, ba_worst, ba_from, ba_to, f1_worst, f1_from,
    f1_to), ...])。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "valgatebatchtrend 产物必须是 bytes，得到 %s"
            % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("valgatebatchtrend 产物不得含 UTF-8 BOM")
    if b"\r" in raw:
        raise ValueError("valgatebatchtrend 产物不得含回车（CR）")
    if not raw.endswith(b"\n"):
        raise ValueError("valgatebatchtrend 产物必须以恰好一个 LF 结尾")
    body = raw[:-1]
    if body.endswith(b"\n"):
        raise ValueError("valgatebatchtrend 产物末尾仅可有一个 LF")
    _reject_json_whitespace(body)
    _reject_json_constants(body)
    text = body.decode("utf-8")
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("valgatebatchtrend 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _VALGATEBATCHTREND_TOP_KEYS:
        raise ValueError(
            "valgatebatchtrend 产物顶层键必须依次为 "
            "report_count、gate_count、results、pass"
        )

    report_count = doc["report_count"]
    if isinstance(report_count, bool) or not isinstance(report_count, int):
        raise TypeError(
            "report_count 必须是 int，得到 %s"
            % type(report_count).__name__
        )
    if report_count < 2:
        raise ValueError(
            "report_count 必须 >= 2，得到 %d" % report_count
        )

    gate_count = doc["gate_count"]
    if isinstance(gate_count, bool) or not isinstance(gate_count, int):
        raise TypeError(
            "gate_count 必须是 int，得到 %s" % type(gate_count).__name__
        )
    if gate_count < 2:
        raise ValueError("gate_count 必须 >= 2，得到 %d" % gate_count)

    overall_pass = doc["pass"]
    if not isinstance(overall_pass, bool):
        raise TypeError(
            "顶层 pass 必须是 bool，得到 %s"
            % type(overall_pass).__name__
        )

    results = doc["results"]
    if not isinstance(results, list):
        raise TypeError(
            "results 必须是 list，得到 %s" % type(results).__name__
        )
    if len(results) != gate_count:
        raise ValueError(
            "results 长度必须等于 gate_count（%d），得到 %d"
            % (gate_count, len(results))
        )

    items = []
    names = set()
    all_pass = True
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError("results[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _VALGATEBATCHTREND_ITEM_KEYS:
            raise ValueError(
                "results[%d] 的键必须依次为 name、ba、f1、pass" % idx
            )
        name = item["name"]
        if not isinstance(name, str) or len(name) == 0:
            raise TypeError(
                "results[%d].name 必须是非空 str，得到 %s"
                % (idx, type(name).__name__)
            )
        if name in names:
            raise ValueError(
                "results 的 name 必须唯一，重复：%r" % name
            )
        names.add(name)

        item_pass = item["pass"]
        if not isinstance(item_pass, bool):
            raise TypeError(
                "results[%d].pass 必须是 bool，得到 %s"
                % (idx, type(item_pass).__name__)
            )

        metric_worst = []
        metric_passes = []
        for metric_name in ("ba", "f1"):
            block = item[metric_name]
            if not isinstance(block, dict):
                raise TypeError(
                    "results[%d].%s 必须是 JSON 对象"
                    % (idx, metric_name)
                )
            if list(block.keys()) != _VALGATEBATCHTREND_METRIC_KEYS:
                raise ValueError(
                    "results[%d].%s 的键必须依次为 worst_delta、"
                    "from_index、to_index、pass" % (idx, metric_name)
                )
            worst_delta = block["worst_delta"]
            if isinstance(worst_delta, bool) or not isinstance(
                worst_delta, float
            ):
                raise TypeError(
                    "results[%d].%s.worst_delta 必须是 float，得到 %s"
                    % (idx, metric_name, type(worst_delta).__name__)
                )
            if not math.isfinite(worst_delta):
                raise ValueError(
                    "results[%d].%s.worst_delta 必须有限"
                    % (idx, metric_name)
                )
            for edge_key in ("from_index", "to_index"):
                edge = block[edge_key]
                if isinstance(edge, bool) or not isinstance(edge, int):
                    raise TypeError(
                        "results[%d].%s.%s 必须是 int，得到 %s"
                        % (idx, metric_name, edge_key, type(edge).__name__)
                    )
            from_index = block["from_index"]
            to_index = block["to_index"]
            if not (0 <= from_index < to_index < report_count):
                raise ValueError(
                    "results[%d].%s 的下标必须满足 0 <= from_index < "
                    "to_index < report_count" % (idx, metric_name)
                )
            metric_pass = block["pass"]
            if not isinstance(metric_pass, bool):
                raise TypeError(
                    "results[%d].%s.pass 必须是 bool，得到 %s"
                    % (idx, metric_name, type(metric_pass).__name__)
                )
            if metric_pass != (worst_delta >= 0.0):
                raise ValueError(
                    "results[%d].%s.pass 须与 worst_delta >= 0 自洽"
                    % (idx, metric_name)
                )
            metric_worst.append((worst_delta, from_index, to_index))
            metric_passes.append(metric_pass)

        if item_pass != (metric_passes[0] and metric_passes[1]):
            raise ValueError(
                "results[%d].pass 必须为 ba/f1 pass 之与" % idx
            )
        if not item_pass:
            all_pass = False

        ba_worst, ba_from, ba_to = metric_worst[0]
        f1_worst, f1_from, f1_to = metric_worst[1]
        items.append(
            (
                name,
                ba_worst,
                ba_from,
                ba_to,
                f1_worst,
                f1_from,
                f1_to,
            )
        )

    if overall_pass != all_pass:
        raise ValueError("顶层 pass 必须为各项 pass 之与")

    return gate_count, items


def _trendstats_compute(trend_paths):
    """读取多份 valgatebatchtrend 产物并逐组汇总回归统计。

    每份产物均按 _load_valgatebatchtrend_output 的严格契约校验；全部
    产物 gate_count 必须相等，且各 results 项的 name 顺序必须逐份逐项
    一致（数量相同但顺序/名称不同一律拒绝）。对每个 name 的 ba/f1：
    regressions 为 worst_delta < 0 的产物份数；worst_delta 取所有产物
    该值的最小值，平局取 trends 中最早的一份，trend 记该份零基序号，
    from/to 照录该份产物的 from_index/to_index。指标 pass 仅当
    regressions 为 0；项 pass 为 ba/f1 pass 之与；顶层 pass 为各项
    pass 之与。输入不可读或契约不符抛
    OSError/UnicodeDecodeError/ValueError/TypeError。

    返回 dict（键依次为 trend_count、gate_count、results、pass）：
    results 每项键依次为 name、ba、f1、pass；ba/f1 键依次为
    regressions（int）、worst_delta（float）、trend（int，trends 零基
    下标）、from、to（int，对应份的 report 零基下标）、pass（bool）。
    """
    loaded = []
    gate_count = None
    for trend_idx, path in enumerate(trend_paths):
        with open(path, "rb") as f:
            raw = f.read()
        count, items = _load_valgatebatchtrend_output(raw)
        if gate_count is None:
            gate_count = count
        elif count != gate_count:
            raise ValueError(
                "trends[%d] 的 gate_count 与首份产物不一致：%d != %d"
                % (trend_idx, count, gate_count)
            )
        loaded.append(items)

    first_names = [item[0] for item in loaded[0]]
    for trend_idx in range(1, len(loaded)):
        names = [item[0] for item in loaded[trend_idx]]
        if names != first_names:
            raise ValueError(
                "trends[%d] 与首份产物的 results name 顺序不一致"
                % trend_idx
            )

    results = []
    overall = True
    for gate_idx, name in enumerate(first_names):
        metric_blocks = []
        item_pass = True
        # (worst_delta 下标, from 下标, to 下标)
        for metric_name, value_idx, from_idx, to_idx in (
            ("ba", 1, 2, 3),
            ("f1", 4, 5, 6),
        ):
            regressions = 0
            worst_delta = None
            worst_trend = None
            for trend_idx in range(len(loaded)):
                delta = loaded[trend_idx][gate_idx][value_idx]
                if not math.isfinite(delta):
                    raise ValueError(
                        "results[%d].%s 的 worst_delta 非有限"
                        % (gate_idx, metric_name)
                    )
                if delta < 0.0:
                    regressions += 1
                # 严格小于才替换：平局保留最早一份。
                if worst_delta is None or delta < worst_delta:
                    worst_delta = delta
                    worst_trend = trend_idx
            if worst_delta == 0.0:
                worst_delta = 0.0
            from_index = loaded[worst_trend][gate_idx][from_idx]
            to_index = loaded[worst_trend][gate_idx][to_idx]
            metric_pass = regressions == 0
            if not metric_pass:
                item_pass = False
            metric_blocks.append(
                (
                    metric_name,
                    {
                        "regressions": regressions,
                        "worst_delta": worst_delta,
                        "trend": worst_trend,
                        "from": from_index,
                        "to": to_index,
                        "pass": bool(metric_pass),
                    },
                )
            )
        if not item_pass:
            overall = False
        results.append(
            {
                "name": name,
                "ba": metric_blocks[0][1],
                "f1": metric_blocks[1][1],
                "pass": bool(item_pass),
            }
        )

    return {
        "trend_count": len(trend_paths),
        "gate_count": gate_count,
        "results": results,
        "pass": bool(overall),
    }


def _cmd_trendstats(manifest_path, output_path):
    """trendstats 子命令主体；全部趋势无回归退出 0、合法但存在回归
    退出 3、参数契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    MANIFEST 的严格契约见 _load_trendstats_manifest：trends 数 >= 2，
    均为非空 str，相对清单目录解析（绝对路径原样使用）且绝对化后两两
    不同。相对 OUTPUT 以 MANIFEST 所在目录解析（绝对 OUTPUT 原样
    使用）；绝对化后 MANIFEST、OUTPUT 与各输入产物路径须两两不同，任一
    冲突即失败退出 1 且不触碰 OUTPUT。每份输入完全复用
    _load_valgatebatchtrend_output 的 valgatebatchtrend 严格产物契约，
    gate_count 与 results 的 name 顺序须跨份一致；任一份读取/校验失败
    即整体失败退出 1，OUTPUT 原样保留（全部读取与计算完成后才一次性
    原子写盘）。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    trend_count（int）、gate_count（int）、results（list）、pass
    （bool，各项 pass 之与）；results 按 name 顺序，每项键依次为
    name（str）、ba、f1、pass（bool）；ba/f1 键依次为 regressions
    （int）、worst_delta（有限 float，固定 12 位小数、负零归零）、
    trend（int，trends 零基下标）、from、to（int）、pass（bool，仅当
    regressions 为 0）；同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        manifest_abs = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_abs)
        # 相对 OUTPUT 以 MANIFEST 所在目录解析（绝对路径原样使用）。
        out_abs = os.path.abspath(os.path.join(manifest_dir, output_path))
        if out_abs == manifest_abs:
            raise ValueError("OUTPUT 与 MANIFEST 不能是同一路径")

        with open(manifest_path, "rb") as f:
            manifest_raw = f.read()
        trend_paths = _load_trendstats_manifest(manifest_raw, manifest_dir)

        # 清单内输入路径两两不同已由加载器保证；绝对化后 MANIFEST、
        # OUTPUT 与各输入路径亦须两两不同。
        for path in trend_paths:
            if path == out_abs:
                raise ValueError(
                    "OUTPUT 与输入路径不能是同一路径：%s" % path
                )
            if path == manifest_abs:
                raise ValueError(
                    "MANIFEST 与输入路径不能是同一路径：%s" % path
                )

        report = _trendstats_compute(trend_paths)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行趋势汇总可配置质量门禁：
# python convnet.py trendgate STATS CONFIG OUTPUT
# ---------------------------------------------------------------------------

_TRENDGATE_TOP_KEYS = ["ba", "f1"]
_TRENDGATE_LIMIT_KEYS = ["max_regressions", "min_worst_delta"]
_TRENDGATE_OUTPUT_TOP_KEYS = ["gate_count", "limits", "results", "pass"]
_TRENDGATE_OUTPUT_ITEM_KEYS = ["name", "ba", "f1", "pass"]
_TRENDGATE_OUTPUT_METRIC_KEYS = ["regressions", "worst_delta", "pass"]


def _load_trendstats_output(raw):
    """按 trendstats 产物契约从原始字节严格解析一份趋势汇总产物。

    与 trendstats 写出的逐字节产物一致：不得含 UTF-8 BOM，须以恰好一个
    LF 结尾（拒绝 CR 与多余空行），其前正文不得含字符串字面量之外的任何
    JSON 空白（紧凑 JSON），拒绝 NaN/Infinity 常量与重复键。正文顶层键
    须依次为 trend_count、gate_count、results、pass：trend_count 与
    gate_count 均为 >= 2 的 int（拒绝 bool），pass 为 bool；results 长度
    恰为 gate_count。每项键须依次为 name/ba/f1/pass：name 为非空 str 且
    全局唯一，pass 为 bool；ba/f1 键须依次为 regressions、worst_delta、
    trend、from、to、pass：regressions 为满足
    0 <= regressions <= trend_count 的 int，worst_delta 为有限 float，
    trend 为满足 0 <= trend < trend_count 的 int，from/to 为满足
    0 <= from < to 的 int（照录自对应份 valgatebatchtrend 产物），
    pass 为 bool 且须与 regressions == 0 自洽；各项 pass 须恰为 ba/f1
    pass 之与，顶层 pass 须恰为各项 pass 之与。非法 UTF-8/JSON 或契约
    不符分别抛 UnicodeDecodeError/ValueError/TypeError。返回
    (trend_count, gate_count,
    [(name, ba_regressions, ba_worst, f1_regressions, f1_worst), ...])。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "trendstats 产物必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("trendstats 产物不得含 UTF-8 BOM")
    if b"\r" in raw:
        raise ValueError("trendstats 产物不得含回车（CR）")
    if not raw.endswith(b"\n"):
        raise ValueError("trendstats 产物必须以恰好一个 LF 结尾")
    body = raw[:-1]
    if body.endswith(b"\n"):
        raise ValueError("trendstats 产物末尾仅可有一个 LF")
    _reject_json_whitespace(body)
    _reject_json_constants(body)
    text = body.decode("utf-8")
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("trendstats 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _TRENDSTATS_OUTPUT_KEYS:
        raise ValueError(
            "trendstats 产物顶层键必须依次为 "
            "trend_count、gate_count、results、pass"
        )

    trend_count = doc["trend_count"]
    if isinstance(trend_count, bool) or not isinstance(trend_count, int):
        raise TypeError(
            "trend_count 必须是 int，得到 %s" % type(trend_count).__name__
        )
    if trend_count < 2:
        raise ValueError("trend_count 必须 >= 2，得到 %d" % trend_count)

    gate_count = doc["gate_count"]
    if isinstance(gate_count, bool) or not isinstance(gate_count, int):
        raise TypeError(
            "gate_count 必须是 int，得到 %s" % type(gate_count).__name__
        )
    if gate_count < 2:
        raise ValueError("gate_count 必须 >= 2，得到 %d" % gate_count)

    overall_pass = doc["pass"]
    if not isinstance(overall_pass, bool):
        raise TypeError(
            "顶层 pass 必须是 bool，得到 %s"
            % type(overall_pass).__name__
        )

    results = doc["results"]
    if not isinstance(results, list):
        raise TypeError(
            "results 必须是 list，得到 %s" % type(results).__name__
        )
    if len(results) != gate_count:
        raise ValueError(
            "results 长度必须等于 gate_count（%d），得到 %d"
            % (gate_count, len(results))
        )

    items = []
    names = set()
    all_pass = True
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError("results[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _TRENDSTATS_OUTPUT_ITEM_KEYS:
            raise ValueError(
                "results[%d] 的键必须依次为 name、ba、f1、pass" % idx
            )
        name = item["name"]
        if not isinstance(name, str) or len(name) == 0:
            raise TypeError(
                "results[%d].name 必须是非空 str，得到 %s"
                % (idx, type(name).__name__)
            )
        if name in names:
            raise ValueError("results 的 name 必须唯一，重复：%r" % name)
        names.add(name)

        item_pass = item["pass"]
        if not isinstance(item_pass, bool):
            raise TypeError(
                "results[%d].pass 必须是 bool，得到 %s"
                % (idx, type(item_pass).__name__)
            )

        metric_values = []
        metric_passes = []
        for metric_name in ("ba", "f1"):
            block = item[metric_name]
            if not isinstance(block, dict):
                raise TypeError(
                    "results[%d].%s 必须是 JSON 对象" % (idx, metric_name)
                )
            if list(block.keys()) != _TRENDSTATS_OUTPUT_METRIC_KEYS:
                raise ValueError(
                    "results[%d].%s 的键必须依次为 regressions、"
                    "worst_delta、trend、from、to、pass"
                    % (idx, metric_name)
                )

            regressions = block["regressions"]
            if isinstance(regressions, bool) or not isinstance(
                regressions, int
            ):
                raise TypeError(
                    "results[%d].%s.regressions 必须是 int，得到 %s"
                    % (idx, metric_name, type(regressions).__name__)
                )
            if not (0 <= regressions <= trend_count):
                raise ValueError(
                    "results[%d].%s.regressions 必须满足 "
                    "0 <= regressions <= trend_count" % (idx, metric_name)
                )

            worst_delta = block["worst_delta"]
            if isinstance(worst_delta, bool) or not isinstance(
                worst_delta, float
            ):
                raise TypeError(
                    "results[%d].%s.worst_delta 必须是 float，得到 %s"
                    % (idx, metric_name, type(worst_delta).__name__)
                )
            if not math.isfinite(worst_delta):
                raise ValueError(
                    "results[%d].%s.worst_delta 必须有限"
                    % (idx, metric_name)
                )

            trend = block["trend"]
            if isinstance(trend, bool) or not isinstance(trend, int):
                raise TypeError(
                    "results[%d].%s.trend 必须是 int，得到 %s"
                    % (idx, metric_name, type(trend).__name__)
                )
            if not (0 <= trend < trend_count):
                raise ValueError(
                    "results[%d].%s.trend 必须满足 "
                    "0 <= trend < trend_count" % (idx, metric_name)
                )

            from_index = block["from"]
            to_index = block["to"]
            for edge_key, edge in (("from", from_index), ("to", to_index)):
                if isinstance(edge, bool) or not isinstance(edge, int):
                    raise TypeError(
                        "results[%d].%s.%s 必须是 int，得到 %s"
                        % (idx, metric_name, edge_key, type(edge).__name__)
                    )
            if not (0 <= from_index < to_index):
                raise ValueError(
                    "results[%d].%s 的下标必须满足 0 <= from < to"
                    % (idx, metric_name)
                )

            metric_pass = block["pass"]
            if not isinstance(metric_pass, bool):
                raise TypeError(
                    "results[%d].%s.pass 必须是 bool，得到 %s"
                    % (idx, metric_name, type(metric_pass).__name__)
                )
            if metric_pass != (regressions == 0):
                raise ValueError(
                    "results[%d].%s.pass 须与 regressions == 0 自洽"
                    % (idx, metric_name)
                )
            metric_values.append((regressions, worst_delta))
            metric_passes.append(metric_pass)

        if item_pass != (metric_passes[0] and metric_passes[1]):
            raise ValueError(
                "results[%d].pass 必须为 ba/f1 pass 之与" % idx
            )
        if not item_pass:
            all_pass = False

        items.append(
            (
                name,
                metric_values[0][0],
                metric_values[0][1],
                metric_values[1][0],
                metric_values[1][1],
            )
        )

    if overall_pass != all_pass:
        raise ValueError("顶层 pass 必须为各项 pass 之与")

    return trend_count, gate_count, items


def _load_trendgate_config(raw):
    """按 trendgate 契约从 CONFIG 原始字节解析并严格校验，返回阈值 dict。

    raw 不得含 UTF-8 BOM 或字符串字面量之外的任何 JSON 空白（空格、
    制表、换行、回车），须为紧凑 UTF-8 JSON 对象，拒绝 NaN/Infinity
    常量与重复键（object_pairs_hook 逐对象查重）。顶层键仅依次为 ba、f1；
    两项均为对象，键仅依次为 max_regressions、min_worst_delta：前者为
    非负 int（拒绝 bool），后者为 [-1,0] 内有限 float（拒绝 int/bool）。
    非法 UTF-8/JSON 或契约不符分别抛
    UnicodeDecodeError/ValueError/TypeError。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "CONFIG 必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("CONFIG 不得含 UTF-8 BOM")
    _reject_json_whitespace(raw)
    _reject_json_constants(raw)
    text = raw.decode("utf-8")
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("CONFIG 顶层必须是 JSON 对象")
    if list(doc.keys()) != _TRENDGATE_TOP_KEYS:
        raise ValueError("CONFIG 键必须依次为 ba、f1")

    limits = {}
    for metric_name in _TRENDGATE_TOP_KEYS:
        block = doc[metric_name]
        if not isinstance(block, dict):
            raise TypeError(
                "CONFIG.%s 必须是 JSON 对象，得到 %s"
                % (metric_name, type(block).__name__)
            )
        if list(block.keys()) != _TRENDGATE_LIMIT_KEYS:
            raise ValueError(
                "CONFIG.%s 的键必须依次为 max_regressions、"
                "min_worst_delta" % metric_name
            )

        max_regressions = block["max_regressions"]
        if isinstance(max_regressions, bool) or not isinstance(
            max_regressions, int
        ):
            raise TypeError(
                "CONFIG.%s.max_regressions 必须是非负 int，得到 %s"
                % (metric_name, type(max_regressions).__name__)
            )
        if max_regressions < 0:
            raise ValueError(
                "CONFIG.%s.max_regressions 必须非负，得到 %d"
                % (metric_name, max_regressions)
            )

        min_worst_delta = block["min_worst_delta"]
        if isinstance(min_worst_delta, bool) or isinstance(
            min_worst_delta, int
        ) or not isinstance(min_worst_delta, float):
            raise TypeError(
                "CONFIG.%s.min_worst_delta 必须是 float（拒绝 int/bool），"
                "得到 %s" % (metric_name, type(min_worst_delta).__name__)
            )
        if not math.isfinite(min_worst_delta) or not (
            -1.0 <= min_worst_delta <= 0.0
        ):
            raise ValueError(
                "CONFIG.%s.min_worst_delta 必须是 [-1,0] 内的有限 float"
                % metric_name
            )

        limits[metric_name] = {
            "max_regressions": max_regressions,
            "min_worst_delta": min_worst_delta,
        }
    return limits


def _trendgate_compute(stats_raw, config_raw):
    """按 trendgate 契约校验 STATS/CONFIG 并逐 name 判定质量门禁。

    STATS 的严格契约见 _load_trendstats_output（trendstats 严格产物），
    CONFIG 的严格契约见 _load_trendgate_config。对每个 name 的 ba/f1：
    指标 pass 当且仅当 regressions <= max_regressions 且
    worst_delta >= min_worst_delta；项 pass 为 ba/f1 pass 之与；顶层
    pass 为各项 pass 之与。契约不符抛 ValueError/TypeError。

    返回 dict（键依次为 gate_count、limits、results、pass）：limits 同
    CONFIG（键依次 ba、f1，各键依次 max_regressions、min_worst_delta）；
    results 保持 STATS 的 name 顺序，每项键依次为 name、ba、f1、pass；
    ba/f1 键依次为 regressions（int）、worst_delta（float）、pass
    （bool）。
    """
    _trend_count, gate_count, items = _load_trendstats_output(stats_raw)
    limits = _load_trendgate_config(config_raw)

    results = []
    overall = True
    for name, ba_reg, ba_worst, f1_reg, f1_worst in items:
        metric_blocks = []
        item_pass = True
        for metric_name, regressions, worst_delta in (
            ("ba", ba_reg, ba_worst),
            ("f1", f1_reg, f1_worst),
        ):
            metric_limits = limits[metric_name]
            metric_pass = (
                regressions <= metric_limits["max_regressions"]
                and worst_delta >= metric_limits["min_worst_delta"]
            )
            if not metric_pass:
                item_pass = False
            metric_blocks.append(
                (
                    metric_name,
                    {
                        "regressions": regressions,
                        "worst_delta": worst_delta,
                        "pass": bool(metric_pass),
                    },
                )
            )
        if not item_pass:
            overall = False
        results.append(
            {
                "name": name,
                "ba": metric_blocks[0][1],
                "f1": metric_blocks[1][1],
                "pass": bool(item_pass),
            }
        )

    return {
        "gate_count": gate_count,
        "limits": limits,
        "results": results,
        "pass": bool(overall),
    }


def _cmd_trendgate(stats_path, config_path, output_path):
    """trendgate 子命令主体；全部 name 的 ba/f1 均达标退出 0、合法但未
    通过退出 3、契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    STATS、CONFIG、OUTPUT 三路径绝对化后必须两两不同。STATS 必须是
    trendstats 的严格产物（契约见 _load_trendstats_output），CONFIG 的
    严格契约见 _load_trendgate_config；任一读取/校验失败即整体失败退出
    1，OUTPUT 原样保留（全部读取与计算完成后才一次性原子写盘）。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    gate_count（int）、limits（同 CONFIG）、results（list）、pass
    （bool，各项 pass 之与）；results 保持 name 顺序，每项键依次为
    name（str）、ba、f1、pass（bool）；ba/f1 键依次为 regressions
    （int）、worst_delta（有限 float，固定 12 位小数、负零归零）、
    pass（bool）；同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        stats_abs = os.path.abspath(stats_path)
        config_abs = os.path.abspath(config_path)
        out_abs = os.path.abspath(output_path)
        if out_abs == stats_abs:
            raise ValueError("OUTPUT 与 STATS 不能是同一路径")
        if out_abs == config_abs:
            raise ValueError("OUTPUT 与 CONFIG 不能是同一路径")
        if stats_abs == config_abs:
            raise ValueError("STATS 与 CONFIG 不能是同一路径")

        with open(stats_path, "rb") as f:
            stats_raw = f.read()
        with open(config_path, "rb") as f:
            config_raw = f.read()

        report = _trendgate_compute(stats_raw, config_raw)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行趋势审计（内存串联 trendstats + trendgate）：
# python convnet.py trendaudit MANIFEST CONFIG OUTPUT
# ---------------------------------------------------------------------------


def _trendaudit_compute(manifest_raw, config_raw, manifest_dir):
    """按 trendaudit 契约在内存内串联 trendstats 与 trendgate。

    MANIFEST 沿用 trendstats 的输入契约
    （_load_trendstats_manifest，相对路径以 manifest_dir 解析），CONFIG
    沿用 trendgate 的配置契约（_load_trendgate_config）。先在内存内完成
    trendstats 计算并序列化为 trendstats 的严格字节产物（不落任何中间
    文件），再将该字节直接交给 _trendgate_compute 完成门禁判定，故两份
    结果与同输入单独运行 trendstats、trendgate 的产物在字段类型、数组
    顺序与逐值上完全一致。契约不符或输入读取失败抛
    OSError/UnicodeDecodeError/ValueError/TypeError。

    返回 (stats_report, gate_report)，结构分别见 _trendstats_compute 与
    _trendgate_compute。
    """
    trend_paths = _load_trendstats_manifest(manifest_raw, manifest_dir)
    stats_report = _trendstats_compute(trend_paths)
    stats_raw = (_dump_compact(stats_report) + "\n").encode("utf-8")
    gate_report = _trendgate_compute(stats_raw, config_raw)
    return stats_report, gate_report


def _cmd_trendaudit(manifest_path, config_path, output_path):
    """trendaudit 子命令主体；门禁全部通过退出 0、合法但未通过退出 3、
    契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    MANIFEST 沿用 trendstats 的清单契约（_load_trendstats_manifest，
    相对条目以清单目录解析），CONFIG 沿用 trendgate 的配置契约
    （_load_trendgate_config）；CONFIG、OUTPUT 的相对路径均以 MANIFEST
    所在目录解析（绝对路径原样使用）。绝对化后 MANIFEST、CONFIG、OUTPUT
    与清单内各输入产物路径须两两不同，任一冲突即失败退出 1 且不触碰
    OUTPUT。trendstats 与 trendgate 全程在内存内执行，不写任何中间文件。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为 stats、gate：
    stats 为 trendstats 单独运行产物的正文对象（键依次 trend_count、
    gate_count、results、pass），gate 为 trendgate 单独运行产物的正文
    对象（键依次 gate_count、limits、results、pass，limits 复制 CONFIG），
    二者字段类型、数组顺序、逐值及 pass 关系与同输入单独运行两命令完全
    一致；即使 gate.pass 为假亦照常写盘，退出码以 gate.pass 为准。同一
    输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        manifest_abs = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_abs)
        # 相对 CONFIG/OUTPUT 以 MANIFEST 所在目录解析（绝对路径原样使用）。
        config_abs = os.path.abspath(os.path.join(manifest_dir, config_path))
        out_abs = os.path.abspath(os.path.join(manifest_dir, output_path))
        if manifest_abs == config_abs:
            raise ValueError("MANIFEST 与 CONFIG 不能是同一路径")
        if manifest_abs == out_abs:
            raise ValueError("MANIFEST 与 OUTPUT 不能是同一路径")
        if config_abs == out_abs:
            raise ValueError("CONFIG 与 OUTPUT 不能是同一路径")

        with open(manifest_abs, "rb") as f:
            manifest_raw = f.read()
        with open(config_abs, "rb") as f:
            config_raw = f.read()

        # 清单内输入路径两两不同已由加载器保证；绝对化后 MANIFEST、
        # CONFIG、OUTPUT 与各输入路径亦须两两不同。
        trend_paths = _load_trendstats_manifest(manifest_raw, manifest_dir)
        for path in trend_paths:
            if path == manifest_abs:
                raise ValueError(
                    "MANIFEST 与输入路径不能是同一路径：%s" % path
                )
            if path == config_abs:
                raise ValueError(
                    "CONFIG 与输入路径不能是同一路径：%s" % path
                )
            if path == out_abs:
                raise ValueError(
                    "OUTPUT 与输入路径不能是同一路径：%s" % path
                )

        stats_report, gate_report = _trendaudit_compute(
            manifest_raw, config_raw, manifest_dir
        )
        overall = gate_report["pass"]
        report = {
            "stats": stats_report,
            "gate": gate_report,
        }
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行批量趋势审计（内存逐项串联 trendaudit）：
# python convnet.py trendauditbatch MANIFEST OUTPUT
# ---------------------------------------------------------------------------

_TRENDAUDITBATCH_MANIFEST_KEYS = ["audits"]
_TRENDAUDITBATCH_ITEM_KEYS = ["name", "manifest", "config"]
_TRENDAUDITBATCH_OUTPUT_KEYS = [
    "audit_count",
    "passed_count",
    "results",
    "pass",
]
_TRENDAUDITBATCH_OUTPUT_ITEM_KEYS = ["name", "stats", "gate", "pass"]


def _load_trendauditbatch_manifest(raw, manifest_dir):
    """按 trendauditbatch 契约从 MANIFEST 原始字节解析并严格校验。

    raw 不得含 UTF-8 BOM 或字符串字面量之外的任何 JSON 空白（空格、
    制表、换行、回车），须为紧凑 UTF-8 JSON 对象；唯一键 audits 为长度
    >= 2 的 list，重复/缺失/额外/错序键（含各项对象内部）一律拒绝
    （object_pairs_hook 逐对象查重）并拒绝 NaN/Infinity 常量。各项须为
    JSON 对象，键仅依次为 name、manifest、config；三值均为非空 str
    （拒绝其他任何 JSON 类型），且 name 全局唯一。manifest 相对批清单
    目录解析（绝对路径原样使用）为 abspath 并全局两两不同；config 暂存
    原始相对值，其绝对化改以对应 manifest 所在目录解析、连同其余全部
    路径的两两互斥在 _cmd_trendauditbatch 预检阶段统一保证。返回按清单
    顺序的 (name, manifest_abspath, config_rel) 列表；非法 UTF-8/JSON
    或契约不符分别抛 UnicodeDecodeError/ValueError/TypeError。
"""
    if not isinstance(raw, bytes):
        raise TypeError(
            "MANIFEST 必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("MANIFEST 不得含 UTF-8 BOM")
    text = raw.decode("utf-8")
    _reject_json_whitespace(raw)
    _reject_json_constants(raw)
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("MANIFEST 顶层必须是 JSON 对象")
    if list(doc.keys()) != _TRENDAUDITBATCH_MANIFEST_KEYS:
        raise ValueError("MANIFEST 唯一键必须为 audits")
    audits = doc["audits"]
    if not isinstance(audits, list):
        raise TypeError(
            "audits 必须是 list，得到 %s" % type(audits).__name__
        )
    if len(audits) < 2:
        raise ValueError("audits 长度必须 >= 2，得到 %d" % len(audits))

    items = []
    names = set()
    seen_manifests = set()
    for idx, item in enumerate(audits):
        if not isinstance(item, dict):
            raise TypeError("audits[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _TRENDAUDITBATCH_ITEM_KEYS:
            raise ValueError(
                "audits[%d] 的键必须依次为 name、manifest、config" % idx
            )
        values = {}
        for key in _TRENDAUDITBATCH_ITEM_KEYS:
            value = item[key]
            if not isinstance(value, str):
                raise TypeError(
                    "audits[%d].%s 必须是非空 str，得到 %s"
                    % (idx, key, type(value).__name__)
                )
            if len(value) == 0:
                raise ValueError(
                    "audits[%d].%s 不得为空字符串" % (idx, key)
                )
            values[key] = value
        name = values["name"]
        if name in names:
            raise ValueError("audits 的 name 必须唯一，重复：%r" % name)
        names.add(name)

        manifest_abs = os.path.abspath(
            os.path.join(manifest_dir, values["manifest"])
        )
        if manifest_abs in seen_manifests:
            raise ValueError(
                "audits 的 manifest 路径绝对化后必须两两不同，"
                "重复指向 %s" % manifest_abs
            )
        seen_manifests.add(manifest_abs)
        items.append((name, manifest_abs, values["config"]))
    return items


def _trendauditbatch_compute(entries):
    """逐项在内存执行 trendaudit，汇总批量审计报告。

    entries 为预检阶段产出的
    (name, manifest_abs, config_abs, manifest_raw, trend_paths) 列表，
    全部路径的两两互斥已在 _cmd_trendauditbatch 用全局路径集保证。这里
    逐项读取 config，随后完全复用 _trendaudit_compute：在内存内串联
    trendstats 与 trendgate（其清单内 trends 条目以该 manifest 目录解析，
    与预检解析出的 trend_paths 同源），全程不落任何中间文件。任一文件
    读取或契约/计算失败抛 OSError/UnicodeDecodeError/ValueError/TypeError。

    返回 dict（键依次为 audit_count、passed_count、results、pass）：
    results 保持 audits 的输入顺序，每项键依次为 name、stats、gate、
    pass；stats、gate 为该项单独 trendaudit 输出中同名对象的逐层等值
    副本（结构分别见 _trendstats_compute 与 _trendgate_compute），项
    pass 取 gate.pass，passed_count 为通过项数，顶层 pass 为各项 pass
    之与。
    """
    results = []
    passed_count = 0
    for name, manifest_abs, config_abs, manifest_raw, _trend_paths in entries:
        with open(config_abs, "rb") as f:
            config_raw = f.read()
        stats_report, gate_report = _trendaudit_compute(
            manifest_raw, config_raw, os.path.dirname(manifest_abs)
        )
        item_pass = gate_report["pass"]
        if item_pass:
            passed_count += 1
        results.append(
            {
                "name": name,
                "stats": stats_report,
                "gate": gate_report,
                "pass": bool(item_pass),
            }
        )

    return {
        "audit_count": len(entries),
        "passed_count": passed_count,
        "results": results,
        "pass": bool(passed_count == len(entries)),
    }


def _cmd_trendauditbatch(manifest_path, output_path):
    """trendauditbatch 子命令主体；全部审计项门禁通过退出 0、合法但未
    全过退出 3、契约/路径/I-O 失败退出 1 且不改 OUTPUT。

    MANIFEST 的严格契约见 _load_trendauditbatch_manifest：audits 数 >= 2，
    name 唯一，manifest 相对批清单目录解析且绝对化后两两不同；每项
    config 相对对应 manifest 所在目录解析。相对 OUTPUT 以批清单所在目录
    解析（绝对 OUTPUT 原样使用）。预检阶段逐项解析其 manifest（沿用
    _load_trendstats_manifest 契约），将批清单、OUTPUT、全部
    manifest/config 以及各项清单内全部子趋势文件路径纳入同一全局集合，
    绝对化后须两两不同，任一冲突即失败退出 1 且不触碰 OUTPUT。随后各项
    在内存内执行 trendaudit（_trendaudit_compute），不落任何中间文件；
    任一项读取/校验/计算失败即整体失败退出 1，OUTPUT 原样保留（全部项
    完成后才一次性原子写盘）。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    audit_count（int）、passed_count（int）、results（list）、pass
    （bool，各项 pass 之与）；results 保持输入顺序，每项键依次为
    name（str）、stats、gate、pass（bool，同 gate.pass）；stats、gate
    分别为该项单独 trendaudit 输出中同名对象的逐层等值副本（float 固定
    12 位小数、负零归零）；合法但未全过仍照常写盘，退出码以顶层 pass 为
    准。同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        manifest_abs = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_abs)
        # 相对 OUTPUT 以批清单所在目录解析（绝对路径原样使用）。
        out_abs = os.path.abspath(os.path.join(manifest_dir, output_path))

        with open(manifest_path, "rb") as f:
            manifest_raw = f.read()
        items = _load_trendauditbatch_manifest(manifest_raw, manifest_dir)

        # 全局路径集：批清单、OUTPUT、全部 manifest/config 及子趋势文件
        # 绝对化后须两两不同。预检同时完成各 item manifest 的读取与清单
        # 解析（contract 失败即整体失败，OUTPUT 不被触碰）。
        all_paths = set()

        def _claim(path, label):
            if path in all_paths:
                raise ValueError(
                    "%s 与其他路径不能相同：%s" % (label, path)
                )
            all_paths.add(path)

        _claim(manifest_abs, "批 MANIFEST")
        _claim(out_abs, "OUTPUT")

        entries = []
        for name, item_manifest_abs, config_rel in items:
            item_manifest_dir = os.path.dirname(item_manifest_abs)
            config_abs = os.path.abspath(
                os.path.join(item_manifest_dir, config_rel)
            )
            _claim(item_manifest_abs, "审计项 %r 的 manifest" % name)
            _claim(config_abs, "审计项 %r 的 config" % name)

            with open(item_manifest_abs, "rb") as f:
                item_manifest_raw = f.read()
            trend_paths = _load_trendstats_manifest(
                item_manifest_raw, item_manifest_dir
            )
            for trend_path in trend_paths:
                _claim(trend_path, "审计项 %r 的子趋势文件" % name)

            entries.append(
                (
                    name,
                    item_manifest_abs,
                    config_abs,
                    item_manifest_raw,
                    trend_paths,
                )
            )

        report = _trendauditbatch_compute(entries)
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


# ---------------------------------------------------------------------------
# 命令行批量趋势审计基线对比：
# python convnet.py trendauditbatchdiff BASELINE CURRENT OUTPUT
# ---------------------------------------------------------------------------

# trendauditbatchdiff 产物键序（输入复用上方 trendauditbatch 的
# _TRENDAUDITBATCH_OUTPUT_KEYS / _TRENDAUDITBATCH_OUTPUT_ITEM_KEYS）。
_TRENDAUDITBATCHDIFF_TOP_KEYS = ["audit_count", "results", "pass"]
_TRENDAUDITBATCHDIFF_ITEM_KEYS = [
    "name",
    "regressions_delta",
    "worst_delta_delta",
    "pass",
]


def _load_trendauditbatch_output(raw):
    """按 trendauditbatch 严格产物契约从原始字节解析一份批量审计产物。

    与 trendauditbatch 写出的逐字节产物一致：不得含 UTF-8 BOM，须以恰好
    一个 LF（b"\\n"）结尾（拒绝 CR 与多余空行），其前正文不得含字符串
    字面量之外的任何 JSON 空白（紧凑 JSON），拒绝 NaN/Infinity 常量与
    重复键。正文顶层键须依次为 audit_count、passed_count、results、
    pass：audit_count 为 >= 2 的 int（拒绝 bool），passed_count 为
    [0, audit_count] 内 int，pass 为 bool，results 长度恰为
    audit_count。每项键须依次为 name、stats、gate、pass：name 为非空
    str 且在本份产物内唯一、顺序保留，项 pass 为 bool 且须恰为该项
    gate.pass。

    stats 为该项单独 trendstats 产物的正文对象（键依次 trend_count、
    gate_count、results、pass），整体复用 _load_trendstats_output 的
    严格契约（重新序列化为规范紧凑字节后校验，含类型/区间/自洽）；
    gate 为该项单独 trendgate 产物的正文对象（键依次 gate_count、
    limits、results、pass）：gate_count 须与 stats.gate_count 相等，
    limits 复用 _load_trendgate_config 的严格契约（键依次 ba、f1，各键
    依次 max_regressions、min_worst_delta），gate.results 长度恰为
    gate_count 且与 stats.results 逐项下标对齐——name 相同，ba/f1 键
    依次 regressions、worst_delta、pass，regressions/worst_delta 须与
    stats 同名指标逐值相等，指标 pass 须与
    regressions <= max_regressions 且 worst_delta >= min_worst_delta
    自洽，gate 项 pass 为 ba/f1 pass 之与，gate.pass 为各项 pass 之
    与。顶层 passed_count 须恰为项 pass 为真的数量，顶层 pass 须恰为
    各项 pass 之与。非法 UTF-8/JSON 或契约不符分别抛
    UnicodeDecodeError/ValueError/TypeError。

    返回 (audit_count, items, shapes)：items 为每项
    (name, regressions_sum, worst_min)——按 stats 的 ba/f1 聚合，
    regressions 求和、worst_delta 取最小值，顺序同 results；shapes 为
    每项 (stats.gate_count, 内层 results 的 name 顺序 tuple)，供两份
    产物逐审计项核对 gate_count 与内层 name 顺序。
    """
    if not isinstance(raw, bytes):
        raise TypeError(
            "trendauditbatch 产物必须是 bytes，得到 %s" % type(raw).__name__
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("trendauditbatch 产物不得含 UTF-8 BOM")
    if b"\r" in raw:
        raise ValueError("trendauditbatch 产物不得含回车（CR）")
    if not raw.endswith(b"\n"):
        raise ValueError("trendauditbatch 产物必须以恰好一个 LF 结尾")
    body = raw[:-1]
    if body.endswith(b"\n"):
        raise ValueError("trendauditbatch 产物末尾仅可有一个 LF")
    _reject_json_whitespace(body)
    _reject_json_constants(body)
    text = body.decode("utf-8")
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(doc, dict):
        raise TypeError("trendauditbatch 产物顶层必须是 JSON 对象")
    if list(doc.keys()) != _TRENDAUDITBATCH_OUTPUT_KEYS:
        raise ValueError(
            "trendauditbatch 产物顶层键必须依次为 "
            "audit_count、passed_count、results、pass"
        )

    audit_count = doc["audit_count"]
    if isinstance(audit_count, bool) or not isinstance(audit_count, int):
        raise TypeError(
            "audit_count 必须是 int，得到 %s"
            % type(audit_count).__name__
        )
    if audit_count < 2:
        raise ValueError("audit_count 必须 >= 2，得到 %d" % audit_count)

    passed_count = doc["passed_count"]
    if isinstance(passed_count, bool) or not isinstance(passed_count, int):
        raise TypeError(
            "passed_count 必须是 int，得到 %s"
            % type(passed_count).__name__
        )
    if not (0 <= passed_count <= audit_count):
        raise ValueError(
            "passed_count 必须在 [0, audit_count] 内，得到 %d"
            % passed_count
        )

    overall_pass = doc["pass"]
    if not isinstance(overall_pass, bool):
        raise TypeError(
            "顶层 pass 必须是 bool，得到 %s"
            % type(overall_pass).__name__
        )

    results = doc["results"]
    if not isinstance(results, list):
        raise TypeError(
            "results 必须是 list，得到 %s" % type(results).__name__
        )
    if len(results) != audit_count:
        raise ValueError(
            "results 长度必须等于 audit_count（%d），得到 %d"
            % (audit_count, len(results))
        )

    items = []
    shapes = []
    names = set()
    counted_pass = 0
    all_pass = True
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError("results[%d] 必须是 JSON 对象" % idx)
        if list(item.keys()) != _TRENDAUDITBATCH_OUTPUT_ITEM_KEYS:
            raise ValueError(
                "results[%d] 的键必须依次为 name、stats、gate、pass" % idx
            )
        name = item["name"]
        if not isinstance(name, str) or len(name) == 0:
            raise TypeError(
                "results[%d].name 必须是非空 str，得到 %s"
                % (idx, type(name).__name__)
            )
        if name in names:
            raise ValueError("results 的 name 必须唯一，重复：%r" % name)
        names.add(name)

        item_pass = item["pass"]
        if not isinstance(item_pass, bool):
            raise TypeError(
                "results[%d].pass 必须是 bool，得到 %s"
                % (idx, type(item_pass).__name__)
            )

        # stats 为 trendstats 正文对象：重新序列化为规范紧凑字节后整体
        # 复用 trendstats 严格产物契约（键序/类型/区间/逐层自洽）。
        stats = item["stats"]
        if not isinstance(stats, dict):
            raise TypeError(
                "results[%d].stats 必须是 JSON 对象，得到 %s"
                % (idx, type(stats).__name__)
            )
        stats_raw = (_dump_compact(stats) + "\n").encode("utf-8")
        (
            _trend_count,
            stats_gate_count,
            stats_items,
        ) = _load_trendstats_output(stats_raw)

        # gate 为 trendgate 正文对象，与 stats 同源：gate_count、results
        # 的 name 顺序与逐值（regressions/worst_delta）必须逐层一致。
        gate = item["gate"]
        if not isinstance(gate, dict):
            raise TypeError(
                "results[%d].gate 必须是 JSON 对象，得到 %s"
                % (idx, type(gate).__name__)
            )
        if list(gate.keys()) != _TRENDGATE_OUTPUT_TOP_KEYS:
            raise ValueError(
                "results[%d].gate 的键必须依次为 gate_count、limits、"
                "results、pass" % idx
            )

        gate_count = gate["gate_count"]
        if isinstance(gate_count, bool) or not isinstance(gate_count, int):
            raise TypeError(
                "results[%d].gate.gate_count 必须是 int，得到 %s"
                % (idx, type(gate_count).__name__)
            )
        if gate_count < 2:
            raise ValueError(
                "results[%d].gate.gate_count 必须 >= 2，得到 %d"
                % (idx, gate_count)
            )
        if gate_count != stats_gate_count:
            raise ValueError(
                "results[%d].gate.gate_count 与 stats.gate_count 不一致："
                "%d != %d" % (idx, gate_count, stats_gate_count)
            )

        limits_block = gate["limits"]
        if not isinstance(limits_block, dict):
            raise TypeError(
                "results[%d].gate.limits 必须是 JSON 对象，得到 %s"
                % (idx, type(limits_block).__name__)
            )
        limits = _load_trendgate_config(
            _dump_compact(limits_block).encode("utf-8")
        )

        gate_results = gate["results"]
        if not isinstance(gate_results, list):
            raise TypeError(
                "results[%d].gate.results 必须是 list，得到 %s"
                % (idx, type(gate_results).__name__)
            )
        if len(gate_results) != gate_count:
            raise ValueError(
                "results[%d].gate.results 长度必须等于 gate_count（%d），"
                "得到 %d" % (idx, gate_count, len(gate_results))
            )

        gate_overall_pass = gate["pass"]
        if not isinstance(gate_overall_pass, bool):
            raise TypeError(
                "results[%d].gate.pass 必须是 bool，得到 %s"
                % (idx, type(gate_overall_pass).__name__)
            )

        gate_all_pass = True
        # 聚合整个 stats（全部 gate 结果 × ba/f1）：regressions 求和、
        # worst_delta 取最小值（平局任取，标量结果不受影响）。
        item_regressions_sum = 0
        item_worst_min = None
        inner_names = []
        for gate_idx, (
            stats_name,
            stats_ba_reg,
            stats_ba_worst,
            stats_f1_reg,
            stats_f1_worst,
        ) in enumerate(stats_items):
            inner_names.append(stats_name)
            gate_item = gate_results[gate_idx]
            if not isinstance(gate_item, dict):
                raise TypeError(
                    "results[%d].gate.results[%d] 必须是 JSON 对象"
                    % (idx, gate_idx)
                )
            if list(gate_item.keys()) != _TRENDGATE_OUTPUT_ITEM_KEYS:
                raise ValueError(
                    "results[%d].gate.results[%d] 的键必须依次为 name、"
                    "ba、f1、pass" % (idx, gate_idx)
                )
            gate_name = gate_item["name"]
            if not isinstance(gate_name, str) or len(gate_name) == 0:
                raise TypeError(
                    "results[%d].gate.results[%d].name 必须是非空 str，"
                    "得到 %s" % (idx, gate_idx, type(gate_name).__name__)
                )
            if gate_name != stats_name:
                raise ValueError(
                    "results[%d].gate.results[%d].name 与 stats 不一致："
                    "%r != %r" % (idx, gate_idx, gate_name, stats_name)
                )

            gate_item_pass = gate_item["pass"]
            if not isinstance(gate_item_pass, bool):
                raise TypeError(
                    "results[%d].gate.results[%d].pass 必须是 bool，得到 %s"
                    % (idx, gate_idx, type(gate_item_pass).__name__)
                )

            metric_pass_all = True
            for metric_name, stats_reg, stats_worst in (
                ("ba", stats_ba_reg, stats_ba_worst),
                ("f1", stats_f1_reg, stats_f1_worst),
            ):
                metric_block = gate_item[metric_name]
                if not isinstance(metric_block, dict):
                    raise TypeError(
                        "results[%d].gate.results[%d].%s 必须是 JSON 对象"
                        % (idx, gate_idx, metric_name)
                    )
                if (
                    list(metric_block.keys())
                    != _TRENDGATE_OUTPUT_METRIC_KEYS
                ):
                    raise ValueError(
                        "results[%d].gate.results[%d].%s 的键必须依次为 "
                        "regressions、worst_delta、pass"
                        % (idx, gate_idx, metric_name)
                    )

                regressions = metric_block["regressions"]
                if isinstance(regressions, bool) or not isinstance(
                    regressions, int
                ):
                    raise TypeError(
                        "results[%d].gate.results[%d].%s.regressions 必须"
                        "是 int，得到 %s"
                        % (
                            idx,
                            gate_idx,
                            metric_name,
                            type(regressions).__name__,
                        )
                    )
                if regressions != stats_reg:
                    raise ValueError(
                        "results[%d].gate.results[%d].%s.regressions 与 "
                        "stats 逐值不一致：%d != %d"
                        % (idx, gate_idx, metric_name, regressions, stats_reg)
                    )

                worst_delta = metric_block["worst_delta"]
                if isinstance(worst_delta, bool) or not isinstance(
                    worst_delta, float
                ):
                    raise TypeError(
                        "results[%d].gate.results[%d].%s.worst_delta 必须"
                        "是 float，得到 %s"
                        % (
                            idx,
                            gate_idx,
                            metric_name,
                            type(worst_delta).__name__,
                        )
                    )
                if not math.isfinite(worst_delta):
                    raise ValueError(
                        "results[%d].gate.results[%d].%s.worst_delta 必须"
                        "有限" % (idx, gate_idx, metric_name)
                    )
                if worst_delta != stats_worst:
                    raise ValueError(
                        "results[%d].gate.results[%d].%s.worst_delta 与 "
                        "stats 逐值不一致" % (idx, gate_idx, metric_name)
                    )

                metric_pass = metric_block["pass"]
                if not isinstance(metric_pass, bool):
                    raise TypeError(
                        "results[%d].gate.results[%d].%s.pass 必须是 bool，"
                        "得到 %s"
                        % (
                            idx,
                            gate_idx,
                            metric_name,
                            type(metric_pass).__name__,
                        )
                    )
                expected_metric_pass = (
                    regressions
                    <= limits[metric_name]["max_regressions"]
                    and worst_delta
                    >= limits[metric_name]["min_worst_delta"]
                )
                if metric_pass != expected_metric_pass:
                    raise ValueError(
                        "results[%d].gate.results[%d].%s.pass 与阈值/实测"
                        "值不自洽" % (idx, gate_idx, metric_name)
                    )
                if not metric_pass:
                    metric_pass_all = False

                item_regressions_sum += regressions
                if item_worst_min is None or worst_delta < item_worst_min:
                    item_worst_min = worst_delta

            if gate_item_pass != metric_pass_all:
                raise ValueError(
                    "results[%d].gate.results[%d].pass 必须为 ba/f1 pass "
                    "之与" % (idx, gate_idx)
                )
            if not gate_item_pass:
                gate_all_pass = False

        if gate_overall_pass != gate_all_pass:
            raise ValueError(
                "results[%d].gate.pass 必须为各项 pass 之与" % idx
            )
        if item_pass != gate_overall_pass:
            raise ValueError(
                "results[%d].pass 必须与 gate.pass 一致" % idx
            )

        if item_pass:
            counted_pass += 1
        else:
            all_pass = False

        # 负零归零（写出时 _fmt_float 亦会归一）。
        if item_worst_min == 0.0:
            item_worst_min = 0.0
        items.append((name, item_regressions_sum, item_worst_min))
        shapes.append((stats_gate_count, tuple(inner_names)))

    if passed_count != counted_pass:
        raise ValueError(
            "passed_count 必须等于 results 中 pass 为真的项数"
        )
    if overall_pass != all_pass:
        raise ValueError("顶层 pass 必须为各项 pass 之与")
    return audit_count, items, shapes


def _trendauditbatchdiff_compute(baseline_path, current_path):
    """读取并逐项对比两份 trendauditbatch 严格产物，返回 diff 报告 dict。

    两份输入均按 _load_trendauditbatch_output 的 trendauditbatch 产物
    契约严格校验；audit_count 必须相等，且各 results 项 name 顺序必须
    逐项一致（数量相同但顺序/名称不同一律拒绝）。对同名审计项，其 stats
    的 gate_count 与内层 results 的 name 顺序亦须逐项一致，否则拒绝。
    每项的聚合 stats 取 ba/f1：regressions 求和、worst_delta 取最小
    值；regressions_delta、worst_delta_delta 均为当前汇总减基线汇总
    （未舍入 int/float 直接运算），项 pass 当且仅当
    regressions_delta <= 0 且 worst_delta_delta >= 0.0。任何
    worst_delta_delta 非有限即失败（契约上不会发生，防御性校验）。

    返回 dict（键依次为 audit_count、results、pass）：results 每项键
    依次为 name（str）、regressions_delta（int）、worst_delta_delta
    （float）、pass（bool）；顶层 pass 为各项 pass 之与。输入不可读或
    契约/对应不符抛 OSError/UnicodeDecodeError/ValueError/TypeError。
    """
    with open(baseline_path, "rb") as f:
        baseline_raw = f.read()
    with open(current_path, "rb") as f:
        current_raw = f.read()

    base_count, base_items, base_shapes = _load_trendauditbatch_output(
        baseline_raw
    )
    cur_count, cur_items, cur_shapes = _load_trendauditbatch_output(
        current_raw
    )
    if base_count != cur_count:
        raise ValueError(
            "两份产物 audit_count 不一致：%d != %d"
            % (base_count, cur_count)
        )
    for idx, (base_item, cur_item) in enumerate(zip(base_items, cur_items)):
        if base_item[0] != cur_item[0]:
            raise ValueError(
                "results[%d] 的 name 顺序不一致：%r != %r"
                % (idx, base_item[0], cur_item[0])
            )
        base_gate_count, base_inner = base_shapes[idx]
        cur_gate_count, cur_inner = cur_shapes[idx]
        if base_gate_count != cur_gate_count:
            raise ValueError(
                "审计项 %r 的 stats.gate_count 不一致：%d != %d"
                % (base_item[0], base_gate_count, cur_gate_count)
            )
        if base_inner != cur_inner:
            raise ValueError(
                "审计项 %r 的 stats results 内层 name 顺序不一致"
                % base_item[0]
            )

    results = []
    overall = True
    for idx, (
        (name, base_reg, base_worst),
        (_cur_name, cur_reg, cur_worst),
    ) in enumerate(zip(base_items, cur_items)):
        regressions_delta = cur_reg - base_reg
        worst_delta_delta = cur_worst - base_worst
        if not math.isfinite(worst_delta_delta):
            raise ValueError(
                "results[%d].worst_delta_delta 的差值非有限" % idx
            )
        # 负零归零（-0.0 >= 0 为真，写出时 _fmt_float 亦会归一）。
        if worst_delta_delta == 0.0:
            worst_delta_delta = 0.0
        item_pass = (
            regressions_delta <= 0 and worst_delta_delta >= 0.0
        )
        if not item_pass:
            overall = False
        results.append(
            {
                "name": name,
                "regressions_delta": regressions_delta,
                "worst_delta_delta": worst_delta_delta,
                "pass": bool(item_pass),
            }
        )

    return {
        "audit_count": base_count,
        "results": results,
        "pass": bool(overall),
    }


def _cmd_trendauditbatchdiff(baseline_path, current_path, output_path):
    """trendauditbatchdiff 子命令主体；各项均无回归退出 0、合法但存在
    回归退出 3、契约/对应/路径/计算/I-O 失败退出 1 且不改 OUTPUT。

    BASELINE、CURRENT 为两份 trendauditbatch 严格产物，契约见
    _load_trendauditbatch_output：二者 audit_count 与各 results 项
    name 顺序必须一致；每项聚合 stats 的 ba/f1（regressions 求和、
    worst_delta 取最小值）后，regressions_delta、worst_delta_delta
    均为当前汇总减基线汇总（按未舍入值计算），项 pass 仅当
    regressions_delta <= 0 且 worst_delta_delta >= 0。BASELINE、
    CURRENT、OUTPUT 三路径绝对化后须两两不同。

    OUTPUT 原子写出紧凑 UTF-8 JSON（末尾 LF），顶层键依次为
    audit_count（int）、results（list）、pass（bool，各项 pass 之与）；
    results 每项键依次为 name（str）、regressions_delta（int）、
    worst_delta_delta（有限 float，固定 12 位小数、负零归零）、pass
    （bool）；合法但存在回归仍照常原子写盘，退出码以顶层 pass 为准。
    同一输入重复运行逐字节相同，标准输出为空。
    """
    overall = False
    try:
        base_abs = os.path.abspath(baseline_path)
        cur_abs = os.path.abspath(current_path)
        out_abs = os.path.abspath(output_path)
        if base_abs == cur_abs:
            raise ValueError("BASELINE 与 CURRENT 不能是同一路径")
        if base_abs == out_abs:
            raise ValueError("BASELINE 与 OUTPUT 不能是同一路径")
        if cur_abs == out_abs:
            raise ValueError("CURRENT 与 OUTPUT 不能是同一路径")

        report = _trendauditbatchdiff_compute(
            baseline_path, current_path
        )
        overall = report["pass"]
        payload = (_dump_compact(report) + "\n").encode("utf-8")
        _atomic_write_output(out_abs, payload)
    except (ValueError, TypeError, OSError):
        return 1
    return 0 if overall else 3


def main(argv):
    """命令行入口：接受 train/fitcnn/fitnorm/benchmark/benchmark_batches/
    benchmarkdeep OUTPUT、evaluate/evalcnn/evalnorm WEIGHTS OUTPUT、fitdata DATA OUTPUT、
    evaldata/predictdata WEIGHTS DATA OUTPUT、benchmark_data TRAIN VAL
    OUTPUT、gradcheck CONFIG OUTPUT、convcheck CONFIG OUTPUT、
    resumenorm INPUT EPOCHS OUTPUT、resumedata DATA INPUT EPOCHS OUTPUT、
    resumeval TRAIN VAL INPUT EPOCHS OUTPUT、
    resumevalstats TRAIN VAL INPUT EPOCHS OUTPUT、
    valreport STATS VAL OUTPUT、
    valgate STATS VAL CONFIG OUTPUT、
    valgatebatch MANIFEST OUTPUT、
    valgatebatchdiff BASELINE CURRENT OUTPUT、
    valgatebatchtrend MANIFEST OUTPUT、trendstats MANIFEST OUTPUT、
    trendgate STATS CONFIG OUTPUT、
    trendaudit MANIFEST CONFIG OUTPUT、
    trendauditbatch MANIFEST OUTPUT 与
    trendauditbatchdiff BASELINE CURRENT OUTPUT。

    成功 0、参数数目错 2、其余失败 1。
    """
    if len(argv) == 3 and argv[1] == "train":
        return _cmd_train(argv[2])
    if len(argv) == 4 and argv[1] == "evaluate":
        return _cmd_evaluate(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "fitcnn":
        return _cmd_fitcnn(argv[2])
    if len(argv) == 4 and argv[1] == "evalcnn":
        return _cmd_evalcnn(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "fitnorm":
        return _cmd_fitnorm(argv[2])
    if len(argv) == 3 and argv[1] == "benchmark":
        return _cmd_benchmark(argv[2])
    if len(argv) == 3 and argv[1] == "benchmark_batches":
        return _cmd_benchmark_batches(argv[2])
    if len(argv) == 3 and argv[1] == "benchmarkdeep":
        return _cmd_benchmarkdeep(argv[2])
    if len(argv) == 4 and argv[1] == "evalnorm":
        return _cmd_evalnorm(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "fitdata":
        return _cmd_fitdata(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "evaldata":
        return _cmd_evaldata(argv[2], argv[3], argv[4])
    if len(argv) == 5 and argv[1] == "benchmark_data":
        return _cmd_benchmarkdata(argv[2], argv[3], argv[4])
    if len(argv) == 5 and argv[1] == "predictdata":
        return _cmd_predictdata(argv[2], argv[3], argv[4])
    if len(argv) == 4 and argv[1] == "gradcheck":
        return _cmd_gradcheck(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "convcheck":
        return _cmd_convcheck(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "resumenorm":
        return _cmd_resumenorm(argv[2], argv[3], argv[4])
    if len(argv) == 6 and argv[1] == "resumedata":
        return _cmd_resumedata(argv[2], argv[3], argv[4], argv[5])
    if len(argv) == 7 and argv[1] == "resumeval":
        return _cmd_resumeval(argv[2], argv[3], argv[4], argv[5], argv[6])
    if len(argv) == 7 and argv[1] == "resumevalstats":
        return _cmd_resumevalstats(
            argv[2], argv[3], argv[4], argv[5], argv[6]
        )
    if len(argv) == 5 and argv[1] == "valreport":
        return _cmd_valreport(argv[2], argv[3], argv[4])
    if len(argv) == 6 and argv[1] == "valgate":
        return _cmd_valgate(argv[2], argv[3], argv[4], argv[5])
    if len(argv) == 4 and argv[1] == "valgatebatch":
        return _cmd_valgatebatch(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "valgatebatchdiff":
        return _cmd_valgatebatchdiff(argv[2], argv[3], argv[4])
    if len(argv) == 4 and argv[1] == "valgatebatchtrend":
        return _cmd_valgatebatchtrend(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "trendstats":
        return _cmd_trendstats(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "trendgate":
        return _cmd_trendgate(argv[2], argv[3], argv[4])
    if len(argv) == 5 and argv[1] == "trendaudit":
        return _cmd_trendaudit(argv[2], argv[3], argv[4])
    if len(argv) == 4 and argv[1] == "trendauditbatch":
        return _cmd_trendauditbatch(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "trendauditbatchdiff":
        return _cmd_trendauditbatchdiff(argv[2], argv[3], argv[4])
    if len(argv) >= 2 and argv[1] in (
        "train",
        "evaluate",
        "fitcnn",
        "evalcnn",
        "fitnorm",
        "benchmark",
        "benchmark_batches",
        "benchmarkdeep",
        "evalnorm",
        "fitdata",
        "evaldata",
        "benchmark_data",
        "predictdata",
        "gradcheck",
        "convcheck",
        "resumenorm",
        "resumedata",
        "resumeval",
        "resumevalstats",
        "valreport",
        "valgate",
        "valgatebatch",
        "valgatebatchdiff",
        "valgatebatchtrend",
        "trendstats",
        "trendgate",
        "trendaudit",
        "trendauditbatch",
        "trendauditbatchdiff",
    ):
        return 2
    # 其他入口保持现状（信息打印）。
    print("convnet.py：从零实现的卷积神经网络库（仅标准库）。")
    print("当前可用组件：Conv2D(weights, bias, stride=1, padding=0, dilation=1)")
    print("              MaxPool2D(kernel_size, stride=None, padding=0)")
    print("              Flatten()")
    print("              Linear(weights, bias)")
    print("              ReLU()")
    print("              Dropout(p=0.5, seed=0)")
    print("              Dropout2D(p=0.5, seed=0)")
    print("              BatchNorm2D(gamma, beta, eps=1e-5, momentum=0.1)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
