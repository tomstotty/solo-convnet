"""convnet.py — 从零实现的卷积神经网络组件（仅 Python 标准库）。

当前提供：
- Conv2D 层：NCHW 嵌套 list、互相关（不翻转核）、零补边；stride 可为
  正 int 或 (SH, SW) tuple，padding 可为非负 int 或 (PT,PB,PL,PR) tuple。
- MaxPool2D 层：NCHW 嵌套 list、逐通道最大池化、补边位置不参与比较。
- Flatten 层：NCHW 嵌套 list 展平为 [N][C*H*W]（按 c→h→w 顺序）。
- Linear 层：全连接，weights [O][I]、bias [O]，输入 [N][I] 输出 [N][O]。
- ReLU 层：逐元素 max(0, v)，限二维 [N][D]。
- Dropout 层：NCHW 嵌套 list，训练态按概率 p 置零并放大保留项，推理态原样复制。
- BatchNorm2D 层：NCHW 嵌套 list，逐通道批归一化；训练态按批次统计并更新
  running_mean/running_var，推理态使用运行统计仿射。
- SoftmaxCrossEntropy 层：二维 logits [N][K] 与 labels [N] 的批均
  softmax 交叉熵损失，反向返回对 logits 的梯度。

- 公开推理 API（仅标准库）：
- load_model(path)：严格校验 train 产物后返回键序为 values、bias 的新 dict。
- predict_batch(model, x)：对 [N][1][1][4] 输入逐样本计算 logit，返回预测类别。

公开训练 API（仅标准库）：
- train_norm_step(layers, x, labels, lr=0.1)：对
  Conv2D→BatchNorm2D→Dropout→MaxPool2D→Flatten→Linear→SoftmaxCrossEntropy
  七层按序前向、自损失层起逆序反传，做一次同步 SGD 更新，返回 float
  批均损失；任何异常都把七层恢复到入口状态。
- train_norm(layers, x, labels, epochs=20, lr=0.1)：连续训练 epochs 轮，
  返回各轮更新前批均损失的新 list[float]；末轮更新后仅以更新后的
  Conv2D 输出做一次训练态 BN 前向刷新运行统计；任何异常都把七层
  整体恢复到函数入口状态。

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
- `python convnet.py gradcheck CONFIG OUTPUT`：读取 UTF-8 JSON CONFIG
  （键依次为 x、labels、eps、atol、rtol，x 为有限数的规则
  list[N][1][2][2]、N≥1），新建与 fitnorm 初始参数、层配置相同的七层
  训练链并以 SoftmaxCrossEntropy 为损失层调用 check_train_gradients，
  将 ok、max_e、max_r 以紧凑 JSON 原子写入 OUTPUT；ok 真退出 0、
  假退出 1，其余失败退出 1 且不改 OUTPUT。
"""

import json
import math
import os
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


def _zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class Conv2D:
    """二维互相关层（NCHW，嵌套 list，零补边）。

    weights: [O][C][KH][KW]，bias: [O]，输入 x: [N][C][H][W]。
    stride: 正 int（展开为 (S, S)）或恰含 (SH, SW) 的正 int tuple。
    padding: 非负 int（展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边。
    输出: [N][O][floor((H+PT+PB-KH)/SH)+1][floor((W+PL+PR-KW)/SW)+1]；
    不能整除时舍弃底部或右侧余量。
    """

    def __init__(self, weights, bias, stride=1, padding=0):
        sh_, sw_ = _check_stride2d(stride)
        pt_, pb_, pl_, pr_ = _check_padding2d(padding)

        _require_list(weights, "weights")
        _require_list(bias, "bias")
        w_shape = _shape_of(weights, 4, "weights")
        b_shape = _shape_of(bias, 1, "bias")
        if b_shape[0] != w_shape[0]:
            raise ValueError(
                "bias 长度 %d 与 weights 输出通道数 %d 不符"
                % (b_shape[0], w_shape[0])
            )

        self._weights = weights
        self._bias = bias
        self._stride = (sh_, sw_)
        self._padding = (pt_, pb_, pl_, pr_)
        self._w_shape = w_shape  # (O, C, KH, KW)

        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][C][H][W] 做零补边互相关，返回嵌套 list 并缓存输入。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        o_ch, w_c, kh_, kw_ = self._w_shape
        if c_ != w_c:
            raise ValueError(
                "输入通道数 %d 与 weights 通道数 %d 不符" % (c_, w_c)
            )
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        if kh_ > h_ + pt_ + pb_ or kw_ > w_ + pl_ + pr_:
            raise ValueError("核在补边后仍越界：核尺寸大于补边后的输入")
        oh_ = (h_ + pt_ + pb_ - kh_) // sh_ + 1
        ow_ = (w_ + pl_ + pr_ - kw_) // sw_ + 1

        weights = self._weights
        bias = self._bias
        out = []
        for n in range(n_):
            out_n = []
            for o in range(o_ch):
                out_o = []
                for oh in range(oh_):
                    row = []
                    base_h = oh * sh_ - pt_
                    for ow in range(ow_):
                        base_w = ow * sw_ - pl_
                        acc = bias[o]
                        for c in range(c_):
                            x_c = x[n][c]
                            w_c_o = weights[o][c]
                            for kh in range(kh_):
                                ih = base_h + kh
                                if ih < 0 or ih >= h_:
                                    continue
                                x_row = x_c[ih]
                                w_row = w_c_o[kh]
                                for kw in range(kw_):
                                    iw = base_w + kw
                                    if 0 <= iw < w_:
                                        acc += x_row[iw] * w_row[kw]
                        row.append(acc)
                    out_o.append(row)
                out_n.append(out_o)
            out.append(out_n)

        self._x = x
        self._out_shape = (n_, o_ch, oh_, ow_)
        return out

    def backward(self, dy):
        """根据上游梯度 dy 返回 (dx, dweights, dbias)。

        dy 的形状必须等于最近一次成功 forward 的输出形状。
        未成功 forward 前调用一律抛 ValueError。补边位置不产生 dx，
        多个输出位置对同一输入坐标的梯度在此累加。
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
        _, c_, kh_, kw_ = self._w_shape
        h_ = len(x[0][0])
        w_ = len(x[0][0][0])
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding

        dx = _zeros((n_, c_, h_, w_))
        dw = _zeros(self._w_shape)
        db = _zeros((o_ch,))

        for n in range(n_):
            for o in range(o_ch):
                for oh in range(oh_):
                    base_h = oh * sh_ - pt_
                    for ow in range(ow_):
                        g = dy[n][o][oh][ow]
                        db[o] += g
                        base_w = ow * sw_ - pl_
                        for c in range(c_):
                            x_c = x[n][c]
                            dx_c = dx[n][c]
                            w_c_o = weights[o][c]
                            dw_c_o = dw[o][c]
                            for kh in range(kh_):
                                ih = base_h + kh
                                if ih < 0 or ih >= h_:
                                    continue
                                x_row = x_c[ih]
                                dx_row = dx_c[ih]
                                w_row = w_c_o[kh]
                                dw_row = dw_c_o[kh]
                                for kw in range(kw_):
                                    iw = base_w + kw
                                    if 0 <= iw < w_:
                                        dw_row[kw] += g * x_row[iw]
                                        dx_row[iw] += g * w_row[kw]
        return dx, dw, db


class MaxPool2D:
    """二维最大池化层（NCHW，嵌套 list，逐通道池化）。

    kernel_size: 正 int（双轴同值，展开为 (K, K)）或恰含 (KH, KW) 的
    正 int tuple。
    stride: 正 int（双轴同值）或恰含 (SH, SW) 的正 int tuple；为 None
    时取 (KH, KW)。
    padding: 非负 int（四边同值，展开为 (P, P, P, P)）或恰含
    (PT, PB, PL, PR) 的非负 int tuple，分别为上/下/左/右补边；且
    PT、PB < KH，PL、PR < KW。补边位置不参与比较。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]，
    OH = (H + PT + PB - KH) // SH + 1，
    OW = (W + PL + PR - KW) // SW + 1。
    不能整除时舍弃底部或右侧余量。窗口内按 kh→kw 扫描，并列最大只取
    首个真实坐标（补边不计入扫描）。
    """

    def __init__(self, kernel_size, stride=None, padding=0):
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

        self._kernel_size = (kh_, kw_)
        self._stride = (sh_, sw_)
        self._padding = (pt_, pb_, pl_, pr_)

        self._x_shape = None   # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状
        self._winners = None   # 每个输出位置的最大值来源 (ih, iw)

    def forward(self, x):
        """对 x: [N][C][H][W] 做最大池化，返回新 list 并缓存获胜位置。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        kh_, kw_ = self._kernel_size
        sh_, sw_ = self._stride
        pt_, pb_, pl_, pr_ = self._padding
        if kh_ > h_ + pt_ + pb_ or kw_ > w_ + pl_ + pr_:
            raise ValueError("池化窗口在补边后仍越界：kernel_size 大于补边后的输入")
        oh_ = (h_ + pt_ + pb_ - kh_) // sh_ + 1
        ow_ = (w_ + pl_ + pr_ - kw_) // sw_ + 1

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
                            ih = base_h + kh
                            if ih < 0 or ih >= h_:
                                continue
                            x_row = x_c[ih]
                            for kw in range(kw_):
                                iw = base_w + kw
                                if 0 <= iw < w_:
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
        """根据上游梯度 dy 返回与输入同形状的 dx。

        dy 的形状必须等于最近一次成功 forward 的输出形状；
        梯度按 n→c→oh→ow 顺序累加到 forward 记录的获胜位置。
        未成功 forward 前调用一律抛 ValueError。
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
                        dx_c[ih][iw] += dy_row[ow]
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

    forward(logits, labels)：logits 为非空规则嵌套 list[N][K]（N、K ≥ 1），
    元素为有限 int/float（拒绝 bool）；labels 为长度 N 的 list，元素为
    [0, K) 内的 int（拒绝 bool）。逐行取 m = max(row)，按 k 递增求
    e[k] = exp(row[k] - m)、s = Σe、p[k] = e[k]/s，再按 n 递增累计
    m + log(s) - row[label] 并除以 N，返回 float 批均损失。
    仅成功时以新 list 缓存 p 与 labels 并覆盖旧缓存，失败保留旧缓存。

    backward()：返回新 list[N][K]，元素为
    (p[n][k] - (k == label[n] ? 1 : 0)) / N，均为 float。
    未成功 forward 前调用一律抛 ValueError；重复调用返回等值独立列表。
    两个方法均不修改实参。
    """

    def __init__(self):
        self._probs = None     # 最近一次成功 forward 的 softmax 概率
        self._labels = None    # 最近一次成功 forward 的标签副本
        self._out_shape = None  # 最近一次成功 forward 的 (N, K)

    def forward(self, logits, labels):
        """计算批均 softmax 交叉熵损失，返回 float 并缓存概率与标签。"""
        _require_list(logits, "logits")
        _require_list(labels, "labels")
        n_, k_ = _shape_of(logits, 2, "logits")
        if len(labels) != n_:
            raise ValueError(
                "labels 长度 %d 与 logits 样本数 %d 不符"
                % (len(labels), n_)
            )
        for n in range(n_):
            label = labels[n]
            if isinstance(label, list):
                raise ValueError("labels 的层级过深：标量位置出现了 list")
            if isinstance(label, bool) or not isinstance(label, int):
                raise TypeError(
                    "labels 的元素必须是 int（拒绝 bool），得到 %s"
                    % type(label).__name__
                )
            if label < 0 or label >= k_:
                raise ValueError(
                    "labels[%d] = %d 超出 [0, %d) 范围" % (n, label, k_)
                )

        probs = []
        loss_sum = 0.0
        for n in range(n_):
            row = logits[n]
            m = row[0]
            for k in range(1, k_):
                if row[k] > m:
                    m = row[k]
            exps = []
            s = 0.0
            for k in range(k_):
                e = math.exp(row[k] - m)
                exps.append(e)
                s += e
            p = [e / s for e in exps]
            probs.append(p)
            loss_sum += m + math.log(s) - row[labels[n]]
        loss = loss_sum / n_

        for n in range(n_):
            for k in range(k_):
                if not math.isfinite(probs[n][k]):
                    raise ValueError("前向计算产生非有限值（NaN/inf）")
        if not math.isfinite(loss):
            raise ValueError("前向计算产生非有限值（NaN/inf）")

        self._probs = probs
        self._labels = list(labels)
        self._out_shape = (n_, k_)
        return loss

    def backward(self):
        """返回 logits 的梯度新 list[N][K]，元素均为 float。

        元素为 (p[n][k] - (k 等于 label[n] 时为 1，否则为 0)) / N；
        未成功 forward 前调用一律抛 ValueError。
        """
        if self._probs is None:
            raise ValueError("尚未成功执行 forward，无法 backward")

        n_, k_ = self._out_shape
        probs = self._probs
        labels = self._labels
        dx = []
        for n in range(n_):
            p_row = probs[n]
            label = labels[n]
            row = []
            for k in range(k_):
                row.append(
                    (p_row[k] - (1.0 if k == label else 0.0)) / n_
                )
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

    layer 限 Conv2D/MaxPool2D/Flatten/Linear/ReLU/Dropout/BatchNorm2D
    实例，其余抛 TypeError。解析梯度 a 取自原值 forward(x) 后 backward(dy)
    的对应返回：Conv2D/Linear 还包含 dweights、dbias；BatchNorm2D 按
    x、gamma、beta 顺序检查 dx、dgamma、dbeta；Dropout 只检查 x。
    对每个标量 v，定义标量损失 L：acc=0.0，按输出嵌套索引从外到内递增
    执行 acc += y*dy（y 为前向输出），数值梯度
    n = (L(v+eps) - L(v-eps)) / (2*eps)，各目标内部标量按嵌套序遍历。

    BatchNorm2D 仅在训练态检查，推理态一律抛 ValueError；每次数值前向都
    重新按当前批次统计（gamma/beta 扰动不影响归一化值 z）。
    Dropout 训练态以入口随机状态 _s 为基准：解析梯度前向及每次正、负
    扰动前向之前都把 _s 恢复为入口值，使各次前向重放同一掩码，故同一
    入口状态结果确定；推理态不推进随机状态，按恒等映射检查。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    校验顺序固定为 layer、eps/atol/rtol、BatchNorm2D 模式、forward(x)、
    backward(dy)：layer 或数值参数类型错抛 TypeError；BatchNorm2D 推理态、
    eps 非正、容差为负或任一参数非有限抛 ValueError；x、dy 的校验及异常
    完全沿用对应层的 forward/backward（形状或非有限错误抛 ValueError）；
    计算产生非有限值抛 ValueError。x、dy、参数、训练/推理模式、Dropout
    随机状态与掩码、BatchNorm2D 运行统计与旧缓存在所有成功或异常路径均
    原样恢复，实参内容不变。
    """
    if not isinstance(
        layer,
        (Conv2D, MaxPool2D, Flatten, Linear, ReLU, Dropout, BatchNorm2D),
    ):
        raise TypeError(
            "layer 必须是 Conv2D/MaxPool2D/Flatten/Linear/ReLU/"
            "Dropout/BatchNorm2D 实例，得到 %s" % type(layer).__name__
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
    is_dropout = isinstance(layer, Dropout)
    if is_batchnorm and not layer._training:
        raise ValueError("BatchNorm2D 仅在训练态支持梯度检查")

    saved_state = dict(layer.__dict__)
    dropout_entry_s = layer._s if is_dropout else None
    try:
        # Dropout 训练态：解析梯度前向先回到入口随机状态，掩码随后可重放。
        if is_dropout and layer._training:
            layer._s = dropout_entry_s
        layer.forward(x)  # x 的校验沿用该层 forward
        grad = layer.backward(dy)  # dy 的校验沿用该层 backward
        if isinstance(layer, (Conv2D, Linear)):
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
            # MaxPool2D/Flatten/ReLU/Dropout（训练态与推理态）只检查 x。
            targets = (("x", x, grad),)

        def loss(x_arg):
            y = layer.forward(x_arg)
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

    窗口/步长/补边规则与 MaxPool2D.forward 完全一致（补边位置不参与比较）：
    窗口内同一最大值出现两次及以上即视为并列。inp 须为池化层合法四维输入。
    """
    kh_, kw_ = pool._kernel_size
    sh_, sw_ = pool._stride
    pt_, pb_, pl_, pr_ = pool._padding
    n_ = len(inp)
    c_ = len(inp[0])
    h_ = len(inp[0][0])
    w_ = len(inp[0][0][0])
    oh_ = (h_ + pt_ + pb_ - kh_) // sh_ + 1
    ow_ = (w_ + pl_ + pr_ - kw_) // sw_ + 1
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
                        ih = base_h + kh
                        if ih < 0 or ih >= h_:
                            continue
                        x_row = x_c[ih]
                        for kw in range(kw_):
                            iw = base_w + kw
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


def check_cnn_gradients(
    conv, pool, flatten, linear, x, dy, eps=1e-6, atol=1e-6, rtol=1e-4
):
    """用中心差分数值梯度检验 Conv2D→MaxPool2D→Flatten→Linear 整链。

    conv/pool/flatten/linear 须依次为 Conv2D/MaxPool2D/Flatten/Linear 实例，
    其余抛 TypeError。前向按 conv→pool→flatten→linear 执行，解析梯度按
    linear→flatten→pool→conv 逆序取各层 backward 结果。标量损失 L：
    acc=0.0，按线性层输出的嵌套索引从外到内递增执行 acc += y*dy（y 为
    整链前向输出）。

    数值梯度依次扰动 x、conv 的 weights/bias、linear 的 weights/bias
    （各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    任一次前向中任一池化有效窗口并列最大（补边位置不参与比较）一律抛
    ValueError——max 在并列点梯度无定义。

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
    if not isinstance(pool, MaxPool2D):
        raise TypeError(
            "pool 必须是 MaxPool2D 实例，得到 %s" % type(pool).__name__
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
            _check_pool_window_ties(pool, conv_out)
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
        # 池化并列最大在此一并检出。
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
    """用中心差分数值梯度检验 Conv2D→BatchNorm2D→Dropout→MaxPool2D→Flatten→Linear 训练链。

    conv/bn/dropout/pool/flatten/linear 须依次为 Conv2D/BatchNorm2D/Dropout/
    MaxPool2D/Flatten/Linear 实例，其余抛 TypeError；bn 或 dropout 非训练态
    抛 ValueError。前向按 conv→bn→dropout→pool→flatten→linear 执行，解析
    梯度按 linear→flatten→pool→dropout→bn→conv 逆序取各层 backward 结果。
    标量损失 L：acc=0.0，按线性层输出的嵌套索引从外到内递增执行
    acc += y*dy（y 为整链前向输出）。

    数值梯度依次扰动 x、conv 的 weights/bias、bn 的 gamma/beta、linear 的
    weights/bias（各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    每次前向（含解析梯度前向与每次正、负扰动前向）之前都把 dropout 的随机
    状态 _s 恢复为入口值，使各次前向重放同一掩码，故同一入口状态结果确定。
    BatchNorm2D 每次数值前向都重新按当前批次统计。任一次前向中任一池化
    有效窗口并列最大（补边位置不参与比较）一律抛 ValueError——max 在
    并列点梯度无定义。

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
    if not isinstance(pool, MaxPool2D):
        raise TypeError(
            "pool 必须是 MaxPool2D 实例，得到 %s" % type(pool).__name__
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
            _check_pool_window_ties(pool, drop_out)
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
        # forward，池化并列最大在此一并检出。
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
    """用中心差分数值梯度检验 Conv2D→BatchNorm2D→Dropout→MaxPool2D→Flatten→Linear→SoftmaxCrossEntropy 训练链。

    conv/bn/dropout/pool/flatten/linear 须依次为 Conv2D/BatchNorm2D/Dropout/
    MaxPool2D/Flatten/Linear 实例，loss 须为 SoftmaxCrossEntropy 实例，
    其余抛 TypeError；bn 或 dropout 非训练态抛 ValueError。前向按
    conv→bn→dropout→pool→flatten→linear 执行得 logits，标量损失
    L = loss.forward(logits, labels) 返回的 float 批均损失；解析梯度自
    loss.backward() 起按 linear→flatten→pool→dropout→bn→conv 逆序取各层
    backward 结果。

    数值梯度依次扰动 x、conv 的 weights/bias、bn 的 gamma/beta、linear 的
    weights/bias（各张量内部按嵌套序），n = (L(v+eps) - L(v-eps)) / (2*eps)。
    每次前向（含解析梯度前向与每次正、负扰动前向）之前都把 dropout 的随机
    状态 _s 恢复为入口值，使各次前向重放同一掩码，故同一入口状态结果确定。
    BatchNorm2D 每次数值前向都重新按当前批次统计。任一次前向中任一池化
    有效窗口并列最大（补边位置不参与比较）一律抛 ValueError——max 在
    并列点梯度无定义。

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
    if not isinstance(pool, MaxPool2D):
        raise TypeError(
            "pool 必须是 MaxPool2D 实例，得到 %s" % type(pool).__name__
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
            _check_pool_window_ties(pool, drop_out)
            pool_out = pool.forward(drop_out)
            flat = flatten.forward(pool_out)
            return linear.forward(flat)

        def loss_value(x_arg):
            return loss.forward(chain_forward(x_arg), labels)

        # 前向按 conv→bn→dropout→pool→flatten→linear→loss；x 的校验沿
        # 各层 forward，labels 的校验沿 loss.forward，池化并列最大在此
        # 一并检出。
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
# 命令行数据驱动训练：python convnet.py fitdata DATA OUTPUT
# ---------------------------------------------------------------------------

_FITDATA_KEYS = ["x", "labels"]


def _load_fitdata(data_path):
    """读取并严格校验 fitdata 的 DATA，返回 (x, labels)。

    DATA 须为 UTF-8 JSON 对象，键仅依次为 x、labels，重复、缺失、额外或
    错序键一律非法；x 须为有限 int/float（拒绝 bool）的规则
    list[N][1][2][2] 且 N>=2；labels 须为与 x 等长的 list，元素为 int
    0 或 1（拒绝 bool）。文件不可读、UTF-8/JSON 非法或结构/类型/取值
    不符时抛 OSError/ValueError/TypeError，返回的列表不与原文共享标量
    之外的可变结构（json 解析结果本身即为新建）。
    """
    with open(data_path, "rb") as f:
        raw = f.read()
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


# ---------------------------------------------------------------------------
# 公开训练 API：train_norm_step(layers, x, labels, lr=0.1) 与
# train_norm(layers, x, labels, epochs=20, lr=0.1)
# ---------------------------------------------------------------------------

# 七层每个实例的全部可变状态：缓存（每次成功 forward 覆盖）、模式无关的
# 随机/运行统计，以及被更新的参数 list 本体。
def _snapshot_layers(layers):
    conv, bn, dropout, pool, flatten, linear, loss = layers
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
        "pool": {
            "x_shape": pool._x_shape, "out_shape": pool._out_shape,
            "winners": pool._winners,
        },
        "flatten": {
            "x_shape": flatten._x_shape, "out_shape": flatten._out_shape,
        },
        "linear": {
            "weights": linear._weights, "bias": linear._bias,
            "x": linear._x, "out_shape": linear._out_shape,
        },
        "loss": {
            "probs": loss._probs, "labels": loss._labels,
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
    pool._x_shape, pool._out_shape, pool._winners = (
        snap["pool"]["x_shape"], snap["pool"]["out_shape"],
        snap["pool"]["winners"],
    )
    flatten._x_shape, flatten._out_shape = (
        snap["flatten"]["x_shape"], snap["flatten"]["out_shape"]
    )
    linear._weights, linear._bias = (
        snap["linear"]["weights"], snap["linear"]["bias"]
    )
    linear._x, linear._out_shape = (
        snap["linear"]["x"], snap["linear"]["out_shape"]
    )
    loss._probs, loss._labels, loss._out_shape = (
        snap["loss"]["probs"], snap["loss"]["labels"],
        snap["loss"]["out_shape"],
    )


def _validate_train_norm_args(layers, lr):
    """train_norm_step 与 train_norm 共用的 layers/lr 校验。

    layers 必须是恰含 Conv2D/BatchNorm2D/Dropout/MaxPool2D/Flatten/
    Linear/SoftmaxCrossEntropy 七层实例（类型与顺序均固定）的 list：
    容器或成员类型错抛 TypeError，长度错抛 ValueError；BatchNorm2D 与
    Dropout 必须处于训练态，否则抛 ValueError。lr 必须是正的有限
    int/float（拒绝 bool）：类型错抛 TypeError，非有限或非正抛
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
        Conv2D, BatchNorm2D, Dropout, MaxPool2D,
        Flatten, Linear, SoftmaxCrossEntropy,
    )
    names = (
        "Conv2D", "BatchNorm2D", "Dropout", "MaxPool2D",
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
    """七层网络（Conv2D/BN/Dropout/MaxPool/Flatten/Linear/SoftmaxCE）的
    一步标准化训练：按列表顺序前向，自损失层 backward() 起逆序反传，
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


def train_norm(layers, x, labels, epochs=20, lr=0.1):
    """七层网络（Conv2D/BN/Dropout/MaxPool/Flatten/Linear/SoftmaxCE）的
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


def main(argv):
    """命令行入口：接受 train/fitcnn/fitnorm OUTPUT、
    evaluate/evalcnn/evalnorm WEIGHTS OUTPUT、fitdata DATA OUTPUT、
    evaldata WEIGHTS DATA OUTPUT 与 gradcheck CONFIG OUTPUT。

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
    if len(argv) == 4 and argv[1] == "evalnorm":
        return _cmd_evalnorm(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "fitdata":
        return _cmd_fitdata(argv[2], argv[3])
    if len(argv) == 5 and argv[1] == "evaldata":
        return _cmd_evaldata(argv[2], argv[3], argv[4])
    if len(argv) == 4 and argv[1] == "gradcheck":
        return _cmd_gradcheck(argv[2], argv[3])
    if len(argv) >= 2 and argv[1] in (
        "train",
        "evaluate",
        "fitcnn",
        "evalcnn",
        "fitnorm",
        "evalnorm",
        "fitdata",
        "evaldata",
        "gradcheck",
    ):
        return 2
    # 其他入口保持现状（信息打印）。
    print("convnet.py：从零实现的卷积神经网络库（仅标准库）。")
    print("当前可用组件：Conv2D(weights, bias, stride=1, padding=0)")
    print("              MaxPool2D(kernel_size, stride=None, padding=0)")
    print("              Flatten()")
    print("              Linear(weights, bias)")
    print("              ReLU()")
    print("              Dropout(p=0.5, seed=0)")
    print("              BatchNorm2D(gamma, beta, eps=1e-5, momentum=0.1)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
