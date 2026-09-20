"""convnet.py — 从零实现的卷积神经网络组件（仅 Python 标准库）。

当前提供：
- Conv2D 层：NCHW 嵌套 list、互相关（不翻转核）、零补边。
- MaxPool2D 层：NCHW 嵌套 list、逐通道最大池化、补边位置不参与比较。
- Flatten 层：NCHW 嵌套 list 展平为 [N][C*H*W]（按 c→h→w 顺序）。
- Linear 层：全连接，weights [O][I]、bias [O]，输入 [N][I] 输出 [N][O]。
- ReLU 层：逐元素 max(0, v)，限二维 [N][D]。
"""

import math


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


def _zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class Conv2D:
    """二维互相关层（NCHW，嵌套 list，零补边）。

    weights: [O][C][KH][KW]，bias: [O]，输入 x: [N][C][H][W]。
    输出: [N][O][floor((H+2P-KH)/S)+1][floor((W+2P-KW)/S)+1]。
    """

    def __init__(self, weights, bias, stride=1, padding=0):
        stride = _check_nonnegative_int(stride, "stride")
        if stride <= 0:
            raise ValueError("stride 必须为正整数")
        padding = _check_nonnegative_int(padding, "padding")
        if padding < 0:
            raise ValueError("padding 必须为非负整数")

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
        self._stride = stride
        self._padding = padding
        self._w_shape = w_shape  # (O, C, KH, KW)

        self._x = None           # 最近一次成功 forward 的输入
        self._out_shape = None   # 最近一次成功 forward 的输出形状

    def forward(self, x):
        """对 x: [N][C][H][W] 做互相关，返回嵌套 list 并缓存输入。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        o_ch, w_c, kh_, kw_ = self._w_shape
        if c_ != w_c:
            raise ValueError(
                "输入通道数 %d 与 weights 通道数 %d 不符" % (c_, w_c)
            )
        s = self._stride
        p = self._padding
        if kh_ > h_ + 2 * p or kw_ > w_ + 2 * p:
            raise ValueError("核在补边后仍越界：核尺寸大于补边后的输入")
        oh_ = (h_ + 2 * p - kh_) // s + 1
        ow_ = (w_ + 2 * p - kw_) // s + 1

        weights = self._weights
        bias = self._bias
        out = []
        for n in range(n_):
            out_n = []
            for o in range(o_ch):
                out_o = []
                for oh in range(oh_):
                    row = []
                    base_h = oh * s - p
                    for ow in range(ow_):
                        base_w = ow * s - p
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
        未成功 forward 前调用一律抛 ValueError。
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
        s = self._stride
        p = self._padding

        dx = _zeros((n_, c_, h_, w_))
        dw = _zeros(self._w_shape)
        db = _zeros((o_ch,))

        for n in range(n_):
            for o in range(o_ch):
                for oh in range(oh_):
                    base_h = oh * s - p
                    for ow in range(ow_):
                        g = dy[n][o][oh][ow]
                        db[o] += g
                        base_w = ow * s - p
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

    kernel_size: 正方形窗口边长 K（正 int）。
    stride: 步长 S（正 int）；为 None 时取 S = K。
    padding: 零补边宽度 P（非负 int 且 P < K）；补边位置不参与比较。
    输入 x: [N][C][H][W]，输出: [N][C][OH][OW]，
    OH = (H + 2P - K) // S + 1，OW = (W + 2P - K) // S + 1。
    窗口内按 kh→kw 扫描，并列最大只取首个位置。
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        kernel_size = _check_nonnegative_int(kernel_size, "kernel_size")
        if kernel_size <= 0:
            raise ValueError("kernel_size 必须为正整数")
        if stride is None:
            stride = kernel_size
        else:
            stride = _check_nonnegative_int(stride, "stride")
            if stride <= 0:
                raise ValueError("stride 必须为正整数")
        padding = _check_nonnegative_int(padding, "padding")
        if padding < 0:
            raise ValueError("padding 必须为非负整数")
        if padding >= kernel_size:
            raise ValueError("padding 必须小于 kernel_size")

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self._x_shape = None   # 最近一次成功 forward 的输入形状
        self._out_shape = None  # 最近一次成功 forward 的输出形状
        self._winners = None   # 每个输出位置的最大值来源 (ih, iw)

    def forward(self, x):
        """对 x: [N][C][H][W] 做最大池化，返回新 list 并缓存获胜位置。"""
        _require_list(x, "x")
        n_, c_, h_, w_ = _shape_of(x, 4, "x")
        k = self._kernel_size
        s = self._stride
        p = self._padding
        if k > h_ + 2 * p or k > w_ + 2 * p:
            raise ValueError("池化窗口在补边后仍越界：kernel_size 大于补边后的输入")
        oh_ = (h_ + 2 * p - k) // s + 1
        ow_ = (w_ + 2 * p - k) // s + 1

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
                    base_h = oh * s - p
                    row = []
                    win_row = []
                    for ow in range(ow_):
                        base_w = ow * s - p
                        best = None
                        best_pos = None
                        for kh in range(k):
                            ih = base_h + kh
                            if ih < 0 or ih >= h_:
                                continue
                            x_row = x_c[ih]
                            for kw in range(k):
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

    layer 限 Conv2D/MaxPool2D/Flatten/Linear/ReLU 实例，其余抛 TypeError。
    解析梯度 a 取自原值 forward(x) 后 backward(dy) 的对应返回（有参层
    还包含 dweights、dbias）。对每个标量 v，定义标量损失 L：acc=0.0，
    按输出嵌套索引从外到内递增执行 acc += y*dy（y 为前向输出），数值
    梯度 n = (L(v+eps) - L(v-eps)) / (2*eps)。

    令 e = abs(a - n)、r = e / max(abs(a), abs(n), 1e-12)，返回
    (ok, max(e), max(r))，类型固定 (bool, float, float)，不舍入；
    ok 当且仅当每项 e <= atol + rtol * max(abs(a), abs(n))。

    eps/atol/rtol 须为有限 int/float（拒绝 bool）：类型错抛 TypeError；
    eps 非正、容差为负或任一非有限抛 ValueError。x、dy 的校验及异常
    完全沿用对应层的 forward/backward；计算产生非有限值抛 ValueError。
    x、dy、参数及实例状态在所有成功或异常路径均原样恢复。
    """
    if not isinstance(layer, (Conv2D, MaxPool2D, Flatten, Linear, ReLU)):
        raise TypeError(
            "layer 必须是 Conv2D/MaxPool2D/Flatten/Linear/ReLU 实例，得到 %s"
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

    saved_state = dict(layer.__dict__)
    try:
        layer.forward(x)  # x 的校验沿用该层 forward
        grad = layer.backward(dy)  # dy 的校验沿用该层 backward
        if isinstance(layer, (Conv2D, Linear)):
            dx, dw, db = grad
            targets = (
                ("x", x, dx),
                ("weights", layer._weights, dw),
                ("bias", layer._bias, db),
            )
        else:
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
        layer.__dict__.clear()
        layer.__dict__.update(saved_state)

    return (bool(ok), float(max_e), float(max_r))


if __name__ == "__main__":
    print("convnet.py：从零实现的卷积神经网络库（仅标准库）。")
    print("当前可用组件：Conv2D(weights, bias, stride=1, padding=0)")
    print("              MaxPool2D(kernel_size, stride=None, padding=0)")
    print("              Flatten()")
    print("              Linear(weights, bias)")
    print("              ReLU()")
