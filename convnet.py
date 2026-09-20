"""convnet.py — 从零实现的卷积神经网络组件（仅 Python 标准库）。

当前提供：
- Conv2D 层：NCHW 嵌套 list、互相关（不翻转核）、零补边。
- MaxPool2D 层：NCHW 嵌套 list、逐通道最大池化、补边位置不参与比较。
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


if __name__ == "__main__":
    print("convnet.py：从零实现的卷积神经网络库（仅标准库）。")
    print("当前可用组件：Conv2D(weights, bias, stride=1, padding=0)")
    print("              MaxPool2D(kernel_size, stride=None, padding=0)")
