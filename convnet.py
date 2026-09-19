"""从零实现的卷积神经网络构件（仅使用 Python 标准库）。

本模块提供 :class:`Conv2D`，一个使用 NCHW 嵌套 ``list`` 表示数据的
二维卷积层。前向采用互相关（核不翻转），边界补零；反向返回
``(dx, dweights, dbias)``。全程不做静默类型转换，也不修改传入的列表。
"""

import argparse
import math
import sys

__all__ = ["Conv2D"]


def _is_real_number(value):
    """元素必须是严格的 int/float，明确拒绝 bool。"""
    return type(value) in (int, float)


def _validate_tensor(name, value, ndim):
    """校验嵌套 list 张量，返回其形状元组。

    - 顶层不是 list -> TypeError
    - 内部层级应为 list 却不是 / 空维 / 不规则 -> ValueError
    - 叶子不是 int/float（含 bool）-> TypeError
    - 叶子为 inf/nan -> ValueError
    """
    if not isinstance(value, list):
        raise TypeError("%s 必须是 list，实际类型为 %s" % (name, type(value).__name__))

    def walk(seq, level):
        if not isinstance(seq, list):
            raise ValueError(
                "%s 嵌套层级错误：第 %d 层应为 list，实际为 %s"
                % (name, level, type(seq).__name__)
            )
        if len(seq) == 0:
            raise ValueError("%s 的第 %d 维为空，要求各维非空" % (name, level))
        if level == ndim - 1:
            for v in seq:
                if isinstance(v, list):
                    raise ValueError(
                        "%s 嵌套层级过多：第 %d 层应为标量元素，实际仍为 list"
                        % (name, level)
                    )
                if not _is_real_number(v):
                    raise TypeError(
                        "%s 的元素必须是 int/float 且不能为 bool，实际为 %s"
                        % (name, type(v).__name__)
                    )
                if not math.isfinite(v):
                    raise ValueError("%s 的元素必须有限，得到 %r" % (name, v))
            return (len(seq),)
        sub_shape = None
        for item in seq:
            shape = walk(item, level + 1)
            if sub_shape is None:
                sub_shape = shape
            elif shape != sub_shape:
                raise ValueError("%s 的形状不规则（嵌套子列表长度不一致）" % name)
        return (len(seq),) + sub_shape

    return walk(value, 0)


class Conv2D(object):
    """二维卷积（互相关）层。

    参数形状：
      weights: ``[O][C][KH][KW]``
      bias:    ``[O]``
      x:       ``[N][C][H][W]``

    输出形状：
      ``[N][O][floor((H+2P-KH)/S)+1][floor((W+2P-KW)/S)+1]``
    """

    def __init__(self, weights, bias, stride=1, padding=0):
        w_shape = _validate_tensor("weights", weights, 4)
        b_shape = _validate_tensor("bias", bias, 1)

        # stride / padding 先查类型（bool 不算 int），再查取值。
        if type(stride) is not int:
            raise TypeError("stride 必须是正整数 int，实际为 %s" % type(stride).__name__)
        if type(padding) is not int:
            raise TypeError("padding 必须是非负整数 int，实际为 %s" % type(padding).__name__)
        if stride < 1:
            raise ValueError("stride 必须为正整数，得到 %d" % stride)
        if padding < 0:
            raise ValueError("padding 必须为非负整数，得到 %d" % padding)

        out_channels, in_channels, kh, kw = w_shape
        if b_shape[0] != out_channels:
            raise ValueError(
                "bias 长度 %d 与 weights 输出通道数 %d 不符"
                % (b_shape[0], out_channels)
            )

        self.weights = weights
        self.bias = bias
        self.stride = stride
        self.padding = padding

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kh = kh
        self.kw = kw

        # 最近一次成功 forward 的输入与输出形状；未 forward 前为 None。
        self._x = None
        self._x_shape = None
        self._out_shape = None

    def forward(self, x):
        """前向互相关。成功时缓存输入，返回新建的嵌套 list。"""
        x_shape = _validate_tensor("x", x, 4)
        n, c, h, w = x_shape
        if c != self.in_channels:
            raise ValueError(
                "x 的通道数 %d 与 weights 的输入通道数 %d 不符"
                % (c, self.in_channels)
            )

        p = self.padding
        s = self.stride
        padded_h = h + 2 * p
        padded_w = w + 2 * p
        if padded_h < self.kh:
            raise ValueError(
                "核高 %d 在补边后（有效高度 %d）仍然越界" % (self.kh, padded_h)
            )
        if padded_w < self.kw:
            raise ValueError(
                "核宽 %d 在补边后（有效宽度 %d）仍然越界" % (self.kw, padded_w)
            )
        out_h = (padded_h - self.kh) // s + 1
        out_w = (padded_w - self.kw) // s + 1

        weights = self.weights
        bias = self.bias
        kh = self.kh
        kw = self.kw

        output = []
        for ni in range(n):
            out_n = []
            for oi in range(self.out_channels):
                out_o = []
                w_o = weights[oi]
                for oh in range(out_h):
                    row = []
                    for ow in range(out_w):
                        total = bias[oi]
                        # 按 c -> kh -> kw 的顺序累加，保证确定性。
                        for ci in range(c):
                            w_oc = w_o[ci]
                            x_nc = x[ni][ci]
                            for khi in range(kh):
                                ih = oh * s - p + khi
                                if 0 <= ih < h:
                                    w_row = w_oc[khi]
                                    x_row = x_nc[ih]
                                    for kwi in range(kw):
                                        iw = ow * s - p + kwi
                                        if 0 <= iw < w:
                                            total += w_row[kwi] * x_row[iw]
                        row.append(total)
                    out_o.append(row)
                out_n.append(out_o)
            output.append(out_n)

        # 只有全部计算成功后才更新缓存。
        self._x = x
        self._x_shape = x_shape
        self._out_shape = (n, self.out_channels, out_h, out_w)
        return output

    def backward(self, dy):
        """对最近一次 forward 的输入求梯度。

        返回 ``(dx, dweights, dbias)``，形状分别与 x、weights、bias 相同。
        未成功执行过 forward 时一律抛出 ValueError。
        """
        if self._x is None:
            raise ValueError("在成功执行 forward 之前不能调用 backward")

        dy_shape = _validate_tensor("dy", dy, 4)
        if dy_shape != self._out_shape:
            raise ValueError(
                "dy 形状 %s 与最近输出形状 %s 不符"
                % (dy_shape, self._out_shape)
            )

        x = self._x
        n, c, h, w = self._x_shape
        o_count, out_h, out_w = self.out_channels, self._out_shape[2], self._out_shape[3]
        kh, kw = self.kh, self.kw
        s, p = self.stride, self.padding
        weights = self.weights

        # 用整数 0 初始化：全整数运算时梯度保持精确整数。
        dx = [[[[0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
              for _ in range(n)]
        dweights = [[[[0 for _ in range(kw)] for _ in range(kh)]
                     for _ in range(c)] for _ in range(o_count)]
        dbias = [0 for _ in range(o_count)]

        for ni in range(n):
            for oi in range(o_count):
                w_o = weights[oi]
                dw_o = dweights[oi]
                for oh in range(out_h):
                    for ow in range(out_w):
                        g = dy[ni][oi][oh][ow]
                        dbias[oi] += g
                        for ci in range(c):
                            x_nc = x[ni][ci]
                            w_oc = w_o[ci]
                            dw_oc = dw_o[ci]
                            for khi in range(kh):
                                ih = oh * s - p + khi
                                if not (0 <= ih < h):
                                    continue
                                for kwi in range(kw):
                                    iw = ow * s - p + kwi
                                    if not (0 <= iw < w):
                                        continue
                                    dx[ni][ci][ih][iw] += g * w_oc[khi][kwi]
                                    dw_oc[khi][kwi] += g * x_nc[ih][iw]

        return dx, dweights, dbias


def _selftest():
    # 手工算例：3x3 输入、2x2 核、stride 1、无补边。
    weights = [[[[1, 2], [3, 4]]]]
    bias = [5]
    x = [[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]
    layer = Conv2D(weights, bias)
    y = layer.forward(x)
    assert y == [[[[42, 52], [72, 82]]]], y
    dx, dw, db = layer.backward([[[[1, 1], [1, 1]]]])
    assert db == [4], db
    assert dw == [[[[12, 16], [24, 28]]]], dw
    assert dx == [[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]], dx
    print("selftest OK")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="纯标准库 Conv2D 实现")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest", help="运行内置数值自检")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        return _selftest()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
