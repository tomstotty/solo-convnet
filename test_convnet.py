import copy
import math
import unittest

from convnet import Conv2D


def ref_forward(weights, bias, x, stride, padding):
    """朴素参考互相关实现。"""
    o, c, kh, kw = len(weights), len(weights[0]), len(weights[0][0]), len(weights[0][0][0])
    n = len(x)
    h, w = len(x[0][0]), len(x[0][0][0])
    p = padding
    s = stride
    oh = (h + 2 * p - kh) // s + 1
    ow = (w + 2 * p - kw) // s + 1

    def get(ni, ci, ih, iw):
        if ih < 0 or ih >= h or iw < 0 or iw >= w:
            return 0
        return x[ni][ci][ih][iw]

    out = [[[[0.0 for _ in range(ow)] for _ in range(oh)] for _ in range(o)]
           for _ in range(n)]
    for ni in range(n):
        for oi in range(o):
            for a in range(oh):
                for b in range(ow):
                    total = bias[oi]
                    for ci in range(c):
                        for u in range(kh):
                            for v in range(kw):
                                total += weights[oi][ci][u][v] * get(
                                    ni, ci, a * s - p + u, b * s - p + v)
                    out[ni][oi][a][b] = total
    return out


def ref_backward(weights, x, dy, stride, padding):
    o, c, kh, kw = len(weights), len(weights[0]), len(weights[0][0]), len(weights[0][0][0])
    n = len(x)
    h, w = len(x[0][0]), len(x[0][0][0])
    p, s = padding, stride
    oh, ow = len(dy[0][0]), len(dy[0][0][0])

    dx = [[[[0 for _ in range(w)] for _ in range(h)] for _ in range(c)] for _ in range(n)]
    dw = [[[[0 for _ in range(kw)] for _ in range(kh)] for _ in range(c)] for _ in range(o)]
    db = [0 for _ in range(o)]
    for ni in range(n):
        for oi in range(o):
            for a in range(oh):
                for b in range(ow):
                    g = dy[ni][oi][a][b]
                    db[oi] += g
                    for ci in range(c):
                        for u in range(kh):
                            for v in range(kw):
                                ih, iw = a * s - p + u, b * s - p + v
                                if 0 <= ih < h and 0 <= iw < w:
                                    dx[ni][ci][ih][iw] += g * weights[oi][ci][u][v]
                                    dw[oi][ci][u][v] += g * x[ni][ci][ih][iw]
    return dx, dw, db


def shape4(t):
    return (len(t), len(t[0]), len(t[0][0]), len(t[0][0][0]))


class TestForward(unittest.TestCase):
    def test_basic_values(self):
        weights = [[[[1, 2], [3, 4]]]]
        bias = [5]
        x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        y = Conv2D(weights, bias).forward(x)
        self.assertEqual(y, [[[[42, 52], [72, 82]]]])

    def test_multi_channel_batch(self):
        weights = [
            [[[1, 0]], [[0, 1]]],
            [[[-1, 1]], [[1, -1]]],
        ]
        bias = [0, 0]
        x = [
            [[[1, 2]], [[3, 4]]],
            [[[5, 6]], [[7, 8]]],
        ]
        y = Conv2D(weights, bias).forward(x)
        # o0: c0 取左 1 + c1 取右 4 => n0: 1+4=5 ; n1: 5+8=13
        # o1: -左c0 + 右c0 + 左c1 - 右c1 => n0: -1+2+3-4=0 ; n1: -5+6+7-8=0
        self.assertEqual(y, [[[[5]], [[0]]], [[[13]], [[0]]]])

    def test_stride_and_padding_against_reference(self):
        x = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]]
        weights = [[[[1, -1], [2, -2]]]]
        bias = [3]
        for stride in (1, 2, 3):
            for padding in (0, 1, 2):
                layer = Conv2D(copy.deepcopy(weights), bias,
                               stride=stride, padding=padding)
                y = layer.forward(copy.deepcopy(x))
                expected = ref_forward(weights, bias, x, stride, padding)
                self.assertEqual(shape4(y), shape4(expected))
                self.assertEqual(y, expected)

    def test_padding_output_shape(self):
        x = [[[[1, 2], [3, 4]]]]
        w = [[[[1]]]]
        y = Conv2D(w, [0], padding=2).forward(x)
        self.assertEqual(shape4(y), (1, 1, 6, 6))

    def test_does_not_mutate_inputs(self):
        weights = [[[[1.0, 2.0], [3.0, 4.0]]]]
        bias = [0.5]
        x = [[[[1.0, 2.0], [3.0, 4.0]]]]
        w0, b0, x0 = copy.deepcopy(weights), copy.deepcopy(bias), copy.deepcopy(x)
        Conv2D(weights, bias).forward(x)
        self.assertEqual(weights, w0)
        self.assertEqual(bias, b0)
        self.assertEqual(x, x0)

    def test_integer_stays_integer(self):
        y = Conv2D([[[[1]]]], [2]).forward([[[[3]]]])
        self.assertEqual(y, [[[[5]]]])
        self.assertIsInstance(y[0][0][0][0], int)


class TestBackward(unittest.TestCase):
    def test_basic_gradients(self):
        weights = [[[[1, 2], [3, 4]]]]
        bias = [5]
        x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        layer = Conv2D(weights, bias)
        layer.forward(x)
        dy = [[[[1, 1], [1, 1]]]]
        dx, dw, db = layer.backward(dy)
        self.assertEqual(db, [4])
        self.assertEqual(dw, [[[[12, 16], [24, 28]]]])
        self.assertEqual(dx, [[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]])

    def test_against_reference(self):
        cases = [
            (2, 3, 2, 2, 1, 4, 4, 1, 0),
            (1, 2, 2, 3, 2, 5, 5, 1, 1),
            (3, 2, 1, 3, 2, 4, 6, 2, 1),
            (2, 1, 3, 3, 1, 5, 5, 2, 2),
            (1, 1, 2, 2, 3, 3, 3, 3, 1),
        ]
        idx = 0
        for n, c, kh, kw, o, h, w, stride, padding in cases:
            x = [[[[((i + 7 * j + 3 * ci + 5 * ni + idx) % 11) - 5
                    for i in range(w)] for j in range(h)]
                  for ci in range(c)] for ni in range(n)]
            weights = [[[[((a + 2 * b + ci + 3 * oi + idx) % 7) - 3
                          for b in range(kw)] for a in range(kh)]
                       for ci in range(c)] for oi in range(o)]
            bias = [(oi * 2 + idx) % 5 - 2 for oi in range(o)]
            layer = Conv2D(weights, bias, stride=stride, padding=padding)
            y = layer.forward(x)
            oh, ow = shape4(y)[2], shape4(y)[3]
            dy = [[[[(ni + oi + a + b + idx) % 5 - 2 for b in range(ow)]
                    for a in range(oh)] for oi in range(o)] for ni in range(n)]
            dx, dw, db = layer.backward(dy)
            edx, edw, edb = ref_backward(weights, x, dy, stride, padding)
            self.assertEqual(dx, edx)
            self.assertEqual(dw, edw)
            self.assertEqual(db, edb)
            idx += 1

    def test_caches_latest_input(self):
        layer = Conv2D([[[[1]]]], [0])
        layer.forward([[[[10]]]])
        layer.forward([[[[20]]]])
        dx, dw, db = layer.backward([[[[1]]]])
        self.assertEqual(dw, [[[[20]]]])
        self.assertEqual(dx, [[[[1]]]])

    def test_backward_returns_new_lists(self):
        layer = Conv2D([[[[1, 1], [1, 1]]]], [0])
        x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        layer.forward(x)
        ones = [[[[1, 1], [1, 1]]]]
        dx, dw, db = layer.backward(ones)
        dx[0][0][0][0] = 999
        dw[0][0][0][0] = 999
        db[0] = 999
        dx2, dw2, db2 = layer.backward(ones)
        self.assertEqual(dx2[0][0][0][0], 1)
        self.assertEqual(dw2[0][0][0][0], 12)
        self.assertEqual(db2[0], 4)


class TestErrors(unittest.TestCase):
    def test_backward_before_forward(self):
        layer = Conv2D([[[[1]]]], [0])
        with self.assertRaises(ValueError):
            layer.backward([[[[1]]]])

    def test_bool_element_rejected(self):
        with self.assertRaises(TypeError):
            Conv2D([[[[True]]]], [0])
        with self.assertRaises(TypeError):
            Conv2D([[[[1]]]], [True])
        layer = Conv2D([[[[1]]]], [0])
        layer.forward([[[[1]]]])
        with self.assertRaises(TypeError):
            layer.backward([[[[True]]]])

    def test_non_numeric_element(self):
        with self.assertRaises(TypeError):
            Conv2D([[[[1, "2"]]]], [0])
        with self.assertRaises(TypeError):
            Conv2D("nope", [0])
        layer = Conv2D([[[[1]]]], [0])
        with self.assertRaises(TypeError):
            layer.forward([[[[None]]]])

    def test_non_finite_rejected(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                Conv2D([[[[bad]]]], [0])
            layer = Conv2D([[[[1]]]], [0])
            with self.assertRaises(ValueError):
                layer.forward([[[[bad]]]])

    def test_empty_dims(self):
        with self.assertRaises(ValueError):
            Conv2D([], [])
        with self.assertRaises(ValueError):
            Conv2D([[[[]]]], [0])
        layer = Conv2D([[[[1]]]], [0])
        with self.assertRaises(ValueError):
            layer.forward([])

    def test_ragged(self):
        with self.assertRaises(ValueError):
            Conv2D([[[[1, 2], [3]]]], [0])
        layer = Conv2D([[[[1]]]], [0])
        with self.assertRaises(ValueError):
            layer.forward([[[[1, 2], [3]]]])
        with self.assertRaises(ValueError):
            layer.forward([[[[1]]], [[[1, 2]]]])

    def test_wrong_nesting(self):
        with self.assertRaises(ValueError):
            Conv2D([[[1]]], [0])
        with self.assertRaises(ValueError):
            Conv2D([[[[[1]]]]], [0])

    def test_bias_length_mismatch(self):
        with self.assertRaises(ValueError):
            Conv2D([[[[1]]], [[[1]]]], [0])

    def test_channel_mismatch(self):
        layer = Conv2D([[[[1]]]], [0])
        with self.assertRaises(ValueError):
            layer.forward([[[[1]], [[1]]]])  # N=1, C=2

    def test_kernel_out_of_bounds_after_padding(self):
        layer = Conv2D([[[[1, 1, 1, 1]]]], [0], padding=1)
        with self.assertRaises(ValueError):
            layer.forward([[[[1]]]])  # W=1，补边后宽度 3 < 核宽 4
        layer2 = Conv2D([[[[1], [1], [1], [1]]]], [0])
        with self.assertRaises(ValueError):
            layer2.forward([[[[1], [2]]]])  # H=2 < 核高 4

    def test_illegal_stride_padding(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                Conv2D([[[[1]]]], [0], stride=bad)
        with self.assertRaises(ValueError):
            Conv2D([[[[1]]]], [0], padding=-1)
        for bad in (1.0, True, "1", 1.5):
            with self.assertRaises(TypeError):
                Conv2D([[[[1]]]], [0], stride=bad)
            with self.assertRaises(TypeError):
                Conv2D([[[[1]]]], [0], padding=bad)

    def test_dy_shape_mismatch(self):
        layer = Conv2D([[[[1, 1], [1, 1]]]], [0])
        layer.forward([[[[1, 2], [3, 4]]]])  # 输出 1x1
        # 合法 dy 为 [[[[1]]]]；以下形状均不符：
        with self.assertRaises(ValueError):
            layer.backward([[[[1, 1]]]])          # (1,1,1,2)
        with self.assertRaises(ValueError):
            layer.backward([[[[1]], [[1]]]])      # (1,2,1,1)
        with self.assertRaises(ValueError):
            layer.backward([[[[1]]], [[[1]]]])    # (2,1,1,1)
        with self.assertRaises(ValueError):
            layer.backward([[[[[1]]]]])           # 嵌套层级错误


if __name__ == "__main__":
    unittest.main()
