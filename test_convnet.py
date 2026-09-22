"""fitnorm/evalnorm/resumenorm 子命令的可发现测试（仅标准库）。

覆盖：
- fitnorm 训练→evalnorm 评估的完整成功路径与产物键序/类型/形状契约；
- 末次 loss 小于首次 loss、推理预测为 [0,1]、accuracy 为 1.0；
- 重复运行逐字节相同；
- evalnorm 拒绝重复/缺失/额外/错序键、类型/形状错误与非有限值，
  失败退出 1 且不改动 OUTPUT；
- resumenorm 分段续训与总轮数连续训练的 loss 逐项、检查点逐字节相同，
  0 轮保持状态，JSON/轮数/路径等失败退出 1、空 stdout、不改 OUTPUT，
  参数数目错退出 2；load_norm_checkpoint 拒绝 JSON 常量与非规范 hex；
- 参数数目错误退出 2；
- 其余既有入口 train/evaluate/fitcnn/evalcnn 仍可成功运行。
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "convnet.py")


def _run(*args):
    """在仓库目录下运行 convnet.py，返回 (returncode, stdout, stderr)。"""
    proc = subprocess.run(
        [sys.executable, _SCRIPT] + list(args),
        cwd=_HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _ordered_load(path):
    """按 JSON 出现顺序加载键（dict 本身保序），便于断言键序。"""
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


class FitNormTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_fitnorm_then_evalnorm_contract(self):
        weights = self._path("norm.json")
        rc, _, err = _run("fitnorm", weights)
        self.assertEqual(rc, 0, err.decode())

        doc = _ordered_load(weights)
        # 顶层与 model 键序。
        self.assertEqual(list(doc.keys()), ["model", "metrics"])
        model = doc["model"]
        self.assertEqual(list(model.keys()), ["conv", "batchnorm", "linear"])
        self.assertEqual(list(model["conv"].keys()), ["values", "bias"])
        self.assertEqual(list(model["linear"].keys()), ["values", "bias"])

        # conv 形状 [2][1][1][1]、bias [2]。
        self.assertEqual(
            model["conv"]["values"],
            [[[[model["conv"]["values"][0][0][0][0]]]],
             [[[model["conv"]["values"][1][0][0][0]]]]],
        )
        self.assertEqual(len(model["conv"]["values"]), 2)
        self.assertEqual(len(model["conv"]["bias"]), 2)

        # batchnorm 键序与四个 float[2]（拒绝 int/bool）。
        bn = model["batchnorm"]
        self.assertEqual(
            list(bn.keys()),
            ["gamma", "beta", "running_mean", "running_var"],
        )
        for key in ("gamma", "beta", "running_mean", "running_var"):
            self.assertEqual(len(bn[key]), 2)
            for v in bn[key]:
                self.assertIsInstance(v, float)
                self.assertNotIsInstance(v, bool)
                self.assertTrue(v == v and abs(v) != float("inf"))

        # linear 形状 [2][2]、bias [2]。
        self.assertEqual(len(model["linear"]["values"]), 2)
        self.assertTrue(all(len(r) == 2 for r in model["linear"]["values"]))
        self.assertEqual(len(model["linear"]["bias"]), 2)

        # metrics 键序与取值类型/数值契约。
        metrics = doc["metrics"]
        self.assertEqual(
            list(metrics.keys()),
            ["epochs", "lr", "seed", "loss", "accuracy"],
        )
        self.assertIs(metrics["epochs"], 20)
        self.assertIsInstance(metrics["epochs"], int)
        self.assertEqual(metrics["lr"], 0.1)
        self.assertIsInstance(metrics["lr"], float)
        self.assertIs(metrics["seed"], 7)
        self.assertIsInstance(metrics["seed"], int)
        loss = metrics["loss"]
        self.assertEqual(len(loss), 20)
        for v in loss:
            self.assertIsInstance(v, float)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIsInstance(metrics["accuracy"], float)

        # 末次 loss 必须严格小于首次 loss。
        self.assertLess(loss[-1], loss[0])

        # evalnorm 以保存的统计推理：2 个样本、预测 [0,1]、accuracy 1.0。
        out = self._path("eval.json")
        rc, _, err = _run("evalnorm", weights, out)
        self.assertEqual(rc, 0, err.decode())
        result = _ordered_load(out)
        self.assertEqual(
            list(result.keys()), ["sample_count", "predictions", "accuracy"]
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["predictions"], [0, 1])
        self.assertEqual(result["accuracy"], 1.0)

    def test_repeated_run_byte_identical(self):
        first = self._path("first.json")
        second = self._path("second.json")
        rc1, _, err1 = _run("fitnorm", first)
        rc2, _, err2 = _run("fitnorm", second)
        self.assertEqual((rc1, rc2), (0, 0), err1 + err2)
        with open(first, "rb") as f:
            a = f.read()
        with open(second, "rb") as f:
            b = f.read()
        self.assertEqual(a, b)

    def test_evalnorm_rejects_malformed_artifacts(self):
        weights = self._path("norm.json")
        rc, _, err = _run("fitnorm", weights)
        self.assertEqual(rc, 0, err.decode())
        with open(weights, "rb") as f:
            good_text = f.read().decode("utf-8").strip()
        good = json.loads(good_text)

        compact = lambda d: json.dumps(d, separators=(",", ":"))
        mdl, met = good["model"], good["metrics"]
        bn = mdl["batchnorm"]

        # 直接给出的原始文本（重复键、NaN、inf、非对象、非法 JSON）。
        dup_top = (
            '{"model":' + compact(mdl) + ',"metrics":' + compact(met)
            + ',"metrics":{}}'
        )
        dup_bn = (
            '{"model":{"conv":' + compact(mdl["conv"])
            + ',"batchnorm":{"gamma":[1.0,1.0],"beta":[0.0,0.0],'
            '"gamma":[1.0,1.0],"running_mean":[0.0,0.0],'
            '"running_var":[1.0,1.0]},"linear":' + compact(mdl["linear"])
            + '},"metrics":' + compact(met) + "}"
        )
        bn_inf = (
            '{"model":{"conv":' + compact(mdl["conv"])
            + ',"batchnorm":{"gamma":[1e999,1.0],"beta":[0.0,0.0],'
            '"running_mean":[0.0,0.0],"running_var":[1.0,1.0]},'
            '"linear":' + compact(mdl["linear"]) + '},"metrics":'
            + compact(met) + "}"
        )
        bn_nan = (
            '{"model":{"conv":' + compact(mdl["conv"])
            + ',"batchnorm":{"gamma":[1.0,1.0],"beta":[0.0,0.0],'
            '"running_mean":[NaN,0.0],"running_var":[1.0,1.0]},'
            '"linear":' + compact(mdl["linear"]) + '},"metrics":'
            + compact(met) + "}"
        )

        # 通过改动合法产物构造的畸形变体。
        bn_gamma_int = dict(bn, gamma=[1, 1])
        bn_gamma_bool = dict(bn, gamma=[True, 1.0])
        bn_gamma_len3 = dict(bn, gamma=[1.0, 1.0, 1.0])
        variants = {
            "dup_top": dup_top,
            "dup_bn": dup_bn,
            "missing_model": compact({"metrics": met}),
            "extra_top": compact(dict(good, zz=1)),
            "reorder_top": compact({"metrics": met, "model": mdl}),
            "reorder_model": compact(
                {"model": {"linear": mdl["linear"],
                           "batchnorm": mdl["batchnorm"],
                           "conv": mdl["conv"]}, "metrics": met}
            ),
            "missing_batchnorm": compact(
                {"model": {"conv": mdl["conv"], "linear": mdl["linear"]},
                 "metrics": met}
            ),
            "extra_batchnorm": compact(
                {"model": {"conv": mdl["conv"],
                           "batchnorm": dict(bn, zz=1),
                           "linear": mdl["linear"]}, "metrics": met}
            ),
            "reorder_batchnorm": compact(
                {"model": {"conv": mdl["conv"],
                           "batchnorm": {"beta": bn["beta"],
                                         "gamma": bn["gamma"],
                                         "running_mean": bn["running_mean"],
                                         "running_var": bn["running_var"]},
                           "linear": mdl["linear"]}, "metrics": met}
            ),
            "reorder_metrics": compact(
                {"model": mdl,
                 "metrics": {"epochs": 20, "lr": 0.1, "loss": met["loss"],
                             "seed": 7, "accuracy": 1.0}}
            ),
            "missing_seed": compact(
                {"model": mdl,
                 "metrics": {"epochs": 20, "lr": 0.1, "loss": met["loss"],
                             "accuracy": 1.0}}
            ),
            "gamma_int": compact(
                {"model": {"conv": mdl["conv"], "batchnorm": bn_gamma_int,
                           "linear": mdl["linear"]}, "metrics": met}
            ),
            "gamma_bool": compact(
                {"model": {"conv": mdl["conv"], "batchnorm": bn_gamma_bool,
                           "linear": mdl["linear"]}, "metrics": met}
            ),
            "gamma_len3": compact(
                {"model": {"conv": mdl["conv"], "batchnorm": bn_gamma_len3,
                           "linear": mdl["linear"]}, "metrics": met}
            ),
            "seed_string": compact(
                {"model": mdl, "metrics": dict(met, seed="7")}
            ),
            "epochs_string": compact(
                {"model": mdl, "metrics": dict(met, epochs="20")}
            ),
            "lr_int": compact(
                {"model": mdl, "metrics": dict(met, lr=1)}
            ),
            "loss_short": compact(
                {"model": mdl, "metrics": dict(met, loss=[0.1] * 19)}
            ),
            "conv_wrong_shape": compact(
                {"model": {"conv": {"values": [[[[1.0]]]], "bias": [0.0]},
                           "batchnorm": bn, "linear": mdl["linear"]},
                 "metrics": met}
            ),
            "gamma_inf": bn_inf,
            "running_mean_nan": bn_nan,
            "top_level_list": "[]",
            "bad_json": "{not json",
        }

        sentinel = b"UNCHANGED-SENTINEL"
        for name, text in variants.items():
            bad_path = self._path("bad_%s.json" % name)
            out_path = self._path("bad_%s.out" % name)
            with open(bad_path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(out_path, "wb") as f:
                f.write(sentinel)
            rc, _, _ = _run("evalnorm", bad_path, out_path)
            self.assertEqual(rc, 1, "应当拒绝：%s" % name)
            with open(out_path, "rb") as f:
                self.assertEqual(
                    f.read(), sentinel, "失败时不得改写 OUTPUT：%s" % name
                )

        # 权重文件不存在同样退出 1 且不写 OUTPUT。
        out_path = self._path("missing.out")
        with open(out_path, "wb") as f:
            f.write(sentinel)
        rc, _, _ = _run("evalnorm", self._path("nope.json"), out_path)
        self.assertEqual(rc, 1)
        with open(out_path, "rb") as f:
            self.assertEqual(f.read(), sentinel)

    def test_metrics_values_do_not_affect_inference(self):
        # 推理只依赖保存的权重与 BN 统计；metrics 取其他合法类型值
        # （epochs=19、seed=8、accuracy=0.5）仍应成功且预测不变。
        weights = self._path("norm.json")
        rc, _, err = _run("fitnorm", weights)
        self.assertEqual(rc, 0, err.decode())
        good = _ordered_load(weights)
        tweaked = {
            "model": good["model"],
            "metrics": dict(
                good["metrics"], epochs=19, seed=8, accuracy=0.5
            ),
        }
        tweaked_path = self._path("tweaked.json")
        with open(tweaked_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(tweaked, separators=(",", ":")))
        out_path = self._path("tweaked.out")
        rc, _, err = _run("evalnorm", tweaked_path, out_path)
        self.assertEqual(rc, 0, err.decode())
        result = _ordered_load(out_path)
        self.assertEqual(result["predictions"], [0, 1])
        self.assertEqual(result["accuracy"], 1.0)

    def test_fitnorm_failure_leaves_output_untouched(self):
        # 换到无 data 目录的临时目录运行，数据校验必失败。
        out_path = self._path("should_not_exist.json")
        sentinel = b"UNCHANGED-SENTINEL"
        with open(out_path, "wb") as f:
            f.write(sentinel)
        proc = subprocess.run(
            [sys.executable, _SCRIPT, "fitnorm", out_path],
            cwd=self.tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(proc.returncode, 1)
        with open(out_path, "rb") as f:
            self.assertEqual(f.read(), sentinel)

    def test_wrong_argc_exits_2(self):
        self.assertEqual(_run("fitnorm")[0], 2)
        self.assertEqual(_run("fitnorm", "a", "b")[0], 2)
        self.assertEqual(_run("evalnorm", "w")[0], 2)
        self.assertEqual(_run("evalnorm")[0], 2)


class ResumeNormTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        sys.path.insert(0, _HERE)
        import convnet
        self.convnet = convnet

    def tearDown(self):
        sys.path.pop(0)
        self._tmp.cleanup()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _resume(self, inp, epochs, out_name="out.json"):
        return _run("resumenorm", inp, str(epochs), self._path(out_name))

    def test_fresh_run_contract(self):
        rc, out, err = self._resume("-", 5, "c5.json")
        self.assertEqual(rc, 0, err.decode())
        self.assertTrue(out.endswith(b"\n"))
        doc = json.loads(out.decode("utf-8"))
        self.assertEqual(
            list(doc.keys()), ["start_epoch", "added_epochs", "loss"]
        )
        self.assertIs(doc["start_epoch"], 0)
        self.assertIs(doc["added_epochs"], 5)
        self.assertEqual(len(doc["loss"]), 5)
        for v in doc["loss"]:
            self.assertIsInstance(v, float)
            self.assertNotIsInstance(v, bool)
        # 固定 12 位小数：loss 数组中的每个数恰含 12 位小数、紧凑无空格。
        text = out.decode("utf-8")
        self.assertNotIn(" ", text)
        m = re.search(r'"loss":\[(.*?)\]}\n$', text)
        self.assertIsNotNone(m)
        tokens = m.group(1).split(",") if m.group(1) else []
        self.assertEqual(len(tokens), 5)
        for token in tokens:
            self.assertRegex(token, r"^-?[0-9]+\.[0-9]{12}$")
        # 负零归零：文本中不得出现 -0.000000000000。
        self.assertNotIn("-0.000000000000", text)
        # OUTPUT 为合法检查点，epoch=5。
        with open(self._path("c5.json"), "rb") as f:
            layers, epoch = self.convnet.load_norm_checkpoint(f.read())
        self.assertEqual(epoch, 5)
        self.assertEqual(len(layers), 7)

    def test_segmented_matches_continuous(self):
        rc1, out1, err1 = self._resume("-", 3, "c3.json")
        rc2, out2, err2 = _run(
            "resumenorm", self._path("c3.json"), "2", self._path("c5.json")
        )
        rc3, out3, err3 = self._resume("-", 5, "c5cont.json")
        self.assertEqual((rc1, rc2, rc3), (0, 0, 0),
                         err1 + err2 + err3)
        # 检查点逐字节相同。
        with open(self._path("c5.json"), "rb") as f:
            seg = f.read()
        with open(self._path("c5cont.json"), "rb") as f:
            cont = f.read()
        self.assertEqual(seg, cont)
        # 分段 loss 拼接后逐项相同。
        l1 = json.loads(out1.decode())["loss"]
        l2 = json.loads(out2.decode())["loss"]
        l3 = json.loads(out3.decode())["loss"]
        self.assertEqual(json.loads(out2.decode())["start_epoch"], 3)
        self.assertEqual(l1 + l2, l3)

    def test_long_chain_matches_continuous(self):
        self.assertEqual(self._resume("-", 3, "a.json")[0], 0)
        rc, _, err = _run(
            "resumenorm", self._path("a.json"), "2", self._path("b.json")
        )
        self.assertEqual(rc, 0, err.decode())
        rc, _, err = _run(
            "resumenorm", self._path("b.json"), "15", self._path("c20.json")
        )
        self.assertEqual(rc, 0, err.decode())
        rc, _, err = self._resume("-", 20, "r20.json")
        self.assertEqual(rc, 0, err.decode())
        with open(self._path("c20.json"), "rb") as f:
            a = f.read()
        with open(self._path("r20.json"), "rb") as f:
            b = f.read()
        self.assertEqual(a, b)

    def test_zero_epochs_preserves_state(self):
        self.assertEqual(self._resume("-", 3, "c3.json")[0], 0)
        with open(self._path("c3.json"), "rb") as f:
            before = f.read()
        rc, out, err = _run(
            "resumenorm", self._path("c3.json"), "0", self._path("c3z.json")
        )
        self.assertEqual(rc, 0, err.decode())
        doc = json.loads(out.decode())
        self.assertEqual(doc["start_epoch"], 3)
        self.assertIs(doc["added_epochs"], 0)
        self.assertEqual(doc["loss"], [])
        with open(self._path("c3z.json"), "rb") as f:
            self.assertEqual(f.read(), before)

        # 从初值 0 轮：检查点等于初值网络的直接 dump。
        rc, _, err = self._resume("-", 0, "init.json")
        self.assertEqual(rc, 0, err.decode())
        ref = self.convnet.dump_norm_checkpoint(
            self.convnet._build_norm_layers(), 0
        )
        with open(self._path("init.json"), "rb") as f:
            self.assertEqual(f.read(), ref)

    def test_bad_epochs_fail_without_touching_output(self):
        sentinel = b"UNCHANGED-SENTINEL"
        bad_epochs = ("01", "00", "-1", "+1", "1.0", "abc", "",
                      " 1", "1 ", "0x1", "1_00", "1e0")
        for idx, epochs in enumerate(bad_epochs):
            out_path = self._path("e%d.out" % idx)
            with open(out_path, "wb") as f:
                f.write(sentinel)
            rc, stdout, _ = _run("resumenorm", "-", epochs, out_path)
            self.assertEqual(rc, 1, "应拒绝 EPOCHS=%r" % epochs)
            self.assertEqual(stdout, b"")
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), sentinel)

    def test_bad_or_missing_input(self):
        sentinel = b"UNCHANGED-SENTINEL"
        good_rc, _, _ = self._resume("-", 1, "c1.json")
        self.assertEqual(good_rc, 0)
        with open(self._path("c1.json"), "rb") as f:
            good = f.read()

        variants = {
            "missing": None,
            "bad_json": b"{not json",
            "nan": good.replace(b'"epoch":1', b'"epoch":NaN'),
            "infinity": good.replace(b'"epoch":1', b'"epoch":Infinity'),
            "minus_infinity":
                good.replace(b'"epoch":1', b'"epoch":-Infinity'),
        }
        # 非规范 hex 叶值。
        doc = json.loads(good.decode())
        doc["model"]["conv"]["values"][0][0][0][0] = "0X1.0P+0"
        variants["uppercase_hex"] = json.dumps(
            doc, separators=(",", ":")
        ).encode()
        doc = json.loads(good.decode())
        doc["model"]["conv"]["values"][0][0][0][0] = "-0x0.0p+0"
        variants["neg_zero_hex"] = json.dumps(
            doc, separators=(",", ":")
        ).encode()

        for name, payload in variants.items():
            in_path = self._path("in_%s.json" % name)
            if payload is not None:
                with open(in_path, "wb") as f:
                    f.write(payload)
            out_path = self._path("out_%s.bin" % name)
            with open(out_path, "wb") as f:
                f.write(sentinel)
            rc, stdout, _ = _run("resumenorm", in_path, "1", out_path)
            self.assertEqual(rc, 1, "应拒绝输入：%s" % name)
            self.assertEqual(stdout, b"")
            with open(out_path, "rb") as f:
                self.assertEqual(
                    f.read(), sentinel, "不得改写 OUTPUT：%s" % name
                )

    def test_path_conflict_and_argc(self):
        self.assertEqual(self._resume("-", 1, "same.json")[0], 0)
        rc, stdout, _ = _run(
            "resumenorm", self._path("same.json"), "1", self._path("same.json")
        )
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, b"")

        self.assertEqual(_run("resumenorm")[0], 2)
        self.assertEqual(_run("resumenorm", "-")[0], 2)
        self.assertEqual(_run("resumenorm", "-", "3")[0], 2)
        self.assertEqual(
            _run("resumenorm", "-", "3", "o", "extra")[0], 2
        )


class NormCheckpointStrictTests(unittest.TestCase):
    """load_norm_checkpoint：JSON 常量与规范 hex 的直接 API 测试。"""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, _HERE)
        import convnet
        cls.convnet = convnet
        x, labels = convnet._load_cnn_samples()
        layers = convnet._build_norm_layers()
        convnet.train_norm_step(layers, x, labels, 0.1)
        cls.good = convnet.dump_norm_checkpoint(layers, 1)

    @classmethod
    def tearDownClass(cls):
        sys.path.pop(0)

    def _load(self, data):
        return self.convnet.load_norm_checkpoint(data)

    def test_roundtrip(self):
        layers, epoch = self._load(self.good)
        self.assertEqual(epoch, 1)
        self.assertEqual(len(layers), 7)
        # 重复 dump 逐字节一致。
        self.assertEqual(
            self.convnet.dump_norm_checkpoint(layers, epoch), self.good
        )

    def test_rejects_non_bytes(self):
        with self.assertRaises(TypeError):
            self._load(self.good.decode("utf-8"))

    def test_rejects_json_constants(self):
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            bad = self.good.replace(b'"epoch":1', b'"epoch":' + token)
            with self.assertRaises(ValueError, msg=token):
                self._load(bad)
            # 常量出现在 dropout_state 标量位置同样先于解析被拒绝。
            bad2 = re.sub(
                rb'"dropout_state":\d+',
                b'"dropout_state":' + token,
                self.good,
            )
            with self.assertRaises(ValueError, msg=token):
                self._load(bad2)

    def test_constant_inside_string_is_not_scalar(self):
        # 字符串字面量内的 NaN/Infinity 不得被误判为 JSON 常量；
        # 但作为 hex 叶值仍因非规范而 ValueError。
        doc = json.loads(self.good.decode())
        doc["model"]["conv"]["values"][0][0][0][0] = "NaN"
        payload = json.dumps(doc, separators=(",", ":")).encode()
        with self.assertRaises(ValueError):
            self._load(payload)

    def test_rejects_noncanonical_hex(self):
        for spelling in (
            "0X1.0P+0", "0x1.0000p+0", "0x2.0p-1", "0x.8p+0",
            "1.0", "-0x0.0p+0", "0x0.0p-0", "0x1p+0", "0x1.0P+0",
            "+0x1.0p+0", "0x1.0p0", "0xa.0p+0", " 0x1.0p+0",
            "0x1.0p+0 ", "0x1.0p+00",
        ):
            doc = json.loads(self.good.decode())
            doc["model"]["conv"]["values"][0][0][0][0] = spelling
            payload = json.dumps(doc, separators=(",", ":")).encode()
            with self.assertRaises(ValueError, msg=spelling):
                self._load(payload)

    def test_canonical_zero_accepted(self):
        doc = json.loads(self.good.decode())
        doc["model"]["conv"]["values"][0][0][0][0] = "0x0.0p+0"
        payload = json.dumps(doc, separators=(",", ":")).encode()
        layers, _ = self._load(payload)
        self.assertEqual(layers[0]._weights[0][0][0][0], 0.0)


class ExistingEntryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_other_entries_still_work(self):
        tr = self._path("tr.json")
        tr_out = self._path("tr.eval.json")
        self.assertEqual(_run("train", tr)[0], 0)
        self.assertEqual(_run("evaluate", tr, tr_out)[0], 0)
        self.assertEqual(_ordered_load(tr_out)["accuracy"], 1.0)

        cnn = self._path("cnn.json")
        cnn_out = self._path("cnn.eval.json")
        self.assertEqual(_run("fitcnn", cnn)[0], 0)
        self.assertEqual(_run("evalcnn", cnn, cnn_out)[0], 0)
        result = _ordered_load(cnn_out)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["predictions"], [0, 1])
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
