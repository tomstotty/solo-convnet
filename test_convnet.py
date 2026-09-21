"""fitnorm/evalnorm 子命令的可发现测试（仅标准库）。

覆盖：
- fitnorm 训练→evalnorm 评估的完整成功路径与产物键序/类型/形状契约；
- 末次 loss 小于首次 loss、推理预测为 [0,1]、accuracy 为 1.0；
- 重复运行逐字节相同；
- evalnorm 拒绝重复/缺失/额外/错序键、类型/形状错误与非有限值，
  失败退出 1 且不改动 OUTPUT；
- 参数数目错误退出 2；
- 其余既有入口 train/evaluate/fitcnn/evalcnn 仍可成功运行。
"""

import json
import os
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
