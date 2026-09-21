"""convnet.py 命令行子命令的可发现测试（仅标准库）。

覆盖 fitnorm/evalnorm 的产物契约、确定性、严格校验与失败退出行为，
并附带 fitcnn/evalcnn 往返冒烟测试确保既有入口不变。
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "convnet.py")


def run_cli(args, cwd=ROOT):
    """以仓库根为工作目录运行 convnet.py 子命令，返回 CompletedProcess。"""
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class FitnormTests(unittest.TestCase):
    """fitnorm 产物契约与确定性。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = os.path.join(self._tmp.name, "norm.json")

    def _run_fitnorm(self):
        result = run_cli(["fitnorm", self.out])
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out, "rb") as f:
            self.payload = f.read()
        self.doc = json.loads(self.payload.decode("utf-8"))

    def test_artifact_key_order_and_types(self):
        self._run_fitnorm()
        doc = self.doc
        self.assertEqual(list(doc.keys()), ["model", "metrics"])
        self.assertEqual(
            list(doc["model"].keys()), ["conv", "batchnorm", "linear"]
        )
        self.assertEqual(
            list(doc["model"]["conv"].keys()), ["values", "bias"]
        )
        self.assertEqual(
            list(doc["model"]["batchnorm"].keys()),
            ["gamma", "beta", "running_mean", "running_var"],
        )
        self.assertEqual(
            list(doc["model"]["linear"].keys()), ["values", "bias"]
        )
        self.assertEqual(
            list(doc["metrics"].keys()),
            ["epochs", "lr", "seed", "loss", "accuracy"],
        )

        bn = doc["model"]["batchnorm"]
        for key in ("gamma", "beta", "running_mean", "running_var"):
            self.assertEqual(len(bn[key]), 2)
            for v in bn[key]:
                self.assertIs(type(v), float)

        metrics = doc["metrics"]
        self.assertIs(type(metrics["epochs"]), int)
        self.assertEqual(metrics["epochs"], 20)
        self.assertIs(type(metrics["lr"]), float)
        self.assertEqual(metrics["lr"], 0.1)
        self.assertIs(type(metrics["seed"]), int)
        self.assertEqual(metrics["seed"], 7)
        self.assertEqual(len(metrics["loss"]), 20)
        for v in metrics["loss"]:
            self.assertIs(type(v), float)
        self.assertIs(type(metrics["accuracy"]), float)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_loss_decreases_and_accuracy_perfect(self):
        self._run_fitnorm()
        loss = self.doc["metrics"]["loss"]
        self.assertLess(loss[-1], loss[0])
        self.assertEqual(self.doc["metrics"]["accuracy"], 1.0)

    def test_deterministic_byte_identical(self):
        self._run_fitnorm()
        first = self.payload
        other = os.path.join(self._tmp.name, "norm2.json")
        result = run_cli(["fitnorm", other])
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(other, "rb") as f:
            self.assertEqual(f.read(), first)

    def test_missing_data_exits_1_without_output(self):
        # 在无 data/ 的目录下运行：数据缺失须退出 1 且不写产物。
        with tempfile.TemporaryDirectory() as empty:
            out = os.path.join(empty, "norm.json")
            result = run_cli(["fitnorm", out], cwd=empty)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(os.path.exists(out))

    def test_wrong_arg_count_exits_2(self):
        self.assertEqual(run_cli(["fitnorm"]).returncode, 2)
        self.assertEqual(run_cli(["fitnorm", "a", "b"]).returncode, 2)


class EvalnormTests(unittest.TestCase):
    """evalnorm 输出契约、严格校验与失败行为。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.weights = os.path.join(cls._tmp.name, "norm.json")
        result = run_cli(["fitnorm", cls.weights])
        if result.returncode != 0:
            raise RuntimeError("fitnorm 前置失败：%s" % result.stderr)
        with open(cls.weights, "rb") as f:
            cls.weights_bytes = f.read()
        cls.doc_template = json.loads(cls.weights_bytes.decode("utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.doc = copy.deepcopy(self.doc_template)
        self.out = os.path.join(self._tmp.name, "eval.json")
        if os.path.exists(self.out):
            os.unlink(self.out)

    def _write_weights(self, doc=None, raw=None):
        path = os.path.join(self._tmp.name, "tampered.json")
        data = raw if raw is not None else json.dumps(doc).encode("utf-8")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _assert_rejected(self, path):
        result = run_cli(["evalnorm", path, self.out])
        self.assertEqual(result.returncode, 1)
        self.assertFalse(os.path.exists(self.out))

    def test_output_contract(self):
        result = run_cli(["evalnorm", self.weights, self.out])
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
        self.assertEqual(
            list(doc.keys()), ["sample_count", "predictions", "accuracy"]
        )
        self.assertEqual(doc["sample_count"], 2)
        self.assertEqual(doc["predictions"], [0, 1])
        self.assertEqual(doc["accuracy"], 1.0)

    def test_deterministic_byte_identical(self):
        result = run_cli(["evalnorm", self.weights, self.out])
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.out, "rb") as f:
            first = f.read()
        other = os.path.join(self._tmp.name, "eval2.json")
        result = run_cli(["evalnorm", self.weights, other])
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(other, "rb") as f:
            self.assertEqual(f.read(), first)

    def test_rejects_duplicate_keys(self):
        # 在 batchnorm 内注入第二个 gamma 键。
        raw = self.weights_bytes.replace(
            b'"beta"', b'"gamma":[0.0,0.0],"beta"', 1
        )
        self.assertNotEqual(raw, self.weights_bytes)
        self._assert_rejected(self._write_weights(raw=raw))

    def test_rejects_wrong_model_key_order(self):
        doc = self.doc
        doc["model"] = {
            "batchnorm": doc["model"]["batchnorm"],
            "conv": doc["model"]["conv"],
            "linear": doc["model"]["linear"],
        }
        self._assert_rejected(self._write_weights(doc))

    def test_rejects_missing_key(self):
        doc = self.doc
        del doc["model"]["batchnorm"]["running_var"]
        self._assert_rejected(self._write_weights(doc))

    def test_rejects_extra_key(self):
        doc = self.doc
        doc["model"]["extra"] = {}
        self._assert_rejected(self._write_weights(doc))

    def test_rejects_wrong_shape(self):
        doc = self.doc
        doc["model"]["batchnorm"]["gamma"] = [1.0]
        self._assert_rejected(self._write_weights(doc))

    def test_rejects_non_finite(self):
        doc = self.doc
        doc["model"]["batchnorm"]["running_mean"][0] = float("nan")
        self._assert_rejected(self._write_weights(doc))

    def test_rejects_wrong_metrics_type(self):
        doc = self.doc
        doc["metrics"]["seed"] = 7.0
        self._assert_rejected(self._write_weights(doc))

    def test_missing_weights_exits_1(self):
        missing = os.path.join(self._tmp.name, "no-such.json")
        self._assert_rejected(missing)

    def test_failure_preserves_existing_output(self):
        with open(self.out, "wb") as f:
            f.write(b"sentinel")
        doc = self.doc
        doc["model"]["linear"]["bias"] = [0.0]
        result = run_cli(
            ["evalnorm", self._write_weights(doc), self.out]
        )
        self.assertEqual(result.returncode, 1)
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), b"sentinel")

    def test_wrong_arg_count_exits_2(self):
        self.assertEqual(run_cli(["evalnorm"]).returncode, 2)
        self.assertEqual(
            run_cli(["evalnorm", "a", "b", "c"]).returncode, 2
        )


class ExistingEntrySmokeTests(unittest.TestCase):
    """既有入口 fitcnn/evalcnn 保持可用。"""

    def test_fitcnn_evalcnn_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "cnn.json")
            out = os.path.join(tmp, "eval.json")
            self.assertEqual(run_cli(["fitcnn", weights]).returncode, 0)
            self.assertEqual(
                run_cli(["evalcnn", weights, out]).returncode, 0
            )
            with open(out, "rb") as f:
                doc = json.loads(f.read().decode("utf-8"))
            self.assertEqual(doc["sample_count"], 2)
            self.assertEqual(doc["predictions"], [0, 1])
            self.assertEqual(doc["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
