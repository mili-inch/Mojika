import json
from pathlib import Path
import tempfile
import unittest

from mojika.assets import verify_weights
from mojika.prepare_data import read_descriptions
from mojika.server.config import ServerConfig


class ReleaseSetupTests(unittest.TestCase):
    def test_relative_paths_resolve_from_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "assets/weights").mkdir(parents=True)
            (root / "assets/weights/weights.pt").touch()
            for name in ("sarashina2.2-0.5b", "clip-japanese-base"):
                (root / "assets/models" / name).mkdir(parents=True)
            (root / "web/dist").mkdir(parents=True)
            (root / "web/dist/index.html").write_text("<html></html>")
            template = Path(__file__).resolve().parents[1] / "configs/studio.toml"
            config_path = root / "configs/studio.toml"
            config_path.write_text(template.read_text())
            config = ServerConfig.read(config_path)
            self.assertEqual(config.checkpoint, root / "assets/weights/weights.pt")
            self.assertEqual(config.client_dist, root / "web/dist")
            config.checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                ServerConfig.read(config_path)

    def test_weight_checksum_rejects_wrong_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"not the published weights")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_weights(path)

    def test_descriptions_reject_missing_and_malformed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.json"
            for data in (
                {},
                [],
                {"山川": ["地形"]},
                {"山": []},
                {"山": [""]},
                {"山": [1]},
            ):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    read_descriptions(path)
            data = {"山": ["高い地形。"], "魚": ["水中で暮らす動物。"]}
            path.write_text(json.dumps(data))
            self.assertEqual(read_descriptions(path), data)
