import json
import pathlib
import tempfile
import unittest
from unittest import mock

import sync_scenes


class SyncScenesTests(unittest.TestCase):
    def test_sync_validates_assets_and_rewrites_image_url(self):
        manifest = [{"id": "leeds", "name": "Leeds", "thumbnailUrl": "old"}]
        responses = {
            "https://source/manifest.json": json.dumps(manifest).encode(),
            "https://source/leeds.json": b'{"imageWidth":1200}',
            "https://source/leeds.webp": b"RIFF\x04\x00\x00\x00WEBPdata",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "sync_scenes.download", side_effect=lambda url: responses[url]
        ):
            count = sync_scenes.sync(
                "https://source", "https://r2/scenes/runs/1", temporary
            )
            output = pathlib.Path(temporary)
            rewritten = json.loads((output / "manifest.json").read_text())
            self.assertEqual(count, 1)
            self.assertEqual(
                rewritten[0]["thumbnailUrl"],
                "https://r2/scenes/runs/1/leeds.webp",
            )
            self.assertTrue((output / "leeds.json").exists())
            self.assertTrue((output / "leeds.webp").exists())


if __name__ == "__main__":
    unittest.main()
