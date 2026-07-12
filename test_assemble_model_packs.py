import json
import pathlib
import tempfile
import unittest

import numpy as np

import assemble_model_packs as assembler
import fetch_weather_data as weather
import validate_consensus as packed


class AssembleModelPacksTests(unittest.TestCase):
    def test_assembles_two_independent_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            lon = np.array([0.0, 1.0], dtype=np.float32)
            lat = np.array([1.0, 0.0], dtype=np.float32)
            for provider, temperature in (("gfs", 280.0), ("ifs", 284.0)):
                directory = root / provider
                directory.mkdir()
                tile = weather.build_binary_tile_bytes(
                    lon,
                    lat,
                    1.0,
                    -1.0,
                    [(0, np.full(4, temperature, dtype=np.float32))],
                )
                weather.write_tile_pack(directory / "N0_E0.gpack", [("0", tile)])
                (root / f"{provider}.json").write_text(json.dumps({
                    "forecast_hours": [0],
                    "day_count": 1,
                    "pack_layout": "complete_horizon",
                    "fields": ["temperature"],
                }))

            assembler.assemble(root, "2026-07-12T12:00:00")

            manifest = json.loads((root / "manifest.json").read_text())
            self.assertIn("consensus", manifest["models"])
            consensus = packed.read_pack(root / "consensus" / "N0_E0.gpack")
            values = consensus["0"][4][0]
            self.assertTrue(np.allclose(values, 280.0, atol=0.01))


if __name__ == "__main__":
    unittest.main()
