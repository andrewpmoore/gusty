import json
import pathlib
import tempfile
import unittest

import numpy as np

import assemble_model_packs as assembler
import fetch_weather_data as weather
import validate_consensus as packed


class AssembleModelPacksTests(unittest.TestCase):
    def test_bias_corrected_blend_stays_inside_model_envelope(self):
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

            learning_state = {
                "weights": {"gfs": 0.5, "ifs": 0.5},
                "regions": {
                    "-10_+000": {
                        "lead_buckets": {
                            "0-24": {"weights": {"gfs": 0.1, "ifs": 0.9}}
                        },
                        "temperature_bias": {
                            "day": {"value": 1.0, "samples": 3}
                        },
                    }
                },
            }
            assembler.assemble(
                root, "2026-07-12T12:00:00", learning_state
            )

            manifest = json.loads((root / "manifest.json").read_text())
            self.assertIn("consensus", manifest["models"])
            consensus = packed.read_pack(root / "consensus" / "N0_E0.gpack")
            values = consensus["0"][4][0]
            self.assertTrue(np.allclose(values, 284.0, atol=0.01))
            combined = packed.read_pack(root / "multi" / "N0_E0.gpack")
            temperature = packed.read_pack(
                root / "multi" / "temperature" / "N0_E0.gpack"
            )
            self.assertEqual(
                set(combined["0"][4]),
                set(temperature["0"][4]),
            )

    def test_adds_icon_temperature_to_map_tiles_and_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "models"
            icon = root / "icon"
            temperature = base / "temp"
            icon.mkdir(parents=True)
            temperature.mkdir()
            lon = np.array([0.0, 1.0], dtype=np.float32)
            lat = np.array([1.0, 0.0], dtype=np.float32)
            icon_tile = weather.build_binary_tile_bytes(
                lon, lat, 1.0, -1.0,
                [(0, np.full(4, 284.0, dtype=np.float32))],
            )
            weather.write_tile_pack(icon / "N0_E0.gpack", [("0", icon_tile)])
            base_tile = weather.build_binary_tile_bytes(
                lon, lat, 1.0, -1.0,
                [(0, np.full(4, 280.0, dtype=np.float32))],
            )
            weather.write_tile_pack(
                temperature / "temp_0h_N0_E0.gpack",
                [("N0_E0", base_tile), ("N0_E0_60", base_tile)],
            )

            assembler.add_compact_model_temperature_channels(root)

            result = packed.read_pack(temperature / "temp_0h_N0_E0.gpack")
            self.assertTrue(np.allclose(result["N0_E0"][4][15], 284.0))
            self.assertTrue(np.allclose(result["N0_E0_60"][4][15], 284.0))


if __name__ == "__main__":
    unittest.main()
