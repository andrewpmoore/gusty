import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fetch_radar_data as radar


class FetchRadarDataCoreTests(unittest.TestCase):
    def _model_frame(self, lead_minutes, value):
        latitudes = np.array([1.0, 0.0], dtype=np.float64)
        longitudes = np.array([0.0, 1.0], dtype=np.float64)
        values = np.full((2, 2), value, dtype=np.float32)
        return radar.create_frame(
            provider_id="noaa_gfs",
            name="Fixture GFS",
            valid_time="2026-06-29T00:00:00+00:00",
            lead_minutes=lead_minutes,
            latitudes=latitudes,
            longitudes=longitudes,
            rain=values,
            snow=values * 0.5,
            mixed=None,
            attribution="Fixture",
            source_url="https://example.com",
            source_kind="model-grib2",
            metadata={"model_fallback": True},
        )

    def test_split_precip_by_wet_bulb(self):
        precip = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        wet_bulb = np.array([[-1.0, 0.8, 3.0]], dtype=np.float32)

        rain, snow, mixed = radar.split_precip_by_wet_bulb(precip, wet_bulb)

        np.testing.assert_allclose(rain, [[0.0, 0.0, 3.0]])
        np.testing.assert_allclose(snow, [[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(mixed, [[0.0, 2.0, 0.0]])

    def test_polish_precip_keeps_transparent_zero_and_softens_core(self):
        values = np.zeros((7, 7), dtype=np.float32)
        values[3, 3] = 10.0

        polished = radar.polish_precip(values)

        self.assertGreater(polished[3, 3], 0.0)
        self.assertGreater(polished[3, 2], 0.0)
        self.assertEqual(float(polished[0, 0]), 0.0)

    def test_write_tile_pack_round_trips_header(self):
        lon = np.array([0.0, 0.1], dtype=np.float64)
        lat = np.array([1.0, 0.9], dtype=np.float64)
        values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        tile = radar.build_binary_tile_bytes(lon, lat, 0.1, -0.1, values)

        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = os.path.join(tmpdir, "test.gpack")
            radar.write_tile_pack(pack_path, [("N0_E0", tile)])
            with open(pack_path, "rb") as handle:
                payload = handle.read()

        self.assertEqual(payload[:4], radar.PACK_MAGIC)
        self.assertIn(b"N0_E0", payload)
        self.assertIn(radar.MAGIC, payload)

    def test_aggregate_product_results_includes_all_forecast_offsets(self):
        by_offset = {}
        for offset in radar.FORECAST_OFFSETS_MINUTES:
            by_offset[offset] = {
                product_id: {
                    "forecast_offset_minutes": offset,
                    "status": "ok",
                    "tiles": 1,
                    "packs": 1,
                    "max_value": 2.0,
                    "ref_time": "2026-06-29T00:00:00+00:00",
                    "source_provider_ids": ["fixture"],
                    "attribution": "Fixture",
                    "source_url": "https://example.com",
                    "model_fallback_active": False,
                }
                for product_id in radar.RADAR_SCALAR_LAYER_IDS
            }

        products = radar.aggregate_product_results(by_offset)

        self.assertEqual({product["id"] for product in products}, set(radar.RADAR_SCALAR_LAYER_IDS))
        for product in products:
            self.assertEqual(product["forecast_offsets_minutes"], radar.FORECAST_OFFSETS_MINUTES)
            self.assertEqual(len(product["frames"]), len(radar.FORECAST_OFFSETS_MINUTES))

    def test_global_model_fallback_is_enabled_by_default(self):
        far_from_observations = (20.0, -20.0, 40.0, 0.0)

        self.assertTrue(radar.ENABLE_GLOBAL_MODEL_FALLBACK)
        self.assertTrue(radar.tile_has_model_permission(far_from_observations, []))

    def test_gfs_interpolation_uses_hourly_frames_for_display_offsets(self):
        low = self._model_frame(60, 2.0)
        high = self._model_frame(120, 6.0)

        frame = radar.select_or_interpolate_gfs_frame([low, high], 90)

        self.assertEqual(frame.lead_minutes, 90)
        np.testing.assert_allclose(frame.rain_mm_h, np.full((2, 2), 4.0))
        np.testing.assert_allclose(frame.snow_mm_h, np.full((2, 2), 2.0))
        self.assertEqual(frame.metadata["interpolated_from_minutes"], [60, 120])


if __name__ == "__main__":
    unittest.main()
