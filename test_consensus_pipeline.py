import datetime as dt
import json
import pathlib
import tempfile
import unittest

import consensus_pipeline as pipeline


class ConsensusPipelineTests(unittest.TestCase):
    def test_ema_rewards_lower_error(self):
        result = pipeline.update_scores(
            {}, {"gfs": [0.25], "icon": [2.0]}, alpha=1.0
        )
        self.assertGreater(result["weights"]["gfs"], result["weights"]["icon"])
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0)

    def test_previous_score_decays_instead_of_resetting(self):
        previous = {"scores": {"gfs": 2.0}, "sample_counts": {"gfs": 3}}
        result = pipeline.update_scores(previous, {"gfs": [0.0]}, alpha=0.25)
        self.assertAlmostEqual(result["scores"]["gfs"], 1.5)
        self.assertEqual(result["sample_counts"]["gfs"], 4)

    def test_nearest_forecast_enforces_tolerance(self):
        observed = dt.datetime(2026, 7, 12, 12, tzinfo=dt.timezone.utc)
        rows = [{
            "station": "EGLL",
            "valid_time": "2026-07-12T16:00:00Z",
            "temperature_c": 20,
        }]
        self.assertIsNone(pipeline.nearest_forecast(rows, "EGLL", observed))

    def test_region_key_matches_twenty_degree_pack_boundaries(self):
        self.assertEqual(pipeline.region_key(53.8, -1.5), "+50_-020")
        self.assertEqual(pipeline.region_key(-89.0, 179.0), "-90_+160")

    def test_regional_lead_scores_bias_and_holdout_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            providers = []
            forecasts = {
                "gfs": (20.0, 10.0),
                "icon": (24.0, 30.0),
            }
            for provider, (temperature, wind) in forecasts.items():
                path = root / f"{provider}.json"
                path.write_text(json.dumps({
                    "provider": provider,
                    "forecasts": [{
                        "station": "EGLL",
                        "valid_time": "2026-07-12T12:00:00Z",
                        "lead_hours": 48,
                        "temperature_c": temperature,
                        "wind_kmh": wind,
                    }, {
                        "station": "EGCC",
                        "valid_time": "2026-07-12T12:00:00Z",
                        "lead_hours": 48,
                        "temperature_c": temperature,
                        "wind_kmh": wind,
                    }],
                }))
                providers.append(path)
            observations = [{
                "icaoId": "EGLL",
                "obsTime": "2026-07-12T12:00:00Z",
                "latitude": 51.47,
                "longitude": -0.45,
                "temp": 20.0,
                "wspd": 5.4,
            }, {
                "icaoId": "EGCC",
                "obsTime": "2026-07-12T12:00:00Z",
                "latitude": 53.35,
                "longitude": -2.27,
                "temp": 21.0,
                "wspd": 6.0,
            }]

            result = pipeline.score_providers(providers, observations, {}, 1.0)
            regional = result["regions"]["+50_-020"]
            weights = regional["lead_buckets"]["25-72"]["weights"]
            self.assertGreater(weights["gfs"], weights["icon"])
            self.assertGreater(regional["temperature_bias"]["day"]["samples"], 0)
            validation = result["validation"]["+50_-020"]["25-72"]
            self.assertIn("consensus", validation)
            self.assertEqual(result["holdout_station_count"], 1)

    def test_error_history_is_bounded(self):
        previous = {"error_history": {"gfs": list(range(100))}}
        result = pipeline.update_scores(previous, {"gfs": [1.0]}, alpha=0.1)
        self.assertEqual(len(result["error_history"]["gfs"]), pipeline.HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
