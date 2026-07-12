import datetime as dt
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


if __name__ == "__main__":
    unittest.main()
