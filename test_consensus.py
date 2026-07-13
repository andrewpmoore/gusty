import unittest

import numpy as np

import fetch_weather_data as weather
import validate_consensus as validation


class ConsensusTests(unittest.TestCase):
    def test_validation_accepts_nanosecond_model_timestamp(self):
        parsed = validation.parse_time("2026-07-13T06:00:00.123456789")
        self.assertEqual(parsed.microsecond, 123456)

    def test_missing_values_use_reserved_sentinel(self):
        values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        packed, scale, offset = weather.quantize_int16(values)
        self.assertEqual(int(packed[1]), -32768)
        decoded = packed.astype(np.float32) * scale + offset
        self.assertAlmostEqual(float(decoded[0]), 1.0, places=3)
        self.assertAlmostEqual(float(decoded[2]), 3.0, places=3)

    def test_related_models_share_one_family_vote(self):
        names = ["gfs", "aigfs", "ifs", "icon"]
        weights = weather._family_balanced_weights(names)
        self.assertAlmostEqual(float(weights[0] + weights[1]), 1.0)
        self.assertAlmostEqual(float(weights[2]), 1.0)
        self.assertAlmostEqual(float(weights[3]), 1.0)

    def test_family_balanced_median_resists_duplicate_model(self):
        stack = np.array([280.0, 282.0, 290.0, 292.0], dtype=np.float32)
        stack = stack.reshape(4, 1, 1)
        weights = weather._family_balanced_weights(
            ["gfs", "aigfs", "ifs", "icon"]
        )
        result = weather._weighted_median(stack, weights)
        self.assertAlmostEqual(float(result[0, 0]), 290.0)

    def test_skill_weights_are_normalized_within_model_family(self):
        weights = weather._family_balanced_weights(
            ["gfs", "aigfs", "ifs", "icon"],
            {"gfs": 0.75, "aigfs": 0.25, "ifs": 0.4, "icon": 0.8},
        )
        self.assertAlmostEqual(float(weights[0]), 0.375)
        self.assertAlmostEqual(float(weights[1]), 0.125)
        self.assertAlmostEqual(float(weights[2]), 0.4)
        self.assertAlmostEqual(float(weights[3]), 0.8)

    def test_condition_ties_prefer_consequential_family(self):
        stack = np.array([
            weather.CONDITION_CODES["Clear"],
            weather.CONDITION_CODES["MostlyClear"],
            weather.CONDITION_CODES["Rain"],
            weather.CONDITION_CODES["Cloudy"],
        ], dtype=np.float32).reshape(4, 1, 1)
        weights = weather._family_balanced_weights(
            ["gfs", "aigfs", "ifs", "icon"]
        )
        result = weather._condition_consensus(stack, weights)
        self.assertEqual(int(result[0, 0]), weather.CONDITION_CODES["Rain"])


if __name__ == "__main__":
    unittest.main()
