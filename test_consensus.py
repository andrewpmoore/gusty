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

    def test_temperature_sanitizer_rejects_zero_kelvin(self):
        values = weather.xr.DataArray(
            np.array([[0.0, 273.15, 400.0]], dtype=np.float32)
        )
        sanitized = weather.sanitize_temperature_kelvin(values).values
        self.assertTrue(np.isnan(sanitized[0, 0]))
        self.assertAlmostEqual(float(sanitized[0, 1]), 273.15, places=2)
        self.assertTrue(np.isnan(sanitized[0, 2]))

    def test_daily_temperature_missing_values_remain_missing(self):
        values = weather.prepare_grid_values(
            np.array([[np.nan, 280.0]], dtype=np.float32),
            1,
            preserve_missing=True,
        )
        packed, _, _ = weather.quantize_int16(values)
        self.assertEqual(int(packed[0]), weather.MISSING_VALUE)

    def test_multi_forecast_summary_includes_gfs_and_other_fields(self):
        channels = {
            weather.MODEL_FIELD_CHANNELS["temperature"]: np.array(
                [[280.0]], dtype=np.float32
            ),
            weather.MODEL_FIELD_CHANNELS["wind_u"]: np.array(
                [[3.0]], dtype=np.float32
            ),
            weather.MODEL_FIELD_CHANNELS["wind_v"]: np.array(
                [[4.0]], dtype=np.float32
            ),
            weather.MODEL_FIELD_CHANNELS["pressure"]: np.array(
                [[101300.0]], dtype=np.float32
            ),
            weather.MODEL_FIELD_CHANNELS["humidity"]: np.array(
                [[0.75]], dtype=np.float32
            ),
        }
        components = dict(weather.multi_forecast_daily_components({
            "gfs": [channels],
            "blend": [channels],
        }))
        field_channels = weather.MULTI_FORECAST_FIELD_CHANNELS
        self.assertIn(field_channels["temperature"]["high"]["gfs"], components)
        wind_channel = field_channels["wind_speed"]["high"]["gfs"]
        self.assertAlmostEqual(float(components[wind_channel][0]), 5.0)
        self.assertIn(field_channels["pressure"]["low"]["gfs"], components)
        self.assertIn(field_channels["humidity"]["high"]["gfs"], components)
        blend_channels = weather.MULTI_FORECAST_BLEND_CHANNELS
        self.assertIn(blend_channels["temperature"]["high"], components)
        self.assertIn(blend_channels["wind_speed"]["low"], components)

    def test_daily_extrema_require_broad_time_coverage(self):
        self.assertTrue(weather.has_daily_extrema_coverage([24, 30, 36, 42]))
        self.assertFalse(weather.has_daily_extrema_coverage([120]))
        self.assertFalse(weather.has_daily_extrema_coverage([30, 36]))

    def test_multi_forecast_field_filter_keeps_only_requested_channels(self):
        components = [
            (channel, np.array([float(channel)], dtype=np.float32))
            for field_name in weather.MULTI_FORECAST_FIELD_CHANNELS
            for channel in weather.multi_forecast_field_channel_numbers(
                field_name
            )
        ]
        temperature = weather.multi_forecast_components_for_field(
            components, "temperature"
        )
        temperature_channels = {channel for channel, _ in temperature}
        self.assertEqual(
            temperature_channels,
            weather.multi_forecast_field_channel_numbers("temperature"),
        )
        self.assertTrue(temperature_channels.isdisjoint(
            weather.multi_forecast_field_channel_numbers("wind_speed")
        ))

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
