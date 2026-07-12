import unittest

import prepare_metar_data as metar


class PrepareMetarTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_active_stations(self):
        rows = [
            {
                "station_id": "EGLL",
                "observation_time": "2026-07-12T12:00:00Z",
                "latitude": "51.48",
                "longitude": "-0.45",
                "temp_c": "21.0",
                "dewpoint_c": "12.0",
                "wind_speed_kt": "10",
                "sea_level_pressure_mb": "1013.2",
                "precip_in": "0.01",
            },
            {
                "station_id": "EGLL",
                "observation_time": "2026-07-12T13:00:00Z",
                "latitude": "51.48",
                "longitude": "-0.45",
                "temp_c": "22.0",
                "wind_speed_kt": "M",
            },
        ]
        observations, stations = metar.normalize(rows)
        self.assertEqual(len(observations), 2)
        self.assertEqual(stations[0]["icao"], "EGLL")
        self.assertEqual(observations[0]["dewpoint"], 12.0)
        self.assertEqual(observations[0]["pressure_hpa"], 1013.2)
        self.assertEqual(observations[0]["precipitation_in"], 0.01)
        self.assertIsNone(observations[1]["wspd"])


if __name__ == "__main__":
    unittest.main()
