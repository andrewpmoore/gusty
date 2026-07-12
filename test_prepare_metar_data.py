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
                "wind_speed_kt": "10",
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
        self.assertIsNone(observations[1]["wspd"])


if __name__ == "__main__":
    unittest.main()
