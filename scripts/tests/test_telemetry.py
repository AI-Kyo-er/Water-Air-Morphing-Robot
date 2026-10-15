"""正式验证后续实测日志的接入和指标计算。"""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.import_telemetry import REQUIRED, read_telemetry, telemetry_metrics


class TelemetryTests(unittest.TestCase):
    def write_input(self, path, times):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED)
            writer.writeheader()
            for t in times:
                row = {name: 0.0 for name in REQUIRED}
                row["time_s"], row["x_m"] = t, 1.0
                writer.writerow(row)

    def test_analytic_position_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            self.write_input(path, [0, 0.1, 0.2])
            result = telemetry_metrics(read_telemetry(path))
            self.assertEqual(result["position_rmse_m"], 1.0)
            self.assertEqual(result["tilt_rate_max_deg_s"], 0.0)

    def test_repeated_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            self.write_input(path, [0, 0])
            with self.assertRaises(ValueError):
                read_telemetry(path)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            path.write_text("time_s,x_m\n0,0\n1,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_telemetry(path)


if __name__ == "__main__":
    unittest.main()
