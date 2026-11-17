"""Exercise the plotting CLI using the committed result CSVs."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/2026-11-a100-sharing"
FILES = ("controlled.csv", "fullstack.csv", "fullstack-unlimited.csv")


class PlotTest(unittest.TestCase):
    def run_plot(self, *args):
        return subprocess.run(
            [sys.executable, "-W", "error", str(ROOT / "plot.py"), *map(str, args)],
            cwd=self.work,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLCONFIGDIR": str(self.work / "mpl-cache")},
            timeout=60,
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_default_data_and_reproducible_exports(self):
        output = self.work / "nested/throughput.svg"
        preview = self.work / "preview/throughput.png"
        result = self.run_plot("--output", output, "--preview", preview)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = output.read_bytes()
        self.assertTrue(preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        description = ET.fromstring(original).find(
            ".//{http://purl.org/dc/elements/1.1/}description"
        ).text
        metadata = json.loads(description.split("\n", 1)[1])
        for name in FILES:
            self.assertEqual(
                metadata["source_sha256"][name],
                hashlib.sha256((DATA / name).read_bytes()).hexdigest(),
            )
        with (DATA / "controlled.csv").open() as stream:
            fixed = {(r["mode"], r["configuration"]): r for r in csv.DictReader(stream)}
        regular = {}
        for name in FILES[1:]:
            with (DATA / name).open() as stream:
                for row in csv.DictReader(stream):
                    if name == "fullstack.csv" or row["configuration"] == "HAMi pair":
                        regular[(row["mode"], row["configuration"], int(row["total_concurrency_per_gpu"]))] = row
        self.assertEqual(len(metadata["controlled"]), 8)
        self.assertEqual(len(metadata["regular"]), 50)
        for point in metadata["controlled"]:
            row = fixed[(point["mode"], point["configuration"])]
            self.assertEqual(point["mean"], float(row["tps_per_gpu_mean"]))
            self.assertEqual(point["sample_std"], float(row["tps_per_gpu_sample_std"]))
        for point in metadata["regular"]:
            row = regular[(point["mode"], point["configuration"], point["concurrency"])]
            self.assertEqual(point["tps"], float(row["tps_per_gpu"]))
        result = self.run_plot("--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(original, output.read_bytes())

    def test_invalid_inputs_preserve_existing_output(self):
        for case in ("missing", "duplicate", "negative"):
            with self.subTest(case=case):
                data = self.work / case
                data.mkdir()
                for name in FILES:
                    shutil.copy(DATA / name, data / name)
                path = data / "controlled.csv"
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    rows = list(reader)
                if case == "missing":
                    rows.pop()
                elif case == "duplicate":
                    rows.append(rows[0].copy())
                else:
                    rows[0]["tps_per_gpu_mean"] = "-1"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                output = data / "preserved.svg"
                output.write_text("existing figure")
                result = self.run_plot("--data-dir", data, "--output", output)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), "existing figure")
                self.assertIn("plot.py:", result.stderr)


if __name__ == "__main__":
    unittest.main()
