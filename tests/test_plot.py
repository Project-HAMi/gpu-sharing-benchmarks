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
FIGURES = ("gpu-sharing-benchmarks-fixed-cache", "gpu-sharing-benchmarks-standard")


class PlotTest(unittest.TestCase):
    def run_plot(self, *args):
        return subprocess.run(
            [sys.executable, "-W", "error", str(ROOT / "plot.py"), *map(str, args)],
            cwd=self.work,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", str(self.work / "mpl-cache"))},
            timeout=60,
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_default_data_and_reproducible_exports(self):
        output_dir = self.work / "nested"
        result = self.run_plot("--output-dir", output_dir, "--preview")
        self.assertEqual(result.returncode, 0, result.stderr)
        originals = {}
        metadata = {}
        for figure, kind in zip(FIGURES, ("controlled", "regular")):
            original = (output_dir / f"{figure}.svg").read_bytes()
            originals[figure] = original
            self.assertTrue((output_dir / f"{figure}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            root = ET.fromstring(original)
            description = root.find(".//{http://purl.org/dc/elements/1.1/}description").text
            plotted = json.loads(description.split("\n", 1)[1])
            metadata[kind] = plotted[kind]
            self.assertEqual(set(plotted), {"source_sha256", kind})
            inputs = FILES[:1] if kind == "controlled" else FILES[1:]
            self.assertEqual(set(plotted["source_sha256"]), set(inputs))
            for name in inputs:
                self.assertEqual(plotted["source_sha256"][name], hashlib.sha256((DATA / name).read_bytes()).hexdigest())
            ids = {element.get("id") for element in root.iter()}
            self.assertEqual({i for i in ids if i and i.startswith("axes_")}, {"axes_1", "axes_2"})
            self.assertEqual("legend_1" in ids, kind == "regular")
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
        result = self.run_plot("--output-dir", output_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        for figure, original in originals.items():
            self.assertEqual(original, (output_dir / f"{figure}.svg").read_bytes())

    def test_invalid_inputs_preserve_existing_output(self):
        for case in ("missing", "duplicate", "negative", "invalid_regular"):
            with self.subTest(case=case):
                data = self.work / case
                data.mkdir()
                for name in FILES:
                    shutil.copy(DATA / name, data / name)
                path = data / ("fullstack.csv" if case == "invalid_regular" else "controlled.csv")
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    rows = list(reader)
                if case == "missing":
                    rows.pop()
                elif case == "duplicate":
                    rows.append(rows[0].copy())
                elif case == "negative":
                    rows[0]["tps_per_gpu_mean"] = "-1"
                else:
                    rows[0]["tps_per_gpu"] = "nan"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                outputs = [data / f"{figure}.svg" for figure in FIGURES]
                for output in outputs:
                    output.write_text("existing figure")
                result = self.run_plot("--data-dir", data, "--output-dir", data)
                self.assertNotEqual(result.returncode, 0)
                for output in outputs:
                    self.assertEqual(output.read_text(), "existing figure")
                self.assertIn("plot.py:", result.stderr)


if __name__ == "__main__":
    unittest.main()
