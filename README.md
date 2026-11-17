# GPU sharing benchmarks

- English (v2.10.0): [Report](https://project-hami.io/docs/v2.10.0/userguide/nvidia-device/gpu-sharing-benchmarks) · [Reproduction guide](https://project-hami.io/docs/v2.10.0/userguide/nvidia-device/gpu-sharing-reproduction)
- 简体中文（v2.10.0）：[测试报告](https://project-hami.io/zh/docs/v2.10.0/userguide/nvidia-device/gpu-sharing-benchmarks) · [复现指南](https://project-hami.io/zh/docs/v2.10.0/userguide/nvidia-device/gpu-sharing-reproduction)
- [Results / 测试结果](results/)

## Plot throughput results

The plotting script reads the committed CSVs and generates a four-panel figure:
fixed-cache Graph/Eager throughput with sample-standard-deviation error bars, and
standard Graph/Eager throughput across the five concurrency levels. It runs on a
CPU and does not require CUDA or a Kubernetes cluster.

Use Python 3.12 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-plot.txt
.venv/bin/python plot.py --preview out/throughput.png
```

The default input is `results/2026-11-a100-sharing/`, and the default output is
`out/throughput.svg`. English labels use Matplotlib's bundled DejaVu Serif font.
The SVG embeds glyph outlines and includes the input CSV SHA-256 hashes and
plotted values in its metadata.

For Chinese labels, install a Chinese font such as Noto Serif CJK SC, or pass a
font file explicitly:

```sh
.venv/bin/python plot.py --language zh --font /path/to/chinese-font.otf \
  --output out/throughput-zh.svg --preview out/throughput-zh.png
```

Use `--data-dir` to select another directory containing `controlled.csv`,
`fullstack.csv`, and `fullstack-unlimited.csv`. The script expects the same
configuration names and concurrency levels as the committed dataset. It checks
for missing, duplicate, and invalid plotted values before writing the figure.

Run the plotting checks with:

```sh
.venv/bin/python -m unittest discover -s tests
```
