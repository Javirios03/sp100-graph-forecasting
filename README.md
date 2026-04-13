# S&P100 Graph Forecasting

This repository implements Geometric Deep Learning techniques to investigate whether graph-based representations of the S&amp;P100 improve weekly stock movement forecasting compared to traditional tabular and temporal models.

## Environment setup

```bash
uv python install 3.11
uv python pin 3.11
uv venv
uv sync
```

Run code with:

```bash
uv run python src/script.py
```

### Optional: Force kernel for Jupyter notebooks

```bash
uv run python -m ipykernel install --user --name sp100-graph-forecasting
```

### Optional: Enable CUDA for PyTorch and PyTorch Geometric

The Project TOML file doesn't explicitly document what the system is to do with all PyTorch related dependencies, so we will need to install them manually. The versions of PyTorch and PyTorch Geometric specified in the Project TOML file are compatible with CUDA 12.4, so we recommend installing that version of CUDA to ensure optimal performance.

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then, you may run the next command to check if CUDA is properly enabled:

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device count:', torch.cuda.device_count())"
```

which should return something like:

```torch: 2.6.0+cu124
cuda available: True
cuda version: 12.4
device count: 1
```

Then, and only once you have verified that CUDA is properly enabled, you can proceed to install PyTorch Geometric with CUDA support:

```bash
uv pip install torch-geometric
```

which should automatically install the compatible versions of `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv` with CUDA support.

## Scripts Execution

### Tests

In order to check that all the processed data files are consistent and valid (before feeding them to the models), we have implemented a set of basic tests in `tests/`. These tests check for consistency in the number of unique tickers and dates across all data files, as well as the validity of the target classes and split labels.

To run the tests, execute the following command:

```bash
uv run pytest tests/ -v -s -n auto
```

The flags used in the command are:

- `-v`: verbose output, which provides more detailed information about each test being run.
- `-s`: allows the output from print statements in the tests to be displayed in the console, which can be helpful for debugging.
- `-n auto`: enables parallel test execution using multiple CPU cores, which can significantly speed up the testing process, especially when there are many tests to run.
