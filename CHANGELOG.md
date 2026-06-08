# Changelog

All notable changes to this project are documented here.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `requirements.txt` listing all Python dependencies.
- `-o` / `--output-dir` CLI flag to `evaluate.py` for configurable figure output path.
- `getClassPixelCounts()` in `evaluate.py` to report per-class GT pixel counts alongside IoU metrics.
- `--test` and `--pred-dir` CLI flags to `writeImageLabelPathPairsToTxtFile.py`, exposing the previously unreachable `writeTestPredImageLabelPathPairsToTxtFile` function.
- Image-extension filtering (`IMAGE_EXTENSIONS`) across all directory-scanning scripts to skip non-image files such as `.DS_Store` or `Thumbs.db`.
- Size-mismatch validation in `blendImageAndLabel.py` before calling `cv2.addWeighted`.
- `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE` (Apache 2.0), and `CHANGELOG.md`.
- Google-style docstrings and module-level documentation for all source files.

### Changed
- **`colorTransformer.py`** — `clr2id` and the inline hash in `transform` now use base-256 instead of base-255, eliminating potential RGB colour collisions (e.g. `[255,0,0]` and `[0,1,0]` previously both hashed to 255).
- **`addToConfusionMatrix.pyx`** — Corrected `height_ui` / `width_ui` variable assignments (`shape[0]` = height, `shape[1]` = width) and fixed the argument order in the C call to match the C function signature.
- **`evaluate.py`** — `getMeanIOU` now uses `numpy.nanmean` instead of `numpy.mean` so that classes absent from the ground truth are excluded from the mean rather than propagating NaN.
- **`evaluate.py`** — `normalize_confusion_matrix` and `visualizeConfusionMatrix` guard against division by zero when a class has zero GT pixels.
- **`evaluate.py`** — `getPixelAccuracy` simplified to `numpy.trace(cm) / cm.sum()`.
- **`evaluate.py`** — `visualizeMeanIOUforClasses` replaced the O(n²) `labels.index(l)` lookup inside the loop with `enumerate`.
- **`evaluate.py`** — Bare `except:` replaced with `except ImportError:` in the Cython import block.
- **`writeImageLabelPathPairsToTxtFile.py`** — Replaced `img_path.replace('Images', 'Labels')` with explicit `osp.join` path construction to avoid incorrect substitution when `'Images'` appears in a parent directory name.
- **`writeImageLabelPathPairsToTxtFile.py`** — Removed unused `sys` import.
- **`blendImageAndLabel.py`** — Fixed incorrect argparse description (copy-paste error).
- **`setup.py`** — Replaced deprecated `distutils.core.setup` (removed in Python 3.12) with `setuptools.setup`.
- **`setup.py`** — Bare `except:` replaced with `except ImportError:`.
- All directory-scanning loops now call `sorted()` on `os.listdir()` results for deterministic, reproducible ordering.

---

## [0.1.0] — Initial Release

### Added
- `colorTransformer.py` — `UAVidColorTransformer` for bidirectional RGB ↔ train-ID conversion.
- `evaluate.py` — Pixel-wise IoU and accuracy evaluation with optional matplotlib visualisation.
- `prepareTrainIdFiles.py` — Batch RGB label → train-ID conversion.
- `convertTrainIdFiles2Color.py` — Batch train-ID → RGB colour conversion.
- `blendImageAndLabel.py` — Alpha-blend camera images with colour label overlays.
- `writeImageLabelPathPairsToTxtFile.py` — Generate image–label path-pair text files.
- `addToConfusionMatrix.pyx` / `addToConfusionMatrix_impl.c` — Cython/C++ confusion-matrix accumulation extension.
- `setup.py` — Build script for the Cython extension.
