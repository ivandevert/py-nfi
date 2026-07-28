# py-nfi

[![PyPI version](https://img.shields.io/pypi/v/py-nfi.svg)](https://pypi.org/project/py-nfi/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-nfi.svg)](https://pypi.org/project/py-nfi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21632630.svg)](https://doi.org/10.5281/zenodo.21632630)

Compute the **normalized frequency index (nFI)** from earthquake spectra. nFI is a
data-driven measure of the relative high-frequency content of a seismic source,
providing an observation-based alternative to stress-drop estimation for
characterizing source behavior at high frequencies.

## Installation

```bash
pip install py-nfi
```

or, with conda:

```bash
conda install -c conda-forge py-nfi
```

py-nfi requires the following packages:
- numpy
- pandas
- scipy
- tqdm

## Quick start

```python
import numpy as np
import pandas as pd
from nfi import nFIEstimator

# df_records: one row per event-station record (a single observed spectrum)
# spectra:    N x nf array of signal amplitude spectra, aligned row-for-row
#             with df_records (the signal windows; noise spectra are used
#             upstream for signal-to-noise screening, not passed here). Each
#             record in spectra has already passed signal-to-noise criteria.
# f:          length-nf frequency array (0 to Nyquist)

est = nFIEstimator(df_records, spectra, f, save_dir="results/")
est.compute()

# per-event results, including the 'nfi' column, are written to results/
# and available as est.df_events
```
### df_records structure
`df_records` must contain the following for each record at minimum:
- `event_name`, the recorded event's **unique** identifier;
- `channel_name`, the recording station's **unique** identifier;
- `emag`, `elat`, `elon`, and `edep` (km), the event's magnitude and location ;
- `slat`, `slon`, and `selev`, the station's location; and
- `deldist`, the horizontal event-station distance in kilometers;

This pandas DataFrame should be of length N, with each row corresponding to the nth row in `spectra`. `e*` fields describe the event and `s*` fields describe the recording station.

## Method

nFI is computed from spectra in three stages.

**1. Frequency index.** For every observed spectrum, we take the log ratio of the
median amplitude in a high-frequency band to that in a low-frequency band:

```
logbeta = log10(A_HF) - log10(A_LF)
```

This quantifies the richness of high frequencies relative to low frequencies for a
single recording. The low band should sample the flat portion of the displacement
spectrum below the corner frequency, and the high band should sample the decay
above it.

**2. Path and station correction.** Observed spectra carry path and station effects
that must be removed to isolate source variability. For each target event, we
identify small nearby "calibration" earthquakes recorded at the same stations.
Differencing `logbeta` between a target and a calibration event at a shared station
cancels the common station and path terms. Taking the median of these differences
across calibration events per station, then the median across stations, yields the
corrected source term `dlogbeta` for each event. Calibration events are selected
within a narrow magnitude range and a limited horizontal and vertical distance of
the target.

**3. Magnitude correction.** Larger earthquakes radiate proportionally less
high-frequency energy, so `dlogbeta` retains a magnitude dependence. We estimate
this trend by taking the median `dlogbeta` in magnitude bins, smoothing it, and
subtracting the interpolated trend from each event. The result is **nFI**: by
construction it has approximately zero median at any magnitude, with positive
values indicating high-frequency enrichment and negative values indicating
depletion.

For full detail and validation of the method, see Vandevert et al. (2026) below.

## Reproducing the Ridgecrest results

Clone this repository:
```bash
git clone https://github.com/ivandevert/py-nfi your/local/save/path/
```

Download the precomputed, signal-to-noise passing spectra from [Zenodo](https://zenodo.org/records/21463045) into the repo's `ridgecrest/data/` subdirectory, unzip, and remove the compressed folder:
```bash
wget https://zenodo.org/records/21463045/files/spectra.zip spectra.zip

md5sum -c - <<< "10e714977e73dd396721194e88fd88e0  spectra.zip"

unzip spectra.zip -d py-nfi/ridgecrest/data

rm spectra.zip
```

Launch `run_nfi_compute.ipynb` and run all cells.

## Citation

If you use this software, please cite:

> Vandevert, I., Shearer, P. M., & Fan, W. (2026). Using a High-to-Low-Frequency
> Spectral Ratio to Distinguish Variations in Earthquake Source Properties.
> *Bulletin of the Seismological Society of America.*
> https://doi.org/10.1785/0120250171

## License

Released under the [MIT License](LICENSE).
