# Data

**Breast Cancer Wisconsin (Diagnostic)** dataset (Street, Wolberg & Mangasarian, 1993), UCI Machine Learning Repository.
Loaded programmatically through `sklearn.datasets.load_breast_cancer`, so no download is required.

- **Samples:** 569 (212 malignant, 357 benign). No missing values.
- **Features:** 30 numeric values. Ten nuclear measurements (radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension) each summarised as
  *mean*, *standard error* and *worst* (mean of the three largest values).
- **Origin:** computed from digitised images of fine-needle aspirates of breast masses. The images are not included.
- **Target:** this project uses `malignant = 1`, `benign = 0`. scikit-learn's own encoding is the reverse
  (malignant = 0); `src/data_loader.py` flips it.

`python -m src.data_loader` writes a CSV copy (`breast_cancer_wisconsin.csv`) into this folder.

## Caveats
Single institution, small, and already cleaned. Not representative of other scanners, protocols or populations.
Check the UCI licence/citation terms before redistributing.
