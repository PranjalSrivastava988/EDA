# MAGIC Gamma Telescope Data Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%20|%20Pandas%20|%20Matplotlib%20|%20Seaborn%20|%20Scikit--learn-orange)

A comprehensive exploratory data analysis of the UCI MAGIC04 Gamma Telescope dataset for gamma particle detection.

## 📋 Overview

This project provides an in-depth analysis of the MAGIC Gamma Telescope dataset, which contains 19,020 samples of simulated gamma and hadron events in high-energy gamma astronomy. The goal is to distinguish gamma (signal) from hadron (background) events using their characteristic patterns.

<p align="center">
  <br>
  <em>PCA visualization showing separation between gamma and hadron events</em>
</p>

## 🔍 Dataset

The MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope dataset consists of:
- 19,020 instances
- 10 continuous feature attributes
- Binary target: gamma (signal) vs hadron (background)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)

## ✨ Features

- **Complete data pipeline** from loading to analysis
- **Modular, PEP8-compliant code** with comprehensive documentation
- **Insightful visualizations** with proper labeling and interpretation
- **Statistical analysis** including distribution characteristics and correlations
- **Advanced techniques** including PCA and t-SNE visualizations

## 📊 Key Findings

- Most discriminative features: `alpha`, `m3_long`, and `concentration`
- Significant class imbalance with more hadron than gamma events
- Several features exhibit high skewness requiring transformations
- High correlation between certain feature pairs (`length`/`size`, `concentration`/`concentration1`)
- Notable outliers in features like `asymmetry` and `m3_long`
- PCA shows promising class separation using just two principal components

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Installation

```bash
git clone https://github.com/username/magic-gamma-analysis.git
cd magic-gamma-analysis
```

### Usage

```python
# Run the full analysis pipeline
python magic_analysis.py

# Or import specific functions
from magic_analysis import load_and_clean_data, plot_correlation_heatmap
```

## 📁 Project Structure

```
.
├── eda.py                # Main analysis script
├── magic04.data          # Dataset file
├── magic04.name          # Dataset name
├── README.md             # This file
└── images/               # Generated visualizations
    ├── correlation.png
    ├── class_balance.png
    └── pca_visualization.png
```

## 📈 Visualizations

### Feature Distributions
<p align="center">
 
</p>

### Correlation Heatmap
<p align="center">
  
</p>

## 🔬 Analysis Components

1. **Data Loading & Cleaning**
   - Missing value detection
   - Type optimization
   - Class encoding (gamma → 1, hadron → 0)

2. **Univariate Analysis**
   - Distribution visualization
   - Skewness & kurtosis calculation
   - Outlier detection

3. **Bivariate & Correlation Analysis**
   - Feature correlation heatmap
   - Identification of highly correlated pairs
   - Feature importance for class discrimination

4. **Class Balance & Statistics**
   - Class distribution visualization
   - Group-by statistical summary
   - Violin plots for key features

5. **Outlier Detection & Handling**
   - IQR-based outlier identification
   - Transformation suggestions
   - Visual outlier inspection

6. **Dimensionality Reduction**
   - PCA visualization and interpretation
   - t-SNE clustering
   - Feature importance in principal components

## 💡 Recommendations

Based on the analysis, we recommend:

1. **Feature Transformations**: Apply log or sqrt transformations to skewed features
2. **Outlier Handling**: Cap extreme values or use robust scaling
3. **Feature Selection**: Remove highly correlated features
4. **Class Imbalance**: Use appropriate resampling techniques
5. **Feature Engineering**: Create new features from combinations of existing ones

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- Bock, R.K., et al. (2004). Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 516(2-3), 511-528.
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  <i>If you use this code in your research, please cite this repository.</i>
</p>
