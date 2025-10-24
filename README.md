# FRM–SD Network Analysis for Cryptocurrencies

This repository implements a comprehensive pipeline to compute Financial Risk Measures (FRM) for cryptocurrencies using penalized quantile-lasso, construct Stochastic Dominance (SD) networks from FRM signals with multiple test methods, extract network-centrality features, build network-risk factors, and evaluate performance via cross-sectional regressions.

## ✨ Recent Updates (v2.0)

- **� Multiple SD Test Methods**: 6 different stochastic dominance testing approaches
- **🐛 Fixed Regression Issues**: Resolved NaN handling and timezone alignment
- **🎬 Network Animation**: Create animated GIFs of network evolution over time
- **⚙️ Configurable Experiments**: Multiple configuration files for different time horizons
- **📊 Enhanced Analysis**: Improved cross-sectional regression and factor construction

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/ralupu/FRM_SD_GNN.git
cd FRM_SD_GNN
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
# Run with default config (all data, strict SD test)
python run_crypto.py --config config.yml --with_network

# Run specific experiment (180-day right-tail test)
python run_crypto.py --config config_right_tail_180d.yml --with_network

# Run without network generation (faster)
python run_crypto.py --config config.yml
```

### 3. Create Network Animations
```bash
# Create transparent GIF animation (last 90 days, every 5th frame)
python animate_network_separate.py outputs/right_tail_180d/networks.pkl -o network_evolution.gif -t -l 90 -s 5

# Create opaque animation with all frames
python animate_network.py outputs/networks.pkl -o full_animation.gif -i 500
```

---

## 📊 SD Test Methods

The pipeline now supports **6 different stochastic dominance testing methods**:

| Method | Description | Use Case |
|--------|-------------|----------|
| `strict` | Classical SD definition (λᵢ > λⱼ) | Conservative, fewer edges |
| `pysdtest` | Uses pysdtest library statistical tests | Statistical significance |
| `epsilon` | Allows small violations (ε-dominance) | Robust to noise |
| `left_tail` | Focus on lower quantiles only | Downside risk emphasis |
| `right_tail` | Focus on upper quantiles only | Upside potential emphasis |
| `bootstrap_prob` | Bootstrap probability > threshold | Probabilistic dominance |

### Configure SD Test Method
```yaml
# In config.yml
sd_network:
  test_mode: "right_tail"    # Choose from above methods
  alpha_tail: 0.25           # For tail-based methods
  epsilon: 0.05              # For epsilon method
  bootstrap_prob_threshold: 0.7  # For bootstrap method
```

---

## 🔧 Configuration Files

### Pre-configured Experiments
- **`config.yml`**: Base configuration (all data, strict test)
- **`config_right_tail_180d.yml`**: Right-tail test, last 180 days
- **`config_right_tail_730d.yml`**: Right-tail test, last 730 days  
- **`config_right_tail_full.yml`**: Right-tail test, full dataset
- **`config_pysdtest_180d.yml`**: Statistical test, last 180 days
- **`config_pysdtest_730d.yml`**: Statistical test, last 730 days

### Key Configuration Options
```yaml
# Time window configuration
test_window:
  last_days: 180      # Use last N days only (null = all data)
  
# Output directory
out_dir: "outputs/right_tail_180d"

# SD testing method
sd_network:
  test_mode: "right_tail"
  alpha_tail: 0.25
  
# Factor construction
factor:
  lag_days: 30        # Lag between network and returns
  top_k: 3           # Top K assets for long leg
  bottom_k: 3        # Bottom K assets for short leg
```

---

## 📂 Enhanced Repository Structure

```
.
├── data/
│   ├── daily_log_returns.csv         # Daily crypto returns
│   ├── daily_volumes.csv             # Daily crypto volumes  
│   ├── monthly_log_returns.csv       # Monthly returns (generated)
│   ├── monthly_volumes.csv           # Monthly volumes (generated)
│   └── cryptos/                      # Individual crypto CSV files
│
├── analysis/
│   ├── crypto_prep.py         # Data loading & preprocessing
│   ├── frm_asgl.py           # Quantile-lasso FRM estimation
│   ├── sd_utils.py           # 🆕 Multiple SD testing methods
│   ├── sd_network.py         # SD network construction
│   ├── factor.py             # Network-risk factor building
│   ├── features.py           # Centrality feature extraction
│   ├── econ_test.py          # 🔧 Fixed cross-sectional regressions
│   └── evaluate.py           # Performance evaluation
│
├── outputs/                   # Experiment results
│   ├── right_tail_180d/      # 180-day right-tail experiment
│   ├── right_tail_730d/      # 730-day right-tail experiment
│   ├── right_tail_full/      # Full dataset right-tail
│   ├── pysdtest_180d/        # 180-day statistical test
│   └── pysdtest_730d/        # 730-day statistical test
│
├── config*.yml               # 🆕 Multiple experiment configs
├── run_crypto.py             # 🔧 Enhanced pipeline runner
├── animate_network.py        # 🆕 Network animation (matplotlib)
├── animate_network_separate.py  # 🆕 Frame-isolated animation (PIL)
├── test_regression.py        # 🆕 Regression testing utilities
└── plots.py                  # 🔧 Enhanced visualization
```

---

## 🎬 Network Animation Features

### Animation Scripts

1. **`animate_network.py`** - Standard matplotlib animation
   ```bash
   python animate_network.py outputs/networks.pkl --output animation.gif --transparent --interval 500
   ```

2. **`animate_network_separate.py`** - Frame-isolated animation (recommended)
   ```bash
   python animate_network_separate.py outputs/networks.pkl --output clean_animation.gif -t -l 90 -s 3
   ```

### Animation Options
- `--transparent, -t`: Create transparent background GIFs
- `--last N, -l N`: Animate only last N frames  
- `--skip N, -s N`: Show every Nth frame (amplifies changes)
- `--interval N, -i N`: Milliseconds per frame
- `--output PATH, -o PATH`: Output file path

### Examples
```bash
# Transparent 90-day animation, every 5th frame
python animate_network_separate.py outputs/right_tail_180d/networks.pkl -o network_90d.gif -t -l 90 -s 5

# Weekly view (every 7th frame) with fast transitions  
python animate_network.py outputs/networks.pkl -o weekly.gif -s 7 -i 300

# Full dataset animation (may be large!)
python animate_network_separate.py outputs/right_tail_full/networks.pkl -o full_evolution.gif -s 10
```

---

## 🧮 Enhanced Methodology

### 1. FRM Computation with Multiple Horizons
```python
# Supports different time windows
λ_{i,t} = min { λ | β_j(λ) = 0 for all j ≠ i }
# Now configurable for last N days or full dataset
```

### 2. Advanced SD Network Construction  
```python
# Multiple test methods available:
if test_mode == "right_tail":
    edge_ij = (λ_i >= quantile(pooled_λ, 1-α)) and (λ_i > λ_j)
elif test_mode == "pysdtest":
    edge_ij = statistical_test(λ_i, λ_j, method="bootstrap")
# ... other methods
```

### 3. Robust Factor Construction
```python
# Fixed timezone and NaN handling
NetworkRisk_t = Avg(Returns[top_k_eigenvector]) - Avg(Returns[bottom_k_eigenvector])
# With proper alignment and missing data handling
```

### 4. Enhanced Cross-Sectional Analysis
```python
# Improved regression with proper error handling
(r_{i,t} - r_{f,t}) = α_i + β_i * NetworkRisk_t + ε_{i,t}
# Now handles NaN values and timezone misalignment
```

---

## 📈 Outputs Explanation

Each experiment generates comprehensive outputs:

### Core Results
- **`NetworkRisk_daily.csv`**: Daily network-risk factor values
- **`frm_index_full.csv`**: Complete FRM λ matrix over time  
- **`centralities.csv`**: Node centrality measures (in/out degree, PageRank, eigenvector)
- **`cross_sectional_factor_loadings.csv`**: Regression coefficients and statistics

### Network Files
- **`networks.pkl`**: Pickled time series of network objects for animation
- **`lambda_mat_full.pkl`**: Pickled FRM λ matrices for further analysis

### Visualizations  
- **`plot_*.png`**: Various plots (FRM evolution, centrality heatmaps, factor performance)
- **`sd_network_*.gif`**: Animated network evolution (if generated)

---

## 🔬 Running Experiments

### Standard Workflows

1. **Quick Test (180 days, right-tail)**
   ```bash
   python run_crypto.py --config config_right_tail_180d.yml --with_network
   python animate_network_separate.py outputs/right_tail_180d/networks.pkl -o test.gif -t -l 30 -s 3
   ```

2. **Statistical Comparison (pysdtest vs right-tail)**
   ```bash
   python run_crypto.py --config config_pysdtest_180d.yml --with_network
   python run_crypto.py --config config_right_tail_180d.yml --with_network
   # Compare results in outputs/pysdtest_180d/ vs outputs/right_tail_180d/
   ```

3. **Full Dataset Analysis**
   ```bash
   python run_crypto.py --config config_right_tail_full.yml --with_network
   python animate_network_separate.py outputs/right_tail_full/networks.pkl -o full.gif -s 20
   ```

### Custom Configuration
```yaml
# Create your own config_custom.yml
frequency: daily
test_window:
  last_days: 365
out_dir: "outputs/custom_365d"
sd_network:
  test_mode: "epsilon" 
  epsilon: 0.03
factor:
  lag_days: 21
  top_k: 5
  bottom_k: 5
```

---

## 🐛 Troubleshooting

### Common Issues

1. **"SVD did not converge" in regressions**
   - ✅ **Fixed**: NaN handling and timezone alignment implemented

2. **Transparent GIFs show frame accumulation**
   - ✅ **Fixed**: Use `animate_network_separate.py` for clean animations

3. **Missing dependencies**
   ```bash
   pip install networkx scipy scikit-learn matplotlib pillow pandas numpy
   ```

4. **Memory issues with large datasets**
   - Use `test_window.last_days` to limit data size
   - Increase frame skipping (`-s` parameter) in animations

### Performance Tips
- Set `last_days` for faster experimentation
- Use `--skip` parameter for smoother animations
- Skip `--with_network` flag if you don't need network files
- Use `animate_network_separate.py` for large datasets

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-sd-method`
3. Add your SD test method to `analysis/sd_utils.py`
4. Update configuration schema in `config.yml` 
5. Add tests in `test_*.py`
6. Commit changes: `git commit -am 'Add new SD test method'`
7. Push branch: `git push origin feature/new-sd-method`
8. Open Pull Request

### Adding New SD Test Methods
```python
# In analysis/sd_utils.py
def _sd_gap_your_method(z_pooled, alpha, **kwargs):
    """Your custom SD test implementation"""
    # Implement your logic here
    threshold = your_threshold_calculation(z_pooled, alpha)
    return threshold

# Register in Mode enum and sd_compare function
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 🔗 Citation

If you use this code in your research, please cite:
```bibtex
@software{frm_sd_gnn,
  title={FRM-SD Network Analysis for Cryptocurrencies},
  author={Your Name},
  year={2025},
  url={https://github.com/ralupu/FRM_SD_GNN}
}
```
