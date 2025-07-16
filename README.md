# FRM–SD Network Analysis for Cryptocurrencies

This repository implements a pipeline to compute a Financial Risk Measure (FRM) for cryptocurrencies using penalized quantile‐lasso, construct monthly Stochastic Dominance (SD) networks from FRM signals, extract network‐centrality features, build a network‐risk factor, and evaluate its performance via Fama–French‐style regressions.

---

## 🚀 Quick Start

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/ralupu/FRM_SD_GNN.git
   cd FRM_SD_GNN
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Place daily crypto price CSV at:
   ```
   data/crypto_prices.csv
   ```

3. **Configure parameters**  
   Edit `config.yml`:
   ```yaml
   frequency: monthly
   window: 12
   step: 1
   quantile: 0.05
   bootstrap_draws: 0
   ```

4. **Run crypto pipeline**
   ```bash
   python run_crypto.py --config config.yml
   ```

5. **Inspect outputs**  
   All results are saved in `outputs/`:
   - `NetworkRisk.csv` — H–L factor series  
   - `frm_lambdas.csv` — monthly FRM λ values  
   - `centralities.csv` — in‐degree, out‐degree, PageRank, eigenvector  
   - `econ_results.csv` — regression coefficients & summary stats  

---

## 📂 Repository Structure

```
.
├── data/
│   └── crypto_prices.csv       # Daily price data
│
├── analysis/
│   ├── crypto_prep.py         # Load & resample crypto data
│   ├── frm_asgl.py            # Quantile‐lasso FRM λ estimation
│   ├── sd_utils.py            # Nonparametric SD tests
│   ├── sd_network.py          # Build scalar SD networks
│   ├── factor.py              # High–Low network‐risk factor
│   ├── features.py            # Prepare centrality features
│   └── econ_test.py           # Fama–French‐style regressions
│
├── config.yml                 # Pipeline parameters
├── run_crypto.py              # Driver script for crypto pipeline
└── requirements.txt           # Python dependencies
```

---

## 🧮 Methodology Overview

1. **Compute FRM λ**  
   \[
     \lambda_{i,t}
     = \min \{\lambda \mid eta_j(\lambda)=0,\ orall j\neq i\}
   \]

2. **Construct scalar SD network**  
   Edge \(i \to j\) if \(\lambda_{i,t} > \lambda_{j,t}\).

3. **Extract centralities**  
   In‐degree, out‐degree, PageRank, eigenvector.

4. **Build NetworkRisk factor**  
   \[
     \text{NetworkRisk}_t
     = \overline{R}_{\text{High},t}
     - \overline{R}_{\text{Low},t}
   \]

5. **Evaluate via regressions**  
   \[
     r_{i,t} - r_{f,t}
     = \alpha_i + \beta_i\,\text{NetworkRisk}_t + \varepsilon_{i,t}
   \]

---

## 🤝 Contributing

1. Fork the repo  
2. Create a new branch:  
   ```bash
   git checkout -b feature/XYZ
   ```  
3. Commit your changes & push:  
   ```bash
   git push origin feature/XYZ
   ```  
4. Open a Pull Request for review

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
