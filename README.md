# GraphCursorPy

**GraphCursorPy** is a Python toolkit for processing PKP precursors based on Graph Neural Networks (GNNs) and related seismic methods.

---

## 🌟 Key Features

- GNN-based multistation detection of PKP precursors
- Estimation of scatterer locations and relative strengths
- Built-in example data and precomputed travel-time tables for quick testing

---

## 📦 Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Marshon36/GraphCursorPy.git
cd GraphCursorPy
pip install -e .
```

### 🔧 Required Dependencies

Before using GraphCursorPy, make sure the following external tools are installed:

* **TauP** – developed by Crotwell et al. and available from [https://www.seis.sc.edu/taup/](https://www.seis.sc.edu/taup/), for theoretical travel-time computation
* **VesPy** – developed by NeilWilkins and available from [https://github.com/NeilWilkins/VesPy](https://github.com/NeilWilkins/VesPy), for seismic array analysis

## 🚀 Quick Start

Run the provided example script:

```
cd graphcursorpy
python example.py
```

## Reference

Ma, S., Li, Z., Sun, D., Su, Y., Li, J., Si, X., & Zhu, J. (2025). Global Search of PKP Precursors With Graph Neural Network: Implications for Scatterers in the Lowermost Mantle. Geophysical Research Letters, 52(17), e2025GL115952. https://doi.org/10.1029/2025GL115952
