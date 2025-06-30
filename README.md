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

* **[TauP](https://www.seis.sc.edu/taup/)** – for theoretical travel-time computation
* **[VesPy](https://github.com/NeilWilkins/VesPy)** – for seismic array analysis

## 🚀 Quick Start

Run the provided example script:

```
cd graphcursorpy
python example.py
```
