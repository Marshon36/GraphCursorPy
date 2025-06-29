# GraphCursorPy

**GraphCursorPy** is a Python toolkit for detecting PKP precursors using Graph Neural Networks (GNNs) and related deep learning methods. It is designed for multistation waveform analysis.

---

## 🌟 Key Features

- GNN-based multistation detection of PKP precursors
- Integrated support for waveform processing, travel-time matching, and prediction
- Extendable to other weak seismic phases with similar dominant frequency (1 Hz) as PKP precursors (e.g., SKKKP)
- Built-in example data and precomputed travel-time tables for quick testing

---

## 📦 Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/your-username/GraphCursorPy.git
cd GraphCursorPy
pip install -e .
```

### 🔧 Required Dependencies

Before using GraphCursorPy, make sure the following external tools are installed:

* **[TauP](https://www.seis.sc.edu/taup/)** – for theoretical travel-time computation
* **[VesPy](https://github.com/NeilWilkins/VesPy)** – for seismic array analysis

## 🚀 Quick Start

### Run the example:

```
python graphcursorpy/example.py
```
