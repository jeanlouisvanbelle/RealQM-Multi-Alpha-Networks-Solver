# RealQM Multi-Alpha Network Solver

This repository hosts an open-source, first-principles computational physics framework designed to model multi-alpha nuclear clusters ($^{12}\text{C}$ and $^{16}\text{O}$) as self-organizing networks of phase-locked current loops. Moving away from traditional, static single-particle potential wells, the RealQM model investigates how structural macroscopic binding energy emerges natively from three-dimensional geometric optimization, phase synchronization, and classical electrodynamics.

## 🗂️ Core Numerical Pipelines

* **`sensitivity_analysis.py`**: This script contains a highly vectorized, closed-loop implementation of the Neumann double-line integral engine. It calculates the raw magnetic mutual inductance tensor across thousands of intersecting grid segments simultaneously. It utilizes an L-BFGS-B optimization framework to dynamically tilt, yaw, and relax 32 concurrent angular degrees of freedom for an $^{16}\text{O}$ macro-tetrahedron. It features an automated sensitivity matrix sweep that tests network stability across varied internal neutron current coherence factors ($\eta$ from 0.600 to 0.750), outputting precise energy yields and tracking structural convergence paths in real-time.
* **`topology_check.py`**: An independent mathematical validator that applies spectral graph theory and Kuramoto phase-locking equations to different network sizes (2-node Deuteron links, 4-node Alpha meshes, and 16-node Nested macro-tetrahedrons). By calculating the spectral radius and maximum eigenvalues ($\lambda_{\max}$) of each geometric adjacency matrix, this tool models how the collective phase-locking work multiplier ($\kappa$) shifts non-linearly based purely on the network’s topological complexity.

## 🚀 System Requirements & Quick Start

All scripts are written in standard Python 3 and require only the common scientific computing libraries `numpy` and `scipy`. They are designed to run natively within a standard terminal or command line prompt, eliminating browser timeouts and ensuring absolute numerical reproducibility.

To run the pipeline natively on your computer, clone this repository and execute the scripts directly via terminal or PowerShell:

```bash
python sensitivity_analysis.py
python topology_check.py
```

## ⚖️ License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the LICENSE file for details.

## 🤝 Authorship & Citation
Developed by the **RealQM AI Research Node (Gemini & DeepSeek)**. If you utilize this computational framework or data in your research, please cite the core working papers accordingly.

