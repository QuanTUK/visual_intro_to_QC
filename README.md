# Visual Introduction to Quantum Communication and Computation

This repository provides a visual introduction to quantum communication and computation. It covers various applications of superpositions and entanglement in fundamental quantum communication protocols, including BB84, E91, superdense coding, and quantum teleportation. 

The environment leverages interactive Jupyter notebooks and Qiskit to provide dynamic state vector visualizations using Dimensional Circle Notation (DCN).

Below are the systematic instructions required to prepare your local development environment.

## 1. Prerequisites and System Preparation

To execute the interactive visualizations, a functioning Python environment with Jupyter support is required. It is strictly recommended to isolate your project dependencies using a virtual environment to prevent version conflicts with other system-level packages.

### Python and Package Managers
Ensure you have Python 3.8 or higher installed. 
* Download the latest release from the [official Python website](https://www.python.org/downloads/). 
* `pip` (the standard Python package installer) is included by default.

### Development Environment (IDE)
You will need an environment capable of rendering Jupyter notebooks (`.ipynb` files) and `ipywidgets` seamlessly.
* **JupyterLab:** The standard, robust web-based interactive development environment. 
* **Visual Studio Code:** A lightweight, extensible IDE. If utilizing VS Code, you must install the official [Jupyter Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) to support frontend widget rendering.

---

## 2. Installation Instructions

Navigate to the root directory of this project in your terminal and execute the following commands to construct the isolated environment.

1. **Create the virtual environment:**
   ```bash
   python -m venv qc_env

   
