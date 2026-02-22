Visual Introduction to Quantum Communication and Computation

This repository provides a visual introduction to quantum communication and computation. It covers various applications of superpositions and entanglement in fundamental quantum communication protocols, including BB84, E91, superdense coding, and quantum teleportation.

The environment leverages interactive Jupyter notebooks and Qiskit to provide dynamic state vector visualizations using Dimensional Circle Notation (DCN).

Below are the systematic instructions required to prepare your local environment.
1. Prerequisites and System Preparation

To execute the interactive visualizations, a functioning Python environment with Jupyter support is required. We strongly recommend isolating your project dependencies using a virtual environment to prevent version conflicts with other system packages.
Python and Package Managers

Ensure you have Python 3.8 or higher installed.

    Standard Installation: Download the latest release from the official Python website. pip (the Python package installer) is included by default.

    Alternative Solution (Recommended for Scientific Computing): Install Miniconda or Anaconda. Conda manages both Python execution versions and complex binary dependencies highly effectively.

Development Environment (IDE)

You will need an environment capable of rendering Jupyter notebooks (.ipynb files) and ipywidgets.

    JupyterLab: The standard, robust web-based interactive development environment.

    Visual Studio Code: A lightweight, extensible IDE. If you choose this route, ensure you install the official Jupyter Extension for VS Code to support widget rendering.

2. Installation Instructions

Navigate to the project directory in your terminal and follow one of the two methods below to install the necessary packages.
Method A: Using standard pip and venv

    Create a virtual environment:
    Bash

    python -m venv qc_env

    Activate the environment:

        Windows: qc_env\Scripts\activate

        macOS/Linux: source qc_env/bin/activate

    Install the core dependencies (including Qiskit and visualization tools):
    Bash

    pip install qiskit qiskit-aer matplotlib ipywidgets numpy jupyterlab

Method B: Using conda

    Create a dedicated conda environment:
    Bash

    conda create -n qc_env python=3.10

    Activate the environment:
    Bash

    conda activate qc_env

    Install dependencies from the conda-forge channel:
    Bash

    conda install -c conda-forge qiskit matplotlib ipywidgets numpy jupyterlab

3. Launching the Project

Once the dependencies are resolved, launch the Jupyter interface from your terminal:
Bash

jupyter lab

Open the primary notebook files to begin interacting with the DCN Viewer and exploring the quantum protocols.
