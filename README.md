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

2. **Activate the virtual environment:**
 * On Windows:
   ```bash
   qc_env\Scripts\activate
 * On macOS and Linux: 
   ```bash
   source qc_env/bin/activate

3. Install the required dependencies:
Ensure your environment is activated (indicated by (`qc_env`) in your terminal prompt), then install the necessary packages. Note: The `qc_education_package` and `qiskit` are required for the interactive visualizations.
   ```bash
   pip install -r requirements.txt

4. Launch the Jupyter environment:

   ```bash
   jupyter notebook


Alternatively, open the project directory in VS Code and select the `qc_env` Python interpreter for your notebooks.

## 3. Module Structure

The repository is structured sequentially to build your understanding from fundamental qubit mechanics to more complex, multi-qubit entanglement protocols. It is highly recommended to progress through the notebooks in numerical order.

`0_overview_of_the_course.ipynb`

High-level introduction to the course methodology and the three pillars of quantum technologies (sensing, computation, and communication).

`1_Introduction_to_qubits.ipynb`

Superpositions: From Bits to Qubits. Introduces the mathematical representation of a qubit as a vector in a two-dimensional complex Hilbert space.

`2_single_qubit_operations.ipynb`

Visualizing Quantum Processes. Examines unitary operators (Pauli-X, Pauli-Z, and Hadamard gates) and the intricacies of quantum measurements.

`3_BB84_protocol.ipynb`

Secure Quantum Communication. Implements the first quantum cryptography protocol proposed by Bennett and Brassard, leveraging the no-cloning theorem and superposition to detect eavesdroppers.

`4_introduction_to_entanglement.ipynb`

Transitioning from single to multi-qubit systems. Explores CNOT gates, the creation of fundamental Bell states, and visualization via Dimensional Circle Notation (DCN).

`5_Superdense_Coding.ipynb`

Superdense Coding and Bell Measurements. Demonstrates how to transmit two bits of classical information utilizing a single entangled qubit and local operations.

`6_The_Bell_experiment.ipynb`

Surprising Consequences of Entanglement. Explores local realism, hidden variables, and the theoretical violation of the CHSH inequality.

`7_Ekert_91_protocol.ipynb`

Entanglement-based Security. Applies the principles of the Bell experiment to Quantum Key Distribution, mathematically certifying a secure channel via CHSH inequality violations.

`8_Quantum_Teleportation.ipynb`

Detailed breakdown and implementation of transferring quantum state information across space utilizing an entangled resource pair and classical communication channels.

`9_communication_to_computation.ipynb`

Universality and Error Correction. An outlook connecting communication protocols to computational paradigms, including oracle-based algorithms and logical qubit stabilization.

## 4. Credits and Feedback

This course material was authored by Jonas Bley. The qc_education_package was developed by Nikolas Longen, Patrick Pfau, and Jonas Bley. The project was under the supervision of Maximilian Kiefer-Emmanouilidis. Generative AI was utilized to generate parts of the text and code. AI Content was humanly reviewed.

Continuous improvement is vital for educational resources. Upon concluding the notebooks, please submit your feedback via the official survey: [Course Evaluation Form](https://forms.gle/FQo587RCB5DxJETT8).

