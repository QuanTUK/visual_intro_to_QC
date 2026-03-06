# Quantum Viewer

An interactive quantum circuit simulator and educational visualization suite. Built on modern Qiskit (2.x) and Jupyter/Voilà, this package provides a state-aware Single Page Application (SPA) designed to teach quantum algorithms, statevector manipulation, and entanglement through real-time, deterministic visualizations.

________________________________________
## 1. Local Installation
Ensure your Python environment (virtual environment recommended) satisfies the optimized dependencies.

### Compilation Prerequisites (Rust & C++)
Some underlying dependencies require compiling extensions written in Rust. If your system lacks the necessary compilers, the package installation will fail. Please ensure the following are installed before proceeding:
* **Rust and Cargo:** Install the Rust toolchain globally via [rustup.rs](https://rustup.rs/). Run the installer and proceed with the default settings.
* **C++ Build Tools (Windows Only):** Windows users must also install the Microsoft C++ linker. Download the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/), run the installer, and check the box for the **Desktop development with C++** workload before installing.
*(Note: You must completely close and restart your terminal after installing these tools so your system recognizes the new environment variables.)*

### Installation Steps

```bash
# Clone the repository
git clone [https://github.com/your-repo/quantum-education-suite.git](https://github.com/your-repo/quantum-education-suite.git)
cd quantum-education-suite

# Install dependencies
pip install -r requirements.txt
```

Dependency Requirements

To prevent dependency resolver conflicts while maintaining mathematical stability, ensure your environment meets these minimum baselines:
* `qiskit>=2.3.0`
* `numpy>=2.4.0`
* `jupyterlab>=4.5.0`
* `voila>=0.5.0`
* `matplotlib>=3.10.0`
* `Pillow>=9.0.0`

________________________________________

## 2. API & Usage Instructions
The package can be executed either via standard Jupyter Notebook cells for interactive research or launched as a standalone web application.
A. Core Viewer Classes
You can instantiate interactive UI instances directly within a Jupyter Notebook.

### The Interactive Sandbox:

```Python
from qc_interactive_education_package import InteractiveViewer

# Initialize a 3-qubit sandbox
viewer = InteractiveViewer(num_qubits=3)

# Render a large, highly-detailed 12x8 inch plot bounded inside the Jupyter container
viewer.display(figsize=(12.0, 8.0), show_circuit=True)
```
### The Challenge Environment:
Evaluates student circuits against target statevectors using fidelity calculations ($F = |\langle \psi_{target} | \psi_{student} \rangle|^2$).
```Python
from qc_interactive_education_package import ChallengeViewer

# Initialize a challenge: Transition from |-> to |0>
viewer = ChallengeViewer(
    num_qubits=1,
    initial_state=[1, -1],  # Unnormalized input; class handles normalization
    target_state=[1, 0]     # Target |0>
)
viewer.display(show_circuit=True)
```

### Standalone Visualizations
For analysis without the interactive UI, you can pass a Qiskit Statevector directly into the visualization registry.
```Python
from qiskit.quantum_info import Statevector
import numpy as np
from qc_interactive_education_package.visualization import DimensionalCircleNotation

psi = Statevector([np.sqrt(3)/2, 0.5j], dims=(2,))
vis = DimensionalCircleNotation.from_qiskit(psi)
vis.show()

# Export graphics
# vis.exportPNG("current_state.png", title="My Quantum State")
```

### SPA Launchers

To run the pre-configured Single Page Application modes via IPC (Inter-Process Communication) and Voilà:
```Python
from qc_interactive_education_package import launch_app

# Launch the master Single Page Application
launch_app()

# Or launch a specific algorithmic challenge directly into the browser
launch_app(
    mode='challenge',
    num_qubits=2,
    initial_state=[1.0, 0.0, 0.0, 0.0],       
    target_state=[0.707, 0.0, 0.0, 0.707],    # Target Bell State
)
```
________________________________________

## 3. Cloud & Docker Deployment
The repository includes a highly optimized Dockerfile tailored for headless deployment to Google Cloud Run, AWS AppRunner, or local containerization.
Architecture Notes
* Base Image: Uses python:3.13-slim for a minimal attack surface and reduced overhead.
* Graphics Backend: Enforces ENV MPLBACKEND=Agg to ensure Matplotlib operates perfectly without a physical display server (X11).
* Routing: Automatically exposes the Voilà DOM server to port 8080 and disables XSRF checks for seamless cross-origin integration inside mobile WebViews.
Build and Run Instructions
1.	Build the Image:
```Bash
docker build -t quantum-app .
```
2.	Execute the Container:
```Bash
docker run -p 8080:8080 quantum-app
```
3.	Access: Navigate your browser to http://localhost:8080.
________________________________________

