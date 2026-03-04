import sys
import subprocess
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import json
import qiskit.qasm2
from qiskit.quantum_info import Statevector

# Import your viewer classes
from qc_interactive_education_package import InteractiveViewer, ChallengeViewer
from quantum_library import QuantumCurriculum


def launch_tool(num_qubits=3, initial_state=None, show_circuit=True, preloaded_circuit=None):
    """
    Launches the Voilà server for the interactive quantum sandbox.
    """
    print(f"Initializing Quantum Sandbox Environment ({num_qubits} Qubits)...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "app.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'app.ipynb' at {notebook_path}")
        return

    custom_env = os.environ.copy()
    custom_env["VIEWER_QUBITS"] = str(num_qubits)
    custom_env["VIEWER_INITIAL"] = json.dumps(initial_state)
    custom_env["VIEWER_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # NEW: Serialize the Qiskit circuit to a QASM string
    if preloaded_circuit is not None:
        import qiskit.qasm2
        try:
            custom_env["VIEWER_PRELOADED_QASM"] = qiskit.qasm2.dumps(preloaded_circuit)
        except Exception as e:
            print(f"Warning: Failed to serialize preloaded circuit to QASM. {e}")

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
        "--Voila.log_level=ERROR",
        "--ServerApp.log_level=CRITICAL"
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Sandbox...")

def launch_app():
    """
    Launches the master Single Page Application (SPA) in the browser via Voilà.
    """
    print("Initializing Quantum Education Suite SPA...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "index.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'index.ipynb' at {notebook_path}")
        return

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
        "--Voila.log_level=ERROR",
        "--ServerApp.log_level=CRITICAL"
    ]

    try:
        # No custom_env needed here, as the SPA handles its own parameters internally
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Education Suite...")


def launch_challenge(num_qubits=1, initial_state=[1, 0], target_state=[1, -1], show_circuit=True, preloaded_circuit=None):
    """
    Launches the Voilà server with dynamically injected quantum states and an optional solution.
    """
    print(f"Initializing Quantum Challenge Environment ({num_qubits} Qubits)...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "challenge.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'challenge.ipynb' at {notebook_path}")
        return

    custom_env = os.environ.copy()
    custom_env["CHALLENGE_QUBITS"] = str(num_qubits)
    custom_env["CHALLENGE_INITIAL"] = json.dumps(initial_state)
    custom_env["CHALLENGE_TARGET"] = json.dumps(target_state)
    custom_env["CHALLENGE_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # NEW: Serialize the Qiskit circuit to a QASM string
    if preloaded_circuit is not None:
        import qiskit.qasm2
        try:
            custom_env["CHALLENGE_PRELOADED_QASM"] = qiskit.qasm2.dumps(preloaded_circuit)
        except Exception as e:
            print(f"Warning: Failed to serialize preloaded circuit to QASM. {e}")

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
        "--Voila.log_level=ERROR",
        "--ServerApp.log_level=CRITICAL"
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Challenge...")


# ==========================================
# 1. JSON INGESTION & PARSING PIPELINE
# ==========================================

def load_libraries(filepath="library.json"):
    """Reads the static JSON configuration and parses it into mathematical objects."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Critical Error: Configuration file '{filepath}' not found.")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"❌ Critical Error: Invalid JSON syntax in '{filepath}'. Details: {e}")
        return {}, {}

    # Parse Algorithms (QASM -> QuantumCircuit)
    algos = {}
    for name, qasm_str in data.get("algorithms", {}).items():
        try:
            algos[name] = qiskit.qasm2.loads(qasm_str)
        except Exception as e:
            print(f"Warning: Failed to compile QASM for '{name}'. {e}")

    # Parse Challenges (Lists -> Complex Numpy Arrays)
    challenges = {}
    for name, chal_data in data.get("challenges", {}).items():
        try:
            # Reconstruct complex tensors from [real, imag] pairs or standard floats
            init_parsed = _parse_complex_array(chal_data["initial_state"])
            targ_parsed = _parse_complex_array(chal_data["target_state"])

            challenges[name] = {
                "num_qubits": chal_data["num_qubits"],
                "initial_state": init_parsed,
                "target_state": targ_parsed
            }
        except Exception as e:
            print(f"Warning: Failed to parse state array for '{name}'. {e}")

    return algos, challenges


def _parse_complex_array(state_list):
    """Safely converts generic JSON arrays into typed Python complex arrays."""
    if not state_list:
        return None
    # If standard 1D array of floats (purely real state)
    if isinstance(state_list[0], (int, float)):
        return [complex(x, 0.0) for x in state_list]
    # If 2D array of [real, imaginary] pairs
    elif isinstance(state_list[0], list):
        return [complex(x[0], x[1]) for x in state_list]
    raise ValueError("Unrecognized array format in JSON.")


# ==========================================
# 2. SPA ENTRY POINT APPLICATION
# ==========================================

class QuantumAppLauncher:
    """
    The Single Page Application (SPA) entry point.
    """

    def __init__(self):
        self.output = widgets.Output()

        # Load native Qiskit objects directly from memory
        self.algos = QuantumCurriculum.get_algorithms()
        self.challenges = QuantumCurriculum.get_challenges()

        self.title = widgets.HTML("<h1 style='text-align: center; color: #2c3e50;'>Quantum Education Suite</h1>")
        self.subtitle = widgets.HTML(
            "<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 20px;'>Select your learning environment</h4>")
        self.header = widgets.VBox([self.title, self.subtitle], layout={'width': '100%'})

        self.tab = widgets.Tab(layout={'width': '500px', 'min_height': '250px'})

        self.tab_sandbox = self._build_sandbox_tab()
        self.tab_algo = self._build_algorithm_tab()
        self.tab_challenge = self._build_challenge_tab()

        self.tab.children = [self.tab_sandbox, self.tab_algo, self.tab_challenge]
        self.tab.titles = ('Sandbox', 'Algorithms', 'Challenges')

        self.menu_container = widgets.VBox(
            [self.header, self.tab],
            layout=widgets.Layout(align_items='center', justify_content='center', width='100%', margin='30px 0px')
        )

    def _build_sandbox_tab(self):
        self.sb_qubits = widgets.Dropdown(options=[1, 2, 3, 4, 5, 6], value=3, description='Qubits:')
        self.sb_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)
        self.sb_states = {"|0...0⟩ (Ground State)": None, "|+...+⟩ (Equal Superposition)": "superposition"}
        self.sb_initial = widgets.Dropdown(options=list(self.sb_states.keys()), value="|0...0⟩ (Ground State)",
                                           description='Initial State:')

        btn = widgets.Button(description="Launch Sandbox", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
        btn.style.button_color = '#3498db';
        btn.style.text_color = 'white';
        btn.style.font_weight = 'bold'
        btn.on_click(self._launch_sandbox)
        return widgets.VBox([self.sb_qubits, self.sb_initial, self.sb_circuit, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _build_algorithm_tab(self):
        options = list(self.algos.keys()) if self.algos else ["No algorithms loaded"]
        self.algo_dropdown = widgets.Dropdown(options=options, description='Algorithm:', layout={'width': '350px'})
        self.algo_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)

        btn = widgets.Button(description="Study Algorithm", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
        btn.style.button_color = '#9b59b6';
        btn.style.text_color = 'white';
        btn.style.font_weight = 'bold'
        btn.disabled = not bool(self.algos)
        btn.on_click(self._launch_algorithm)
        return widgets.VBox([self.algo_dropdown, self.algo_circuit, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _build_challenge_tab(self):
        options = list(self.challenges.keys()) if self.challenges else ["No challenges loaded"]
        self.chal_dropdown = widgets.Dropdown(options=options, description='Challenge:', layout={'width': '400px'})
        self.chal_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)

        btn = widgets.Button(description="Start Challenge", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
        btn.style.button_color = '#e67e22';
        btn.style.text_color = 'white';
        btn.style.font_weight = 'bold'
        btn.disabled = not bool(self.challenges)
        btn.on_click(self._launch_challenge)
        return widgets.VBox([self.chal_dropdown, self.chal_circuit, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _launch_sandbox(self, b):
        num_qubits = self.sb_qubits.value
        state_key = self.sb_initial.value
        initial_state = None
        if self.sb_states[state_key] == "superposition":
            dim = 2 ** num_qubits
            initial_state = [1.0 / np.sqrt(dim)] * dim

        with self.output:
            clear_output(wait=True)
            viewer = InteractiveViewer(num_qubits=num_qubits, initial_state=initial_state,
                                       show_circuit=self.sb_circuit.value)
            viewer.display()

    def _launch_algorithm(self, b):
        qc_algo = self.algos[self.algo_dropdown.value]
        # Native Qiskit object simulation ensures 100% mathematical fidelity
        initial_state = ([1.0] + [0.0] * ((2 ** qc_algo.num_qubits) - 1))
        target_sv = Statevector.from_instruction(qc_algo).data.tolist()

        with self.output:
            clear_output(wait=True)
            viewer = ChallengeViewer(num_qubits=qc_algo.num_qubits, initial_state=initial_state, target_state=target_sv,
                                     preloaded_circuit=qc_algo, show_circuit=self.algo_circuit.value)
            viewer.display()

    def _launch_challenge(self, b):
        chal_data = self.challenges[self.chal_dropdown.value]
        with self.output:
            clear_output(wait=True)
            viewer = ChallengeViewer(num_qubits=chal_data["num_qubits"], initial_state=chal_data["initial_state"],
                                     target_state=chal_data["target_state"], preloaded_circuit=None,
                                     show_circuit=self.chal_circuit.value)
            viewer.display()

    def display(self):
        with self.output:
            clear_output(wait=True)
            display(self.menu_container)
        display(self.output)