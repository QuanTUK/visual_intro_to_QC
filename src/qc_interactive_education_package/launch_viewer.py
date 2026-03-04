import sys
import subprocess
import os
import json


def launch_app(num_qubits=3, initial_state=None, show_circuit=True, preloaded_circuit=None):
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