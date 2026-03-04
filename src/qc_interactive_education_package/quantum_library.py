import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, grover_operator
from qiskit.quantum_info import Statevector


class QuantumCurriculum:
    """
    A native Python constructor for the quantum education suite.
    Bypasses serialization to preserve modern Qiskit architectures,
    including advanced phase gates and custom instructions.
    """

    @staticmethod
    def get_algorithms():
        algos = {}

        # --- Algorithm 1: Bell State ---
        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        algos["Bell State Entanglement"] = qc_bell

        # --- Algorithm 2: GHZ State ---
        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        algos["GHZ State Entanglement"] = qc_ghz

        # --- Algorithm 3: Quantum Fourier Transform ---
        qc_qft = QuantumCircuit(3)
        qc_qft.append(QFTGate(3), [0, 1, 2])
        # .decompose() unpacks it one level, revealing the pure p and cp gates
        algos["Quantum Fourier Transform (3Q)"] = qc_qft.decompose()

        # --- Algorithm 4: Grover's Search (|101>) ---
        qc_grover = QuantumCircuit(3)
        qc_grover.h([0, 1, 2])
        oracle = QuantumCircuit(3)
        oracle.cz(0, 2)
        qc_grover = qc_grover.compose(grover_operator(oracle))
        algos["Grover's Search (Target |101>)"] = qc_grover

        return algos

    @staticmethod
    def get_challenges():
        challenges = {}

        # Pre-calculate the irrational amplitude for absolute precision
        inv_sq2 = 1.0 / np.sqrt(2)

        # --- Level 1 & 2: Single Qubit Operations ---
        challenges["Level 1: Create a Superposition (|+⟩)"] = {
            "num_qubits": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [inv_sq2, inv_sq2]
        }

        challenges["Level 2: Phase Flip (|1⟩ to |-⟩)"] = {
            "num_qubits": 1,
            "initial_state": [0.0, 1.0],
            "target_state": [inv_sq2, -inv_sq2]
        }

        # --- Level 3: Bell State (Dynamically Derived) ---
        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        challenges["Level 3: Construct the Bell State"] = {
            "num_qubits": 2,
            "initial_state": [1.0, 0.0, 0.0, 0.0],
            "target_state": Statevector.from_instruction(qc_bell).data.tolist()
        }

        # --- NEW - Level 4: GHZ State (Dynamically Derived) ---
        # Automatically simulates the mathematical target to eliminate human error
        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        challenges["Level 4: Construct a GHZ state"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,  # Clean generation of ground state
            "target_state": Statevector.from_instruction(qc_ghz).data.tolist()
        }

        # --- NEW - Level 5: Quantum Teleportation ---
        # Translated from your provided nested JSON arrays to clean 1D Python lists
        challenges["Level 5: Quantum Teleportation"] = {
            "num_qubits": 3,
            "initial_state": [inv_sq2, -inv_sq2, 0.0, 0.0, 0.0, 0.0, inv_sq2, -inv_sq2],
            "target_state": [inv_sq2, -inv_sq2, inv_sq2, -inv_sq2, -inv_sq2, inv_sq2, inv_sq2, -inv_sq2]
        }

        # --- NEW - Search Challenge: Amplify |101⟩ ---
        # Floating point artifacts (-0.0) from the JSON have been mathematically zeroed.
        challenges["Search Challenge: Amplify |101⟩"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": [0.0, 0.0, 0.0, 0.0, 0.0, inv_sq2, 0.0, inv_sq2]
        }

        return challenges