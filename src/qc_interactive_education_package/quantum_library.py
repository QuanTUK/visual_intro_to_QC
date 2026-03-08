import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector


class QuantumCurriculum:
    """
    A native Python constructor for the Quantum Viewer.
    """

    @staticmethod
    def annotate(qc, step_index, text):
        """
        Attaches a pedagogical annotation to a specific mathematical step in the timeline.
        Step 0 = Initial State. Step 1 = After the first gate operation, etc.
        """
        if qc.metadata is None:
            qc.metadata = {}
        if 'annotations' not in qc.metadata:
            qc.metadata['annotations'] = {}
        qc.metadata['annotations'][step_index] = text
        return qc

    @staticmethod
    def get_algorithms():
        algos = {}

        # --- Algorithm 1: Bell State ---
        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)

        QuantumCurriculum.annotate(qc_bell, 0,
                                   r"We start in the absolute ground state $|00\rangle$. Both qubits possess deterministic, classical values.")
        QuantumCurriculum.annotate(qc_bell, 1,
                                   r"The **Hadamard** gate (`HGate`) on $q_0$ creates a perfect superposition. The system is now separated into two equal branches: $\frac{|00\rangle + |01\rangle}{\sqrt{2}}$.")
        QuantumCurriculum.annotate(qc_bell, 2,
                                   r"The **CNOT** gate permanently entangles the qubits. Notice how the amplitudes have shifted into the mathematically inseparable Bell State: $| \Phi^+ \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$.")
        algos["Bell State Entanglement"] = qc_bell

        # --- Algorithm 1.5: GHZ State ---
        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)

        QuantumCurriculum.annotate(qc_ghz, 0, r"We begin in the $|000\rangle$ ground state.")
        QuantumCurriculum.annotate(qc_ghz, 1,
                                   r"The **Hadamard** gate places the first qubit into a superposition, yielding $\frac{|000\rangle + |001\rangle}{\sqrt{2}}$.")
        QuantumCurriculum.annotate(qc_ghz, 2,
                                   r"The first **CNOT** entangles $q_0$ and $q_1$. We now have a bipartite entangled pair tensored with a deterministic $q_2$.")
        QuantumCurriculum.annotate(qc_ghz, 3,
                                   r"The second **CNOT** cascades the entanglement to $q_2$. We have successfully generated the maximally entangled tripartite **GHZ State**: $\frac{|000\rangle + |111\rangle}{\sqrt{2}}$.")
        algos["GHZ State Entanglement"] = qc_ghz

        # --- Algorithm 2: Unitary Quantum Teleportation ---

        # We use a standard 3-qubit circuit with no classical registers to ensure
        # perfect compatibility with the ChallengeViewer's Statevector extraction.
        qc_teleport = QuantumCircuit(3)

        # Step 0: Prepare an arbitrary, non-trivial state on Qubit 0
        psi = random_statevector(2)

        # Remove global phase: rotate so the global phase is 0 (first element becomes real). This doesn't change the physics.
        psi = Statevector(np.exp(-1j * np.angle(psi.data[0])) * psi.data)
        full_sv = Statevector.from_label('00').tensor(psi)
        qc_teleport.initialize(full_sv, [0, 1, 2])

        # Step 1: Create Entanglement
        qc_teleport.h(1)
        qc_teleport.cx(1, 2)
        qc_teleport.barrier()

        # Step 2: Alice's local operations (Projecting into the Bell Basis)
        qc_teleport.cx(0, 1)
        qc_teleport.h(0)
        qc_teleport.barrier()

        # Step 3 & 4: Quantum Feedforward (Deferred Measurement Principle)
        # Instead of measuring Qubit 1 to conditionally apply X, we use a CNOT.
        qc_teleport.cx(1, 2)

        # Instead of measuring Qubit 0 to conditionally apply Z, we use a CZ.
        qc_teleport.cz(0, 2)

        # --- Annotations ---

        QuantumCurriculum.annotate(qc_teleport, 0,
                                   "**Initialization:** Qubit 0 holds the unknown state $|\\psi\\rangle$. Qubits 1 and 2 act as the blank computational medium.")

        QuantumCurriculum.annotate(qc_teleport, 1,
                                   "**Entanglement Generation:** A Hadamard and CNOT gate construct the maximally entangled Bell state $|\\Phi^+\\rangle$ between Qubits 1 and 2. This establishes the non-local correlation.")

        QuantumCurriculum.annotate(qc_teleport, 4,
                                   "**Basis Transformation:** Alice interacts her message qubit (Qubit 0) with her half of the entangled pair (Qubit 1) using a CNOT followed by a Hadamard. This operation effectively projects the system into the Bell basis.")

        QuantumCurriculum.annotate(qc_teleport, 7,
                                   "**Quantum Feedforward:** To satisfy the requirements of a unitary Statevector simulator, we utilize the *Deferred Measurement Principle*. Instead of extracting classical bits, Alice's qubits act as direct quantum controls for the corrective Pauli operations on Bob's qubit. The exact state $|\\psi\\rangle$ is successfully reconstructed on Qubit 2.")

        algos["Quantum Teleportation"] = qc_teleport

        # --- Algorithm 3: Quantum Fourier Transform ---
        qc_qft = QuantumCircuit(3)
        qc_qft.h(2)
        qc_qft.cp(np.pi / 2, 1, 2)
        qc_qft.cp(np.pi / 4, 0, 2)
        qc_qft.h(1)
        qc_qft.cp(np.pi / 2, 0, 1)
        qc_qft.h(0)
        qc_qft.swap(0, 2)

        #
        QuantumCurriculum.annotate(qc_qft, 0,
                                   r"We initialize the Quantum Fourier Transform (QFT). The QFT maps the computational basis into the Fourier (phase) basis.")
        QuantumCurriculum.annotate(qc_qft, 1,
                                   r"A **Hadamard** on the Most Significant Bit (MSB, $q_2$) begins the phase fractionalization.")
        QuantumCurriculum.annotate(qc_qft, 2,
                                   r"A Controlled-Phase gate ($\pi/2$) applies a fractional kickback conditional on $q_1$.")
        QuantumCurriculum.annotate(qc_qft, 3,
                                   r"A Controlled-Phase gate ($\pi/4$) applies a finer fractional kickback conditional on $q_0$. The MSB is now fully encoded in the Fourier basis.")
        QuantumCurriculum.annotate(qc_qft, 7,
                                   r"Finally, the **SWAP** gate reverses the qubit ordering to mathematically align the output with standard Qiskit little-endian notation.")
        algos["Quantum Fourier Transform (3Q)"] = qc_qft

        # ==========================================
        # GROVER'S SEARCH DYNAMIC BUILDER
        # ==========================================
        def build_grover(target_bitstring):
            n = len(target_bitstring)
            qc = QuantumCircuit(n)

            QuantumCurriculum.annotate(qc, 0, r"Grover's Search begins in the absolute ground state $|0...0\rangle$.")

            qc.h(range(n))
            # Broadcasting H to n qubits adds n instructions.
            QuantumCurriculum.annotate(qc, n,
                                       r"We apply **Hadamard** gates to all qubits, spreading the amplitude evenly across all $2^n$ basis states to create a uniform superposition.")

            oracle = QuantumCircuit(n, name="Oracle")
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0':
                    oracle.x(i)
            oracle.h(n - 1)
            oracle.mcx(list(range(n - 1)), n - 1)
            oracle.h(n - 1)
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0':
                    oracle.x(i)

            oracle_gate = oracle.to_gate(label="Oracle")

            diffuser = QuantumCircuit(n, name="Diffuser")
            diffuser.h(range(n))
            diffuser.x(range(n))
            diffuser.h(n - 1)
            diffuser.mcx(list(range(n - 1)), n - 1)
            diffuser.h(n - 1)
            diffuser.x(range(n))
            diffuser.h(range(n))

            diffuser_gate = diffuser.to_gate(label="Diffuser")

            optimal_iterations = int(np.floor((np.pi / 4.0) * np.sqrt(2 ** n)))

            step_counter = n + 1
            for i in range(optimal_iterations):
                qc.append(oracle_gate, range(n))
                QuantumCurriculum.annotate(qc, step_counter,
                                           rf"Iteration {i + 1}: The **Oracle** isolates the target string $|{target_bitstring}\rangle$ and applies a geometric reflection, flipping its phase (amplitude) to negative.")
                step_counter += 1

                qc.append(diffuser_gate, range(n))
                QuantumCurriculum.annotate(qc, step_counter,
                                           rf"Iteration {i + 1}: The **Diffuser** performs an inversion about the mean. Notice how this geometrically drains the amplitude from the uniform states and physically amplifies the target state.")
                step_counter += 1

            return qc

        algos["Grover's Search: Target |1011⟩ (4Q)"] = build_grover("1011")
        algos["Grover's Search: Target |10101⟩ (5Q)"] = build_grover("10101")
        #

        # --- Algorithm 6: 3-Qubit Bit-Flip Error Correction ---
        qc_3q_err = QuantumCircuit(3)
        QuantumCurriculum.annotate(qc_3q_err, 0,
                                   r"We start in the $|000\rangle$ ground state. The objective is to encode a single logical qubit across three physical qubits to protect against a bit-flip error.")
        qc_3q_err.x(0)
        QuantumCurriculum.annotate(qc_3q_err, 1,
                                   r"We initialize $q_0$ to the state $|1\rangle$. This is the raw logical state we wish to protect.")
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        QuantumCurriculum.annotate(qc_3q_err, 3,
                                   r"Encoding complete. By cascading CNOT gates, we have mapped the logical state into the physical repetition code $|111\rangle$.")
        qc_3q_err.x(0)
        QuantumCurriculum.annotate(qc_3q_err, 4,
                                   r"⚠️ **ERROR INJECTED:** A quantum noise event (X-gate) strikes $q_0$, flipping it back to $|0\rangle$. The system is now corrupted into the state $|011\rangle$.")
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        QuantumCurriculum.annotate(qc_3q_err, 6,
                                   r"Syndrome Measurement: We compute the parity of the qubits using CNOTs. This maps the error syndrome into the ancilla space without collapsing the logical superposition.")
        qc_3q_err.ccx(1, 2, 0)
        QuantumCurriculum.annotate(qc_3q_err, 7,
                                   r"Correction: The Toffoli (CCX) gate acts as an autonomous classical logic switch, flipping $q_0$ back to its correct state exclusively if the syndrome flags an error. The state is restored to $|111\rangle$.")
        algos["Error Correction: 3-Qubit Bit-Flip"] = qc_3q_err

        # --- Algorithm 7: 7-Qubit Steane Code (Logical |0>) ---
        #
        qc_steane = QuantumCircuit(7)
        QuantumCurriculum.annotate(qc_steane, 0,
                                   r"We initialize the Steane [[7,1,3]] Error Correction code. This CSS code can simultaneously correct both bit-flip (X) and phase-flip (Z) errors.")
        qc_steane.h([0, 1, 2])
        QuantumCurriculum.annotate(qc_steane, 3,
                                   r"Hadamard gates prepare the three data qubits into a uniform superposition.")
        qc_steane.cx(0, 3);
        qc_steane.cx(1, 3);
        qc_steane.cx(0, 4)
        qc_steane.cx(2, 4);
        qc_steane.cx(1, 5);
        qc_steane.cx(2, 5)
        qc_steane.cx(0, 6);
        qc_steane.cx(1, 6);
        qc_steane.cx(2, 6)

        # 3 H-gates + 9 CNOTs = 12 timeline steps
        QuantumCurriculum.annotate(qc_steane, 12,
                                   r"A highly specific array of parity operations encodes the logical $|0\rangle_L$ state into the 7 physical qubits using the classical Hamming code topology.")
        algos["Error Correction: Steane [[7,1,3]] Code"] = qc_steane

        # --- Algorithm 8: Shor's Period Finding (a=2, N=3) ---
        #
        qc_shor = QuantumCircuit(4)
        QuantumCurriculum.annotate(qc_shor, 0,
                                   r"We initialize a 4-qubit Shor's algorithm. Qubits 0-1 serve as the counting register, and Qubits 2-3 are the auxiliary work register.")
        qc_shor.h([0, 1])
        qc_shor.x(3)
        QuantumCurriculum.annotate(qc_shor, 3,
                                   r"We prepare the counting register in a uniform superposition and excite the auxiliary register to $|1\rangle$ (the multiplicative identity).")
        qc_shor.cswap(0, 2, 3)
        qc_shor.swap(0, 1)
        qc_shor.h(1)
        qc_shor.cp(-1 * np.pi / 2, 0, 1)
        qc_shor.h(0)
        QuantumCurriculum.annotate(qc_shor, 8,
                                   r"After executing the modular exponentiation and the Inverse QFT, the counting register collapses into the periodic phase shift, revealing the mathematical period of the function.")
        algos["Shor's Algorithm: Period Finding"] = qc_shor

        # ==========================================
        # SHOR'S PERIOD FINDING (6Q, a=5, N=6)
        # ==========================================
        qc_shor_6q = QuantumCircuit(6)
        n_count = 3

        prep = QuantumCircuit(6, name="Initialize")
        for q in range(n_count):
            prep.h(q)
        prep.x(n_count)

        qc_shor_6q.append(prep.to_gate(label="Initialization"), range(6))
        QuantumCurriculum.annotate(qc_shor_6q, 0,
                                   r"We initialize the counting register in a uniform superposition and set the auxiliary work register to the multiplicative identity $|1\rangle$ (Decimal 1).")

        from qiskit.circuit.library import UnitaryGate
        U_matrix = np.eye(8)
        U_matrix[1, 1] = 0;
        U_matrix[1, 5] = 1
        U_matrix[5, 5] = 0;
        U_matrix[5, 1] = 1
        U_matrix[2, 2] = 0;
        U_matrix[2, 4] = 1
        U_matrix[4, 4] = 0;
        U_matrix[4, 2] = 1

        U_gate = UnitaryGate(U_matrix, label="5^1 mod 6").control(1)
        U_identity = UnitaryGate(np.eye(8), label="5^2 mod 6 (Identity)").control(1)
        U_identity2 = UnitaryGate(np.eye(8), label="5^4 mod 6 (Identity)").control(1)

        qc_shor_6q.append(U_gate, [0, 3, 4, 5])
        QuantumCurriculum.annotate(qc_shor_6q, 2,
                                   r"We apply $5^1 \bmod 6$. This unitary rotation mathematically shifts the amplitudes and accumulates a phase in the counting register via Phase Kickback.")

        qc_shor_6q.append(U_identity, [1, 3, 4, 5])
        QuantumCurriculum.annotate(qc_shor_6q, 3,
                                   r"We apply $5^2 \bmod 6$. Because $5^2 = 25 \equiv 1 \bmod 6$, this operator trivially collapses to the Identity matrix. No amplitude shift occurs.")

        qc_shor_6q.append(U_identity2, [2, 3, 4, 5])
        QuantumCurriculum.annotate(qc_shor_6q, 4,
                                   r"We apply $5^4 \bmod 6$. This also evaluates to Identity. The early saturation of these higher-order exponentiations explicitly proves the period is $r=2$.")

        def qft_dagger(n):
            qc = QuantumCircuit(n, name="QFT†")
            for qubit in range(n // 2):
                qc.swap(qubit, n - qubit - 1)
            for j in range(n):
                for m in range(j):
                    qc.cp(-np.pi / float(2 ** (j - m)), m, j)
                qc.h(j)
            return qc.to_gate(label="Inverse QFT")

        qc_shor_6q.append(qft_dagger(n_count), range(n_count))
        QuantumCurriculum.annotate(qc_shor_6q, 5,
                                   r"The **Inverse Quantum Fourier Transform** perfectly resolves the accumulated kickback phases, collapsing the counting register exclusively into the fraction periods $0$ and $4$. ($4/8 = 1/2$, yielding period $r=2$).")

        algos["Shor's Algorithm: Period Finding (a=5, N=6)"] = qc_shor_6q

        # ==========================================
        # SHOR'S ALGORITHM (8Q) - SEQUENTIAL EXECUTION
        # ==========================================
        qc_shor_8q = QuantumCircuit(8)
        n_count = 4
        a = 7

        prep = QuantumCircuit(8, name="Initialize")
        for q in range(n_count):
            prep.h(q)
        prep.x(n_count)

        qc_shor_8q.append(prep.to_gate(label="Initialization"), range(8))
        QuantumCurriculum.annotate(qc_shor_8q, 0,
                                   r"We initialize the 8Q Shor algorithm. The 4-qubit counting register is placed into uniform superposition, and the auxiliary register is excited to $|1\rangle$.")

        def c_amod15(base, power):
            if base not in [2, 4, 7, 8, 11, 13]:
                raise ValueError("'base' must be 2,4,7,8,11 or 13")
            U = QuantumCircuit(4)
            for _ in range(power):
                if base in [2, 13]:
                    U.swap(0, 1);
                    U.swap(1, 2);
                    U.swap(2, 3)
                if base in [7, 8]:
                    U.swap(2, 3);
                    U.swap(1, 2);
                    U.swap(0, 1)
                if base in [4, 11]:
                    U.swap(1, 3);
                    U.swap(0, 2)
                if base in [7, 11, 13]:
                    for q in range(4):
                        U.x(q)
            U_gate = U.to_gate()
            U_gate.name = f"{base}^{power} mod 15"
            return U_gate.control(1)

        for q in range(n_count):
            qc_shor_8q.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])

        QuantumCurriculum.annotate(qc_shor_8q, 5,
                                   r"We have completed the sequential modular exponentiation. Notice the highly complex entanglement topology mapping between the counting and work registers.")

        qc_shor_8q.append(qft_dagger(n_count), range(n_count))
        QuantumCurriculum.annotate(qc_shor_8q, 6,
                                   r"The **Inverse QFT** isolates the periodic interference patterns. Measuring the counting register now yields one of the exact eigenvalues required to extract the period classically.")

        algos["Shor's Algorithm: Factor 15 (8Q)"] = qc_shor_8q

        return algos

    @staticmethod
    def get_challenges():
        challenges = {}
        inv_sq2 = 1.0 / np.sqrt(2)

        def make_hint(num_qubits, text):
            qc_hint = QuantumCircuit(num_qubits)
            QuantumCurriculum.annotate(qc_hint, 0, text)
            return qc_hint

        challenges["Level 1: Create a Superposition (|+⟩)"] = {
            "num_qubits": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [inv_sq2, inv_sq2],
            "preloaded_circuit": make_hint(1,
                                           r"To map a deterministic classical state into a uniform superposition, apply a **Hadamard** (`H`) gate."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 1
        }

        challenges["Level 2: Phase Flip (|1⟩ to |-⟩)"] = {
            "num_qubits": 1,
            "initial_state": [0.0, 1.0],
            "target_state": [inv_sq2, -inv_sq2],
            "preloaded_circuit": make_hint(1,
                                           r"You are starting in the $|1\rangle$ state. Applying a Hadamard gate to $|1\rangle$ will yield the targeted negative phase superposition automatically."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 1
        }

        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        challenges["Level 3: Construct a Bell State"] = {
            "num_qubits": 2,
            "initial_state": [1.0, 0.0, 0.0, 0.0],
            "target_state": Statevector.from_instruction(qc_bell).data.tolist(),
            "preloaded_circuit": make_hint(2,
                                           r"Entanglement requires two steps: First, create a superposition on $q_0$ using an `H` gate. Then, correlate $q_1$ to $q_0$ using a Controlled-NOT (`CX`) gate."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 2
        }

        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        challenges["Level 4: Construct a GHZ state"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": Statevector.from_instruction(qc_ghz).data.tolist(),
            "preloaded_circuit": make_hint(3,
                                           r"To build a tripartite GHZ state, first construct a standard Bell State between $q_0$ and $q_1$, and then cascade a second `CX` gate from $q_1$ to $q_2$."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 3
        }

        challenges["Level 5: Quantum Teleportation"] = {
            "num_qubits": 3,
            "initial_state": [inv_sq2, -inv_sq2, 0.0, 0.0, 0.0, 0.0, inv_sq2, -inv_sq2],
            "target_state": [inv_sq2, -inv_sq2, inv_sq2, -inv_sq2, -inv_sq2, inv_sq2, inv_sq2, -inv_sq2],
            "preloaded_circuit": make_hint(3,
                                           r"You must bind Bob ($q_2$) to Alice ($q_1$) using a Bell State, then mathematically fuse Alice's random state ($q_0$) into the Bell pair using `CX` and `H` gates. The measurement, in the end, would then yield a state that can be correct to the initial state of $q_0$."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 6  # Mathematical Minimum: H, CX, CX, H, CX, CZ
        }

        challenges["Level 6: Search Challenge: Amplify |101⟩"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": [0.0, 0.0, 0.0, 0.0, 0.0, inv_sq2, 0.0, inv_sq2],
            "preloaded_circuit": make_hint(3,
                                           r"To amplify $|101\rangle$, you must construct an Oracle that isolates the target using $X$-gates on the zero-bits, and phase-flips it using a Multi-Controlled Z-gate. (For a 3-qubit challenge, standard `H` gates and `CCX` work perfectly)."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 26
        }

        # --- Phase-Flip Error Correction ---
        qc_phase_target = QuantumCircuit(3)
        qc_phase_target.x(0)
        qc_phase_target.cx(0, 1)
        qc_phase_target.cx(0, 2)
        qc_phase_target.h([0, 1, 2])

        qc_phase_init = qc_phase_target.copy()
        qc_phase_init.z(0)

        challenges["Level 7: Correct a Phase-Flip Error"] = {
            "num_qubits": 3,
            "initial_state": Statevector.from_instruction(qc_phase_init).data.tolist(),
            "target_state": Statevector.from_instruction(qc_phase_target).data.tolist(),
            "preloaded_circuit": make_hint(3,
                                           r"The system is currently corrupted by a $Z$-error. Because Phase ($Z$) errors become Bit ($X$) errors when rotated into the Hadamard basis, you must sandwich your standard Repetition Code syndrome measurement (`CX`, `CCX`) between two arrays of `H` gates."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 9
        }

        return challenges