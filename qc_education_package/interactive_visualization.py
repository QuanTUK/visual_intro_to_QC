import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, RXGate, RYGate, RZGate
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import traceback
import numpy as np
import base64

from .visualization import DimensionalCircleNotation


class InteractiveDCNViewer:
    """
    An encapsulated interactive UI for dynamically visualizing quantum circuits.
    Features parametric controls, state vector normalization, global phase alignment,
    projective measurements, deterministic state-snapshotting undo mechanism,
    and a live Dirac notation readout with array export capabilities.
    """

    def __init__(self, num_qubits=3, initial_state=None):
        self.num_qubits = num_qubits
        self.render_figsize = (8.0, 6.0)

        self.initial_state = self._normalize_state(initial_state) if initial_state is not None else None
        self._circuit_history = []
        self._init_circuit()

        # --- UI Components ---
        self.gate_dropdown = widgets.Dropdown(options=['H', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'], value='H',
                                              description='Gate:', layout={'width': '180px'})
        self.controlled_checkbox = widgets.Checkbox(value=False, description='Controlled', indent=False,
                                                    disabled=(self.num_qubits < 2),
                                                    layout={'width': '100px', 'margin': '0px 10px 0px 10px'})

        qubit_options = list(range(1, self.num_qubits + 1))
        self.target_selector = widgets.SelectMultiple(options=qubit_options, value=(1,), description='Target(s):',
                                                      rows=min(4, self.num_qubits), layout={'width': '160px'})
        self.control_selector = widgets.SelectMultiple(options=qubit_options,
                                                       value=(2,) if self.num_qubits > 1 else tuple(),
                                                       description='Control(s):', disabled=True,
                                                       rows=min(4, self.num_qubits), layout={'width': '160px'})
        self.angle_input = widgets.FloatSlider(value=0.5, min=-2.0, max=2.0, step=0.0625, description='Angle (× π):',
                                               disabled=True, layout={'width': '250px'}, readout_format='.3f')

        # --- Buttons (Icons Removed, Spacing Corrected) ---
        self.apply_btn = widgets.Button(description="Apply",
                                        layout=widgets.Layout(width='85px', height='32px', border='1px solid #2b5797',
                                                              border_radius='4px'))
        self.apply_btn.style.button_color = '#2d89ef'
        self.apply_btn.style.text_color = 'white'
        self.apply_btn.style.font_weight = 'bold'

        self.measure_btn = widgets.Button(description="Measure",
                                          layout=widgets.Layout(width='95px', height='32px', border='1px solid #8e44ad',
                                                                border_radius='4px'))
        self.measure_btn.style.button_color = '#9b59b6'
        self.measure_btn.style.text_color = 'white'
        self.measure_btn.style.font_weight = 'bold'

        self.zero_phase_btn = widgets.Button(description="0-Phase",
                                             layout=widgets.Layout(width='90px', height='32px',
                                                                   border='1px solid #d37c15', border_radius='4px'))
        self.zero_phase_btn.style.button_color = '#f39c12'
        self.zero_phase_btn.style.text_color = 'white'
        self.zero_phase_btn.style.font_weight = 'bold'

        self.undo_btn = widgets.Button(description="Undo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.undo_btn.style.button_color = '#95a5a6'
        self.undo_btn.style.text_color = 'white'
        self.undo_btn.style.font_weight = 'bold'

        self.reset_btn = widgets.Button(description="Reset",
                                        layout=widgets.Layout(width='80px', height='32px', border='1px solid #b91d47',
                                                              border_radius='4px'))
        self.reset_btn.style.button_color = '#ee1111'
        self.reset_btn.style.text_color = 'white'
        self.reset_btn.style.font_weight = 'bold'

        # --- State Inspector & Array Extraction ---
        self.state_inspector = widgets.HTML(layout={'width': '80%', 'margin': '10px 0px 10px 0px'})

        self.show_array_btn = widgets.Button(description="Raw Array",
                                             layout=widgets.Layout(width='110px', height='32px',
                                                                   border='1px solid #2c3e50',
                                                                   border_radius='4px'))
        self.show_array_btn.style.button_color = '#34495e'
        self.show_array_btn.style.text_color = 'white'
        self.show_array_btn.style.font_weight = 'bold'

        # Combine the Dirac readout and the extraction button into a single horizontal band
        self.inspector_row = widgets.HBox([self.state_inspector, self.show_array_btn],
                                          layout={'width': '100%', 'align_items': 'center',
                                                  'justify_content': 'space-around'})

        self.image_widget = widgets.Image(format='png', layout={'min_height': '400px', 'max_width': '100%'})
        self.console = widgets.Output(layout={'border': '1px solid red', 'width': '100%'})

        # --- Event Binding ---
        self.gate_dropdown.observe(self._toggle_angle_slider, names='value')
        self.controlled_checkbox.observe(self._toggle_control_selector, names='value')

        self.apply_btn.on_click(self._apply_gate)
        self.measure_btn.on_click(self._measure_qubits)
        self.zero_phase_btn.on_click(self._zero_global_phase)
        self.undo_btn.on_click(self._undo_action)
        self.reset_btn.on_click(self._reset_circuit)
        self.show_array_btn.on_click(self._show_state_array)

        # --- Layout ---
        controls_top = widgets.HBox(
            [self.gate_dropdown, self.controlled_checkbox, self.control_selector, self.target_selector],
            layout={'align_items': 'center'})

        controls_bottom = widgets.HBox(
            [self.angle_input, self.apply_btn, self.measure_btn, self.zero_phase_btn, self.undo_btn, self.reset_btn])

        self.ui = widgets.VBox([controls_top, controls_bottom, self.inspector_row, self.image_widget, self.console],
                               layout={'align_items': 'center'})

        self._update_plot()

    def _normalize_state(self, statevector):
        sv_array = np.array(statevector, dtype=complex)
        norm = np.linalg.norm(sv_array)
        if np.isclose(norm, 0.0): raise ValueError(
            "A null vector cannot be normalized to represent a physical quantum state.")
        return (sv_array / norm).tolist()

    def _init_circuit(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        self._circuit_history.clear()
        if self.initial_state is not None:
            self.circuit.initialize(self.initial_state, self.circuit.qubits)

    def set_initial_state(self, statevector):
        with self.console:
            self.console.clear_output()
            try:
                self.initial_state = self._normalize_state(statevector)
                self._init_circuit()
                self._update_plot()
            except Exception as e:
                print(f"Initialization Error: {type(e).__name__}: {str(e)}")

    def _toggle_angle_slider(self, change):
        self.angle_input.disabled = (change.new not in ['Rx', 'Ry', 'Rz'])

    def _toggle_control_selector(self, change):
        self.control_selector.disabled = not change.new

    def _apply_gate(self, b):
        gate_str = self.gate_dropdown.value
        is_controlled = self.controlled_checkbox.value
        angle_radians = self.angle_input.value * np.pi

        targets = [t - 1 for t in self.target_selector.value]
        controls = [c - 1 for c in self.control_selector.value] if is_controlled else []

        with self.console:
            self.console.clear_output()
            if not targets:
                print("Validation Error: At least one Target qubit must be selected.")
                return
            if is_controlled and not controls:
                print("Validation Error: At least one Control qubit must be selected when 'Controlled' is active.")
                return
            if set(targets).intersection(controls):
                print(
                    f"Validation Error: Intersection detected. Qubits {set(targets).intersection(controls)} cannot serve as both control and target.")
                return

        self._circuit_history.append(self.circuit.copy())

        if gate_str == 'H':
            base_gate = HGate()
        elif gate_str == 'X':
            base_gate = XGate()
        elif gate_str == 'Y':
            base_gate = YGate()
        elif gate_str == 'Z':
            base_gate = ZGate()
        elif gate_str == 'Rx':
            base_gate = RXGate(angle_radians)
        elif gate_str == 'Ry':
            base_gate = RYGate(angle_radians)
        elif gate_str == 'Rz':
            base_gate = RZGate(angle_radians)

        if is_controlled:
            controlled_gate = base_gate.control(len(controls))
            for t in targets: self.circuit.append(controlled_gate, controls + [t])
        else:
            for t in targets: self.circuit.append(base_gate, [t])

        self._update_plot()

    def _measure_qubits(self, b):
        targets = [t - 1 for t in self.target_selector.value]

        with self.console:
            self.console.clear_output()
            if not targets:
                print("Validation Error: Please select at least one Target qubit to measure.")
                return

        self._circuit_history.append(self.circuit.copy())

        try:
            sv_current = Statevector.from_instruction(self.circuit)
            outcome_str, sv_collapsed = sv_current.measure(targets)

            self.circuit = QuantumCircuit(self.num_qubits)
            self.circuit.initialize(sv_collapsed.data, self.circuit.qubits)

            results = []
            for t, bit in zip(targets, reversed(outcome_str)):
                results.append(f"Qubit {t + 1}: {bit}")

            self._update_plot()

            with self.console:
                print(f"💥 Measurement Result: {', '.join(results)}")

        except Exception as e:
            self._circuit_history.pop()
            with self.console:
                print(f"Measurement Error: {type(e).__name__}: {str(e)}")

    def _zero_global_phase(self, b):
        with self.console:
            try:
                sv_data = Statevector.from_instruction(self.circuit).data
                self._circuit_history.append(self.circuit.copy())
                for amplitude in sv_data:
                    if np.abs(amplitude) > 1e-6:
                        self.circuit.global_phase -= np.angle(amplitude)
                        break
                self._update_plot()
            except Exception as e:
                self._circuit_history.pop()
                self.console.clear_output()
                print(f"Phase Calculation Error: {type(e).__name__}: {str(e)}")

    def _undo_action(self, b):
        if self._circuit_history:
            self.circuit = self._circuit_history.pop()
            self._update_plot()
            with self.console:
                self.console.clear_output()
                print("Reverted to previous state.")
        else:
            with self.console:
                self.console.clear_output()
                print("Undo Stack Empty: You are at the initial circuit state.")

    def _reset_circuit(self, b):
        try:
            self._init_circuit()
            self._update_plot()
            self.console.clear_output()
        except Exception as e:
            with self.console:
                print(f"Reset Error: {str(e)}")

    def _show_state_array(self, b):
        """
        Extracts the statevector and prints it as a standard Python list of complex numbers
        to the console output widget, ensuring safe, environment-agnostic copy-pasting.
        """
        with self.console:
            self.console.clear_output()
            try:
                sv_data = Statevector.from_instruction(self.circuit).data
                # Round to 6 decimal places to prevent floating point drift representations
                formatted_list = [complex(round(c.real, 6), round(c.imag, 6)) for c in sv_data]

                print("📋 Raw Statevector Array (Copy the list below):")
                print(repr(formatted_list))
            except Exception as e:
                print(f"Array Extraction Error: {type(e).__name__}: {str(e)}")

    def _format_dirac_notation(self, sv_data):
        terms = []
        for index, amplitude in enumerate(sv_data):
            if np.abs(amplitude) > 1e-5:
                real = np.real(amplitude)
                imag = np.imag(amplitude)

                if np.abs(imag) < 1e-5:
                    amp_str = f"{real:.3f}"
                elif np.abs(real) < 1e-5:
                    amp_str = f"{imag:.3f}i"
                else:
                    sign = "+" if imag > 0 else "-"
                    amp_str = f"({real:.3f} {sign} {np.abs(imag):.3f}i)"

                basis_state = f"|{index:0{self.num_qubits}b}⟩"
                terms.append(f"{amp_str}{basis_state}")

        equation = " + ".join(terms).replace("+ -", "- ")
        return f"<div style='text-align: center; font-family: monospace; font-size: 16px; font-weight: bold;'>|ψ⟩ = {equation}</div>"

    def _update_plot(self):
        with self.console:
            try:
                sv_current = Statevector.from_instruction(self.circuit)
                self.state_inspector.value = self._format_dirac_notation(sv_current.data)

                vis = DimensionalCircleNotation.from_qiskit(self.circuit)
                with plt.rc_context({'figure.figsize': self.render_figsize}):
                    b64_str = vis.exportBase64(formatStr='png')
                png_bytes = base64.b64decode(b64_str)
                self.image_widget.value = png_bytes
            except Exception as e:
                print("An error occurred during visualization generation:")
                traceback.print_exc()

    def display(self, figsize=None, ui_width=None):
        if figsize is not None:
            self.render_figsize = figsize
            self._update_plot()
        if ui_width is not None:
            self.image_widget.layout.width = ui_width
        display(self.ui)


class ChallengeDCNViewer(InteractiveDCNViewer):
    """
    An assessment-driven subclass of InteractiveDCNViewer.
    Evaluates the current state against a defined target state, automatically
    calculating fidelity to verify completion up to a global phase.
    """

    def __init__(self, num_qubits, initial_state, target_state):
        self.status_banner = widgets.HTML("<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>")

        shared_layout = widgets.Layout(
            min_height='320px',
            width='100%',
            object_fit='contain',
            justify_content='center'
        )

        self.target_image_widget = widgets.Image(format='png', layout=shared_layout)
        self._raw_target_state = target_state

        super().__init__(num_qubits=num_qubits, initial_state=initial_state)

        self.image_widget.layout = shared_layout
        self.render_figsize = (5.0, 4.0)

        self.target_state = self._normalize_state(self._raw_target_state)
        self._render_target()
        self._update_plot()

        controls_top = self.ui.children[0]
        controls_bottom = self.ui.children[1]
        inspector_row = self.ui.children[2]  # References the new combined HBox
        console = self.ui.children[4]

        comparison_box = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3 style='text-align: center; color: #555; margin-bottom: 0px;'>Current State</h3>"),
                self.image_widget
            ], layout={'align_items': 'center', 'width': '50%'}),

            widgets.VBox([
                widgets.HTML("<h3 style='text-align: center; color: #555; margin-bottom: 0px;'>Target State</h3>"),
                self.target_image_widget
            ], layout={'align_items': 'center', 'width': '50%'})
        ], layout={'width': '100%', 'justify_content': 'space-around', 'align_items': 'flex-start'})

        self.ui = widgets.VBox([
            self.status_banner,
            controls_top,
            controls_bottom,
            inspector_row,
            comparison_box,
            console
        ], layout={'align_items': 'center', 'width': '100%'})

        self._check_success()

    def _render_target(self):
        qc_target = QuantumCircuit(self.num_qubits)
        qc_target.initialize(self.target_state, qc_target.qubits)

        try:
            vis = DimensionalCircleNotation.from_qiskit(qc_target)
            with plt.rc_context({'figure.figsize': self.render_figsize}):
                b64_str = vis.exportBase64(formatStr='png')
            self.target_image_widget.value = base64.b64decode(b64_str)
        except Exception as e:
            with self.console:
                print(f"Target Render Error: {type(e).__name__}: {str(e)}")

    def _update_plot(self):
        super()._update_plot()

        if hasattr(self, 'status_banner'):
            self._check_success()

    def _check_success(self):
        try:
            sv_current = Statevector.from_instruction(self.circuit)
            sv_target = Statevector(self.target_state)

            fidelity = state_fidelity(sv_current, sv_target)

            if np.isclose(fidelity, 1.0, atol=1e-5):
                self.status_banner.value = "<h2 style='text-align: center; color: #27ae60;'>Status: Challenge Completed! 🎉</h2>"
            else:
                self.status_banner.value = "<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>"
        except Exception:
            pass