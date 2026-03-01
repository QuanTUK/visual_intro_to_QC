import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, PhaseGate, RXGate, RYGate, RZGate
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import traceback
import numpy as np
import base64
from io import BytesIO
import os
from PIL import Image

from .visualization import DimensionalCircleNotation


class InteractiveDCNViewer:
    """
    An encapsulated interactive UI for dynamically visualizing quantum circuits.
    Features parametric controls, state vector normalization, global phase alignment,
    projective measurements, deterministic state-snapshotting undo/redo mechanism,
    a live Dirac notation readout with discrete exports, and an optional circuit diagram.
    """

    def __init__(self, num_qubits=3, initial_state=None, preloaded_circuit=None):
        self.num_qubits = num_qubits
        self.show_circuit = False
        self.render_figsize = (8.0, 6.0)

        # Initialize primary and secondary (redo) tracking arrays
        self._circuit_history = []
        self._action_history = []
        self._redo_circuit_history = []
        self._redo_action_history = []

        self.initial_state = self._normalize_state(initial_state) if initial_state is not None else None
        self._init_circuit()

        # --- UI Components (Controls) ---
        self.gate_dropdown = widgets.Dropdown(options=['H', 'X', 'Y', 'Z', 'P', 'Rx', 'Ry', 'Rz'], value='H',
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

        # --- Base Control Buttons ---
        self.apply_btn = widgets.Button(description="Apply",
                                        layout=widgets.Layout(width='85px', height='32px', border='1px solid #2b5797',
                                                              border_radius='4px'))
        self.apply_btn.style.button_color = '#2d89ef'; self.apply_btn.style.text_color = 'white'; self.apply_btn.style.font_weight = 'bold'

        self.measure_btn = widgets.Button(description="Measure",
                                          layout=widgets.Layout(width='95px', height='32px', border='1px solid #8e44ad',
                                                                border_radius='4px'))
        self.measure_btn.style.button_color = '#9b59b6'; self.measure_btn.style.text_color = 'white'; self.measure_btn.style.font_weight = 'bold'

        self.zero_phase_btn = widgets.Button(description="0-Phase", layout=widgets.Layout(width='90px', height='32px',
                                                                                          border='1px solid #d37c15',
                                                                                          border_radius='4px'))
        self.zero_phase_btn.style.button_color = '#f39c12'; self.zero_phase_btn.style.text_color = 'white'; self.zero_phase_btn.style.font_weight = 'bold'

        self.undo_btn = widgets.Button(description="Undo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.undo_btn.style.button_color = '#95a5a6'; self.undo_btn.style.text_color = 'white'; self.undo_btn.style.font_weight = 'bold'

        # NEW: Redo Button
        self.redo_btn = widgets.Button(description="Redo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.redo_btn.style.button_color = '#95a5a6'; self.redo_btn.style.text_color = 'white'; self.redo_btn.style.font_weight = 'bold'

        self.reset_btn = widgets.Button(description="Reset",
                                        layout=widgets.Layout(width='80px', height='32px', border='1px solid #b91d47',
                                                              border_radius='4px'))
        self.reset_btn.style.button_color = '#ee1111'; self.reset_btn.style.text_color = 'white'; self.reset_btn.style.font_weight = 'bold'

        # --- Base State Inspector ---
        self.state_inspector = widgets.HTML(layout={'width': '100%', 'margin': '10px 0px 5px 0px'})

        btn_layout = widgets.Layout(width='115px', height='32px', border_radius='4px')

        self.show_array_btn = widgets.Button(description="Raw Array", layout=btn_layout)
        self.show_array_btn.style.button_color = '#34495e'; self.show_array_btn.style.text_color = 'white'; self.show_array_btn.style.font_weight = 'bold'; self.show_array_btn.layout.border = '1px solid #2c3e50'

        self.export_png_btn = widgets.Button(description="DCN PNG", layout=btn_layout)
        self.export_png_btn.style.button_color = '#1abc9c'; self.export_png_btn.style.text_color = 'white'; self.export_png_btn.style.font_weight = 'bold'; self.export_png_btn.layout.border = '1px solid #16a085'

        self.export_svg_btn = widgets.Button(description="DCN SVG", layout=btn_layout)
        self.export_svg_btn.style.button_color = '#2ecc71'; self.export_svg_btn.style.text_color = 'white'; self.export_svg_btn.style.font_weight = 'bold'; self.export_svg_btn.layout.border = '1px solid #27ae60'

        self.export_circ_png_btn = widgets.Button(description="Circ PNG", layout=btn_layout)
        self.export_circ_png_btn.style.button_color = '#3498db'; self.export_circ_png_btn.style.text_color = 'white'; self.export_circ_png_btn.style.font_weight = 'bold'; self.export_circ_png_btn.layout.border = '1px solid #2980b9'

        self.export_circ_svg_btn = widgets.Button(description="Circ SVG", layout=btn_layout)
        self.export_circ_svg_btn.style.button_color = '#2980b9'; self.export_circ_svg_btn.style.text_color = 'white'; self.export_circ_svg_btn.style.font_weight = 'bold'; self.export_circ_svg_btn.layout.border = '1px solid #1c5980'

        # --- Automatic Environment Detection ---
        is_voila = 'voila' in os.environ.get('SERVER_SOFTWARE', '').lower()
        active_buttons = [self.show_array_btn]

        if is_voila:
            active_buttons.extend([
                self.export_png_btn,
                self.export_svg_btn,
                self.export_circ_png_btn,
                self.export_circ_svg_btn
            ])

        self.extraction_buttons_row = widgets.HBox(
            active_buttons,
            layout={'width': '100%', 'justify_content': 'center', 'margin': '5px 0px 15px 0px', 'grid_gap': '10px'}
        )

        self.bottom_section = widgets.VBox([self.state_inspector, self.extraction_buttons_row],
                                           layout={'width': '100%'})

        # --- Output Canvases ---
        self.image_widget = widgets.Image(format='png', layout={'min_height': '400px', 'max_width': '100%', 'margin': '10px 0px'})
        self.circuit_image_widget = widgets.Image(format='png', layout={'max_width': '100%', 'margin': '10px 0px', 'display': 'none'})
        self.console = widgets.Output(layout={'border': '1px solid #ccc', 'width': '100%'})

        # --- Event Binding ---
        self.gate_dropdown.observe(self._toggle_angle_slider, names='value')
        self.controlled_checkbox.observe(self._toggle_control_selector, names='value')

        self.apply_btn.on_click(self._apply_gate)
        self.measure_btn.on_click(self._measure_qubits)
        self.zero_phase_btn.on_click(self._zero_global_phase)
        self.undo_btn.on_click(self._undo_action)
        self.redo_btn.on_click(self._redo_action)  # Bind the new method
        self.reset_btn.on_click(self._reset_circuit)
        self.show_array_btn.on_click(self._show_state_array)
        self.export_png_btn.on_click(self._export_png)
        self.export_svg_btn.on_click(self._export_svg)
        self.export_circ_png_btn.on_click(self._export_circ_png)
        self.export_circ_svg_btn.on_click(self._export_circ_svg)

        # --- Layout Assembly ---
        self.controls_top = widgets.HBox(
            [self.gate_dropdown, self.controlled_checkbox, self.control_selector, self.target_selector],
            layout={'align_items': 'center'})

        # Inject Redo Button into the bottom controls row
        self.controls_bottom = widgets.HBox(
            [self.angle_input, self.apply_btn, self.measure_btn, self.zero_phase_btn, self.undo_btn, self.redo_btn, self.reset_btn])

        ui_elements = [
            self.controls_top,
            self.controls_bottom,
            self.image_widget,
            self.circuit_image_widget,
            self.bottom_section,
            self.console
        ]

        self.ui = widgets.VBox(ui_elements, layout={'align_items': 'center', 'width': '100%'})

        if preloaded_circuit is not None:
            self._load_timeline(preloaded_circuit)

        self._update_plot()

    def _normalize_state(self, statevector):
        sv_array = np.array(statevector, dtype=complex)
        norm = np.linalg.norm(sv_array)
        if np.isclose(norm, 0.0): raise ValueError("A null vector cannot be normalized to represent a physical quantum state.")
        return (sv_array / norm).tolist()

    def _init_circuit(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        self._circuit_history.clear()
        self._action_history.clear()

        # Ensure Redo state is purged upon initialization
        self._redo_circuit_history.clear()
        self._redo_action_history.clear()

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

    def _load_timeline(self, qc: QuantumCircuit):
        """
        Internal method to parse a provided Qiskit circuit, generate the
        chronological state history, and reverse-load it into the Redo stack.
        """
        if qc.num_qubits != self.num_qubits:
            with self.console:
                self.console.clear_output()
                print(
                    f"Timeline Error: Provided circuit has {qc.num_qubits} qubits, but viewer expects {self.num_qubits}.")
            return

        forward_circs = []
        forward_actions = []
        temp_circ = self.circuit.copy()

        for instruction in qc.data:
            op = instruction.operation
            name = op.name.capitalize()

            try:
                qubit_indices = [qc.find_bit(q).index + 1 for q in instruction.qubits]
            except Exception:
                qubit_indices = []

            if name.upper() == 'MEASURE':
                action_desc = f"Measure Target(s): {qubit_indices}"
            elif len(qubit_indices) == 1:
                if hasattr(op, 'params') and op.params:
                    try:
                        angle_pi = float(op.params[0]) / np.pi
                        action_desc = f"Apply {name}({angle_pi:.3f}π) on Target(s): {qubit_indices}"
                    except (TypeError, ValueError):
                        action_desc = f"Apply {name} on Target(s): {qubit_indices}"
                else:
                    action_desc = f"Apply {name} on Target(s): {qubit_indices}"
            elif len(qubit_indices) > 1:
                controls = qubit_indices[:-1]
                target = qubit_indices[-1]
                action_desc = f"Apply Controlled-{name} on Target(s): [{target}] with Control(s): {controls}"
            else:
                action_desc = f"Apply {name}"

            temp_circ.append(instruction)
            forward_circs.append(temp_circ.copy())
            forward_actions.append(action_desc)

        # Execute LIFO reversal into the redo arrays
        self._redo_circuit_history = forward_circs[::-1]
        self._redo_action_history = forward_actions[::-1]

        with self.console:
            self.console.clear_output()
            print(f"Loaded a {len(forward_circs)}-step quantum algorithm. Click 'Redo' to step forward.")

    def _toggle_angle_slider(self, change):
        self.angle_input.disabled = (change.new not in ['P', 'Rx', 'Ry', 'Rz'])

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
                print(f"Validation Error: Intersection detected. Qubits {set(targets).intersection(controls)} cannot serve as both control and target.")
                return

        targets_ui = [t + 1 for t in targets]
        controls_ui = [c + 1 for c in controls]
        if is_controlled:
            action_desc = f"Apply Controlled-{gate_str} on Target(s): {targets_ui} with Control(s): {controls_ui}"
        else:
            if gate_str in ['P', 'Rx', 'Ry', 'Rz']:
                action_desc = f"Apply {gate_str}({self.angle_input.value:.3f}π) on Target(s): {targets_ui}"
            else:
                action_desc = f"Apply {gate_str} on Target(s): {targets_ui}"

        # Prune the divergent Redo branch upon new action
        self._redo_circuit_history.clear()
        self._redo_action_history.clear()

        self._circuit_history.append(self.circuit.copy())
        self._action_history.append(action_desc)

        if gate_str == 'H': base_gate = HGate()
        elif gate_str == 'X': base_gate = XGate()
        elif gate_str == 'Y': base_gate = YGate()
        elif gate_str == 'Z': base_gate = ZGate()
        elif gate_str == 'P': base_gate = PhaseGate(angle_radians)
        elif gate_str == 'Rx': base_gate = RXGate(angle_radians)
        elif gate_str == 'Ry': base_gate = RYGate(angle_radians)
        elif gate_str == 'Rz': base_gate = RZGate(angle_radians)

        if is_controlled:
            controlled_gate = base_gate.control(len(controls))
            for t in targets: self.circuit.append(controlled_gate, controls + [t])
        else:
            for t in targets: self.circuit.append(base_gate, [t])

        self._update_plot()

    def _measure_qubits(self, b):
        targets = [t - 1 for t in self.target_selector.value]
        targets_ui = [t + 1 for t in targets]

        with self.console:
            self.console.clear_output()
            if not targets:
                print("Validation Error: Please select at least one Target qubit to measure.")
                return

        action_desc = f"Measure Target(s): {targets_ui}"

        # Prune the divergent Redo branch upon new measurement
        self._redo_circuit_history.clear()
        self._redo_action_history.clear()

        self._circuit_history.append(self.circuit.copy())
        self._action_history.append(action_desc)

        try:
            sv_current = Statevector.from_instruction(self.circuit)
            outcome_str, sv_collapsed = sv_current.measure(targets)
            self.circuit = QuantumCircuit(self.num_qubits)
            self.circuit.initialize(sv_collapsed.data, self.circuit.qubits)

            results = [f"Qubit {t + 1}: {bit}" for t, bit in zip(targets, reversed(outcome_str))]
            self._update_plot()
            with self.console:
                print(f"💥 Measurement Result: {', '.join(results)}")
        except Exception as e:
            self._circuit_history.pop(); self._action_history.pop()
            with self.console:
                print(f"Measurement Error: {type(e).__name__}: {str(e)}")

    def _zero_global_phase(self, b):
        with self.console:
            try:
                sv_data = Statevector.from_instruction(self.circuit).data

                # Prune the divergent Redo branch upon new phase alignment
                self._redo_circuit_history.clear()
                self._redo_action_history.clear()

                self._circuit_history.append(self.circuit.copy())
                self._action_history.append("Zero Global Phase")
                for amplitude in sv_data:
                    if np.abs(amplitude) > 1e-6:
                        self.circuit.global_phase -= np.angle(amplitude)
                        break
                self._update_plot()
            except Exception as e:
                self._circuit_history.pop(); self._action_history.pop()
                self.console.clear_output()
                print(f"Phase Calculation Error: {type(e).__name__}: {str(e)}")

    def _undo_action(self, b):
        if self._circuit_history:
            # Shift the current state into the future stack before reverting
            self._redo_circuit_history.append(self.circuit.copy())
            self._redo_action_history.append(self._action_history.pop())

            self.circuit = self._circuit_history.pop()
            self._update_plot()
            with self.console:
                self.console.clear_output()
                print("Reverted to previous state.")
        else:
            with self.console:
                self.console.clear_output()
                print("Undo Stack Empty: You are at the initial circuit state.")

    def _redo_action(self, b):
        if self._redo_circuit_history:
            # Shift the current state into the past stack before advancing
            self._circuit_history.append(self.circuit.copy())
            self._action_history.append(self._redo_action_history.pop())

            self.circuit = self._redo_circuit_history.pop()
            self._update_plot()
            with self.console:
                self.console.clear_output()
                print("Restored subsequent state.")
        else:
            with self.console:
                self.console.clear_output()
                print("Redo Stack Empty: No future states available to restore.")

    def _reset_circuit(self, b):
        try:
            self._init_circuit()
            self._update_plot()
            self.console.clear_output()
        except Exception as e:
            with self.console:
                print(f"Reset Error: {str(e)}")

    def _show_state_array(self, b):
        with self.console:
            self.console.clear_output()
            try:
                sv_data = Statevector.from_instruction(self.circuit).data
                formatted_list = [complex(round(c.real, 6), round(c.imag, 6)) for c in sv_data]
                array_str = repr(formatted_list)

                copy_html = f"""
                <div style="margin: 15px 0px; padding: 15px; border: 1px solid #bdc3c7; border-radius: 4px; background-color: #f8f9fa;">
                    <div style="color: #2c3e50; font-family: sans-serif; font-weight: bold; margin-bottom: 8px;">📋 Raw Statevector Array:</div>
                    <textarea id="array_output_box" readonly style="width: 100%; height: 60px; font-family: monospace; font-size: 13px; border: 1px solid #ccc; border-radius: 3px; padding: 8px; box-sizing: border-box; resize: none;">{array_str}</textarea>
                    <button onclick="let ta = document.getElementById('array_output_box'); ta.select(); document.execCommand('copy'); this.innerText='&#10004; Copied to Clipboard!'; this.style.backgroundColor='#27ae60';" 
                            style="margin-top: 10px; padding: 8px 16px; background-color: #34495e; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif; transition: 0.3s;">
                        Copy to Clipboard
                    </button>
                </div>
                """
                display(widgets.HTML(copy_html))
            except Exception as e:
                print(f"Array Extraction Error: {type(e).__name__}: {str(e)}")

    def _export_png(self, b):
        with self.console:
            self.console.clear_output()
            try:
                vis = DimensionalCircleNotation.from_qiskit(self.circuit)
                with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
                    b64_str = vis.exportBase64(formatStr='png')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/png;base64,{b64_str}" download="quantum_state_dcn.png"
                       style="background-color: #1abc9c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; font-family: sans-serif; border: 1px solid #16a085;">
                       &#128190; Save High-Res PNG
                    </a>
                </div>
                """
                display(widgets.HTML(download_html))
            except Exception as e:
                print(f"PNG Export Error: {type(e).__name__}: {str(e)}")

    def _export_svg(self, b):
        with self.console:
            self.console.clear_output()
            try:
                vis = DimensionalCircleNotation.from_qiskit(self.circuit)
                with plt.rc_context({'figure.figsize': self.render_figsize, 'svg.fonttype': 'none'}):
                    b64_str = vis.exportBase64(formatStr='svg')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/svg+xml;base64,{b64_str}" download="quantum_state_dcn.svg"
                       style="background-color: #2ecc71; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; font-family: sans-serif; border: 1px solid #27ae60;">
                       &#128190; Save Scalable Vector (SVG)
                    </a>
                </div>
                """
                display(widgets.HTML(download_html))
            except Exception as e:
                print(f"SVG Export Error: {type(e).__name__}: {str(e)}")

    def _export_circ_png(self, b):
        with self.console:
            self.console.clear_output()
            try:
                fig = self.circuit.draw(output='mpl')
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/png;base64,{b64_str}" download="quantum_circuit.png"
                       style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; font-family: sans-serif; border: 1px solid #2980b9;">
                       &#128190; Save Circuit PNG
                    </a>
                </div>
                """
                display(widgets.HTML(download_html))
            except Exception as e:
                print(f"Circuit PNG Export Error: {type(e).__name__}: {str(e)}")

    def _export_circ_svg(self, b):
        with self.console:
            self.console.clear_output()
            try:
                fig = self.circuit.draw(output='mpl')
                buf = BytesIO()

                with plt.rc_context({'svg.fonttype': 'none'}):
                    fig.savefig(buf, format='svg', bbox_inches='tight')
                plt.close(fig)

                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/svg+xml;base64,{b64_str}" download="quantum_circuit.svg"
                       style="background-color: #2980b9; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; font-family: sans-serif; border: 1px solid #1c5980;">
                       &#128190; Save Scalable Circuit (SVG)
                    </a>
                </div>
                """
                display(widgets.HTML(download_html))
            except Exception as e:
                print(f"Circuit SVG Export Error: {type(e).__name__}: {str(e)}")

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
        return f"|ψ⟩ = {equation}"

    def _update_plot(self):
        with self.console:
            try:
                html_content = "<div style='text-align: left; font-family: monospace; font-size: 14px; max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 4px;'>"
                for i, past_circ in enumerate(self._circuit_history):
                    sv = Statevector.from_instruction(past_circ)
                    dirac_str = self._format_dirac_notation(sv.data)
                    if i == 0:
                        html_content += f"<div style='margin-bottom: 5px;'><b>Initial State:</b> {dirac_str}</div>"
                    else:
                        html_content += f"<div style='margin-bottom: 5px;'><b>{self._action_history[i - 1]}</b> &#8594; {dirac_str}</div>"

                sv_current = Statevector.from_instruction(self.circuit)
                current_dirac = self._format_dirac_notation(sv_current.data)
                if not self._circuit_history:
                    html_content += f"<div style='margin-bottom: 5px;'><b>Initial State:</b> {current_dirac}</div>"
                else:
                    html_content += f"<div style='margin-bottom: 5px;'><b>{self._action_history[-1]}</b> &#8594; {current_dirac}</div>"
                html_content += "</div>"
                self.state_inspector.value = html_content

                vis = DimensionalCircleNotation.from_qiskit(self.circuit)
                with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
                    b64_str = vis.exportBase64(formatStr='png')
                self.image_widget.value = base64.b64decode(b64_str)

                # Conditional Circuit Rendering
                # Conditional Circuit Rendering with Ghost Overlay
                # Conditional Circuit Rendering with Ghost Overlay
                # Conditional Circuit Rendering with Ghost Overlay
                # Conditional Circuit Rendering with Ghost Overlay
                if self.show_circuit:
                    # 1. Render Current Circuit
                    fig_curr = self.circuit.draw(output='mpl', scale=0.4, style={'backgroundcolor': 'none'})
                    buf_curr = BytesIO()
                    fig_curr.savefig(buf_curr, format='png', bbox_inches='tight', dpi=300)
                    plt.close(fig_curr)

                    if not self._redo_circuit_history:
                        # No future states exist. Output the normal circuit.
                        self.circuit_image_widget.value = buf_curr.getvalue()
                    else:
                        # 2. Render the furthest future "Ghost" Circuit
                        future_circ = self._redo_circuit_history[0]
                        fig_fut = future_circ.draw(output='mpl', scale=0.4, style={'backgroundcolor': 'none'})
                        buf_fut = BytesIO()
                        fig_fut.savefig(buf_fut, format='png', bbox_inches='tight', dpi=300)
                        plt.close(fig_fut)

                        # Rewind the byte streams
                        buf_curr.seek(0)
                        buf_fut.seek(0)

                        img_curr = Image.open(buf_curr).convert("RGBA")
                        img_fut = Image.open(buf_fut).convert("RGBA")

                        # 3. Chroma-Key Operation (Force White to Transparent)
                        # Process Future Image (Ghosting)
                        fut_data = img_fut.getdata()
                        new_fut = []
                        for r, g, b, a in fut_data:
                            if r > 240 and g > 240 and b > 240:
                                new_fut.append((255, 255, 255, 0))  # Eradicate white background
                            else:
                                new_fut.append((r, g, b, int(a * 0.35)))  # Fade remaining drawing to 35% opacity
                        img_fut.putdata(new_fut)

                        # Process Current Image (Opaque Foreground)
                        curr_data = img_curr.getdata()
                        new_curr = []
                        for r, g, b, a in curr_data:
                            if r > 240 and g > 240 and b > 240:
                                new_curr.append((255, 255, 255, 0))  # Eradicate white background
                            else:
                                new_curr.append((r, g, b, a))  # Keep drawing opaque
                        img_curr.putdata(new_curr)

                        # 4. Dynamic Canvas Compositing
                        # Create a canvas large enough to hold the maximum bounds of both images
                        max_width = max(img_fut.width, img_curr.width)
                        max_height = max(img_fut.height, img_curr.height)
                        canvas = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))

                        # Center both images vertically to perfectly align the horizontal qubit wires
                        y_fut = (max_height - img_fut.height) // 2
                        y_curr = (max_height - img_curr.height) // 2

                        # Paste Base Layer (Ghost) then Top Layer (Current)
                        canvas.paste(img_fut, (0, y_fut), img_fut)
                        canvas.paste(img_curr, (0, y_curr), img_curr)

                        # 5. Export to UI
                        final_buf = BytesIO()
                        canvas.save(final_buf, format="PNG")
                        self.circuit_image_widget.value = final_buf.getvalue()

            except Exception as e:
                print("An error occurred during visualization generation:")
                traceback.print_exc()

    def show(self, show_circuit=None):
        """
        Spawns native OS windows displaying the current state.
        If show_circuit is True, it spawns two separate Matplotlib windows.
        """
        if show_circuit is not None:
            self.show_circuit = show_circuit

        try:
            if self.show_circuit:
                circ_fig = self.circuit.draw(output='mpl',scale=0.4)
                circ_fig.suptitle("Quantum Circuit Pipeline")

            vis = DimensionalCircleNotation.from_qiskit(self.circuit)
            with plt.rc_context({'figure.figsize': self.render_figsize}):
                vis.draw()
                if hasattr(vis, 'fig') and vis.fig is not None:
                    vis.fig.suptitle("DCN Quantum State Viewer")
                    plt.show(block=True)
                else:
                    print("Error: The visualization class failed to generate a Matplotlib 'fig'.")
        except Exception as e:
            with self.console:
                print(f"Standalone Render Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    def display(self, figsize=None, ui_width=None, show_circuit=None):
        """Renders the UI in a Jupyter Notebook or Voilà browser environment."""
        if figsize is not None:
            self.render_figsize = figsize

        if show_circuit is not None:
            self.show_circuit = show_circuit
            self.circuit_image_widget.layout.display = 'block' if self.show_circuit else 'none'

        self._update_plot()

        if ui_width is not None:
            self.image_widget.layout.width = ui_width

        from IPython.display import display as ipy_display
        ipy_display(self.ui)

class ChallengeDCNViewer(InteractiveDCNViewer):
    """
    An assessment-driven subclass of InteractiveDCNViewer.
    Evaluates the current state against a defined target state.
    Allows for an optional preloaded optimal solution path.
    """

    def __init__(self, num_qubits, initial_state, target_state, preloaded_circuit=None):
        self.status_banner = widgets.HTML("<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>")

        shared_layout = widgets.Layout(min_height='320px', width='100%', object_fit='contain', justify_content='center')
        self.target_image_widget = widgets.Image(format='png', layout=shared_layout)
        self._raw_target_state = target_state

        # Pass the preloaded_circuit up to the parent constructor to build the timeline
        super().__init__(num_qubits=num_qubits, initial_state=initial_state, preloaded_circuit=preloaded_circuit)

        self.image_widget.layout = shared_layout
        self.render_figsize = (5.0, 4.0)

        self.target_state = self._normalize_state(self._raw_target_state)
        self._render_target()
        self._update_plot()

        comparison_box = widgets.HBox([
            widgets.VBox(
                [widgets.HTML("<h3 style='text-align: center; color: #555; margin-bottom: 0px;'>Current State</h3>"),
                 self.image_widget], layout={'align_items': 'center', 'width': '50%'}),
            widgets.VBox(
                [widgets.HTML("<h3 style='text-align: center; color: #555; margin-bottom: 0px;'>Target State</h3>"),
                 self.target_image_widget], layout={'align_items': 'center', 'width': '50%'})
        ], layout={'width': '100%', 'justify_content': 'space-around', 'align_items': 'flex-start'})

        # Apply the reordered vertical layout hierarchy
        ui_elements = [
            self.status_banner,
            self.controls_top,
            self.controls_bottom,
            comparison_box,             # DCN Visualization Middle 1
            self.circuit_image_widget,  # Quantum Circuit Middle 2
            self.bottom_section,        # History and Exports Bottom
            self.console
        ]

        self.ui = widgets.VBox(ui_elements, layout={'align_items': 'center', 'width': '100%'})
        self._check_success()

    def _render_target(self):
        qc_target = QuantumCircuit(self.num_qubits)
        qc_target.initialize(self.target_state, qc_target.qubits)
        try:
            vis = DimensionalCircleNotation.from_qiskit(qc_target)
            with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
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
            if np.isclose(state_fidelity(sv_current, sv_target), 1.0, atol=1e-5):
                self.status_banner.value = "<h2 style='text-align: center; color: #27ae60;'>Status: Challenge Completed! 🎉</h2>"
            else:
                self.status_banner.value = "<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>"
        except Exception:
            pass