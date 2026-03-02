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

from .visualization import DimensionalCircleNotation, CircleNotation
from .dim_Bloch_spheres import DimensionalBlochSpheres

# --- Global Visualization Registry ---
VISUALIZATION_REGISTRY = {
    'Dimensional Circle Notation': DimensionalCircleNotation,
    'Circle Notation': CircleNotation,
    'Dimensional Bloch Spheres': DimensionalBlochSpheres
}

def register_visualization(name: str, vis_class: type):
    """
    Dynamically injects a new visualization paradigm into the global registry.
    This allows external scripts to add custom visualizers without modifying the core package.
    """
    VISUALIZATION_REGISTRY[name] = vis_class


class InteractiveViewer:
    """
    An encapsulated interactive UI for dynamically visualizing quantum circuits.
    Features polymorphic visualization rendering, parametric controls, state vector
    normalization, projective measurements, deterministic undo/redo mechanism,
    a live Dirac notation readout, and an optional ghosted circuit diagram.
    """

    def __init__(self, num_qubits=3, initial_state=None, preloaded_circuit=None):
        self.num_qubits = num_qubits
        self.show_circuit = False
        self.render_figsize = (8.0, 6.0)

        # Persist the preloaded circuit to allow reconstruction upon reset
        self._preloaded_circuit = preloaded_circuit

        # Initialize primary and secondary (redo) tracking arrays
        self._circuit_history = []
        self._action_history = []
        self._redo_circuit_history = []
        self._redo_action_history = []

        self.initial_state = self._normalize_state(initial_state) if initial_state is not None else None
        self._init_circuit()

        # --- UI Components (Controls) ---

        # We convert the keys to a list and default to the first registered visualization.
        registry_keys = list(VISUALIZATION_REGISTRY.keys())
        initial_vis = registry_keys[0] if registry_keys else None

        # Visualization Switcher Dropdown
        self.vis_dropdown = widgets.Dropdown(
            options=registry_keys,
            value=initial_vis,
            description='Visualization:',
            layout={'width': '320px'}
        )

        # Dynamic Qubit Selector for Bloch Spheres
        # Cap the maximum selectable qubit to 3 per your constraint
        bloch_options = list(range(1, min(self.num_qubits, 3) + 1))
        self.bloch_qubit_dropdown = widgets.Dropdown(
            options=bloch_options,
            value=1,
            description='Focus Qubit:',
            layout={'display': 'none', 'width': '160px'}  # Hidden by default
        )

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
        self.apply_btn.style.button_color = '#2d89ef';
        self.apply_btn.style.text_color = 'white';
        self.apply_btn.style.font_weight = 'bold'

        self.measure_btn = widgets.Button(description="Measure",
                                          layout=widgets.Layout(width='95px', height='32px', border='1px solid #8e44ad',
                                                                border_radius='4px'))
        self.measure_btn.style.button_color = '#9b59b6';
        self.measure_btn.style.text_color = 'white';
        self.measure_btn.style.font_weight = 'bold'

        self.zero_phase_btn = widgets.Button(description="0-Phase", layout=widgets.Layout(width='90px', height='32px',
                                                                                          border='1px solid #d37c15',
                                                                                          border_radius='4px'))
        self.zero_phase_btn.style.button_color = '#f39c12';
        self.zero_phase_btn.style.text_color = 'white';
        self.zero_phase_btn.style.font_weight = 'bold'

        self.undo_btn = widgets.Button(description="Undo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.undo_btn.style.button_color = '#95a5a6';
        self.undo_btn.style.text_color = 'white';
        self.undo_btn.style.font_weight = 'bold'

        self.redo_btn = widgets.Button(description="Redo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.redo_btn.style.button_color = '#95a5a6';
        self.redo_btn.style.text_color = 'white';
        self.redo_btn.style.font_weight = 'bold'

        self.reset_btn = widgets.Button(description="Reset",
                                        layout=widgets.Layout(width='80px', height='32px', border='1px solid #b91d47',
                                                              border_radius='4px'))
        self.reset_btn.style.button_color = '#ee1111';
        self.reset_btn.style.text_color = 'white';
        self.reset_btn.style.font_weight = 'bold'

        # --- System Size Controls ---
        self.attach_btn = widgets.Button(description="Attach |0⟩",
                                         layout=widgets.Layout(width='90px', height='32px', border='1px solid #27ae60',
                                                               border_radius='4px'))
        self.attach_btn.style.button_color = '#2ecc71';
        self.attach_btn.style.text_color = 'white';
        self.attach_btn.style.font_weight = 'bold'

        self.detach_btn = widgets.Button(description="Detach",
                                         layout=widgets.Layout(width='85px', height='32px', border='1px solid #c0392b',
                                                               border_radius='4px'))
        self.detach_btn.style.button_color = '#e74c3c';
        self.detach_btn.style.text_color = 'white';
        self.detach_btn.style.font_weight = 'bold'

        # Bind the events
        self.attach_btn.on_click(self._attach_qubit)
        self.detach_btn.on_click(self._detach_qubit)

        # --- Base State Inspector ---
        self.state_inspector = widgets.HTML(layout={'width': '100%', 'margin': '10px 0px 5px 0px'})

        btn_layout = widgets.Layout(width='115px', height='32px', border_radius='4px')

        self.show_array_btn = widgets.Button(description="Raw Array", layout=btn_layout)
        self.show_array_btn.style.button_color = '#34495e';
        self.show_array_btn.style.text_color = 'white';
        self.show_array_btn.style.font_weight = 'bold';
        self.show_array_btn.layout.border = '1px solid #2c3e50'

        # Generalized generic nomenclature for export buttons
        self.export_png_btn = widgets.Button(description="State PNG", layout=btn_layout)
        self.export_png_btn.style.button_color = '#1abc9c';
        self.export_png_btn.style.text_color = 'white';
        self.export_png_btn.style.font_weight = 'bold';
        self.export_png_btn.layout.border = '1px solid #16a085'

        self.export_svg_btn = widgets.Button(description="State SVG", layout=btn_layout)
        self.export_svg_btn.style.button_color = '#2ecc71';
        self.export_svg_btn.style.text_color = 'white';
        self.export_svg_btn.style.font_weight = 'bold';
        self.export_svg_btn.layout.border = '1px solid #27ae60'

        self.export_circ_png_btn = widgets.Button(description="Circ PNG", layout=btn_layout)
        self.export_circ_png_btn.style.button_color = '#3498db';
        self.export_circ_png_btn.style.text_color = 'white';
        self.export_circ_png_btn.style.font_weight = 'bold';
        self.export_circ_png_btn.layout.border = '1px solid #2980b9'

        self.export_circ_svg_btn = widgets.Button(description="Circ SVG", layout=btn_layout)
        self.export_circ_svg_btn.style.button_color = '#2980b9';
        self.export_circ_svg_btn.style.text_color = 'white';
        self.export_circ_svg_btn.style.font_weight = 'bold';
        self.export_circ_svg_btn.layout.border = '1px solid #1c5980'

        # --- Automatic Environment Detection ---
        is_voila = 'voila' in os.environ.get('SERVER_SOFTWARE', '').lower()
        active_buttons = [self.show_array_btn]

        if is_voila:
            active_buttons.extend(
                [self.export_png_btn, self.export_svg_btn, self.export_circ_png_btn, self.export_circ_svg_btn])

        self.extraction_buttons_row = widgets.HBox(active_buttons, layout={'width': '100%', 'justify_content': 'center',
                                                                           'margin': '5px 0px 15px 0px',
                                                                           'grid_gap': '10px'})
        self.bottom_section = widgets.VBox([self.state_inspector, self.extraction_buttons_row],
                                           layout={'width': '100%'})

        # --- Output Canvases ---
        self.image_widget = widgets.Image(format='png',
                                          layout={'min_height': '400px', 'max_width': '100%', 'margin': '10px 0px'})
        self.circuit_image_widget = widgets.Image(format='png',
                                                  layout={'max_width': '100%', 'margin': '10px 0px', 'display': 'none'})
        self.console = widgets.Output(layout={'border': '1px solid #ccc', 'width': '100%'})

        # --- Event Binding ---
        self.vis_dropdown.observe(self._on_vis_change, names='value')
        self.bloch_qubit_dropdown.observe(self._on_vis_change, names='value')  # Triggers redraw on change

        self.gate_dropdown.observe(self._toggle_angle_slider, names='value')
        self.controlled_checkbox.observe(self._toggle_control_selector, names='value')

        self.apply_btn.on_click(self._apply_gate)
        self.measure_btn.on_click(self._measure_qubits)
        self.zero_phase_btn.on_click(self._zero_global_phase)
        self.undo_btn.on_click(self._undo_action)
        self.redo_btn.on_click(self._redo_action)
        self.reset_btn.on_click(self._reset_circuit)
        self.show_array_btn.on_click(self._show_state_array)
        self.export_png_btn.on_click(self._export_png)
        self.export_svg_btn.on_click(self._export_svg)
        self.export_circ_png_btn.on_click(self._export_circ_png)
        self.export_circ_svg_btn.on_click(self._export_circ_svg)

        # --- Layout Assembly ---
        self.controls_header = widgets.HBox(
            [self.vis_dropdown, self.bloch_qubit_dropdown],
            layout={'width': '100%', 'justify_content': 'center', 'margin': '0px 0px 15px 0px', 'grid_gap': '15px'}
        )

        self.controls_top = widgets.HBox(
            [self.gate_dropdown, self.controlled_checkbox, self.control_selector, self.target_selector],
            layout={'align_items': 'center'})

        self.controls_bottom = widgets.HBox(
            [self.angle_input, self.apply_btn, self.measure_btn, self.zero_phase_btn,
             self.undo_btn, self.redo_btn, self.attach_btn, self.detach_btn, self.reset_btn],
            layout={'align_items': 'center', 'grid_gap': '5px'}
        )

        ui_elements = [
            self.controls_header,
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

    def _get_active_vis_class(self):
        """Strategy pattern resolver pulling from the global visualization registry."""

        # Fallback to the first available class in the registry, or None if empty
        default_class = next(iter(VISUALIZATION_REGISTRY.values()), None)

        return VISUALIZATION_REGISTRY.get(self.vis_dropdown.value, default_class)

    def _on_vis_change(self, change):
        """Toggles specialized UI components and triggers a redraw."""
        if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
            self.bloch_qubit_dropdown.layout.display = 'flex'
        else:
            self.bloch_qubit_dropdown.layout.display = 'none'

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
        self._action_history.clear()
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

        self._redo_circuit_history = forward_circs[::-1]
        self._redo_action_history = forward_actions[::-1]

        with self.console:
            self.console.clear_output()
            print(f"Loaded a {len(forward_circs)}-step quantum algorithm. Click 'Redo' to step forward.")

    def _toggle_angle_slider(self, change):
        self.angle_input.disabled = (change.new not in ['P', 'Rx', 'Ry', 'Rz'])

    def _toggle_control_selector(self, change):
        self.control_selector.disabled = not change.new

    def _refresh_ui_selectors(self):
        """Safely updates the UI dropdowns to reflect the new system size."""
        new_options = list(range(1, self.num_qubits + 1))

        self.target_selector.value = (1,)
        self.control_selector.value = tuple() if self.num_qubits < 2 else (2,)

        self.target_selector.options = new_options
        self.control_selector.options = new_options
        self.controlled_checkbox.disabled = (self.num_qubits < 2)

        # Enforce the 3-qubit maximum constraint for the Bloch Sphere visualizer
        bloch_options = list(range(1, min(self.num_qubits, 3) + 1))

        # Prevent TraitError by ensuring the active value exists in the new bounds
        if self.bloch_qubit_dropdown.value not in bloch_options:
            self.bloch_qubit_dropdown.value = 1

        self.bloch_qubit_dropdown.options = bloch_options

    def _attach_qubit(self, b):
        with self.console:
            self.console.clear_output()
            try:
                # 1. Extract current state
                sv_current = Statevector.from_instruction(self.circuit).data

                # 2. Compute Kronecker Product: |0⟩ ⊗ |ψ_current⟩
                # In Qiskit's little-endian ordering, prepending [1, 0] adds the new qubit
                # at the most significant bit (the bottom wire of the circuit).
                sv_new = np.kron(np.array([1.0, 0.0], dtype=complex), sv_current)

                # 3. Update State History
                self._redo_circuit_history.clear()
                self._redo_action_history.clear()
                self._circuit_history.append(self.circuit.copy())
                self._action_history.append(f"Attach Qubit {self.num_qubits + 1} in |0⟩")

                # 4. Rebuild Circuit Architecture
                self.num_qubits += 1
                self.circuit = QuantumCircuit(self.num_qubits)
                self.circuit.initialize(sv_new, self.circuit.qubits)

                self._refresh_ui_selectors()
                self._update_plot()

            except Exception as e:
                print(f"Attach Error: {type(e).__name__}: {str(e)}")

    def _detach_qubit(self, b):
        with self.console:
            self.console.clear_output()

            if self.num_qubits <= 1:
                print("Validation Error: Cannot detach the final remaining qubit.")
                return

            # Safely extract the target qubit from the UI.
            # If the user deselected all options (empty tuple), default to the highest indexed qubit.
            if not self.target_selector.value:
                target_ui = self.num_qubits
            else:
                target_ui = self.target_selector.value[0]

            k = target_ui - 1  # Convert to 0-based integer for mathematical arrays

            try:
                sv_current = Statevector.from_instruction(self.circuit).data

                # 1. Reshape the 1D array into an N-dimensional tensor of shape (2, 2, ..., 2)
                tensor = sv_current.reshape((2,) * self.num_qubits)

                # 2. Determine the correct tensor axis based on Qiskit's Endianness
                # Qiskit orders axes as (q_{N-1}, ..., q_1, q_0).
                # Therefore, qubit k corresponds to axis (N - 1 - k).
                axis = self.num_qubits - 1 - k

                # 3. Project the axis onto |0⟩ (index 0)
                slices = [slice(None)] * self.num_qubits
                slices[axis] = 0
                sv_projected = tensor[tuple(slices)].flatten()

                # 4. Validate and Re-normalize
                norm = np.linalg.norm(sv_projected)
                if np.isclose(norm, 0.0):
                    print(
                        f"Mathematical Error: Qubit {target_ui} is deterministically in state |1⟩. Projecting it to |0⟩ yields a null state. Please apply an X-gate before detaching.")
                    return
                sv_new = sv_projected / norm

                # 5. Update State History
                self._redo_circuit_history.clear()
                self._redo_action_history.clear()
                self._circuit_history.append(self.circuit.copy())

                # Update the history string to explicitly state which qubit was defaulted/targeted
                self._action_history.append(f"Detach Qubit {target_ui} (Traced out via |0⟩ projection)")

                # 6. Rebuild Circuit Architecture
                self.num_qubits -= 1
                self.circuit = QuantumCircuit(self.num_qubits)
                self.circuit.initialize(sv_new, self.circuit.qubits)

                self._refresh_ui_selectors()
                self._update_plot()

            except Exception as e:
                print(f"Detach Error: {type(e).__name__}: {str(e)}")

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

        targets_ui = [t + 1 for t in targets]
        controls_ui = [c + 1 for c in controls]
        if is_controlled:
            action_desc = f"Apply Controlled-{gate_str} on Target(s): {targets_ui} with Control(s): {controls_ui}"
        else:
            if gate_str in ['P', 'Rx', 'Ry', 'Rz']:
                action_desc = f"Apply {gate_str}({self.angle_input.value:.3f}π) on Target(s): {targets_ui}"
            else:
                action_desc = f"Apply {gate_str} on Target(s): {targets_ui}"

        self._redo_circuit_history.clear()
        self._redo_action_history.clear()

        self._circuit_history.append(self.circuit.copy())
        self._action_history.append(action_desc)

        if gate_str == 'H':
            base_gate = HGate()
        elif gate_str == 'X':
            base_gate = XGate()
        elif gate_str == 'Y':
            base_gate = YGate()
        elif gate_str == 'Z':
            base_gate = ZGate()
        elif gate_str == 'P':
            base_gate = PhaseGate(angle_radians)
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
        targets_ui = [t + 1 for t in targets]

        with self.console:
            self.console.clear_output()
            if not targets:
                print("Validation Error: Please select at least one Target qubit to measure.")
                return

        # Initial descriptor before collapse
        action_desc = f"Measure Target(s): {targets_ui}"

        self._redo_circuit_history.clear()
        self._redo_action_history.clear()

        self._circuit_history.append(self.circuit.copy())
        self._action_history.append(action_desc)

        try:
            # 1. Perform the mathematical measurement
            sv_current = Statevector.from_instruction(self.circuit)
            outcome_str, sv_collapsed = sv_current.measure(targets)

            from qiskit.circuit.library import UnitaryGate
            import numpy as np

            # 2. Create an invisible (Identity) gate, but label it for the UI
            measure_gate = UnitaryGate(np.eye(2), label="M")
            measure_gate.name = "M"

            # 3. Append the visual placeholder to all measured wires
            for t in targets:
                self.circuit.append(measure_gate, [t])

            # 4. Snap the simulation to the newly collapsed state
            # This acts as a hidden checkpoint for Statevector.from_instruction()
            self.circuit.initialize(sv_collapsed.data, self.circuit.qubits)

            # 5. Format the measurement results
            results = [f"Qubit {t + 1}: {bit}" for t, bit in zip(targets, reversed(outcome_str))]

            # 6. CRITICAL UPDATE: Mutate the action history string to include the result.
            # This keeps the history stacks perfectly symmetric while updating the UI text.
            self._action_history[-1] = f"{action_desc} 💥 Outcome: [{', '.join(results)}]"

            # 7. Redraw the UI (The inspector will now catch the mutated action string)
            self._update_plot()

        except Exception as e:
            # Revert the symmetry if the measurement strictly fails
            self._circuit_history.pop()
            self._action_history.pop()
            with self.console:
                print(f"Measurement Error: {type(e).__name__}: {str(e)}")

    def _zero_global_phase(self, b):
        with self.console:
            try:
                sv_data = Statevector.from_instruction(self.circuit).data

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
                self._circuit_history.pop();
                self._action_history.pop()
                self.console.clear_output()
                print(f"Phase Calculation Error: {type(e).__name__}: {str(e)}")

    def _undo_action(self, b):
        if self._circuit_history:
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

            # Re-inject the preloaded timeline if it exists
            if self._preloaded_circuit is not None:
                self._load_timeline(self._preloaded_circuit)

            self._update_plot()
            self.console.clear_output()
        except Exception as e:
            with self.console:
                print(f"Reset Error: {str(e)}")

    def _get_drawable_circuit(self, circ: QuantumCircuit) -> QuantumCircuit:
        """
        Creates a temporary copy of the circuit and strips ALL Qiskit 'initialize'
        instructions to prevent visual clutter in the UI, without altering the simulation math.
        """
        drawable_circ = circ.copy()

        # Filter out all initialization checkpoints so they do not render on the screen
        clean_data = []
        for inst in drawable_circ.data:
            if inst.operation.name != 'initialize':
                clean_data.append(inst)

        drawable_circ.data = clean_data
        return drawable_circ

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
                vis_class = self._get_active_vis_class()

                if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
                    # REMOVED the '- 1' to pass the 1-based index directly
                    vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
                else:
                    vis = vis_class.from_qiskit(self.circuit)

                with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
                    b64_str = vis.exportBase64(formatStr='png')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/png;base64,{b64_str}" download="quantum_state_vis.png"
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
                vis_class = self._get_active_vis_class()

                if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
                    # REMOVED the '- 1' to pass the 1-based index directly
                    vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
                else:
                    vis = vis_class.from_qiskit(self.circuit)

                with plt.rc_context({'figure.figsize': self.render_figsize, 'svg.fonttype': 'none'}):
                    b64_str = vis.exportBase64(formatStr='svg')

                download_html = f"""
                <div style="margin: 15px 0px 10px 0px; text-align: center;">
                    <a href="data:image/svg+xml;base64,{b64_str}" download="quantum_state_vis.svg"
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
                drawable_circ = self._get_drawable_circuit(self.circuit)
                fig = drawable_circ.draw(output='mpl')
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

                # Polymorphic visualization generation based on strategy
                vis_class = self._get_active_vis_class()

                if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
                    vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
                else:
                    vis = vis_class.from_qiskit(self.circuit)

                with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
                    b64_str = vis.exportBase64(formatStr='png')
                self.image_widget.value = base64.b64decode(b64_str)

                # Conditional Circuit Rendering with Ghost Overlay
                if self.show_circuit:
                    drawable_curr = self._get_drawable_circuit(self.circuit)
                    fig_curr = drawable_curr.draw(output='mpl', scale=0.4, style={'backgroundcolor': 'none'})
                    buf_curr = BytesIO()
                    fig_curr.savefig(buf_curr, format='png', bbox_inches='tight', dpi=300)
                    plt.close(fig_curr)

                    if not self._redo_circuit_history:
                        self.circuit_image_widget.value = buf_curr.getvalue()
                    else:
                        future_circ = self._redo_circuit_history[0]
                        drawable_fut = self._get_drawable_circuit(future_circ)
                        fig_fut = drawable_fut.draw(output='mpl', scale=0.4, style={'backgroundcolor': 'none'})
                        buf_fut = BytesIO()
                        fig_fut.savefig(buf_fut, format='png', bbox_inches='tight', dpi=300)
                        plt.close(fig_fut)

                        buf_curr.seek(0)
                        buf_fut.seek(0)

                        img_curr = Image.open(buf_curr).convert("RGBA")
                        img_fut = Image.open(buf_fut).convert("RGBA")

                        fut_data = img_fut.getdata()
                        new_fut = []
                        for r, g, b, a in fut_data:
                            if r > 240 and g > 240 and b > 240:
                                new_fut.append((255, 255, 255, 0))
                            else:
                                new_fut.append((r, g, b, int(a * 0.35)))
                        img_fut.putdata(new_fut)

                        curr_data = img_curr.getdata()
                        new_curr = []
                        for r, g, b, a in curr_data:
                            if r > 240 and g > 240 and b > 240:
                                new_curr.append((255, 255, 255, 0))
                            else:
                                new_curr.append((r, g, b, a))
                        img_curr.putdata(new_curr)

                        max_width = max(img_fut.width, img_curr.width)
                        max_height = max(img_fut.height, img_curr.height)
                        canvas = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))

                        y_fut = (max_height - img_fut.height) // 2
                        y_curr = (max_height - img_curr.height) // 2

                        canvas.paste(img_fut, (0, y_fut), img_fut)
                        canvas.paste(img_curr, (0, y_curr), img_curr)

                        final_buf = BytesIO()
                        canvas.save(final_buf, format="PNG")
                        self.circuit_image_widget.value = final_buf.getvalue()

            except Exception as e:
                print("An error occurred during visualization generation:")
                traceback.print_exc()

    def show(self, show_circuit=None):
        if show_circuit is not None:
            self.show_circuit = show_circuit

        try:
            if self.show_circuit:
                drawable_circ = self._get_drawable_circuit(self.circuit)
                circ_fig = drawable_circ.draw(output='mpl', scale=0.4)
                circ_fig.suptitle("Quantum Circuit Pipeline")

            # Extract the active class logic here as well
            vis_class = self._get_active_vis_class()

            if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
                vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
            else:
                vis = vis_class.from_qiskit(self.circuit)

            with plt.rc_context({'figure.figsize': self.render_figsize}):
                vis.draw()
                if hasattr(vis, 'fig') and vis.fig is not None:
                    # Dynamically set the window title based on the dropdown string
                    vis.fig.suptitle(f"{self.vis_dropdown.value} Viewer")
                    plt.show(block=True)
                else:
                    print("Error: The visualization class failed to generate a Matplotlib 'fig'.")
        except Exception as e:
            with self.console:
                print(f"Standalone Render Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    def display(self, figsize=None, ui_width=None, show_circuit=None):
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


class ChallengeViewer(InteractiveViewer):
    """
    An assessment-driven subclass of InteractiveViewer.
    Evaluates the current state against a defined target state.
    """

    def __init__(self, num_qubits, initial_state, target_state, preloaded_circuit=None):
        self.status_banner = widgets.HTML("<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>")

        shared_layout = widgets.Layout(min_height='320px', width='100%', object_fit='contain', justify_content='center')
        self.target_image_widget = widgets.Image(format='png', layout=shared_layout)
        self._raw_target_state = target_state

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

        # Insert the newly decoupled dropdown header directly under the banner
        ui_elements = [
            self.status_banner,
            self.controls_header,
            self.controls_top,
            self.controls_bottom,
            comparison_box,
            self.circuit_image_widget,
            self.bottom_section,
            self.console
        ]

        self.ui = widgets.VBox(ui_elements, layout={'align_items': 'center', 'width': '100%'})
        self._check_success()

    def _on_vis_change(self, change):
        """Overrides the parent method to ensure the target state visual updates as well."""
        self._render_target()
        super()._on_vis_change(change)

    def _render_target(self):
        qc_target = QuantumCircuit(self.num_qubits)
        qc_target.initialize(self.target_state, qc_target.qubits)
        try:
            # Route target generation through the polymorphic parser
            vis_class = self._get_active_vis_class()

            if self.vis_dropdown.value == 'Dimensional Bloch Spheres':
                vis = vis_class.from_qiskit(qc_target, select_qubit=self.bloch_qubit_dropdown.value)
            else:
                vis = vis_class.from_qiskit(qc_target)

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
            # 1. extract the target system size from the statevector length
            target_dim = len(self.target_state)
            target_qubits = int(np.log2(target_dim))

            # 2. Prevent linear algebra errors by halting execution on dimension mismatch
            if self.num_qubits != target_qubits:
                msg = f"Status: Dimension Mismatch (System: {self.num_qubits}Q, Target: {target_qubits}Q) ⚠️"
                self.status_banner.value = f"<h2 style='text-align: center; color: #f39c12;'>{msg}</h2>"
                return

            # 3. Calculate state fidelity within the equivalent Hilbert spaces
            sv_current = Statevector.from_instruction(self.circuit)
            sv_target = Statevector(self.target_state)

            if np.isclose(state_fidelity(sv_current, sv_target), 1.0, atol=1e-5):
                self.status_banner.value = "<h2 style='text-align: center; color: #27ae60;'>Status: Challenge Completed! 🎉</h2>"
            else:
                self.status_banner.value = "<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>"

        except Exception as e:
            # Divert unexpected mathematical errors to the UI console rather than silently failing
            with self.console:
                print(f"Fidelity Evaluation Error: {type(e).__name__}: {str(e)}")