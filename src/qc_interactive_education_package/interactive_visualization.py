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
from collections import OrderedDict

from .visualization import DimensionalCircleNotation, CircleNotation
from .dim_Bloch_spheres import SphereNotation

# --- Global Visualization Registry ---
VISUALIZATION_REGISTRY = {
    'Dimensional Circle Notation': DimensionalCircleNotation,
    'Circle Notation': CircleNotation,
    'Sphere Notation': SphereNotation
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

    def __init__(self, num_qubits=3, initial_state=None, preloaded_circuit=None, show_circuit=True):
        # Enforce strict maximum of 9 qubits to prevent exponential rendering latency
        self.num_qubits = min(num_qubits, 9)
        self.show_circuit = show_circuit
        self.render_figsize = (8.0, 6.0)

        # Persist the preloaded circuit to allow reconstruction upon reset
        self._preloaded_circuit = preloaded_circuit

        # Initialize primary and secondary (redo) tracking arrays
        self._circuit_history = []
        self._action_history = []
        self._redo_circuit_history = []
        self._redo_action_history = []

        # Decoupled LRU Caches to prevent state collisions and RAM overflow
        self._vis_cache = OrderedDict()
        self._circ_cache = OrderedDict()

        self.initial_state = self._normalize_state(initial_state) if initial_state is not None else None
        self._init_circuit()

        # --- UI Components (Controls) ---

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
        bloch_options = list(range(1, min(self.num_qubits, 3) + 1))
        self.bloch_qubit_dropdown = widgets.Dropdown(
            options=bloch_options,
            value=1,
            description='Focus Qubit:',
            layout={'display': 'none', 'width': '160px'}
        )

        # Dynamic Zoom Slider
        # Establish default zoom based on initial qubit count
        if self.num_qubits < 6:
            initial_zoom = 60
        else:
            initial_zoom = 100

        self.zoom_slider = widgets.IntSlider(
            value=initial_zoom,
            min=10, max=150, step=5,
            description='Zoom %:',
            layout={'width': '220px'}
        )

        self.gate_dropdown = widgets.Dropdown(options=['H', 'X', 'Y', 'Z', 'P', 'Rx', 'Ry', 'Rz'], value='H',
                                              description='Gate:', layout={'width': '180px'})
        self.controlled_checkbox = widgets.Checkbox(value=False, description='Controlled', indent=False,
                                                    disabled=(self.num_qubits < 2),
                                                    layout={'width': '100px', 'margin': '0px 10px 0px 10px'})

        qubit_options = list(range(1, self.num_qubits + 1))

        self.target_selector = widgets.SelectMultiple(
            options=qubit_options, value=(1,), description='Target(s):',
            rows=self.num_qubits, layout={'width': '160px'}
        )
        self.control_selector = widgets.SelectMultiple(
            options=qubit_options, value=(2,) if self.num_qubits > 1 else tuple(),
            description='Control(s):', disabled=True,
            rows=self.num_qubits, layout={'width': '160px'}
        )
        # We drop the '(x pi)' from the description and disable the native readout
        self.angle_input = widgets.FloatSlider(value=0.5, min=-2.0, max=2.0, step=0.0625,
                                               description='Angle:', disabled=True,
                                               layout={'width': '190px'}, readout=False)

        # We inject a custom, reactive readout that supports Unicode symbology
        self.angle_readout = widgets.HTML(
            value="<div style='font-family: sans-serif; font-size: 14px; color: #95a5a6;'>0.500π</div>",
            layout={'width': '60px', 'margin': '0px 10px 0px -5px'})

        # --- Base Control Buttons ---
        self.apply_btn = widgets.Button(description="⚙️ Apply",
                                        layout=widgets.Layout(width='85px', height='32px', border='1px solid #2b5797',
                                                              border_radius='4px'))
        self.apply_btn.style.button_color = '#2d89ef';
        self.apply_btn.style.text_color = 'white';
        self.apply_btn.style.font_weight = 'bold'

        self.measure_btn = widgets.Button(description="💥 Measure",
                                          layout=widgets.Layout(width='100px', height='32px',
                                                                border='1px solid #8e44ad', border_radius='4px'))
        self.measure_btn.style.button_color = '#9b59b6';
        self.measure_btn.style.text_color = 'white';
        self.measure_btn.style.font_weight = 'bold'

        self.zero_phase_btn = widgets.Button(description="⊘ 0-Phase",
                                             layout=widgets.Layout(width='95px', height='32px',
                                                                   border='1px solid #d37c15', border_radius='4px'))
        self.zero_phase_btn.style.button_color = '#f39c12';
        self.zero_phase_btn.style.text_color = 'white';
        self.zero_phase_btn.style.font_weight = 'bold'

        self.undo_btn = widgets.Button(description="↩ Undo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.undo_btn.style.button_color = '#95a5a6';
        self.undo_btn.style.text_color = 'white';
        self.undo_btn.style.font_weight = 'bold'

        self.redo_btn = widgets.Button(description="↪ Redo",
                                       layout=widgets.Layout(width='80px', height='32px', border='1px solid #7f8c8d',
                                                             border_radius='4px'))
        self.redo_btn.style.button_color = '#95a5a6';
        self.redo_btn.style.text_color = 'white';
        self.redo_btn.style.font_weight = 'bold'

        self.reset_btn = widgets.Button(description="🔄 Reset",
                                        layout=widgets.Layout(width='85px', height='32px', border='1px solid #b91d47',
                                                              border_radius='4px'))
        self.reset_btn.style.button_color = '#ee1111';
        self.reset_btn.style.text_color = 'white';
        self.reset_btn.style.font_weight = 'bold'

        # --- System Size Controls ---
        self.attach_btn = widgets.Button(description="➕ Attach",
                                         layout=widgets.Layout(width='95px', height='32px', border='1px solid #27ae60',
                                                               border_radius='4px'))
        self.attach_btn.style.button_color = '#2ecc71';
        self.attach_btn.style.text_color = 'white';
        self.attach_btn.style.font_weight = 'bold'

        self.detach_btn = widgets.Button(description="➖ Detach",
                                         layout=widgets.Layout(width='95px', height='32px', border='1px solid #c0392b',
                                                               border_radius='4px'))
        self.detach_btn.style.button_color = '#e74c3c';
        self.detach_btn.style.text_color = 'white';
        self.detach_btn.style.font_weight = 'bold'

        # --- Base State Inspector & Extraction ---
        self.state_inspector = widgets.HTML(layout={'width': '100%', 'margin': '10px 0px 5px 0px'})

        # Expanded width to 130px to safely accommodate the injected Unicode glyphs
        btn_layout = widgets.Layout(width='120px', height='32px', border_radius='4px')

        self.show_array_btn = widgets.Button(description="📋 Raw Array", layout=btn_layout)
        self.show_array_btn.style.button_color = '#34495e';
        self.show_array_btn.style.text_color = 'white';
        self.show_array_btn.style.font_weight = 'bold';
        self.show_array_btn.layout.border = '1px solid #2c3e50'

        self.export_png_btn = widgets.Button(description="🖼️ State PNG", layout=btn_layout)
        self.export_png_btn.style.button_color = '#1abc9c';
        self.export_png_btn.style.text_color = 'white';
        self.export_png_btn.style.font_weight = 'bold';
        self.export_png_btn.layout.border = '1px solid #16a085'

        self.export_svg_btn = widgets.Button(description="📐 State SVG", layout=btn_layout)
        self.export_svg_btn.style.button_color = '#2ecc71';
        self.export_svg_btn.style.text_color = 'white';
        self.export_svg_btn.style.font_weight = 'bold';
        self.export_svg_btn.layout.border = '1px solid #27ae60'

        self.export_circ_png_btn = widgets.Button(description="🔌 Circ PNG", layout=btn_layout)
        self.export_circ_png_btn.style.button_color = '#3498db';
        self.export_circ_png_btn.style.text_color = 'white';
        self.export_circ_png_btn.style.font_weight = 'bold';
        self.export_circ_png_btn.layout.border = '1px solid #2980b9'

        self.export_circ_svg_btn = widgets.Button(description="📐 Circ SVG", layout=btn_layout)
        self.export_circ_svg_btn.style.button_color = '#2980b9';
        self.export_circ_svg_btn.style.text_color = 'white';
        self.export_circ_svg_btn.style.font_weight = 'bold';
        self.export_circ_svg_btn.layout.border = '1px solid #1c5980'

        # --- Automatic Environment Detection ---
        is_voila = 'voila' in os.environ.get('SERVER_SOFTWARE', '').lower()
        active_buttons = [self.show_array_btn]
        if is_voila:
            active_buttons.extend([self.export_png_btn, self.export_svg_btn, self.export_circ_png_btn, self.export_circ_svg_btn])

        self.extraction_buttons_row = widgets.Box(
            active_buttons,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='center',
                align_items='center',
                width='100%',
                margin='5px 0px 15px 0px',
                grid_gap='10px'
            )
        )
        self.bottom_section = widgets.VBox([self.state_inspector, self.extraction_buttons_row], layout={'width': '100%'})

        # --- Output Canvases ---

        # Use object_fit='contain' to guarantee aspect ratio is preserved.
        # Remove min_height so the browser can shrink the image vertically as needed.
        self.image_widget = widgets.Image(
            format='png',
            layout=widgets.Layout(
                width=f"{initial_zoom}%",
                object_fit='contain',
                margin='0px'  # Removed vertical margin
            )
        )

        self.circuit_image_widget = widgets.Image(
            format='png',
            layout=widgets.Layout(
                max_width='100%',
                object_fit='contain',
                margin='0px',  # Removed vertical margin
                display='block' if self.show_circuit else 'none'
            )
        )
        self.console = widgets.Output(layout={'border': '1px solid #ccc', 'width': '100%'})

        # --- Event Binding ---
        self.vis_dropdown.observe(self._on_vis_change, names='value')
        self.bloch_qubit_dropdown.observe(self._on_vis_change, names='value')
        self.zoom_slider.observe(self._on_zoom_change, names='value') # Bind zoom event
        self.angle_input.observe(self._sync_angle_readout, names='value')

        self.attach_btn.on_click(self._attach_qubit)
        self.detach_btn.on_click(self._detach_qubit)
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
        self.controls_header = widgets.Box(
            [self.vis_dropdown, self.bloch_qubit_dropdown, self.zoom_slider],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='center',
                align_items='center',
                width='100%',
                margin='0px 0px 15px 0px',
                grid_gap='15px'
            )
        )

        self.controls_top = widgets.Box(
            [self.gate_dropdown, self.controlled_checkbox, self.control_selector, self.target_selector],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='center',
                align_items='center',
                width='100%',
                margin='0px 0px 10px 0px',
                grid_gap='10px'
            )
        )

        self.controls_bottom = widgets.Box(
            [self.angle_input, self.angle_readout, self.apply_btn, self.measure_btn, self.zero_phase_btn,
             self.undo_btn, self.redo_btn, self.attach_btn, self.detach_btn, self.reset_btn],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='center',
                align_items='center',
                width='100%',
                grid_gap='5px'
            )
        )

        # Responsive Flexbox Container for Visualizations
        self.visualization_row = widgets.Box(
            [self.circuit_image_widget, self.image_widget],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',  # Enforce strict vertical stacking
                align_items='center',
                justify_content='center',
                width='100%',
                grid_gap='0px'  # Obliterated gap space
            )
        )

        ui_elements = [
            self.controls_header,
            self.controls_top,
            self.controls_bottom,
            self.visualization_row,
            self.bottom_section,
            self.console
        ]

        self.ui = widgets.VBox(ui_elements, layout={'align_items': 'center', 'width': '100%'})

        if preloaded_circuit is not None:
            self._load_timeline(preloaded_circuit)

        self._update_plot()

    def _on_zoom_change(self, change):
        """Instantly scales the visualization image via CSS width manipulation."""
        self.image_widget.layout.width = f"{change.new}%"
        # If running inside ChallengeViewer, the target image shares the layout
        # and will automatically scale down as well.

    def _get_active_vis_class(self):
        default_class = next(iter(VISUALIZATION_REGISTRY.values()), None)
        return VISUALIZATION_REGISTRY.get(self.vis_dropdown.value, default_class)

    def _on_vis_change(self, change):
        if self.vis_dropdown.value == 'Sphere Notation':
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
                print(f"Timeline Error: Provided circuit has {qc.num_qubits} qubits, but viewer expects {self.num_qubits}.")
            return

        forward_circs = []
        forward_actions = []
        temp_circ = self.circuit.copy()

        for instruction in qc.data:
            op = instruction.operation
            name = op.name.capitalize()

            # INJECTION: Skip initialization instructions so they do not bloat the UI history stack
            if op.name == 'initialize':
                continue

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
        is_disabled = (change.new not in ['P', 'Rx', 'Ry', 'Rz'])
        self.angle_input.disabled = is_disabled

        # Instantly update the custom readout color to match the disabled state
        color = "#95a5a6" if is_disabled else "#2c3e50"
        self.angle_readout.value = f"<div style='font-family: sans-serif; font-size: 14px; color: {color};'>{self.angle_input.value:.3f}π</div>"

    def _sync_angle_readout(self, change):
        # Dynamically inject the pi symbol as the slider is dragged
        color = "#95a5a6" if self.angle_input.disabled else "#2c3e50"
        self.angle_readout.value = f"<div style='font-family: sans-serif; font-size: 14px; color: {color};'>{change.new:.3f}π</div>"

    def _toggle_control_selector(self, change):
        self.control_selector.disabled = not change.new

    def _refresh_ui_selectors(self):
        new_options = list(range(1, self.num_qubits + 1))

        # 1. Update the available options
        self.target_selector.options = new_options
        self.control_selector.options = new_options

        # 2. NEW: Dynamically expand/contract the box height to prevent scrollbars
        self.target_selector.rows = self.num_qubits
        self.control_selector.rows = self.num_qubits

        # 3. Reset the selection values
        self.target_selector.value = (1,)
        self.control_selector.value = tuple() if self.num_qubits < 2 else (2,)

        self.controlled_checkbox.disabled = (self.num_qubits < 2)

        bloch_options = list(range(1, min(self.num_qubits, 3) + 1))
        if self.bloch_qubit_dropdown.value not in bloch_options:
            self.bloch_qubit_dropdown.value = 1
        self.bloch_qubit_dropdown.options = bloch_options

        # Proactively lock the system size controls
        self.attach_btn.disabled = (self.num_qubits >= 9)
        self.detach_btn.disabled = (self.num_qubits <= 1)

    def _attach_qubit(self, b):
        with self.console:
            self.console.clear_output()

            if self.num_qubits >= 9:
                print("Hardware Limit Reached: Cannot exceed 9 qubits (512-dimensional Hilbert space).")
                return

            try:
                sv_current = Statevector.from_instruction(self.circuit).data
                sv_new = np.kron(np.array([1.0, 0.0], dtype=complex), sv_current)
                self._redo_circuit_history.clear()
                self._redo_action_history.clear()
                self._circuit_history.append(self.circuit.copy())
                self._action_history.append(f"Attach Qubit {self.num_qubits + 1} in |0⟩")
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
            if not self.target_selector.value:
                target_ui = self.num_qubits
            else:
                target_ui = self.target_selector.value[0]
            k = target_ui - 1
            try:
                sv_current = Statevector.from_instruction(self.circuit).data
                tensor = sv_current.reshape((2,) * self.num_qubits)
                axis = self.num_qubits - 1 - k
                slices = [slice(None)] * self.num_qubits
                slices[axis] = 0
                sv_projected = tensor[tuple(slices)].flatten()
                norm = np.linalg.norm(sv_projected)
                if np.isclose(norm, 0.0):
                    print(f"Mathematical Error: Qubit {target_ui} is deterministically in state |1⟩. Projecting it to |0⟩ yields a null state. Please apply an X-gate before detaching.")
                    return
                sv_new = sv_projected / norm
                self._redo_circuit_history.clear()
                self._redo_action_history.clear()
                self._circuit_history.append(self.circuit.copy())
                self._action_history.append(f"Detach Qubit {target_ui} (Traced out via |0⟩ projection)")
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
        self._redo_circuit_history.clear()
        self._redo_action_history.clear()
        self._circuit_history.append(self.circuit.copy())
        self._action_history.append(action_desc)

        try:
            sv_current = Statevector.from_instruction(self.circuit)
            outcome_str, sv_collapsed = sv_current.measure(targets)
            from qiskit.circuit.library import UnitaryGate
            import numpy as np
            import math
            from fractions import Fraction

            measure_gate = UnitaryGate(np.eye(2), label="M")
            measure_gate.name = "M"
            for t in targets:
                self.circuit.append(measure_gate, [t])
            self.circuit.initialize(sv_collapsed.data, self.circuit.qubits)
            results = [f"Qubit {t + 1}: {bit}" for t, bit in zip(targets, reversed(outcome_str))]
            self._action_history[-1] = f"{action_desc} 💥 Outcome: [{', '.join(results)}]"

            # Print standard measurement results
            for res in results:
                print(res)

            # ==========================================
            # QUANTUM TELEPORTATION INTELLIGENT PARSER
            # ==========================================
            # Trigger if N=3 and Alice measures both her qubits (1 and 2 in the UI)
            if self.num_qubits == 3 and set(targets_ui) == {1, 2}:
                # Verify Teleportation setup: Only Qubit 1 (index 0) should have an initial superposition
                is_teleportation = True
                for i in range(2, 8):
                    if np.abs(self.initial_state[i]) > 1e-6:
                        is_teleportation = False
                        break

                if is_teleportation:
                    alpha = complex(self.initial_state[0])
                    beta = complex(self.initial_state[1])

                    print(f"\n--- 🌌 Quantum Teleportation Analysis ---")

                    def fmt_amp(c):
                        real = np.real(c)
                        imag = np.imag(c)
                        if np.abs(imag) < 1e-5: return f"{real:.3f}"
                        if np.abs(real) < 1e-5: return f"{imag:.3f}i"
                        sign = "+" if imag >= 0 else "-"
                        return f"({real:.3f} {sign} {np.abs(imag):.3f}i)"

                    orig_state_str = f"{fmt_amp(alpha)}|0⟩ + {fmt_amp(beta)}|1⟩"
                    print(f"Original Random State (Alice's Qubit 1): {orig_state_str}")

                    # Extract Bob's state from the collapsed statevector using bitwise index masking
                    # In Qiskit's little-endian system, Bob's Qubit is q2 (the most significant bit, value 4)
                    sv_data = sv_collapsed.data
                    bob_alpha, bob_beta = 0.0j, 0.0j

                    for idx, amp in enumerate(sv_data):
                        if np.abs(amp) > 1e-6:
                            if (idx & 4) == 0:  # MSB (Qubit 3) is 0
                                bob_alpha = amp
                            else:  # MSB (Qubit 3) is 1
                                bob_beta = amp

                    # Mathematically align the global phase for a clean visual UI comparison
                    if np.abs(alpha) > 1e-6 and np.abs(bob_alpha) > 1e-6:
                        phase_shift = np.angle(alpha) - np.angle(bob_alpha)
                        bob_alpha *= np.exp(1j * phase_shift)
                        bob_beta *= np.exp(1j * phase_shift)
                    elif np.abs(beta) > 1e-6 and np.abs(bob_beta) > 1e-6:
                        phase_shift = np.angle(beta) - np.angle(bob_beta)
                        bob_alpha *= np.exp(1j * phase_shift)
                        bob_beta *= np.exp(1j * phase_shift)

                    bob_state_str = f"{fmt_amp(bob_alpha)}|0⟩ + {fmt_amp(bob_beta)}|1⟩"
                    print(f"Teleported State (Bob's Qubit 3):      {bob_state_str}")

                    # Exact Mathematical Fidelity Check
                    orig_vec = np.array([alpha, beta])
                    bob_vec = np.array([bob_alpha, bob_beta])
                    fidelity = np.abs(np.vdot(orig_vec, bob_vec)) ** 2

                    if np.isclose(fidelity, 1.0):
                        print("🎉 Success! The mathematical fidelity is 100%. Alice's state was teleported.")
                    else:
                        print("⚠️ Mismatch detected. The deferred measurement gates (CX, CZ) may be missing.")

            # ==========================================
            # SHOR'S ALGORITHM INTELLIGENT PARSER
            # ==========================================
            # Trigger only if N=8 and the 4 counting qubits are targeted
            if self.num_qubits == 8 and set(targets_ui) == {1, 2, 3, 4}:
                # Reconstruct the exact binary string from the Qiskit measurement ordering
                measured_bin = "".join(reversed(outcome_str))
                measured_dec = int(measured_bin, 2)
                phase = measured_dec / 16.0  # 2^4 = 16 basis states in the counting register

                print(f"\n--- 🔍 Shor's Algorithm Analysis (N=15, a=7) ---")
                print(f"Counting Register Measured: |{measured_bin}⟩ (Decimal: {measured_dec})")
                print(f"Calculated Phase: {measured_dec} / 16 = {phase}")

                if measured_dec == 0:
                    print("Result: Trivial phase (0). The period cannot be extracted.")
                    print("Action: Click 'Undo' and measure again to collapse into a non-trivial eigenstate.")
                else:
                    # Apply continued fractions to extract the period (r)
                    frac = Fraction(phase).limit_denominator(15)
                    r = frac.denominator
                    print(f"Continued Fraction: {frac.numerator}/{r}  ➔  Period (r) = {r}")

                    if r % 2 != 0:
                        print("Result: The extracted period is odd. The arithmetic fails.")
                        print("Action: Click 'Undo' and measure again.")
                    else:
                        a = 7
                        N = 15
                        # Calculate factors using the classically efficient gcd operation
                        factor1 = math.gcd(a ** (r // 2) - 1, N)
                        factor2 = math.gcd(a ** (r // 2) + 1, N)

                        print(f"Factor 1: gcd({a}^{r // 2} - 1, {N}) = {factor1}")
                        print(f"Factor 2: gcd({a}^{r // 2} + 1, {N}) = {factor2}")

                        # Validate the extracted factors against the target modulus
                        if (factor1 * factor2 == N) or (factor1 in [3, 5] and factor2 in [3, 5]):
                            print("\n🎉 Success! The integer 15 has been successfully factored into 3 and 5.")
                        else:
                            print("\nResult: Trivial factors found. Click 'Undo' and measure again.")

            self._update_plot()

        except Exception as e:
            self._circuit_history.pop()
            self._action_history.pop()
            with self.console:
                print(f"Measurement Error: {type(e).__name__}: {str(e)}")

        except Exception as e:
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
                self._circuit_history.pop()
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
            if self._preloaded_circuit is not None:
                self._load_timeline(self._preloaded_circuit)
            self._update_plot()
            self.console.clear_output()
        except Exception as e:
            with self.console:
                print(f"Reset Error: {str(e)}")

    def _get_drawable_circuit(self, circ: QuantumCircuit) -> QuantumCircuit:
        drawable_circ = circ.copy()
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
                if self.vis_dropdown.value == 'Sphere Notation':
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
                if self.vis_dropdown.value == 'Sphere Notation':
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
                drawable_circ = self._get_drawable_circuit(self.circuit)
                fig = drawable_circ.draw(output='mpl')
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
        self.undo_btn.disabled = not bool(self._circuit_history)
        self.redo_btn.disabled = not bool(self._redo_circuit_history)
        with self.console:
            try:
                # ==========================================
                # PHASE 1: Update the Dirac Notation DOM
                # ==========================================
                html_content = "<div style='text-align: left; font-family: monospace; font-size: 14px; max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 4px;'>"

                # 1a. Print historical states
                for i, past_circ in enumerate(self._circuit_history):
                    sv = Statevector.from_instruction(past_circ)
                    dirac_str = self._format_dirac_notation(sv.data)
                    if i == 0:
                        html_content += f"<div style='margin-bottom: 5px;'><b>Initial State:</b> {dirac_str}</div>"
                    else:
                        html_content += f"<div style='margin-bottom: 5px;'><b>{self._action_history[i - 1]}</b> &#8594; {dirac_str}</div>"

                # 1b. Calculate and print the current active state
                sv_current = Statevector.from_instruction(self.circuit)
                current_dirac = self._format_dirac_notation(sv_current.data)

                if not self._circuit_history:
                    html_content += f"<div style='margin-bottom: 5px;'><b>Initial State:</b> {current_dirac}</div>"
                else:
                    html_content += f"<div style='margin-bottom: 5px;'><b>{self._action_history[-1]}</b> &#8594; {current_dirac}</div>"

                html_content += "</div>"
                self.state_inspector.value = html_content

                # ==========================================
                # PHASE 2: Decoupled LRU Caching Pipeline
                # ==========================================

                # --- A. Resolve State Visualization ---
                # Key strictly depends on the math and the dropdowns
                stable_tensor_bytes = np.round(sv_current.data, decimals=5).tobytes()
                vis_cache_key = (
                    hash(stable_tensor_bytes),
                    self.vis_dropdown.value,
                    self.bloch_qubit_dropdown.value
                )

                if vis_cache_key in self._vis_cache:
                    # LRU Hit: Refresh position and use cached bytes
                    vis_bytes = self._vis_cache.pop(vis_cache_key)
                    self._vis_cache[vis_cache_key] = vis_bytes
                    self.image_widget.value = vis_bytes
                else:
                    # Cache Miss: Generate Matplotlib graphic
                    vis_class = self._get_active_vis_class()
                    if self.vis_dropdown.value == 'Sphere Notation':
                        vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
                    else:
                        vis = vis_class.from_qiskit(self.circuit)

                    with plt.rc_context({'figure.figsize': self.render_figsize, 'savefig.dpi': 300}):
                        b64_str = vis.exportBase64(formatStr='png')

                    vis_bytes = base64.b64decode(b64_str)
                    self.image_widget.value = vis_bytes

                    # Store and enforce 50-item LRU limit
                    self._vis_cache[vis_cache_key] = vis_bytes
                    if len(self._vis_cache) > 50:
                        self._vis_cache.popitem(last=False)

                # --- B. Resolve Circuit Diagram ---
                if self.show_circuit:
                    # Key strictly depends on exact gate history and future redo state
                    circ_id = hash(str(self.circuit.data))
                    redo_id = hash(str(self._redo_circuit_history[0].data)) if self._redo_circuit_history else None
                    circ_cache_key = (circ_id, redo_id)

                    if circ_cache_key in self._circ_cache:
                        # LRU Hit
                        circ_bytes = self._circ_cache.pop(circ_cache_key)
                        self._circ_cache[circ_cache_key] = circ_bytes
                        self.circuit_image_widget.value = circ_bytes
                    else:
                        # Cache Miss: Generate PIL Circuit
                        drawable_curr = self._get_drawable_circuit(self.circuit)
                        fig_curr = drawable_curr.draw(output='mpl', scale=0.4, style={'backgroundcolor': 'none'})
                        buf_curr = BytesIO()
                        fig_curr.savefig(buf_curr, format='png', bbox_inches='tight', dpi=300)
                        plt.close(fig_curr)

                        if not self._redo_circuit_history:
                            circ_bytes = buf_curr.getvalue()
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

                            def apply_vectorized_ghosting(img, alpha_multiplier=1.0):
                                arr = np.array(img)
                                white_mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
                                arr[white_mask, 3] = 0
                                if alpha_multiplier < 1.0:
                                    arr[~white_mask, 3] = (arr[~white_mask, 3] * alpha_multiplier).astype(np.uint8)
                                return Image.fromarray(arr)

                            img_curr_clean = apply_vectorized_ghosting(img_curr, alpha_multiplier=1.0)
                            img_fut_ghost = apply_vectorized_ghosting(img_fut, alpha_multiplier=0.5)

                            max_width = max(img_fut_ghost.width, img_curr_clean.width)
                            max_height = max(img_fut_ghost.height, img_curr_clean.height)
                            canvas = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))

                            y_fut = (max_height - img_fut_ghost.height) // 2
                            y_curr = (max_height - img_curr_clean.height) // 2

                            canvas.paste(img_fut_ghost, (0, y_fut), img_fut_ghost)
                            canvas.paste(img_curr_clean, (0, y_curr), img_curr_clean)

                            final_buf = BytesIO()
                            canvas.save(final_buf, format="PNG")
                            circ_bytes = final_buf.getvalue()

                        self.circuit_image_widget.value = circ_bytes

                        # Store and enforce 50-item LRU limit
                        self._circ_cache[circ_cache_key] = circ_bytes
                        if len(self._circ_cache) > 50:
                            self._circ_cache.popitem(last=False)

            except Exception as e:
                print("An error occurred during visualization generation:")
                traceback.print_exc()

            finally:
                # Prevent memory leaks
                plt.close('all')

    def show(self, show_circuit=None):
        if show_circuit is not None:
            self.show_circuit = show_circuit
        try:
            if self.show_circuit:
                drawable_circ = self._get_drawable_circuit(self.circuit)
                circ_fig = drawable_circ.draw(output='mpl', scale=0.4)
                circ_fig.suptitle("Quantum Circuit Pipeline")

            vis_class = self._get_active_vis_class()
            if self.vis_dropdown.value == 'Sphere Notation':
                vis = vis_class.from_qiskit(self.circuit, select_qubit=self.bloch_qubit_dropdown.value)
            else:
                vis = vis_class.from_qiskit(self.circuit)

            with plt.rc_context({'figure.figsize': self.render_figsize}):
                vis.draw()
                if hasattr(vis, 'fig') and vis.fig is not None:
                    vis.fig.suptitle(f"{self.vis_dropdown.value} Viewer")
                    plt.show(block=True)
                else:
                    print("Error: The visualization class failed to generate a Matplotlib 'fig'.")
        except Exception as e:
            with self.console:
                print(f"Standalone Render Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

    def display(self, figsize=None, ui_width=None, show_circuit=None):
        """Renders the unified application to the Jupyter DOM."""

        if figsize is not None:
            self.render_figsize = figsize
        if show_circuit is not None:
            self.show_circuit = show_circuit

        self.circuit_image_widget.layout.display = 'block' if self.show_circuit else 'none'
        self._update_plot()

        if ui_width is not None:
            self.image_widget.layout.width = ui_width

        # 5. Render the Polyfill and the parent Flexbox container
        from IPython.display import display as ipy_display, HTML

        ipy_display(HTML("""
        <script>
        window.MathJax = window.MathJax || {};
        window.MathJax.Hub = window.MathJax.Hub || {Queue: function(){}};
        </script>
        """))

        ipy_display(self.ui)


class ChallengeViewer(InteractiveViewer):
    """
    An assessment-driven subclass of InteractiveViewer.
    Evaluates the current state against a defined target state.
    Can be toggled into a purely comparative 'algorithm' mode.
    Dynamically resolves missing initial and target states to reduce instantiation boilerplate.
    """

    def __init__(self, num_qubits, initial_state=None, target_state=None, preloaded_circuit=None, show_circuit=True,
                 is_assessment=True):
        self.is_assessment = is_assessment

        # ==========================================
        # DYNAMIC STATE RESOLUTION PIPELINE
        # ==========================================

        # 1. Architecturally resolve the initial state
        # ==========================================
        # DYNAMIC STATE RESOLUTION PIPELINE
        # ==========================================

        # 1. Architecturally resolve the initial state
        if initial_state is None:
            if preloaded_circuit is not None:
                # Scan the circuit data for an explicit initialization vector
                for inst in preloaded_circuit.data:
                    if inst.operation.name == 'initialize':
                        initial_state = list(inst.operation.params)
                        break

            # If no initialization gate exists, mathematically enforce the absolute ground state
            if initial_state is None:
                dim = 2 ** num_qubits
                initial_state = [1.0] + [0.0] * (dim - 1)

        # 2. Architecturally resolve the target state
        if target_state is None:
            if preloaded_circuit is None:
                raise ValueError(
                    "Instantiation Error: A target_state or a preloaded_circuit must be provided to derive the final state.")

            # To guarantee 100% mathematical fidelity with the user's timeline, we replicate
            # the exact assembly pipeline used by InteractiveViewer._load_timeline.
            temp_qc = QuantumCircuit(num_qubits)
            temp_qc.initialize(initial_state, temp_qc.qubits)

            for inst in preloaded_circuit.data:
                # Strip pre-existing initializations to prevent amplitude overwriting
                if inst.operation.name != 'initialize':
                    temp_qc.append(inst)

            target_state = Statevector.from_instruction(temp_qc).data.tolist()

        # ==========================================

        if self.is_assessment:
            self.status_banner = widgets.HTML(
                "<h2 style='text-align: center; color: #e74c3c;'>Status: Incomplete ❌</h2>")

        shared_layout = widgets.Layout(width='100%', object_fit='contain', justify_content='center')
        self.target_image_widget = widgets.Image(format='png', layout=shared_layout)
        self._raw_target_state = target_state

        # Pass the dynamically resolved parameters to the parent class
        super().__init__(
            num_qubits=num_qubits,
            initial_state=initial_state,
            preloaded_circuit=preloaded_circuit,
            show_circuit=show_circuit
        )

        self.image_widget.layout = shared_layout
        self.render_figsize = (5.0, 4.0)

        self.target_state = self._normalize_state(self._raw_target_state)
        self._render_target()
        self._update_plot()

        # Dynamically assign the column header based on the execution context
        right_column_header = "Target State" if self.is_assessment else "Final State"

        # Force the comparison box to scale with the zoom slider
        self.comparison_box = widgets.HBox([
            # LEFT SIDE: Current State (Strict 50% mathematical partition)
            widgets.VBox(
                [widgets.HTML(
                    "<h3 style='text-align: center; width: 100%; color: #555; margin-bottom: 0px;'>Current State</h3>"),
                    self.image_widget],
                layout={'align_items': 'center', 'width': '50%', 'margin': '0px 5px', 'padding': '0px'}),

            # RIGHT SIDE: Target/Final State (Strict 50% mathematical partition)
            widgets.VBox(
                [widgets.HTML(
                    f"<h3 style='text-align: center; width: 100%; color: #555; margin-bottom: 0px;'>{right_column_header}</h3>"),
                    self.target_image_widget],
                layout={'align_items': 'center', 'width': '50%', 'margin': '0px 5px', 'padding': '0px'})

        ], layout={'width': f"{self.zoom_slider.value}%", 'justify_content': 'center', 'margin': '0px'})

        # Responsive Flexbox Container for Challenge Visualizations
        self.visualization_row = widgets.Box(
            [self.circuit_image_widget, self.comparison_box],
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',  # Strict vertical stacking
                align_items='center',
                justify_content='center',
                width='100%',
                grid_gap='0px'  # Obliterated gap space
            )
        )

        # Dynamically construct the DOM payload
        ui_elements = []
        if self.is_assessment:
            ui_elements.append(self.status_banner)

        ui_elements.extend([
            self.controls_header,
            self.controls_top,
            self.controls_bottom,
            self.visualization_row,
            self.bottom_section,
            self.console
        ])

        self.ui = widgets.VBox(ui_elements, layout={'align_items': 'center', 'width': '100%'})

        if self.is_assessment:
            self._check_success()

    def _on_vis_change(self, change):
        """Overrides the parent method to ensure the target state visual updates as well."""
        self._render_target()
        super()._on_vis_change(change)

    def _on_zoom_change(self, change):
        """Overrides the parent method to dynamically scale the outer container instead of the inner images."""
        if hasattr(self, 'comparison_box'):
            # 1. Scale the outer parent container to the slider's percentage
            self.comparison_box.layout.width = f"{change.new}%"

            # 2. lock the inner images to fill their 50% partitions
            self.image_widget.layout.width = '100%'
            self.target_image_widget.layout.width = '100%'
        else:
            # Fallback for events triggered during the super().__init__ boot sequence
            super()._on_zoom_change(change)

    def _render_target(self):
        qc_target = QuantumCircuit(self.num_qubits)
        qc_target.initialize(self.target_state, qc_target.qubits)
        try:
            # Route target generation through the polymorphic parser
            vis_class = self._get_active_vis_class()

            if self.vis_dropdown.value == 'Sphere Notation':
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
        # Only evaluate fidelity against the target matrix if assessment mode is active
        if getattr(self, 'is_assessment', False) and hasattr(self, 'status_banner'):
            self._check_success()

    def _check_success(self):
        # Architectural guard against the super().__init__ race condition
        if not hasattr(self, 'target_state'):
            return
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