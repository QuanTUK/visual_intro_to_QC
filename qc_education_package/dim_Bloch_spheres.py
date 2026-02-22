import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
# import matplotlib.cm as cm
import hsluv

from qc_education_package import Simulator, Visualization, DimensionalCircleNotation

import matplotlib
matplotlib.use('Tkagg')

class BlochSphere:
    def __init__(self,
                 bloch_radius=0.8,
                 rotation_angle=0,  # Used for inner sphere color mapping (phase).
                 vector_theta=0,
                 vector_phi=0,
                 outer_radius=1,
                 rotation_axis='z'):
        """
        Initialize the BlochSphere.
        """
        self.bloch_radius = bloch_radius
        self.rotation_angle = rotation_angle
        self.vector_theta = vector_theta
        self.vector_phi = vector_phi
        self.outer_radius = outer_radius
        self.rotation_axis = rotation_axis

    def plot(self, ax=None, figsize=(6, 6), offset=(0, 0, 0)):
        """
        Plot the Bloch sphere with everything rotated 90° clockwise about z.
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        dx, dy, dz = offset

        # Create spherical coordinates.
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        # --- Outer sphere (low zorder) ---
        x_outer = self.outer_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = self.outer_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = self.outer_radius * np.outer(np.ones_like(u), np.cos(v))

        # Rotate 90° CW: (x, y) -> (y, -x)
        x_outer, y_outer = y_outer, -x_outer

        ax.plot_surface(
            x_outer + dx, y_outer + dy, z_outer + dz,
            color='gray', alpha=0.01, edgecolor='none'
        )

        if self.bloch_radius > 1e-3:
            # --- Internal Bloch vector ---
            x_raw = self.bloch_radius * np.sin(self.vector_theta) * np.cos(self.vector_phi)
            y_raw = self.bloch_radius * np.sin(self.vector_theta) * np.sin(self.vector_phi)
            z_vec = self.bloch_radius * np.cos(self.vector_theta)

            # Rotate the vector too: (x, y) -> (y, -x)
            x_vec, y_vec = y_raw, -x_raw

            # --- Fixed coordinate axes (low zorder) ---
            axes = np.array([
                [self.outer_radius, 0, 0],
                [0, self.outer_radius, 0],
                [0, 0, self.outer_radius]
            ])

            # Draw the Bloch vector
            ax.quiver(
                dx, dy, dz,
                x_vec, y_vec, z_vec,
                color='#e31b4c', arrow_length_ratio=0.1, linewidth=2
            )
            ax.text(
                dx + 1.05 * x_vec, dy + 1.05 * y_vec, dz + 1.05 * z_vec,
                'v', color='#e31b4c', fontsize=12
            )

            # Draw X, Y, Z axes (rotated)
            rot_axes = np.array([[ay, -ax, az] for ax, ay, az in axes])

            for vec, label in zip(rot_axes, ('X', 'Y', 'Z')):
                ax.quiver(
                    dx, dy, dz,
                    vec[0], vec[1], vec[2],
                    color='black', arrow_length_ratio=0.1, alpha=0.5
                )
                ax.text(
                    dx + vec[0] * 1.1,
                    dy + vec[1] * 1.1,
                    dz + vec[2] * 1.1,
                    label, color='black', fontsize=12
                )

            # --- Inner sphere (high zorder, transparent) ---
            x_inner = self.bloch_radius * np.outer(np.cos(u), np.sin(v))
            y_inner = self.bloch_radius * np.outer(np.sin(u), np.sin(v))
            z_inner = self.bloch_radius * np.outer(np.ones_like(u), np.cos(v))

            # Rotate the inner sphere
            x_inner, y_inner = y_inner, -x_inner

            # --- HUSL COLOR MAPPING LOGIC ---
            # 1. Normalize angle to degrees [0, 360]
            #    We use the offset +5*pi/4 per your original logic to match alignment
            phase_radians = (self.rotation_angle + 5 * np.pi / 4) % (2 * np.pi)
            degrees = np.degrees(phase_radians)

            # 2. Convert HUSL to Hex
            #    H = degrees, S = 100 (saturation), L = 50 (lightness)
            inner_color = hsluv.hsluv_to_hex([degrees, 100, 50])

            ax.plot_surface(
                x_inner + dx, y_inner + dy, z_inner + dz,
                color=inner_color, alpha=0.3, edgecolor='none'
            )

        # Keep aspect ratio equal
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')  # Clean up axes if preferred

        return ax




def normalize_vector(v):
    """
    Normalize a complex vector so that the sum of the squared absolute values equals 1.

    Parameters:
        v (array-like): Input vector (can be a list or a NumPy array) with complex entries.

    Returns:
        np.ndarray: Normalized vector.

    Raises:
        ValueError: If the input vector is the zero vector.
    """
    # Convert the input to a NumPy array if it isn't already
    v = np.array(v, dtype=complex)

    # Compute the norm: sqrt(sum(|v_i|^2))
    norm = np.sqrt(np.sum(np.abs(v) ** 2))

    if norm == 0:
        return v, norm

    return (v / norm, norm)

# class MultiBlochSpheres2D:
#     def __init__(self, bloch_spheres, nrows=1, ncols=None, figsize=(12, 6), row_labels=None, col_labels=None):
#         """
#         Initialize with:
#          - bloch_spheres: List of BlochSphere instances.
#          - nrows: Number of rows in the grid.
#          - ncols: Number of columns in the grid (if None, set to len(bloch_spheres)).
#          - figsize: Overall figure size.
#          - row_labels: List of labels for each row.
#          - col_labels: List of labels for each column.
#         """
#         self.bloch_spheres = bloch_spheres
#         if ncols is None:
#             ncols = len(bloch_spheres)
#         self.nrows = nrows
#         self.ncols = ncols
#         self.figsize = figsize
#         if row_labels is None:
#             self.row_labels = [str(i) for i in range(nrows)]
#         else:
#             self.row_labels = row_labels
#         if col_labels is None:
#             self.col_labels = [str(i) for i in range(ncols)]
#         else:
#             self.col_labels = col_labels
#
#     def plot(self):
#         fig, ax_array = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)
#         # Flatten ax_array into a list if it is not already.
#         if self.nrows * self.ncols == 1:
#             axes = [ax_array]
#         else:
#             axes = ax_array.flatten()
#
#         # Create subplots and plot each BlochSphere.
#         for i, bloch in enumerate(self.bloch_spheres):
#             bloch.plot(ax=axes[i])
#         # Hide any unused axes.
#         for j in range(i + 1, len(axes)):
#             axes[j].axis('off')
#
#         # Create a ScalarMappable for the HSV colormap with normalization from 0 to 2π.
#         norm = colors.Normalize(vmin=0, vmax=2 * np.pi)
#         sm = cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
#         sm.set_array([])
#
#         # Define ticks at π/2, π, 3π/2, and 2π.
#         ticks = [0,np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
#
#         # Create a dedicated axis for the horizontal colorbar positioned at the bottom.
#         cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
#         cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=ticks)
#         cbar.ax.set_xticklabels([r'$0$',r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#         cbar.set_label('Phase (rad)', fontsize=12)
#
#         plt.tight_layout(rect=[0, 0.1, 1, 1])
#         plt.show()
#         return fig

class DimensionalBlochSpheres(Visualization):
    """A Visualization subclass for the Dimensional Circle
    Notation (DCN) representation.
    """

    def __init__(self, simulator, select_qubit=1, parse_math=True, version=2):
        """Constructor for the Dimensional Circle Notation
        representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        print(f"Setting up DCN Visualization in version {version}.")

        self._params.update({
            'version': version,
            'labels_dirac': True if version == 1 else False,
            'color_edge': 'black',
            'color_bg': 'white',
            'color_fill': '#77b6baff',
            'color_phase': 'black',
            'color_cube': '#8a8a8a',
            'width_edge': .7,
            'width_phase': .7,
            'width_cube': .5,
            'width_textwidth': .1,
            'offset_registerLabel': 1.3,
            'offset_registerValues': .6,
            # Set default text sizes for visualization
            'textsize_register': 10 * 0.7 ** ((self._sim._n - 3) // 2),
            'textsize_magphase': 8 * 0.7 ** ((self._sim._n - 3) // 2),
            'textsize_axislbl': 10 * 0.7 ** ((self._sim._n - 3) // 2),
            'bloch_outer_radius': 1
        })

        # Set default arrow style
        self._arrowStyle = {
            "width": 0.03 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_width": 0.2 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_length": 0.3 * 0.7 ** ((self._sim._n - 3) // 2),
            "edgecolor": None,
            "facecolor": 'black',
        }
        # Set default text style
        self._textStyle = {
            "size": self._params["textsize_register"],
            "horizontalalignment": "center",
            "verticalalignment": "center",
        }
        self._plotStyle = {
            "color": 'black',
            "linewidth": 0.7 ** ((self._sim._n - 3) // 2),
            "linestyle": "solid",
            "zorder": 0.7 ** ((self._sim._n - 3) // 2),
        }
        # Create empty variables for later use
        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._bloch_values = None
        self.select_qubit = select_qubit
        self._lx, self._ly = None, None

    def draw(self):
        """Draw Dimensional Circle Notation representation of current
        simulator state.
        """
        # Setup pyplot figure
        self.fig = plt.figure(layout="compressed")
        plt.get_current_fig_manager().set_window_title(
            "Dimensional Bloch spheres"
        )
        self._ax = self.fig.gca()

        self._ax.set_aspect("equal")
        # Get arrays with magnitude and phase of the register
        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        # Get x, y components of the phase for drawing phase dial inside
        # circles
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        # Explicit positions for the qubits do not specify the bit-order
        # Bitorder can be changed by flipping the value, phase, label arrays
        self._axis_labels = np.arange(
            1, self._sim._n + 1)[:: self._params["bitOrder"]]

        # self._sim._n is amount of Qubits.
        amount_qubits = self._sim._n
        # select qubit to extract
        select_qubit = self.select_qubit
        register_as_vector = self._sim._register.flatten()
        # set distance
        d = 3.5

        # Get arrays with magnitude and phase of the register
        self._bloch_values = multi_complex_to_Bloch(amount_qubits, select_qubit, register_as_vector, bitorder = self._params["bitOrder"])

        # Check whether the input is between 1 and 9
        if not 0 < amount_qubits < 10 or not isinstance(amount_qubits, int):
            raise NotImplementedError(
                "Please enter a valid number between 1 and 9."
            )

        ### Hard coded visualization for 1-3 Qubits - Dynamically coded 4+ Qubits ###

        # Origin x and y coordinate and
        # length of a tick mark on the axis
        x, y, len_tick = -2, 7, .2

        # Set position of circles in DCN
        if amount_qubits >= 1:
            # 1+ Qubits:
            self._coords = np.array([[0, 1], [1, 1]], dtype=float)
            self._bloch_coords = np.array([[0.5,1]], dtype=float)
            # Set distance
            self._coords *= d
            self._bloch_coords *= d

            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                x_pos = x + 1
                y_pos = y - 2
                if amount_qubits == 2:
                    x_pos += -0.5
                    y_pos += 0.3
                elif amount_qubits > 2:
                    x_pos += -1.2
                    y_pos += 0.3
                self._ax.text(
                    x_pos + 1.2,  # TODO hier auch evtl auch mit x und y
                    y_pos + 0.3,
                    "Qubit 1",
                    **self._textStyle
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x_pos, y_pos, 2.3, 0, **self._arrowStyle)
            # DCN V2: different coordinate axes
            else:
                # Horizontal axis (x,y,dx,dy, **kwargs)
                if amount_qubits == 1:
                    self._ax.arrow(x + 0.5, y - 2, 6.3, 0, **self._arrowStyle)
                    y = 5
                elif amount_qubits == 2:
                    self._ax.arrow(x, y - 2, 6.5, 0, **self._arrowStyle)
                    y = 5
                else:
                    self._ax.arrow(x, y, 6.5, 0, **self._arrowStyle)

                tick_y = [y - len_tick, y + len_tick]
                # 1st tick on x axis
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,
                    **self._plotStyle
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2.5 * len_tick,
                    "0",
                    **self._textStyle,
                )
                # 2nd tick on x axis
                self._ax.plot(
                    [self._coords[0, 1], self._coords[0, 1]],
                    tick_y,
                    **self._plotStyle,
                )
                self._ax.text(
                    self._coords[0, 1],
                    y + 2.5 * len_tick,
                    "1",
                    **self._textStyle,
                )
                self._ax.text(
                    self._coords[0, 1] / 2,
                    y + 3 * len_tick,
                    "Qubit 1",
                    **self._textStyle,
                )
            if amount_qubits == 1:
                # Set axis limits
                self._ax.set_xlim([-1.6, 5.3])
                self._ax.set_ylim([2.3, 5.5])
        # 2+ Qubits:
        if amount_qubits >= 2:
            self._coords = np.concatenate((self._coords, np.array([[0, 0], [d, 0]])))
            if select_qubit == 1:
                self._bloch_coords = np.array([[0.5, 1],[0.5, 0]], dtype=float)
            else:
                self._bloch_coords = np.array([[0, 0.5],[1, 0.5]], dtype=float)

            self._bloch_coords *= d
            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                x_pos = x + 0.35
                y_pos = y - 2.75
                if amount_qubits > 2:
                    x_pos -= 0.7
                self._ax.text(
                    x_pos - 0.15,
                    y_pos,
                    "Qubit 2",
                    **self._textStyle,
                    rotation=90
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x_pos + 0.15, y_pos + 1.05, 0, -2.3, **self._arrowStyle)

            # DCN V2: different coordinate axes
            else:
                # Vertical axis
                if amount_qubits == 2:
                    self._ax.arrow(x, y, 0, -6, **self._arrowStyle)
                else:
                    self._ax.arrow(x, y, 0, -8, **self._arrowStyle)
                tick_x = [x - len_tick, x + len_tick]
                # 1st tick on y axis
                self._ax.plot(
                    tick_x,
                    [self._coords[0, 1], self._coords[0, 1]],
                    **self._plotStyle,
                )
                self._ax.text(
                    x - 2.5 * len_tick,
                    self._coords[0, 1],
                    "0",
                    **self._textStyle,
                    rotation=90
                )
                # 2nd tick on y axis
                self._ax.plot(
                    tick_x,
                    [self._coords[3, 1], self._coords[3, 1]],
                    **self._plotStyle,
                )
                self._ax.text(
                    x - 2.5 * len_tick,
                    self._coords[3, 1],
                    "1",
                    **self._textStyle,
                    rotation=90
                )
                self._ax.text(
                    x - 3 * len_tick,
                    self._coords[0, 1] / 2,
                    "Qubit 2",
                    **self._textStyle,
                    rotation=90,
                )
            if amount_qubits == 2:
                # Set axis limits
                self._ax.set_xlim([-2.8, 5])
                self._ax.set_ylim([-1.5, 5.8])

        # 3+ Qubits:
        if amount_qubits >= 3:
            # Double the array
            self._coords = np.concatenate((self._coords, self._coords))
            # Offset 3rd dim circles to the rear from position of the first 4 circles
            self._coords[4:] += d/2

            if select_qubit == 1 or select_qubit == 2:
                self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))
                self._bloch_coords[2:] += d/2
            else:
                self._bloch_coords = np.array([[0,d],[d,d],[0, 0], [d, 0]])
                self._bloch_coords += d/2

            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                self._ax.text(
                    x + 0.55,
                    y - 0.55,
                    "Qubit 3",
                    **self._textStyle,
                    rotation=45
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x - 0.2, y - 1.7, d/2-0.1, d/2-0.1, **self._arrowStyle)

            # DCN V2: different coordinate axes
            else:
                # Diagonal axis
                self._ax.arrow(x, y, d-0.2, d-0.2, **self._arrowStyle)
                len_tick_z = len_tick / np.sqrt(2)
                off1, off2 = 0.8, 2.2
                # 1st tick on z axis
                self._ax.plot(
                    [x + off1 + len_tick_z, x + off1 - len_tick_z],
                    [y + off1 - len_tick_z, y + off1 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x + off1 - 2.5 * len_tick_z,
                    y + off1 + 2.5 * len_tick_z,
                    "0",
                    **self._textStyle,
                    rotation=45
                )
                # 2nd tick on z axis
                self._ax.plot(
                    [x + off2 + len_tick_z, x + off2 - len_tick_z],
                    [y + off2 - len_tick_z, y + off2 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x + off2 - 2.5 * len_tick_z,
                    y + off2 + 2.5 * len_tick_z,
                    "1",
                    **self._textStyle,
                    rotation=45
                )
                middle_ticks = (off2 - off1) / 2
                self._ax.text(
                    x + off1 + (middle_ticks) - 5.5 * len_tick_z,
                    y + off1 + (middle_ticks) + 5.5 * len_tick_z,
                    "Qubit 3",
                    **self._textStyle,
                    rotation=45
                )
            if amount_qubits == 3:
                # Set axis limits
                if self._params['version'] == 1:
                    self._ax.set_ylim([-1.5, 7.5])
                else:
                    self._ax.set_ylim([-1.5, 10.8])
                self._ax.set_xlim([-4.8, 8.5])
        # 4+ Qubits:
        if amount_qubits >= 4:  # Setting up remaining qubits and axis labels for qubits 4+
            if select_qubit >= 4:
                orig_array = np.array([[0,d],[d,d],[0, 0], [d, 0]])
                self._bloch_coords = np.concatenate((orig_array, orig_array))
                self._bloch_coords[len(self._bloch_coords) // 2:] += d / 2

            iter_control = 0
            for i in range(4, amount_qubits + 1):
                quarter_axis_length = (2 ** int(i / 2))
                self._coords = np.concatenate((self._coords, self._coords))
                if select_qubit < 4:
                    self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))

                if select_qubit >= 4 and iter_control == 1:
                    self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))

                iter_control = 1

                if (i % 2 == 0):  # Horizontal axes
                    # Shift it along the X-axis
                    self._coords[len(self._coords) // 2:, 0] += 2 ** (i / 2 + 1)

                    if select_qubit < 4:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 0] += 2 ** (i / 2 + 1)
                    elif select_qubit % 2 == 0:
                        self._bloch_coords[:, 0] += 2 ** (i / 2)
                    elif select_qubit % 2 == 1:
                        self._bloch_coords[:, 1] -= 2 ** (i / 2)


                    self._ax.arrow(x, y + i, 4 * quarter_axis_length, 0, **self._arrowStyle)
                    self._ax.plot(  # |0> area
                        [x + quarter_axis_length / 6, x + quarter_axis_length * 1.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x + quarter_axis_length,
                        y + i + 2.5 * len_tick,
                        "0",
                        **self._textStyle,
                    )
                    self._ax.plot(  # |1> area
                        [x + 2.125 * quarter_axis_length, x + quarter_axis_length * 3.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        3 * quarter_axis_length - 2,
                        y + i + 2.5 * len_tick,
                        "1",
                        **self._textStyle,
                    )
                    self._ax.text(
                        2 * quarter_axis_length - 2,
                        y + i + 3 * len_tick,
                        f"Qubit {i}",
                        **self._textStyle,
                    )
                else:  # Vertical axes
                    # Shift it along the Y-axis
                    self._coords[len(self._coords) // 2:, 1] -= 2 ** ((i + 1) / 2)

                    if select_qubit < 4:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 1] -= 2 ** ((i + 1) / 2)

                    elif select_qubit % 2 == 0:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 1] -= 2 ** (i / 2 + 1)
                    elif select_qubit % 2 == 1:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 0] += 2 ** (i / 2 + 1)

                    x_pos = x + 3 - i
                    self._ax.arrow(x_pos, y, 0, -4 * quarter_axis_length, **self._arrowStyle)
                    self._ax.plot(  # |0> area
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - quarter_axis_length / 6, y - quarter_axis_length * 1.875],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x_pos - 2.5 * len_tick,
                        y - quarter_axis_length,
                        "0",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.plot(  # |1> area
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - 2.125 * quarter_axis_length, y - quarter_axis_length * 3.875],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x_pos - 2.5 * len_tick,
                        y - 3 * quarter_axis_length,
                        "1",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.text(
                        x_pos - 3 * len_tick,
                        y - 2 * quarter_axis_length,
                        f"Qubit {i}",
                        **self._textStyle,
                        rotation=90,
                    )
            # Set axis limits according to plot size (grows with n for 4+ Qubits)
            self._ax.set_xlim([x - 1 - 2 * ((amount_qubits - 3) // 2), x + 1 + 2 ** (amount_qubits // 2 + 2)])
            self._ax.set_ylim([y - 1 - 2 ** ((amount_qubits + 1) // 2 + 1), y + 1 + 2 * ((amount_qubits) // 2)])

        self._bloch_coords = self._bloch_coords[::self._params["bitOrder"]]
        # Draw all circles
        self._draw_all_spheres()

        self._ax.set_axis_off()

        # Flip axis labels if bitOrder is set to 1
        self._axis_labels = np.arange(1, amount_qubits + 1)[:: self._params["bitOrder"]]

        try:
            # --- PREPARE HUSL COLORMAP ---
            # We generate a list of RGB values for hues 0 to 360.
            # We use 256 steps for smoothness.
            # Saturation=100, Lightness=50 gives the standard vibrant HUSL look.
            husl_rgb_list = [hsluv.hsluv_to_rgb([h, 100, 50]) for h in np.linspace(0, 360, 256)]

            # Create the Matplotlib colormap object
            husl_cmap = ListedColormap(husl_rgb_list, name='husl_phase')

            # --- YOUR PLOTTING CODE ---

            # 1. Create the Polar Axes
            # [left, bottom, width, height]
            cbar_ax = self.fig.add_axes([0.89, 0.40, 0.1, 0.1], projection='polar')

            # 2. Create the data for the ring
            n_segments = 360
            theta = np.linspace(0, 2 * np.pi, n_segments)
            r = np.linspace(0.6, 1, 2)
            Theta, R = np.meshgrid(theta, r)

            # 3. Shift the Color Mapping
            # Formula: (Theta + Shift) % 2pi
            # The shift needed is 5pi/4.
            ColorVals = (Theta + 5 * np.pi / 4) % (2 * np.pi)

            # 4. Plot the color wheel
            # Use the custom 'husl_cmap' instead of 'plt.cm.hsv'
            mesh = cbar_ax.pcolormesh(Theta, R, ColorVals, cmap=husl_cmap, shading='auto', vmin=0, vmax=2 * np.pi)

            # 5. Configure Ticks
            cbar_ax.set_yticks([])  # Remove radial ticks

            tick_locs = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            cbar_ax.set_xticks(tick_locs)
            cbar_ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])

            # Remove outer spine
            cbar_ax.spines['polar'].set_visible(False)

            # 6. Label
            cbar_ax.text(0.5, 0.5, 'Phase\n(rad)',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=cbar_ax.transAxes, fontsize=11, fontweight='bold')

        except Exception as e:
            print(f"Error creating circular colorbar: {e}")

    def _draw_all_spheres(self):
        """Internal method to iterate through calculated coordinates and
        Bloch parameters, drawing each sphere using inset_axes.
        """
        if self._bloch_coords is None or self._bloch_values is None:
            print("Warning: Cannot draw spheres, coordinates or values not calculated.")
            return
        if len(self._bloch_coords) != len(self._bloch_values):
            print(f"Warning: Mismatch between number of coordinates ({len(self._bloch_coords)}) and values ({len(self._bloch_values)}). Skipping draw.")
            return

        print(f"Drawing {len(self._bloch_coords)} Bloch spheres using inset axes...")

        outer_radius = self._params['bloch_outer_radius']
        # Determine the approximate size needed for the inset axis in data coordinates
        # Make it slightly larger than the sphere diameter to include labels/padding
        inset_diameter_data = 2 * outer_radius * 2 # e.g., 30% larger than sphere radius for padding

        # Store inset axes if needed later (optional)
        self._inset_axes = []

        for i in range(len(self._bloch_coords)):
            # Target center coordinate (x, y) in the main axes data coordinates
            cx, cy = self._bloch_coords[i]
            bloch_params = self._bloch_values[i]   # (theta, phi, radius, angle)

            # --- Parameter Extraction ---
            try:
                radius = bloch_params[0]
                angle = bloch_params[1]
                theta = bloch_params[2]
                phi = bloch_params[3]
            except (IndexError, TypeError) as e:
                print(f"Error extracting parameters for sphere {i}: {e}. Using defaults.")
                radius, angle , theta, phi = 0, 0, 0, 0

            # --- Create Inset Axis ---
            # Calculate bottom-left corner for inset_axes in data coordinates
            bl_x = cx - inset_diameter_data / 2
            bl_y = cy - inset_diameter_data / 2

            # Create the inset axis at the calculated position and size (in data coords)
            # zorder places it above background but potentially below main axes lines/labels
            inset_ax = self._ax.inset_axes(
                [bl_x, bl_y, inset_diameter_data, inset_diameter_data],
                transform=self._ax.transData, # Use data coordinates for positioning
                zorder=2,
                projection = '3d'
            )
            self._inset_axes.append(inset_ax) # Store if needed

            # --- Create BlochSphere2D Instance ---
            sphere = BlochSphere(
                bloch_radius=radius,
                rotation_angle=angle,
                vector_theta=theta,
                vector_phi=phi,
                outer_radius=outer_radius
            )

            # --- Plot the Sphere onto the Inset Axis ---
            # Use global_offset=(0, 0) because the sphere should be centered *within* its own inset axis.
            # The inset_ax itself is already positioned correctly in the main plot.
            sphere.plot(
                ax=inset_ax
            )

            # Set the facecolor of the Axes object itself to transparent
            # Use an RGBA tuple or the string 'none' might work in newer matplotlib
            inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
            # Or alternatively try: inset_ax.set_facecolor('none')

            # Make the background panes transparent (keep this for clarity in 3D)
            pane_color = (1.0, 1.0, 1.0, 0.0)
            inset_ax.xaxis.set_pane_color(pane_color)
            inset_ax.yaxis.set_pane_color(pane_color)
            inset_ax.zaxis.set_pane_color(pane_color)

            # Also make the grid lines transparent (belt-and-suspenders with set_axis_off)
            # Note: Using internal '_axinfo' might be less stable across versions
            try:  # Wrap in try-except in case structure changes
                inset_ax.xaxis._axinfo["grid"]['color'] = pane_color
                inset_ax.yaxis._axinfo["grid"]['color'] = pane_color
                inset_ax.zaxis._axinfo["grid"]['color'] = pane_color
            except (KeyError, AttributeError):
                # If this fails, set_axis_off() should still hide them
                pass
            # inset_ax.set_frame_on(False) # Hide the bounding box of the inset axis

            inset_ax.set_axis_off() # Alternative way to hide everything


    # Helper function (needs implementation based on ordering)
    def _get_fixed_state_label(self, index, n_qubits, selected_qubit):
         """ Calculates the binary label for the fixed qubits corresponding to sphere index."""
         if n_qubits == 1: return ""
         num_fixed = n_qubits - 1
         # Assumes standard binary order for fixed qubits
         binary_format = "{:0" + str(num_fixed) + "b}"
         binary_string = binary_format.format(index)

         # Insert '-' for the selected qubit's position
         label_list = list(binary_string)
         label_list.insert(selected_qubit - 1, '-') # selected_qubit is 1-based
         return "|" + "".join(label_list) + ">"


def complex_to_bloch(vector):
    """
    Converts a qubit state given by two complex numbers (alpha, beta)
    to its Bloch sphere coordinates (theta, phi).

    The qubit is assumed to be in the form:
        |ψ⟩ = α|0⟩ + β|1⟩
    with normalization |α|^2 + |β|^2 = 1.

    The corresponding Bloch sphere parameters are defined as:
        α = cos(θ/2)
        β = e^(i φ) sin(θ/2)

    This function normalizes the input state, removes the global phase
    (by making α real and nonnegative), and then computes:
        θ = 2 arccos(|α|)
        φ = arg(β)

    Parameters:
        alpha (complex): The amplitude for |0⟩.
        beta (complex):  The amplitude for |1⟩.

    Returns:
        theta (float): The polar angle, in radians (0 ≤ θ ≤ π).
        phi (float): The azimuthal angle, in radians (0 ≤ φ < 2π).
    """
    vector, norm = normalize_vector(vector)

    # Ensure the inputs are of type complex
    alpha = complex(vector[0])
    beta = complex(vector[1])

    # Remove global phase: rotate so that alpha becomes real and nonnegative
    if alpha != 0:
        global_phase = np.angle(alpha)
    else:
        global_phase = np.angle(beta)

    alpha *= np.exp(-1j * global_phase)
    beta *= np.exp(-1j * global_phase)

    # Compute theta using the relation: alpha = cos(theta/2)
    # Clip to [0, 1] to avoid numerical issues.
    theta = 2 * np.arccos(np.clip(np.real(alpha), 0, 1))

    # Compute phi from beta = e^(i φ) sin(theta/2)
    phi = np.angle(beta)
    # Optionally convert phi from (-pi, pi] to [0, 2*pi)
    # if phi < 0:
        # phi += 2 * np.pi
    return norm, global_phase, theta, phi

def select_qubits(n,sel_qubit):
    reordered_list = []
    dif = 2**(sel_qubit-1)
    N = 2**(n-1)
    for i in range(2*N):
        if i not in reordered_list:
            reordered_list.append(i)
            reordered_list.append(i+dif)
    pairs = [[reordered_list[x*2],reordered_list[x*2+1]] for x in range(N)]
    return pairs


def multi_complex_to_Bloch(n,sel_qubit,vector, bitorder=1):
    if isinstance(vector, Simulator):
        vector = vector._register.flatten()
    vector = normalize_vector(vector)[0][::bitorder]
    multi_bloch = []
    for pair in select_qubits(n,sel_qubit):
        if vector[pair[0]] == 0 and vector[pair[1]] == 0:
            multi_bloch.append([0,0,0,0])
        else:
            norm, global_phase, theta, phi = complex_to_bloch([vector[pair[0]], vector[pair[1]]])
            multi_bloch.append([norm,global_phase,theta,phi])
    return multi_bloch[::bitorder]

# Example usage:
if __name__ == '__main__':
    # Create four BlochSphere instances with different rotations for a 2x2 grid.
    sel_qubit = 1
    vector_1 = [0,1]
    vector_2 = [1+1j,1j,1,0]
    vector_3 = [0, 1, 1, 0, 1, 0, 0, 0]
    vector_3_ = [1, 0, 0, 0, 0, 0, 0, 1]
    vector_4 = [0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1]
    vector_5 = [0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1]
    vector_6 = [0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1,0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,0, 0, 1j, 1, 0, 1, 0, 1]

    sim_1 = Simulator(1)
    sim_1.writeComplex(vector_1)
    sim_2 = Simulator(2)
    sim_2.writeComplex(vector_2)
    sim_3 = Simulator(3)
    sim_3.writeComplex(vector_3)
    sim_3_ = Simulator(3)
    sim_3_.writeComplex(vector_3_)
    sim_4 = Simulator(4)
    sim_4.writeComplex(vector_4)
    sim_4.had(4)

    sim_5 = Simulator(5)
    sim_5.writeComplex(vector_5)
    sim_6 = Simulator(6)
    sim_6.writeComplex(vector_6)

    Bloch_vis = DimensionalBlochSpheres(sim_6, select_qubit=sel_qubit)
    Bloch_vis.draw()
    Bloch_vis = DimensionalBlochSpheres(sim_6, select_qubit=sel_qubit)
    Bloch_vis.draw()

    DCN_vis = DimensionalCircleNotation(sim_3)
    DCN_vis.draw()

    plt.show()
