# ----------------------------------------------------------------------------
# Created By: Nikolas Longen, nlongen@rptu.de
# Modified By: Patrick Pfau, ppfau@rptu.de
# Reviewed By: Maximilian Kiefer-Emmanouilidis, maximilian.kiefer@rptu.de
# Created: March 2023
# Project: DCN QuanTUK
# ----------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# from matplotlib.textpath import TextPath
# from matplotlib.patches import PathPatch
from io import BytesIO
from base64 import b64encode
import numpy as np

# TODO: Implement logging


class Visualization:
    """Superclass for all visualizations of quantum computer states.
    This way all visualizations inherit export methods.
    Alle subclasses must implement/overwrite a draw method and should also
    overwrite the __init__ method.
    """

    def __init__(self, simulator, parse_math=True):
        """Constructor for Visualization superclass.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
        """
        self._sim = simulator
        self.fig = None
        self._lastState = None
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["text.parse_math"] = parse_math
        # common settings
        self._params = {
            "dpi": 300,
            "transparent": True,
            "showValues": False,
            "bitOrder": simulator._bitOrder,
        }

    def exportPNG(self, fname: str, title=""):
        """Export the current visualization as PNG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        self._export(fname, "png", title)

    def exportPDF(self, fname: str, title=""):
        """Export the current visualization as PDF file to given path.

        Args:
            fname (str): fname or path to export file to.
        """
        self._export(fname, "pdf", title)

    def exportSVG(self, fname: str, title=""):
        """Export the current visualization as SVG image to given path.

        Args:
            fname (str): fname or path to export image to.
        """
        mpl.rcParams["svg.fonttype"] = "none"  # Export as text and not paths
        self._export(fname, "svg", title)

    def exportBase64(self, formatStr="png"):
        """Exports the figure to a base64 string."""
        from io import BytesIO
        import base64

        self._redraw()
        buf = BytesIO()

        # Save to buffer
        self.fig.savefig(buf, format=formatStr, bbox_inches='tight', transparent=True)

        # CLEANUP: Close the figure so the user doesn't have to
        plt.close(self.fig)

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _exportBuffer(self, formatStr, title=""):
        """Export current visualization in format into IO buffer.

        Args:
            formatStr (str, optional): Format for image. Defaults to 'png'.

        Returns:
            BytesIO: returns view of buffer containing image using
            BytesIO.getbuffer()
        """
        buf = BytesIO()
        self._export(buf, formatStr, title)
        return buf.getbuffer()

    def _export(self, target: str, formatStr: str, title: str):
        """General export method to save current pyplot figure, so all all
        exports will share same form factor, res etc.

        Args:
            target (plt.savefig compatible object): Target to save pyplot
            figure to.
            formatStr (str, optional): Format for image. Defaults to 'png'.
        """
        self._redraw()
        self.fig.suptitle(title)
        self.fig.savefig(
            target,
            format=formatStr,
            bbox_inches="tight",
            pad_inches=0,
            dpi=self._params["dpi"],
            transparent=self._params["transparent"],
        )

    def show(self):
        """Method to show current figure. Checks for Jupyter environment
        to ensure inline rendering.
        """
        self._redraw()

        try:
            # explicit import to check if we are in a notebook environment
            from IPython.display import display
            # Render the figure object directly inline
            display(self.fig)
            # Close the figure to prevent Double-Rendering
            # (standard inline backend would try to plot it again at cell end)
            plt.close(self.fig)
        except ImportError:
            # Fallback for standard Python scripts (non-notebook)
            plt.show()

    def _redraw(self):
        """Checks if simulator state is changed and redraws the image if so."""
        if self._lastState != self._sim:
            self._lastState = self._sim.copy()
            self.draw()

    def showMagnPhase(self, show_values: bool):
        """Switch showing magnitude and phase of each product state in a
        register

        Args:
            show_values (bool): Show value fir true, else do not show
        """
        self._params.update({"showValues": show_values})

    @classmethod
    def from_qiskit(cls, qiskit_obj):
        """
        Factory method to create a visualization instance from a Qiskit object.
        Accepts either a QuantumCircuit or a Statevector.

        Args:
            qiskit_obj: A qiskit.QuantumCircuit or qiskit.quantum_info.Statevector.

        Returns:
            Visualization: An initialized instance of the visualization class.
        """
        # Import necessary Qiskit tools inside the method
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        # Robust import for the Simulator
        try:
            from .simulator import Simulator
        except ImportError:
            from simulator import Simulator

        # 1. Handle Input Type
        if isinstance(qiskit_obj, QuantumCircuit):
            # If it's a circuit, simulate it to get the statevector
            sv = Statevector.from_instruction(qiskit_obj)
        elif isinstance(qiskit_obj, Statevector):
            # If it's already a statevector, use it directly
            sv = qiskit_obj
        else:
            raise TypeError("Input must be a Qiskit QuantumCircuit or Statevector.")

        # 2. Initialize Custom Simulator
        # sv.num_qubits works for both circuit-derived and raw statevectors
        sim = Simulator(sv.num_qubits)

        # 3. Load the complex amplitudes
        sim.writeComplex(sv.data)

        # 4. Return the Visualization Instance
        return cls(sim)

    def draw(self):
        # TODO: Add style guide for draw method
        pass

    def _createLabel(self, number: int):
        """Creates a binary label for a given index with zero padding fitting
        to the number of qubits.

        Args:
            number (int): Number to convert

        Returns:
            str: binary label fot the given number in braket
        """
        # NOTE: width is deprecated since numpy 1.12.0
        return np.binary_repr(number, width=self._sim._n)

    def hist(self, qubit=None, size=100
             ) -> tuple[np.array, mpl.figure, mpl.axes.Axes]:
        """Create a histogram plot for repeated measurements of the simulator
        state. Here the state of the simulator will not collaps after a
        measurement. Arguments are passed to simulator.read(). If no qubit is
        given (qubit=None) all qubit are measured.

        Args:
            qubit (int or list(int), optional): qubit to read.
            Defaults to None.
            size (int), optional): Repeat the measurement size times.
            Default 1 Measurement.

        Returns:
            (np.array, mpl.figure, mpl.axes.Axes): Measurement results and
            pyplot figure and axes of the histogram plot to further manipulate
            if needed
        """
        _, result = self._sim.read(qubit, size)
        histFig = plt.figure(0)
        # plt.get_current_fig_manager().set_window_title("Histogram plot")
        ax = histFig.subplots()
        ax.hist(result, density=True)
        ax.set_xlabel("Measured state")
        ax.set_ylabel("N")
        ax.set_title(
            f"Measured all qubits {size} times."
            if qubit is None
            else f"Measured qubit {qubit} {size} times."
        )
        return result, histFig, ax


class CircleNotation(Visualization):
    """A Visualization subclass for the well known Circle Notation
    representation.
    """

    def __init__(self, simulator, **kwargs):
        """Constructor for the Circle Notation representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
            cols (int, optional): Arrange Circle Notation into a set of
            columns. Defaults to None.
        """
        super().__init__(simulator)  # Execute constructor of superclass

        self._params.update(
            {
                "color_edge": "black",
                "color_fill": "#77b6ba",
                "color_phase": "black",
                "width_edge": 0.7,
                "width_phase": 0.7,
                "textsize_register": 10,
                "textsize_magphase": 8,
                "dist_circles": 3,
                "offset_registerLabel": -1.35,
                "offset_registerValues": -2.3,
            }
        )

        self.fig = None

    def draw(self, cols=8): # TODO wirklich 8?
        """Draw Circle Notation representation of current simulator state."""
        if self._sim._n > 6:
            raise NotImplementedError(
                "Circle notation is only implemented for up to 6 qubits."
            )
        self._cols = cols if cols is not None else 2**self._sim._n
        circles = 2**self._sim._n
        self._c = self._params["dist_circles"]
        x_max = self._c * self._cols 
        y_max = self._c * circles / self._cols if circles > self._cols else self._c
        y_max *= 1 if not self._params["showValues"] else 1.5
        xpos = self._c / 2
        ypos = y_max - self._c / 2

        self.fig = plt.figure(layout="compressed", dpi=self._params["dpi"])
        # plt.get_current_fig_manager().set_window_title("Circle Notation")
        ax = self.fig.gca()

        val = np.abs(self._sim._register)
        phi = -np.angle(self._sim._register, deg=False).flatten()
        lx, ly = np.sin(phi), np.cos(phi)

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_axis_off()
        ax.set_aspect("equal")

        # Scale textsizes such that ratio circles to textsize constant
        # automatic relative to length of y axis
        if (self._sim._n<6):
            factor = 0.8
        else:
            factor = 0.6 if not self._params["showValues"] else 0.4

        self._params["textsize_register"] *= factor
        self._params["textsize_magphase"] *= factor

        for i in range(2**self._sim._n):
            if val[i] > 1e-3:
                fill = mpatches.Circle(
                    (xpos, ypos),
                    radius=val[i],
                    color=self._params["color_fill"],
                    edgecolor=None,
                )
                phase = mlines.Line2D(
                    [xpos, xpos + lx[i]],
                    [ypos, ypos + ly[i]],
                    color=self._params["color_phase"],
                    linewidth=self._params["width_phase"],
                )
                ax.add_artist(fill)
                ax.add_artist(phase)
            ring = mpatches.Circle(
                (xpos, ypos),
                radius=1,
                fill=False,
                edgecolor=self._params["color_edge"],
                linewidth=self._params["width_edge"],
            )
            ax.add_artist(ring)
            label = self._createLabel(i)
            ax.text(
                xpos,
                ypos + self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                size=self._params["textsize_register"],
                horizontalalignment="center",
                verticalalignment="center",
            )
            # NOTE text vs TextPath:
            # text can easily be centered
            # textpath size is fixed when zooming
            # tp = TextPath((xpos-0.2*len(label),
            # ypos - 1.35),
            # f'|{label:s}>',
            # size=0.4)
            # ax.add_patch(PathPatch(tp, color="black"))
            if self._params["showValues"]:
                ax.text(
                    xpos,
                    ypos + self._params["offset_registerValues"],
                    f"{val[i]:+2.3f}\n{np.rad2deg(phi[i]):+2.0f}°",
                    size=self._params["textsize_magphase"],
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            xpos += self._c
            if (i + 1) % self._cols == 0:
                xpos = self._c / 2
                ypos -= self._c if not self._params["showValues"] else self._c * 1.5


class DimensionalCircleNotation(Visualization):
    """A Visualization subclass for the Dimensional Circle
    Notation (DCN) representation.
    """

    def __init__(self, simulator, parse_math=True, version=2):
        """Constructor for the Dimensional Circle Notation
        representation.

        Args:
            simulator (qc_simulator.simulator): Simulator object to be
            visualized.
        """
        super().__init__(simulator)  # Execute constructor of superclass
        # print(f"Setting up DCN Visualization in version {version}.")

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
            'textsize_register': 10*0.7**((self._sim._n-3)//2),
            'textsize_magphase': 8*0.7**((self._sim._n-3)//2),
            'textsize_axislbl': 10*0.7**((self._sim._n-3)//2),
            'bloch_outer_radius': 0.8  # Adjust for desired size/spacing
            })
        
        # Set default arrow style
        self._arrowStyle = {
            "width": 0.03*0.7**((self._sim._n-3)//2),
            "head_width": 0.2*0.7**((self._sim._n-3)//2),
            "head_length": 0.3*0.7**((self._sim._n-3)//2),
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
            "linewidth": 0.7**((self._sim._n-3)//2),
            "linestyle": "solid",
            "zorder": 0.7**((self._sim._n-3)//2),
        }
        # Create empty variables for later use
        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None

    def draw(self):
        """Draw Dimensional Circle Notation representation of current
        simulator state.
        """
        # Setup pyplot figure
        self.fig = plt.figure(layout="compressed")
        # plt.get_current_fig_manager().set_window_title(
        #     "Dimensional Circle Notation"
        # )
        self._ax = self.fig.gca()
        self._ax.set_axis_off()
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
        
        # Check whether the input is between 1 and 9
        if not 0<amount_qubits<10 or not isinstance(amount_qubits, int):
            raise NotImplementedError(
                "Please enter a valid number between 1 and 9."
            )
        
        ### Hard coded visualization for 1-3 Qubits - Dynamically coded 4+ Qubits ###
        
        # Origin x and y coordinate and 
        # length of a tick mark on the axis
        x, y, len_tick = -2, 7, .2

        # Set position of circles in DCN
        if amount_qubits>=1:
            # 1+ Qubits:
            self._coords = np.array([[0, 1],[1, 1]],dtype=float)
            # Set distance
            self._coords *= 3.5
            
            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                x_pos=x+1
                y_pos=y-2
                if amount_qubits==2:
                    x_pos+=-0.5
                    y_pos+=0.3
                elif amount_qubits>2:
                    x_pos+=-1.2
                    y_pos+=0.3
                self._ax.text(
                    x_pos+1.2, # TODO hier auch evtl auch mit x und y
                    y_pos+0.3,
                    "Qubit 1",
                    **self._textStyle
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x_pos, y_pos, 2.3, 0, **self._arrowStyle)
            # DCN V2: different coordinate axes
            else:
                # Horizontal axis (x,y,dx,dy, **kwargs)
                if amount_qubits==1:
                    self._ax.arrow(x+0.5, y-2, 6.3, 0, **self._arrowStyle)
                    y=5
                elif amount_qubits==2:
                    self._ax.arrow(x, y-2, 6.5, 0, **self._arrowStyle)
                    y=5
                else:
                    self._ax.arrow(x, y, 6.5, 0, **self._arrowStyle)
                    
                tick_y = [y-len_tick, y+len_tick]
                # 1st tick on x axis
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,
                    **self._plotStyle
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2.5*len_tick,
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
                    y + 2.5*len_tick,
                    "1",
                    **self._textStyle,
                )
                self._ax.text(
                    self._coords[0, 1]/2,
                    y + 3*len_tick,
                    "Qubit 1",
                    **self._textStyle,
                )
            if amount_qubits==1:
                # Set axis limits
                self._ax.set_xlim([-1.6, 5.3])
                self._ax.set_ylim([2.3, 5.5])
        # 2+ Qubits:
        if amount_qubits >= 2:
            self._coords = np.concatenate((self._coords, np.array([[0, 0],[3.5, 0]])))
            
            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                x_pos=x+0.35
                y_pos=y-2.75
                if amount_qubits>2:
                    x_pos-=0.7
                self._ax.text(
                    x_pos-0.15,
                    y_pos,
                    "Qubit 2",
                    **self._textStyle,
                    rotation=90
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x_pos+0.15, y_pos+1.05, 0, -2.3, **self._arrowStyle)

            # DCN V2: different coordinate axes
            else:
                # Vertical axis
                if amount_qubits==2:
                    self._ax.arrow(x, y, 0, -6, **self._arrowStyle)
                else:
                    self._ax.arrow(x, y, 0, -8, **self._arrowStyle)
                tick_x = [x-len_tick, x+len_tick]
                # 1st tick on y axis
                self._ax.plot(
                    tick_x,
                    [self._coords[0, 1], self._coords[0, 1]],
                    **self._plotStyle,
                )
                self._ax.text(
                    x - 2.5*len_tick,
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
                    x - 2.5*len_tick,
                    self._coords[3, 1],
                    "1",
                    **self._textStyle,
                    rotation=90
                )
                self._ax.text(
                    x - 3*len_tick,
                    self._coords[0, 1]/2,
                    "Qubit 2",
                    **self._textStyle,
                    rotation=90,
                )
            if amount_qubits==2:
                # Set axis limits
                self._ax.set_xlim([-2.8, 5])
                self._ax.set_ylim([-1.5, 5.8])
        # 3+ Qubits:
        if amount_qubits >= 3:
            # Double the array
            self._coords = np.concatenate((self._coords, self._coords))
            # Offset 3rd dim circles to the rear from position of the first 4 circles
            self._coords[4:] += 1.75

            # old style dcn coordinate axes
            # dirac labels are coonfigured in init already
            if self._params['version'] == 1:
                self._ax.text(
                    x+0.55,
                    y-0.55,
                    "Qubit 3",
                    **self._textStyle,
                    rotation=45
                )
                # Arrows for coordinate axis (x,y,dx,dy, **kwargs)
                self._ax.arrow(x-0.2, y-1.7, 1.65, 1.65, **self._arrowStyle)
                
            # DCN V2: different coordinate axes
            else:
                # Diagonal axis
                self._ax.arrow(x, y, 3.3, 3.3, **self._arrowStyle)
                len_tick_z = len_tick/np.sqrt(2)
                off1, off2 = 0.8, 2.2
                # 1st tick on z axis
                self._ax.plot(
                    [x+off1+len_tick_z, x+off1-len_tick_z],
                    [y+off1-len_tick_z, y+off1+len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x+off1-2.5*len_tick_z,
                    y+off1+2.5*len_tick_z,
                    "0",
                    **self._textStyle,
                    rotation=45
                )
                # 2nd tick on z axis
                self._ax.plot(
                    [x+off2+len_tick_z, x+off2-len_tick_z],
                    [y+off2-len_tick_z, y+off2+len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x+off2-2.5*len_tick_z,
                    y+off2+2.5*len_tick_z,
                    "1",
                    **self._textStyle,
                    rotation=45
                )
                middle_ticks = (off2-off1)/2
                self._ax.text(
                    x+off1+(middle_ticks)-5.5*len_tick_z,
                    y+off1+(middle_ticks)+5.5*len_tick_z,
                    "Qubit 3",
                    **self._textStyle,
                    rotation=45
                )
            if amount_qubits==3:
                # Set axis limits
                if self._params['version'] == 1:
                    self._ax.set_ylim([-1.5, 7.5])
                else:
                    self._ax.set_ylim([-1.5, 10.8])
                self._ax.set_xlim([-4.8, 8.5])
        # 4+ Qubits:
        if amount_qubits>=4: # Setting up remaining qubits and axis labels for qubits 4+
            for i in range(4, amount_qubits + 1):
                quarter_axis_length = (2**int(i/2))
                self._coords = np.concatenate((self._coords, self._coords))
                if (i%2==0): # Horizontal axes
                    # Shift it along the X-axis
                    self._coords[len(self._coords)//2:,0] += 2**(i/2+1)
                    self._ax.arrow(x, y+i, 4*quarter_axis_length, 0, **self._arrowStyle)
                    self._ax.plot( #|0> area
                        [x+quarter_axis_length/6, x+quarter_axis_length*1.875],
                        [y+i-0.3*len_tick, y+i-0.3*len_tick],
                        color= 'black',
                        linewidth= 2,
                        linestyle= "solid",
                        zorder= 1
                    )
                    self._ax.text(
                        x+quarter_axis_length,
                        y+i+2.5*len_tick,
                        "0",
                        **self._textStyle,
                    )
                    self._ax.plot( #|1> area
                        [x+2.125*quarter_axis_length, x+quarter_axis_length*3.875],
                        [y+i-0.3*len_tick, y+i-0.3*len_tick],
                        color= 'black',
                        linewidth= 2,
                        linestyle= "solid",
                        zorder= 1
                    )
                    self._ax.text(
                        3*quarter_axis_length-2,
                        y+i+2.5*len_tick,
                        "1",
                        **self._textStyle,
                    )
                    self._ax.text(
                        2*quarter_axis_length-2,
                        y+i+3*len_tick,
                        f"Qubit {i}",
                        **self._textStyle,
                    )
                else: # Vertical axes
                    # Shift it along the Y-axis
                    self._coords[len(self._coords)//2:,1] -= 2**((i+1)/2)
                    x_pos = x+3-i
                    self._ax.arrow(x_pos, y, 0, -4*quarter_axis_length, **self._arrowStyle)
                    self._ax.plot( #|0> area
                        [x_pos+0.3*len_tick, x_pos+0.3*len_tick],
                        [y-quarter_axis_length/6, y-quarter_axis_length*1.875],
                        color= 'black',
                        linewidth= 2,
                        linestyle= "solid",
                        zorder= 1
                    )
                    self._ax.text(
                        x_pos-2.5*len_tick,
                        y-quarter_axis_length,
                        "0",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.plot( #|1> area
                        [x_pos+0.3*len_tick, x_pos+0.3*len_tick],
                        [y-2.125*quarter_axis_length, y-quarter_axis_length*3.875],
                        color= 'black',
                        linewidth= 2,
                        linestyle= "solid",
                        zorder= 1
                    )
                    self._ax.text(
                        x_pos-2.5*len_tick,
                        y-3*quarter_axis_length,
                        "1",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.text(
                        x_pos-3*len_tick,
                        y-2*quarter_axis_length,
                        f"Qubit {i}",
                        **self._textStyle,
                        rotation=90,
                    )
            # Set axis limits according to plot size (grows with n for 4+ Qubits)
            self._ax.set_xlim([x-1-2*((amount_qubits-3)//2), x+1+2**(amount_qubits//2+2)])
            self._ax.set_ylim([y-1-2**((amount_qubits+1)//2+1), y+1+2*((amount_qubits)//2)])

        # Draw all circles
        self.draw_all_circles(amount_qubits)


        # Flip axis labels if bitOrder is set to 1
        self._axis_labels = np.arange(1, amount_qubits + 1
                                      )[:: self._params["bitOrder"]]

    def draw_all_circles(self, amount_qubits=None):
        # In the following the index is used to draw wires of circles
        if amount_qubits<3:
            self._drawLine([0, 1])
        if amount_qubits==2:
            self._drawLine([0, 2, 3, 1])
        for i in range(2**amount_qubits):
            if (i%8==0 and amount_qubits>2):
                # Draw wires of each cube
                self._drawLine([i, i+4, i+5])
                self._drawLine([i+1, i+5, i+7, i+3])
                self._drawLine([i, i+2, i+3, i+1])
                self._drawLine([i, i+1])
                self._drawDottedLine([i+2, i+6, i+7])
                self._drawDottedLine([i+4, i+6])
            # Draw all circles
            self._drawCircle(i)

    def _drawDottedLine(self, index):
        """Helper method:
        Draw dotted lines connecting points at given index. The coordinates of
        the points a defined internal in the _coords array

        Args:
            index (nested list([float, float]): List of indices
        """
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="dotted",
            zorder=1,
        )

    def _drawLine(self, index):
        """Helper method:
        Draw lines connecting the given points at given index. The coordinates
        of the points a defined internal in the _coords array.

        Args:
            index (nested list([float, float]): List of indices
        """
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="solid",
            zorder=1,
        )

    def _drawCircle(self, index):
        """Helper method:
        Draw single circle for DCN. Position and values of the circle are
        provided internal. Hand over the corect index

        Args:
            index (int): Index of the circle to be drawn.
        """
        xpos, ypos = self._coords[index]
        # White bg circle area of unit circle
        bg = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            color=self._params["color_bg"],
            edgecolor=None
        )
        self._ax.add_artist(bg)
        # Fill area of unit circle
        if self._val[index] >= 1e-3:
            fill = mpatches.Circle(
                (xpos, ypos),
                radius=self._val[index],
                color=self._params["color_fill"],
                edgecolor=None,
            )
            self._ax.add_artist(fill)
        # Black margin for circles
        ring = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            fill=False,
            edgecolor=self._params["color_edge"],
            linewidth=self._params["width_edge"],
        )
        self._ax.add_artist(ring)
        # Indicator for phase
        if self._val[index] >= 1e-3:
            phase = mlines.Line2D(
                [xpos, xpos + self._lx[index]],
                [ypos, ypos + self._ly[index]],
                color=self._params["color_phase"],
                linewidth=self._params["width_phase"],
            )
            self._ax.add_artist(phase)

        label = self._createLabel(index)
        if self._sim._n == 3:
            place = -1 if int(label[1]) else 1
        elif self._sim._n == 2:
            place = -1 if int(label[0]) else 1
        else:
            place = -1

        if self._params['labels_dirac']:
            # Add dirac label to circle
            self._ax.text(
                xpos,
                ypos + place * self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                **self._textStyle,
            )
        if self._params["showValues"]:
            self._ax.text(
                xpos,
                ypos
                + place
                * (
                    self._params["offset_registerLabel"]
                    + self._params["offset_registerValues"]
                ),
                f"{self._val[index]:+2.3f}\n"
                + f"{np.rad2deg(self._phi[index]):+2.0f}°",
                size=self._params["textsize_magphase"],
                horizontalalignment="center",
                verticalalignment="center",
            )


