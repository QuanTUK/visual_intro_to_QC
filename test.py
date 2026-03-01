import numpy as np
# import matplotlib
#
# matplotlib.use('Tkagg')

# Quantum Key Distribution

# Register as i-th row of comp. basis matrix 
# n = 1
# sim = Simulator(n)
# # sim.writeComplex([0, 1, 1, 0, 0, 0, 0, 0])
#
# vis = Visualization(sim, version=2)
#
# # sim.cSwap([1], 2,3)
# vis.show()

from qc_interactive_education_package import launch_app, launch_challenge
launch_app(num_qubits=3,show_circuit=True)
# launch_challenge(num_qubits=2,initial_state=[1,0,0,1], target_state=[0,1,-1,0], show_circuit=True)