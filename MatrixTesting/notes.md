So, this matrix analysis are based on:
1) the simulations (99 of them) that where generated using HPC (High Performance Computing) and the code for: `GitFitScriptINDPENDENTGPR.py`, `GPFitScriptAutoRegressive.py`
2) Then, with the 100 fits using either of those scripts, the generated kernels where paired with the matrices properties (eigenvalues, condition number, etc) and the results where saved in a csv file.
3) Then, we analyzed to see if the different kernels could be associated with the different matrix properties. 