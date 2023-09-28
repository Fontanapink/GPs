This graphs were generated using the autostat V2 repository. The graphs are generated using the `autostat.py` script. The script is run using the following command:

```python
import pretty_errors
import autostat
from autostat.test_data.test_data_loader import load_matlab_test_data_by_file_num

# `file_num=2` loads the famous Mauna Loa CO2 data set
#x,y = load_matlab_test_data_by_file_num(file_num=2)

# load simulations0.csv as numpy arrays
import pandas as pd
import numpy as np
data = pd.read_csv(r'test_GMLV/simulations0.csv')
# x is the first column of the csv file
# y is the second column of the csv file
x = np.array(data.iloc[:,0]).reshape(-1,1)
y = np.array(data.iloc[:,1]).reshape(-1,1)



abcd_model = autostat.ABCDModel(autostat.KernelSearchSettings(max_search_depth = 7, num_cpus = 8))

abcd_model.fit(x,y)

#autostat.plots.plot_decomposition(abcd_model)
```