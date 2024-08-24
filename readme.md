# Bearing Only Solver
This repository contains a Python implementation of a bearing-only solver. The solver is designed to compute the relative pose between two sets of points given their bearings.

## `bearing_linear_solver` Class
The `bearing_linear_solver` class implements the linear bearing solver algorithm proposed in [1]. It provides methods to solve for the rotation matrix, translation vector, and error given 2D coordinates, bearing angles, and other parameters.

> Cooperative Localisation of a GPS-Denied UAV using Direction-of-Arrival Measurements. JS Russell, M Ye, BDO Anderson, H Hmam, P Sarunic. IEEE Transactions on Aerospace and Electronic Systems， 2019

```python
class bearing_linear_solver():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def compute_reduced_Ab_matrix(uA: np.ndarray, vA: np.ndarray, wA: np.ndarray, phi: np.ndarray, theta: np.ndarray, 
                                k: int, xB: np.ndarray, yB: np.ndarray, zB: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    ...

    @staticmethod
    def solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
```

### Usage
To use the bearing-only solver, run the `bearing_only_solver.py` script with the appropriate folder and file prefix:
```python
import numpy as np
from bearing_only_solver import bearing_linear_solver

# Example data
p1 = np.array([[0, 0], [1, 1], [2, 2]])
p2 = np.array([[0, 0], [1, 1], [2, 2]])
bearing = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Solve using the BGPnP algorithm
R, T = bearing_linear_solver.solve(p1, p2, bearing)

print("Rotation Matrix:", R)
print("Translation Vector:", T)
```

## `bgpnp` Class

The `bgpnp` class implements the Bearing Generalized Perspective-n-Point (BGPnP) algorithm. It provides methods to solve for the rotation matrix, translation vector, and error given 2D coordinates, bearing angles, and other parameters.

### Class Definition

```python
class bgpnp:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        # Method implementation

    @staticmethod
    def define_control_points() -> np.ndarray:
        # Method implementation

    @staticmethod
    def compute_alphas(Xw: np.ndarray, Cw: np.ndarray) -> np.ndarray:
        # Method implementation

    @staticmethod 
    def myProcrustes(X, Y):
        # Method implementation

    @staticmethod
    def KernelPnP(Cw: np.ndarray, Km: np.ndarray, dims: int = 4, sol_iter: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        # Method implementation

    @staticmethod
    def kernel_noise(M: np.ndarray, b: np.ndarray, dimker: int = 4) -> np.ndarray:
        # Method implementation

    @staticmethod
    def prepare_data(p: np.ndarray, bearing: np.ndarray, pb: np.ndarray, Cw: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Method implementation

    @staticmethod
    def skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
        # Method implementation

    @staticmethod
    def compute_Mb(bearing: np.ndarray, Alph: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Method implementation
```
### Usage Example
```python
import numpy as np
from bearing_only_solver import bgpnp

# Example data
p1 = np.array([[0, 0], [1, 1], [2, 2]])
p2 = np.array([[0, 0], [1, 1], [2, 2]])
bearing = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Solve using the BGPnP algorithm
R, T, err = bgpnp.solve(p1, p2, bearing)

print("Rotation Matrix:", R)
print("Translation Vector:", T)
print("Error:", err)
```

### Description
The bgpnp class provides several static methods to perform various computations required by the BGPnP algorithm:
- solve(p1, p2, bearing, sol_iter): Computes the rotation matrix, translation vector, and error.



## Dependencies
- numpy
- scipy
- logging
Install the dependencies using pip:
```bash
pip install numpy scipy
```


# Simulation Data Processing
This project contains scripts to process and save simulation data. The main functionalities include loading, processing, and saving simulation data in a specific format.

## Files
- `load_data.py`: Contains functions to load and save simulation data.
## Functions
`save_simulation_data(simulation_data, folder: str, file: str)`

This function saves the simulation data to text files.

- Parameters:
    - simulation_data: List of dictionaries containing simulation data.
    - folder: The folder where the files will be saved.
    - file: The base name of the files.

- Functionality:
    - Iterates through the simulation_data.
    - For each data entry, it creates a text file.
    - Writes p1, p2, bearing, Rgt, and tgt data to the file.
    - Each element of Rgt and tgt is written one by one to ensure no extra brackets are included.

`gen_simulation_data_dtu()`

This function generates simulation data for DTU.

- Functionality:
    - Prepares data for two agents (agent_a and agent_b) using the prepare_data function.
    - Logs the length of agent_a.

## Usage
1. Ensure you have the necessary data files in the specified folder.
2. Call the save_simulation_data function with the appropriate parameters to save the data.
3. Use the gen_simulation_data_dtu function to generate and prepare simulation data for DTU.

## Example
```python
from load_data import save_simulation_data, gen_simulation_data_dtu

# Example usage of save_simulation_data
simulation_data = [
    {
        "p1": np.array([[1, 2], [3, 4]]),
        "p2": np.array([[5, 6], [7, 8]]),
        "bearing": np.array([[9, 10], [11, 12]]),
        "Rgt": np.array([[13, 14], [15, 16]]),
        "tgt": np.array([[17, 18], [19, 20]])
    }
]
save_simulation_data(simulation_data, './data/', 'simulation_')

# Example usage of gen_simulation_data_dtu
gen_simulation_data_dtu()
```

## Requirements
- Python 3.x
- NumPy

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

