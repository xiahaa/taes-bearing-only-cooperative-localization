# Bearing Only Solver
This repository contains a Python implementation of a bearing-only solver. The solver is designed to compute the relative pose between two sets of points given their bearings.

`bearing_only_solver`
```python
def bearing_only_solver(foler: str, file: str):
    ...
```

Main function to solve the bearing-only problem. It loads the data, computes the necessary matrices, and solves for the relative pose.

## Usage
To use the bearing-only solver, run the `bearing_only_solver.py` script with the appropriate folder and file prefix:
```python
python bearing_only_solver.py
```

Make sure to update the folder and file prefix in the script as needed.

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

