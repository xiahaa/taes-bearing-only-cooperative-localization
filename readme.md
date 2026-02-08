# Bearing-Only Cooperative Localization

This repository contains Python implementations of bearing-only localization algorithms for GPS-denied environments. The algorithms compute the relative pose (rotation and translation) between two agents using only bearing angle measurements to common landmarks.

## Algorithms

### `bearing_linear_solver` Class
The `bearing_linear_solver` class implements linear and semidefinite programming (SDP) based bearing solvers proposed in [1]. It provides methods to solve for the rotation matrix and translation vector given 3D point coordinates and bearing measurements.

> **Reference:** Cooperative Localisation of a GPS-Denied UAV using Direction-of-Arrival Measurements. 
> JS Russell, M Ye, BDO Anderson, H Hmam, P Sarunic. 
> IEEE Transactions on Aerospace and Electronic Systems, 2019

#### Available Methods

- **`solve(uvw, xyz, bearing)`** - Linear least-squares solver with automatic regularization
  - Constructs measurement matrix from bearing constraints
  - Automatically detects ill-conditioned matrices (condition number > 1e10)
  - Applies Tikhonov regularization when needed to improve robustness
  - Fast and numerically stable even with poorly distributed bearing vectors

- **`solve_with_sdp_sdr(uvw, xyz, bearing)`** - SDP with Semidefinite Relaxation
  - Formulates as semidefinite program with SO(3) constraints
  - Uses MOSEK solver (requires license - free for academic use)
  - Performs rank-1 approximation via SVD for final solution
  - More robust to noise but slower

- **`ransac_solve(uvw, xyz, bearing, ...)`** - RANSAC with linear solver
  - Robust to outliers in bearing measurements
  - Iteratively samples minimal sets and finds best inlier set

- **`ransac_solve_with_sdp_sdr(uvw, xyz, bearing, ...)`** - RANSAC with SDP solver
  - Combines outlier robustness with SDP accuracy

#### Class Methods

**Note:** All methods are decorated with `@timeit` which returns `((result), elapsed_time)` tuples.

```python
class bearing_linear_solver():
    @staticmethod
    def solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linear solver returning ((R, t), time)"""
    
    @staticmethod
    def solve_with_sdp_sdr(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SDP solver returning ((R, t), time)"""
    
    @staticmethod
    def ransac_solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray,
                     num_iterations: int = 500, threshold: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC-based linear solver for outlier rejection, returning ((R, t), time)"""
    
    @staticmethod
    def ransac_solve_with_sdp_sdr(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray,
                                   num_iterations: int = 500, threshold: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC-based SDP solver for outlier rejection, returning ((R, t), time)"""
```

#### Usage Example
```python
import numpy as np
from bearing_only_solver import bearing_linear_solver, load_simulation_data

# Load simulation data
data = load_simulation_data("batch_0.txt")
uvw = data["p1"]      # Global frame positions (3×n)
xyz = data["p2"]      # Local frame positions (3×n)  
bearing = data["bearing"]  # Bearing vectors (3×n)

# Linear solver (returns ((R, t), elapsed_time) due to @timeit decorator)
(R, t), time = bearing_linear_solver.solve(uvw, xyz, bearing)

# SDP solver (more accurate but slower)
(R, t), time = bearing_linear_solver.solve_with_sdp_sdr(uvw, xyz, bearing)

# RANSAC for outlier rejection
(R, t), time = bearing_linear_solver.ransac_solve(uvw, xyz, bearing, 
                                                   num_iterations=500, 
                                                   threshold=1e-2)

print("Rotation Matrix:", R)
print("Translation Vector:", t)
print("Elapsed Time:", time, "seconds")
```

### `bgpnp` Class

The `bgpnp` class implements the Bearing Generalized Perspective-n-Point (BGPnP) algorithm. It uses a control point representation and kernel-based optimization to solve for pose.

#### Available Methods

- **`solve(p1, p2, bearing, sol_iter=True, enforce_manifold=False)`** - Main BGPnP solver
  - Uses control points and kernel decomposition
  - Optional iterative refinement for improved accuracy
  - **New**: `enforce_manifold` parameter enables SO(3) manifold constraints for noise robustness
  - Returns (R, t, error)

- **`ransac_solve(p1, p2, bearing, ...)`** - RANSAC variant
  - Robust to outlier measurements
  - Samples minimal sets (6 points), finds best inliers
  - Refits final solution on inlier set

- **`solve_new_loss(p1, p2, bearing, ...)`** - Alternative loss function
  - Experimental variant with different error metric

#### Robustness in Noisy Conditions

The `bgpnp` solver can now enforce **SO(3) manifold constraints** similar to the SDP-SDR approach, improving robustness in noisy bearing measurements:

**When to use `enforce_manifold=True`:**
- Bearing measurements have significant noise (> 1 degree)
- Need robustness similar to SDP-SDR but faster
- Willing to trade computation time for accuracy

**What it does:**
1. **Tikhonov regularization** in kernel computation (adaptive based on condition number)
2. **SO(3) projection** after each Procrustes iteration to enforce proper rotation properties
3. **Determinant enforcement** ensures det(R) = +1 (proper rotation, not reflection)

**Performance comparison** (rotation error):
| Noise Level | Standard BGPnP | With Manifold | Improvement |
|-------------|---------------|---------------|-------------|
| 1° noise    | 161.2°        | 38.0°         | **81%**     |
| 2° noise    | 171.5°        | 77.8°         | **55%**     |
| 5° noise    | 159.8°        | 88.9°         | **44%**     |

**Trade-off**: Slower (~100-300x slower due to regularization and iterations in noisy conditions) but significantly more accurate in high noise scenarios.

#### Class Methods

**Note:** All methods are decorated with `@timeit` which returns `((result), elapsed_time)` tuples.

```python
class bgpnp:
    @staticmethod
    def solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, 
              sol_iter: bool = True, enforce_manifold: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve BGPnP problem.
        
        Args:
            p1: 3D points in global frame (n×3)
            p2: 3D points in local frame (n×3)
            bearing: Bearing vectors (n×3)
            sol_iter: Whether to use iterative refinement
            enforce_manifold: Enable SO(3) constraints for noise robustness
            
        Returns:
            ((R, t, error), elapsed_time) where:
                R: Rotation matrix (3×3)
                t: Translation vector (3,)
                error: Reprojection error
                elapsed_time: Computation time in seconds
        """
    
    @staticmethod
    def ransac_solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray,
                     num_iterations: int = 500, threshold: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, float]:
        """RANSAC-based BGPnP solver for outlier rejection, returning ((R, t, error), time)"""
```
#### Usage Example
```python
import numpy as np
from bearing_only_solver import bgpnp, load_simulation_data

# Load simulation data
data = load_simulation_data("batch_0.txt")
p1 = data["p1"].T  # Convert to n×3
p2 = data["p2"].T
bearing = data["bearing"].T

# Standard BGPnP solve (returns ((R, t, error), elapsed_time) due to @timeit decorator)
(R, t, error), time = bgpnp.solve(p1, p2, bearing, sol_iter=True)

# For noisy bearings, use manifold constraints (similar robustness to SDP-SDR)
(R, t, error), time = bgpnp.solve(p1, p2, bearing, sol_iter=True, enforce_manifold=True)

# RANSAC for outlier rejection
(R, t, error), time = bgpnp.ransac_solve(p1, p2, bearing,
                                         num_iterations=500,
                                         threshold=1e-2)

print("Rotation Matrix:", R)
print("Translation Vector:", t)
print("Reprojection Error:", error)
print("Elapsed Time:", time, "seconds")
```

## Experiments

The repository includes several experimental scripts for benchmarking and testing:

### `exp_tcst_data.py` - TCST Paper Benchmark

Evaluates all solvers on data from the TCST 2021 reference paper:

> N. T. Hung, F. F. C. Rego and A. M. Pascoal, "Cooperative Distributed Estimation and Control of Multiple Autonomous Vehicles for Range-Based Underwater Target Localization and Pursuit," IEEE Transactions on Control Systems Technology, 2021.

**Usage:**
```bash
python exp_tcst_data.py --folder ../2/
```

**Output:** Comparative plots of rotation/translation errors and execution times for BGPnP, Linear Solver, and SDP solver.

### `exp_random_data.py` - Synthetic Data Benchmark

Generates random test cases with configurable noise levels and evaluates solver performance.

**Features:**
- Configurable batch size (number of landmarks)
- Adjustable bearing noise (standard deviation in degrees)
- Generates ground truth poses and noisy bearing measurements

**Usage:**
```bash
# Generate data
python exp_random_data.py --generate --batch_size 6 --folder ../3/ --std_noise 2.0

# Run benchmark
python exp_random_data.py --folder ../3/
```

### `exp_outlier_test.py` - Outlier Robustness Testing

Tests RANSAC variants against varying outlier ratios.

**Features:**
- Configurable outlier percentage (default: 20%)
- Compares standard vs RANSAC solvers
- Evaluates degradation with increasing outliers

**Usage:**
```bash
# Generate data with 20% outliers
python exp_outlier_test.py --generate --ratio 0.2 --folder ../5/

# Run benchmark
python exp_outlier_test.py --folder ../5/
```

### `exp_real_data.py` - Real UAV Flight Data

Processes actual IMU/GPS data from UAV experiments (DTU and TAES datasets).

**Features:**
- Loads MATLAB format sensor data
- Converts GPS/IMU to relative poses
- Tests bearing-only localization on real measurements

**Usage:**
```bash
python exp_real_data.py
```

## Data Format

Simulation data files store one test case per file with the following format:

```
<p1 values as space-separated floats (3n values for n points)>
<p2 values as space-separated floats (3n values)>
<bearing values as space-separated floats (3n values)>
<Rgt values as space-separated floats (9 values, row-major 3×3)>
<tgt values as space-separated floats (3 values)>
```

**Example with 2 points:**
```
1.0 2.0 3.0 4.0 5.0 6.0
7.0 8.0 9.0 10.0 11.0 12.0
0.707 0.707 0.0 0.577 0.577 0.577
0.866 -0.5 0.0 0.5 0.866 0.0 0.0 0.0 1.0
1.5 2.5 3.5
```
Line 1: p1 = [[1,2,3], [4,5,6]]^T (2 points in global frame)  
Line 2: p2 = [[7,8,9], [10,11,12]]^T (2 points in local frame)  
Line 3: bearing = [[0.707,0.707,0], [0.577,0.577,0.577]]^T (2 bearing vectors)  
Line 4: 3×3 rotation matrix Rgt (row-major order)  
Line 5: 3D translation vector tgt

Load using:
```python
from bearing_only_solver import load_simulation_data
data = load_simulation_data("batch_0.txt")
# Returns dict with keys: p1, p2, bearing, Rgt, tgt
# Note: p1, p2, bearing are returned as 3×n arrays (transposed)
```

## Installation

### Requirements

**Core Dependencies:**
- Python 3.x
- NumPy - Array operations
- SciPy - Linear algebra and optimization
- cvxpy - Convex optimization framework

**Optional Dependencies:**
- sophuspy - Lie group operations (for SO(3) sampling in experiments)
- matplotlib - Plotting and visualization
- seaborn - Statistical plotting
- pandas - Data analysis

### Install

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy scipy cvxpy

# Optional: for experiments and visualization
pip install sophuspy matplotlib seaborn pandas
```

### MOSEK License (Required for SDP Solver)

The SDP solver uses MOSEK, which requires a license:
1. **Academic users**: Get a free academic license at https://www.mosek.com/products/academic-licenses/
2. **Personal/hobby use**: Free personal academic license available
3. Follow MOSEK installation instructions to install the license file

Without MOSEK, you can still use:
- `bearing_linear_solver.solve()` (linear solver)
- `bgpnp.solve()` and `bgpnp.ransac_solve()` (BGPnP algorithms)

## Quick Start

```python
from bearing_only_solver import bearing_linear_solver, bgpnp, load_simulation_data
import numpy as np

# Load test data
data = load_simulation_data("taes/simu_0.txt")

# Method 1: Linear solver (fastest)
# Note: Methods decorated with @timeit return ((result), elapsed_time)
(R, t), time = bearing_linear_solver.solve(data["p1"], data["p2"], data["bearing"])

# Method 2: BGPnP (good accuracy)
(R, t, err), time = bgpnp.solve(data["p1"].T, data["p2"].T, data["bearing"].T)

# Method 3: SDP solver (best accuracy, requires MOSEK)
(R, t), time = bearing_linear_solver.solve_with_sdp_sdr(data["p1"], data["p2"], data["bearing"])

# With outliers: use RANSAC
(R, t), time = bearing_linear_solver.ransac_solve(data["p1"], data["p2"], data["bearing"])
(R, t, err), time = bgpnp.ransac_solve(data["p1"].T, data["p2"].T, data["bearing"].T)
```

## Numerical Robustness for Ill-Conditioned Scenarios

The performance of the linear solver depends on the distribution of bearing vectors. When bearing vectors are poorly distributed (e.g., nearly parallel or concentrated in similar directions), the action matrix A becomes ill-conditioned, which can lead to numerical instability and inaccurate results.

### Automatic Regularization

The `solve()` method now includes automatic detection and handling of ill-conditioned matrices:

1. **Condition Number Monitoring**: The solver computes the condition number of the A matrix
2. **Automatic Regularization**: When the condition number exceeds 1e10, Tikhonov (ridge) regularization is automatically applied
3. **Adaptive Parameter Selection**: The regularization parameter is computed adaptively based on the singular values of A

### Understanding Condition Number

The condition number measures how sensitive the solution is to small perturbations in the input:
- **Well-conditioned**: cond(A) ≈ 1 to 1e3 - stable, accurate solutions
- **Moderately ill-conditioned**: cond(A) ≈ 1e3 to 1e10 - may have some numerical errors
- **Severely ill-conditioned**: cond(A) > 1e10 - regularization applied automatically

### When Does Ill-Conditioning Occur?

Ill-conditioning typically occurs when:
- Bearing vectors are nearly parallel (pointing in similar directions)
- Bearing vectors are concentrated in a small angular region
- Insufficient diversity in bearing directions relative to the geometry

### Manual Control (Advanced Users)

For advanced users who want explicit control over regularization:

```python
from bearing_only_solver import bearing_linear_solver
import numpy as np

# Compute condition number manually
A = bearing_linear_solver.compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], 
                                           phi, theta, n_points)
cond_num = bearing_linear_solver.compute_condition_number(A)
print(f"Condition number: {cond_num:.2e}")

# Apply regularization with custom parameter
b = bearing_linear_solver.compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], 
                                           phi, theta, n_points)
x = bearing_linear_solver.solve_with_regularization(A, b, regularization=0.001)
```

## Algorithm Comparison

| Algorithm | Speed | Accuracy | Outlier Robust | Noise Robust | License Required |
|-----------|-------|----------|----------------|--------------|------------------|
| Linear Solver | Fast | Good | No | Moderate | No |
| BGPnP (standard) | Medium | Better | No | Moderate | No |
| **BGPnP (enforce_manifold)** | **Slow** | **Better** | **No** | **High** | **No** |
| SDP + SDR | Slow | Best | No | High | MOSEK |
| RANSAC + Linear | Medium | Good | Yes | Moderate | No |
| RANSAC + BGPnP | Slow | Better | Yes | Moderate | No |
| RANSAC + SDP | Very Slow | Best | Yes | High | MOSEK |

**Noise Robustness Levels:**
- **High**: Robust to 5+ degrees of bearing noise (SDP-SDR, BGPnP with manifold)
- **Moderate**: Works well with < 1 degree of bearing noise (standard methods)

**Recommendations:**
- **Real-time applications**: Use `bearing_linear_solver.solve()` or `bgpnp.solve()`
- **High accuracy needed**: Use `bearing_linear_solver.solve_with_sdp_sdr()` (if MOSEK available)
- **Noisy bearings (> 1° noise)**: Use `bgpnp.solve(enforce_manifold=True)` or SDP-SDR
- **Outlier-prone data**: Use any RANSAC variant
- **Ill-conditioned bearing distributions**: The linear solver now handles this automatically

## Project Structure

```
.
├── readme.md              # This file
├── requirements.txt       # Python dependencies
├── src/
│   ├── bearing_only_solver.py          # Main algorithms
│   ├── bearing_only_solver_3andmore.py # Extended solver variants
│   ├── exp_tcst_data.py                # TCST benchmark
│   ├── exp_random_data.py              # Synthetic data benchmark
│   ├── exp_outlier_test.py             # Outlier robustness test
│   ├── exp_real_data.py                # Real UAV data processing
│   ├── load_data.py                    # Data loading utilities
│   ├── test_bgpnp.py                   # BGPnP unit tests
│   └── test_load_data.py               # Data loading tests
└── taes/
    └── simu_0.txt         # Example simulation data
```

## References

[1] JS Russell, M Ye, BDO Anderson, H Hmam, P Sarunic. "Cooperative Localisation of a GPS-Denied UAV using Direction-of-Arrival Measurements." IEEE Transactions on Aerospace and Electronic Systems, 2019.

[2] N. T. Hung, F. F. C. Rego, A. M. Pascoal. "Cooperative Distributed Estimation and Control of Multiple Autonomous Vehicles for Range-Based Underwater Target Localization and Pursuit." IEEE Transactions on Control Systems Technology, 2021.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.
