# Fisher Information Matrix and Condition Number Analysis

## Overview

This document describes the relationship between Fisher Information Matrix (FIM) and condition number in bearing-only cooperative localization, and provides heuristics for improving observability.

## Background

### Fisher Information Matrix (FIM)

The Fisher Information Matrix quantifies the amount of information that bearing measurements carry about the pose parameters (rotation and translation). For bearing-only localization:

```
FIM = J^T * Σ^{-1} * J
```

where:
- `J` is the measurement Jacobian (relating pose changes to bearing changes)
- `Σ` is the measurement noise covariance

For isotropic Gaussian noise with standard deviation σ:
```
FIM = (J^T * J) / σ^2
```

### Condition Number

The condition number measures how sensitive a linear system's solution is to perturbations in the input. For a matrix A:

```
cond(A) = ||A|| * ||A^{-1}|| = σ_max / σ_min
```

where σ_max and σ_min are the largest and smallest singular values.

## Key Finding: FIM-Condition Number Relationship

**Mathematical Relationship:**

```
cond(FIM) ≈ [cond(J)]^2
```

This relationship holds because `FIM = J^T * J / σ^2`, and the condition number of `A^T * A` is approximately the square of the condition number of `A`.

**Practical Implications:**

1. **Both metrics indicate the same underlying issues**: If the Jacobian is ill-conditioned, the FIM will also be ill-conditioned
2. **Improving one improves the other**: Actions that reduce the Jacobian's condition number also improve the FIM
3. **FIM is more sensitive**: Since condition numbers are squared, FIM condition numbers can become very large even for moderate Jacobian condition numbers

## Observability Metrics from FIM

The FIM provides several metrics for assessing observability:

### 1. Condition Number
- **< 100**: GOOD observability
- **100 - 1000**: MODERATE observability  
- **1000 - 1e6**: POOR observability
- **> 1e6**: VERY POOR observability (severely ill-conditioned)

### 2. Determinant
- **Higher is better**: Larger determinant indicates better overall observability
- Used in D-optimality criterion for experiment design

### 3. Trace
- **Higher is better**: Sum of eigenvalues, indicates average observability
- Used in A-optimality criterion (most common)

### 4. Minimum Eigenvalue
- **Higher is better**: Indicates worst observable direction
- Used in E-optimality criterion
- Small minimum eigenvalue suggests poor observability in some directions

### 5. Eigenvalue Ratio
- **λ_max / λ_min**: Indicates anisotropy in observability
- Large ratio suggests uneven observability across different directions

## Condition Number-Based Heuristics for Improving Observability

### Strategy 1: Maximize Trace of FIM (A-Optimality)

The most commonly used criterion. Maximizes the sum of eigenvalues:

```python
objective = trace(FIM)
```

**When to use**: General-purpose observability improvement

### Strategy 2: Maximize Determinant of FIM (D-Optimality)

Maximizes the product of eigenvalues (geometric mean):

```python
objective = log(det(FIM))
```

**When to use**: When you want to ensure good observability in all directions

### Strategy 3: Maximize Minimum Eigenvalue (E-Optimality)

Improves the worst observable direction:

```python
objective = λ_min(FIM)
```

**When to use**: When you want to avoid blind spots in observability

### Strategy 4: Minimize Condition Number

Balances best and worst observable directions:

```python
objective = 1 / cond(FIM)
```

**When to use**: When numerical stability is critical

## Practical Guidelines for Improving Observability

### What Causes Poor Observability?

1. **Nearly parallel bearing vectors**: All bearings pointing in similar directions
2. **Planar configurations**: Bearings lack 3D diversity
3. **Concentrated angular distribution**: Bearings clustered in small angular region
4. **Insufficient bearing diversity**: Too few distinct bearing directions

### How to Improve Observability

#### 1. Diversify Bearing Directions
- Ensure bearings are spread across multiple directions
- Avoid concentrating bearings in one region of space
- Aim for 3D diversity, not just 2D

#### 2. Use FIM Analysis for Trajectory Planning
```python
from fisher_information_matrix import FisherInformationAnalyzer

# Current configuration
uvw = ...  # Current bearing points
R, t = ...  # Current pose estimate

# Get improvement suggestions
suggestions = FisherInformationAnalyzer.suggest_improvements(uvw, R, t)
print(suggestions['rotation_suggestion'])
print(suggestions['translation_suggestion'])
```

#### 3. Monitor Condition Number During Operation
```python
analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
print(analysis['analysis'])

if analysis['FIM_metrics']['condition_number'] > 1e6:
    print("WARNING: Poor observability - adjust trajectory")
```

#### 4. Compare Multiple Trajectory Options
```python
# Evaluate multiple candidate configurations
configs = [uvw1, uvw2, uvw3]
results = FisherInformationAnalyzer.compare_configurations(configs, R, t)
# Choose configuration with best determinant or lowest condition number
```

## Usage Examples

### Example 1: Basic Observability Analysis

```python
from fisher_information_matrix import FisherInformationAnalyzer
import numpy as np

# Points in global frame (3 x n)
uvw = np.random.randn(3, 8) * 10

# Current pose estimate
R = np.eye(3)
t = np.array([1.0, 2.0, 3.0])

# Analyze observability
analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)

print(analysis['analysis'])
print(f"Condition number: {analysis['FIM_metrics']['condition_number']:.2e}")
print(f"Determinant: {analysis['FIM_metrics']['determinant']:.2e}")
```

### Example 2: Trajectory Guidance

```python
# Test different agent positions
best_objective = -np.inf
best_position = None

for x in range(10):
    t_test = np.array([x, 0, 0])
    
    # Compute objective for this position
    obj = FisherInformationAnalyzer.compute_condition_based_objective(
        uvw, R, t_test, objective_type='trace'
    )
    
    if obj > best_objective:
        best_objective = obj
        best_position = t_test

print(f"Best position: {best_position}")
print(f"Objective value: {best_objective:.2e}")
```

### Example 3: Compare Scenarios

```python
# Scenario 1: Well-distributed bearings
uvw_good = np.array([
    [10, -10, 10, -10, 0, 0],
    [10, 10, -10, -10, 10, -10],
    [5, 5, 5, 5, 10, 10]
], dtype=float)

# Scenario 2: Nearly parallel bearings
uvw_poor = np.ones((3, 6)) * 10
uvw_poor += np.random.randn(3, 6) * 0.1

# Compare
configs = [uvw_good, uvw_poor]
results = FisherInformationAnalyzer.compare_configurations(configs, R, t)
```

## Relationship to Existing Condition Number Implementation

The repository already has condition number monitoring in `bearing_linear_solver`:

```python
# In bearing_linear_solver.solve()
cond_num = bearing_linear_solver.compute_condition_number(A)
if cond_num > 1e10:
    logger.info('Ill-conditioned matrix detected. Applying regularization.')
```

The FIM analysis complements this by:
1. Providing a **theoretical foundation** for why condition number matters
2. Offering **multiple metrics** beyond just condition number
3. Enabling **trajectory optimization** for observability improvement
4. Giving **directional guidance** on how to improve observability

## Demonstration

Run the comprehensive demonstration:

```bash
python demo_fim_analysis.py
```

This demonstrates:
1. The mathematical relationship between FIM and condition number
2. Observability analysis for different scenarios
3. Condition number-based improvement heuristics
4. Guidance strategies for trajectory planning
5. Analysis on real simulation data

## Testing

Run the test suite:

```bash
python test_fisher_information.py
```

Tests cover:
- FIM computation correctness
- Jacobian computation
- Observability metrics
- FIM-condition number relationship
- Improvement suggestions
- Objective functions

## API Reference

### `FisherInformationAnalyzer` Class

#### Methods

**`compute_fisher_information_matrix(uvw, R, t, measurement_noise_std=0.01)`**
- Computes the Fisher Information Matrix
- Returns: 6×6 numpy array

**`compute_fim_metrics(FIM)`**
- Computes various metrics from FIM
- Returns: Dictionary with trace, determinant, condition_number, eigenvalues, etc.

**`analyze_observability(uvw, R, t, measurement_noise_std=0.01)`**
- Comprehensive observability analysis
- Returns: Dictionary with FIM, metrics, Jacobian, and text analysis

**`suggest_improvements(uvw, R, t, measurement_noise_std=0.01)`**
- Suggests trajectory modifications to improve observability
- Returns: Dictionary with suggestions for rotation and translation

**`compute_condition_based_objective(uvw, R, t, objective_type='trace')`**
- Computes objective function for guidance
- Objective types: 'trace', 'determinant', 'min_eigenvalue', 'inverse_condition'
- Returns: Scalar objective value

**`compare_configurations(uvw_configs, R, t)`**
- Compares multiple bearing configurations
- Returns: Dictionary mapping configuration index to metrics

## Conclusion

The Fisher Information Matrix provides a rigorous mathematical framework for understanding and improving observability in bearing-only localization. The strong relationship with condition number (`cond(FIM) ≈ [cond(J)]^2`) confirms that:

1. **Condition number is a valid proxy** for observability
2. **Both FIM and condition number** capture the same fundamental issues
3. **Heuristics based on condition number** are well-founded
4. **Multiple optimization criteria** (trace, determinant, min eigenvalue) can be used depending on the application

By monitoring and optimizing these metrics, researchers can design better guidance strategies for cooperative localization systems.
