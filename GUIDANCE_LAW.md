# Guidance Law Derivation from Condition Matrix

## Overview

This document describes how to derive guidance laws from the Fisher Information Matrix (FIM) and condition matrix for bearing-only cooperative localization. The guidance laws optimize observability while achieving mission objectives such as target pursuit or trajectory following.

## Background

### From Fisher Information to Guidance

The Fisher Information Matrix (FIM) quantifies how much information bearing measurements provide about the pose (rotation and translation). We have established that:

```
cond(FIM) ≈ [cond(J)]^2
```

where J is the Jacobian of bearing measurements with respect to pose parameters.

**Key Insight:** The eigenstructure of the FIM reveals:
1. **Best observable directions** (large eigenvalues)
2. **Worst observable directions** (small eigenvalues)
3. **Optimal motion directions** (gradients of FIM objectives)

This enables us to **derive guidance laws** that optimize observability by steering agents toward favorable configurations.

## Universal Guidance Law

### Concept

The universal guidance law computes optimal motion directions to maximize observability based on various FIM-derived objectives.

### Available Objectives

#### 1. Trace Maximization (A-Optimality)
```python
objective = trace(FIM)
```
- **What it optimizes:** Sum of all eigenvalues
- **Interpretation:** Average observability across all directions
- **Use case:** General-purpose guidance
- **Pros:** Fast to compute, widely used in optimal experiment design
- **Cons:** May not prevent blind spots

#### 2. Determinant Maximization (D-Optimality)
```python
objective = log(det(FIM))
```
- **What it optimizes:** Product of eigenvalues (geometric mean)
- **Interpretation:** Balanced observability in all directions
- **Use case:** Robust localization avoiding blind spots
- **Pros:** Ensures no eigenvalue becomes too small
- **Cons:** More sensitive to ill-conditioning

#### 3. Minimum Eigenvalue Maximization (E-Optimality)
```python
objective = min(eigenvalues(FIM))
```
- **What it optimizes:** Smallest eigenvalue
- **Interpretation:** Worst-case observability
- **Use case:** Safety-critical applications
- **Pros:** Directly improves weakest direction
- **Cons:** May ignore other directions

#### 4. Condition Number Minimization
```python
objective = 1 / cond(FIM)
```
- **What it optimizes:** Ratio of largest to smallest eigenvalue
- **Interpretation:** Numerical stability
- **Use case:** Ill-conditioned scenarios
- **Pros:** Prevents numerical issues
- **Cons:** Indirect measure of observability

### Usage

#### Basic Optimal Direction

```python
from src.guidance_law import GuidanceLaw
import numpy as np

# Landmarks in global frame
uvw = np.random.randn(3, 8) * 10

# Current pose estimate
R = np.eye(3)
t = np.array([5.0, 5.0, 5.0])

# Compute optimal direction (trace maximization)
direction, current_obj = GuidanceLaw.compute_optimal_direction(
    uvw, R, t, objective_type='trace'
)

print(f"Move in direction: {direction}")
print(f"Current objective: {current_obj}")
```

#### Guidance Command with Dynamics

```python
# Current velocity
velocity = np.array([2.0, 0.0, 0.0])

# Compute acceleration command
acceleration = GuidanceLaw.compute_guidance_command(
    uvw, R, t, velocity,
    objective_type='trace',
    max_acceleration=1.0,  # m/s^2
    dt=0.1  # time step
)

# Update velocity and position
velocity_new = velocity + acceleration * 0.1
position_new = t + velocity_new * 0.1
```

#### Trajectory Evaluation

```python
# Define a trajectory
trajectory = [
    np.array([0, 0, 0]),
    np.array([1, 0, 0]),
    np.array([2, 1, 0]),
    np.array([3, 1, 1]),
]

# Evaluate observability along trajectory
results = GuidanceLaw.evaluate_trajectory(
    uvw, trajectory, R, objective_type='trace'
)

print(f"Objectives: {results['objectives']}")
print(f"Condition numbers: {results['condition_numbers']}")
print(f"Determinants: {results['determinants']}")
```

## Two-Agent Pursuit Guidance

### Scenario

- **Agent 1 (Target):** Flies in a straight line at constant velocity
- **Agent 2 (Pursuer):** Pursues the target while optimizing observability using bearing measurements to landmarks

### Guidance Strategy

The pursuer's guidance law balances two objectives:

1. **Pursuit:** Move toward predicted target intercept point
2. **Observability:** Move to improve FIM-based observability metrics

The blending is controlled by weights:
```python
velocity = pursuit_gain * pursuit_direction + observability_gain * obs_direction
```

### Usage

#### Initialize Pursuit Guidance

```python
from src.guidance_law import TwoAgentPursuitGuidance

# Create pursuit guidance
guidance = TwoAgentPursuitGuidance(
    pursuer_speed=12.0,           # Pursuer speed (m/s)
    pursuit_gain=0.6,             # Weight for pursuit (0-1)
    observability_gain=0.4,       # Weight for observability (0-1)
    objective_type='trace'        # FIM objective to optimize
)
```

**Note:** Gains are automatically normalized to sum to 1.

#### Compute Velocity Command

```python
# Landmarks
uvw = np.random.randn(3, 8) * 10

# Current state
R = np.eye(3)
pursuer_position = np.array([0, 0, 0])
target_position = np.array([10, 5, 0])
target_velocity = np.array([1, 0, 0])

# Get velocity command
velocity = guidance.compute_guidance_velocity(
    uvw, R, pursuer_position, target_position, target_velocity
)

print(f"Pursuer velocity command: {velocity}")
print(f"Speed: {np.linalg.norm(velocity)}")
```

#### Simulate Complete Pursuit

```python
# Initial conditions
pursuer_pos_0 = np.array([0, 0, 0])
target_pos_0 = np.array([20, 0, 0])
target_vel = np.array([2, 0.5, 0])

# Simulate
results = guidance.simulate_pursuit(
    uvw, R,
    pursuer_pos_0, target_pos_0, target_vel,
    duration=10.0,  # seconds
    dt=0.1          # time step
)

# Analyze results
print(f"Final distance: {results['distances'][-1]}")
print(f"Minimum distance: {np.min(results['distances'])}")
print(f"Average condition number: {np.mean(results['observability_metrics']['condition_numbers'])}")

# Access trajectories
pursuer_traj = results['pursuer_trajectory']  # (n_steps, 3)
target_traj = results['target_trajectory']    # (n_steps, 3)
```

### Strategy Comparison

| Strategy | pursuit_gain | observability_gain | Characteristics |
|----------|--------------|-------------------|-----------------|
| Pure Pursuit | 1.0 | 0.0 | Fastest intercept, may have poor observability |
| Pure Observability | 0.0 | 1.0 | Best observability, may not intercept |
| Balanced | 0.5 | 0.5 | Trades off both objectives equally |
| Pursuit-Heavy | 0.8 | 0.2 | Prioritizes intercept, maintains some observability |
| Obs-Heavy | 0.2 | 0.8 | Prioritizes observability, slow pursuit |

### Choosing the Right Strategy

**When to use Pure Pursuit (1.0, 0.0):**
- Target must be intercepted quickly
- Localization accuracy is secondary
- Short-duration missions

**When to use Balanced (0.5, 0.5):**
- Both intercept and localization are important
- Medium-duration missions
- Good general-purpose choice

**When to use Observability-Heavy (0.2, 0.8):**
- Localization accuracy is critical
- Can tolerate longer intercept times
- Scientific missions requiring precise measurements

## Examples

### Example 1: Real-Time Observability Monitoring

```python
from src.guidance_law import GuidanceLaw
from src.fisher_information_matrix import FisherInformationAnalyzer
import numpy as np

# During flight, monitor observability
def monitor_and_guide(uvw, R, t, velocity):
    # Analyze current observability
    analysis = FisherInformationAnalyzer.analyze_observability(uvw, R, t)
    
    if analysis['FIM_metrics']['condition_number'] > 1e6:
        print("WARNING: Poor observability detected!")
        
        # Compute guidance to improve observability
        direction, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, t, objective_type='inverse_condition'
        )
        
        # Apply guidance
        accel = GuidanceLaw.compute_guidance_command(
            uvw, R, t, velocity,
            objective_type='inverse_condition',
            max_acceleration=2.0
        )
        
        return accel
    else:
        # Observability is good, continue current trajectory
        return np.zeros(3)
```

### Example 2: Trajectory Planning

```python
# Plan trajectory that maximizes observability
def plan_optimal_trajectory(uvw, R, start_pos, end_pos, n_waypoints=10):
    waypoints = []
    current_pos = start_pos.copy()
    
    for i in range(n_waypoints):
        # Compute direction to end
        to_end = end_pos - current_pos
        dist = np.linalg.norm(to_end)
        
        if dist < 1e-3:
            break
        
        # Compute observability-optimal direction
        obs_dir, _ = GuidanceLaw.compute_optimal_direction(
            uvw, R, current_pos, objective_type='trace'
        )
        
        # Blend with direct path
        blend = 0.7 * (to_end / dist) + 0.3 * obs_dir
        blend = blend / np.linalg.norm(blend)
        
        # Step
        step_size = min(dist, 1.0)
        current_pos = current_pos + blend * step_size
        waypoints.append(current_pos.copy())
    
    return waypoints
```

### Example 3: Multi-Objective Pursuit

```python
# Custom pursuit with mission-specific objectives
class CustomPursuitGuidance:
    def __init__(self):
        self.base_guidance = TwoAgentPursuitGuidance(
            pursuer_speed=10.0,
            pursuit_gain=0.6,
            observability_gain=0.4
        )
    
    def compute_velocity(self, uvw, R, pursuer_pos, target_pos, target_vel, fuel_remaining):
        # Adjust gains based on fuel
        if fuel_remaining < 0.2:
            # Low fuel: prioritize intercept
            self.base_guidance.pursuit_gain = 0.9
            self.base_guidance.observability_gain = 0.1
        else:
            # Normal operation: balanced
            self.base_guidance.pursuit_gain = 0.6
            self.base_guidance.observability_gain = 0.4
        
        return self.base_guidance.compute_guidance_velocity(
            uvw, R, pursuer_pos, target_pos, target_vel
        )
```

## Implementation Details

### Gradient Computation

The optimal direction is computed using finite differences:

```python
∇f(t) ≈ [f(t + εe_i) - f(t - εe_i)] / (2ε)
```

where `f` is the FIM objective and `e_i` are unit vectors along coordinate axes.

### Pursuit Direction

The pursuer uses proportional navigation to predict intercept:

1. Compute line-of-sight vector to target
2. Estimate time to intercept based on speed difference
3. Predict target position at intercept time
4. Compute direction to predicted position

### Observability Direction

Computed as the gradient of the FIM objective with respect to the agent's position:

```python
direction = ∇_t [objective(FIM(uvw, R, t))]
```

This points in the direction of maximum observability improvement.

## Performance Considerations

### Computational Cost

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Optimal direction | O(n) FIM computations | ~1-5 ms |
| Guidance command | O(n) | ~1-5 ms |
| Trajectory evaluation | O(k·n) | ~k ms (k = trajectory length) |
| Pursuit simulation | O(m·n) | ~10-100 ms (m = time steps) |

where n is the number of landmarks.

### Numerical Stability

- All directions are normalized to unit vectors
- FIM computation uses robust eigenvalue decomposition
- Determinant objectives use log-transform to avoid overflow
- Gradients use central differences for better accuracy

## Testing

Run the comprehensive test suite:

```bash
python test_guidance_law.py
```

Tests cover:
- Optimal direction computation (all objectives)
- Guidance command generation
- Trajectory evaluation
- Two-agent pursuit (all strategies)
- Mathematical properties
- Edge cases

Run the demonstration:

```bash
python demo_guidance_law.py
```

This shows:
- Universal guidance law with different objectives
- Two-agent pursuit with multiple strategies
- Mathematical properties validation
- Performance comparisons

## Theoretical Foundation

### Why These Objectives?

The four objectives correspond to classical optimal experiment design criteria:

1. **A-optimality (trace):** Minimizes average variance of parameter estimates
2. **D-optimality (determinant):** Minimizes volume of confidence ellipsoid
3. **E-optimality (min eigenvalue):** Minimizes maximum variance
4. **Condition minimization:** Ensures numerical stability

### Connection to Observability

Higher FIM eigenvalues → Lower estimation variance → Better observability

The guidance laws maximize eigenvalues in different ways:
- **Trace:** Increases average eigenvalue
- **Determinant:** Balances all eigenvalues (geometric mean)
- **Min eigenvalue:** Focuses on weakest direction
- **Condition:** Balances largest and smallest

## Limitations

### What the Guidance Law Cannot Do

1. **Cannot overcome fundamental unobservability**
   - If a direction is inherently unobservable (e.g., scale in monocular vision), guidance cannot fix it
   
2. **Cannot guarantee global optimality**
   - Uses gradient-based method (local optimization)
   - May converge to local minima
   
3. **Does not account for obstacles**
   - Computes pure observability guidance
   - Needs collision avoidance wrapper for real deployment

4. **Assumes static landmarks**
   - Landmark positions must be known and stationary
   - Moving landmarks require extended framework

### Recommended Practices

1. **Monitor condition numbers**
   - Alert when cond(FIM) > 1e6
   - Fall back to safe maneuver if needed

2. **Use appropriate objectives**
   - Start with trace (A-optimality) for general use
   - Switch to min eigenvalue if blind spots appear
   - Use condition minimization for numerical issues

3. **Blend with mission objectives**
   - Pure observability guidance may conflict with mission
   - Use weighted blending (as in pursuit scenario)

4. **Validate in simulation**
   - Test guidance laws in simulation first
   - Verify observability improvement
   - Check for unintended behaviors

## API Reference

### GuidanceLaw Class

**Static Methods:**

```python
GuidanceLaw.compute_optimal_direction(uvw, R, t, objective_type, step_size)
→ (direction, objective_value)
```

```python
GuidanceLaw.compute_guidance_command(uvw, R, t, current_velocity, objective_type, max_acceleration, dt)
→ acceleration
```

```python
GuidanceLaw.evaluate_trajectory(uvw, trajectory, R, objective_type)
→ {'objectives', 'condition_numbers', 'determinants'}
```

### TwoAgentPursuitGuidance Class

**Constructor:**
```python
TwoAgentPursuitGuidance(pursuer_speed, pursuit_gain, observability_gain, objective_type)
```

**Methods:**
```python
compute_target_state(target_position_0, target_velocity, time)
→ (position, velocity)
```

```python
compute_pursuit_direction(pursuer_position, target_position, target_velocity)
→ direction
```

```python
compute_observability_direction(uvw, R, pursuer_position)
→ direction
```

```python
compute_guidance_velocity(uvw, R, pursuer_position, target_position, target_velocity)
→ velocity
```

```python
simulate_pursuit(uvw, R, pursuer_position_0, target_position_0, target_velocity, duration, dt)
→ {'time', 'pursuer_trajectory', 'target_trajectory', 'pursuer_velocities', 'distances', 'observability_metrics'}
```

## Conclusion

The guidance laws derived from the Fisher Information Matrix provide a rigorous, theoretically-grounded approach to trajectory planning for bearing-only cooperative localization. By optimizing observability while achieving mission objectives, these guidance laws enable more accurate and robust localization in GPS-denied environments.

**Key Takeaways:**
1. ✓ Universal guidance law supports multiple FIM-based objectives
2. ✓ Two-agent pursuit balances intercept and observability
3. ✓ Guidance laws are computationally efficient (suitable for real-time use)
4. ✓ Flexible framework allows mission-specific customization
5. ✓ Validated through comprehensive testing and demonstrations

---

**Repository:** https://github.com/xiahaa/taes-bearing-only-cooperative-localization

**See Also:**
- [FIM_ANALYSIS.md](FIM_ANALYSIS.md) - Fisher Information Matrix fundamentals
- [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md) - Research context and results
- [demo_guidance_law.py](demo_guidance_law.py) - Working demonstrations
