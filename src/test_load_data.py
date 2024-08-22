import numpy as np
from load_data import angle2dcm

# Test case 1: Check if the function returns a 3x3 numpy matrix
yaw = 0.1
pitch = 0.2
roll = 0.3
Rnav2body = angle2dcm(yaw, pitch, roll)
assert isinstance(Rnav2body, np.matrix)
assert Rnav2body.shape == (3, 3)

# Test case 2: Check if the function returns the correct transformation matrix
yaw = 0.1
pitch = 0.2
roll = 0.3
expected_result = np.matrix([
    [0.93629336, -0.27509585, 0.21835066],
    [0.28962948, 0.95642505, -0.03617541],
    [-0.19866933, 0.0978434, 0.97517033]
])
Rnav2body = angle2dcm(yaw, pitch, roll)
assert np.allclose(Rnav2body, expected_result)

# Test case 3: Check if the function handles input angles in degrees
yaw = 90
pitch = 15
roll = 0
expected_result = np.matrix([
    [0, -0.25881905, 0.96592583],
    [0, 0.96592583, 0.25881905],
    [-1, 0, 0]
])
Rnav2body = angle2dcm(yaw, pitch, roll, input_units='deg')
assert np.allclose(Rnav2body, expected_result)

# Test case 4: Check if the function handles a different rotation sequence
yaw = 0.1
pitch = 0.2
roll = 0.3
rotation_sequence = '123'
expected_result = np.nan
Rnav2body = angle2dcm(yaw, pitch, roll, rotation_sequence=rotation_sequence)
assert np.isnan(Rnav2body)

print("All tests passed!")

