import numpy as np
import os
import logging
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

def compute_reduced_Ab_matrix(uA, vA, wA, phi, theta, k, xB, yB, zB, R):
    r = R.flatten()
    A = np.zeros((2 * k, 3))
    b = np.zeros(2 * k)
    
    for i in range(k):
        A[2 * i, 0] = np.sin(phi[i])
        A[2 * i, 1] = 0
        A[2 * i, 2] = -np.cos(theta[i]) * np.cos(phi[i])
        
        A[2 * i + 1, 0] = 0
        A[2 * i + 1, 1] = np.sin(phi[i])
        A[2 * i + 1, 2] = -np.sin(theta[i]) * np.cos(phi[i])
        
        AA = np.zeros((2, 9))
        AA[0, 0] = uA[i] * np.sin(phi[i])
        AA[0, 1] = vA[i] * np.sin(phi[i])
        AA[0, 2] = wA[i] * np.sin(phi[i])
        AA[0, 3] = 0
        AA[0, 4] = 0
        AA[0, 5] = 0
        AA[0, 6] = -uA[i] * np.cos(theta[i]) * np.cos(phi[i])
        AA[0, 7] = -vA[i] * np.cos(theta[i]) * np.cos(phi[i])
        AA[0, 8] = -wA[i] * np.cos(theta[i]) * np.cos(phi[i])
        
        AA[0 + 1, 0] = 0
        AA[0 + 1, 1] = 0
        AA[0 + 1, 2] = 0
        AA[0 + 1, 3] = uA[i] * np.sin(phi[i])
        AA[0 + 1, 4] = vA[i] * np.sin(phi[i])
        AA[0 + 1, 5] = wA[i] * np.sin(phi[i])
        AA[0 + 1, 6] = -uA[i] * np.sin(theta[i]) * np.cos(phi[i])
        AA[0 + 1, 7] = -vA[i] * np.sin(theta[i]) * np.cos(phi[i])
        AA[0 + 1, 8] = -wA[i] * np.sin(theta[i]) * np.cos(phi[i])
        
        residual = AA.dot(r)
        print(f'Residual: {residual}')
        
        b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i] - residual[0]
        b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i] -  residual[1]
        
    return A, b


def compute_A_matrix(uA, vA, wA, phi, theta, k):
    A = np.zeros((2 * k, 12))

    for i in range(k):
        A[2 * i, 0] = uA[i] * np.sin(phi[i])
        A[2 * i, 1] = vA[i] * np.sin(phi[i])
        A[2 * i, 2] = wA[i] * np.sin(phi[i])
        A[2 * i, 3] = 0
        A[2 * i, 4] = 0
        A[2 * i, 5] = 0
        A[2 * i, 6] = -uA[i] * np.cos(theta[i]) * np.cos(phi[i])
        A[2 * i, 7] = -vA[i] * np.cos(theta[i]) * np.cos(phi[i])
        A[2 * i, 8] = -wA[i] * np.cos(theta[i]) * np.cos(phi[i])
        A[2 * i, 9] = np.sin(phi[i])
        A[2 * i, 10] = 0
        A[2 * i, 11] = -np.cos(theta[i]) * np.cos(phi[i])

        A[2 * i + 1, 0] = 0
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = 0
        A[2 * i + 1, 3] = uA[i] * np.sin(phi[i])
        A[2 * i + 1, 4] = vA[i] * np.sin(phi[i])
        A[2 * i + 1, 5] = wA[i] * np.sin(phi[i])
        A[2 * i + 1, 6] = -uA[i] * np.sin(theta[i]) * np.cos(phi[i])
        A[2 * i + 1, 7] = -vA[i] * np.sin(theta[i]) * np.cos(phi[i])
        A[2 * i + 1, 8] = -wA[i] * np.sin(theta[i]) * np.cos(phi[i])
        A[2 * i + 1, 9] = 0
        A[2 * i + 1, 10] = np.sin(phi[i])
        A[2 * i + 1, 11] = -np.sin(theta[i]) * np.cos(phi[i])

    return A

def compute_b_vector(xB, yB, zB, phi, theta, k):
    b = np.zeros(2 * k)

    for i in range(k):
        b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i]
        b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i]

    return b


import numpy as np

def load_simulation_data(filename: str):
    with open(filename, 'r') as f:
        data = {}
        lines = f.readlines()

        # Read p1
        p1_lines = lines[0].strip().split(' ')
        p1 = np.array([float(x) for x in p1_lines]).reshape(-1,3)

        # Read p2
        p2_lines = lines[1].strip().split(' ')
        p2 = np.array([float(x) for x in p2_lines]).reshape(-1,3)

        # Read bearing
        bearing_lines = lines[2].strip().split(' ')
        bearing = np.array([float(x) for x in bearing_lines]).reshape(-1,3)

        # Read Rgt
        Rgt_lines = lines[3].strip().split(' ')
        Rgt = np.array([float(x) for x in Rgt_lines]).reshape(3, 3)

        # Read tgt
        tgt_lines = lines[4].strip().split(' ')
        tgt = np.array([float(x) for x in tgt_lines])

        data["p1"] = p1.T
        data["p2"] = p2.T
        data["bearing"] = bearing.T
        data["Rgt"] = Rgt
        data["tgt"] = tgt
        
    return data

def orthogonal_procrustes(Rgt):
    U, S, Vt = np.linalg.svd(Rgt)
    
    D = np.dot(Vt.T, U.T)
    if np.linalg.det(D) < 0:
        Vt[-1, :] *= -1
        D = np.dot(Vt.T, U.T)
    
    Ropt = D.T
    return Ropt


def bearing_only_solver(foler: str, file: str):
    files = [os.path.join(foler, f) for f in os.listdir(foler) if file in f]

    for f in files:
        data = load_simulation_data(f)
        logger.debug(data["p1"])
        logger.debug(data["p2"].shape)
        logger.debug(data["Rgt"])
        logger.debug(data["tgt"])
        
        uvw = data["p1"]
        xyz = data["p2"]
        
        bearing_angle = np.zeros((2, data["bearing"].shape[1]))
        for i in range(data["bearing"].shape[1]):
            vec = data["bearing"][:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])           


        A = compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        b = compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        
        logger.debug(A.shape)
        
        # Solve for x using least squares
        from scipy.linalg import solve, lstsq
        x = solve(A, b)
        logger.debug(f'Solution x: {x}')

        R = x[:9].reshape(3, 3)
        R = orthogonal_procrustes(R)
        # t = x[9:]
        logger.debug(f'R: {R}')
        # logger.debug(f't: {t}')
        
        A1,b1 = compute_reduced_Ab_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1], xyz[0,:], xyz[1,:], xyz[2,:], R)
        
        x1, res, rnk, s = lstsq(A1, b1)
        logger.debug(f'Solution x: {x1}')
        t = x1[:3]        
        logger.debug(f't: {t}')

    
if __name__ == "__main__":
    bearing_only_solver('../taes/', 'simu_')
    
    