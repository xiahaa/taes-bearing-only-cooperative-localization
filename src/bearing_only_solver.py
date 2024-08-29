import numpy as np
import os
import logging
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt
from typing import Optional, Tuple
import cvxpy as cp

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def load_simulation_data(filename: str) -> dict:
    """
    Load simulation data from a file.

    Args:
        filename (str): The path to the file containing the simulation data.

    Returns:
        dict: A dictionary containing the loaded simulation data. The dictionary
              has the following keys:
              - "p1": A 2D numpy array of shape (3, n) representing the p1 data.
              - "p2": A 2D numpy array of shape (3, n) representing the p2 data.
              - "bearing": A 2D numpy array of shape (3, n) representing the bearing data.
              - "Rgt": A 2D numpy array of shape (3, 3) representing the Rgt data.
              - "tgt": A 1D numpy array of shape (n,) representing the tgt data.
    """
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
class bearing_linear_solver():
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_reduced_Ab_matrix(uA: np.ndarray, vA: np.ndarray, wA: np.ndarray, phi: np.ndarray, theta: np.ndarray,
                                k: int, xB: np.ndarray, yB: np.ndarray, zB: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reduced A and b matrices for the bearing-only solver.

        Parameters:
        - uA (np.ndarray): Array of uA values. Shape: (k,)
        - vA (np.ndarray): Array of vA values. Shape: (k,)
        - wA (np.ndarray): Array of wA values. Shape: (k,)
        - phi (np.ndarray): Array of phi values. Shape: (k,)
        - theta (np.ndarray): Array of theta values. Shape: (k,)
        - k (int): Number of elements in the arrays.
        - xB (np.ndarray): Array of xB values. Shape: (k,)
        - yB (np.ndarray): Array of yB values. Shape: (k,)
        - zB (np.ndarray): Array of zB values. Shape: (k,)
        - R (np.ndarray): Array of R values. Shape: (3, 3)

        Returns:
        - A (np.ndarray): The reduced A matrix. Shape: (2 * k, 3)
        - b (np.ndarray): The reduced b matrix. Shape: (2 * k,)
        """
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
            logger.debug(f'Residual: {residual}')

            b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i] - residual[0]
            b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i] -  residual[1]

        return A, b

    @staticmethod
    def compute_A_matrix(uA: np.ndarray, vA: np.ndarray, wA: np.ndarray, phi: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the A matrix for bearing-only solver.

        Parameters:
        uA (np.ndarray): Array of uA values. Shape: (k,)
        vA (np.ndarray): Array of vA values. Shape: (k,)
        wA (np.ndarray): Array of wA values. Shape: (k,)
        phi (np.ndarray): Array of phi values. Shape: (k,)
        theta (np.ndarray): Array of theta values. Shape: (k,)
        k (int): Number of elements in the arrays.

        Returns:
        A (np.ndarray): A matrix of shape (2 * k, 12).
        """
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

    @staticmethod
    def compute_b_vector(xB: np.ndarray, yB: np.ndarray, zB: np.ndarray, phi: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the b vector for the bearing-only solver.

        Parameters:
        - xB (np.ndarray): x-coordinates of the bearings. Shape: (k,)
        - yB (np.ndarray): y-coordinates of the bearings. Shape: (k,)
        - zB (np.ndarray): z-coordinates of the bearings. Shape: (k,)
        - phi (np.ndarray): azimuth angles of the bearings. Shape: (k,)
        - theta (np.ndarray): elevation angles of the bearings. Shape: (k,)
        - k (int): number of bearings.

        Returns:
        - b (np.ndarray): computed b vector. Shape: (2 * k,)
        """
        b = np.zeros(2 * k)

        for i in range(k):
            b[2 * i] = -np.cos(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * xB[i]
            b[2 * i + 1] = -np.sin(theta[i]) * np.cos(phi[i]) * zB[i] + np.sin(phi[i]) * yB[i]

        return b

    @staticmethod
    def orthogonal_procrustes(Rgt: np.ndarray) -> np.ndarray:
        """
        Perform the orthogonal Procrustes analysis to find the optimal rotation matrix.

        Parameters:
        - Rgt (np.ndarray): A 2D numpy array of shape (3, 3) representing the input rotation matrix.

        Returns:
        - Ropt (np.ndarray): A 2D numpy array of shape (3, 3) representing the optimal rotation matrix.
        """
        U, S, Vt = np.linalg.svd(Rgt)

        D = np.dot(Vt.T, U.T)
        if np.linalg.det(D) < 0:
            Vt[-1, :] *= -1
            D = np.dot(Vt.T, U.T)

        Ropt = D.T
        return Ropt

    @staticmethod
    def solve(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the bearing-only problem given the sensor positions, target positions, and bearing angles.

        Parameters:
        - uvw (np.ndarray): Array of sensor positions. Shape: (3, n)
        - xyz (np.ndarray): Array of target positions. Shape: (3, n)
        - bearing (np.ndarray): Array of bearing angles. Shape: (3, n)

        Returns:
        - R (np.ndarray): The rotation matrix. Shape: (3, 3)
        - t (np.ndarray): The translation vector. Shape: (3,)
        """
        bearing_angle = np.zeros((2, bearing.shape[1]))
        for i in range(bearing.shape[1]):
            vec = bearing[:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])

        A = bearing_linear_solver.compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        b = bearing_linear_solver.compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])

        # Solve for x using least squares
        from scipy.linalg import solve, lstsq
        x = solve(A, b)
        logger.debug(f'Solution x: {x}')

        R = x[:9].reshape(3, 3)
        R = bearing_linear_solver.orthogonal_procrustes(R)
        # t = x[9:]
        logger.debug(f'R: {R}')

        A1,b1 = bearing_linear_solver.compute_reduced_Ab_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1], xyz[0,:], xyz[1,:], xyz[2,:], R)

        x1, res, rnk, s = lstsq(A1, b1)
        logger.debug(f'Solution x: {x1}')
        t = x1[:3]
        logger.debug(f't: {t}')

        return R, t

    @staticmethod
    def solve_with_sdp_sdr(uvw: np.ndarray, xyz: np.ndarray, bearing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        bearing_angle = np.zeros((2, bearing.shape[1]))
        for i in range(bearing.shape[1]):
            vec = bearing[:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])


        A = bearing_linear_solver.compute_A_matrix(uvw[0,:], uvw[1,:], uvw[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])
        b = bearing_linear_solver.compute_b_vector(xyz[0,:], xyz[1,:], xyz[2,:], bearing_angle[1,:], bearing_angle[0,:], bearing_angle.shape[1])

        P1 = np.hstack([A, b.reshape(-1,1)])
        # P is 13 x 13
        P = P1.T@P1

        Q = []
        ndim = P.shape[0]
        """
        generate matrix for the following constraints
        [ a1^2 + a2^2 + a3^2, - 1
          a1*a4 + a2*a5 + a3*a6,
          a1*a7 + a2*a8 + a3*a9]
        [   a4^2 + a5^2 + a6^2 - 1,
            a4*a7 + a5*a8 + a6*a9]
        [   a7^2 + a8^2 + a9^2 - 1]
        """
        Q1 = np.zeros((ndim,ndim))
        Q1[[0,1,2],[0,1,2]] = 1; Q1[12,12] = -1
        Q2 = np.zeros((ndim,ndim))
        Q2[[0,1,2],[3,4,5]] = 1
        Q3 = np.zeros((ndim,ndim))
        Q3[[0,1,2],[6,7,8]] = 1
        Q4 = np.zeros((ndim,ndim))
        Q4[[3,4,5],[3,4,5]] = 1; Q4[12,12] = -1
        Q5 = np.zeros((ndim,ndim))
        Q5[[3,4,5],[6,7,8]] = 1
        Q6 = np.zeros((ndim,ndim))
        Q6[[6,7,8],[6,7,8]] = 1; Q6[12,12] = -1
        Q.append(Q1)
        Q.append(Q2)
        Q.append(Q3)
        Q.append(Q4)
        Q.append(Q5)
        Q.append(Q6)

        """
        generate matrix for the following constraints
        [ a1^2 + a4^2 + a7^2 - 1,
          a1*a2 + a4*a5 + a7*a8,
          a1*a3 + a4*a6 + a7*a9]
        [   a2^2 + a5^2 + a8^2 - 1,
            a2*a3 + a5*a6 + a8*a9]
        [   a3^2 + a6^2 + a9^2 - 1]
        """
        Q7 = np.zeros((ndim,ndim))
        Q7[[0,3,6],[0,3,6]] = 1; Q7[12,12] = -1
        Q8 = np.zeros((ndim,ndim))
        Q8[[0,3,6],[1,4,7]] = 1
        Q9 = np.zeros((ndim,ndim))
        Q9[[0,3,6],[2,5,8]] = 1
        Q10 = np.zeros((ndim,ndim))
        Q10[[1,4,7],[1,4,7]] = 1; Q10[12,12] = -1
        Q11 = np.zeros((ndim,ndim))
        Q11[[1,4,7],[2,5,8]] = 1
        Q12 = np.zeros((ndim,ndim))
        Q12[[2,5,8],[2,5,8]] = 1; Q12[12,12] = -1
        Q.append(Q7)
        Q.append(Q8)
        Q.append(Q9)
        Q.append(Q10)
        Q.append(Q11)
        Q.append(Q12)
        """
        generate matrix for the following constraints
        [a1 - a5*a9 + a6*a8,
         a2 + a4*a9 - a6*a7,
         a3 - a4*a8 + a5*a7]
        """
        Q13 = np.zeros((ndim,ndim))
        Q13[0,12] = -1; Q13[4,8] = -1; Q13[5,7] = 1
        Q14 = np.zeros((ndim,ndim))
        Q14[1,12] = -1; Q14[3,8] =  1; Q14[5,6] = -1
        Q15 = np.zeros((ndim,ndim))
        Q15[2,12] = -1; Q15[3,7] = -1; Q15[4,6] = 1
        Q.append(Q13)
        Q.append(Q14)
        Q.append(Q15)
        """
        generate matrix for the following constraints
        [a4 + a2*a9 - a3*a8,
         a5 - a1*a9 + a3*a7,
         a6 + a1*a8 - a2*a7]
        """
        Q16 = np.zeros((ndim,ndim))
        Q16[3,12] = -1; Q16[1,8] =  1; Q16[2,7] = -1
        Q17 = np.zeros((ndim,ndim))
        Q17[4,12] = -1; Q17[0,8] =  -1; Q17[2,6] = 1
        Q18 = np.zeros((ndim,ndim))
        Q18[5,12] = -1; Q18[0,7] = 1; Q18[1,6] = -1
        Q.append(Q16)
        Q.append(Q17)
        Q.append(Q18)
        """
        generate matrix for the following constraints
        [a7 - a2*a6 + a3*a5,
         a8 + a1*a6 - a3*a4,
         a9 - a1*a5 + a2*a4]
        """
        Q19 = np.zeros((ndim,ndim))
        Q19[6,12] = -1; Q19[1,5] = -1; Q19[2,4] = 1
        Q20 = np.zeros((ndim,ndim))
        Q20[7,12] = -1; Q20[0,5] =  1; Q20[2,3] = -1
        Q21 = np.zeros((ndim,ndim))
        Q21[8,12] = -1; Q21[0,4] = -1; Q21[1,3] = 1
        Q.append(Q19)
        Q.append(Q20)
        Q.append(Q21)
        Q22 = np.zeros((ndim, ndim))
        Q22[12,12] = 1
        Q.append(Q22)
        # [  a1^2,  a1*a2,  a1*a3,  a1*a4,  a1*a5,  a1*a6,  a1*a7,  a1*a8,  a1*a9,  a1*a10,  a1*a11,  a1*a12,  -a1]
        # [ a1*a2,   a2^2,  a2*a3,  a2*a4,  a2*a5,  a2*a6,  a2*a7,  a2*a8,  a2*a9,  a2*a10,  a2*a11,  a2*a12,  -a2]
        # [ a1*a3,  a2*a3,   a3^2,  a3*a4,  a3*a5,  a3*a6,  a3*a7,  a3*a8,  a3*a9,  a3*a10,  a3*a11,  a3*a12,  -a3]
        # [ a1*a4,  a2*a4,  a3*a4,   a4^2,  a4*a5,  a4*a6,  a4*a7,  a4*a8,  a4*a9,  a4*a10,  a4*a11,  a4*a12,  -a4]
        # [ a1*a5,  a2*a5,  a3*a5,  a4*a5,   a5^2,  a5*a6,  a5*a7,  a5*a8,  a5*a9,  a5*a10,  a5*a11,  a5*a12,  -a5]
        # [ a1*a6,  a2*a6,  a3*a6,  a4*a6,  a5*a6,   a6^2,  a6*a7,  a6*a8,  a6*a9,  a6*a10,  a6*a11,  a6*a12,  -a6]
        # [ a1*a7,  a2*a7,  a3*a7,  a4*a7,  a5*a7,  a6*a7,   a7^2,  a7*a8,  a7*a9,  a7*a10,  a7*a11,  a7*a12,  -a7]
        # [ a1*a8,  a2*a8,  a3*a8,  a4*a8,  a5*a8,  a6*a8,  a7*a8,   a8^2,  a8*a9,  a8*a10,  a8*a11,  a8*a12,  -a8]
        # [ a1*a9,  a2*a9,  a3*a9,  a4*a9,  a5*a9,  a6*a9,  a7*a9,  a8*a9,   a9^2,  a9*a10,  a9*a11,  a9*a12,  -a9]
        # [a1*a10, a2*a10, a3*a10, a4*a10, a5*a10, a6*a10, a7*a10, a8*a10, a9*a10,   a10^2, a10*a11, a10*a12, -a10]
        # [a1*a11, a2*a11, a3*a11, a4*a11, a5*a11, a6*a11, a7*a11, a8*a11, a9*a11, a10*a11,   a11^2, a11*a12, -a11]
        # [a1*a12, a2*a12, a3*a12, a4*a12, a5*a12, a6*a12, a7*a12, a8*a12, a9*a12, a10*a12, a11*a12,   a12^2, -a12]
        # [   -a1,    -a2,    -a3,    -a4,    -a5,    -a6,    -a7,    -a8,    -a9,    -a10,    -a11,    -a12,    1]
        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((ndim,ndim), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        constraints += [
            cp.trace(Q[i] @ X) == 0 for i in range(21)
        ]
        constraints += [cp.trace(Q[-1] @ X) == 1]
        # logger.warning(cp.trace(Q[-1]@X))
        logger.debug(constraints)
        prob = cp.Problem(cp.Minimize(cp.trace(P @ X)),
                  constraints)
        prob.solve(solver = cp.MOSEK)

        # Print result.
        logger.debug("The optimal value is", prob.value)
        logger.debug("A solution X is")
        logger.debug(X.value)
        sol = X.value

        # Perform SVD on the solution matrix
        U, S, Vt = np.linalg.svd(sol)

        # Keep only the largest singular value
        S_rank1 = np.zeros_like(S)
        S_rank1[0] = S[0]

        # Construct the rank-1 approximation
        sol_rank1 = U[:, 0:1] @ np.diag(S_rank1[0:1]) @ Vt[0:1, :]

        # Log the rank-1 approximation
        logger.debug("Rank-1 approximation of the solution X is")
        logger.debug(sol_rank1)

        xsol = np.sqrt(np.diag(sol_rank1))

        R = xsol[:9].reshape(3, 3)
        t = xsol[9:-1]
        R = bearing_linear_solver.orthogonal_procrustes(R)
        logger.debug(f'Solution xsol: {xsol}')

        return R, t




class bgpnp():
    def __init__(self) -> None:
        pass

    @staticmethod
    def solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the Bearing Generalized Perspective-n-Point (BGPnP) algorithm.

        Args:
            p1 (np.ndarray): The 2D coordinates of the points in the first image.
            p2 (np.ndarray): The 2D coordinates of the points in the second image.
            bearing (np.ndarray): The bearing angles of the points.
            sol_iter (bool): Flag indicating whether to perform iterative refinement.

        Returns:
            tuple: A tuple containing the rotation matrix (R), translation vector (T), and error (err).
        """
        M, b, Alph, Cw = bgpnp.prepare_data(p1, bearing, p2)
        Km = bgpnp.kernel_noise(M, b, dimker=4)
        R, t, err = bgpnp.KernelPnP(Cw, Km, dims=4, sol_iter=True)

        return R, t, err

    @staticmethod
    def define_control_points() -> np.ndarray:
        """
        Define control points.

        Returns:
            np.ndarray: A 4x3 matrix of control points.
        """
        Cw = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        return Cw

    @staticmethod
    def compute_alphas(Xw:np.ndarray, Cw:np.ndarray) -> np.ndarray:
        """
        Compute alphas (linear combination of the control points to represent the 3D points).

        Args:
            Xw (np.ndarray): 3D points, shape (n, 3).
            Cw (np.ndarray): Control points, shape (4, 3).

        Returns:
            np.ndarray: Alphas.
        """
        n = Xw.shape[0]  # number of 3D points
        logger.debug(f'Xw: {Xw.shape}')
        logger.debug(f'Cw: {Cw.shape}')

        # Generate auxiliary matrix to compute alphas
        C = np.vstack((Cw.T, np.ones((1, 4)))) # 4x4
        X = np.vstack((Xw.T, np.ones((1, n)))) # 4xn

        logger.debug(f'C: {C.shape}')
        logger.debug(f'C: {C}')

        Alph_ = np.linalg.inv(C) @ X # 4xn

        Alph = Alph_.T # nx4
        return Alph

    @staticmethod
    def myProcrustes(X, Y):
        """
        Perform Procrustes analysis to find the best transformation between two sets of points.

        Parameters:
        - X: Dictionary containing the reference points.
        - Y: Array-like object containing the target points.

        Returns:
        - R: The rotation matrix.
        - b: The scaling factor.
        - mc: The translation vector.

        The function calculates the best rotation, scaling, and translation that aligns the target points (Y)
        with the reference points (X). It uses the Procrustes analysis method to find the optimal transformation.

        The reference points (X) should be provided as a dictionary with the following keys:
        - 'nP': The normalized reference points.
        - 'norm': The normalization factor.
        - 'mP': The mean of the reference points.

        The target points (Y) should be an array-like object with shape (n, d), where n is the number of points
        and d is the number of dimensions.

        The function returns the rotation matrix (R), scaling factor (b), and translation vector (mc) that
        transform the target points (Y) to align with the reference points (X).

        Note: The function assumes that the number of dimensions in the target points (Y) is the same as the
        number of dimensions in the reference points (X).

        """
        dims = Y.shape[1]
        mY = np.mean(Y, axis=1)
        cY = Y - mY.reshape(Y.shape[0], 1)
        ncY = np.linalg.norm(cY)
        tcY = cY / ncY

        A = np.dot(X['nP'], tcY.T)
        L, D, Mt = np.linalg.svd(A)

        logger.debug(f'A: {A}')
        logger.debug(f'L: {L}')
        logger.debug(f'D: {D}')
        logger.debug(f'M: {Mt.T}')

        R = Mt.T @ np.diag([1, 1, np.sign(np.linalg.det(Mt.T @ L.T))]) @ L.T
        logger.debug(f'R: {R}')

        b = np.sum(np.diag(D)) * X['norm'] / ncY
        logger.debug(f'b: {b}')
        c = X['mP'] - np.dot(b, np.dot(R.T, mY))
        logger.debug(f'c: {c}')

        mc = np.tile(c, (dims,1)).T
        logger.debug(f'mc: {mc}')

        return R, b, mc

    def KernelPnP(Cw: np.ndarray, Km: np.ndarray, dims: int = 4, sol_iter: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes the Kernel Perspective-n-Point (KernelPnP) algorithm.

        Args:
            Cw (numpy.ndarray): The 3D world coordinates of the points.
            Km (numpy.ndarray): The kernel matrix.
            dims (int): The number of dimensions.
            sol_iter (bool): Flag indicating whether to perform iterative refinement.

        Returns:
            tuple: A tuple containing the rotation matrix (R), translation vector (T), and error (err).
        """
        vK = np.reshape(Km[:, -1], (dims, -1)).T
        logger.debug(f'vK: {vK}')
        # precomputations
        X = {}
        X['P'] = Cw.T
        logger.debug(X['P'])
        X['mP'] = np.mean(X['P'], axis=1)
        logger.debug(X['mP'])
        X['cP'] = X['P'] - X['mP'].reshape(3, 1)
        logger.debug(X['cP'])

        X['norm'] = np.linalg.norm(X['cP'])
        logger.debug(X['norm'])
        X['nP'] = X['cP'] / X['norm']
        logger.debug(X['nP'])

        # procrustes solution for the first kernel vector
        R, b, mc = bgpnp.myProcrustes(X, vK)

        solV = b * vK
        solR = R
        solmc = mc

        # procrustes solution using 4 kernel eigenvectors
        err = np.inf
        if sol_iter:
            n_iterations = 500
            for iter in range(n_iterations):
                # projection of previous solution into the null space
                A = R @ (-mc + X['P'])
                abcd = np.linalg.lstsq(Km, A.T.flatten(), rcond=None)[0]
                newV = np.reshape(Km @ abcd, (dims, -1)).T

                logger.debug(f'Iteration: {iter}')
                logger.debug(f'A: {A}')
                logger.debug(f'abcd: {abcd}')
                logger.debug(f'newV: {newV}')

                # euclidean error
                newerr = np.linalg.norm(R.T @ newV + mc - X['P'],2)
                logger.debug(f'newerr: {newerr}')

                if ((newerr > err) and (iter > 2)) or newerr < 1e-6:
                    break
                else:
                    # procrustes solution
                    R, b, mc = bgpnp.myProcrustes(X, newV)
                    solV = b * newV

                    solmc = mc
                    solR = R
                    err = newerr

        R = solR
        mV = np.mean(solV, axis=1)

        T = mV - R @ X['mP']
        logger.info(f'Final solution: {R}, {T}')
        return R, T, err

    @staticmethod
    def kernel_noise(M: np.ndarray, b: np.ndarray, dimker: int = 4) -> np.ndarray:
        """
        Computes the kernel noise matrix for a given input matrix M and vector b.

        Parameters:
        - M: Input matrix of shape (3n, 12)
        - b: Input vector of shape (3n,)
        - dimker: Dimension of the kernel noise matrix (default: 4)

        Returns:
        - K: Kernel noise matrix of shape (12, dimker)
        """
        K = np.zeros((M.shape[1], dimker))
        U, S, V = np.linalg.svd(M)
        V = V.T
        logger.debug(f'U: {V}')

        K[:, 0:dimker-1] = V[:, -dimker+1:]
        logger.debug(f'K: {K}')
        logger.debug(f'np.linalg.pinv(M) @ b: {np.linalg.pinv(M) @ b}')
        if np.linalg.matrix_rank(M) < 12:
            K[:, -1] = np.linalg.pinv(M) @ b
        else:
            K[:, -1] = np.linalg.pinv(M) @ b

        return K


    @staticmethod
    def prepare_data(p: np.ndarray, bearing: np.ndarray, pb: np.ndarray, Cw: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for the bearing-only solver.

        Args:
            p (np.ndarray): Array of sensor positions.
            bearing (np.ndarray): Array of bearing measurements.
            pb (np.ndarray): Array of target positions.
            Cw (np.ndarray, optional): Array of control points. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the computed matrices M and b, the array of alphas Alph, and the array of control points Cw.
        """
        if Cw is None:
            logger.info('Control points not provided. Defining control points.')
            Cw = bgpnp.define_control_points()

        logger.debug(f'Cw: {Cw}')
        Alph = bgpnp.compute_alphas(p, Cw)
        M, b = bgpnp.compute_Mb(bearing, Alph, pb)

        return M, b, Alph, Cw

    @staticmethod
    def skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
        """
        Generate the skew-symmetric matrix for a vector.

        Args:
            v (np.ndarray): A vector of shape (3,).

        Returns:
            np.ndarray: A 3x3 skew-symmetric matrix.
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def compute_Mb(bearing: np.ndarray, Alph: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the M and b matrix.

        Args:
            bearing (np.ndarray): Bearing vector, shape (n, 3).
            Alph (np.ndarray): Alphas, shape (n, 4).
            p2 (np.ndarray): 3D points, shape (n, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - M (np.ndarray): The M matrix of shape (3n, 12).
                - b (np.ndarray): The b vector of shape (3n,).
        """
        M = np.zeros((3 * bearing.shape[0], 12))
        b = np.zeros(3 * bearing.shape[0])
        for i in range(bearing.shape[0]):
            S = bgpnp.skew_symmetric_matrix(bearing[i])
            logger.debug(f'bearing: {bearing[i]}')
            logger.debug(f'S: {S}')
            logger.debug(f'Alph: {Alph[i]}')

            M[3 * i:3 * i + 3, :] = np.kron(Alph[i], S)
            b[3 * i:3 * i + 3] = S.dot(p2[i])

        return M, b


def bearing_only_solver(folder: str, file: str):
    """
    Solve the bearing-only problem given a folder and a file.

    Args:
        folder (str): The folder path where the files are located.
        file (str): The file name to search for in the folder.

    Returns:
        None
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder) if file in f]

    solver = bearing_linear_solver()

    for f in files:
        data = load_simulation_data(f)
        logger.debug(data["p1"])
        logger.debug(data["p2"].shape)
        logger.debug(data["Rgt"])
        logger.debug(data["tgt"])

        uvw = data["p1"]
        xyz = data["p2"]
        bearing = data["bearing"]

        R, t = solver.solve(uvw, xyz, bearing)

        logger.info(f'Solution R: {R}')
        logger.info(f't: {t}')

        R1, t1 = solver.solve_with_sdp_sdr(uvw, xyz, bearing)
        logger.info(f'Solution R: {R1}')
        logger.info(f't: {t1}')


if __name__ == "__main__":
    bearing_only_solver('../taes/', 'simu_')
