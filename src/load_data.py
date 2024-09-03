import scipy.io as sio
from typing import List, Optional
import numpy as np
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def refine_timestamp(imudata, rate: float = 200):
    for i in range(0, len(imudata)-1, 2):
        t1 = imudata[i,0]
        t2 = imudata[i+1,0]
        if t1 == t2:
            imudata[i+1,0] = t1 + 1 / rate
    return imudata

def angle2dcm(yaw, pitch, roll, input_units='rad', rotation_sequence='321'):
    """
    Returns a transformation matrix (aka direction cosine matrix or DCM) which
    transforms from navigation to body frame.  Other names commonly used,
    besides DCM, are `Cbody2nav` or `Rbody2nav`.  The rotation sequence
    specifies the order of rotations when going from navigation-frame to
    body-frame.  The default is '321' (i.e Yaw -> Pitch -> Roll).
    Parameters
    ----------
    yaw   : yaw angle, units of input_units.
    pitch : pitch angle, units of input_units.
    roll  : roll angle , units of input_units.
    input_units: units for input angles {'rad', 'deg'}, optional.
    rotationSequence: assumed rotation sequence {'321', others can be
                                                implemented in the future}.
    Returns
    -------
    Rnav2body: 3x3 transformation matrix (numpy matrix data type).  This can be
               used to convert from navigation-frame (e.g NED) to body frame.

    Notes
    -----
    Since Rnav2body is a proper transformation matrix, the inverse
    transformation is simply the transpose.  Hence, to go from body->nav,
    simply use: Rbody2nav = Rnav2body.T
    Examples:
    ---------
    >>> import numpy as np
    >>> from nav import angle2dcm
    >>> g_ned = np.matrix([[0, 0, 9.8]]).T # gravity vector in NED frame
    >>> yaw, pitch, roll = np.deg2rad([90, 15, 0]) # vehicle orientation
    >>> g_body = Rnav2body * g_ned
    >>> g_body
    matrix([[-2.53642664],
            [ 0.        ],
            [ 9.4660731 ]])

    >>> g_ned_check = Rnav2body.T * g_body
    >>> np.linalg.norm(g_ned_check - g_ned) < 1e-10 # should match g_ned
    True
    Reference
    ---------
    [1] Equation 2.4, Aided Navigation: GPS with High Rate Sensors, Jay A. Farrel 2008
    [2] eul2Cbn.m function (note, this function gives body->nav) at:
    http://www.gnssapplications.org/downloads/chapter7/Chapter7_GNSS_INS_Functions.tar.gz
    """
    # Apply necessary unit transformations.
    if input_units == 'rad':
        pass
    elif input_units == 'deg':
        yaw, pitch, roll = np.radians([yaw, pitch, roll])

    # Build transformation matrix Rnav2body.
    s_r, c_r = sin(roll) , cos(roll)
    s_p, c_p = sin(pitch), cos(pitch)
    s_y, c_y = sin(yaw)  , cos(yaw)

    if rotation_sequence == '321':
        # This is equivalent to Rnav2body = R(roll) * R(pitch) * R(yaw)
        # where R() is the single axis rotation matrix.  We implement
        # the expanded form for improved efficiency.
        Rnav2body = np.matrix([
                [c_y*c_p               ,  s_y*c_p              , -s_p    ],
                [-s_y*c_r + c_y*s_p*s_r,  c_y*c_r + s_y*s_p*s_r,  c_p*s_r],
                [ s_y*s_r + c_y*s_p*c_r, -c_y*s_r + s_y*s_p*c_r,  c_p*c_r]])

    else:
        # No other rotation sequence is currently implemented
        logger.warning('WARNING (angle2dcm): requested rotation_sequence is unavailable.')
        logger.warning('                     NaN returned.')
        Rnav2body = np.nan

    return Rnav2body

def dcm2q(R):
    """
    Convert a direction cosine matrix (DCM) to a quaternion.
    Parameters
    ----------
    R : 3x3 numpy array
        Direction cosine matrix.
    Returns
    -------
    q : 4x1 numpy array
        Quaternion.
    Notes
    -----
    The quaternion is ordered as [w, x, y, z].
    """
    q = np.zeros(4)
    q[0] = 0.5 * sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    q[1] = -(R[2,1] - R[1,2]) / (4*q[0])
    q[2] = -(R[0,2] - R[2,0]) / (4*q[0])
    q[3] = -(R[1,0] - R[0,1]) / (4*q[0])
    return q

class ImuGPS():
    def __init__(self) -> None:
        self.time = np.empty(0)
        self.GPStime = np.empty(0)
        self.p = np.empty(0)
        self.vb = np.empty(0)
        self.ab = np.empty(0)
        self.roll = np.empty(0)
        self.pitch = np.empty(0)
        self.yaw = np.empty(0)
        self.w = np.empty(0)
        self.acc_bias = np.empty(0)
        self.gyrp_bias = np.empty(0)
        self.dt = None
        self.vg = np.empty(0)
        self.ag = np.empty(0)
        self.q = np.empty(0)

    @staticmethod
    def get_vg_ag(vb: np.ndarray, ab: np.ndarray, \
        roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
        vg = np.zeros(vb.shape)
        ag = np.zeros(ab.shape)
        q = np.zeros((4, vb.shape[0]))
        for i in range(len(vb)):
            R = angle2dcm(yaw[i], pitch[i], roll[i])
            vg[i] = R.dot(vb[i])
            ag[i] = R.dot(ab[i])
            q[:, i] = dcm2q(R)

        return vg, ag, q

    def convert(self, imugps: np.ndarray):
        # refine timestamp
        imugps = refine_timestamp(imugps)
        localtime = imugps[:,0] - imugps[0,0]
        self.time = localtime
        self.p = imugps[:,1:4]
        self.vb = imugps[:,4:7]
        self.ab = imugps[:,7:10]
        self.roll = imugps[:,10]
        self.pitch = imugps[:,11]
        self.yaw = imugps[:,12]
        self.w = imugps[:,13:16]
        self.acc_bias = imugps[:,16:19]
        self.gyrp_bias = imugps[:,19:22]
        self.GPStime = imugps[:,0]
        dts = imugps[1:,0] - imugps[:-1,0]
        self.dt = np.mean(dts)
        # from angle to rad
        self.roll = np.deg2rad(self.roll)
        self.pitch = np.deg2rad(self.pitch)
        self.yaw = np.deg2rad(self.yaw)
        # from deg/s to rad/s
        self.w = np.deg2rad(self.w)
        self.gyrp_bias = np.deg2rad(self.gyrp_bias)

        vg, ag, q = self.get_vg_ag(self.vb, self.ab, self.roll, self.pitch, self.yaw)
        self.vg = vg
        self.ag = ag
        self.q = q

    def __str__(self) -> str:
        return "{}".format(self.p)

    def __len__(self) -> int:
        return len(self.time)

    def remove_unmatched(self, keep_idx: List[int]):
        idx = [i for i in range(len(self.time)) if i not in keep_idx]
        self.time = np.delete(self.time, idx)
        self.GPStime = np.delete(self.GPStime, idx)
        self.p = np.delete(self.p, idx)
        self.vb = np.delete(self.vb, idx)
        self.ab = np.delete(self.ab, idx)
        self.roll = np.delete(self.roll, idx)
        self.pitch = np.delete(self.pitch, idx)
        self.yaw = np.delete(self.yaw, idx)
        self.w = np.delete(self.w, idx)
        self.acc_bias = np.delete(self.acc_bias, idx)
        self.gyrp_bias = np.delete(self.gyrp_bias, idx)
        self.vg = np.delete(self.vg, idx)
        self.ag = np.delete(self.ag, idx)
        self.q = np.delete(self.q, idx)

        logger.warning("remove {} unmatched data: {}".format(len(idx), idx))

def prepare_data(folder: str, file: str) -> np.ndarray:
    mat_fname = folder + file
    mat_contents = sio.loadmat(mat_fname)

    keys = [key for key in mat_contents.keys() if 'imugps' in key]
    imugps = mat_contents[keys[0]]

    imugps_loader = ImuGPS()
    imugps_loader.convert(imugps)

    return imugps_loader

def align_data(agent_1, agent_2, tol = 1e-3):
    matched_pairs = []
    for i in range(len(agent_1)):
        time_diff = np.abs(agent_1.time[i] - agent_2.time)
        min_val, min_idx = np.min(time_diff), np.argmin(time_diff)
        if min_val < tol:
            logger.debug("{}-{}: {}".format(i, min_idx, min_val))
            matched_pairs.append((i, min_idx))
        else:
            print('time not aligned')

    idx1 = [pair[0] for pair in matched_pairs]
    idx2 = [pair[1] for pair in matched_pairs]

    agent_1.remove_unmatched(idx1)
    agent_2.remove_unmatched(idx2)

    return agent_1, agent_2


def test():
    mat_fname = '../1/imugps_bird.mat'
    mat_contents = sio.loadmat(mat_fname)
    imugps = mat_contents['imugps_bird']

    imu = ImuGPS()
    imu.convert(imugps)

    print((imugps).shape)

    # ok, pass matlab toolbox test
    yaw = 10
    pitch = 20.323
    roll = 30.323
    R = angle2dcm(yaw, pitch, roll, input_units='deg')
    print(R)
    q = dcm2q(R)
    print(q)

    return

def prepare_bearing_data(agent1, agent2, batch_size: int = 6, max_size: int = 2000):
    # prepare bearing data， todo: there is a bug in the bearing calculation
    simulation_data = []

    for i in range(0, min(max_size, len(agent1) - batch_size), batch_size):
        # rotation matrix from agent1's body frame to global frame
        # R1 = angle2dcm(agent1.yaw[i], agent1.pitch[i], agent1.roll[i])
        # rotation matrix from agent2's body frame to global frame
        R2_i = angle2dcm(agent2.yaw[i], agent2.pitch[i], agent2.roll[i])
        p2_i = agent2.p[i]
        R2_i_T = R2_i.T
        ## Agent B records its own position in the INS frame p^B2_B(k). xyz in equation 2
        p2_relative = np.zeros((3, batch_size))
        for j in range(batch_size):
            # R2_j = angle2dcm(agent2.yaw[i+j], agent2.pitch[i+j], agent2.roll[i+j])
            # R_j2i = R2_i.T.dot(R2_j)
            p2 = agent2.p[i+j]
            p2_relative[:, j] = (R2_i_T.dot(p2 - p2_i))

        ## Agent A records and broadcasts its position in the global frame p^A1_A(k). uvw in equation 1
        p1_global = np.zeros((3, batch_size))
        for j in range(batch_size):
            p1 = agent1.p[i+j]
            p1_global[:, j] = p1

        ## now compute the bearing vector from agent2 to agent1
        # This directional measurement is therefore naturally referenced to the body-fixed frame B4.
        # Agent B’s attitude, i.e. orientation with respect to the INS frames B2 and B3 is known.
        # An expression for the DOA measurement referenced to the axes INS frames B3 can therefore be easily calculated.
        # let B3 denote the body-centred INS frame of Agent B (axes of frames B2 and B3 are parallel by definition),
        bearing = np.zeros((3, batch_size))
        for j in range(batch_size):
            p1 = agent1.p[i+j]
            p2 = agent2.p[i+j]
            vec = p1 - p2
            # normalize
            vec = vec / np.linalg.norm(vec)
            # transform to agent2's body frame: theoretically, the bearing vector will be in B4 frame,
            # as the paper assumes that we know the transformation from B4 to B2,
            # so we just need to transform the bearing vector from B4 to B2
            bearing[:, j] = (R2_i_T.dot(vec))

        #  The localisation problem can be reduced to solving for R^B2_A1 \in SO(3) with entries rij and t^B2_A1 \in R3 with entries ti.
        simulation_data.append({"p1": p1_global, "p2": p2_relative, "bearing": bearing, "Rgt": R2_i.T, "tgt": -R2_i.dot(p2_i).reshape(3,-1)})

    return simulation_data


def save_simulation_data(simulation_data, folder: str, file: str):
    for i, data in enumerate(simulation_data):
        mat_fname = folder + file + str(i) + '.txt'
        with open(mat_fname, 'w') as f:
            p1 = data["p1"]
            p2 = data["p2"]
            bearing = data["bearing"]
            Rgt = data["Rgt"]
            tgt = data["tgt"]
            for j in range(p1.shape[1]):
                f.write(' '.join([str(x) for x in p1[:, j]]))
                f.write(' ')
            f.write('\n')
            for j in range(p2.shape[1]):
                f.write(' '.join([str(x) for x in p2[:, j]]))
                f.write(' ')
            f.write('\n')
            for j in range(bearing.shape[1]):
                f.write(' '.join([str(x) for x in bearing[:, j]]))
                f.write(' ')
            f.write('\n')
            Rgt = Rgt.flatten().squeeze().tolist()
            tgt = tgt.flatten().squeeze().tolist()

            if type(Rgt[0]) is list:
                Rgt = Rgt[0]
            if type(tgt[0]) is list:
                tgt = tgt[0]

            for element in Rgt:
                f.write(f"{element} ")
            f.write('\n')
            for element in tgt:
                f.write(f"{element} ")
            f.write('\n')


def gen_simulation_data_dtu():
    folder = '../1/'
    file = 'imugps.mat'
    agent_a = prepare_data(folder, file)
    file = 'imugps_bird.mat'
    agent_b = prepare_data(folder, file)

    logger.debug(len(agent_a))
    logger.debug(len(agent_b))

    # align data: remove unmatched data, it seems that data has been already synchronized.
    # align_data(agent_a, agent_b)
    # logger.debug(len(agent_a))
    # logger.debug(len(agent_b))

    logger.debug((agent_b.p.shape))

    simulation_data = prepare_bearing_data(agent_a, agent_b, batch_size=6, max_size=2000)

    save_simulation_data(simulation_data, '../1/simulation_data/', 'batch_')


def orthogonal_procrustes(Rgt):
    U, S, Vt = np.linalg.svd(Rgt)

    D = np.dot(Vt.T, U.T)
    if np.linalg.det(D) < 0:
        Vt[-1, :] *= -1
        D = np.dot(Vt.T, U.T)

    Ropt = D.T
    return Ropt

def gen_simulation_data_taes():
    Rgt = np.array([[1, -0.032, 3.78e-5],[0.032, 1, 0.002], [-0.98e-5, -0.002, 1]])
    Tgt = np.array([[854.87, 6.18, 1.93]]).T
    Ropt = orthogonal_procrustes(Rgt)

    logger.debug(Ropt)
    logger.debug(Rgt)

    p1_global = np.array([[349.1, -924.1, 374.4],
                          [781.0, -870.3, 372.5],
                          [1007.0, -522.7, 373.3],
                          [869.8, -91.3, 373.2],
                          [431.4, 56.6, 373.1],
                          [33.9, -262.2, 373.6]]).T

    p2_relative = np.array([[1039.2, 574.2, 311.3],
                            [1486.1, 519.4, 310.9],
                            [1946.2, 458.2, 310.2],
                            [2140.4, 746.9, 309.8],
                            [2201.6, 1166.4, 308.8],
                            [2032.8, 1477.7, 310.2]]).T

    bearing_angle = np.array([[-1.4403, 0.0447],
                            [-1.4409, 0.0474],
                            [-1.6430, 0.0697],
                            [-2.0459, 0.0723],
                            [-2.2708, 0.0464],
                            [-2.1512, 0.0317]]).T

    # p2_global = np.array([[202.5, 561.3, 310.4],
    #                      [647.3, 492.1, 309.9],
    #                      [1105.2, 416.2, 309.1],
    #                      [1308.6, 698.5, 309.2],
    #                      [1383.2, 1115.8, 309.0],
    #                      [1224.5, 1432.5, 310.9]]).T

    p2_global = np.zeros((3, p2_relative.shape[1]))
    for i in range(p2_relative.shape[1]):
        p2_global[:, i] = (Ropt.T@(p2_relative[:, i].reshape(3,1) - Tgt)).flatten()

    # logger.debug(p2_global)
    # logger.debug(p2_global)

    bearing_computed = np.zeros((3, p2_relative.shape[1]))
    for i in range(p2_global.shape[1]):
        vec = p1_global[:,i] - p2_global[:,i]
        vec = vec / np.linalg.norm(vec)
        bearing_computed[:, i] = Ropt.dot(vec)

    bearing_angle = np.zeros((2, bearing_angle.shape[1]))
    for i in range(bearing_computed.shape[1]):
        vec = bearing_computed[:, i]
        phi = asin(vec[2])
        theta = atan2(vec[1], vec[0])
        bearing_angle[:, i] = np.array([theta, phi])

    logger.info(bearing_angle)

    def from_bearingangle_to_bearing_vec(theta, phi):
        return np.array([cos(theta) * cos(phi), cos(phi) * sin(theta), sin(phi)])

    bearing = np.zeros((3, bearing_angle.shape[1]))
    for i in range(bearing_angle.shape[1]):
        bearing[:, i] = from_bearingangle_to_bearing_vec(bearing_angle[0, i], bearing_angle[1, i])

    ## check bearing correctness

    logger.info(bearing)
    logger.info(bearing_computed)

    # logger.debug(bearing_angle)
    # logger.debug(bearingang_computed)

    simulation_data = []

    logger.info(Ropt)
    simulation_data.append({"p1": p1_global, "p2": p2_relative, "bearing": bearing_computed, "Rgt": Ropt, "tgt": Tgt})

    save_simulation_data(simulation_data, '../taes/', 'simu_')

if __name__ == '__main__':
    gen_simulation_data_taes()
