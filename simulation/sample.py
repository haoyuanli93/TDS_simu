import numpy as np


def get_rotation_matrix_and_g(theta, phi, chi):
    """
    Get the rotation matrix for the sample given the theta, phi, chi angles.
    Assume that the machine first rotate theta, then chi, then phi.
    Assume that the default operation is left multiplication

    :param theta: The angle of the largest circle. This is a rotation around y axis of the
                    sample frame. Counter clockwise is positive
    :param phi: The angle of the stage on the largest circle. This is the rotation around the x axis
                of the sample frame. Counter clockwise is positive
    :param chi: The angle of the plate on the stage. This is the rotation around the z axis
                of the sample frame. Counter clockwise is positive
    :return: The rotation matrix.
    """
    # theta is around the sample frame's y axis. Counter-clockwise is positive
    rtheta = np.array([[np.cos(theta), 0, - np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])

    # chi is around the sample frame's x axis. Counter-clockwise is positive
    rchi = np.array([[1, 0, 0],
                     [0, np.cos(chi), np.sin(chi)],
                     [0, - np.sin(chi), np.cos(chi)]])

    # Phi is around the sample frame's z axis. Counter-clockwise is positive
    rphi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

    # Assume that the machine first rotate theta, then chi, then phi.
    # Assume that the default operation is left multiplication
    rot_mat = np.dot(rphi, np.dot(rchi, rtheta))
    return rot_mat, np.dot(rot_mat, g)
