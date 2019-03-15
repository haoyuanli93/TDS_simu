"""
This file contains all the calculation functions that will be useful in the simulation
"""

import numba
import numpy as np

er = 2.8179403227e-15  # the classical electron radius in meter
er_square = er * er


@numba.vectorize
def thomson_scattering_intensity(intensity, d_square, angular_effect):
    """
    Given the intensty the distance square and the polarization factor, return the scattering
    intensity of a thomson scattering scattering process.

    :param intensity:
    :param d_square:
    :param angular_effect:
    :return:
    """
    return intensity * er_square / d_square * angular_effect


def get_polarization_correction(pixel_center, polarization):
    """
    Obtain the polarization correction.
    :param pixel_center: The position of each pixel in real space.
    :param polarization: The polarization vector of the incident beam.
    :return: The polarization correction array.
    """

    # Get pixel array shape
    pixel_shape = pixel_center.shape
    pixel_num = np.prod([pixel_shape[:-1]])

    # reshape the array into a 1d position array
    pixel_center_1d = np.reshape(pixel_center, [pixel_num, pixel_shape[-1]])

    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d), axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:, np.newaxis]

    # Calculate the polarization correction
    polarization_norm = np.sqrt(np.sum(np.square(polarization)))
    polarization_direction = polarization / polarization_norm

    polarization_correction_1d = np.sum(np.square(np.cross(pixel_center_direction,
                                                           polarization_direction)), axis=1)

    # print polarization_correction_1d.shape
    polarization_correction = np.reshape(polarization_correction_1d, pixel_center.shape[:-1])

    return polarization_correction
