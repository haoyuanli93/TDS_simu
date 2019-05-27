import numpy as np

"""
        Notice:
        
    The code from this script is adopted from another repo
    https://github.com/haoyuanli93/pysingfel
    
    5/27/2019
    Haoyuan Li
    hyli16@stanford.edu

"""


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2_norm_batch(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def reshape_pixels_position_arrays_to_1d(array):
    """
    Only an abbreviation.

    :param array: The position array.
    :return: Reshaped array.
    """
    array_1d = np.reshape(array, [np.prod(array.shape[:-1]), 3])
    return array_1d


def get_reciprocal_space_pixel_position(pixel_position, wave_vector):
    """
    Obtain the coordinate of each pixel in the reciprocal space.
    Notice that unit of the output of this calculation only depends on the unit of the wave_vector.

    :param pixel_position: The coordinate of the pixel in real space in m^-1
    :param wave_vector: The wavevector in m^-1
    :return: The array containing the pixel coordinates.
    """
    # reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_position)

    # Calculate the reciprocal position of each pixel
    wave_vector_norm = l2_norm(wave_vector)
    wave_vector_direction = wave_vector / wave_vector_norm

    pixel_center_norm = l2_norm_batch(pixel_center_1d)
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:, np.newaxis]

    pixel_position_reciprocal_1d = wave_vector_norm * (pixel_center_direction - wave_vector_direction)

    # restore the pixels shape
    pixel_position_reciprocal = np.reshape(pixel_position_reciprocal_1d, pixel_position.shape)

    return pixel_position_reciprocal


def get_polarization_correction(pixel_position, polarization):
    """
    Obtain the polarization correction. Notice that this is the correction of the polarization on the
    intensity rather than the complex magnitude.

    :param pixel_position: The position of each pixel in real space.
    :param polarization: The polarization vector of the incident beam.
    :return: The polarization correction array.
    """
    # reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_position)

    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d), axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:, np.newaxis]

    # Calculate the polarization correction
    polarization_norm = l2_norm(polarization)
    polarization_direction = polarization / polarization_norm

    polarization_correction_1d = np.sum(np.square(np.cross(pixel_center_direction,
                                                           polarization_direction)), axis=1)

    # print polarization_correction_1d.shape

    polarization_correction = np.reshape(polarization_correction_1d, pixel_position.shape[0:-1])

    return polarization_correction


def get_solid_angle(pixel_position, pixel_area, orientation):
    """
    Calculate the solid angle for each pixel.

    :param pixel_position: The position of each pixel in real space.
    :param orientation: The orientation of the detector.
    :param pixel_area: The pixel area for each pixel.
    :return: Solid angle of each pixel.
    """

    # Use 1D format
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_position)
    pixel_center_norm_1d = np.sqrt(np.sum(np.square(pixel_center_1d), axis=-1))
    pixel_area_1d = np.reshape(pixel_area, np.prod(pixel_area.shape))

    # Calculate the direction of each pixel.
    pixel_center_direction_1d = pixel_center_1d / pixel_center_norm_1d[:, np.newaxis]

    # Normalize the orientation vector
    orientation_norm = np.sqrt(np.sum(np.square(orientation)))
    orientation_normalized = orientation / orientation_norm

    # The correction induced by projection which is a factor of cosine.
    cosine_1d = np.abs(np.dot(pixel_center_direction_1d, orientation_normalized))

    # Calculate the solid angle ignoring the projection
    solid_angle_1d = np.divide(pixel_area_1d, np.square(pixel_center_norm_1d))
    solid_angle_1d = np.multiply(cosine_1d, solid_angle_1d)

    # Restore the pixel stack format
    solid_angle_stack = np.reshape(solid_angle_1d, pixel_area.shape)

    return solid_angle_stack


def get_reciprocal_position_and_corrections(pixel_position, pixel_area,
                                            wave_vector, polarization, orientation):
    """
    Calculate the pixel positions in reciprocal space and all the related corrections.

    :param pixel_position: The position of the pixel in real space.
    :param wave_vector: The wavevector.
    :param polarization: The polarization vector.
    :param orientation: The normal direction of the detector.
    :param pixel_area: The pixel area for each pixel. In pixel stack format.
    :return: pixel_position_reciprocal, pixel_position_reciprocal_norm, polarization_correction,
            geometry_correction
    """
    # Calculate the position and distance in reciprocal space
    pixel_position_reciprocal = get_reciprocal_space_pixel_position(pixel_position=pixel_position,
                                                                    wave_vector=wave_vector)
    pixel_position_reciprocal_norm = np.sqrt(np.sum(np.square(pixel_position_reciprocal), axis=-1))

    # Calculate the corrections.
    polarization_correction = get_polarization_correction(pixel_position=pixel_position,
                                                          polarization=polarization)

    # Because the pixel area in this function is measured in m^2,
    # therefore,the distance has to be in m
    solid_angle_array = get_solid_angle(pixel_position=pixel_position,
                                        pixel_area=pixel_area,
                                        orientation=orientation)

    return (pixel_position_reciprocal, pixel_position_reciprocal_norm,
            polarization_correction, solid_angle_array)


class Detector(object):
    """
    This is the base object for all detector object.
    This class contains some basic operations of the detector.
    It provides interfaces for the other modules.
    """

    def __init__(self, pixel_position, pixel_area, orientation):
        """

        :param pixel_position: numpy array of shape [n,m,3]. Here, n is the pixel number along 1 direction
                                m is the pixel number along the other direction.
                                3 represent the 3d space. By default, 0 is x, 1 is y, 2 is z.
                                this is the position of the pixels in the lab frame.
                                The unit is m.

        :param orientation: The normal direction of the detector. This is an array of shape [3]
        :param pixel_area: The area of each pixel in m^2. This is an array of shape [n,m]
        """

        n, m, _ = pixel_position.shape
        ########################
        #  Geometry
        ########################
        self.pixel_num_y = n  # number of pixels in y
        self.pixel_num_z = m  # number of pixels in z
        self.pixel_num_total = self.pixel_num_z * self.pixel_num_y  # total number of pixels

        # Default position of the detector which is in the y-z plane
        self.pixel_area = pixel_area  # (m^2)
        self.pixel_position = pixel_position  # (m)  The pixel position in the lab frame

        self.orientation = orientation  # The normal direction of the detector

        ########################
        # pixel momentum and corrections. These are the useful info we may use later for processing
        ########################
        # pixel information in reciprocal space
        self.pixel_position_reciprocal = None  # (m^-1)
        self.pixel_distance_reciprocal = None  # (m^-1)

        # Corrections
        self.solid_angle_pixel = np.ones((self.pixel_num_y, self.pixel_num_z))  # solid angle
        self.polarization_correction = np.ones((self.pixel_num_y, self.pixel_num_z))  # Polarization correction

        """
        The theoretical differential cross section of an electron ignoring the 
        polarization effect is,
                do/dO = ( e^2/(4*Pi*epsilon0*m*c^2) )^2  *  ( 1 + cos(xi)^2 )/2 
        Therefore, one needs to includes the leading constant factor which is the 
        following numerical value.
        """
        # Tompson Scattering factor
        self.Thomson_factor = 2.817895019671143 * 2.817895019671143 * 1e-30

    def get_reciprocal_pixel_position_and_corrections(self, wavevector_m, polar_vector):
        """
        Calculate the momentum for each pixel and the corresponding correction coefficient.
        :param wavevector_m: The wave vector in meter
        :param polar_vector: The polarization of the incident x-ray pulse. This should be a vector.
        :return:
        """

        # Get the reciprocal positions and the corrections
        (self.pixel_position_reciprocal,
         self.pixel_distance_reciprocal,
         self.polarization_correction,
         self.solid_angle_pixel) = get_reciprocal_position_and_corrections(
            pixel_position=self.pixel_position,
            polarization=polar_vector,
            wave_vector=wavevector_m,
            pixel_area=self.pixel_area,
            orientation=self.orientation)

    def rotate_two_theta(self, two_theta):
        """
        Rotate the two_theta angle. Get the corresponding pixel position in real space
        and the orientation of the detector
        :param two_theta:
        :return:
        """
        # First get the rotation matrix
        rot_mat = np.zeros((3, 3), dtype=np.float64)
        rot_mat[0, 0] = np.cos(two_theta)
        rot_mat[0, 2] = -np.sin(two_theta)
        rot_mat[1, 1] = 1.
        rot_mat[2, 0] = np.sin(two_theta)
        rot_mat[2, 2] = np.cos(two_theta)

        # Rotate the detector
        self.general_rotation(rot_mat=rot_mat)

    def general_rotation(self, rot_mat):
        """
        Rotate the detector generally. Update the corresponding pixel position
        and the detector orientation.

        :param rot_mat:
        :return:
        """

        # Notice that, usually, we use column matrix to represent vectors.
        # Therefore, by default, the rotate matrix should act on the left. However,
        # In my implementation, for pixel_position, it is the row matrix that represents the
        # vector. Therefore, I need to use the transformation of the rotmat and act
        # from right.
        self.pixel_position = np.dot(self.pixel_position, rot_mat.T)

        # This is different from the previous calculation because this is a column matrix
        self.orientation = np.dot(rot_mat, self.orientation)
