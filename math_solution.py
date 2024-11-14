import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    rotation_matrix1 = camera_rotation1.T
    rotation_matrix2 = camera_rotation2.T

    translation_vector1 = -rotation_matrix1 @ camera_position1
    translation_vector2 = -rotation_matrix2 @ camera_position2

    P1 = np.dot(camera_matrix, np.hstack((rotation_matrix1, translation_vector1)))
    P2 = np.dot(camera_matrix, np.hstack((rotation_matrix2, translation_vector2)))

    coordinates1 = image_points1
    coordinates2 = image_points2

    points_3d = []
    for p1, p2 in zip(coordinates1, coordinates2):
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        X /= X[3]
        points_3d.append(X[:3])
    return np.array(points_3d)
