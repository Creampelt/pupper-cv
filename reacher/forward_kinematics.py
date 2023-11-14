import numpy as np

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10  # length of link 1
LOWER_LEG_OFFSET = 0.13  # length of link 2


def rotation_matrix(axis, angle):
    """
    Create a 3x3 rotation matrix which rotates about a specific axis

    Args:
      axis:  Array.  Unit vector in the direction of the axis of rotation
      angle: Number. The amount to rotate about the axis in radians

    Returns:
      3x3 rotation matrix as a numpy array
    """
    a = np.cos(angle / 2.0)
    b, c, d = np.array(axis) * np.sin(angle / 2.0)
    # Euler Rodrigues formula
    return np.array([
        [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a ** 2 + c ** 2 - b ** 2 - d ** 2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 + d ** 2 - b ** 2 - c ** 2]
    ])


def homogenous_transformation_matrix(axis, angle, v_A):
    """
    Create a 4x4 transformation matrix which transforms from frame A to frame B

    Args:
      axis:  Array.  Unit vector in the direction of the axis of rotation
      angle: Number. The amount to rotate about the axis in radians
      v_A:   Vector. The vector translation from A to B defined in frame A

    Returns:
      4x4 transformation matrix as a numpy array
    """
    translation_mat = np.eye(4)
    translation_mat[:, 3] = v_A + [1.0]
    rotation_mat = np.vstack(
        (np.hstack((rotation_matrix(axis, angle), np.atleast_2d(np.zeros(3)).T)), [0.0, 0.0, 0.0, 1.0]))
    return translation_mat @ rotation_mat


def fk_hip(joint_angles):
    """
    Use forward kinematics equations to calculate the xyz coordinates of the hip
    frame given the joint angles of the robot

    Args:
      joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle,
                    elbow_angle]. Angles are in radians
    Returns:
      4x4 matrix representing the pose of the hip frame in the base frame
    """
    return homogenous_transformation_matrix([0.0, 0.0, 1.0], joint_angles[0], [0.0, 0.0, 0.0])


def fk_shoulder(joint_angles):
    """
    Use forward kinematics equations to calculate the xyz coordinates of the shoulder
    joint given the joint angles of the robot

    Args:
      joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle,
                    elbow_angle]. Angles are in radians
    Returns:
      4x4 matrix representing the pose of the shoulder frame in the base frame
    """

    shoulder_frame_from_hip = homogenous_transformation_matrix([0.0, 1.0, 0.0], joint_angles[1],
                                                               [0.0, -HIP_OFFSET, 0.0])
    return fk_hip(joint_angles) @ shoulder_frame_from_hip


def fk_elbow(joint_angles):
    """
    Use forward kinematics equations to calculate the xyz coordinates of the elbow
    joint given the joint angles of the robot

    Args:
      joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle,
                    elbow_angle]. Angles are in radians
    Returns:
      4x4 matrix representing the pose of the elbow frame in the base frame
    """
    elbow_frame_from_shoulder = homogenous_transformation_matrix([0.0, 1.0, 0.0], joint_angles[2],
                                                                 [0.0, 0.0, UPPER_LEG_OFFSET])
    return fk_shoulder(joint_angles) @ elbow_frame_from_shoulder


def fk_foot(joint_angles):
    """
    Use forward kinematics equations to calculate the xyz coordinates of the foot given
    the joint angles of the robot

    Args:
      joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle,
                    elbow_angle]. Angles are in radians
    Returns:
      4x4 matrix representing the pose of the end effector frame in the base frame
    """

    foot_frame_from_elbow = homogenous_transformation_matrix([1.0, 0.0, 0.0], 0.0, [0.0, 0.0, LOWER_LEG_OFFSET])
    return fk_elbow(joint_angles) @ foot_frame_from_elbow
