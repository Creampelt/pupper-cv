import numpy as np
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10  # length of link 1
LOWER_LEG_OFFSET = 0.13  # length of link 2
TOLERANCE = 0.01  # tolerance for inverse kinematics
PERTURBATION = 0.0001  # perturbation for finite difference method
MAX_ITERATIONS = 10


def get_end_effector(joint_angles):
    return forward_kinematics.fk_foot(joint_angles)[0:3, 3]


def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    end_effector_guess = get_end_effector(guess)
    return np.linalg.norm(end_effector_pos - end_effector_guess)


def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    # Initialize Jacobian to zero
    J = np.zeros((3, 3))
    actual_pose = get_end_effector(joint_angles)
    for i in range(0, 3):
        adjusted_angles = np.array(joint_angles)
        adjusted_angles[i] += delta
        pose = get_end_effector(adjusted_angles) - actual_pose
        for j in range(0, 3):
            J[j][i] = pose[j] / delta
    return J


def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """
    # Initialize previous cost to infinity
    previous_cost = np.inf
    for iters in range(MAX_ITERATIONS):
        J = calculate_jacobian_FD(guess, PERTURBATION)
        e = end_effector_pos - get_end_effector(guess)
        J_pinv = np.linalg.pinv(J)

        # Take a full Newton step to update the guess for joint angles
        guess += J_pinv @ e
        cost = ik_cost(end_effector_pos, guess)
        # Calculate the cost based on the updated guess
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost

    return guess
