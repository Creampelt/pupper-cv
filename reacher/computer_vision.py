import numpy as np
import cv2 as cv

# Distance between robot and inner left corner of checkerboard
# NOTE: We are assuming robot angle is 0, since measuring it would be too difficult
ROBOT_X = -3.17
ROBOT_Y = 8.5
ROBOT_Z = -11.43

# For calibration
CHECKER_SIZE = 2.4  # checker size in cm
CHECKERBOARD_HEIGHT = 46.99  # height of top of bottom row of checkerboard (i.e. lowest dot) from ground in cm
CHECKER_WIDTH = 9  # number of columns in checkerboard
CHECKER_HEIGHT = 7  # number of rows in checkerboard
DILATION_ITERATIONS = 3  # number of iterations for image dilation (increase for less precise but fewer contour boxes


def col_vec(x):
    return x[:, np.newaxis]


def calculate_world_coords(u, v, camera_mat, rotation, translation):
    # Prepare matrices
    img_vec = col_vec(np.array([u, v, 1], dtype=np.float32))
    r, _ = cv.Rodrigues(rotation)

    # Calculate inverses
    a_inv = np.linalg.inv(camera_mat)
    r_inv = np.linalg.inv(r)

    # Calculate s from z value (since z = 0 is known)
    lhs = r_inv @ a_inv @ img_vec
    rhs = r_inv @ translation
    s = (0.0 + rhs[2, 0]) / lhs[2, 0]
    # Return coordinates
    return (r_inv @ (s * a_inv @ img_vec - translation)).T[0]


def get_valid_contours(contours):
    return list(filter(lambda cnt: cnt[2] < 20 and cnt[3] < 20, contours))


def find_nearest_to_center(contours, img_shape):
    if len(contours) == 0:
        return None
    center = (img_shape[1] / 2.0, img_shape[0] / 2.0)
    contour_centers = np.array([get_rectangle_center(cnt) for cnt in contours])
    i = np.linalg.norm(contour_centers - center, axis=1).argmin()
    return contours[i]


def get_rectangle_center(rect):
    return rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0


def visualize_image(img, best, contours):
    """
    Draw contour box on image and display image
    """
    for contour in contours:
        x, y, w, h = contour
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if best is not None:
        x, y, w, h = best
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center_x, center_y = get_rectangle_center((x, y, w, h))
        cv.putText(img, "({}, {})".format(center_x, center_y), (x, y + h + 5), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow("Result", img)


def pick_best_contour(contours, img):
    """
    Filter contours by size (should be <20 pixels in width and height), picks center-most
    """
    valid_contours = get_valid_contours(contours)
    best = find_nearest_to_center(valid_contours, img.shape[:2])
    visualize_image(img, best, valid_contours)
    return best


def find_red_dot(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Mask to red pixels only
    lower_red = np.array([340 / 2, 50, 50])
    upper_red = np.array([360 / 2, 255, 255])
    mask = cv.inRange(img_hsv, lower_red, upper_red)
    img_hsv[np.where(mask == 0)] = 0

    # Binarize image
    gray = cv.cvtColor(img_hsv, cv.COLOR_BGR2GRAY)
    ret2, th2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Apply morphological operations to dilate the image (reduce noise)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(th2, kernel, iterations=DILATION_ITERATIONS)

    # Find contours (boxes around connected components)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = pick_best_contour([cv.boundingRect(cnt) for cnt in contours], img)
    if contour is None:
        return False, None, None
    return True, *get_rectangle_center(contour)


def world_to_robot_coords(world):
    robot_frame = world - [ROBOT_X, ROBOT_Y, ROBOT_Z]
    # We have y being up/down and z being forward/back, but inverse kinematics is vice versa (bc we are silly)
    robot_frame[1], robot_frame[2] = robot_frame[2], robot_frame[1]
    # We also did our math in cm, while inverse kinematics is in meters
    return robot_frame / 100.0


def calibrate_camera(cap):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKER_WIDTH * CHECKER_HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKER_WIDTH, (CHECKER_HEIGHT - 1):-1:-1].T.reshape(-1, 2)
    # Scale objp to be in cm
    objp *= CHECKER_SIZE
    # Translate objp y-values to have y=0 at ground level
    objp[:, 1] += CHECKERBOARD_HEIGHT
    # objp[:, 2] = 1.0
    object_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    while len(object_points) < 10:
        success, img = cap.read()
        if cv.waitKey(1) & 0xFF == ord('q'):
            return None
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (CHECKER_WIDTH, CHECKER_HEIGHT),
                                                flags=cv.CALIB_CB_ADAPTIVE_THRESH)
        # If found, add object points, image points (after refining them)
        if ret:
            object_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (CHECKER_WIDTH, CHECKER_HEIGHT), corners2, ret)
            cv.imshow("Result", img)
            cv.waitKey(500)
        else:
            cv.imshow("Result", img)
    ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera(object_points, img_points,
                                                                                      gray.shape[::-1], None, None)
    print("Error in projection: ", ret)
    return ret, camera_mat, distortion, rotation_vecs, translation_vecs


def calculate_cv_ik_xyz(cap, camera_mat, rotation_vecs, translation_vecs):
    success, img = cap.read()
    if not success:
        return None
    success, x, y = find_red_dot(img)
    if not success:
        return None
    world_coords = calculate_world_coords(x, y, camera_mat, rotation_vecs[-1], translation_vecs[-1])
    robot_coords = world_to_robot_coords(world_coords)
    return robot_coords
