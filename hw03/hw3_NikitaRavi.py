# Name: Nikita Ravi
# Class: ECE 66100
# Homework #3
# Deadline: 09/20/2022

# Import Modules
import numpy as np
import cv2
import math

def display_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def compute_homography(distorted, undistorted):
    n = distorted.shape[1]
    A = np.zeros((2*n, 2*n))
    b = np.zeros((2*n, 1))
    H = np.zeros((3,3))

    for idx in range(n):
        A[2*idx] = [undistorted[0][idx], undistorted[1][idx], 1, 0, 0, 0, (-undistorted[0][idx] * distorted[0][idx]), (-undistorted[1][idx] * distorted[0][idx])]
        A[2*idx + 1] = [0, 0, 0, undistorted[0][idx], undistorted[1][idx], 1, (-undistorted[0][idx] * distorted[1][idx]), (-undistorted[1][idx] * distorted[1][idx])]
        b[2*idx] = distorted[0][idx]
        b[2*idx + 1] = distorted[1][idx]

    h = np.matmul(np.linalg.inv(A), b)

    row = 0
    for idx in range(0, len(h), 3):
        spliced = h[idx:idx+3]
        if(len(spliced) == 3):
            H[row] = spliced.T
        else:
            H[row] = np.append(spliced, [1])
        row += 1

    return H

def create_blank_image(H, height, width):
    corners = np.array([[0,0,1], [0,width,1], [height, width, 1], [height,0,1]])
    cornerPrime = np.matmul(H, corners.T).T
    for idx in range(len(cornerPrime)):
        cornerPrime[idx] = cornerPrime[idx] / cornerPrime[idx][2]

    minimum, maximum = cornerPrime.min(axis = 0), cornerPrime.max(axis = 0)
    y_min, y_max, x_min, x_max = int(math.floor(minimum[0])), int(math.ceil(maximum[0])), int(math.floor(minimum[1])), int(math.ceil(maximum[1]))

    blank = np.zeros((y_max - y_min, x_max - x_min, 3))
    return blank, y_min, x_min

def transform(undistorted, blank, H, y_min, x_min, path):
    undistorted_height, undistorted_width, undistorted_channel = undistorted.shape
    blank_height, blank_width, blank_channel = blank.shape

    for i in range(blank_height):
        for j in range(blank_width):
            xPrime = np.array((i+y_min,j+x_min,1),dtype = 'l')
            X = np.matmul(np.linalg.inv(H), xPrime.T)
            X /= X[-1]
            X = X.astype(int)
            
            if X[0]<undistorted_height and X[0]>=0 and X[1]<undistorted_width and X[1]>=0:
                blank[i][j] = undistorted[X[0]][X[1]]   

    cv2.imwrite(path, blank)

def point_to_point_correspondence(undistorted_img, undistorted_points, distorted_points, path):
    H = compute_homography(undistorted_points, distorted_points)
    height, width, channel = undistorted_img.shape
    blank, y_min, x_min = create_blank_image(H, height, width)
    transform(undistorted_img, blank, H, y_min, x_min, path)

def calculate_lines_from_points(pt1, pt2):
    line = np.cross(pt1, pt2)
    return line

def calculate_point_from_lines(l1, l2):
    point = np.cross(l1, l2)
    return point

def homography_to_remove_projective_distortion(P,Q,R,S):
    l1 = calculate_lines_from_points(P, S) #Line PS
    l2 = calculate_lines_from_points(Q, R) #Line QR
    vanishingPoint1 = calculate_point_from_lines(l1, l2)
    vanishingPoint1 = vanishingPoint1 / vanishingPoint1[-1]
    
    l3 = calculate_lines_from_points(P, Q) # Line PQ
    l4 = calculate_lines_from_points(S, R) # Line SR
    vanishingPoint2 = calculate_point_from_lines(l3, l4)
    vanishingPoint2 = vanishingPoint2 / vanishingPoint2[-1]

    vanishingLine = calculate_lines_from_points(vanishingPoint1, vanishingPoint2)
    vanishingLine = vanishingLine / vanishingLine[-1]
    
    H = np.zeros((3,3))
    H[0][0] = H[1][1] = 1
    H[2] = vanishingLine

    return H

def homography_to_remove_affine_distortion(P,Q,R,S):
    """
    P ------- S
    |         |
    |         |
    |         |
    |         |
    Q ------- R
    """

    lpq = calculate_lines_from_points(P, Q) # Line PQ
    lpq = lpq / lpq[-1]

    lsr = calculate_lines_from_points(S, R) # Line SR
    lsr = lsr / lsr[-1]

    lps = calculate_lines_from_points(P, S) # Line PS
    lps = lps / lps[-1]

    lqr = calculate_lines_from_points(Q, R) # Line QR
    lqr = lqr / lqr[-1]

    A = np.zeros((2, 2))
    A[0][0] = lpq[0] * lqr[0]
    A[0][1] = lpq[0] * lqr[1] + lpq[1] * lqr[0]
    A[1][0] = lps[0] * lsr[0]
    A[1][1] = lps[0] * lsr[1] + lps[1] * lsr[0]

    b = np.zeros((2,1))
    b[0] = -lpq[1] * lqr[1]
    b[1] = -lps[1] * lsr[1]

    s11_s12 = np.matmul(np.linalg.inv(A), b)
    S = np.array([[s11_s12[0], s11_s12[1]], [s11_s12[1], 1]], dtype=float)
    
    u, s, vh = np.linalg.svd(S) # S = V * D^2 * V.T
    
    D = np.sqrt(np.diag(s))
    A = np.matmul(np.matmul(u, D), u.T)

    # H = np.zeros((3,3))
    H = np.append(A,[[0,0]], axis=0)
    H = np.append(H,[[0],[0],[1]], axis=1)
    return H

def two_step_method(image, X, path):
    P,Q,R,S = X[0], X[1], X[2], X[3]
    remove_projective_homography = homography_to_remove_projective_distortion(P,Q,R,S)
    image_with_only_affine = np.matmul(remove_projective_homography, X.T).T
    
    P,Q,R,S = image_with_only_affine[0], image_with_only_affine[1], image_with_only_affine[2], image_with_only_affine[3]
    remove_affine_homography = homography_to_remove_affine_distortion(P,Q,R,S)

    H = np.matmul(np.linalg.inv(remove_affine_homography), remove_projective_homography)
    height, width, channel = image.shape
    blank, y_min, x_min = create_blank_image(H, height, width)
    transform(image, blank, H, y_min, x_min, path)

def homography_to_remove_projective_and_affine(P,Q,R,S, P_sq, Q_sq, R_sq, S_sq):
    """
    P ------- S
    |         |
    |         |
    |         |
    |         |
    Q ------- R
    """

    lpq = np.cross(P, Q) 
    lpq = lpq / lpq[-1]

    lsr = np.cross(S, R) 
    lsr  = lsr / lsr[-1]

    lps = np.cross(P, S) 
    lps = lps / lps[-1]

    lqr = np.cross(Q, R) 
    lqr = lqr / lqr[-1]

    lpr_sq = np.cross(P_sq, R_sq) 
    lpr_sq = lpr_sq / lpr_sq[-1]

    lsq_sq = np.cross(Q_sq,S_sq) 
    lsq_sq = lsq_sq / lsq_sq[-1]

    A = np.zeros((5,5))
    b = np.zeros((5,1))

    # PQ and QR
    A[0][0] = lpq[0] * lqr[0]
    A[0][1] = (lpq[0] * lqr[1] + lpq[1] * lqr[0]) / 2
    A[0][2] = lpq[1] * lqr[1]
    A[0][3] = (lpq[0] * lqr[2] + lpq[2] * lqr[0]) / 2
    A[0][4] = (lpq[2] * lqr[1] + lpq[1] * lqr[2]) / 2
    b[0] = -lpq[2] * lqr[2]

    # PQ and PS
    A[1][0] = lpq[0] * lps[0]
    A[1][1] = (lpq[0] * lps[1] + lpq[1] * lps[0]) / 2
    A[1][2] = lpq[1] * lps[1]
    A[1][3] = (lpq[2] * lps[0] + lpq[0] * lps[2]) / 2
    A[1][4] = (lpq[2] * lps[1] + lpq[1] * lps[2]) / 2
    b[1] = -lpq[2] * lps[2]

    # SR and PS
    A[2][0] = lsr[0] * lps[0]
    A[2][1] = (lsr[0] * lps[1] + lsr[1] * lps[0]) / 2
    A[2][2] = lsr[1] * lps[1]
    A[2][3] = (lsr[2] * lps[0] + lsr[0] * lps[2]) / 2
    A[2][4] = (lsr[2] * lps[1] + lsr[1] * lps[2]) / 2
    b[2] = -lsr[2] * lps[2]

    # SR and QR
    A[3][0] = lsr[0] * lqr[0]
    A[3][1] = (lsr[0] * lqr[1] + lsr[1] * lqr[0]) / 2
    A[3][2] = lsr[1] * lqr[1]
    A[3][3] = (lsr[2] * lqr[0] + lsr[0] * lqr[2]) / 2
    A[3][4] = (lsr[2] * lqr[1] + lsr[1] * lqr[2]) / 2
    b[3] = -lsr[2] * lqr[2]

    # PR_sq and SQ_sq
    A[4][0] = lpr_sq[0] * lsq_sq[0]
    A[4][1] = (lpr_sq[0] * lsq_sq[1] + lpr_sq[1] * lsq_sq[0]) / 2
    A[4][2] = lpr_sq[1] * lsq_sq[1]
    A[4][3] = (lpr_sq[2] * lsq_sq[0] + lpr_sq[0] * lsq_sq[2]) / 2
    A[4][4] = (lpr_sq[2] * lsq_sq[1] + lpr_sq[1] * lsq_sq[2]) / 2
    b[4] = -lpr_sq[2] * lsq_sq[2]

    s_temp = np.matmul(np.linalg.inv(A), b)
    s_temp = s_temp / np.max(s_temp)

    S = np.zeros((2,2))
    S[0][0] = s_temp[0]
    S[0][1] = S[1][0] = s_temp[1] / 2
    S[1][1] = s_temp[2]

    u, s, vh = np.linalg.svd(S)

    D = np.sqrt(np.diag(s))
    A = np.matmul(np.matmul(u, D), u.T)

    temp = np.array([s_temp[3]/2, s_temp[4]/2])
    v = np.matmul(np.linalg.inv(A), temp)

    H = np.zeros((3,3))
    H[0][0] = A[0][0]
    H[0][1] = A[0][1]
    H[1][0] = A[1][0]
    H[1][1] = A[1][1]
    H[2][0] = v[0]
    H[2][1] = v[1]
    H[2][2] = 1
    return H

def one_step_method(image, X, X_sq, path):
    P,Q,R,S = X[0], X[1], X[2], X[3]
    P_sq, Q_sq, R_sq, S_sq = X_sq[0], X_sq[1], X_sq[2], X_sq[3]
    H = homography_to_remove_projective_and_affine(P,Q,R,S, P_sq, Q_sq, R_sq, S_sq)

    height, width, channel = image.shape
    blank, y_min, x_min = create_blank_image(H, height, width)
    transform(image, blank, H, y_min, x_min, path)

def get_images(path, image):
    path += image
    img = cv2.imread(path)

    return img

def get_width_height_coordinates(image):
    coor = np.array([])
    if(image == "building"):
        coor = np.array([[0,0,1], [0,900,1], [300, 900, 1], [300, 0, 1]])
    elif(image == "nighthawks"):
        coor = np.array([[0,0,1], [0,85,1], [150,85,1], [150,0,1]])
    elif(image == "pikachu"):
        coor = np.array([[0,0,1], [0,25,1], [25,25,1], [25,0,1]])
    elif(image == "calendar"):
        coor = np.array([[0,0,1], [0,42,1], [53,42,1], [53,0,1]]) 

    return coor

def get_pqrs_coordinates(image):
    pqrs = np.array([])
    if(image == "building"):
        pqrs = np.array([[243,204,1], [236,370,1], [296,375,1], [299,218,1]])
    elif(image == "nighthawks"):
        pqrs = np.array([[77,182,1], [77,656,1], [807,620,1], [815,208,1]])
    elif(image == "pikachu"):
        pqrs = np.array([[1071,596,1], [1108,2724,1], [2271,2432,1], [2312,984,1]])
    elif(image == "calendar"):
        pqrs = np.array([[161,342,1], [506,3471,1], [2659,2596,1], [2677,769,1]])

    return pqrs

def get_square_pqrs_coordinates(image):
    pqrs = np.array([])

    if(image == "building"):
        pqrs = np.array([[243,204,1], [239,281,1], [296,291,1], [299,218,1]])
    elif(image == "nighthawks"):
        pqrs = np.array([[1095,271,1], [1095,598,1], [1408,628,1], [1400,236,1]])
    elif(image == "pikachu"):
        pqrs = np.array([[1071,596,1], [1108,2724,1], [2271,2432,1], [2312,984,1]])
    elif(image == "calendar"):
        pqrs = np.array([[748,1006,1], [789,1485,1], [1140,1480,1], [1103,1025,1]])

    return pqrs


if __name__ == "__main__":
    building = get_images(r"./hw3images/", "building.jpg")
    building_world_coordinates = get_width_height_coordinates("building")
    building_PQRS_coordinates = get_pqrs_coordinates("building")
    building_square_pqrs_coordinates = get_square_pqrs_coordinates("building")

    nighthawks = get_images(r"./hw3images/", "nighthawks.jpg")
    nighthawks_world_coordinates = get_width_height_coordinates("nighthawks")
    nighthawks_PQRS_coordinates = get_pqrs_coordinates("nighthawks")
    nighthawks_square_pqrs_coordinates = get_square_pqrs_coordinates("nighthawks")

    pikachu = get_images(r"./hw3images/", "pikachu.jpg")
    pikachu_world_coordinates = get_width_height_coordinates("pikachu")
    pikachu_PQRS_coordinates = get_pqrs_coordinates("pikachu")
    pikachu_square_pqrs_coordinates = get_square_pqrs_coordinates("pikachu")

    calendar = get_images(r"./hw3images/", "calendar.jpg")
    calendar_world_coordinates = get_width_height_coordinates("calendar")
    calendar_PQRS_coordinates = get_pqrs_coordinates("calendar")
    calendar_square_pqrs_coordinates = get_square_pqrs_coordinates("calendar")

    # Task 1.1a
    point_to_point_correspondence(building, building_world_coordinates.T, building_PQRS_coordinates.T, r"./Task 1a/building_transformed.jpg")
    point_to_point_correspondence(nighthawks, nighthawks_world_coordinates.T, nighthawks_PQRS_coordinates.T, r"./Task 1a/nighthawks_transformed.jpg")

    # Task 1.1b
    ## Two-Step
    two_step_method(building, building_PQRS_coordinates, r"./Task 1b/building_transformed.jpg")
    two_step_method(nighthawks, nighthawks_PQRS_coordinates, r"./Task 1b/nighthawks_transformed.jpg")

    ## One-Step
    one_step_method(building, building_PQRS_coordinates, building_square_pqrs_coordinates, r"./Task 1c/building_transformed.jpg")
    one_step_method(nighthawks, nighthawks_PQRS_coordinates, nighthawks_square_pqrs_coordinates, r"./Task 1c/nighthawks_transformed.jpg")

    # Task 2.1a
    point_to_point_correspondence(pikachu, pikachu_world_coordinates.T, pikachu_PQRS_coordinates.T, r"./Task 2a/pikachu_transformed.jpg")
    point_to_point_correspondence(calendar, calendar_world_coordinates.T, calendar_PQRS_coordinates.T, r"./Task 2a/calendar_transformed.jpg")

    # Task 2.1b
    ## Two-Step
    two_step_method(pikachu, pikachu_PQRS_coordinates, r"./Task 2b/pikachu_transformed.jpg")
    two_step_method(calendar, calendar_PQRS_coordinates, r"./Task 2b/calendar_transformed.jpg")

    ## One-Step
    one_step_method(pikachu, pikachu_PQRS_coordinates, pikachu_square_pqrs_coordinates, r"./Task 2c/pikachu_transformed.jpg")
    one_step_method(calendar, calendar_PQRS_coordinates, calendar_square_pqrs_coordinates, r"./Task 2c/calendar_transformed.jpg")

