# Name: Nikita Ravi
# Class: ECE 66100
# Homework #5
# Deadline: 09/28/2022

# Import Modules
import cv2
import math
import numpy as np
import os
from scipy.optimize import least_squares

def get_images(path, task=1):
    path = path + "/Task" + str(task)
    images = [cv2.imread(path + "/" + img) for img in sorted(os.listdir(path)[::-1])]
    return images

def display_images(images, idx=None):
    if(idx):
        cv2.imshow("Window", images[idx])
    else:
        cv2.imshow("Window", images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def compute_homography(domain_coord, range_coord, over=False):
    # From homework #3
    n = domain_coord.shape[1]
    A = np.zeros((2*n, 8))
    b = np.zeros((2*n, 1))
    H = np.zeros((3,3))

    for idx in range(n):
        A[2*idx] = [domain_coord[0][idx], domain_coord[1][idx], 1, 0, 0, 0, (-domain_coord[0][idx] * range_coord[0][idx]), (-domain_coord[1][idx] * range_coord[0][idx])]
        A[2*idx + 1] = [0, 0, 0, domain_coord[0][idx], domain_coord[1][idx], 1, (-domain_coord[0][idx] * range_coord[1][idx]), (-domain_coord[1][idx] * range_coord[1][idx])]
        b[2*idx] = range_coord[0][idx]
        b[2*idx + 1] = range_coord[1][idx]

    h = np.array([])
    if(over):
        A_plus = np.linalg.pinv(np.matmul(A.T, A)) #pseudo-inverse of A
        h = np.dot(np.matmul(A_plus, A.T), b)
    else:
        h = np.dot(np.linalg.pinv(A), b)

    row = 0
    for idx in range(0, len(h), 3):
        spliced = h[idx:idx+3]
        if(len(spliced) == 3):
            H[row] = spliced.T
        else:
            H[row] = np.append(spliced, [1])
        row += 1

    return H

def establish_correspondences(image1, descriptor1, keyPoint1, image2, descriptor2, keyPoint2, path):
    # Citation: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    bruteForceMatcher = cv2.BFMatcher() # It takes the descriptor of one feature in first set and matches it with all other features in second set using some distance calculation - the closest distance is returned
    matches = bruteForceMatcher.match(descriptor1, descriptor2) # returns matches where k is specified by the user
    matches = sorted(matches, key=lambda x: x.distance)
    domain_coord, range_coord = [], []

    for match in matches:
        domain_point = list(map(int, keyPoint1[match.queryIdx].pt))
        domain_point.append(1)
        domain_coord.append(domain_point)
        range_point = list(map(int, keyPoint2[match.trainIdx].pt))
        range_point.append(1)
        range_coord.append(range_point)

    combined = np.hstack((image1, image2))
    result = cv2.drawMatches(image1, keyPoint1, image2, keyPoint2, matches[:100], combined, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(path, result)

    domain_coord, range_coord = domain_coord, range_coord
    return domain_coord, range_coord

def sift(image1, image2, idx1, idx2):
    sift_detector = cv2.xfeatures2d.SIFT_create() # Create SIFT Detector
    keyPoint1, descriptor1 = sift_detector.detectAndCompute(image1, None)
    keyPoint2, descriptor2 = sift_detector.detectAndCompute(image2, None)

    return establish_correspondences(image1, descriptor1, keyPoint1, image2, descriptor2, keyPoint2, "hw05/sift_correspondences/Task1/image" + str(idx1) + "image" + str(idx2) +".jpg")

def find_inliers(H, domain_coord, range_coord, delta):
    range_hat = np.matmul(H, domain_coord.T).T
    range_hat = range_hat.T / range_hat.T[2,:]
    range_hat = range_hat.T
    error = np.sum(np.abs(range_hat - range_coord)**2, axis=1)
    indices = np.where(error <= delta)[0]
    return indices

def ransac(images, task=1, LM=True):
    """
    Constants
    n = 5 (4 < n < 10)
    n_total = number of correspondences
    epsilon = 0.85
    p = 0.99
    N = ln(1-p)/(ln[1-(1-epsilon)^n])
    M = (1-epsilon) * n_total
    delta = 3 * sigma - typical p.11.3
    sigma = 2 - typical p11.3
    """
    # Constants
    n = 5
    epsilon = 0.85
    p = 0.99
    N = int(math.log(1-p) / (math.log(1-(1-epsilon)**n)))
    sigma = 2
    delta = 3 * sigma

    # Inputs
    images = [img.copy() for img in images]
    gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if(len(img.shape) == 3) else img for img in images]
    H_list = []
    best_inliers = []

    for idx in range(len(gray) - 1):
        domain_gray, domain_image = gray[idx], images[idx]
        range_gray, range_image = gray[idx + 1], images[idx + 1]
        domain_coord, range_coord = sift(domain_gray, range_gray, idx, idx + 1)
        domain_coord, range_coord = np.array(domain_coord), np.array(range_coord)
        
        n_total = domain_coord.shape[0]
        M = int((1 - epsilon) * n_total)
        inlier_percentage = 0
        indices = list(range(n_total))

        best_inlier_idx = None
        for trial in range(N): # Conduct N Trials
            random_sample_indices = np.random.choice(indices, n) # Randomly select a sample of indices of n correspondences
            random_domains = domain_coord[random_sample_indices]
            random_ranges = range_coord[random_sample_indices]
            H_with_noise = compute_homography(random_domains.T, random_ranges.T, over=False)
            inlier_idx = find_inliers(H_with_noise, domain_coord, range_coord, delta)

            num_inlier_indices = len(inlier_idx)
            if(num_inlier_indices / n_total >= inlier_percentage):
                inlier_percentage = num_inlier_indices / n_total
                best_inlier_idx = inlier_idx
        
        if(len(best_inlier_idx) < M):
            print("WARNING: Number of inlier points is less than the minimum requirement")

        best_inliers.append(best_inlier_idx)
        H = compute_homography(domain_coord[best_inlier_idx].T, range_coord[best_inlier_idx].T, over=True)
        if(LM):
            h_vector = H.flatten()
            result = least_squares(cost_function, h_vector, args=(domain_coord[best_inlier_idx], range_coord[best_inlier_idx]), method="lm")
            h_prime = result.x
            H = h_prime.reshape((3,3))
        H_list.append(H)
        display_inliers_outliers(best_inliers, domain_coord, range_coord, domain_image, range_image, "hw05/inlier_outlier/Task" + str(task) + "/image" + str(idx) + "image" + str(idx + 1))
    return H_list, best_inliers

def cost_function(h, domain_coord, range_coord):
    X = []
    F = []
    #[(x,y,1),...
    for idx in range(len(domain_coord)):
        xd = domain_coord[idx][0]
        yd = domain_coord[idx][1]
        xr = range_coord[idx][0]
        yr = range_coord[idx][1]
        X.append(xr)
        X.append(yr)

        F.append((h[0] * xd + h[1] * yd + h[2]) / (h[6] * xd + h[7] * yd + h[8]))
        F.append((h[3] * xd + h[4] * yd + h[5]) / (h[6] * xd + h[7] * yd + h[8]))

    return np.array(X) - np.array(F)

def display_inliers_outliers(inlier_indices, domain_coord, range_coord, image1, image2, path):
    combined_inliers, combined_outliers = np.hstack((image1, image2)), np.hstack((image1, image2))
    height, width, channel = image1.shape
    inlier_indices = inlier_indices[0].tolist()
    
    for idx in range(domain_coord.shape[0]):
        # Inliers
        if(idx in inlier_indices):
            pt1 = tuple(domain_coord[idx])
            pt2 = range_coord[idx][0] + width, range_coord[idx][1]

            cv2.circle(combined_inliers, pt1[0:2], 3, (255, 0, 0), 1)
            cv2.circle(combined_inliers, pt2[0:2], 3, (0, 255, 0), 1)
            cv2.line(combined_inliers, pt1[0:2], pt2[0:2], (0, 0, 255), 1)
        
        else:
            # Outliers
            pt1 = tuple(domain_coord[idx])
            pt2 = (range_coord[idx][0] + width, range_coord[idx][1])

            cv2.circle(combined_outliers, pt1[0:2], 3, (255, 0, 0), 1)
            cv2.circle(combined_outliers, pt2[0:2], 3, (0, 255, 0), 1)
            cv2.line(combined_outliers, pt1[0:2], pt2[0:2], (0, 0, 255), 1)

    cv2.imwrite(path + "inliers.jpg", combined_inliers)
    cv2.imwrite(path + "outliers.jpg", combined_outliers)

def generate_panorama(H, images, path):
    images = [img.copy() for img in images]
    N = len(images)
    mid = N // 2

    midH = np.eye(3)
    for idx in range(mid, len(H)):
        midH = np.matmul(midH, np.linalg.inv(H[idx])) #H23, H34
        H[idx] = midH
    
    midH = np.eye(3)
    for idx in range(mid - 1, -1, -1):
        midH = np.matmul(midH, H[idx]) #H12, H01
        H[idx] = midH

    H.insert(mid, np.eye(3))
    tx = 0
    for idx in range(mid):
        tx += images[idx].shape[1]
    translation = np.array([[1,0,tx], [0,1,0], [0,0,1]], dtype=float)

    height, width = 0, 0
    for idx in range(N):
        height = max(height, images[idx].shape[0])
        width += images[idx].shape[1]

    combined = np.zeros((height, width, 3), np.uint8)
    for idx in range(N):
        H_curr = np.matmul(translation, H[idx])
        combined = pixel_mapping(combined, images[idx], H_curr)

    cv2.imwrite(path + "_panorama.jpg", combined)

def pixel_mapping(combined, image, H):
    height, width, _ = image.shape
    combined_height, combined_width, _ = combined.shape
    H = np.linalg.inv(H)

    Xarray, Yarray = np.meshgrid(np.arange(0, combined_width, 1), np.arange(0, combined_height, 1))
    pixels = np.vstack((Xarray.ravel(), Yarray.ravel())).T
    pixels = np.hstack((pixels[:, 0:2], pixels[:, 0:1]*0+1))

    transformed_pixel = np.dot(H, pixels.T)
    transformed_pixel = transformed_pixel / transformed_pixel[2, :]
    transformed_pixel = transformed_pixel.T[:, 0:2].astype('int')

    valid_pixels, valid_transformed = find_valid_pixels(pixels, transformed_pixel, width - 1, height - 1)
    for idx in range(valid_pixels.shape[0]):
        if((combined[valid_pixels[idx, 1], valid_pixels[idx, 0]] != 0).all() == False):
            combined[valid_pixels[idx, 1], valid_pixels[idx, 0]] = image[valid_transformed[idx, 1], valid_transformed[idx, 0]]

    return combined

def find_valid_pixels(pixels, transformed, width, height):
    xMin = transformed[:,0] >= 0
    transformed = transformed[xMin, :]
    pixels = pixels[xMin, :]

    xMax = transformed[:,0] <= width
    transformed = transformed[xMax, :]
    pixels = pixels[xMax, :]

    yMin = transformed[:,1] >= 0
    transformed = transformed[yMin, :]
    pixels = pixels[yMin, :]

    yMax = transformed[:,1] <= height
    transformed = transformed[yMax, :]
    pixels = pixels[yMax, :]    

    return pixels, transformed


if __name__=="__main__":
    # Task 1
    images1 = get_images(r"hw05/Images", task=1)
    optimumH, best_inliers = ransac(images1, task=1, LM=False)
    levenmarq = "without_LM"
    generate_panorama(optimumH, images1, path="hw05/panaroma/Task"+str(1)+"/" + levenmarq)
    
    optimumH, best_inliers = ransac(images1, task=1, LM=True)
    levenmarq = "LM"
    generate_panorama(optimumH, images1, path="hw05/panaroma/Task"+str(1)+"/" + levenmarq)

    # Task 2
    images2 = get_images(r"hw05/Images", task=2)
    optimumH, best_inliers = ransac(images2, task=2, LM=False)
    levenmarq = "without_LM"
    generate_panorama(optimumH, images2, path="hw05/panaroma/Task"+str(2)+"/" + levenmarq)
    
    optimumH, best_inliers = ransac(images2, task=2, LM=True)
    levenmarq = "LM"
    generate_panorama(optimumH, images2, path="hw05/panaroma/Task"+str(2)+"/" + levenmarq)