# Name: Nikita Ravi
# Class: ECE 66100
# Homework #4
# Deadline: 09/28/2022

# Import Modules
import cv2
import math
import numpy as np

def display_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def get_images(path):
    img = cv2.imread(path)

    return img

def get_inputs():
    books_image1 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/books_1.jpeg")
    fountain_image1 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/fountain_1.jpg")
    books_image2 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/books_2.jpeg")
    fountain_image2 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/fountain_2.jpg")
    krannert_image1 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/krannert1.jpg")
    ubs_image1 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/ubs1.jpg")
    krannert_image2 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/krannert2.jpg")
    ubs_image2 = get_images(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/Figures/ubs2.jpg")

    return books_image1, fountain_image1, books_image2, fountain_image2, krannert_image1, ubs_image1, krannert_image2, ubs_image2

def get_path(subdir, sigma=None, filename=None):
    if(sigma and not filename):
        book_directory1 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/books_harris1.jpg"
        fountain_directory1 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/fountain_harris1.jpg"
        book_directory2 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/books_harris2.jpg"
        fountain_directory2 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/fountain_harris2.jpg"
        krannert_directory1 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/krannert1_harris.jpg"
        ubs_directory1 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/ubs1_harris.jpg"
        krannert_directory2 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/krannert2_harris.jpg"
        ubs_directory2 = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+"/sigma" + str(sigma) + "/ubs2_harris.jpg"
        return book_directory1, fountain_directory1, book_directory2, fountain_directory2, krannert_directory1, krannert_directory2, ubs_directory1, ubs_directory2

    elif(filename):
        directory = "/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/"+subdir+filename+str(sigma)+".jpg"
        return directory

def normalize_image(image):
    image = image.astype(np.float64)
    image -= np.min(image)
    image /= np.max(image)

    return image

def generate_haar_kernels(sigma):
    N = int(math.ceil(4 * sigma))
    N = N if not N%2 else N+1

    hx, hy = np.ones((N,N)), np.ones((N,N))
    hx[:, :int(N//2)] = -1
    hy[int(N//2):, :] = -1

    return hx, hy

def perform_thresholding(R):
    threshold = np.sort(R, axis=None)[-int(0.10 * len(R.flatten()))]
    return threshold

def perform_non_maximum_suppression(radius, height, width, threshold, R):
    corners = []
    for y in range(radius, height - radius):
            for x in range(radius, width - radius):
                sub_window = R[y - radius: y + radius, x - radius: x + radius]
                if(R[y, x] == np.amax(sub_window) and np.amax(sub_window) >= threshold):
                    corners.append([x, y])

    return corners

def harris_corner(image, sigma, e=0.001):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image #convert image to grayscale
    image = normalize_image(image) # Normalize the image
    height, width = image.shape 

    hx, hy = generate_haar_kernels(sigma)
    dx = cv2.filter2D(src = image, ddepth = -1, kernel = hx) # convolve image with hx kernel
    dy = cv2.filter2D(src = image, ddepth = -1, kernel = hy) # convolve image with hy kernel

    dx2 = dx * dx
    dy2 = dy * dy
    dxdy = dx * dy

    N = 2 * int((5 * sigma) // 2) + 1
    kernel_for_c = np.ones((N,N))

    sum_dx2 = cv2.filter2D(src = dx2, ddepth = -1, kernel = kernel_for_c)
    sum_dy2 = cv2.filter2D(src = dy2, ddepth = -1, kernel = kernel_for_c)
    sum_dxdy = cv2.filter2D(src = dxdy, ddepth = -1, kernel = kernel_for_c)

    trace = sum_dx2 + sum_dy2
    det = (sum_dx2 * sum_dy2) - (sum_dxdy)**2

    k = det / (trace**2 + e) # r / (1+r)^2
    k = np.sum(k) / (height * width) # Normalize
    
    R = det - k*(trace**2)
    R_threshold = perform_thresholding(R)

    radius = N // 2
    return perform_non_maximum_suppression(radius=radius, height=height, width=width, threshold=R_threshold, R=R)

def add_circles_around_corners(corners, image, color):
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for x, y in corners:
        image = cv2.circle(image, (x, y), radius=3, color=color)

    return image

def Harris(image, sigma, directory, color):
    corners = list(harris_corner(image=image, sigma=sigma))
    image = add_circles_around_corners(corners, image, color)
    cv2.imwrite(directory, image)

    return corners

def draw_correspondences(image1, image2, width1, correspondences, path):
    combined = np.hstack((image1, image2))
    for pt1, pt2, _ in correspondences:
        cv2.circle(combined, (pt1[0], pt1[1]), radius=3, color=(0,0,255))
        cv2.circle(combined, (pt2[0] + width1, pt2[1]), radius=3, color=(0,0,255))
        cv2.line(combined, pt1, (pt2[0] + width1, pt2[1]), color=(10,240,240), thickness=1)

    cv2.imwrite(path, combined)
    
def establish_correspondences(image1, image2, corners1, corners2, mode="ssd", radius=21, path=""):
    height1, width1, _ = image1.shape
    image2 = cv2.resize(image2, (width1, height1))
    height2, width2, _ = image2.shape

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if(len(image1.shape) == 3) else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if(len(image2.shape) == 3) else image2

    gray1 = normalize_image(gray1)
    gray2 = normalize_image(gray2)

    correspondences = []

    for c1 in corners1:
        bestMatch = None
        currBestDistance = float("inf")

        for c2 in corners2:
            distance = calculate_distance(gray1, gray2, c1, c2, mode, radius)
            if(distance < currBestDistance):
                currBestDistance = distance
                bestMatch = c2
        
        correspondences.append((c1, bestMatch, currBestDistance))
    
    draw_correspondences(image1, image2, width1, correspondences, path)

def calculate_distance(image1, image2, coord1, coord2, mode, radius=21):
    x1, y1 = coord1
    x2, y2 = coord2

    height1, width1 = image1.shape
    height2, width2 = image2.shape
    
    radius = min(x1, width1 - x1, y1, height1 - y1, 
        x2, width2 - x2, y2, height2 - y2, 
        radius)
    
    f1 = image1[y1 - radius: y1 + radius, x1 - radius: x1 + radius]
    f2 = image2[y2 - radius: y2 + radius, x2 - radius: x2 + radius]
    distance = None

    if(mode == "ssd"):
        distance = np.sum((f1 - f2)**2)
    elif(mode == "ncc"):
        m1, m2 = np.mean(f1), np.mean(f2)
        numerator = np.sum((f1-m1)*(f2-m2))
        denominator = np.sqrt(np.sum((f1-m1)**2) * np.sum((f2-m2)**2))
        distance = numerator / denominator
        distance = 1 - distance

    return distance

def sift(image1, image2, path=""):
    # Citation: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if (len(image1.shape) == 3) else image1
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if (len(image2.shape) == 3) else image2

    sift_detector = cv2.xfeatures2d.SIFT_create() # Create SIFT Detector
    keyPoint1, descriptor1 = sift_detector.detectAndCompute(image1, None)
    keyPoint2, descriptor2 = sift_detector.detectAndCompute(image2, None)

    bruteForceMatcher = cv2.BFMatcher() # It takes the descriptor of one feature in first set and matches it with all other features in second set using some distance calculation - the closest distance is returned
    matches = bruteForceMatcher.knnMatch(descriptor1, descriptor2, k=2) # returns k best matches where k is specified by the user

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if(m.distance < 0.75 * n.distance):
            good_matches.append([m])

    combined = np.hstack((image1, image2))
    result = cv2.drawMatchesKnn(image1, keyPoint1, image2, keyPoint2, good_matches, combined, flags=2)
    cv2.imwrite("/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw04/SIFT/"+path, result)


if __name__ == "__main__":
    # Get Images
    books_image1, fountain_image1, books_image2, fountain_image2, krannert_image1, ubs_image1, krannert_image2, ubs_image2  = get_inputs()

    # Task 1 
    scale = [1.8,3.2,5.4,7.3]
    for sigma in scale:
        print("Sigma: ", sigma)
        books_copy1, fountain_copy1, books_copy2, fountain_copy2  = books_image1.copy(), fountain_image1.copy(), books_image2.copy(), fountain_image2.copy()
        krannert_copy1, krannert_copy2, ubs_copy1, ubs_copy2 = krannert_image1.copy(), krannert_image2.copy(), ubs_image1.copy(), ubs_image2.copy()
        book_directory1, fountain_directory1, book_directory2, fountain_directory2, krannert_directory1, krannert_directory2, ubs_directory_1, ubs_directory_2 = get_path("Harris", sigma)

        # 1a - Harris Corner Detection
        book_corners1 = Harris(books_copy1, sigma, book_directory1, (0,0,255))
        fountain_corners1 = Harris(fountain_copy1, sigma, fountain_directory1, (0,255,0))

        book_corners2 = Harris(books_copy2, sigma, book_directory2, (0,0,255))
        fountain_corners2 = Harris(fountain_copy2, sigma, fountain_directory2, (0,255,0))

        krannert_corners1 = Harris(krannert_copy1, sigma, krannert_directory1, (0,0,255))
        ubs_corners1 = Harris(ubs_copy1, sigma, ubs_directory_1, (0,255,0))

        krannert_corners2 = Harris(krannert_copy2, sigma, krannert_directory2, (0,0,255))
        ubs_corners2 = Harris(ubs_copy2, sigma, ubs_directory_2, (0,255,0))

        # 1b - Establish Correspondences
        books_copy1, books_copy2, fountain_copy1, fountain_copy2 = books_image1.copy(), books_image2.copy(), fountain_image1.copy(), fountain_image2.copy()
        establish_correspondences(books_copy1, books_copy2, book_corners1, book_corners2, mode="ssd", path=get_path("Harris/correspondences/", sigma=sigma, filename="books_corr_ssd"))
        establish_correspondences(fountain_copy1, fountain_copy2, fountain_corners1, fountain_corners2, mode="ssd", path=get_path("Harris/correspondences/", sigma=sigma, filename="fountain_corr_ssd"))
        establish_correspondences(books_copy1, books_copy2, book_corners1, book_corners2, mode="ncc", path=get_path("Harris/correspondences/", sigma=sigma, filename="books_corr_ncc"))
        establish_correspondences(fountain_copy1, fountain_copy2, fountain_corners1, fountain_corners2, mode="ncc", path=get_path("Harris/correspondences/", sigma=sigma, filename="fountain_corr_ncc"))

        establish_correspondences(krannert_copy1, krannert_copy2, krannert_corners1, krannert_corners2, mode="ssd", path=get_path("Harris/correspondences/", sigma=sigma, filename="krannert_corr_ssd"))
        establish_correspondences(ubs_copy1, ubs_copy2, ubs_corners1, ubs_corners2, mode="ssd", path=get_path("Harris/correspondences/", sigma=sigma, filename="ubs_corr_ssd"))
        establish_correspondences(krannert_copy1, krannert_copy2, krannert_corners1, krannert_corners2, mode="ncc", path=get_path("Harris/correspondences/", sigma=sigma, filename="krannert_corr_ncc"))
        establish_correspondences(ubs_copy1, ubs_copy2, ubs_corners1, ubs_corners2, mode="ncc", path=get_path("Harris/correspondences/", sigma=sigma, filename="ubs_corr_ncc"))


    # Task 2 - SIFT 
    books_copy1, books_copy2, fountain_copy1, fountain_copy2 = books_image1.copy(), books_image2.copy(), fountain_image1.copy(), fountain_image2.copy()
    krannert_copy1, krannert_copy2, ubs_copy1, ubs_copy2 = krannert_image1.copy(), krannert_image2.copy(), ubs_image1.copy(), ubs_image2.copy()
    
    sift(books_copy1, books_copy2, "books.jpg")
    sift(fountain_copy1, fountain_copy2, "fountain.jpg")
    sift(krannert_copy1, krannert_copy2, "krannert.jpg")
    sift(ubs_copy1, ubs_copy2, "ubs.jpg")
