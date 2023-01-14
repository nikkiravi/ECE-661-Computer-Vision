# Import modules
import cv2
import numpy as np
import os

def get_images(path):
    list_of_files = os.listdir(path)[::-1]
    images = []

    for idx in range(len(list_of_files)):
        img = list_of_files[idx]
        images.append(cv2.imread(path + img))

    return images

def display_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def write_image(path, image):
    cv2.imwrite(path, np.uint8(image * 255))

def compute_otsu_threshold(image):
    hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    N = len(image)

    total = sum(hist * bin_edges[:-1])
    w0 = 0 # Probability of it being a background class
    sum_of_background_pixels = 0
    best_fisher_discriminant_numerator = -1
    threshold = 0  

    for k in range(256):
        w0, w1 = sum(hist[:k]), sum(hist[k+1:]) #w1 = probability of it being a foreground class
        if(not w0 or not w1):
            continue

        sum_of_background_pixels += k * hist[k]
        sum_of_foreground_pixels = np.int32(total - sum_of_background_pixels)

        mu0 = sum_of_background_pixels / w0 # Mean of background class
        mu1 = sum_of_foreground_pixels / w1 # Mean of foreground class

        curr_fisher_discriminant_numerator = w0 * w1 * (mu0 - mu1)**2
        if(curr_fisher_discriminant_numerator > best_fisher_discriminant_numerator):
            best_fisher_discriminant_numerator = curr_fisher_discriminant_numerator
            threshold = k
    
    return threshold

def otsu(image, num_iterations, foreground_flag):
    image_flattened = image.flatten()
    for iteration in range(num_iterations):
        threshold = compute_otsu_threshold(image_flattened)
        mask = np.zeros(image.shape, dtype=np.uint8)

        if(foreground_flag): #low level is foreground
            mask[image <= threshold] = 1
            image_flattened = np.array([i for i in image_flattened if i <= threshold])
        else: #high level is foreground
            mask[image > threshold] = 1
            image_flattened = np.array([i for i in image_flattened if i <= threshold])

    return mask

def erosion(mask, size, iteration):
    kernel = np.ones((size, size), dtype=np.uint8)
    mask = cv2.erode(np.float32(mask), kernel, iterations=iteration)

    return mask

def dilation(mask, size, iteration):
    kernel = np.ones((size, size), dtype=np.uint8)
    mask = cv2.dilate(np.float32(mask), kernel, iterations=iteration)

    return mask

def opening(mask, erosion_size, erosion_iteration, dilation_size, dilation_iteration, path):
    result = erosion(mask, erosion_size, erosion_iteration)
    write_image(path + "_erosion.jpg", result)

    result = dilation(result, dilation_size, dilation_iteration)
    write_image(path + "_dilation.jpg", result)

    return result

def closing(mask, erosion_size, erosion_iteration, dilation_size, dilation_iteration, path):
    result = dilation(mask, dilation_size, dilation_iteration)
    write_image(path + "_dilation.jpg", result)

    result = erosion(result, erosion_size, erosion_iteration)
    write_image(path + "_erosion.jpg", result)

    return result

def contour(mask):
    contour = np.zeros(mask.shape, dtype=np.uint8)
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            if(not mask[j][i]):
                continue

            kernel = mask[j-1:j+2, i-1:i+2]
            if(sum(kernel.flatten()) < 9):
                contour[j][i] = 1

    return contour

def texture_segmentation(gray, N=3):
    height, width = gray.shape
    result = np.zeros((height, width))
    half_N = int((N-1)/2)

    for j in range(height):
        for i in range(width):
            xi = max(0, j - half_N)
            xf = min(height, j + half_N + 1)
            yi = max(0, i - half_N)
            yf = min(width, i + half_N + 1)

            kernel = gray[xi:xf, yi:yf]
            result[j][i] = np.var(kernel)
    result = np.uint8(np.round(255 * result / (np.max(result)-np.min(result))))
    return result

def task1(image, name, iterations, labels, foreground_flag, erosion_size, erosion_iter, dilation_size, dilation_iter):
    mask = np.zeros(image.shape, dtype=np.uint8)
    channels = cv2.split(image)
    path = "hw06/rgbSegmentation/" + name + "/"

    for c in range(3):
        layer = channels[c]
        mask[:, :, c] = otsu(layer, num_iterations=iterations[c], foreground_flag=foreground_flag[c])
        write_image(path + labels[c] + ".jpg", mask[:, :, c])
    
    overall_mask = mask[:,:,0] * mask[:,:,1] * mask[:,:,2]
    write_image(path + "overall.jpg", overall_mask)

    closing_result = closing(overall_mask, erosion_size, erosion_iter, dilation_size, dilation_iter, path + "closing")
    opening_result = opening(overall_mask, erosion_size, erosion_iter, dilation_size, dilation_iter, path + "opening")

    just_contour = contour(overall_mask)
    write_image(path + "contour.jpg", just_contour)

    closing_contour = contour(closing_result)
    write_image(path + "contour_closing.jpg", closing_contour)

    opening_contour = contour(opening_result)
    write_image(path + "contour_opening.jpg", opening_contour)

def task2(image, gray, name, iterations, labels, foreground_flag, erosion_size, erosion_iter, dilation_size, dilation_iter):
    mask = np.zeros(image.shape, dtype=np.uint8)
    window_sizes = [int(label) for label in labels]
    path = "hw06/textualSegmentation/" + name + "/"

    for c, N in enumerate(window_sizes):
        texture_layer = texture_segmentation(gray, N)
        mask[:, :, c] = otsu(texture_layer, iterations[c], foreground_flag[c])
        write_image(path + labels[c] + ".jpg", mask[:, :, c])
    
    overall_mask = np.logical_and(mask[:,:,0], mask[:,:,1], mask[:,:,2])
    write_image(path + "overall.jpg", overall_mask)

    closing_result = closing(overall_mask, erosion_size, erosion_iter, dilation_size, dilation_iter, path + "closing")
    opening_result = opening(overall_mask, erosion_size, erosion_iter, dilation_size, dilation_iter, path + "opening")

    just_contour = contour(overall_mask)
    write_image(path + "contour.jpg", just_contour)

    closing_contour = contour(closing_result)
    write_image(path + "contour_closing.jpg", closing_contour)

    opening_contour = contour(opening_result)
    write_image(path + "contour_opening.jpg", opening_contour)

def car(image, gray, name):
    task1(image, name, iterations=[1,1,1], labels=['blue', 'green', 'red'], foreground_flag=[1,0,1], erosion_size=3, erosion_iter=1, dilation_size=2, dilation_iter=1)
    task2(image, gray, name, iterations=[1,1,1], labels=["5", "7", "9"], foreground_flag=[0,0,1], erosion_size=2, erosion_iter=1, dilation_size=2, dilation_iter=1)

def cat(image, gray, name):
    task1(image, name, iterations=[1,1,1], labels=['blue', 'green', 'red'], foreground_flag=[1,1,0], erosion_size=3, erosion_iter=1, dilation_size=2, dilation_iter=1)
    task2(image, gray, name, iterations=[1,1,1], labels=["5", "7", "9"], foreground_flag=[0,0,1], erosion_size=2, erosion_iter=1, dilation_size=2, dilation_iter=1)

def dog(image, gray, name):
    task1(image, name, iterations=[1,1,1], labels=['blue', 'green', 'red'], foreground_flag=[1,1,1], erosion_size=3, erosion_iter=1, dilation_size=2, dilation_iter=1)
    task2(image, gray, name, iterations=[1,1,1], labels=["5", "7", "9"], foreground_flag=[1,0,1], erosion_size=2, erosion_iter=1, dilation_size=2, dilation_iter=1)

def soccer(image, gray, name):
    # task1(image, name, iterations=[1,1,1], labels=['blue', 'green', 'red'], foreground_flag=[0,0,0], erosion_size=3, erosion_iter=1, dilation_size=2, dilation_iter=1)
    task2(image, gray, name, iterations=[1,1,1], labels=["5", "7", "9"], foreground_flag=[1,0,1], erosion_size=2, erosion_iter=1, dilation_size=2, dilation_iter=1)

if __name__ == "__main__":
    images = get_images(r"hw06/Images/")
    names = ["car", "cat", "soccer", "dog"]
    copies = [img.copy() for img in images]
    gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if(len(img.shape) == 3) else img for img in copies]

    car(copies[0], gray[0], names[0])
    cat(copies[1], gray[1], names[1])
    soccer(copies[2], gray[2], names[2])
    dog(copies[3], gray[3], names[3])