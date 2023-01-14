# Name: Nikita Ravi
# Class: ECE 66100
# Homework #9
# Deadline: 11/27/2022

# Import Modules
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
		
def display_image(image, points=True):
	def click_event(event, x, y, flags, params):
		# This function was inspired by https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
		if(event == cv2.EVENT_LBUTTONDOWN):
			print(x, y)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.circle(image, (x, y), 1, (0,0,255), thickness=-1)
			cv2.putText(image, str(x) + ',' +
						str(y), (x,y), font,
						1, (0, 0, 255), 2)
			cv2.imshow('window', image)


	cv2.imshow("window", image)
	if(points):
		cv2.setMouseCallback('window', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	quit()

def draw_circles(path, image, pts):
	for idx, pt in enumerate(pts):
		x, y = int(pt[0]), int(pt[1])
		cv2.circle(image, (x, y), radius=6, color=(0,0,255), thickness=-1)

	cv2.imwrite(path, image)

def normalize_points(correspondences):
	"""
	Normalization is necessary for the 8-point algorithm of constructing the fundamental matrix to be successful. The suggested normalization 
	is a translation and scaling of each image so that the centroid of the reference points is at the origin of the coordinates and the RMS 
	distance of the points from the origin is equal to sqrt(2) (Hartley and Zisserman, 2003).
	"""
	x_coord = correspondences[:, 0]
	y_coord = correspondences[:, 1]

	# Compute the mean distance
	x_mean, y_mean = np.mean(x_coord), np.mean(y_coord)
	distances = np.sqrt((x_coord - x_mean) ** 2 + (y_coord - y_mean) ** 2)
	mean_dist = np.mean(distances)

	# Create normalizing transformation matrix consisting of translation and scaling
	c = np.sqrt(2) / mean_dist
	tMat = np.array([[c, 0, -c*x_mean], 
					[0, c, -c*y_mean],
					[0, 0, 1]])

	# Apply the translation matrices
	normalized_pts = np.hstack((correspondences, np.ones((len(correspondences), 1)))) # Make it homogeneous 
	normalized_pts = np.dot(tMat, normalized_pts.T).T

	return normalized_pts, tMat

def conditionF(F):
	# Fundamental matrix F must be conditioned to enforce the requirement rank(F) = 2 to construct a linear least squares solution
	u, d, vh = np.linalg.svd(F)
	d_prime = np.array([[d[0], 0, 0],
						[0, d[1], 0],
						[0, 0, 0]])
	conditionedF = u @ d_prime @ vh
	return conditionedF

def calculate_fundamental_matrix(normalized_ptsL, normalized_ptsR):
	A = np.zeros((8, 9))
	for corr in range(len(normalized_ptsL)):
		x1, y1 = normalized_ptsL[corr, 0], normalized_ptsL[corr, 1]
		x2, y2 = normalized_ptsR[corr, 0], normalized_ptsR[corr, 1]

		A[corr] = np.array([x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

	u, s, vh = np.linalg.svd(A)
	F_flatten = vh[-1]
	F = np.reshape(F_flatten, (3, 3))
	return conditionF(F)

def compute_Ex(e):
	E_x = np.array([[0, -e[2], e[1]],
					[e[2], 0, -e[0]],
					[-e[1], e[0], 0]])

	return E_x

def find_epipoles(F):
	u,s,vh = np.linalg.svd(F) 
	eL = vh[-1, :].T
	eL = eL / eL[2] # Make it homogeneous 

	eR = u[:, -1] 
	eR = eR / eR[2] # Make it homogeneous

	E_x = compute_Ex(eR)
	return eL, eR, E_x

def compute_projection_matrices(F, eR, s):
	P1 = np.hstack((np.eye(3), np.zeros((3,1))))
	P2 = np.hstack((np.dot(s, F), np.transpose([eR])))

	return P1, P2

def cost_function(f, ptsL, ptsR):
	F = np.reshape(f, (3, 3))
	eL, eR, E_x = find_epipoles(F)
	P1, P2 = compute_projection_matrices(F, eR, E_x)

	ptsL = np.hstack((ptsL, np.ones((len(ptsL), 1)))) # Make it homogeneous
	ptsR = np.hstack((ptsR, np.ones((len(ptsR), 1)))) # Make it homogeneous

	d_geom = []
	for corr in range(len(ptsL)):
		A = np.zeros((4, 4)) # AX = 0
		A[0] = ptsL[corr][0] * P1[2, :] - P1[0, :]
		A[1] = ptsL[corr][1] * P1[2, :] - P1[1, :]
		A[2] = ptsR[corr][0] * P2[2, :] - P2[0, :]
		A[3] = ptsR[corr][1] * P2[2, :] - P2[1, :]

		# Compute X
		u, s, vh = np.linalg.svd(A)
		X = vh[-1, :].T
		Xhat = X / X[3] # Unit Vector of X
		xL = np.dot(P1, Xhat)
		xL = xL / xL[2] # Make it Homogeneous

		xR = np.dot(P2, Xhat)
		xR = xR / xR[2] # Make it Homogeneous

		# Append the re-projection errors 
		d_geom.append(np.linalg.norm(xL - ptsL[corr]) ** 2)
		d_geom.append(np.linalg.norm(xR - ptsR[corr]) ** 2)

	return np.ravel(d_geom)

def compute_homography_rectification(image, corrsL, corrsR, eL, eR):
	height, width = image.shape[:2]

	# Translation Matrix to translate the image such that the center is at the origin
	T = np.array([[1, 0, -width / 2],
				[0, 1, -height / 2],
				[0, 0, 1]]) # Obtained Translation Matrix from https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
	
	##### Calculate HR or H' #####
	# Rotation Matrix to move the epipoles towards the x-axis
	theta = -np.arctan((eR[1] - height/2)/(eR[0] - width/2)) 
	R = np.array([[math.cos(theta), -math.sin(theta), 0],
				[math.sin(theta), math.cos(theta), 0],
				[0, 0, 1]]) # Inspired by 2020 hw#1 solution

	# Create matrix that would send the epipoles to infinity
	f = (eR[0] - width / 2) * math.cos(theta) - (eR[1] - height / 2) * math.sin(theta) # Inspired by 2020 hw#1 solution
	G = np.array([[1, 0, 0],
				[0, 1, 0],
				[-1 / f, 0, 1]]) 

	# Rectify the right image center
	HR_center = np.dot(G, np.dot(R, T))
	Xprime = np.dot(HR_center, np.array([width / 2, height / 2, 1])) # Apply HR_center to rectify the center of the right image (x', y')
	T2 = np.array([[1, 0, width / 2 - Xprime[0]],
					[0, 1, height / 2 - Xprime[1]],
					[0, 0, 1]]) # Move the rectified center back to the true image center
	
	# Rectify the entire right image
	HR = np.dot(np.dot(T2, G), np.dot(R, T))
	HR = HR / HR[2, 2] # Make it Homogeneous

	##### Calculate HL or H #####
	# Create a rotation matrix to move the epipole to the x-axis
	theta = math.atan2(eL[1] - height / 2, (eL[0] - width / 2))
	R = np.array([[math.cos(theta), -math.sin(theta), 0],
				[math.sin(theta), math.cos(theta), 0],
				[0, 0, 1]]) # Inspired by 2020 hw#1 solution

	# Create matrix that would send the epipoles to infinity
	f = math.cos(theta) * (eL[0] - width / 2) - math.sin(theta) * (eL[1] - height / 2) # Inspired by 2020 hw#1 solution
	G = np.array([[1, 0, 0],
				[0, 1, 0],
				[-1 / f, 0, 1]]) 

	# Rectify the left image center
	H0 = G @ R @ T

	# HL is obtained by the matrix that minimizes the least squares fistance 
	# pg 307 (Hartley and Zisserman, 2003)
	ptsL = np.hstack((corrsL, np.ones((len(corrsL), 1)))) # Make it Homogeneous
	ptsR = np.hstack((corrsR, np.ones((len(corrsR), 1)))) # Make it Homogeneous
	
	# Transform the points by their respective homographies
	xL = np.dot(H0, ptsL.T).T
	xR = np.dot(HR, ptsR.T).T
	
	# Make them Homogeneous
	# xL = xL / xL[:, 2]
	xL[:, 0] = xL[:, 0] / xL[:, 2]
	xL[:, 1] = xL[:, 1] / xL[:, 2]
	xL[:, 2] = xL[:, 2] / xL[:, 2]
	
	xR[:, 0] = xR[:, 0] / xR[:, 2] # Only need the x-coordinate as shown on pg. 307 to minimize the dist

	# The least squares solution to an mxn system of equations Ax=b of rank n is given by x = (A^+)b which minimizes ||x||. (pg. 590)
	a,b,c = np.dot(np.linalg.pinv(xL), xR[:, 0])
	Ha = np.array([[a, b, c], [0, 1, 0], [0, 0, 1]]) # Eq. 11.20 pg 306
	HL_center = np.dot(Ha, H0)
	Xprime = np.dot(HL_center, np.array([width / 2, height / 2, 1])) # Apply HR_center to rectify the center of the right image (x', y')
	Xprime = Xprime / Xprime[2] # Make it Homogeneous

	T2 = np.array([[1, 0, width / 2 - Xprime[0]],
					[0, 1, height / 2 - Xprime[1]],
					[0, 0, 1]]) # Move the rectified center back to the true image center

	# Rectify the entire image
	HL = np.dot(T2, HL_center)
	HL = HL / HL[2, 2]

	return HL, HR
	
def transform(domain_image, H):
	height, width = domain_image.shape[:2]
	return cv2.warpPerspective(domain_image, H, dsize=(height, width))

def detect_edges(rectified_img, low_threshold=60, high_threshold=70, edges=5):
	gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY) if len(rectified_img.shape) == 3 else rectified_img
	gray = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=1.3)

	return cv2.Canny(gray, threshold1=high_threshold, threshold2=low_threshold, edges=edges)

def filter_correspondences(imgL, imgR, corrs, kernel_size, maximum_corrs, mode="ncc"):
	half_kernel_size = kernel_size // 2
	if(maximum_corrs > len(corrs)):
		print("Maximum correspondences exceeded the number of correspondences")
		maximum_corrs = len(corrs)

	dist = []
	for corr in corrs:
		corrL, corrR = corr
		f1 = imgL[corrL[1] - half_kernel_size:corrL[1] + half_kernel_size + 1, 
				  corrL[0] - half_kernel_size:corrL[0] + half_kernel_size + 1]

		f2 = imgR[corrR[1] - half_kernel_size:corrR[1] + half_kernel_size + 1, 
				  corrR[0] - half_kernel_size:corrR[0] + half_kernel_size + 1]

		if(f1.shape != f2.shape):
			continue

		if(mode == "ssd"):
			dist.append(np.sum((f1 - f2) ** 2))
		elif(mode == "ncc"):
			m1, m2 = np.mean(f1), np.mean(f2)
			numerator = np.sum((f1-m1)*(f2-m2))
			denominator = np.sqrt(np.sum((f1-m1)**2) * np.sum((f2-m2)**2))
			distance = numerator / denominator
			dist.append(1 - distance)

	# Sort the correspondences and choose the best max_correspondence
	sorted_correspondences = [pt for d, pt in sorted(zip(dist, corrs), key=lambda x: x[0])]
	return sorted_correspondences[0:maximum_corrs]

def get_color():
	r = random.randint(0,255)
	g = random.randint(0,255)
	b = random.randint(0,255)
	return (r,g,b)

def draw_correspondences(imgL, imgR, width1, correspondences, path):
	# From homework 4
	combined = np.hstack((imgL, imgR))
	for idx, corr in enumerate(correspondences):
		if(not idx % 10):
			pt1, pt2 = corr
			cv2.circle(combined, (pt1[0], pt1[1]), radius=3, color=(0,0,255))
			cv2.circle(combined, (pt2[0] + width1, pt2[1]), radius=3, color=(0,0,255))
			cv2.line(combined, pt1, (pt2[0] + width1, pt2[1]), color=get_color(), thickness=3)

	cv2.imwrite(path, combined)

def detect_correspondences(edgesL, edgesR, max_search_area):
	grayEdgesL = cv2.cvtColor(edgesL, cv2.COLOR_BGR2GRAY) if len(edgesL.shape) == 3 else edgesL
	grayEdgesR = cv2.cvtColor(edgesR, cv2.COLOR_BGR2GRAY) if len(edgesR.shape) == 3 else edgesR

	list_of_correspondences = []
	for row in range(grayEdgesL.shape[0]):
		non_zero_idxL = np.where(grayEdgesL[row] > 0)[0] # Get indices of all the edges that are not equal to 0
		if(not np.size(non_zero_idxL)): 
			continue
		
		# For each pixel in the left image, find the left-most pixel in the right image
		for colL in non_zero_idxL:
			search_kernel = edgesR[row, colL:colL + max_search_area + 1]
			potential_non_zero_idxR = np.where(search_kernel > 0)[0]

			if(not np.size(potential_non_zero_idxR)):
				continue

			colR = potential_non_zero_idxR[0] + colL
			edgesR[row, colR] = 0 # Don't count this pixel anymore
			list_of_correspondences.append([[colL, row], [colR, row]])

	return list_of_correspondences

def projective_reconstruction(corrs, PL, PR):
	world_coords = []

	for idx, corr in enumerate(corrs):
		A = np.zeros((4, 4))
		corrL, corrR = corr

		A[0] = corrL[0] * PL[2, :] - PL[0, :]
		A[1] = corrL[1] * PL[2, :] - PL[1, :]
		A[2] = corrR[0] * PR[2, :] - PR[0, :]
		A[3] = corrR[1] * PR[2, :] - PR[1, :]

		u, s, vh = np.linalg.svd(A.T @ A)
		vh = vh[-1, :].T
		world_coords.append(vh / vh[3]) 

	world_coords = np.reshape(world_coords, (len(corrs), 4))
	return world_coords

def transform_pts(pts, H):
	transformed_pts = []
	for pt in pts:
		X = np.array([pt[0], pt[1], 1.])
		Xprime = np.dot(H, X)
		Xprime = Xprime / Xprime[-1]
		transformed_pts.append((Xprime[0], Xprime[1]))
	return transformed_pts

def projective_stereo_reconstruction():
	imgL = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/imgL.JPG")
	imgR = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/imgR.JPG")

	# Correspondence Points for the Left Image and Right Image
	corrsL = np.int32(np.array([[66, 242],
						[122, 417],
						[549, 714],
						[553, 532],
						[846, 314],
						[799, 496],
						[354, 127],
						[364, 424]]))

	corrsR = np.int32(np.array([[71, 239],
						[117, 420],
						[611, 721],
						[640, 531],
						[883, 308],
						[825, 492],
						[355, 122],
						[423, 430]]))

	# Save the inputs with correspondence points
	draw_circles(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/stereoInputs/imgL.jpg", imgL.copy(), corrsL)
	draw_circles(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/stereoInputs/imgR.jpg", imgR.copy(), corrsR)
	
	# Normalize the correspondences
	normalized_ptsL, tMatL = normalize_points(corrsL)
	normalized_ptsR, tMatR = normalize_points(corrsR)

	# calculate the fundamental matrix
	F = calculate_fundamental_matrix(normalized_ptsL, normalized_ptsR)
	denormalized_F = tMatR.T @ F @ tMatL 
	denormalized_F = denormalized_F / denormalized_F[2, 2] # Make it homogeneous

	# Find the epipoles and the camera projection matrices
	eL_norm, eR_norm, Ex_norm = find_epipoles(denormalized_F)
	PL_norm, PR_norm = compute_projection_matrices(denormalized_F, eR_norm, Ex_norm)

	# Using nonlinear least squares minimization find optimum F
	f = np.ravel(denormalized_F)
	F_optimized = least_squares(cost_function, f, args=[corrsL, corrsR], method="lm").x
	F_optimized = conditionF(np.reshape(F_optimized, (3, 3))) # Make the fundamental matrix have a rank of 2
	F_optimized = F_optimized / F_optimized[2, 2] # Make it homogeneous

	# Get the optimized epipolar lines and camera projection matrices
	eL, eR, E_x = find_epipoles(F_optimized)
	PL, PR = compute_projection_matrices(F_optimized, eR, E_x)

	# ===============================IMAGE RECTIFICATION=====================================================
	HL, HR = compute_homography_rectification(imgL, corrsL, corrsR, eL, eR)

	rectified_left_image = transform(imgL, HL)
	cv2.imwrite(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/rectified/imgL.jpg", rectified_left_image)

	rectified_right_image = transform(imgR, HR)
	cv2.imwrite(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/rectified/imgR.jpg", rectified_right_image)

	# ===============================INTEREST POINT DETECTION=====================================================
	# Detect Edges
	edgesL = detect_edges(rectified_left_image)
	cv2.imwrite(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/edges/imgL.jpg", edgesL)

	edgesR = detect_edges(rectified_right_image)
	cv2.imwrite(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/edges/imgR.jpg", edgesR)

	# Detect Correspondences
	corrsDetected = detect_correspondences(edgesL, edgesR, max_search_area=15)
	corrsFiltered = filter_correspondences(rectified_left_image, rectified_right_image, corrsDetected, kernel_size=15, maximum_corrs=500, mode="ssd")
	draw_correspondences(rectified_left_image, rectified_right_image, rectified_left_image.shape[1], corrsFiltered, path=r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/correspondences/corr.jpg")
	
	# ===============================PROJECTIVE RECONSTRUCTION=====================================================
	
	rectifiedCorrsL = transform_pts(corrsL, HL)
	rectifiedCorrsR = transform_pts(corrsR, HR)
	rectified_coordinates = [[[xL, yL], [xR, yR]] for (xL, yL), (xR, yR) in zip(rectifiedCorrsL, rectifiedCorrsR)]

	draw_circles(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/rectified/imgL_circles.jpg", rectified_left_image.copy(), rectifiedCorrsL)
	draw_circles(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/rectified/imgR_circles.jpg", rectified_right_image.copy(), rectifiedCorrsR)

	unfiltered_world_coords = projective_reconstruction(rectified_coordinates, PL, PR)
	filtered_world_coords = projective_reconstruction(corrsFiltered, PL, PR)

	# ===============================3D RECONSTRUCTION=====================================================
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(unfiltered_world_coords[:, 0], unfiltered_world_coords[:, 1], unfiltered_world_coords[:, 2])
	ax.scatter(filtered_world_coords[:, 0], filtered_world_coords[:, 1], filtered_world_coords[:, 2])

	pair_of_lines = [[0, 1],
			[0, 3],
			[0, 6],
			[1, 2],
			[2, 3],
			[2, 5],
			[3, 4],
			[4, 5],
			[4, 6]] # Points that should form a line

	for pair in pair_of_lines:
		ax.plot([unfiltered_world_coords[pair[0]][0], unfiltered_world_coords[pair[1]][0]], \
						[unfiltered_world_coords[pair[0]][1], unfiltered_world_coords[pair[1]][1]], \
						[unfiltered_world_coords[pair[0]][2], unfiltered_world_coords[pair[1]][2]])

	plt.show()

def get_dmax(gtDisp, mul):
	gtDisp = cv2.cvtColor(gtDisp, cv2.COLOR_BGR2GRAY)
	gtDisp = gtDisp.astype(np.float32) / (16 * mul) # divide by mul because disparity scale is 4
	gtDisp = gtDisp.astype(np.uint8) 

	d_max = np.max(gtDisp)  
	return gtDisp, d_max

def census_transform(imgL, imgR, M, d_max):
	height, width = imgL.shape[:2]

	grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) if len(imgL.shape) == 3 else imgL
	grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) if len(imgR.shape) == 3 else imgR

	halfM = M // 2
	border_size = d_max + halfM
	disparity_map = np.zeros((height, width), dtype=np.uint8)

	for rowL in range(border_size, height - border_size):
		print(f"Row: {rowL + 1} / {height - border_size}")
		for colL in range(width - border_size - 1, border_size - 1, - 1):
			cost = []
			leftWindow = grayL[rowL - halfM: rowL + halfM + 1,
								colL - halfM: colL + halfM + 1]

			binary_left_window = np.ravel(np.where(leftWindow > grayL[rowL, colL], 1, 0))
			for d in range(d_max + 1):
				rowR, colR = rowL, colL - d
				rightWindow = grayR[rowR - halfM: rowR + halfM + 1,
								colR - halfM: colR + halfM + 1]

				binary_right_window = np.ravel(np.where(rightWindow > grayR[rowL, colR], 1, 0))
				cost.append(np.sum(np.bitwise_xor(binary_left_window, binary_right_window)))

			disparity_map[rowL, colL] = np.argmin(cost)

	return disparity_map.astype(np.uint8)

def compute_disparity_accuracy(gtDisp, dispMap, d_max, M, delta=2):
	halfM = M // 2
	border_size = halfM + d_max
	dispMap = dispMap[border_size:dispMap.shape[0]-border_size, border_size:dispMap.shape[1]-border_size]
	gtDisp = gtDisp[border_size:gtDisp.shape[0]-border_size, border_size:gtDisp.shape[1]-border_size]

	error = np.abs(dispMap.astype(np.uint16) - gtDisp.astype(np.uint16)).astype(np.uint8)
	valid_pixels = cv2.countNonZero(gtDisp) # Non-black pixels are valid pixels # 74952
	print("The number of valid pixels: ", valid_pixels)

	accuracy = np.sum(error <= delta)
	percentage_of_accuracy = accuracy / valid_pixels

	error_mask = np.where(error <= delta, 255, 0)
	return percentage_of_accuracy, error_mask

def dense_stereo_matching(M=10):
	imgL = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/im2.ppm")
	dispL = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/disp2.pgm")

	imgR = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/im6.ppm")
	dispR = cv2.imread(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/Task3Images/disp6.pgm")

	gtDisp, d_max = get_dmax(dispL, mul=0.25) # d_max = 13 * mul
	print("d_max: ", d_max)
	
	disparity_map = census_transform(imgL, imgR, M, d_max)
	
	save_disparity_map = cv2.normalize(disparity_map, dst=None, alpha=0, beta=255, norm_type = cv2.NORM_MINMAX).astype(np.uint8) 
	cv2.imwrite(r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/disparityMap/" + str(M) + "size.jpg", save_disparity_map)
	
	percentage_of_accuracy, error_mask = compute_disparity_accuracy(gtDisp, disparity_map, d_max, M)
	print("Percentage of Accuracy: ", percentage_of_accuracy)
	cv2.imwrite("/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/ECE 66100/hw09/error_mask/error_mask" + str(M) + ".jpg", error_mask)


if __name__ == "__main__":
	# TASK 1
	projective_stereo_reconstruction() 

	# TASK 2
	dense_stereo_matching(M=30) 
	dense_stereo_matching(M=45) 
	dense_stereo_matching(M=60) 

	