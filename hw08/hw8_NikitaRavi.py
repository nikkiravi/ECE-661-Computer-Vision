# Name: Nikita Ravi
# Class: ECE 66100
# Homework #8
# Deadline: 11/14/2022

# Import Modules
import cv2
import numpy as np
import math
import os
import re
from scipy.optimize import least_squares

def get_images(path):
	files = [file for file in os.listdir(path)[::-1] if file != ".DS_Store"]
	images = [cv2.imread(path + "/" + img) for img in sorted(files, key=lambda x: int(re.sub(r'\D', '', x)))]
	# print(os.listdir(path)[::-1])
	# print(sorted(os.listdir(path)[::-1], key=lambda x: int(re.sub('\D', '', x)))); quit()
	return images

def display_image(image):
	cv2.imshow("window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	quit()

def draw_lines(image, pts1, pts2, color=(0,0,255), thickness=1):
	copy = image.copy()
	for idx in range(pts1.shape[0]):
		pt1 = (int(pts1[idx, 0]), int(pts1[idx, 1]))
		pt2 = (int(pts2[idx, 0]), int(pts2[idx, 1]))
		
		cv2.line(copy, pt1, pt2, color=color, thickness=thickness)
	return copy

def draw_points(image, pts_list, radius, color=(0,0,255), thickness=-1, text=True):
	copy = image.copy()
	for idx in range(pts_list.shape[0]):
		pt1, pt2 = int(pts_list[idx, 0]), int(pts_list[idx, 1])
		cv2.circle(copy, (pt1, pt2), radius=radius, color=color, thickness=thickness)

		if(text):
			cv2.putText(copy, str(idx + 1), (pt1, pt2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
		
	return copy

def create_mesh_grid(grid_size, num_horizontal_lines, num_vertical_lines):
	x = np.linspace(0, grid_size*(num_vertical_lines-1), num_vertical_lines)
	y = np.linspace(0, grid_size*(num_horizontal_lines-1), num_horizontal_lines)

	x_mesh, y_mesh = np.meshgrid(x,y)
	mesh = np.concatenate([x_mesh.reshape((-1,1)), y_mesh.reshape((-1,1))], axis=1)
	return mesh

def create_blurred_image(image, data):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if(data == "data2"):
		gray = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=1.4) # dataset 2

	return gray

def canny_edge_detection(idx, image, data="data1"):
	# Canny edge detection finds all the edges on the checkerboard pattern
	blur = create_blurred_image(image, data)
	edges = cv2.Canny(blur, 255*1.5, 255)
	cv2.imwrite(os.path.join("hw08/" + data + "/cannyEdges", "image_" + str(idx + 1) + ".jpg"), edges)

	return edges

def get_points_from_hough_lines(lines):
	rho, theta = lines[:, 0], lines[:, 1]
	a, b = np.cos(theta), np.sin(theta)
	x0, y0 = a * rho, b * rho
	pts1 = np.array([x0 + 1000 * (-b), y0 + 1000 * (a)])
	pts2 = np.array([x0 - 1000 * (-b), y0 - 1000 * (a)])
	return pts1.T, pts2.T
		
def hough_lines(edge, image, image_id, ratio=0.5, hough_threshold=60, data="data1"):
	# Draw a straight line along the edges of the checkerboard
	lines = cv2.HoughLines(edge, rho=1, theta=ratio*np.pi / 180, threshold=hough_threshold)
	lines = np.squeeze(lines)
	if(lines is not None):
		pt1, pt2 = get_points_from_hough_lines(lines)
		hough_image = draw_lines(image, pt1, pt2)
		cv2.imwrite(os.path.join("hw08/" + data + "/houghLines", "image_" + str(image_id + 1) + ".jpg"), hough_image)

	return lines

def corner_detection(image, image_id, lines, nms_ratio=0.24, data="data1"):
	copy = image.copy()
	rho, theta = lines[:, 0], lines[:, 1]

	vertical_lines = lines[np.where(np.cos(theta) ** 2 > 0.5)]
	vertical_dist = vertical_lines[:, 0] * np.cos(vertical_lines[:, 1])
	vertical_lines = vertical_lines[np.argsort(vertical_dist)]
	vertical_lines = non_maximum_suppression(vertical_lines, nms_ratio, vertical = True)
	vertical_pts1, vertical_pts2 = get_points_from_hough_lines(vertical_lines)

	horizontal_lines = lines[np.where(np.cos(theta) ** 2 <= 0.5)]
	horizontal_dist = horizontal_lines[:, 0] * np.sin(horizontal_lines[:, 1])	
	horizontal_lines = horizontal_lines[np.argsort(horizontal_dist)]
	horizontal_lines = non_maximum_suppression(horizontal_lines, nms_ratio, vertical = False)
	horizontal_pts1, horizontal_pts2 = get_points_from_hough_lines(horizontal_lines)

	refined_lines_image = draw_lines(copy, vertical_pts1, vertical_pts2)
	refined_lines_image = draw_lines(refined_lines_image, horizontal_pts1, horizontal_pts2)

	cv2.imwrite(os.path.join("hw08/" + data + "/refined", "image_" + str(image_id + 1) + ".jpg"), refined_lines_image)

	horizontal_lines_hc, vertical_lines_hc = get_homogeneous_lines( horizontal_pts1, horizontal_pts2, vertical_pts1, vertical_pts2)
	corners = get_intersections(horizontal_lines_hc, vertical_lines_hc)
	corner_image = draw_points(copy, corners, radius=2)

	cv2.imwrite(os.path.join("hw08/" + data + "/corners", "image_" + str(image_id + 1) + ".jpg"), corner_image)
	return corners

def get_homogeneous_lines(h_pt1, h_pt2, v_pt1, v_pt2):
	v_pt1 = np.append(v_pt1, np.ones((v_pt1.shape[0],1)), axis=1)
	v_pt2 = np.append(v_pt2, np.ones((v_pt2.shape[0],1)), axis=1)

	h_pt1 = np.append(h_pt1, np.ones((h_pt1.shape[0],1)), axis=1)
	h_pt2 = np.append(h_pt2, np.ones((h_pt2.shape[0],1)), axis=1)

	vertical_HC = np.cross(v_pt1, v_pt2)  
	horizontal_HC = np.cross(h_pt1, h_pt2)  

	return horizontal_HC, vertical_HC

def get_intersections(horizontal_lines, vertical_lines):
	corners = []
	for idx in range(horizontal_lines.shape[0]):
		corner = np.cross(horizontal_lines[idx], vertical_lines)
		corner = corner[:,:2] / corner[:,2].reshape((-1,1))
		corners.append(corner)
		
	corners = np.array(corners)
	corners = corners.reshape((-1,2))
	return corners

def non_maximum_suppression(lines, nms_ratio=0.24, vertical=True):
	valid_lines = []
	rho, theta = lines[:, 0], lines[:, 1]
	total_num_line_jumps = 7 if vertical else 9

	dist = rho * np.cos(theta) if vertical else rho * np.sin(theta) # Distance to the vertical/horizontal line
	nms_threshold = nms_ratio * (np.max(dist) - np.min(dist)) / total_num_line_jumps # Taking the average over 7 vertical line jumps

	for idx in range(dist.shape[0] - 1):
		if(dist[idx + 1] - dist[idx] > nms_threshold):
			rho_idx, theta_idx = lines[idx, 0], lines[idx, 1]
			valid_lines.append([rho_idx, theta_idx])
		
		if(idx == dist.shape[0] - 2):
			rho_idx, theta_idx = lines[idx + 1, 0], lines[idx + 1, 1]
			valid_lines.append([rho_idx, theta_idx])

	return np.array(valid_lines)

def compute_homography(domain_coord, range_coord):
	# range_coord = domain_coord * H.T
	# From homework #3
	domain_coord, range_coord = domain_coord.T, range_coord.T
	n = domain_coord.shape[1]
	A = np.zeros((2*n, 8))
	b = np.zeros((2*n, 1))
	H = np.zeros((3,3))

	for idx in range(n):
		A[2*idx] = [domain_coord[0][idx], domain_coord[1][idx], 1, 0, 0, 0, (-domain_coord[0][idx] * range_coord[0][idx]), (-domain_coord[1][idx] * range_coord[0][idx])]
		A[2*idx + 1] = [0, 0, 0, domain_coord[0][idx], domain_coord[1][idx], 1, (-domain_coord[0][idx] * range_coord[1][idx]), (-domain_coord[1][idx] * range_coord[1][idx])]
		b[2*idx] = range_coord[0][idx]
		b[2*idx + 1] = range_coord[1][idx]

	h = np.matmul(np.linalg.pinv(A), b)
	row = 0
	for idx in range(0, len(h), 3):
		spliced = h[idx:idx+3]
		if(len(spliced) == 3):
			H[row] = spliced.T
		else:
			H[row] = np.append(spliced, [1])
		row += 1

	return H

def compute_Vij_component(hi, hj):
	V_ij = np.array([hi[0] * hj[0], 
		hi[0] * hj[1] + hi[1] * hj[0], 
		hi[1] * hj[1], 
		hi[2] * hj[0] + hi[0] * hj[2], 
		hi[2] * hj[1] + hi[1] * hj[2], 
		hi[2] * hj[2]])

	return V_ij.T

def compute_omega_matrix(H_list):
	V = []
	for H in H_list:
		h1 = H[:,0]
		h2 = H[:,1]
		h3 = H[:,2]

		v11 = compute_Vij_component(h1,h1)
		v12 = compute_Vij_component(h1,h2)
		v22 = compute_Vij_component(h2,h2)

		V.append(v12.T)
		V.append((v11-v22).T)
	
	# Compute SVD of v to find b
	u, s, vh = np.linalg.svd(V)
	b = vh[-1] # w11, w12, w22, w13, w23, w33
	w11, w12, w22, w13, w23, w33 = b
	w21 = w12
	w31 = w13
	w32 = w23

	omega = np.array([[w11, w12, w13],
				[w21, w22, w23],
				[w31, w32, w33]])

	return omega

def compute_intrinsic_matrix(omega):
	b = omega.flatten()
	w11, w12, w13, w21, w22, w23, w31, w32, w33 = b

	y0 = (w12 * w13 - w11 * w23) / (w11 * w22 - w12 ** 2)
	Lambda = w33 - (w13 ** 2 + y0 * (w12 * w13 - w11 * w23)) / w11
	alpha_x = math.sqrt(Lambda / w11)
	alpha_y = math.sqrt((Lambda * w11) / (w11 * w22 - w12 ** 2))
	s = -(w12 * alpha_y * alpha_x**2) / Lambda
	x0 = (s * y0) / alpha_y - (w13 * alpha_x ** 2)/Lambda

	K = np.array([[alpha_x, s, x0], [0, alpha_y, y0], [0, 0, 1]])
	return K

def compute_extrinsic_matrix(H_list, K):
	R_list = []
	t_list = []
	for H in H_list:
		r12 = np.dot(np.linalg.inv(K), H)
		scaling_factor = 1 / (np.linalg.norm(r12[:,0]))

		r12 = scaling_factor * r12
		r3 = np.cross(r12[:,0], r12[:,1])

		Q = np.vstack((r12[:,0], r12[:,1], r3)).T
		u, s, vh = np.linalg.svd(Q)

		R = np.matmul(u, vh)
		R_list.append(R)
		t_list.append(r12[:, 2])    
	return R_list, t_list

def refining_calibration_parameters(R_list):
	# Rotation matrix has 9 elements but 3 DoF so we create the Rodriguez representation
	W = []
	for R in R_list:
		r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flatten()
		phi = np.arccos((np.trace(R) - 1) / 2)
		w = phi / (2 * np.sin(phi)) * np.array([r32-r23, r13-r31, r21-r12])
		W.append(w)

	return W

def reconstruct_rotation(w):
	phi = np.linalg.norm(w)
	wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
	R = np.eye(3) + (np.sin(phi) / phi) * wx + ((1 - np.cos(phi)) / (phi ** 2)) * np.matmul(wx, wx)

	return R

def compute_projection_matrix(H, domain):
	domain_homogenous = np.hstack((domain, np.ones((domain.shape[0], 1))))
	P = np.dot(H, domain_homogenous.T).T
	P = P[:, :2] / P[:, 2].reshape((domain.shape[0], 1))
	return P
	
def cost_function(parameters, intersections, mesh):
	projection_pattern = []
	R_list, t_list, K = separate_parameters(parameters)

	for R, t in zip(R_list, t_list):
		Rt = np.hstack((R[:, :2], np.reshape(t, (3,1))))
		H = np.dot(K, Rt)
		projection_pattern.append(compute_projection_matrix(H, mesh))
	
	projection_pattern = np.concatenate(projection_pattern, axis=0)
	intersections = np.concatenate(intersections, axis=0)
	d_geom = projection_pattern - intersections
	return d_geom.flatten()

def combine_parameters(R_list, t_list, K):
	Rt = []
	W = refining_calibration_parameters(R_list)
	for w, t in zip(W, t_list):
		Rt.append(np.append(w, t))

	Rt = np.concatenate(Rt, axis=0)
	K_parameters = np.array([K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]])
	parameters = np.append(K_parameters, Rt)
	return parameters

def separate_parameters(parameters):
	num_Rt = (parameters.shape[0]-5) // 6
	k = parameters[:5]
	K = np.array([[k[0], k[1], k[2]], [0, k[3], k[4]], [0, 0, 1]])

	R_list = [] 
	t_list = []
	extrinsic = parameters[5:]

	for idx in range(num_Rt):
		w = extrinsic[idx*6: idx*6+3]
		R = reconstruct_rotation(w)
		R_list.append(R)
		
		t = extrinsic[idx*6+3: idx*6+6]
		t_list.append(t)

	return R_list, t_list, K

def projection(mesh, intersections, valid_image_idx, images, parameters, before="before", data="data1"):
	R_list, t_list, K = separate_parameters(parameters)
	H_list = []
	for R, t in zip(R_list, t_list):
		Rt = np.hstack((R[:, :2], np.reshape(t, (3,1))))
		H = np.dot(K, Rt)
		H_list.append(H)

	differences = []
	for idx, H in enumerate(H_list):
		projection = compute_projection_matrix(H, mesh)
		diff = intersections[idx] - projection
		differences.append(diff)

		valid_idx = valid_image_idx[idx]
		image = images[idx]
		lm_image = draw_points(image, projection, radius=2)
		cv2.imwrite(os.path.join("hw08/" + data + "/LM", "image_" + str(valid_idx + 1) + "_" + before + ".jpg"), lm_image)

	accuracy, error  = compute_accuracy(np.array(differences).flatten()), compute_error(np.array(differences).flatten())
	return accuracy, error

def compute_accuracy(differences):
	differences = np.array(differences).reshape((-1, 2))
	norm = np.linalg.norm(differences, axis=1)
	avg = np.average(norm)
	var = np.var(norm)
	max_distance = np.max(norm)

	return np.array([avg, var, max_distance])

def compute_error(differences):
	differences = np.array(differences).reshape((-1, 2))
	norm = np.linalg.norm(differences, axis=1)
	N = differences.shape[0] // 80
	errors = []
	for idx in range(N):
		error = {}
		current_distance = norm[idx*80: idx*80+80]
		error["mean"] = np.mean(current_distance)
		error["variance"] = np.var(current_distance)
		error["max_dist"] = np.max(current_distance)
		errors.append(error)

	return errors

def main(images, r=0.5, nms=0.24, hough=60, data="data1"):
	mesh_grid = create_mesh_grid(grid_size=10, num_horizontal_lines=10, num_vertical_lines=8)
	images_with_80_corners = []
	H_list = []
	corners_list = []

	for idx, image in enumerate(images):
		edge = canny_edge_detection(idx, image, data) 
		lines = hough_lines(edge, image, idx, ratio=r, hough_threshold=hough, data=data)
		corners = corner_detection(image, idx, lines, nms_ratio=nms, data=data)

		if(corners.shape[0] == 80):
			images_with_80_corners.append(idx)
			corners_list.append(corners)
			H_list.append(compute_homography(mesh_grid, corners))
	
	print("Number of images with 80 corners: ", len(images_with_80_corners))
	print("These images are with indices: ")
	print([idx + 1 for idx in images_with_80_corners])

	omega = compute_omega_matrix(H_list) # Compute the omega matrix
	K = compute_intrinsic_matrix(omega) # Compute Intrinsic K Matrix
	R_list, t_list = compute_extrinsic_matrix(H_list, K) # Compute the rotation matrix
	parameters = combine_parameters(R_list, t_list, K) # Combine extrinsic and intrinsic parameters
	loss_function = cost_function(parameters, corners_list, mesh_grid) # Create loss function
	
	print("=========================Before LM: Intrinsic Matrix=============================")
	print(K)
	print("=========================Before LM: Rotation Matrix=============================")
	print(R_list[0])
	print("=========================Before LM: Translation Matrix=============================")
	print(t_list[0])

	result = least_squares(cost_function, parameters, method="lm", args=[corners_list, mesh_grid]) # Levenberg-Marquadt
	print("#################################################################################################")

	R_list, t_list, K = separate_parameters(result.x)
	print("=========================After LM: Intrinsic Matrix=============================")
	print(K)
	print("=========================After LM: Rotation Matrix=============================")
	print(R_list[0])
	print("=========================After LM: Translation Matrix=============================")
	print(t_list[0])

	acc_before, error_before = projection(mesh_grid, corners_list, images_with_80_corners, images, parameters, before="before", data=data)
	acc_after, error_after = projection(mesh_grid, corners_list, images_with_80_corners, images, result.x, before="after", data=data)

	print("---------------------------------------Image 1----------------------------------------------")
	print("BEFORE")
	score = error_before[0]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("AFTER")
	score = error_after[0]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("---------------------------------------Image 2----------------------------------------------")
	print("BEFORE")
	score = error_before[1]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("AFTER")
	score = error_after[1]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("---------------------------------------Image 3----------------------------------------------")
	print("BEFORE")
	score = error_before[2]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("AFTER")
	score = error_after[2]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("---------------------------------------Image 4----------------------------------------------")
	print("BEFORE")
	score = error_before[3]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])

	print("AFTER")
	score = error_after[3]
	print("Mean: ", score["mean"])
	print("Variance: ", score["variance"])
	print("Max Distance: ", score["max_dist"])


if __name__ == "__main__":
	image_set1 = get_images(r"hw08/HW8-Files/Dataset1")
	image_set1_copy = [image.copy() for image in image_set1]

	image_set2 = get_images(r"hw08/HW8-Files/Dataset2")
	image_set2_copy = [image.copy() for image in image_set2]
	image_set2_copy = [cv2.resize(image, (480, 640)) for image in image_set2_copy]

	dataset = 1

	if(dataset == 1):
		main(image_set1_copy, r=0.5, nms=0.24, hough=60, data="data1")
	else:
		main(image_set2_copy, r=1, nms=0.24, hough=50, data="data2")