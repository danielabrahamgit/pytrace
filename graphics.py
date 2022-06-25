from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

class graphics:

	"""
	Inputs:
		camera_pos (3d np.array) - position of camera
		camera_norm (3d np.array) - normalized camera normal vector
		sun_norm (3d np.array) - normalized vector pointing at the sun
		f (float) - distance from camera to image plane
		res (2d tuple) - resolution in pixels
	"""
	def __init__(self, f=1, res=(200,200)):
		# Set camera stuff
		self.camera_pos = np.array([5.0, 0.0, 0.0])
		self.sun_norm = np.array([1.0, 1.0, 9.0])
		self.sun_norm = self.sun_norm / np.linalg.norm(self.sun_norm)
		self.plane_x_hat = np.array([0.0, 1.0, 0.0])
		self.plane_y_hat = np.array([0.0, 0.0, 1.0])
		self.plane_n_hat = np.cross(self.plane_x_hat, self.plane_y_hat)

		# Store all inputs
		self.f = f 
		self.res = res
		self.up = np.array([0,0,1])
		self.image = np.zeros(self.res)
	
	def translate_camera(self, translation):
		self.camera_pos += translation
	
	def rotate_camera(self, rot_axis='x', angle=0.0):
		theta = angle * np.pi / 180
		if rot_axis == 'x':
			R = np.array([
				[1, 0, 0],
				[0, np.cos(theta), -np.sin(theta)],
				[0, np.sin(theta), np.cos(theta)]])
		elif rot_axis == 'y':
			R = np.array([
				[np.cos(theta), 0, np.sin(theta)],
				[0, 1, 0],
				[-np.sin(theta), 0, np.cos(theta)]])
		elif rot_axis == 'z':
			R = np.array([
				[np.cos(theta), -np.sin(theta), 0],
				[np.sin(theta), np.cos(theta), 0],
				[0, 0, 1]])
		
		self.plane_x_hat = R @ self.plane_x_hat
		self.plane_y_hat = R @ self.plane_y_hat
		self.plane_n_hat = R @ self.plane_n_hat

	def render_scene(self, dist_to_surf, deriv, tol=1e-2):
		# Find plane center
		plane_center = self.camera_pos - self.plane_n_hat * self.f
		
		# Angular resolution
		alpha_x, alpha_y = 1 / self.res[1], 1 / self.res[0]

		# Construct x y coordinates on image plane
		y = np.arange(self.res[0])
		x = np.arange(self.res[1])
		x, y = np.meshgrid(x, y)
		x = x.flatten() - self.res[1] / 2
		y = self.res[0] / 2 - y.flatten()
		
		# 3D pixel coordinates on plane
		pixel_cords  = alpha_x * self.plane_x_hat[:,np.newaxis] * x[np.newaxis,:]
		pixel_cords += alpha_y * self.plane_y_hat[:,np.newaxis] * y[np.newaxis,:]
		pixel_cords += plane_center[:, np.newaxis]

		# Direction (unit vector) from camera to pixel 
		dirs = pixel_cords - self.camera_pos[:, np.newaxis]
		dirs /= np.linalg.norm(dirs, axis=0)

		# Compute all lines passing from camera to pixels
		ds = np.linspace(0, 10, 100)
		lines = ds[np.newaxis,np.newaxis,:] * dirs[:,:,np.newaxis] + pixel_cords[:,:,np.newaxis]
		
		# Points on the line
		xs = lines[0, :, :]
		ys = lines[1, :, :]
		zs = lines[2, :, :]	

		# Check wherever the distance from a surface is small	
		dists = dist_to_surf(xs, ys, zs, tol)
		inds = np.argwhere(dists < tol)
		xs = xs[inds[:, 0], inds[:, 1]]
		ys = ys[inds[:, 0], inds[:, 1]]
		zs = zs[inds[:, 0], inds[:, 1]]

		# compute normal dot sun vec for pixel values
		norms = deriv(xs, ys, zs, tol)
		pixel_values = self.sun_norm @ norms

		# Convert 2d coordinates to rows, cols
		cs = x[inds[:, 0]] + self.res[1] / 2
		cs = cs.astype(np.int)
		rs = self.res[0] / 2 - y[inds[:, 0]]
		rs = rs.astype(np.int)

		# Filter out duplicate pixels
		unique = rs * self.res[0] + cs
		_, unique_inds = np.unique(unique, return_index=True)
		rs = rs[unique_inds]
		cs = cs[unique_inds]
		pixel_values = pixel_values[unique_inds]

		# Construct image
		self.image[rs, cs] = pixel_values

	def show_scene(self):
		plt.imshow(self.image, cmap='gray')
		plt.show()



	