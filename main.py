import cv2
import numpy as np
import matplotlib.pyplot as plt

from graphics import graphics

# Constant parameters
res = (500, 500)
duration = 2
fps = 20
n_frames = int(duration * fps)
focus = 1
render_dir = 'renders/'

def video_from_imgs(imgs, res, fps, name='test'):
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(
		render_dir + name + '.mp4', 
		fourcc, 
		fps, 
		(res[1], res[0]),
		False)
	for img in imgs:
		data = np.random.randint(0, 256, res, dtype='uint8')
		video.write(img)
	video.release()

# x y and z can be scalars or vectors of the same length
# f(x, y, z) = 0 
def dist_to_surf(x, y, z, tol):
	r = 1
	dist_circ =  (x ** 2 + y ** 2 + (z - r) ** 2) - r ** 2
	delta =  0.0 + (dist_circ <= tol)
	return delta *  dist_circ + (1 - delta) * np.abs(z)

def deriv(x, y, z, tol):
	r = 1
	delta = 0.0 + ((x ** 2 + y ** 2 + (z - r) ** 2) <= r ** 2 + tol)
	n_circ = np.array([2 * x , 2 * y, 2 * z])
	n_circ /= np.linalg.norm(n_circ, axis=0)
	n_plane = np.array([np.zeros_like(x), np.zeros_like(y), np.ones_like(z)])
	return 0.9 * n_circ * delta + (1 - delta) * n_plane * 0.7

def main():
	g = graphics(f=focus, res=res)
	g.translate_camera(np.array([0, 0, 5]))
	g.rotate_camera(rot_axis='y', angle=-45)
	imgs = []
	# Rotate around ball 
	thetas = np.linspace(0, np.pi, n_frames)
	dtheta = thetas[1] - thetas[0]
	pos_og = g.camera_pos.copy()
	for i, theta in enumerate(thetas):
		R = np.array([
				[np.cos(theta), -np.sin(theta), 0],
				[np.sin(theta), np.cos(theta), 0],
				[0, 0, 1]])
		g.camera_pos = R @ pos_og
		g.rotate_camera(rot_axis='z', angle=dtheta * 180 / np.pi)
		g.render_scene(dist_to_surf, deriv, tol=1e-1)
		imgs.append((g.image * 255).astype(np.uint8))
	video_from_imgs(imgs, res, fps)
	
if __name__ == '__main__':
	main()
