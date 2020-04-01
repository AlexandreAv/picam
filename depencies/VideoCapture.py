import cv2
from time import time


class VideoCapture:
	def __init__(self, num_cam=0, time_of_cap=True, display=True, size=None):
		self.cap = None
		self.num_cam = num_cam
		self.time_of_cap = time_of_cap
		self.display = display
		self.size = size
		self.t0 = 0
		self.t_fps = 0
		self.fps = 0
		self.num_frame = 0
		self.tab_fps = []

	def read(self):
		return self.cap.read()

	def is_opened(self):
		return self.cap.isOpened()

	def time_is_passed(self):
		if not self.time_of_cap:
			return False

		return abs(self.t0 - time() > self.time_of_cap)

	def get_fps(self):
		self.num_frame += 1
		if self.num_frame == 4:
			self.fps = round(self.num_frame / abs(self.t_fps - time()), 1)
			self.num_frame = 0
			self.t_fps = time()
			self.tab_fps.append(self.fps)

		return self.fps

	def get_mean_fps(self):
		somme = 0

		for fps in self.tab_fps:
			somme += fps

		return round(somme / len(self.tab_fps))

	def __enter__(self):
		self.cap = cv2.VideoCapture(self.num_cam)
		if not self.size:
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

		self.t0 = time()
		return self

	def __exit__(self, type, value, traceback):
		self.cap.release()
		if self.display:
			cv2.destroyAllWindows()
		print('la caméra a été libéré')
