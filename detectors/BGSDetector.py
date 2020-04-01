import cv2
import numpy as np
from depencies.functions import pool_sum_greater_than


class BGSDetector:
	def __init__(self, algorithm="MOG2", motion_units=10):
		if algorithm == "MOG2":
			self.back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
			print('Création détecteur MOG2')
		else:
			self.back_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=16, detectShadows=False)
			print('Création détecteur KNN')

		self.motion_units = motion_units
		self.frame = None

	def run_detection(self, frame):
		self.frame = self.back_sub.apply(frame)
		return self.frame

	def get_motion(self):
		return self.is_there_motion(self.frame)

	def is_there_motion(self, frame):
		fg_mask = (frame >= 240) * np.ones(frame.shape)
		fg_mask = pool_sum_greater_than(x=32, array=fg_mask, kernel_shape=(8, 8))
		fg_mask = pool_sum_greater_than(x=8, array=fg_mask, kernel_shape=(4, 4))

		number_motion_units = np.sum(fg_mask)
		if number_motion_units > self.motion_units:
			return True
		else:
			return False
