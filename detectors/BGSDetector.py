import cv2
import numpy as np
from depencies.functions import pool_sum_greater_than


class BGSDetector:
	"""
			Classe de détection charger d'utiliser les modèles BGS OpenCv.
			Cette classe va permettre de rendre l'utilisation de BGS plus facile

			:param
			@algorithm: algorithm est une chaine de caractère chargé d'indiquer le type du modèle à charger
			@motion_units: motion_units est un nombre charger d'indiquer le nombre d'unité de mouvement nécessaire pour renvoyer True à la question is there motion ?
		"""
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
		"""
		Fonction chargé d'appliquer une détection sur l'image

		:param frame: image au temps t+1
		:return: Renvoie un masque de la soustraction de frame avec l'image précédent ( Background Subtraction )
		"""

		self.frame = self.back_sub.apply(frame)
		return self.frame

	def get_motion(self):
		return self.is_there_motion(self.frame)

	def is_there_motion(self, frame):
		"""
		Fonction chargée d'interpréter le masque obtenu par le BGS pour détecter si il y a un mouvement

		:param frame: mask du BGS
		:return: True ou False, True = mouvement
		"""
		fg_mask = (frame >= 240) * np.ones(frame.shape)  # masque binaire
		fg_mask = pool_sum_greater_than(x=32, array=fg_mask, kernel_shape=(8, 8)) # SumPool 1
		fg_mask = pool_sum_greater_than(x=8, array=fg_mask, kernel_shape=(4, 4)) # SumPool 2

		number_motion_units = np.sum(fg_mask)
		if number_motion_units > self.motion_units:
			return True
		else:
			return False
