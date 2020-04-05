import cv2
from time import time


class VideoCapture:
	"""
	Class chargée de gérer la lecture d'une caméra, le temps de lecture, le nombre d'fps et la fermeture de la lecture
	de la caméra

	:param num_cam: Numéro de la caméra à lire
	:param time_of_cap: Temps de lecture, si True, le temps de lecture est infini
	:param display: Boolean gérant l'affichage où non d'une fênetre de la lecture en cours
	:param size: Redimensionnement de l'image en sortie de lecture
	"""

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

	def read(self):  # Renvoie la frame de la caméra
		return self.cap.read()

	def is_opened(self):  # Vérifie si la caméra a bien été ouverte
		return self.cap.isOpened()

	def time_is_passed(self):  # Vérfie si le temps de capture a été atteint
		if not self.time_of_cap:
			return False

		return abs(self.t0 - time() > self.time_of_cap)

	def get_fps(self):  # Calcul le nombre d'fps pendant la capture, doit être appelée à chaque tour de boucle pour actualiser les fps
		self.num_frame += 1
		if self.num_frame == 4:
			self.fps = round(self.num_frame / abs(self.t_fps - time()), 1)
			self.num_frame = 0
			self.t_fps = time()
			self.tab_fps.append(self.fps)

		return self.fps

	def get_mean_fps(self):  # Calcul la moyenne des fps lors de la capture
		somme = 0

		for fps in self.tab_fps:
			somme += fps

		return round(somme / len(self.tab_fps))

	def __enter__(self): # Initialise la capture à l'entré du context
		self.cap = cv2.VideoCapture(self.num_cam)
		if not self.size:
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

		self.t0 = time()
		return self

	def __exit__(self, type, value, traceback):  # Ferme la capture à la sortie du context
		self.cap.release()
		if self.display:
			cv2.destroyAllWindows()
		print('la caméra a été libéré')
