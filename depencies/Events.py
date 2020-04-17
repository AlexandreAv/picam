import numpy as np
import cv2
from threading import Timer
from datetime import datetime
import time
from pathlib import Path
import os
import json


class Events:
	"""
		Fonction de gestion des logs et des faux positifs. Gère les faux positifs d'un modèle et organise un système de
		logging avec les résultats d'un modèles. Les loggeurs ( partie de code chargé de faire les logs ) sont générés
		par la classse threads_management et donc exécuté dans un autre processus dans différent threads

		:param score_min: un double indiquant le score minimum recquis d'une prédiction pour qu'elle soit consédirée comme valide
		:param IoU_min: double réprésentant le IoU minimum pour qu'une prédiction soit considérée comme valide
		:param save_logs_delay: int réprésentant le nombre de seconde entre chaque sauvegarde des logs générés par les prédictions du modèle
		:param save_video_delay: int réprésentant le nombre de seconde entre chaque sauvegarde des vidéos générées par les prédictions du modèle ( BGS et SSD)
		:param path_logs: chemin indiquant le fichier de sauvegarde des logs
		:param path_videos: chemin indiquant le fichier de sauvegarde des vidéos générés par les modèles ( BGS et SSD )
		:param videos_output_size: dimension de sortie des videos générés par la class event
	"""

	def __init__(self, score_min, IoU_min, save_logs_delay, save_video_delay,
				 path_logs, path_videos, videos_output_size):
		self.score_min = score_min
		self.IoU_min = IoU_min
		self.save_logs_delay = save_logs_delay
		self.save_video_delay = save_video_delay
		self.path_folder_logs = Path(path_logs)
		self.path_folder_video = Path(path_videos)
		self.path_folder_video_mask = self.path_folder_video / "mask"
		self.path_folder_video_image = self.path_folder_video / "image"
		self.videos_output_size = videos_output_size
		self.batch_events = []
		self.video_image = None
		self.video_mask = None
		self.thread_logs = None
		self.thread_videos = None

		init_folders([self.path_folder_video, self.path_folder_video_mask, self.path_folder_video_image, self.path_folder_logs])
		self.timer_logs()

	def false_positive_check(self, batch_preds):
		"""
		:param batch_preds: les prédictions du modèles de la forme [{'detection_boxes':  np.array([[ymin, xmin, ymax, xmax], ...]), 'detection_scores':  np.array([0.X, ...]), 'detection_classes':  np.array([X.0, ...])},
		{'detection_boxes':  np.array([[ymin, xmin, ymax, xmax],...]), 'detection_scores':  np.array([0.X,...]), 'detection_classes':  np.array([X.0, ...])}]
		:return: batch_preds filtered by events class conditions
		"""

		# score verification
		batch_valids_preds = score_verification(batch_preds, self.score_min)

		# IoU verification [id, class_id, bbox, IsToDelete]
		batch_duplicated_elements = get_duplicate_elements(
			batch_valids_preds)  # [[[id, index...], [id, index...], [[id, index...]]]]

		# IoU verification [id, class_id, bbox]
		batch_filtered_elements = apply_IoU_on_dupplicated_elements(self.IoU_min, batch_valids_preds, batch_duplicated_elements)
		# [0, 1.0, array([0.30889422, 0.6696231 , 0.9320558 , 0.8320388 ]

		batch_filtered_preds = []

		for filtered_elements in batch_filtered_elements:
			preds = {'detection_boxes': [], 'detection_scores': [], 'detection_classes': []}
			for filtered_element in filtered_elements:
				if filtered_element:
					if not filtered_element[4]:
						preds['detection_boxes'].append(filtered_element[3])
						preds['detection_scores'].append(filtered_element[1])
						preds['detection_classes'].append(filtered_element[2])

			preds['detection_boxes'] = np.array(preds['detection_boxes'])
			preds['detection_scores'] = np.array(preds['detection_scores'])
			preds['detection_classes'] = np.array(preds['detection_classes'])
			batch_filtered_preds.append(preds)

		return batch_filtered_preds

	def add_events(self, batch_preds):
		"""
		Fonction chargée de stocker les évènement du modèles dans la variable batch_events

		:param batch_preds:
		:return: None
		"""

		now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

		for preds in batch_preds:
			scores = preds['detection_scores']
			class_id = preds['detection_classes']
			bbox = preds['detection_boxes']
			events = list(zip(scores, class_id, bbox))

			for index_1, event in enumerate(self.batch_events):
				for index_2, preds in enumerate(event[1]):
					score, class_id, bbox = preds
					self.batch_events[index_1][1][index_2] = [str(round(score, 1)), str(int(class_id)),
															  str(list(np.around(bbox, decimals=2)))]

			self.batch_events.append([now, events])

	def add_frame(self, image, mask):
		"""
		Fonction chargée d'ajouter les frames des modèles dans un cv2.VideoWritter en vue de transformer l'ensemble de frame en vidéo

		:param image: Frame issu du modèle SSD
		:param mask: Frame issu du modèle BGS
		:return: None
		"""

		if not self.video_image:
			path_video = str(manage_file_name(self.path_folder_video_image, "mk"))
			self.video_image = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'MJPG'), 25, self.videos_output_size)

		if not self.video_mask:
			path_video = str(manage_file_name(self.path_folder_video_mask))
			self.video_mask = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'MJPG'), 25, self.videos_output_size)

		self.video_image.write(image)
		self.video_mask.write(mask)

	def release_videos(self):
		"""
		Fonction charger d'enrengistrer les vidéos fabriqués par les objets cv2.VideoWriter

		:return:
		"""

		num_frames = self.video_image.get(cv2.CAP_PROP_FRAME_COUNT)
		duration = self.save_video_delay

		fps = round(num_frames / duration, 1)

		self.video_image.set(cv2.CAP_PROP_FRAME_COUNT, fps)
		self.video_mask.set(cv2.CAP_PROP_FRAME_COUNT, fps)
		self.video_image.release()
		self.video_mask.release()

		self.video_image = None
		self.video_mask = None

	def save_logs(self):
		"""
		Fonction charger de sauvegarder les logs de batch_events sous un format json dans un fichier

		:return: None
		"""

		path_logs = manage_file_name(self.path_folder_logs)

		json_data = json.dumps(self.batch_events)

		with open(path_logs, "w") as file: # TODO Erreur Handling
			file.write(json_data)

	def timer_logs(self): # TODO remplacer par les fonctions de la classe threads_managements
		if self.thread_logs:
			if self.thread_logs.isAlive():
				self.thread_logs.join()
		self.save_logs()

		self.thread_logs = Timer(self.save_logs_delay, self.timer_logs)
		self.thread_logs.start()

	def timer_videos(self): # TODO remplacer par les fonctions de la classe threads_managements
		if self.thread_videos:
			if self.thread_videos.isAlive():
				self.thread_videos.join()

		self.release_videos()

		self.thread_videos = Timer(self.save_video_delay, self.timer_videos)
		self.thread_videos.start()


#
#  Fonctions basiques, code obscur
#


def get_duplicated_elements(l):
	seen = {}
	dupes = []

	for x in l:
		if x[1] not in seen:
			seen[x[1]] = 1
		else:
			if seen[x[1]] == 1:
				dupes.append(x)
			seen[x[1]] += 1

	return dupes


def score_verification(batch_preds, score_min):
	batch_valids_preds = []

	for prediction in batch_preds:
		valid_preds = []
		tab_class_id = prediction['detection_classes']
		scores = prediction['detection_scores']
		bbox = prediction['detection_boxes']

		for index, score in enumerate(scores):
			if score > score_min:
				valid_preds.append([index, score, tab_class_id[index], bbox[index], False])

		batch_valids_preds.append(valid_preds)

	return batch_valids_preds


def get_duplicate_elements(batch_valids_preds):
	batch_elements_duplicated = []

	for valid_preds in batch_valids_preds:
		# recup class id
		tab_class_id = []

		for pred in valid_preds:
			tab_class_id.append([pred[0], pred[2]])  # id + class id

		tab_class_id_duplicated_elements = get_duplicated_elements(tab_class_id)

		elements_duplicated = []

		if tab_class_id_duplicated_elements:  # si il y a des éléments dupliqués

			for class_id_duplicated in tab_class_id_duplicated_elements:
				# recup id elements duplicate
				tab_index_duplicated = []

				for class_id in tab_class_id:  # TODO: mieux choisir le nom de la variable
					if class_id[1] == class_id_duplicated[0]:
						tab_index_duplicated.append(class_id[0])
				elements_duplicated.append([class_id_duplicated[0], tab_index_duplicated])  # [class_id, [index, ...]]

		else:
			elements_duplicated.append(None)

		batch_elements_duplicated.append(elements_duplicated)

	return batch_elements_duplicated


def get_pred_in_valids_preds(list_id, valids_preds):
	for pred in valids_preds:
		id_element = pred[0]

		if id_element in list_id:
			yield pred


def compute_bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def compute_area(bbox_1):  # ymin, xmin, ymax, xmax = box
	y_min, x_min, y_max, x_max = bbox_1

	return (x_max - x_min) * (y_max - y_min)


def apply_IoU_on_dupplicated_elements(IoU_min, batch_valids_preds, batch_duplicated_elements):
	for index, valids_preds in enumerate(batch_valids_preds):
		for duplicated_elements in batch_duplicated_elements[index]:
			if duplicated_elements:
				class_id, id_elements = duplicated_elements

				for pred_1 in get_pred_in_valids_preds(id_elements, valids_preds):
					for pred_2 in get_pred_in_valids_preds(id_elements, valids_preds):
						if pred_1 is not pred_2:
							bbox_1 = pred_1[3]
							bbox_2 = pred_2[3]

							iou = compute_bb_intersection_over_union(bbox_1, bbox_2)

							if iou > IoU_min:
								if compute_area(bbox_1) > compute_area(bbox_2):
									if not pred_2[4]:
										pred_2[4] = False
								else:
									if not pred_1[4]:
										pred_1[4] = False

	batch_valids_IoU_preds = []

	for valids_preds in batch_valids_preds:
		if valids_preds:
			valids_IoU_preds = []
			for pred in valids_preds:
				if not pred[4]:
					valids_IoU_preds.append(pred)

			batch_valids_IoU_preds.append(valids_IoU_preds)
		else:
			batch_valids_IoU_preds.append([None])

	return batch_valids_IoU_preds


def manage_file_name(path_folder_logs, extension="txt"):
	logs_name = f"{get_current_date()}_1.{extension}"
	path_logs_file = path_folder_logs / logs_name

	if os.path.exists(path_logs_file):
		index_logs = int(path_logs_file[-5])
		path_logs_file.replace(f"_{index_logs}", f"_{index_logs + 1}")

	return path_logs_file


def get_current_date():
	return datetime.now().strftime('%d:%m:%Y-%H:%M:%S')


def get_current_time():
	return time.time()

def init_folders(folders):
	for folder in folders:
		if not os.path.exists(folder):
			os.mkdir(folder)
