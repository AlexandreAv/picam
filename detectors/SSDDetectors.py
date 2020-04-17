import tensorflow as tf
from math import ceil
import cv2
import pdb

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.inter_op_parallelism_threads = 4
config.intra_op_parallelism_threads = 4
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class SSDDetectors:
	"""
		Classe de détection charger d'utiliser les modèles tensorflow.
		Cette classe va permettre de rendre l'utilisation des modèles plus facile

		:param
		@path_model: path_model est une chaine de caractère chargé d'indiquer l'emplacement du modèle tensorflow sauvegarder dans un format SavedModel
		@path_label: path_label est une chaine de caractère d'indiquer l'emplacement d'un fichier textes indiquants les catégories possibles
	"""

	def __init__(self, path_model, path_label):
		self.model = tf.saved_model.load(path_model + '/saved_model')
		self.model = self.model.signatures['serving_default']
		self.list_label = self.read_labels(path_label)

	@staticmethod
	def read_labels(path_label):  # on lit le txt puis on renvoit les valeurs grâce à un split
		data = []
		with open(path_label) as file:
			for line in file.readlines():
				data.append(line.replace("\n", "").split(
					"\t"))  # on supprime la première ligne qui est une légende et non des valeurs utiles

		del data[0]
		return data  # on retourne les valeurs du txt

	def get_id_category(self, id):  # on récupère la catégorie d'un objet grâce à l'identifiant de celui-ci
		for element in self.list_label:
			# pdb.set_trace()
			if int(element[0]) == id:
				return element[1]

		print("l'id {} est inconnu".format(id))
		return ""

	def run_model_and_clean_output(self, batch_img):
		"""
		Fonction chargée de faire passer des images dans le modèle tout application des modification pour rendre
		la sortie exploitable

		:param batch_img: Les images à passer dans le modèle
		:return: Résultat des prédictions du modèle
		"""
		predictions = self.model(batch_img)

		num_detections = predictions.pop('num_detections').numpy().astype(dtype='int')
		len_predictions = len(num_detections)
		batch_preds = [{key: value.numpy()[i][0: num_detections[i]] for key, value in predictions.items()}
					   for i in range(len_predictions)]

		return batch_preds

	def drawn_bounding_boxes(self, batch_img, batch_preds, color=(0, 255, 0)):
		"""
		Fonction chargée de créer les boudings boxe avec les prédictions du modèle

		:param batch_img: Image où les boudings boxes seront crées, tableau d'images [None, size, size, 3]
		:param batch_preds: Prédiction du modèle, list de Dict
		:param color: Couleur des boudings boxes, tuple
		:return: Images avec les boudings boxes, tableau d'images [None, size, size, 3]
		"""
		batch_img = batch_img.numpy()

		for i_1 in range(len(batch_img)):
			img = batch_img[i_1]
			img_w, img_h, _ = img.shape
			prediction = batch_preds[i_1]
			class_id = prediction['detection_classes']
			score = prediction['detection_scores']
			bbox = prediction['detection_boxes']

			for i_2 in range(len(score)):  # ymin, xmin, ymax, xmax = box
				coord_min = (int(bbox[i_2][3] * img_w), int(bbox[i_2][2] * img_h))
				coord_max = (int(bbox[i_2][1] * img_w), int(bbox[i_2][0] * img_h))
				org = (coord_max[0] + 10, coord_max[1] + 20)
				batch_img[i_1] = cv2.rectangle(img, coord_max, coord_min, color, 2)
				text = "{}% {}".format(ceil(score[i_2] * 100), self.get_id_category(int(class_id[i_2])))
				batch_img[i_1] = cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

		return batch_img

	def run_detection(self, batch_img):
		"""
		Fonction charger de passer les images dans le modèle, puis d'écrire les boudings boxes issues des prédictions
		du modèle dans les images

		:param batch_img: Images à passer dans le modèle
		:return: Images avec les boudings boxes
		"""
		batch_preds = self.run_model_and_clean_output(batch_img)

		return self.drawn_bounding_boxes(batch_img, batch_preds)
