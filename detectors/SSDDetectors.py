import tensorflow as tf
from math import ceil
import cv2
import pdb



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.inter_op_parallelism_threads = 8
config.intra_op_parallelism_threads = 8
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
				data.append(line.replace("\n", "").split("\t")) # on supprime la première ligne qui est une légende et non des valeurs utiles

		del data[0]
		return data  # on retourne les valeurs du txt

	def get_id_category(self, id):  #  on récupère la catégorie d'un objet grâce à l'identifiant de celui-ci
		for element in self.list_label:
			# pdb.set_trace()
			if int(element[0]) == id:
				return element[1]

		print("l'id {} est inconnu".format(id))
		return ""

	def run_model_and_clean_output(self, batch_img):
		predictions = self.model(batch_img)

		num_detections = predictions.pop('num_detections').numpy().astype(dtype='int')
		len_predictions = len(num_detections)
		predictions = [{key: value.numpy()[i][0: num_detections[i]] for key, value in predictions.items()}
					   for i in range(len_predictions)]

		return predictions

	def drawn_bounding_boxes(self, batch_img, batch_preds, color=(0, 255, 0)):
		batch_img = batch_img.numpy()

		for i_1 in range(len(batch_img)):
			img = batch_img[i_1]
			img_w, img_h, _ = img.shape
			predictions = batch_preds[i_1]
			score = predictions[1]
			class_id = predictions[2]
			bbox = predictions[0]

			for i_2 in range(len(score)):  # ymin, xmin, ymax, xmax = box
				# pdb.set_trace()
				coord_min = (int(bbox[i_2][3] * img_w), int(bbox[i_2][2] * img_h))
				coord_max = (int(bbox[i_2][1] * img_w), int(bbox[i_2][0] * img_h))
				org = (coord_max[0] + 10, coord_max[1] + 20)
				batch_img[i_1] = cv2.rectangle(img, coord_max, coord_min, color, 2)
				text = "{}% {}".format(ceil(score[i_2] * 100), self.get_id_category(int(class_id[i_2])))
				batch_img[i_1] = cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
											 color)  # TODO présenter correctement les params

		return batch_img

	def run_detection(self, batch_img):
		prediction = self.run_model_and_clean_output(batch_img)
		len_results = len(prediction)

		batch_preds = [list(prediction[i].values()) for i in range(len_results)]

		return self.drawn_bounding_boxes(batch_img, batch_preds)
