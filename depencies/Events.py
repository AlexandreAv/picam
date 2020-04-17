import pdb
import numpy as np

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


def apply_IoU_on_dupplicated_elements(batch_valids_preds, batch_duplicated_elements):
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

							if iou > 0.7:
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


class Events:
	def __init__(self, sucessive_frame_to_valid, score_min, IoU_min, time, path_logs):
		self.sucessive_frame_to_valid = sucessive_frame_to_valid
		self.score_min = score_min
		self.IoU_min = IoU_min
		self.time = time
		self.path_logs = path_logs

	def false_positive_check(self, batch_preds):
		# score verification
		batch_valids_preds = score_verification(batch_preds, 0.6)

		# IoU verification [id, class_id, bbox, IsToDelete]
		batch_duplicated_elements = get_duplicate_elements(
			batch_valids_preds)  # [[[id, index...], [id, index...], [[id, index...]]]]

		# IoU verification [id, class_id, bbox]
		batch_filtered_elements = apply_IoU_on_dupplicated_elements(batch_valids_preds, batch_duplicated_elements)
		# [0, 1.0, array([0.30889422, 0.6696231 , 0.9320558 , 0.8320388 ]

		batch_filtered_preds = []

		for filtered_elements in batch_filtered_elements:
			preds = {'detection_boxes': [], 'detection_scores': [], 'detection_classes': []}
			for filtered_element in filtered_elements:
				if filtered_element:
					if not filtered_element[4]:
						preds['detection_boxes'].append(filtered_element[3])
						preds['detection_scores'] .append(filtered_element[1])
						preds['detection_classes'].append(filtered_element[2])

			preds['detection_boxes'] = np.array(preds['detection_boxes'])
			preds['detection_scores'] = np.array(preds['detection_scores'])
			preds['detection_classes'] = np.array(preds['detection_classes'])
			batch_filtered_preds.append(preds)

		return batch_filtered_preds



