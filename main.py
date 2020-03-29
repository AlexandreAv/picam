import sys
import tensorflow as tf
import numpy as np
import cv2
from time import time
from detectors.SSDDetectors import SSDDetectors
import pdb

# TODO optimiser tflite

def mean(x):
	somme = 0

	for n in x:
		somme += n

	return round(somme / len(x))


class VideoCapture(object):
	def __init__(self, num_cam):
		self.num_cam = num_cam
	def __enter__(self):
		self.cap = cv2.VideoCapture(self.num_cam)
		return self.cap
	def __exit__(self, type, value, traceback):
		self.cap.release()
		print('la caméra a été libéré')



def main(num_cam, path_to_model, path_label, input_size, output_size, display, time_of_cap):
	detector = SSDDetectors(path_to_model, path_label)
	video = cv2.VideoWriter("picam/video.mkv", cv2.VideoWriter_fourcc(*'MJPG'), 25, output_size)
	t0 = time()
	tab_fps = []
	t_fps = time()
	fps = 0
	num_frame = 0


	with VideoCapture(num_cam) as cap:
		while True:
			ret, frame = cap.read()

			# Our operations on the frame come here
			frame = cv2.resize(frame, input_size)
			frame = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.uint8)
			output_frame = detector.run_detection(frame)[0]
			output_frame = cv2.resize(output_frame, output_size)
			output_frame = cv2.putText(output_frame, str(fps), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
			video.write(output_frame)
			# pdb.set_trace()

			# Get fps
			num_frame += 1
			if num_frame == 4:
				fps = round(num_frame / abs(t_fps - time()), 1)
				num_frame = 0
				t_fps = time()
				tab_fps.append(fps)

			# Display the resulting frame
			if display:
				cv2.imshow('frame', output_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			if abs(t0 - time()) > time_of_cap:
				break


		if display:
			cv2.destroyAllWindows()

	video.set(cv2.VIDEOWRITER_PROP_FRAMEBYTES, mean(tab_fps))
	video.release()


if __name__ == '__main__':  # temps en seconde
	with open('picam/detector-config.txt') as file:
		data = []
		for line in file.readlines():
			data.append(line.replace("\n", "").split("#")[0].split('='))

		capture = int(data[0][1])
		path_model = data[1][1]
		path_label = data[2][1]
		inputs_size = (int(data[3][1]), int(data[3][1]))
		outputs_size = (int(data[4][1]), int(data[4][1]))
		show_capture = data[5][1]
		time_cap = int(sys.argv[1]) if len(sys.argv) == 2 else int(data[6][1])

		main(capture, path_model, path_label, inputs_size, outputs_size, show_capture, time_cap)

