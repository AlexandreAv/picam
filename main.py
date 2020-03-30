import sys
import tensorflow as tf
import numpy as np
import cv2
from time import time
from detectors.SSDDetectors import SSDDetectors
from depencies.VideoCapture import VideoCapture
import pdb


# TODO optimiser tflite

def mean(x):
	somme = 0

	for n in x:
		somme += n

	return round(somme / len(x))


def main(num_cam, path_to_model, path_label, input_size, output_size, display, time_of_cap):
	detector = SSDDetectors(path_to_model, path_label)
	video = cv2.VideoWriter("picam/video.mkv", cv2.VideoWriter_fourcc(*'MJPG'), 25, output_size)

	with VideoCapture(num_cam, time_of_cap, display) as cap:
		if cap.is_opened():
			while True:
				ret, frame = cap.read()

				# Our operations the frame come here
				frame = cv2.resize(frame, input_size)


				if is_there_motion:
					frame = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.uint8)
					output_frame = detector.run_detection(frame)[0]

				output_frame = cv2.resize(output_frame, output_size)
				output_frame = cv2.putText(output_frame, cap.get_fps(), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
				video.write(output_frame)
				# pdb.set_trace()


				# Display the resulting frame
				if display:
					cv2.imshow('frame', output_frame)
					if cv2.waitKey(30) & 0xFF == ord('q'):
						break

				if cap.time_is_passed():
					break
		else:
			print('La caméra na pas été ouverte')

	video.set(cv2.VIDEOWRITER_PROP_FRAMEBYTES, cap.get_mean_fps())
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
		show_capture = True if data[5][1] == "True" else False
		time_cap = int(sys.argv[1]) if len(sys.argv) == 2 else int(data[6][1])

		main(capture, path_model, path_label, inputs_size, outputs_size, show_capture, time_cap)
