import cv2
import time
from detectors.BGSDetector import BGSDetector
from depencies.VideoCapture import VideoCapture

with VideoCapture(num_cam=0, time_of_cap=False, display=True, size=(600, 600)) as cap:
	detector = BGSDetector()
	i = 0
	if cap.is_opened():
		while True:
			ret, frame = cap.read()

			output_frame = detector.run_detection(frame)
			print(output_frame.shape)
			print(detector.get_motion())
			cv2.imshow('frame', output_frame)

			if cv2.waitKey(30) & 0xFF == ord('q'):
				break

			if cap.time_is_passed():
				break

			if i == 10:
				print(cap.get_fps() * 10)
				i = 0
			i += 1
		cv2.destroyAllWindows()
	else:
		print('La caméra na pas été ouverte')
