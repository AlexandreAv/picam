import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pipe
import queue
import pdb


class ThreadsManagement:
	def __init__(self, list_functions, sleep_time=0.,
				 max_threads=3):
		self.dict_functions = {index:function for index, function in enumerate(list_functions)} #  [ [function, args, delay, wait_results, put_results_in_queue,] ]
		self.counter_time = 0
		self.sleep_time = sleep_time
		self.max_threads = max_threads
		self.signal_stop = False
		self.process = None
		self.process_executor = None
		self.thread_executor = None
		self.lists_threads = []
		self.list_args = {}
		self.threads_queue = None
		self.parent_conn, self.child_conn = Pipe()

	def start(self):
		self.init()

	def stop(self, wait=True):
		self.signal_stop = True
		self.thread_executor.shutdown(wait)
		self.process_executor.shutdown(wait)

	def init_args_pipe(self): # utiliser pipe pour faire une connexion avec les process
		self.list_args = {}

		for key in self.dict_functions:
			function_data = self.dict_functions[key]
			args = function_data[1]
			self.list_args[key] = args


		self.parent_conn.send(self.list_args)

	def change_args(self, key, args):
		self.list_args[key] = args
		self.parent_conn.send(self.list_args) # envoie à l'autre process

	def update_args(self, key, args):
		self.list_args[key] = args
		self.threads_queue.get()
		self.threads_queue.put(self.list_args)

	def init(self):
		with ProcessPoolExecutor() as executor:
			self.process_executor = executor

			process = executor.submit(self.main_loop)
			process.daemon = True
			self.process = process

# TODO différencier les variables appelés par le threads avec celles de processus

	def main_loop(self):
		self.counter_time = time.time()

		with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
			self.thread_executor = executor
			list_args = self.child_conn.recv()
			while True:
				if not self.child_conn.empty():
					list_args = self.child_conn.recv()

				if not self.stop:
					# Chrono
					current_time = time.time()
					time_passed = current_time - self.counter_time  # en secondes

					for key in self.dict_functions:
						function_data = self.dict_functions[key]
						function, args, delay, wait_results, put_results_in_queue = function_data

						if time_passed >= delay:
							thread = executor.submit(function, args)
							self.lists_threads.append((id, thread))

							if wait_results:
								result = thread.result()

							if put_results_in_queue:
								if not isinstance(result, list):
									self.update_args(key, [result])

					time.sleep(self.sleep_time)
				else:
					break

