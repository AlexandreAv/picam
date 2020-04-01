import numpy as np


def extract_sub_array(array, kernel_shape):
	x_kernel = kernel_shape[1]
	y_kernel = kernel_shape[0]
	x_array = array.shape[1]
	y_array = array.shape[0]

	if 0 != x_array % x_kernel:
		print("hop hop hop, t'as fait une erreure ici mon p'tit bonhomme")  # TODO ajotuer de la gestion d'erreur
		return

	if 0 != y_array % y_kernel:
		print("hop hop hop, t'as fait une erreure là mon p'tit bonhomme")  # TODO ajouter de la gestion d'erreur
		return

	first_iteration = True
	y = 0
	x = 0

	while True:
		if not first_iteration:
			if x % x_array == 0:
				y += y_kernel
				x = 0

		if y == y_array:
			break

		yield array[y: y + y_kernel, x: x + x_kernel]
		x += x_kernel
		first_iteration = False


def pool_sum_greater_than(x, array, kernel_shape, outputs_shape=None):  # Shape(y,x)
	x_kernel = kernel_shape[1]
	y_kernel = kernel_shape[0]
	x_array = array.shape[1]
	y_array = array.shape[0]
	x_out = int(x_array / x_kernel)
	y_out = int(y_array / y_kernel)

	if outputs_shape is None:
		outputs_shape = (y_out, x_out)

	number_elements = x_out * y_out
	number_elements_outputs_shape = outputs_shape[0] * outputs_shape[1]

	if 0 != x_array % x_kernel:
		print("hop hop hop, t'as fait une erreure ici mon p'tit bonhomme")  # TODO ajotuer de la gestion d'erreur
		return

	if 0 != y_array % y_kernel:
		print("hop hop hop, t'as fait une erreure là mon p'tit bonhomme")  # TODO ajouter de la gestion d'erreur
		return

	if number_elements != number_elements_outputs_shape:
		print('Eh, oh, tu fais de la merde ici')  # TODO Ajouter de la gestion d'erreur
		return

	results = []

	for sub_array in extract_sub_array(array, kernel_shape):
		sum = np.sum(sub_array)

		if sum > x:
			results.append(1)
		else:
			results.append(0)

	results = np.array(results)
	return results.reshape(outputs_shape)
