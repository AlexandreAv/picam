import numpy as np


def extract_sub_array(array, kernel_shape):
    """
    Fonction génératrice chargée d'extraire des tableaux en enfant d'un tableaux parent en fonction des itération d'une dimension de kernel passé dans la paramètre kernel shape

    :param array: Tableau parent, va être l'origine des tableaux enfants
    :param kernel_shape: Dimension du kernel permettant de commander le création des tableaux enfants provenant du paramètre array
    :return:
    """

    x_kernel = kernel_shape[1]
    y_kernel = kernel_shape[0]
    x_array = array.shape[1]
    y_array = array.shape[0]

    #  Vérification si les dimentions du kernel sont biens des multiples des dimensions de tableau parrent
    if 0 != x_array % x_kernel:
        print("hop hop hop, t'as fait une erreure ici mon p'tit bonhomme")  # TODO ajotuer de la gestion d'erreur
        return

    if 0 != y_array % y_kernel:
        print("hop hop hop, t'as fait une erreure là mon p'tit bonhomme")  # TODO ajouter de la gestion d'erreur
        return

    # Initialisation des valeurs pour itérer dans le tableau parent, array
    first_iteration = True
    y = 0
    x = 0

    # On parcourt le tableau array de gauche à droite de haut en bas
    while True:
        if not first_iteration:
            if x % x_array == 0:
                y += y_kernel
                x = 0

        if y == y_array:
            break

        yield array[y: y + y_kernel, x: x + x_kernel]  # On renvoit les tableaux enfants grâce à Yield
        x += x_kernel
        first_iteration = False


def pool_sum_greater_than(x, array, kernel_shape, outputs_shape=None):  # Shape(y,x)
    """
    Fonction chargée d'appliquer une Sumpooling sur l'array entrant puis de comparé la valeur obtenu avec un nombre,
    x ici présent et envoyer 1 dans le tableau enfant si la somme du sumpooling est supérieur à x

    :param x: Nombre de comparaison
    :param array: Tableau parent
    :param kernel_shape: Dimension du kernel pour le pooling
    :param outputs_shape: Dimension du tableau enfant sortant
    :return:
    """
    x_kernel = kernel_shape[1]
    y_kernel = kernel_shape[0]
    x_array = array.shape[1]
    y_array = array.shape[0]
    x_out = int(x_array / x_kernel)
    y_out = int(y_array / y_kernel)

    # Valeur par défaut est la division des dimensions du tableaux parent par les dimensions du kernel
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

    # Utilisation de la fonction extract_sub_array pour obtenir des tableaux enfant puis appliquation d'une somme et
    # d'une comparaison sur ce tableau enfant pour créer un nouveau tableau
    for sub_array in extract_sub_array(array, kernel_shape):
        sum = np.sum(sub_array)

        if sum > x:
            results.append(1)
        else:
            results.append(0)

    results = np.array(results)
    return results.reshape(outputs_shape)
