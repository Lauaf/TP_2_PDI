import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from os import path


# Parametros para cada imagen - ajustados a mano despues de varias pruebas
# Cada imagen tiene condiciones distintas (iluminacion, angulo, color de fondo)

# Para detectar LETRAS y NUMEROS de las patentes
PARAMS_LETRAS = [
    {"threshold": 160, "area": [80, 30], "rel_aspect": [1.2, 2.5], "dist": 43},   # img01
    {"threshold": 100, "area": [80, 30], "rel_aspect": [1.2, 2.5], "dist": 43},   # img02
    {"threshold": 120, "area": [180, 5], "rel_aspect": [1, 5], "dist": 43},       # img03
    {"threshold": 160, "area": [80, 30], "rel_aspect": [1.2, 2.5], "dist": 43},   # img04
    {"threshold": 160, "area": [80, 30], "rel_aspect": [1.2, 2.8], "dist": 30},   # img05
    {"threshold": 110, "area": [100, 30], "rel_aspect": [1.2, 2.8], "dist": 30},  # img06
    {"threshold": 110, "area": [100, 10], "rel_aspect": [1.2, 2.8], "dist": 30},  # img07
    {"threshold": 140, "area": [100, 10], "rel_aspect": [1.2, 2.8], "dist": 30},  # img08
    {"threshold": 100, "area": [100, 15], "rel_aspect": [1.2, 2.8], "dist": 30},  # img09
    {"threshold": 130, "area": [100, 30], "rel_aspect": [1.2, 2.5], "dist": 30},  # img10
    {"threshold": 133, "area": [100, 30], "rel_aspect": [1.2, 2.5], "dist": 30},  # img11
    {"threshold": 100, "area": [100, 15], "rel_aspect": [1.8, 2.5], "dist": 30},  # img12
]

# Para detectar la FORMA COMPLETA de la patente
PARAMS_FORMAS = [
    {"threshold": 150, "area": [800, 200], "rel_aspect": [0.1, 0.5]},             # img01
    {"threshold": 100, "area": [1000, 500], "rel_aspect": [0.2, 2.0]},            # img02
    {"threshold": 100, "area": [10000, 100], "rel_aspect": [1, 5]},               # img03 - no anda bien
    {"threshold": 150, "area": [900, 800], "rel_aspect": [0.5, 1.5]},             # img04
    {"threshold": 160, "area": [10000, 2000], "rel_aspect": [0.1, 1.5]},          # img05
    {"threshold": 100, "area": [2000, 1000], "rel_aspect": [0.1, 2.0]},           # img06
    {"threshold": 100, "area": [1500, 500], "rel_aspect": [0.2, 0.5]},            # img07
    {"threshold": 140, "area": [1500, 500], "rel_aspect": [0.2, 1.0]},            # img08
    {"threshold": 100, "area": [1200, 500], "rel_aspect": [0.2, 0.5]},            # img09
    {"threshold": 110, "area": [2000, 1000], "rel_aspect": [0.3, 1.5]},           # img10
    {"threshold": 133, "area": [2000, 500], "rel_aspect": [0.3, 0.5]},            # img11
    {"threshold": 100, "area": [100, 15], "rel_aspect": [1.8, 2.5], "dist": 30},  # img12 - tampoco funciona bien
]


def segmentar_caracteres():
    """
    Segmenta las letras y numeros de las patentes.

    El truco es usar componentes conectados y filtrar por:
    - Area (los caracteres tienen un tamaño similar)
    - Aspect ratio (alto/ancho de cada letra)
    - Distancia entre componentes (las letras estan juntas)
    """

    for i, param in enumerate(PARAMS_LETRAS):
        num_img = i + 1

        # Cargar parametros para esta imagen
        umbral = param["threshold"]
        area_max, area_min = param["area"]
        aspect_min, aspect_max = param["rel_aspect"]
        max_dist = param["dist"]

        # Leer imagen
        img_path = path.join("imagenes", f"img{str(num_img).rjust(2, '0')}.png")
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Umbralizar - esto separa las letras del fondo
        _, img_bin = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)

        # Configurar visualizacion
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        axes[0].imshow(img_bin, cmap="gray")
        axes[0].set_title("Umbralizada")

        # Encontrar componentes conectados
        # Esto agrupa pixeles blancos que estan juntos
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

        # --- Primer filtro: area y aspect ratio ---
        img_filtro1 = np.zeros_like(labels)
        centroids_ok = []  # guardar centroides validos para el siguiente filtro

        for j in range(1, len(stats)):  # empezamos en 1 para saltear el fondo
            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]

            aspect_ratio = h / w

            # Chequear si cumple los rangos
            if (area < area_max and area > area_min and
                aspect_ratio > aspect_min and aspect_ratio < aspect_max):

                centroids_ok.append((centroids[j], j))
                img_filtro1[y:y+h, x:x+w] = labels[y:y+h, x:x+w]

        axes[1].imshow(img_filtro1, cmap="gray")
        axes[1].set_title("Filtro 1: area + aspecto")

        # --- Segundo filtro: distancia entre componentes ---
        # Las letras de una patente estan cerca unas de otras
        # Si un componente esta muy lejos de todo, probablemente no sea parte de la patente
        img_filtro2 = np.zeros_like(labels)

        for k in range(len(centroids_ok)):
            c_i, idx_i = centroids_ok[k]

            # Comparar con los demas centroides
            otros = [t for m, t in enumerate(centroids_ok) if t[1] != idx_i]

            for c_j, idx_j in otros:
                # Calcular distancia euclidiana
                dist = math.sqrt((c_i[0] - c_j[0])**2 + (c_i[1] - c_j[1])**2)

                # Si esta cerca de al menos otro componente, lo incluimos
                if dist < max_dist:
                    x = stats[idx_i, cv2.CC_STAT_LEFT]
                    y = stats[idx_i, cv2.CC_STAT_TOP]
                    w = stats[idx_i, cv2.CC_STAT_WIDTH]
                    h = stats[idx_i, cv2.CC_STAT_HEIGHT]

                    img_filtro2[y:y+h, x:x+w] = labels[y:y+h, x:x+w]
                    break  # ya sabemos que esta cerca de algo

        axes[2].imshow(img_filtro2, cmap="gray")
        axes[2].set_title("Filtro 2: distancia")

        fig.suptitle(f"Imagen {num_img} - Caracteres")
        plt.show(block=True)


def segmentar_formas():
    """
    Intenta segmentar la forma completa de la patente.

    NOTA: Este metodo no funciona muy bien en todas las imagenes
    (por ejemplo img03 y img12 tienen problemas)
    El problema es cuando el color de la patente es muy parecido al del auto
    """

    for i, param in enumerate(PARAMS_FORMAS):
        num_img = i + 1

        umbral = param["threshold"]
        area_max, area_min = param["area"]
        aspect_min, aspect_max = param["rel_aspect"]

        # Cargar imagen
        img_path = path.join("imagenes", f"img{str(num_img).rjust(2, '0')}.png")
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Aplicar threshold
        _, img_bin = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        axes[0].imshow(img_bin, cmap="gray")
        axes[0].set_title('Umbralizada')

        # Componentes conectados con conectividad de 4
        # (probe con 8 pero daba peores resultados)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=4)

        img_result = np.zeros_like(labels)

        # Filtrar por area y aspect ratio
        # La patente completa es mas grande que las letras individuales
        for j in range(1, len(stats)):
            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]

            aspect = h / w

            if (area < area_max and area > area_min and
                aspect > aspect_min and aspect < aspect_max):

                img_result[y:y+h, x:x+w] = labels[y:y+h, x:x+w]

        axes[1].imshow(img_result, cmap="gray")
        axes[1].set_title('Patente detectada')

        fig.suptitle(f"Imagen {num_img} - Forma completa")
        plt.show(block=True)


def main():
    print("=" * 60)
    print("EJERCICIO 2: Segmentacion de Patentes")
    print("=" * 60)
    print("\nEste ejercicio detecta patentes usando componentes conectados")
    print("Hay 12 imagenes de autos con patentes argentinas\n")

    while True:
        print("\nOpciones:")
        print("  [1] Segmentar caracteres (letras y numeros)")
        print("  [2] Segmentar forma completa de la patente")
        print("  [0] Volver al menu principal")

        opcion = input("\nSeleccione una opcion: ").strip()

        if opcion == "1":
            print("\nSegmentando caracteres de las 12 imagenes...")
            print("(Cierre cada ventana para continuar con la siguiente)\n")
            segmentar_caracteres()
            print("\nListo!")

        elif opcion == "2":
            print("\nSegmentando formas completas de las 12 imagenes...")
            print("(Cierre cada ventana para continuar con la siguiente)")
            print("NOTA: img03 e img12 pueden no funcionar bien\n")
            segmentar_formas()
            print("\nListo!")

        elif opcion == "0":
            break
        else:
            print("Opcion invalida")


if __name__ == "__main__":
    main()
