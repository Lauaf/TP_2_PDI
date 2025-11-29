"""
Ejercicio 1: Detección y Clasificación de Monedas y Dados
Procesamiento de Imágenes - Universidad Nacional de Rosario

Descripción: Detecta monedas (10c, 50c, 1p) y dados usando OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


RUTA_IMAGEN = "imagenes/monedas.jpg"

# Configuración de monedas: radio, color, valor
MONEDAS = {
    "10c": {"radio": (120, 145), "color": (255, 0, 0), "valor": 0.10, "nombre": "10 centavos"},
    "1p":  {"radio": (145, 170), "color": (0, 255, 0), "valor": 1.0,  "nombre": "1 peso"},
    "50c": {"radio": (170, 500), "color": (0, 0, 255), "valor": 0.50, "nombre": "50 centavos"}
}

# Parámetros de procesamiento
PARAMS = {
    "gaussian_kernel": (7, 7),
    "canny_threshold": (35, 90),
    "morph_kernel": (3, 3),
    "dilate_iterations": 3,
    "hough_circles": {
        "dp": 1.7,
        "minDist": 300,
        "param1": 255,
        "param2": 100,
        "minRadius": 50,
        "maxRadius": 200
    },
    "hough_dice_dots": {
        "dp": 1.5,
        "minDist": 12,
        "param1": 255,
        "param2": 30,
        "minRadius": 30,
        "maxRadius": 40
    }
}


# ============================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================

def cargar_y_preprocesar(ruta):
    """Carga imagen y aplica preprocesamiento básico."""
    img = cv2.imread(ruta)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, PARAMS["gaussian_kernel"], 0)

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    axes[1].imshow(img_blur, cmap='gray')
    axes[1].set_title('Preprocesada (Blur)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    return img_rgb, img_gray, img_blur


# ============================================================
# FUNCIONES DE DETECCIÓN DE BORDES
# ============================================================

def detectar_bordes(img_blur):
    """Aplica Sobel y Canny para detectar bordes."""
    # Sobel (exploración)
    sobel_x = cv2.Sobel(img_blur, -1, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, -1, 0, 1, ksize=3)
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    _, sobel_thresh = cv2.threshold(sobel_combined, 30, 255, cv2.THRESH_BINARY)

    # Canny (principal)
    canny_img = cv2.Canny(img_blur, *PARAMS["canny_threshold"], apertureSize=3, L2gradient=True)

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(sobel_x, cmap='gray')
    axes[0, 0].set_title('Sobel X')
    axes[0, 1].imshow(sobel_y, cmap='gray')
    axes[0, 1].set_title('Sobel Y')
    axes[1, 0].imshow(sobel_combined, cmap='gray')
    axes[1, 0].set_title('Sobel Combinado')
    axes[1, 1].imshow(canny_img, cmap='gray')
    axes[1, 1].set_title('Canny')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return canny_img


def aplicar_morfologia(canny_img):
    """Aplica operaciones morfológicas y dilatación."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS["morph_kernel"])

    apertura = cv2.morphologyEx(canny_img, cv2.MORPH_OPEN, kernel)
    clausura = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
    gradiente = cv2.morphologyEx(canny_img, cv2.MORPH_GRADIENT, kernel)

    # Dilatación final
    canny_dilate = cv2.dilate(clausura, kernel, iterations=PARAMS["dilate_iterations"])

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(apertura, cmap='gray')
    axes[0, 0].set_title('Apertura')
    axes[0, 1].imshow(clausura, cmap='gray')
    axes[0, 1].set_title('Clausura')
    axes[1, 0].imshow(gradiente, cmap='gray')
    axes[1, 0].set_title('Gradiente')
    axes[1, 1].imshow(canny_dilate, cmap='gray')
    axes[1, 1].set_title('Dilatada')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return clausura, canny_dilate


# ============================================================
# FUNCIONES DE DETECCIÓN DE MONEDAS
# ============================================================

def detectar_monedas(img_rgb, canny_dilate):
    """Detecta y clasifica monedas usando Transformada de Hough."""
    # Detectar círculos
    circles = cv2.HoughCircles(
        canny_dilate,
        cv2.HOUGH_GRADIENT,
        **PARAMS["hough_circles"]
    )

    if circles is None:
        print("No se detectaron monedas")
        return {}, []

    circles = np.uint16(np.around(circles))[0]

    # Visualizar círculos detectados
    img_circles = img_rgb.copy()
    for x, y, r in circles:
        cv2.circle(img_circles, (x, y), r, (255, 0, 0), 20)
        cv2.circle(img_circles, (x, y), 2, (0, 0, 255), 10)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_circles)
    plt.title('Círculos Detectados')
    plt.axis('off')
    plt.show()

    # Clasificar por radio
    monedas_detectadas = {k: {**v, "lista": []} for k, v in MONEDAS.items()}

    for x, y, r in circles:
        for tipo, info in monedas_detectadas.items():
            if info["radio"][0] < r <= info["radio"][1]:
                monedas_detectadas[tipo]["lista"].append([int(x), int(y), int(r)])
                break

    # Visualización clasificada
    output = img_rgb.copy()
    result = np.zeros_like(output)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tipo, info in monedas_detectadas.items():
        for x, y, r in info["lista"]:
            cv2.circle(output, (x, y), r, info["color"], 20)

            mask = np.zeros_like(output, np.uint8)
            cv2.circle(mask, (x, y), r, [255]*3, -1)
            mask = mask[:, :, 0]

            color_src = np.full_like(output, info["color"])
            result = cv2.bitwise_and(color_src, output, dst=result, mask=mask)
            cv2.putText(result, info["nombre"], (x, y), font, 2, (255, 255, 255), 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(output)
    axes[0].set_title('Clasificación')
    axes[0].axis('off')
    axes[1].imshow(result)
    axes[1].set_title('Conteo')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Calcular valor total
    print("\n" + "=" * 50)
    print("RESULTADOS DE MONEDAS")
    print("=" * 50)
    total = 0
    for tipo, info in monedas_detectadas.items():
        cantidad = len(info["lista"])
        subtotal = cantidad * info["valor"]
        total += subtotal
        print(f"{tipo:4s}: {cantidad} moneda(s) | ${subtotal:6.2f}")
    print("-" * 50)
    print(f"TOTAL: ${total:.2f}")
    print("=" * 50)

    return monedas_detectadas, circles


# ============================================================
# FUNCIONES DE DETECCIÓN DE DADOS
# ============================================================

def detectar_dados(img_rgb, clausura, monedas_detectadas):
    """Detecta dados y cuenta sus puntos."""
    # Eliminar monedas de la máscara
    mask_dados = clausura.copy()
    for info in monedas_detectadas.values():
        for x, y, r in info["lista"]:
            cv2.circle(mask_dados, (x, y), r + 50, 0, -1)

    # Dilatar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS["morph_kernel"])
    mask_dados = cv2.dilate(mask_dados, kernel, iterations=10)

    plt.figure(figsize=(10, 6))
    plt.imshow(mask_dados, cmap='gray')
    plt.title('Segmentación de Dados')
    plt.axis('off')
    plt.show()

    # Componentes conectados
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity=8)

    dados = []
    img_dados = img_rgb.copy()

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 10000:  # Solo dados completos
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            p1, p2 = (x, y), (x + w, y + h)

            dado_img = mask_dados[y:y+h, x:x+w]
            dados.append({"coord": (p1, p2), "img": dado_img})
            cv2.rectangle(img_dados, p1, p2, (0, 255, 0), 10)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_dados)
    plt.title('Dados Detectados')
    plt.axis('off')
    plt.show()

    # Detectar puntos en cada dado
    result_dados = np.zeros_like(img_rgb)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("\n" + "=" * 50)
    print("RESULTADOS DE DADOS")
    print("=" * 50)

    for idx, dado in enumerate(dados, 1):
        puntos = cv2.HoughCircles(dado["img"], cv2.HOUGH_GRADIENT, **PARAMS["hough_dice_dots"])

        if puntos is not None:
            valor = len(puntos[0])
            dado["valor"] = valor
            print(f"Dado {idx}: Valor = {valor}")

            # Dibujar dado
            p1, p2 = dado["coord"]
            cv2.rectangle(result_dados, p1, p2, (0, 255, 255), -1)
            cv2.putText(result_dados, f"Value {valor}", p1, font, 2, (255, 255, 255), 10)

            # Dibujar puntos
            for x, y, r in puntos[0]:
                x_abs, y_abs = int(x + p1[0]), int(y + p1[1])
                cv2.circle(result_dados, (x_abs, y_abs), int(r), (255, 255, 0), -1)

    print("=" * 50)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_dados)
    plt.title('Dados con Valores')
    plt.axis('off')
    plt.show()

    return dados


# ============================================================
# VISUALIZACIÓN FINAL
# ============================================================

def visualizar_resultado_final(img_rgb, monedas_detectadas, dados):
    """Genera visualización final combinada."""
    combined = np.zeros_like(img_rgb)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dibujar monedas
    for tipo, info in monedas_detectadas.items():
        for x, y, r in info["lista"]:
            cv2.circle(combined, (x, y), r, info["color"], 20)
            cv2.putText(combined, info["nombre"], (x, y), font, 2, (255, 255, 255), 10)

    # Dibujar dados
    for dado in dados:
        p1, p2 = dado["coord"]
        cv2.rectangle(combined, p1, p2, (0, 255, 0), 10)

        if "valor" in dado:
            puntos = cv2.HoughCircles(dado["img"], cv2.HOUGH_GRADIENT, **PARAMS["hough_dice_dots"])
            if puntos is not None:
                for x, y, r in puntos[0]:
                    x_abs, y_abs = int(x + p1[0]), int(y + p1[1])
                    cv2.circle(combined, (x_abs, y_abs), int(r), (255, 255, 0), -1)
                cv2.putText(combined, f"Value {dado['valor']}", p1, font, 2, (255, 255, 255), 10)

    # Superponer con original
    result = cv2.addWeighted(img_rgb, 0.4, combined, 0.6, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(combined)
    axes[0].set_title('Detecciones')
    axes[0].axis('off')
    axes[1].imshow(result)
    axes[1].set_title('Resultado Final')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    """Ejecuta el flujo completo de procesamiento."""
    print("=" * 60)
    print("EJERCICIO 1: Detección de Monedas y Dados")
    print("=" * 60)

    print("\n[1/5] Cargando y preprocesando imagen...")
    img_rgb, img_gray, img_blur = cargar_y_preprocesar(RUTA_IMAGEN)

    print("[2/5] Detectando bordes...")
    canny_img = detectar_bordes(img_blur)

    print("[3/5] Aplicando operaciones morfológicas...")
    clausura, canny_dilate = aplicar_morfologia(canny_img)

    print("[4/5] Detectando y clasificando monedas...")
    monedas_detectadas, circles = detectar_monedas(img_rgb, canny_dilate)

    print("[5/5] Detectando y clasificando dados...")
    dados = detectar_dados(img_rgb, clausura, monedas_detectadas)

    print("\nGenerando visualización final...")
    visualizar_resultado_final(img_rgb, monedas_detectadas, dados)

    print("\n" + "=" * 60)
    print("Procesamiento completado exitosamente")
    print("=" * 60)


if __name__ == "__main__":
    main()
