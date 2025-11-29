"""
Trabajo Práctico 2 - Procesamiento Digital de Imágenes
Universidad Nacional de Rosario

Sistema de ejecución centralizado para todos los ejercicios del TP.
"""

import sys
import os
from pathlib import Path


def print_header():
    """Imprime el encabezado del programa."""
    print("\n" + "=" * 70)
    print(" " * 15 + "TRABAJO PRÁCTICO 2 - PDI")
    print(" " * 10 + "Universidad Nacional de Rosario")
    print("=" * 70)


def print_menu():
    """Muestra el menú de ejercicios disponibles."""
    print("\nEjercicios disponibles:")
    print("-" * 70)
    print("  [1] Ejercicio 1: Detección y Clasificación de Monedas y Dados")
    print("  [0] Salir")
    print("-" * 70)


def ejecutar_ejercicio_1():
    """Ejecuta el Ejercicio 1."""
    try:
        # Agregar el directorio del ejercicio al path
        ejercicio_path = Path(__file__).parent / "Ejercicio-1"
        sys.path.insert(0, str(ejercicio_path))

        # Cambiar al directorio del ejercicio para rutas relativas
        original_dir = os.getcwd()
        os.chdir(ejercicio_path)

        # Importar y ejecutar
        from ejercicio1 import main as ejercicio1_main

        print("\n" + "=" * 70)
        print("INICIANDO EJERCICIO 1")
        print("=" * 70)

        ejercicio1_main()

        # Restaurar directorio original
        os.chdir(original_dir)
        sys.path.pop(0)

    except ImportError as e:
        print(f"\nError: No se pudo importar el ejercicio 1.")
        print(f"Detalles: {e}")
    except FileNotFoundError as e:
        print(f"\nError: No se encontró un archivo necesario.")
        print(f"Detalles: {e}")
    except Exception as e:
        print(f"\nError inesperado al ejecutar el ejercicio 1:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Detalles: {e}")
    finally:
        # Asegurar que volvemos al directorio original
        os.chdir(original_dir)


def ejecutar_ejercicio_2():
    """Ejecuta el Ejercicio 2 (placeholder para futuro desarrollo)."""
    print("\n" + "!" * 70)
    print("Ejercicio 2: Aún no implementado")
    print("!" * 70)


def ejecutar_ejercicio_3():
    """Ejecuta el Ejercicio 3 (placeholder para futuro desarrollo)."""
    print("\n" + "!" * 70)
    print("Ejercicio 3: Aún no implementado")
    print("!" * 70)


def main():
    """Función principal del programa."""
    print_header()

    # Modo interactivo
    if len(sys.argv) == 1:
        while True:
            print_menu()
            try:
                opcion = input("\nSeleccione un ejercicio (0-1): ").strip()

                if opcion == "0":
                    print("\n" + "=" * 70)
                    print("Gracias por usar el sistema. ¡Hasta pronto!")
                    print("=" * 70 + "\n")
                    break
                elif opcion == "1":
                    ejecutar_ejercicio_1()
                    input("\nPresione ENTER para continuar...")
                elif opcion == "2":
                    ejecutar_ejercicio_2()
                    input("\nPresione ENTER para continuar...")
                elif opcion == "3":
                    ejecutar_ejercicio_3()
                    input("\nPresione ENTER para continuar...")
                else:
                    print("\nOpción no válida. Por favor, seleccione un número del menú.")

            except KeyboardInterrupt:
                print("\n\nInterrumpido por el usuario.")
                print("=" * 70 + "\n")
                break
            except EOFError:
                print("\n\nFin de entrada detectado.")
                break

    # Modo por línea de comandos
    else:
        if len(sys.argv) != 2:
            print("\nUso: python main.py [numero_ejercicio]")
            print("Ejemplo: python main.py 1")
            print("\nO ejecute sin argumentos para modo interactivo:")
            print("python main.py")
            sys.exit(1)

        ejercicio = sys.argv[1]

        if ejercicio == "1":
            ejecutar_ejercicio_1()
        elif ejercicio == "2":
            ejecutar_ejercicio_2()
        elif ejercicio == "3":
            ejecutar_ejercicio_3()
        else:
            print(f"\nError: Ejercicio '{ejercicio}' no reconocido.")
            print("Ejercicios disponibles: 1, 2, 3")
            sys.exit(1)


if __name__ == "__main__":
    main()
