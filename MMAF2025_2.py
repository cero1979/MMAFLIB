# Celda 1: Cargar todas las funciones (VERSIÓN DEFINITIVA)
import requests
import time

# (1) Forzar la descarga de la versión más reciente (evita el caché)
timestamp = int(time.time())
url = f"https://raw.githubusercontent.com/cero1979/MMAFLIB/main/MMAF2025_2.py?t={timestamp}"

print(f"Descargando la biblioteca desde: {url}")

try:
    response = requests.get(url)
    response.raise_for_status() 
    
    # (2) Ejecutar el código en el ÁMBITO GLOBAL (globals())
    # ESTO ES LO QUE SOLUCIONA EL 'NameError'
    exec(response.text, globals())
    
    print("✅ Biblioteca cargada. Funciones disponibles.")
    print("Funciones de demo disponibles:")
    print(" - podar_campo_demo()")
    print(" - lanzar_demo_drones()")
    print(" - crear_interfaz_division_sintetica()")
    print(" - lanzar_demo_viga()")

except Exception as e:
    print(f"❌ ERROR al cargar el script: {e}")