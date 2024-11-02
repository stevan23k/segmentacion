import pandas as pd
import random
from datetime import datetime, timedelta

# Valores posibles para cada columna
generos = ["F", "M"]
frecuencias_visitas = ["Alta", "Media", "Baja"]
preferencias_menu = ["Vegano", "Carnes", "Pescados", "Comida_Rápida", "Ensaladas", "Pastas", "Postres", "Bebidas"]
comentarios_redes = ["Positivo", "Neutro", "Negativo"]
redes_sociales = ["Instagram", "Facebook", "Twitter"]
regiones = ["Norte", "Sur", "Este", "Oeste"]

# Generar datos aleatorios para 100 registros
data = {
    "Cliente_ID": [f"{i+1:03}" for i in range(100)],
    "Edad": [random.randint(18, 60) for _ in range(100)],
    "Género": [random.choice(generos) for _ in range(100)],
    "Ingresos_Mensuales": [random.randint(2500, 7000) for _ in range(100)],
    "Frecuencia_Visitas": [random.choice(frecuencias_visitas) for _ in range(100)],
    "Última_Visita": [(datetime.today() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d") for _ in range(100)],
    "Preferencia_Menu": [random.choice(preferencias_menu) for _ in range(100)],
    "Comentarios_Redes": [random.choice(comentarios_redes) for _ in range(100)],
    "Promedio_Calificación": [round(random.uniform(1.0, 5.0), 1) for _ in range(100)],
    "Red_Social_Favorita": [random.choice(redes_sociales) for _ in range(100)],
    "Número_Comentarios": [random.randint(1, 30) for _ in range(100)],
    "Región": [random.choice(regiones) for _ in range(100)]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
df.to_excel("dataset.xlsx", index=False)

print("Archivo 'dataset.xlsx' generado exitosamente.")
