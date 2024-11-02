import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_excel("dataset.xlsx")

# Seleccionar las variables principales para la segmentación
X = df[['Edad', 'Ingresos_Mensuales']]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Segmento'] = kmeans.fit_predict(X_scaled)

# Analizar los centroides para entender cada segmento
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Edad', 'Ingresos_Mensuales']
)

# Función para asignar nombres basados en las características reales de cada segmento
def assign_segment_names(centroids):
    segment_names = {}
    for i in range(len(centroids)):
        edad = centroids.iloc[i]['Edad']
        ingresos = centroids.iloc[i]['Ingresos_Mensuales']
        
        # Determinar categoría de edad
        if edad < 35:
            edad_cat = 'Jóvenes'
        elif edad < 50:
            edad_cat = 'Adultos'
        else:
            edad_cat = 'Adultos mayores'
            
        # Determinar categoría de ingresos
        if ingresos < df['Ingresos_Mensuales'].quantile(0.33):
            ing_cat = 'bajos ingresos'
        elif ingresos < df['Ingresos_Mensuales'].quantile(0.66):
            ing_cat = 'ingresos medios'
        else:
            ing_cat = 'altos ingresos'
            
        segment_names[i] = f'{edad_cat} con {ing_cat}'
    
    return segment_names

# Asignar nombres basados en las características reales
segment_names = assign_segment_names(centroids)

# Aplicar los nombres descriptivos
df['Nombre_Segmento'] = df['Segmento'].map(segment_names)

# Crear la visualización
plt.figure(figsize=(12, 8))

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for i, segment in enumerate(df['Segmento'].unique()):
    mask = df['Segmento'] == segment
    plt.scatter(
        df[mask]['Edad'],
        df[mask]['Ingresos_Mensuales'],
        c=colors[i],
        label=segment_names[segment],
        alpha=0.6,
        s=100
    )

plt.title('Segmentación de Clientes del Restaurante', fontsize=14, pad=20)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Ingresos Mensuales', fontsize=12)
plt.legend(title='Segmentos de Clientes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Mostrar características de cada segmento
print("\nCaracterísticas de los Segmentos:")
for segment in df['Segmento'].unique():
    segment_data = df[df['Segmento'] == segment]
    print(f"\n{segment_names[segment]}:")
    print(f"- Número de clientes: {len(segment_data)}")
    print(f"- Rango de edad: {int(segment_data['Edad'].min())} - {int(segment_data['Edad'].max())} años")
    print(f"- Rango de ingresos: ${int(segment_data['Ingresos_Mensuales'].min())} - ${int(segment_data['Ingresos_Mensuales'].max())}")
    print(f"- Preferencia de menú más común: {segment_data['Preferencia_Menu'].mode().iloc[0]}")
    print(f"- Frecuencia de visitas más común: {segment_data['Frecuencia_Visitas'].mode().iloc[0]}")