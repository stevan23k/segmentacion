import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Cargar los datos
df = pd.read_excel("dataset.xlsx")

# Convertir Última_Visita a días transcurridos
df['Ultima_Visita_Dias'] = pd.to_datetime(df['Última_Visita']).apply(lambda x: (datetime.now() - x).days)

# Preparar variables categóricas
le = LabelEncoder()
categorical_columns = ['Género', 'Frecuencia_Visitas', 'Preferencia_Menu', 
                      'Comentarios_Redes', 'Red_Social_Favorita', 'Región']
encoded_columns = {}

for col in categorical_columns:
    encoded_columns[col] = f'{col}_Encoded'
    df[encoded_columns[col]] = le.fit_transform(df[col])

# Seleccionar variables para la segmentación
X = df[[
    'Edad',
    'Ingresos_Mensuales',
    encoded_columns['Frecuencia_Visitas'],
    'Promedio_Calificación',
    'Número_Comentarios',
    'Ultima_Visita_Dias',
    encoded_columns['Comentarios_Redes']
]]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar número óptimo de clusters
inertias = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Visualizar método del codo
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.savefig("elbow_method.png")
plt.close()

# Aplicar K-means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Segmento'] = kmeans.fit_predict(X_scaled)

# Función para analizar segmentos
def analyze_segment(segment_data):
    return {
        'tamaño': len(segment_data),
        'edad_promedio': segment_data['Edad'].mean(),
        'ingresos_promedio': segment_data['Ingresos_Mensuales'].mean(),
        'calificacion_promedio': segment_data['Promedio_Calificación'].mean(),
        'comentarios_promedio': segment_data['Número_Comentarios'].mean(),
        'dias_ultima_visita': segment_data['Ultima_Visita_Dias'].mean(),
        'frecuencia_visitas': segment_data['Frecuencia_Visitas'].mode().iloc[0],
        'menu_preferido': segment_data['Preferencia_Menu'].mode().iloc[0],
        'red_social': segment_data['Red_Social_Favorita'].mode().iloc[0],
        'region_predominante': segment_data['Región'].mode().iloc[0],
        'sentimiento_predominante': segment_data['Comentarios_Redes'].mode().iloc[0],
        'genero_predominante': segment_data['Género'].mode().iloc[0],
        'engagement_rate': segment_data['Número_Comentarios'].mean() / len(segment_data)
    }

# Analizar y nombrar segmentos
segment_analysis = {}
for segment in range(n_clusters):
    segment_data = df[df['Segmento'] == segment]
    analysis = analyze_segment(segment_data)
    
    # Determinar nombre del segmento basado en características
    if analysis['calificacion_promedio'] >= 4.5 and analysis['frecuencia_visitas'] == 'Alta':
        nombre = "Clientes VIP Satisfechos"
    elif analysis['dias_ultima_visita'] > 180:
        nombre = "Clientes Inactivos en Riesgo"
    elif analysis['edad_promedio'] < 30 and analysis['engagement_rate'] > df['Número_Comentarios'].mean() / len(df):
        nombre = "Jóvenes Influyentes"
    elif analysis['ingresos_promedio'] > df['Ingresos_Mensuales'].mean() * 1.2:
        nombre = "Profesionales Alto Valor"
    else:
        nombre = "Clientes Regulares"
    
    segment_analysis[nombre] = analysis

# Visualizaciones
plt.figure(figsize=(15, 10))
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# Gráfico principal: Edad vs Ingresos con tamaño basado en calificación
for i, (nombre, analysis) in enumerate(segment_analysis.items()):
    mask = df['Segmento'] == i
    plt.scatter(
        df[mask]['Edad'],
        df[mask]['Ingresos_Mensuales'],
        c=colors[i],
        label=nombre,
        alpha=0.6,
        s=df[mask]['Promedio_Calificación'] * 50  # Tamaño basado en calificación
    )

plt.title('Segmentación de Clientes: Edad vs Ingresos\n(Tamaño del punto indica la calificación promedio)', fontsize=14)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Ingresos Mensuales ($)', fontsize=12)
plt.legend(title='Segmentos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("segmentacion_clientes.png")
plt.close()

# Generar recomendaciones personalizadas
def generate_recommendations(segment_name, analysis):
    recommendations = {
        "Clientes VIP Satisfechos": [
            f"Crear programa VIP exclusivo con énfasis en {analysis['menu_preferido']}",
            f"Campañas especiales en {analysis['red_social']} para mantener engagement",
            "Eventos exclusivos de degustación y maridaje",
            "Servicio de reserva prioritaria y mesa preferencial"
        ],
        "Clientes Inactivos en Riesgo": [
            f"Campaña de reactivación en {analysis['red_social']} con 30% descuento",
            "Encuesta personalizada para entender razones de ausencia",
            f"Promoción especial en {analysis['menu_preferido']} para primera visita",
            "Email marketing con nuevas opciones de menú"
        ],
        "Jóvenes Influyentes": [
            f"Programa de embajadores en {analysis['red_social']}",
            "Eventos instagrameables y experiencias compartibles",
            f"Promociones especiales para {analysis['menu_preferido']} en grupo",
            "Concursos en redes sociales con premios atractivos"
        ],
        "Profesionales Alto Valor": [
            "Servicios de catering para eventos corporativos",
            f"Menús ejecutivos premium con énfasis en {analysis['menu_preferido']}",
            "Programa de fidelización con beneficios empresariales",
            f"Marketing dirigido en {analysis['red_social']} para profesionales"
        ],
        "Clientes Regulares": [
            "Programa de puntos con beneficios escalables",
            f"Promociones especiales en {analysis['menu_preferido']}",
            "Happy hour extendido entre semana",
            "Newsletter mensual con nuevas opciones de menú"
        ]
    }
    return recommendations.get(segment_name, ["No hay recomendaciones específicas disponibles"])

# Generar informe completo
print("\nANÁLISIS DE SEGMENTACIÓN DE CLIENTES Y RECOMENDACIONES")
print("=" * 80)

for nombre, analysis in segment_analysis.items():
    print(f"\n{nombre.upper()}")
    print("-" * 50)
    print(f"Tamaño del segmento: {analysis['tamaño']} clientes ({analysis['tamaño']/len(df):.1%} del total)")
    print(f"Edad promedio: {analysis['edad_promedio']:.1f} años")
    print(f"Ingresos promedio: ${analysis['ingresos_promedio']:.2f}")
    print(f"Calificación promedio: {analysis['calificacion_promedio']:.1f}/5.0")
    print(f"Frecuencia de visitas más común: {analysis['frecuencia_visitas']}")
    print(f"Preferencia de menú: {analysis['menu_preferido']}")
    print(f"Red social preferida: {analysis['red_social']}")
    print(f"Región predominante: {analysis['region_predominante']}")
    print(f"Género predominante: {analysis['genero_predominante']}")
    print(f"Sentimiento en redes: {analysis['sentimiento_predominante']}")
    print(f"Días desde última visita: {analysis['dias_ultima_visita']:.1f}")
    print(f"Tasa de engagement: {analysis['engagement_rate']:.2f}")
    
    print("\nRecomendaciones:")
    for i, rec in enumerate(generate_recommendations(nombre, analysis), 1):
        print(f"{i}. {rec}")

# Métricas de negocio clave
print("\nMÉTRICAS CLAVE DE NEGOCIO")
print("=" * 80)
print(f"Tasa de clientes inactivos: {len(df[df['Ultima_Visita_Dias'] > 180]) / len(df):.1%}")
print(f"Calificación promedio general: {df['Promedio_Calificación'].mean():.1f}/5.0")
print(f"Proporción de clientes altamente satisfechos: {len(df[df['Promedio_Calificación'] >= 4.5]) / len(df):.1%}")
print(f"Red social más efectiva: {df['Red_Social_Favorita'].mode().iloc[0]}")
print(f"Preferencia de menú más popular: {df['Preferencia_Menu'].mode().iloc[0]}")