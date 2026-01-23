"""
Configuración del Severity Model
================================
Parámetros, pesos y configuración de conexión a BigQuery.
"""

from datetime import datetime, timedelta

# =============================================================================
# CONFIGURACIÓN DE GOOGLE CLOUD
# =============================================================================
GCP_PROJECT = "peya-argentina"
DATASET = "automated_tables_reports"

# =============================================================================
# PARÁMETROS DEL MODELO
# =============================================================================

# País (por ahora solo Argentina)
COUNTRY_CODE = "ar"
COUNTRY_ID = 3  # ID de Argentina en fact_orders

# Ventana de tiempo para análisis histórico
LOOKBACK_DAYS = 90  # Últimos 3 meses

# Los datos de mean_delay empiezan el 2025-11-27
DATA_START_DATE = "2025-11-27"

# Fecha de inicio calculada (no puede ser anterior a DATA_START_DATE)
def get_start_date():
    calculated = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return max(calculated, DATA_START_DATE)

# Fechas a excluir del análisis (días atípicos donde la operación se rompe)
EXCLUDED_DATES = [
    "2025-12-24",  # Nochebuena
    "2025-12-25",  # Navidad
    "2025-12-31",  # Fin de año
    "2026-01-01",  # Año nuevo
]

# =============================================================================
# PESOS DEL STRESS SCORE
# =============================================================================
# stress_score = w1*z_score(fail_rate) + w2*z_score(staffing_affection)
# Mide cuanto se desvian las metricas de su baseline por nivel de mean_delay

STRESS_WEIGHTS = {
    "fail_rate": 0.40,           # 40% - Desviacion del fail rate
    "staffing_affection": 0.60   # 60% - Desviacion del staffing affection
}

# Validar que los pesos sumen 1
assert abs(sum(STRESS_WEIGHTS.values()) - 1.0) < 0.001, "Los pesos deben sumar 1.0"

# =============================================================================
# BUCKETS DE STRESS PARA DEFINIR STAGES
# =============================================================================
# El stress_score se normaliza POR ZONA a escala 0-100
# (cada zona usa su propio min/max, no hay comparacion entre zonas)
#
# METODO DE CALCULO DE UMBRALES:
# 1. Se aplica Savitzky-Golay para suavizar la curva de stress
# 2. Se dividen los registros en buckets segun STAGE_STRESS_PERCENTILES del stress suavizado
# 3. El umbral de cada stage es el DELAY PROMEDIO PONDERADO por ordenes del bucket
#
# Buckets (segun config; los valores son cotas superiores):
#   low_delay:      stress en P0-P{low_delay}   -> delay promedio de ese bucket
#   preventive:     stress en P{low_delay}-P{preventive}  -> delay promedio de ese bucket
#   containment_I:  stress en P{preventive}-P{containment_I} -> delay promedio de ese bucket
#   containment_II: stress en P{containment_I}-P{containment_II} -> delay promedio de ese bucket
#   inoperable:     stress en P{containment_II}-P{inoperable} -> delay promedio de ese bucket
#
# Ventajas: Percentiles equidistantes, curva suavizada, pondera por ordenes

STAGE_STRESS_PERCENTILES = {
    # Los valores representan la cota SUPERIOR de cada stage (percentil de stress)
    # low_delay: 0-20, preventive: 20-30, cont_I: 30-45, cont_II: 45-65, inoperable: 65-100
    "low_delay": 20,
    "preventive": 30,
    "containment_I": 45,
    "containment_II": 65,
    "inoperable": 100
}

# =============================================================================
# SUAVIZADO SAVITZKY-GOLAY
# =============================================================================
# Suaviza la curva de stress para eliminar ruido/dientes
# window_length: cuantos puntos vecinos considera (debe ser impar)
# polyorder: grado del polinomio local (debe ser < window_length)

SAVGOL_WINDOW = 11      # Ventana de suavizado (puntos)
SAVGOL_POLYORDER = 3    # Grado del polinomio

# Minimo de registros para considerar un nivel de delay valido
MIN_RECORDS_PER_DELAY = 30

# Minimo de ordenes para considerar un punto (zona, delay) como valido
# Puntos con menos ordenes se filtran para evitar outliers
MIN_ORDERS_PER_POINT = 30

# =============================================================================
# FILTROS DE DELAY (APLICADOS AL INICIO DEL PIPELINE)
# =============================================================================
# 1) Filtro duro para no considerar delays extremos
# 2) Recorte por percentil ponderado (P99) por zona, usando ordenes como peso

MAX_MEAN_DELAY_MINUTE = 30
WEIGHTED_DELAY_P99 = 0.99

# Delay minimo para truncar la curva de stress
# 0 = usar el pico real de stress sin minimo artificial
MIN_DELAY_FOR_TRUNCATION = 0

# =============================================================================
# TABLAS DE BIGQUERY
# =============================================================================
TABLES = {
    "mean_delay_report": "peya-data-origins-pro.cl_hurrier.mean_delay_report",
    "orders_v2": "peya-data-origins-pro.cl_hurrier.orders_v2",
    "logistics_zones": "peya-argentina.automated_tables_reports.logistics_orders_city_zone_AR",
    "fact_orders": "peya-bi-tools-pro.il_core.fact_orders",
    "staffing_daily": "peya-argentina.automated_tables_reports.sa_daily",
    "mean_delay_config": "peya-data-origins-pro.cl_hurrier.mean_delay_config",
    "sessions_zone": "peya-argentina.automated_tables_reports.perseus_sessions_zone"
}

# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_TABLE = f"{GCP_PROJECT}.{DATASET}.severity_model_config"
OUTPUT_CSV_PREFIX = "zone_thresholds"
