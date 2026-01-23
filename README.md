# Severity Model - PedidosYa Argentina

Modelo para calcular umbrales de mean_delay por zona logistica. Define en que valor de delay cada zona activa cada stage operacional.

## Problema que resuelve

Hoy los umbrales se definen manualmente. Este modelo los calcula automaticamente basandose en como cada zona se "estresa" (fail_rate y staffing_affection se desvian de su baseline) en cada nivel de mean_delay.

## Output

Tabla con 5 umbrales por zona:

| zone_id | zone_name    | low_delay | preventive | cont_I | cont_II | inoperable |
|---------|--------------|-----------|------------|--------|---------|------------|
| 123     | Centro Norte | 5         | 9          | 13     | 16      | 20         |
| 456     | Palermo      | 4         | 8          | 12     | 15      | 19         |

**Interpretacion**: Zona "Centro Norte" entra en "Preventive" cuando mean_delay >= 9 min.

## Los 5 Stages Operacionales

| Stage | Descripcion |
|-------|-------------|
| Low Delay | Leve congestion, monitoreo activo |
| Preventive | Se activan medidas preventivas |
| Containment I | Primera fase de contencion |
| Containment II | Contencion agresiva (acciones logisticas) |
| Inoperable | Zona critica, maximas restricciones |

## Como funciona el modelo

### 1. Metricas de entrada
- **PYFR (PedidosYa Fail Rate)**: Ordenes rechazadas atribuibles a PedidosYa/Rider
- **SA (Staffing Affection)**: Ordenes afectadas por falta de repartidores

### 2. Stress Score
Combina ambas metricas en un score unico:
```
stress_score = 0.40 * z_score(PYFR) + 0.60 * z_score(SA)
```
Donde z_score = (valor - baseline) / std_baseline

### 3. Pipeline de procesamiento
```
Datos BigQuery
     |
     v
[1] Calcular baselines por zona (promedio ponderado cuando delay <= P25)
     |
     v
[2] Calcular z-scores de PYFR y SA
     |
     v
[3] Combinar en stress_score y normalizar a 0-100 por zona
     |
     v
[4] Aplicar suavizado Savitzky-Golay a la curva de stress
     |
     v
[5] Truncar curva en el maximo de stress (actuar ANTES del pico)
     |
     v
[6] Dividir en 5 buckets por percentiles equidistantes (P0-20, P20-40, etc.)
     |
     v
[7] Calcular delay promedio ponderado de cada bucket = umbral del stage
```

### 4. Suavizado de curvas
El modelo aplica **Savitzky-Golay** (window=11, polyorder=3) para:
- Eliminar ruido y dientes de sierra en la curva
- Obtener una tendencia suave y representativa
- Identificar claramente el minimo y maximo de stress

### 5. Metodo de buckets con truncado
- La curva se **trunca en el pico de stress** (no usamos datos post-pico)
- Se divide en **5 buckets equidistantes** segun percentiles del stress suavizado
- Cada umbral = **delay promedio ponderado por ordenes** del bucket

## Como correr

```powershell
# 1. Activar entorno virtual
.\venv\Scripts\Activate.ps1

# 2. Instalar dependencias (primera vez)
pip install -r requirements.txt

# 3. Autenticarse en GCP
gcloud auth application-default login

# 4. Abrir notebook y ejecutar celdas en orden
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Estructura del proyecto

```
Severity/
├── config/
│   └── settings.py              # Configuracion y parametros del modelo
├── src/
│   ├── data_extraction.py       # Queries a BigQuery
│   ├── feature_engineering.py   # Baselines, stress score y suavizado
│   ├── stage_calculator.py      # Calculo de umbrales por zona
│   └── visualization.py         # Graficos y diagnostico de zonas
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # Notebook principal (16 celdas)
├── docs/
│   └── severity_score_explicacion.txt # Documentacion tecnica detallada
└── output/
    ├── zone_thresholds.csv      # Umbrales exportados
    └── diagnostico_zonas.pdf    # PDF con graficos por zona (123 paginas)
```

## Parametros principales

En `config/settings.py`:

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `STRESS_WEIGHTS` | FR=40%, SA=60% | Peso de cada metrica en el stress |
| `MIN_ORDERS_PER_POINT` | 30 | Minimo de ordenes para punto valido |
| `SAVGOL_WINDOW` | 11 | Ventana de suavizado Savitzky-Golay |
| `SAVGOL_POLYORDER` | 3 | Grado del polinomio de suavizado |
| `LOOKBACK_DAYS` | 90 | Dias de datos historicos |

## Outputs generados

1. **zone_thresholds.csv**: Tabla con umbrales por zona
2. **diagnostico_zonas.pdf**: 123 paginas con:
   - Curva de stress suavizada (puntos originales + linea suavizada)
   - Marcadores de minimo (verde) y maximo (rojo)
   - Lineas verticales de umbrales por stage
   - Distribucion de sesiones reales por stage

## Documentacion tecnica

Para entender en detalle el modelo (metricas, formulas, edge cases):

**→ [docs/severity_score_explicacion.txt](docs/severity_score_explicacion.txt)**

## Tablas de BigQuery utilizadas

| Alias | Tabla |
|-------|-------|
| mean_delay_report | peya-data-origins-pro.cl_hurrier.mean_delay_report |
| fact_orders | peya-bi-tools-pro.il_core.fact_orders |
| logistics_zones | peya-argentina.automated_tables_reports.logistics_orders_city_zone_AR |
| staffing_daily | peya-argentina.automated_tables_reports.sa_daily |
| sessions_zone | peya-argentina.automated_tables_reports.perseus_sessions_zone |
