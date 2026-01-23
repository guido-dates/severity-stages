"""
Data Extraction Module
======================
Conexión a BigQuery y extracción de datos para el Severity Model.
"""

from google.cloud import bigquery
import pandas as pd
from datetime import datetime
import sys
sys.path.append('..')
from config.settings import (
    GCP_PROJECT, COUNTRY_CODE, COUNTRY_ID,
    TABLES, get_start_date, LOOKBACK_DAYS, EXCLUDED_DATES,
    MAX_MEAN_DELAY_MINUTE
)


def get_bq_client():
    """Obtiene cliente de BigQuery usando gcloud auth."""
    return bigquery.Client(project=GCP_PROJECT)


def extract_mean_delay_data(client: bigquery.Client, start_date: str = None) -> pd.DataFrame:
    """
    Extrae datos de mean delay por zona y minuto.

    Args:
        client: Cliente de BigQuery
        start_date: Fecha de inicio (YYYY-MM-DD). Si es None, usa LOOKBACK_DAYS.

    Returns:
        DataFrame con mean_delay por zona y timestamp
    """
    if start_date is None:
        start_date = get_start_date()

    max_delay_filter = ""
    if MAX_MEAN_DELAY_MINUTE is not None:
        max_delay_filter = f"AND congestion_delay <= {MAX_MEAN_DELAY_MINUTE}"

    query = f"""
    SELECT
        zone_id,
        DATE(report_timestamp, 'America/Argentina/Buenos_Aires') as report_date,
        EXTRACT(HOUR FROM DATETIME(report_timestamp, 'America/Argentina/Buenos_Aires')) as hour,
        AVG(congestion_delay) as avg_delay,
        APPROX_QUANTILES(congestion_delay, 100)[OFFSET(50)] as delay_p50,
        APPROX_QUANTILES(congestion_delay, 100)[OFFSET(75)] as delay_p75,
        APPROX_QUANTILES(congestion_delay, 100)[OFFSET(90)] as delay_p90,
        APPROX_QUANTILES(congestion_delay, 100)[OFFSET(95)] as delay_p95,
        APPROX_QUANTILES(congestion_delay, 100)[OFFSET(99)] as delay_p99,
        MAX(congestion_delay) as delay_max,
        COUNT(*) as records_count
    FROM `{TABLES['mean_delay_report']}`
    WHERE LOWER(country_code) = '{COUNTRY_CODE}'
        AND report_timestamp >= '{start_date}'
        AND congestion_delay >= 0
        {max_delay_filter}
    GROUP BY zone_id, report_date, hour
    """

    print(f"Extrayendo mean_delay desde {start_date}...")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} registros extraídos")
    return df


def extract_zone_info(client: bigquery.Client, start_date: str = None) -> pd.DataFrame:
    """
    Extrae información de zonas (zone_id, zone_name, city_id, city_name).

    Args:
        client: Cliente de BigQuery
        start_date: Fecha de inicio

    Returns:
        DataFrame con info de zonas únicas
    """
    if start_date is None:
        start_date = get_start_date()

    query = f"""
    SELECT DISTINCT
        zone_id,
        zone_name,
        city_id,
        city_name
    FROM `{TABLES['logistics_zones']}`
    WHERE created_date >= '{start_date}'
        AND zone_id IS NOT NULL
    """

    print("Extrayendo información de zonas...")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} zonas encontradas")
    return df


def extract_fail_rate_data(client: bigquery.Client, start_date: str = None) -> pd.DataFrame:
    """
    Extrae datos de fail rate (órdenes completadas vs canceladas) por zona.

    Args:
        client: Cliente de BigQuery
        start_date: Fecha de inicio

    Returns:
        DataFrame con fail_rate por zona y fecha
    """
    if start_date is None:
        start_date = get_start_date()

    query = f"""
    WITH orders_with_zone AS (
        SELECT
            fo.order_id,
            fo.registered_date,
            fo.order_status,
            z.zone_id
        FROM `{TABLES['fact_orders']}` AS fo
        LEFT JOIN `{TABLES['logistics_zones']}` AS z
            ON CAST(fo.order_id AS STRING) = z.order_code
            AND z.created_date >= '{start_date}'
        WHERE fo.registered_date >= '{start_date}'
            AND fo.country_id = {COUNTRY_ID}
            AND z.zone_id IS NOT NULL
    )
    SELECT
        zone_id,
        registered_date,
        COUNT(DISTINCT CASE WHEN order_status = 'CONFIRMED' THEN order_id END) AS completed_orders,
        COUNT(DISTINCT CASE WHEN order_status = 'REJECTED' THEN order_id END) AS cancelled_orders,
        COUNT(DISTINCT order_id) AS total_orders,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN order_status = 'REJECTED' THEN order_id END),
            COUNT(DISTINCT order_id)
        ) * 100 AS fail_rate
    FROM orders_with_zone
    GROUP BY zone_id, registered_date
    """

    print(f"Extrayendo fail_rate desde {start_date}...")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} registros extraídos")
    return df


def extract_staffing_affection_data(client: bigquery.Client, start_date: str = None) -> pd.DataFrame:
    """
    Extrae datos de staffing affection por zona.

    Args:
        client: Cliente de BigQuery
        start_date: Fecha de inicio

    Returns:
        DataFrame con staffing_affection por zona y fecha
    """
    if start_date is None:
        start_date = get_start_date()

    query = f"""
    WITH orders_staffing AS (
        SELECT
            z.zone_id,
            z.created_date,
            z.order_code,
            CASE WHEN s.staffing_over_10 > 0 THEN 1 ELSE 0 END as has_staffing_affection
        FROM `{TABLES['logistics_zones']}` AS z
        LEFT JOIN `{TABLES['staffing_daily']}` AS s
            ON z.order_code = s.order_code
            AND s.created_date >= '{start_date}'
        WHERE z.created_date >= '{start_date}'
            AND z.zone_id IS NOT NULL
    )
    SELECT
        zone_id,
        created_date,
        COUNT(DISTINCT order_code) AS total_orders,
        SUM(has_staffing_affection) AS staffing_affected_orders,
        SAFE_DIVIDE(SUM(has_staffing_affection), COUNT(DISTINCT order_code)) * 100 AS staffing_affection_rate
    FROM orders_staffing
    GROUP BY zone_id, created_date
    """

    print(f"Extrayendo staffing_affection desde {start_date}...")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} registros extraídos")
    return df


def extract_delay_stress_data(client: bigquery.Client, start_date: str | None = None) -> pd.DataFrame:
    """
    Extrae datos de stress (fail_rate, staffing_affection) por zona y nivel de mean_delay.

    Esta es la query principal para el modelo de umbrales.
    Agrupa por (zone_id, mean_delay_minute) para ver como se comportan
    las metricas de stress en cada nivel de delay.

    IMPORTANTE: El join se hace por MINUTO (no por día):
    - mean_delay_report.report_timestamp se convierte de UTC a Argentina
    - Se trunca a minuto y se joinea con fact_orders.registered_at (ya en hora local)
    - Esto asigna el FR/SA correcto al minuto específico de operación

    Args:
        client: Cliente de BigQuery
        start_date: Fecha de inicio

    Returns:
        DataFrame con columnas: zone_id, mean_delay_minute, avg_fail_rate,
        avg_staffing_affection, records_count, total_orders_matched
    """
    if start_date is None:
        start_date = get_start_date()

    # Construir filtro de fechas excluidas
    excluded_dates_str = ", ".join([f"'{d}'" for d in EXCLUDED_DATES])
    max_delay_raw_filter = ""
    max_delay_final_filter = ""
    if MAX_MEAN_DELAY_MINUTE is not None:
        max_delay_raw_filter = f"AND ROUND(congestion_delay) <= {MAX_MEAN_DELAY_MINUTE}"
        max_delay_final_filter = f"AND mean_delay_minute <= {MAX_MEAN_DELAY_MINUTE}"

    query = f"""
    WITH delay_data AS (
        SELECT
            zone_id,
            -- Convertir UTC a Argentina y truncar a minuto
            TIMESTAMP_TRUNC(
                TIMESTAMP(DATETIME(report_timestamp, 'America/Argentina/Buenos_Aires')),
                MINUTE
            ) as report_minute_ar,
            DATE(report_timestamp, 'America/Argentina/Buenos_Aires') as report_date,
            ROUND(congestion_delay) as mean_delay_minute
        FROM `{TABLES['mean_delay_report']}`
        WHERE LOWER(country_code) = '{COUNTRY_CODE}'
            AND report_timestamp >= '{start_date}'
            AND congestion_delay >= 0
            {max_delay_raw_filter}
            AND DATE(report_timestamp, 'America/Argentina/Buenos_Aires') NOT IN ({excluded_dates_str})
    ),
    orders_with_metrics AS (
        SELECT
            z.zone_id,
            -- Truncar registered_at a minuto (ya está en hora local Argentina)
            TIMESTAMP_TRUNC(fo.registered_at, MINUTE) as order_minute,
            fo.order_id,
            fo.order_status,
            fo.fail_rate_owner,
            CASE WHEN s.staffing_over_10 > 0 THEN 1 ELSE 0 END as has_staffing_affection
        FROM `{TABLES['fact_orders']}` AS fo
        INNER JOIN `{TABLES['logistics_zones']}` AS z
            ON CAST(fo.order_id AS STRING) = z.order_code
        LEFT JOIN `{TABLES['staffing_daily']}` AS s
            ON CAST(fo.order_id AS STRING) = s.order_code
        WHERE fo.registered_date >= '{start_date}'
            AND fo.registered_at >= TIMESTAMP('{start_date}')
            AND fo.country_id = {COUNTRY_ID}
            AND z.zone_id IS NOT NULL
    ),
    metrics_by_minute AS (
        SELECT
            zone_id,
            order_minute,
            -- PYFR: solo cuenta REJECTED cuando fail_rate_owner es PedidosYa o Rider
            SAFE_DIVIDE(
                COUNT(DISTINCT CASE
                    WHEN order_status = 'REJECTED' AND fail_rate_owner IN ('PedidosYa', 'Rider')
                    THEN order_id
                END),
                COUNT(DISTINCT order_id)
            ) * 100 AS py_fail_rate,
            SAFE_DIVIDE(
                SUM(has_staffing_affection),
                COUNT(DISTINCT order_id)
            ) * 100 AS staffing_affection_rate,
            COUNT(DISTINCT order_id) as order_count,
            COUNT(DISTINCT CASE WHEN order_status = 'CONFIRMED' THEN order_id END) as completed_orders
        FROM orders_with_metrics
        GROUP BY zone_id, order_minute
    ),
    combined AS (
        SELECT
            d.zone_id,
            d.mean_delay_minute,
            d.report_date,
            m.py_fail_rate,
            m.staffing_affection_rate,
            m.order_count,
            m.completed_orders
        FROM delay_data d
        INNER JOIN metrics_by_minute m
            ON d.zone_id = m.zone_id
            AND d.report_minute_ar = m.order_minute
    )
    SELECT
        zone_id,
        mean_delay_minute,
        AVG(py_fail_rate) as avg_fail_rate,
        AVG(staffing_affection_rate) as avg_staffing_affection,
        COUNT(*) as records_count,
        SUM(order_count) as total_orders_matched
    FROM combined
    WHERE mean_delay_minute >= 0
        {max_delay_final_filter}
        AND completed_orders >= 1
    GROUP BY zone_id, mean_delay_minute
    ORDER BY zone_id, mean_delay_minute
    """

    print(f"Extrayendo datos de stress por nivel de delay desde {start_date}...")
    print(f"  -> Excluyendo fechas: {', '.join(EXCLUDED_DATES)}")
    print(f"  -> Join por MINUTO (timezone UTC -> Argentina)")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} registros (zona, delay_minute) extraídos")
    return df


def extract_current_config(client: bigquery.Client) -> pd.DataFrame:
    """
    Extrae la configuración actual de stages por city_id.

    Args:
        client: Cliente de BigQuery

    Returns:
        DataFrame con configuración actual
    """
    query = f"""
    SELECT
        city_id,
        low_delay,
        preventive,
        containment_I,
        containment_II,
        inoperable,
        config_timestamp
    FROM `{TABLES['mean_delay_config']}`
    WHERE LOWER(country_code) = '{COUNTRY_CODE}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY config_timestamp DESC) = 1
    """

    print("Extrayendo configuración actual de stages...")
    df = client.query(query).to_dataframe()
    print(f"  -> {len(df):,} ciudades con configuración")
    return df


def extract_all_data(start_date: str = None) -> dict:
    """
    Extrae todos los datasets necesarios para el modelo.

    Args:
        start_date: Fecha de inicio opcional

    Returns:
        Diccionario con todos los DataFrames
    """
    client = get_bq_client()

    print("=" * 60)
    print("EXTRACCIÓN DE DATOS - SEVERITY MODEL")
    print("=" * 60)
    print(f"País: {COUNTRY_CODE.upper()}")
    print(f"Período: últimos {LOOKBACK_DAYS} días")
    print("=" * 60)

    data = {
        "delay_stress": extract_delay_stress_data(client, start_date),
        "zone_info": extract_zone_info(client, start_date),
        "current_config": extract_current_config(client)
    }

    print("=" * 60)
    print("EXTRACCIÓN COMPLETADA")
    print("=" * 60)

    return data


if __name__ == "__main__":
    # Test de extracción
    data = extract_all_data()

    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        print(df.head())
