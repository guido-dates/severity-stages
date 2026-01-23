"""
Export Module
=============
Exportación de resultados a CSV y BigQuery.
"""

import pandas as pd
from google.cloud import bigquery
from datetime import datetime
import os
import sys
sys.path.append('..')
from config.settings import GCP_PROJECT, OUTPUT_TABLE, OUTPUT_CSV_PREFIX


def export_to_csv(df: pd.DataFrame, output_dir: str = "output") -> str:
    """
    Exporta DataFrame a CSV.

    Args:
        df: DataFrame a exportar
        output_dir: Directorio de salida

    Returns:
        Path del archivo creado
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Nombre con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_CSV_PREFIX}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Agregar timestamp de cálculo
    df = df.copy()
    df['calculated_at'] = datetime.now().isoformat()

    df.to_csv(filepath, index=False)

    print(f"\n CSV exportado: {filepath}")
    print(f"    -> {len(df):,} filas")

    return filepath


def export_to_bigquery(
    df: pd.DataFrame,
    table_id: str = None,
    if_exists: str = 'replace'
) -> None:
    """
    Exporta DataFrame a BigQuery.

    Args:
        df: DataFrame a exportar
        table_id: ID de tabla destino (proyecto.dataset.tabla)
        if_exists: 'replace', 'append', o 'fail'
    """
    if table_id is None:
        table_id = OUTPUT_TABLE

    client = bigquery.Client(project=GCP_PROJECT)

    # Agregar timestamp
    df = df.copy()
    df['calculated_at'] = datetime.now()

    # Configurar job
    job_config = bigquery.LoadJobConfig()

    if if_exists == 'replace':
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    elif if_exists == 'append':
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY

    job_config.autodetect = True

    print(f"\n Exportando a BigQuery: {table_id}")
    print(f"    -> {len(df):,} filas")
    print(f"    -> Modo: {if_exists}")

    # Cargar data
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Esperar a que termine

    print(f"    -> Completado!")


def prepare_export_dataframe(df_thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el DataFrame final para exportación.

    Args:
        df_thresholds: DataFrame con umbrales calculados

    Returns:
        DataFrame listo para exportar
    """
    # Seleccionar columnas relevantes
    export_cols = [
        'zone_id',
        'zone_name',
        'city_id',
        'city_name',
        'low_delay',
        'preventive',
        'containment_I',
        'containment_II',
        'inoperable',
        'severity_score',
        'assigned_stage',
        'current_delay_p50',
        'current_delay_p90',
        'current_fail_rate',
        'current_staffing_rate'
    ]

    # Filtrar columnas que existen
    existing_cols = [col for col in export_cols if col in df_thresholds.columns]

    df_export = df_thresholds[existing_cols].copy()

    # Redondear valores numéricos
    numeric_cols = df_export.select_dtypes(include=['float64', 'float32']).columns
    df_export[numeric_cols] = df_export[numeric_cols].round(2)

    return df_export


def export_all(
    df_thresholds: pd.DataFrame,
    to_csv: bool = True,
    to_bq: bool = True,
    output_dir: str = "output"
) -> dict:
    """
    Exporta resultados a CSV y/o BigQuery.

    Args:
        df_thresholds: DataFrame con umbrales calculados
        to_csv: Si exportar a CSV
        to_bq: Si exportar a BigQuery
        output_dir: Directorio para CSV

    Returns:
        Diccionario con paths/info de exportación
    """
    print("\n" + "=" * 60)
    print("EXPORTACIÓN DE RESULTADOS")
    print("=" * 60)

    df_export = prepare_export_dataframe(df_thresholds)
    results = {'rows': len(df_export)}

    if to_csv:
        csv_path = export_to_csv(df_export, output_dir)
        results['csv_path'] = csv_path

    if to_bq:
        export_to_bigquery(df_export)
        results['bq_table'] = OUTPUT_TABLE

    print("=" * 60)
    print("EXPORTACIÓN COMPLETADA")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Test con data de ejemplo
    from data_extraction import extract_all_data
    from feature_engineering import merge_all_metrics
    from stage_calculator import calculate_delay_thresholds_per_zone

    data = extract_all_data()
    df_features = merge_all_metrics(data)
    df_thresholds = calculate_delay_thresholds_per_zone(df_features)

    # Exportar solo a CSV para test
    results = export_all(df_thresholds, to_csv=True, to_bq=False)
    print(f"\nResultados: {results}")
