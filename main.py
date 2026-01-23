"""
Severity Model - Main Script
============================
Script principal para ejecutar el modelo de severidad.

Uso:
    python main.py                    # Ejecutar completo (CSV + BQ)
    python main.py --csv-only         # Solo exportar CSV
    python main.py --analyze          # Solo análisis, sin exportar
"""

import argparse
import sys
from datetime import datetime

# Agregar paths
sys.path.insert(0, '.')

from config.settings import COUNTRY_CODE, LOOKBACK_DAYS, WEIGHTS, STAGE_PERCENTILES
from src.data_extraction import extract_all_data
from src.feature_engineering import merge_all_metrics, get_zone_statistics
from src.stage_calculator import (
    calculate_delay_thresholds_per_zone,
    compare_with_current_config,
    get_stage_summary
)
from src.export import export_all


def print_header():
    """Imprime header del modelo."""
    print("\n" + "=" * 70)
    print("  SEVERITY MODEL - Modelo de Stages por Zona Logística")
    print("=" * 70)
    print(f"  País: {COUNTRY_CODE.upper()}")
    print(f"  Período de análisis: últimos {LOOKBACK_DAYS} días")
    print(f"  Pesos: delay={WEIGHTS['delay']}, fail_rate={WEIGHTS['fail_rate']}, staffing={WEIGHTS['staffing_affection']}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def run_model(
    export_csv: bool = True,
    export_bq: bool = True,
    show_analysis: bool = True
) -> dict:
    """
    Ejecuta el modelo completo.

    Args:
        export_csv: Si exportar a CSV
        export_bq: Si exportar a BigQuery
        show_analysis: Si mostrar análisis detallado

    Returns:
        Diccionario con resultados
    """
    print_header()

    # 1. Extracción de datos
    print("\n[1/4] EXTRACCIÓN DE DATOS")
    print("-" * 40)
    data = extract_all_data()

    # 2. Feature Engineering
    print("\n[2/4] FEATURE ENGINEERING")
    print("-" * 40)
    df_features = merge_all_metrics(data)

    if show_analysis:
        print("\nEstadísticas de métricas:")
        print(get_zone_statistics(df_features))

    # 3. Cálculo de stages
    print("\n[3/4] CÁLCULO DE STAGES Y UMBRALES")
    print("-" * 40)
    df_thresholds = calculate_delay_thresholds_per_zone(df_features)

    if show_analysis:
        print("\nResumen por stage:")
        summary = get_stage_summary(df_thresholds)
        print(summary)

        # Comparar con config actual
        df_compare = compare_with_current_config(df_thresholds, data['current_config'])

    # 4. Exportación
    print("\n[4/4] EXPORTACIÓN")
    print("-" * 40)
    if export_csv or export_bq:
        results = export_all(df_thresholds, to_csv=export_csv, to_bq=export_bq)
    else:
        print("Exportación deshabilitada (modo análisis)")
        results = {'rows': len(df_thresholds)}

    # Resumen final
    print("\n" + "=" * 70)
    print("  EJECUCIÓN COMPLETADA")
    print("=" * 70)
    print(f"  Zonas procesadas: {len(df_thresholds):,}")
    if 'csv_path' in results:
        print(f"  CSV: {results['csv_path']}")
    if 'bq_table' in results:
        print(f"  BigQuery: {results['bq_table']}")
    print("=" * 70 + "\n")

    return {
        'data': data,
        'features': df_features,
        'thresholds': df_thresholds,
        'export_results': results
    }


def main():
    """Función principal con argumentos de línea de comando."""
    parser = argparse.ArgumentParser(
        description='Severity Model - Calcula umbrales de stages por zona'
    )
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Solo exportar a CSV (no a BigQuery)'
    )
    parser.add_argument(
        '--bq-only',
        action='store_true',
        help='Solo exportar a BigQuery (no a CSV)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Solo análisis, sin exportar'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso, menos output'
    )

    args = parser.parse_args()

    # Determinar qué exportar
    if args.analyze:
        export_csv = False
        export_bq = False
    elif args.csv_only:
        export_csv = True
        export_bq = False
    elif args.bq_only:
        export_csv = False
        export_bq = True
    else:
        export_csv = True
        export_bq = True

    # Ejecutar modelo
    run_model(
        export_csv=export_csv,
        export_bq=export_bq,
        show_analysis=not args.quiet
    )


if __name__ == "__main__":
    main()
