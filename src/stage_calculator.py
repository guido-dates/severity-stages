"""
Stage Calculator Module
=======================
Calculo de umbrales de mean_delay para cada stage basado en las curvas de stress.

NOTA: El suavizado Savitzky-Golay ahora se aplica en feature_engineering.py
y este modulo usa la columna 'stress_smoothed' precalculada.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
sys.path.append('..')
from config.settings import (
    STAGE_STRESS_PERCENTILES,
    MIN_ORDERS_PER_POINT,
    MIN_DELAY_FOR_TRUNCATION
)

STAGE_ORDER = ["low_delay", "preventive", "containment_I", "containment_II", "inoperable"]


def _build_stage_buckets(
    percentiles: Dict[str, float]
) -> Dict[str, Tuple[float, float]]:
    buckets: Dict[str, Tuple[float, float]] = {}
    prev = 0.0

    for stage in STAGE_ORDER:
        if stage not in percentiles:
            raise ValueError(f"Falta percentil para stage '{stage}' en STAGE_STRESS_PERCENTILES")

        upper = float(percentiles[stage])
        if upper <= prev:
            raise ValueError("STAGE_STRESS_PERCENTILES debe ser estrictamente creciente")
        if upper > 100:
            raise ValueError("STAGE_STRESS_PERCENTILES no puede superar 100")

        buckets[stage] = (prev, upper)
        prev = upper

    if abs(prev - 100.0) > 1e-6:
        raise ValueError("STAGE_STRESS_PERCENTILES debe terminar en 100 (cota superior de inoperable)")

    return buckets


def calculate_weighted_delay_for_bucket(
    zone_data: pd.DataFrame,
    stress_min: float,
    stress_max: float,
    use_smoothed: bool = False
) -> float:
    """
    Calcula el delay promedio ponderado por órdenes para un rango de stress.

    Args:
        zone_data: DataFrame con datos de una zona
        stress_min: Límite inferior del bucket de stress
        stress_max: Límite superior del bucket de stress
        use_smoothed: Si True, usa 'stress_smoothed' en vez de 'stress_score_normalized'

    Returns:
        Delay promedio ponderado por total_orders_matched
    """
    # Determinar qué columna de stress usar
    stress_col = 'stress_smoothed' if use_smoothed and 'stress_smoothed' in zone_data.columns else 'stress_score_normalized'

    # Filtrar registros en el rango de stress
    bucket_data = zone_data[
        (zone_data[stress_col] >= stress_min) &
        (zone_data[stress_col] < stress_max)
    ]

    if len(bucket_data) == 0:
        return np.nan

    # Calcular promedio ponderado por órdenes
    total_orders = bucket_data['total_orders_matched'].sum()
    if total_orders == 0:
        return bucket_data['mean_delay_minute'].mean()

    weighted_delay = (
        bucket_data['mean_delay_minute'] * bucket_data['total_orders_matched']
    ).sum() / total_orders

    return weighted_delay


def find_stress_peak_delay(
    zone_data: pd.DataFrame,
    min_orders: int = None,
    min_delay_cutoff: int = None
) -> Tuple[float, float]:
    """
    Encuentra el delay donde el stress alcanza su máximo.

    La idea es truncar la curva de stress en su punto máximo porque queremos
    actuar ANTES de que el stress llegue al máximo, no después cuando el daño
    ya está hecho.

    Args:
        zone_data: DataFrame con datos de una zona (debe tener stress_score_normalized,
                   mean_delay_minute, total_orders_matched)
        min_orders: Mínimo de órdenes para considerar un punto válido
        min_delay_cutoff: Delay mínimo de corte (si el máximo es antes, usar este)

    Returns:
        Tuple (peak_delay, max_stress_value)
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT
    if min_delay_cutoff is None:
        min_delay_cutoff = MIN_DELAY_FOR_TRUNCATION

    # Filtrar puntos válidos (con suficientes órdenes)
    valid_data = zone_data[zone_data['total_orders_matched'] >= min_orders]
    if len(valid_data) < 3:
        valid_data = zone_data

    # Encontrar el punto con máximo stress
    max_idx = valid_data['stress_score_normalized'].idxmax()
    peak_delay = valid_data.loc[max_idx, 'mean_delay_minute']
    max_stress = valid_data.loc[max_idx, 'stress_score_normalized']

    # Aplicar mínimo de corte para garantizar rango suficiente para 5 stages
    peak_delay = max(peak_delay, min_delay_cutoff)

    return peak_delay, max_stress


def calculate_zone_thresholds(df_stress: pd.DataFrame, min_orders: int = None) -> pd.DataFrame:
    """
    Calcula los umbrales de mean_delay para cada zona basado en las curvas de stress.

    IMPORTANTE: El DataFrame debe tener la columna 'stress_smoothed' precalculada
    por feature_engineering.build_stress_curves().

    Lógica:
    1. Usa 'stress_smoothed' y 'delay_at_max' precalculados
    2. Trunca en el pico de stress (delay_at_max)
    3. Divide en buckets segun percentiles del stress suavizado (STAGE_STRESS_PERCENTILES)
    4. Calcula delay promedio ponderado por órdenes de cada bucket

    Buckets (segun config):
        - P0 a P{low_delay} -> low_delay
        - P{low_delay} a P{preventive} -> preventive
        - P{preventive} a P{containment_I} -> containment_I
        - P{containment_I} a P{containment_II} -> containment_II
        - P{containment_II} a P{inoperable} -> inoperable

    Args:
        df_stress: DataFrame con stress curves (debe incluir stress_smoothed,
                   delay_at_max, zone_id, mean_delay_minute, total_orders_matched)
        min_orders: Mínimo de órdenes por punto (default: MIN_ORDERS_PER_POINT)

    Returns:
        DataFrame con umbrales de delay por zona (incluye peak_delay y points_used)
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    # Verificar que el DataFrame tiene las columnas necesarias
    if 'stress_smoothed' not in df_stress.columns:
        raise ValueError("DataFrame debe tener columna 'stress_smoothed'. "
                        "Ejecutar feature_engineering.build_stress_curves() primero.")

    print("\n" + "=" * 60)
    print("CALCULO DE UMBRALES DE DELAY POR ZONA")
    print("=" * 60)

    print("\nMétodo: DELAY PROMEDIO PONDERADO POR BUCKET DE STRESS SUAVIZADO")
    print(f"Filtrando puntos con menos de {min_orders} órdenes")
    print("Usando curva suavizada precalculada (Savitzky-Golay)")
    print("Truncando curva en el máximo de stress (delay_at_max)")
    print("Percentiles del stress suavizado (segun config):")

    # Definir los rangos de percentiles segun config
    stage_buckets = _build_stage_buckets(STAGE_STRESS_PERCENTILES)

    for stage in STAGE_ORDER:
        p_min, p_max = stage_buckets[stage]
        print(f"  {stage}: stress en P{p_min}-P{p_max} -> delay promedio ponderado")

    thresholds_list = []
    zones = df_stress['zone_id'].unique()
    zones_with_warnings = 0

    for zone_id in zones:
        zone_data_all = df_stress[df_stress['zone_id'] == zone_id].copy()

        # Filtrar puntos con suficientes órdenes
        zone_data = zone_data_all[zone_data_all['total_orders_matched'] >= min_orders].copy()

        # Si no hay suficientes puntos filtrados, usar todos con warning
        if len(zone_data) < 5:
            zone_data = zone_data_all.copy()
            zones_with_warnings += 1

        # Usar delay_at_max precalculado (pico de stress de la curva suavizada)
        if 'delay_at_max' in zone_data_all.columns:
            peak_delay = zone_data_all['delay_at_max'].iloc[0]
            max_stress = zone_data_all['stress_smooth_max'].iloc[0] if 'stress_smooth_max' in zone_data_all.columns else zone_data['stress_smoothed'].max()
        else:
            # Fallback si no hay delay_at_max (compatibilidad)
            peak_delay, max_stress = find_stress_peak_delay(zone_data, min_orders)

        # Truncar datos hasta el pico de stress
        zone_data_truncated = zone_data[zone_data['mean_delay_minute'] <= peak_delay].copy()

        # Si el truncado deja muy pocos puntos, usar todos hasta el pico
        if len(zone_data_truncated) < 5:
            zone_data_truncated = zone_data_all[zone_data_all['mean_delay_minute'] <= peak_delay].copy()

        # Usar los datos truncados para el resto del cálculo
        zone_data = zone_data_truncated

        # Info básica de la zona
        zone_info = {
            'zone_id': zone_id,
            'zone_name': zone_data_all['zone_name'].iloc[0] if 'zone_name' in zone_data_all.columns else '',
            'city_id': zone_data_all['city_id'].iloc[0] if 'city_id' in zone_data_all.columns else '',
            'city_name': zone_data_all['city_name'].iloc[0] if 'city_name' in zone_data_all.columns else '',
            'delay_min': zone_data['mean_delay_minute'].min(),
            'delay_max': zone_data['mean_delay_minute'].max(),
            'peak_delay': peak_delay,
            'max_stress': max_stress,
            'total_records': zone_data_all['records_count'].sum(),
            'valid_points': len(zone_data),
            'total_points': len(zone_data_all),
            'points_after_peak': len(zone_data_all[zone_data_all['mean_delay_minute'] > peak_delay])
        }

        # Ordenar por delay
        zone_data = zone_data.sort_values('mean_delay_minute').reset_index(drop=True)

        # Skip zonas sin datos suficientes
        if len(zone_data) < 5:
            continue

        # Usar stress_smoothed precalculado (ya viene de feature_engineering)
        stress_smooth = zone_data['stress_smoothed'].values

        # Calcular percentiles segun config sobre el stress SUAVIZADO
        bucket_percentiles = [0] + [stage_buckets[stage][1] for stage in STAGE_ORDER]
        percentile_values = {p: np.percentile(stress_smooth, p) for p in bucket_percentiles}

        # Para cada stage, calcular el delay promedio ponderado del bucket
        for stage in STAGE_ORDER:
            p_min, p_max = stage_buckets[stage]
            stress_min = percentile_values[p_min]
            stress_max = percentile_values[p_max]

            zone_info[stage] = calculate_weighted_delay_for_bucket(
                zone_data, stress_min, stress_max, use_smoothed=True
            )

        thresholds_list.append(zone_info)

    df_thresholds = pd.DataFrame(thresholds_list)
    stages_order = STAGE_ORDER

    print(f"\n  -> Umbrales calculados para {len(df_thresholds):,} zonas")
    if zones_with_warnings > 0:
        print(f"  -> WARNING: {zones_with_warnings} zonas con pocos puntos válidos (usaron todos los datos)")

    # Estadísticas del truncado
    print("\nEstadisticas del truncado (delay del pico de stress):")
    print(f"  peak_delay: mean={df_thresholds['peak_delay'].mean():.1f}, "
          f"min={df_thresholds['peak_delay'].min():.0f}, "
          f"max={df_thresholds['peak_delay'].max():.0f}")
    print(f"  puntos ignorados (despues del pico): {df_thresholds['points_after_peak'].sum():,}")

    # Estadísticas de umbrales
    print("\nEstadisticas de umbrales (minutos de mean_delay):")
    for stage in stages_order:
        mean_val = df_thresholds[stage].mean()
        std_val = df_thresholds[stage].std()
        min_val = df_thresholds[stage].min()
        max_val = df_thresholds[stage].max()
        print(f"  {stage}: mean={mean_val:.1f}, std={std_val:.1f}, range=[{min_val:.0f}, {max_val:.0f}]")

    # Verificar monotonicidad
    non_monotonic = 0
    for _, row in df_thresholds.iterrows():
        values = [row[s] for s in stages_order]
        if values != sorted(values):
            non_monotonic += 1

    if non_monotonic > 0:
        print(f"\n  ⚠️  {non_monotonic} zonas con umbrales no ascendentes (revisar)")
    else:
        print(f"\n  ✓ Todas las zonas tienen umbrales estrictamente ascendentes")

    print("=" * 60)

    return df_thresholds


def compare_with_current_config(
    df_new: pd.DataFrame,
    df_current: pd.DataFrame
) -> pd.DataFrame:
    """
    Compara los nuevos umbrales con la configuracion actual.

    Args:
        df_new: DataFrame con nuevos umbrales por zona
        df_current: DataFrame con configuracion actual por ciudad

    Returns:
        DataFrame con comparacion
    """
    print("\n" + "=" * 60)
    print("COMPARACION CON CONFIGURACION ACTUAL")
    print("=" * 60)

    # Merge por city_id
    df_compare = df_new.merge(
        df_current[['city_id', 'low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']],
        on='city_id',
        how='left',
        suffixes=('_new', '_current')
    )

    # Calcular diferencias
    stages = ['low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']
    for stage in stages:
        new_col = f'{stage}_new' if f'{stage}_new' in df_compare.columns else stage
        current_col = f'{stage}_current'

        if current_col in df_compare.columns:
            df_compare[f'{stage}_diff'] = df_compare[new_col] - df_compare[current_col]

    # Resumen de cambios
    if 'low_delay_diff' in df_compare.columns:
        print("\nResumen de cambios vs config actual:")
        for stage in stages:
            diff_col = f'{stage}_diff'
            if diff_col in df_compare.columns:
                mean_diff = df_compare[diff_col].mean()
                zones_higher = (df_compare[diff_col] > 0).sum()
                zones_lower = (df_compare[diff_col] < 0).sum()
                print(f"  {stage}: promedio {mean_diff:+.1f} min ({zones_higher} zonas suben, {zones_lower} bajan)")

    print(f"\n  -> {len(df_compare):,} zonas comparadas")

    return df_compare


def get_thresholds_summary(df_thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadistico de los umbrales calculados.
    """
    stages = ['low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']

    summary = df_thresholds[stages].describe().T
    summary['range'] = summary['max'] - summary['min']

    return summary


def export_thresholds_for_config(df_thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los umbrales en formato listo para exportar a config.

    Returns:
        DataFrame con columnas: zone_id, zone_name, city_id, city_name,
        low_delay, preventive, containment_I, containment_II, inoperable
    """
    cols = ['zone_id', 'zone_name', 'city_id', 'city_name',
            'low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']
    stages = ['low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']

    df_export = df_thresholds[cols].copy()

    # Manejar NaN (buckets vacíos): rellenar con interpolación y forzar monotonía
    for stage in stages:
        df_export[stage] = df_export[stage].ffill().bfill()

    # Forzar monotonía creciente
    for i in range(1, len(stages)):
        prev_stage = stages[i-1]
        curr_stage = stages[i]
        df_export[curr_stage] = df_export[[prev_stage, curr_stage]].max(axis=1)

    # Redondear a enteros
    for stage in stages:
        df_export[stage] = df_export[stage].round().astype(int)

    return df_export


if __name__ == "__main__":
    from data_extraction import extract_all_data
    from feature_engineering import build_stress_curves

    data = extract_all_data()
    df_stress = build_stress_curves(data)
    df_thresholds = calculate_zone_thresholds(df_stress)

    print("\n" + "=" * 60)
    print("MUESTRA DE UMBRALES")
    print("=" * 60)
    print(df_thresholds[['zone_name', 'city_name', 'low_delay', 'preventive',
                         'containment_I', 'containment_II', 'inoperable']].head(10))

    # Comparar con config actual
    df_compare = compare_with_current_config(df_thresholds, data['current_config'])
