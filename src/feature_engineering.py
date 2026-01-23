"""
Feature Engineering Module
==========================
Calculo de baseline y stress score por zona y nivel de mean_delay.
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import interpolate
from scipy.signal import savgol_filter
import sys
sys.path.append('..')
from config.settings import (
    STRESS_WEIGHTS, MIN_RECORDS_PER_DELAY, MIN_ORDERS_PER_POINT,
    SAVGOL_WINDOW, SAVGOL_POLYORDER,
    MAX_MEAN_DELAY_MINUTE, WEIGHTED_DELAY_P99
)

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return np.nan
    if quantile <= 0:
        return float(np.min(values))
    if quantile >= 1:
        return float(np.max(values))

    if weights is None:
        return float(np.quantile(values, quantile))

    weights = np.asarray(weights)
    valid_mask = weights > 0
    if not valid_mask.any():
        return float(np.quantile(values, quantile))

    values = np.asarray(values)[valid_mask]
    weights = weights[valid_mask]

    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]

    cumulative = np.cumsum(weights_sorted)
    cutoff = quantile * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), len(values_sorted) - 1)
    return float(values_sorted[idx])


def apply_delay_filters(
    df: pd.DataFrame,
    delay_col: str = "mean_delay_minute",
    weight_col: str = "total_orders_matched"
) -> pd.DataFrame:
    """
    Aplica filtros de delay al inicio del pipeline:
    1) Filtro duro por MAX_MEAN_DELAY_MINUTE
    2) Recorte por percentil ponderado (P99) por zona
    """
    if df.empty:
        return df

    filtered = df.copy()
    original_len = len(filtered)

    if MAX_MEAN_DELAY_MINUTE is not None and delay_col in filtered.columns:
        before = len(filtered)
        filtered = filtered[filtered[delay_col] <= MAX_MEAN_DELAY_MINUTE].copy()
        removed = before - len(filtered)
        print(f"  -> Filtro max delay <= {MAX_MEAN_DELAY_MINUTE}: {removed:,} puntos removidos")

    if WEIGHTED_DELAY_P99 is not None and delay_col in filtered.columns:
        removed_total = 0
        filtered_list = []

        for zone_id, zone_data in filtered.groupby("zone_id"):
            weights = None
            if weight_col in zone_data.columns:
                weights = zone_data[weight_col].fillna(0)
            if weights is not None and weights.sum() > 0:
                threshold = _weighted_quantile(
                    zone_data[delay_col].to_numpy(),
                    weights.to_numpy(),
                    WEIGHTED_DELAY_P99
                )
            else:
                threshold = float(zone_data[delay_col].quantile(WEIGHTED_DELAY_P99))

            zone_filtered = zone_data[zone_data[delay_col] <= threshold].copy()
            removed_total += len(zone_data) - len(zone_filtered)
            filtered_list.append(zone_filtered)

        if filtered_list:
            filtered = pd.concat(filtered_list, ignore_index=True)

        pctl_label = int(round(WEIGHTED_DELAY_P99 * 100))
        print(f"  -> Recorte por percentil ponderado P{pctl_label}: {removed_total:,} puntos removidos")

    total_removed = original_len - len(filtered)
    print(f"  -> Total puntos removidos por filtros de delay: {total_removed:,}")
    return filtered


def calculate_zone_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el baseline (media y std) de cada zona en condiciones normales.

    El baseline se define como el promedio PONDERADO POR ORDENES de fail_rate
    y staffing_affection cuando la zona esta en su rango bajo de delay (P25 o menos).

    IMPORTANTE: Solo considera puntos con suficientes ordenes (MIN_ORDERS_PER_POINT)
    para evitar que outliers con pocas ordenes distorsionen el baseline.

    Args:
        df: DataFrame con columnas zone_id, mean_delay_minute, avg_fail_rate,
            avg_staffing_affection, records_count, total_orders_matched

    Returns:
        DataFrame con baseline por zona: zone_id, baseline_fail_rate,
        baseline_staffing, std_fail_rate, std_staffing
    """
    print("Calculando baselines por zona (ponderado por ordenes)...")
    print(f"  -> Filtrando puntos con menos de {MIN_ORDERS_PER_POINT} ordenes")

    baselines = []

    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].copy()

        # Filtrar puntos con suficientes ordenes
        zone_data_filtered = zone_data[zone_data['total_orders_matched'] >= MIN_ORDERS_PER_POINT]

        # Si no hay suficientes puntos filtrados, usar todos pero con warning
        if len(zone_data_filtered) < 3:
            zone_data_filtered = zone_data
            low_data_warning = True
        else:
            low_data_warning = False

        # Calcular P25 del delay para esta zona (usando datos filtrados)
        delay_p25 = zone_data_filtered['mean_delay_minute'].quantile(0.25)

        # Filtrar solo los registros en condiciones "normales" (delay bajo)
        normal_data = zone_data_filtered[zone_data_filtered['mean_delay_minute'] <= delay_p25]

        # Si no hay suficientes datos, usar todos los datos con delay < 5
        if len(normal_data) < 5:
            normal_data = zone_data_filtered[zone_data_filtered['mean_delay_minute'] <= 5]

        # Si aun no hay datos, usar los primeros 25% de registros ordenados por delay
        if len(normal_data) < 3:
            normal_data = zone_data_filtered.nsmallest(max(3, len(zone_data_filtered) // 4), 'mean_delay_minute')

        # Calcular baseline PONDERADO por ordenes
        total_orders = normal_data['total_orders_matched'].sum()
        if total_orders > 0:
            baseline_fail_rate = (normal_data['avg_fail_rate'] * normal_data['total_orders_matched']).sum() / total_orders
            baseline_staffing = (normal_data['avg_staffing_affection'] * normal_data['total_orders_matched']).sum() / total_orders
        else:
            baseline_fail_rate = normal_data['avg_fail_rate'].mean()
            baseline_staffing = normal_data['avg_staffing_affection'].mean()

        baselines.append({
            'zone_id': zone_id,
            'baseline_fail_rate': baseline_fail_rate,
            'baseline_staffing': baseline_staffing,
            'std_fail_rate': max(normal_data['avg_fail_rate'].std(), 0.1),  # min std para evitar div/0
            'std_staffing': max(normal_data['avg_staffing_affection'].std(), 0.1),
            'delay_p25': delay_p25,
            'records_baseline': len(normal_data),
            'orders_baseline': int(total_orders),
            'low_data_warning': low_data_warning
        })

    df_baselines = pd.DataFrame(baselines)
    n_warnings = df_baselines['low_data_warning'].sum()
    print(f"  -> Baselines calculados para {len(df_baselines):,} zonas")
    if n_warnings > 0:
        print(f"  -> WARNING: {n_warnings} zonas con pocos datos (< {MIN_ORDERS_PER_POINT} ordenes)")

    return df_baselines


def calculate_stress_scores(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el stress score para cada zona y nivel de delay.

    stress_score = w1 * z_score(fail_rate) + w2 * z_score(staffing)
    donde z_score = (valor - baseline) / std_baseline

    IMPORTANTE: La normalizacion usa promedios PONDERADOS POR ORDENES
    y solo considera puntos con suficientes ordenes para evitar outliers.

    Args:
        df: DataFrame con datos de delay_stress
        baselines: DataFrame con baselines por zona

    Returns:
        DataFrame con stress_score por zona y delay
    """
    print("Calculando stress scores...")

    # Merge con baselines
    df = df.merge(baselines, on='zone_id', how='left')

    # Calcular z-scores
    df['z_fail_rate'] = (df['avg_fail_rate'] - df['baseline_fail_rate']) / df['std_fail_rate']
    df['z_staffing'] = (df['avg_staffing_affection'] - df['baseline_staffing']) / df['std_staffing']

    # Clip z-scores para evitar valores extremos
    df['z_fail_rate'] = df['z_fail_rate'].clip(-5, 10)
    df['z_staffing'] = df['z_staffing'].clip(-5, 10)

    # Calcular stress score ponderado
    df['stress_score'] = (
        STRESS_WEIGHTS['fail_rate'] * df['z_fail_rate'] +
        STRESS_WEIGHTS['staffing_affection'] * df['z_staffing']
    )

    # Marcar puntos con suficientes ordenes
    df['has_enough_orders'] = df['total_orders_matched'] >= MIN_ORDERS_PER_POINT

    # Normalizar stress_score POR ZONA a 0-100
    # USANDO SOLO PUNTOS CON SUFICIENTES ORDENES para calcular baseline y max
    print(f"  -> Normalizando stress POR ZONA (solo puntos con >= {MIN_ORDERS_PER_POINT} ordenes)")
    df['stress_score_normalized'] = 0.0  # Inicializar

    for zone_id in df['zone_id'].unique():
        zone_mask = df['zone_id'] == zone_id
        zone_data = df.loc[zone_mask].copy()

        # Usar solo puntos con suficientes ordenes para calcular baseline/max
        zone_data_valid = zone_data[zone_data['has_enough_orders']]

        # Fallback si no hay suficientes puntos validos
        if len(zone_data_valid) < 3:
            zone_data_valid = zone_data

        # Baseline: stress PONDERADO POR ORDENES cuando delay esta en P25 o menos
        delay_p25 = zone_data_valid['mean_delay_minute'].quantile(0.25)
        low_delay_data = zone_data_valid[zone_data_valid['mean_delay_minute'] <= delay_p25]

        if len(low_delay_data) < 3:
            low_delay_data = zone_data_valid.nsmallest(3, 'mean_delay_minute')

        # Baseline de stress = promedio PONDERADO cuando delay es bajo
        total_orders_low = low_delay_data['total_orders_matched'].sum()
        if total_orders_low > 0:
            stress_baseline = (low_delay_data['stress_score'] * low_delay_data['total_orders_matched']).sum() / total_orders_low
        else:
            stress_baseline = low_delay_data['stress_score'].mean()

        # Maximo de stress: P99 PONDERADO por ordenes
        # Para esto ordenamos por stress y tomamos el valor donde se acumula 99% de ordenes
        zone_sorted = zone_data_valid.sort_values('stress_score')
        zone_sorted['orders_cumsum'] = zone_sorted['total_orders_matched'].cumsum()
        total_orders_zone = zone_sorted['total_orders_matched'].sum()

        if total_orders_zone > 0:
            p99_threshold = total_orders_zone * 0.99
            stress_max_row = zone_sorted[zone_sorted['orders_cumsum'] >= p99_threshold].iloc[0]
            stress_max = stress_max_row['stress_score']
        else:
            stress_max = zone_data_valid['stress_score'].quantile(0.99)

        if stress_max > stress_baseline:
            df.loc[zone_mask, 'stress_score_normalized'] = (
                (zone_data['stress_score'] - stress_baseline) / (stress_max - stress_baseline) * 100
            ).clip(0, 100)
        else:
            # Zona sin variacion de stress
            df.loc[zone_mask, 'stress_score_normalized'] = 0

    n_filtered = (~df['has_enough_orders']).sum()
    print(f"  -> Stress scores calculados para {len(df):,} registros")
    print(f"  -> {n_filtered:,} puntos con < {MIN_ORDERS_PER_POINT} ordenes (marcados pero incluidos)")

    return df


def interpolate_missing_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpola valores de stress para niveles de delay con pocos datos.

    Args:
        df: DataFrame con stress scores por zona y delay

    Returns:
        DataFrame con valores interpolados
    """
    print("Interpolando valores faltantes...")

    interpolated_dfs = []

    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].copy()

        # Filtrar puntos con suficientes registros
        valid_data = zone_data[zone_data['records_count'] >= MIN_RECORDS_PER_DELAY]

        if len(valid_data) < 3:
            # Si no hay suficientes puntos validos, usar todos
            interpolated_dfs.append(zone_data)
            continue

        # Crear interpolador
        delays = valid_data['mean_delay_minute'].values
        stress = valid_data['stress_score_normalized'].values

        try:
            # Interpolacion lineal
            f = interpolate.interp1d(delays, stress, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')

            # Aplicar interpolacion a puntos con pocos datos
            zone_data.loc[zone_data['records_count'] < MIN_RECORDS_PER_DELAY,
                         'stress_score_normalized'] = f(
                             zone_data.loc[zone_data['records_count'] < MIN_RECORDS_PER_DELAY,
                                          'mean_delay_minute']
                         )
            zone_data['interpolated'] = zone_data['records_count'] < MIN_RECORDS_PER_DELAY
        except Exception:
            zone_data['interpolated'] = False

        interpolated_dfs.append(zone_data)

    result = pd.concat(interpolated_dfs, ignore_index=True)
    n_interpolated = result['interpolated'].sum() if 'interpolated' in result.columns else 0
    print(f"  -> {n_interpolated:,} puntos interpolados")

    return result


def apply_savgol_smoothing(df: pd.DataFrame, min_orders: int = None) -> pd.DataFrame:
    """
    Aplica suavizado Savitzky-Golay a las curvas de stress por zona.

    Tambien calcula el minimo y maximo de la curva suavizada para cada zona.

    Args:
        df: DataFrame con stress_score_normalized por zona y delay
        min_orders: Minimo de ordenes para considerar un punto valido

    Returns:
        DataFrame con columnas adicionales: stress_smoothed,
        delay_at_min, delay_at_max, stress_smooth_min, stress_smooth_max
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    print("Aplicando suavizado Savitzky-Golay a curvas de stress...")

    result_dfs = []

    for zone_id in df['zone_id'].unique():
        zone_data = df[df['zone_id'] == zone_id].copy()

        # Filtrar puntos con suficientes ordenes para el suavizado
        valid_mask = zone_data['total_orders_matched'] >= min_orders
        valid_data = zone_data[valid_mask].copy()

        # Fallback si muy pocos puntos validos
        if len(valid_data) < 5:
            valid_data = zone_data.copy()
            valid_mask = pd.Series([True] * len(zone_data), index=zone_data.index)

        # Ordenar por delay
        valid_data = valid_data.sort_values('mean_delay_minute').reset_index(drop=True)
        stress_values = valid_data['stress_score_normalized'].values
        delay_values = valid_data['mean_delay_minute'].values

        # Aplicar Savitzky-Golay
        if len(stress_values) >= SAVGOL_WINDOW:
            stress_smooth = savgol_filter(stress_values, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)
        else:
            # Ajustar window si hay pocos puntos
            window = len(stress_values) if len(stress_values) % 2 == 1 else len(stress_values) - 1
            if window >= 5:
                polyorder = min(SAVGOL_POLYORDER, window - 2)
                stress_smooth = savgol_filter(stress_values, window_length=window, polyorder=polyorder)
            else:
                # Si muy pocos puntos, no suavizar
                stress_smooth = stress_values.copy()

        # Calcular min y max de la curva suavizada
        min_idx = np.argmin(stress_smooth)
        max_idx = np.argmax(stress_smooth)

        delay_at_min = delay_values[min_idx]
        delay_at_max = delay_values[max_idx]
        stress_smooth_min = stress_smooth[min_idx]
        stress_smooth_max = stress_smooth[max_idx]

        # Crear mapeo de delay -> stress_smoothed
        delay_to_smooth = dict(zip(delay_values, stress_smooth))

        # Asignar valores suavizados al dataframe original de la zona
        zone_data['stress_smoothed'] = zone_data['mean_delay_minute'].map(delay_to_smooth)

        # Para puntos que no estaban en valid_data, interpolar o usar el original
        missing_mask = zone_data['stress_smoothed'].isna()
        if missing_mask.any():
            zone_data.loc[missing_mask, 'stress_smoothed'] = zone_data.loc[missing_mask, 'stress_score_normalized']

        # Agregar info de min/max (misma para toda la zona)
        zone_data['delay_at_min'] = delay_at_min
        zone_data['delay_at_max'] = delay_at_max
        zone_data['stress_smooth_min'] = stress_smooth_min
        zone_data['stress_smooth_max'] = stress_smooth_max

        result_dfs.append(zone_data)

    result = pd.concat(result_dfs, ignore_index=True)
    print(f"  -> Suavizado aplicado a {result['zone_id'].nunique()} zonas")
    print(f"  -> Parametros: window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER}")

    return result


def build_stress_curves(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pipeline principal: construye las curvas de stress por zona.

    Args:
        data: Diccionario con DataFrames de delay_stress y zone_info

    Returns:
        DataFrame con stress curves por zona y nivel de delay
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING - STRESS CURVES")
    print("=" * 60)

    df = data['delay_stress'].copy()

    # 0. Filtros iniciales de delay (hard cap + P99 ponderado)
    print("Aplicando filtros iniciales de delay...")
    df = apply_delay_filters(df)

    # 1. Calcular baselines
    baselines = calculate_zone_baselines(df)

    # 2. Calcular stress scores
    df = calculate_stress_scores(df, baselines)

    # 3. Interpolar valores faltantes
    df = interpolate_missing_delays(df)

    # 4. Aplicar suavizado Savitzky-Golay
    df = apply_savgol_smoothing(df)

    # 5. Agregar info de zona
    df = df.merge(data['zone_info'], on='zone_id', how='left')

    # 6. Filtrar zonas sin nombre
    df = df.dropna(subset=['zone_name'])

    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETADO")
    print(f"Zonas procesadas: {df['zone_id'].nunique():,}")
    print(f"Registros totales: {len(df):,}")
    print("=" * 60)

    return df


def get_stress_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadisticas descriptivas del stress score.
    """
    stats = df.groupby('zone_id').agg({
        'stress_score_normalized': ['min', 'max', 'mean', 'std'],
        'mean_delay_minute': ['min', 'max'],
        'records_count': 'sum'
    }).round(2)

    stats.columns = ['stress_min', 'stress_max', 'stress_mean', 'stress_std',
                     'delay_min', 'delay_max', 'total_records']

    return stats.reset_index()


if __name__ == "__main__":
    from data_extraction import extract_all_data

    data = extract_all_data()
    df_stress = build_stress_curves(data)

    print("\n" + "=" * 60)
    print("ESTADISTICAS DE STRESS")
    print("=" * 60)
    print(get_stress_statistics(df_stress).describe())
