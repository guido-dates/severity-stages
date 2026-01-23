"""
Visualization Module
====================
Funciones de visualizacion y diagnostico para zonas del Severity Model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

import sys
sys.path.append('..')
from config.settings import (
    STAGE_STRESS_PERCENTILES,
    MIN_ORDERS_PER_POINT,
    MIN_DELAY_FOR_TRUNCATION,
    MAX_MEAN_DELAY_MINUTE
)

# Colores por stage
STAGE_COLORS = {
    'low_delay': '#2ecc71',       # Verde
    'preventive': '#f1c40f',      # Amarillo
    'containment_I': '#e67e22',   # Naranja
    'containment_II': '#e74c3c',  # Rojo
    'inoperable': '#8e44ad'       # Violeta oscuro
}

STAGE_ORDER = ["low_delay", "preventive", "containment_I", "containment_II", "inoperable"]


def _get_stage_bounds() -> List[Tuple[str, float, float]]:
    bounds: List[Tuple[str, float, float]] = []
    prev = 0.0
    for stage in STAGE_ORDER:
        if stage not in STAGE_STRESS_PERCENTILES:
            raise ValueError(f"Falta percentil para stage '{stage}' en STAGE_STRESS_PERCENTILES")
        upper = float(STAGE_STRESS_PERCENTILES[stage])
        if upper <= prev:
            raise ValueError("STAGE_STRESS_PERCENTILES debe ser estrictamente creciente")
        if upper > 100:
            raise ValueError("STAGE_STRESS_PERCENTILES no puede superar 100")
        bounds.append((stage, prev, upper))
        prev = upper
    if abs(prev - 100.0) > 1e-6:
        raise ValueError("STAGE_STRESS_PERCENTILES debe terminar en 100 (cota superior de inoperable)")
    return bounds


def _get_percentile_edges() -> List[float]:
    bounds = _get_stage_bounds()
    return [0.0] + [upper for _, _, upper in bounds]


def get_zone_diagnostic_info(df_stress: pd.DataFrame, zone_id: int, min_orders: Optional[int] = None) -> Dict:
    """
    Extrae metricas diagnosticas de una zona.

    Args:
        df_stress: DataFrame con stress curves (output de build_stress_curves)
        zone_id: ID de la zona a analizar
        min_orders: Minimo de ordenes para considerar un punto valido

    Returns:
        Dict con metricas diagnosticas
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    zone_data = df_stress[df_stress['zone_id'] == zone_id].copy()

    if len(zone_data) == 0:
        return {'error': f'Zone {zone_id} not found'}

    # Info basica
    zone_name = zone_data['zone_name'].iloc[0] if 'zone_name' in zone_data.columns else 'Unknown'
    city_name = zone_data['city_name'].iloc[0] if 'city_name' in zone_data.columns else 'Unknown'

    # Separar datos validos de filtrados
    valid_data = zone_data[zone_data['total_orders_matched'] >= min_orders]
    n_valid_points = len(valid_data)
    n_total_points = len(zone_data)
    n_filtered_points = n_total_points - n_valid_points

    # Usar datos validos para estadisticas (fallback a todos si no hay suficientes)
    stats_data = valid_data if len(valid_data) >= 3 else zone_data

    # Estadisticas de stress
    stress = stats_data['stress_score_normalized']
    stress_min = stress.min()
    stress_max = stress.max()
    stress_range = stress_max - stress_min
    stress_mean = stress.mean()
    stress_std = stress.std()

    # Percentiles de stress segun config
    percentile_edges = _get_percentile_edges()
    percentile_values = {p: np.percentile(stress, p) for p in percentile_edges}

    # Estadisticas de delay
    delay = stats_data['mean_delay_minute']
    delay_min = delay.min()
    delay_max = delay.max()
    delay_range = delay_max - delay_min

    # Encontrar el pico de stress (delay donde el stress es máximo)
    max_stress_idx = stats_data['stress_score_normalized'].idxmax()
    peak_delay_raw = stats_data.loc[max_stress_idx, 'mean_delay_minute']
    peak_stress = stats_data.loc[max_stress_idx, 'stress_score_normalized']
    # Aplicar mínimo de truncado
    peak_delay = max(peak_delay_raw, MIN_DELAY_FOR_TRUNCATION)
    n_points_before_peak = len(stats_data[stats_data['mean_delay_minute'] <= peak_delay])
    n_points_after_peak = len(stats_data[stats_data['mean_delay_minute'] > peak_delay])

    # Volumen de datos
    n_records = len(zone_data)
    total_orders = zone_data['total_orders_matched'].sum() if 'total_orders_matched' in zone_data.columns else 0
    valid_orders = valid_data['total_orders_matched'].sum() if 'total_orders_matched' in valid_data.columns else 0

    # Baseline info (si existe)
    baseline_fail_rate = zone_data['baseline_fail_rate'].iloc[0] if 'baseline_fail_rate' in zone_data.columns else None
    baseline_staffing = zone_data['baseline_staffing'].iloc[0] if 'baseline_staffing' in zone_data.columns else None

    # Detectar problemas
    problems = []

    # Problema 1: Baja varianza de stress
    if stress_range < 10:
        problems.append('low_variance')

    # Problema 2: Todos los percentiles iguales o muy cercanos
    internal_edges = [p for p in percentile_edges if 0 < p < 100]
    if len(internal_edges) >= 2:
        p_spread = percentile_values[internal_edges[-1]] - percentile_values[internal_edges[0]]
    else:
        p_spread = 0
    if p_spread < 5:
        problems.append('flat_percentiles')

    # Problema 3: Pocos puntos validos
    if n_valid_points < 10:
        problems.append('insufficient_valid_points')

    # Problema 4: Delay sin variacion
    if delay_range < 5:
        problems.append('low_delay_variance')

    # Problema 5: Muchos puntos filtrados (>50% del total)
    if n_filtered_points > n_total_points * 0.5:
        problems.append('many_filtered_points')

    is_problematic = len(problems) > 0

    return {
        'zone_id': zone_id,
        'zone_name': zone_name,
        'city_name': city_name,
        'stress_min': round(stress_min, 2),
        'stress_max': round(stress_max, 2),
        'stress_range': round(stress_range, 2),
        'stress_mean': round(stress_mean, 2),
        'stress_std': round(stress_std, 2),
        'percentiles': {k: round(v, 2) for k, v in percentile_values.items()},
        'delay_min': round(delay_min, 2),
        'delay_max': round(delay_max, 2),
        'delay_range': round(delay_range, 2),
        'peak_delay': round(peak_delay, 2),
        'peak_delay_raw': round(peak_delay_raw, 2),
        'peak_stress': round(peak_stress, 2),
        'n_points_before_peak': n_points_before_peak,
        'n_points_after_peak': n_points_after_peak,
        'n_records': n_records,
        'n_valid_points': n_valid_points,
        'n_filtered_points': n_filtered_points,
        'total_orders': int(total_orders),
        'valid_orders': int(valid_orders),
        'min_orders_threshold': min_orders,
        'baseline_fail_rate': round(baseline_fail_rate, 4) if baseline_fail_rate else None,
        'baseline_staffing': round(baseline_staffing, 4) if baseline_staffing else None,
        'is_problematic': is_problematic,
        'problems': problems
    }


def plot_zone_stress_curve(
    df_stress: pd.DataFrame,
    zone_id: int,
    ax: Optional[plt.Axes] = None,
    show_percentiles: bool = True,
    thresholds: Optional[pd.Series] = None,
    title: Optional[str] = None,
    min_orders: Optional[int] = None,
    show_filtered: bool = True,
    show_peak: bool = True,
    peak_delay: Optional[float] = None,
    use_smoothed: bool = True
) -> plt.Axes:
    """
    Grafica la curva stress_score vs mean_delay para una zona.

    Si use_smoothed=True y existe la columna 'stress_smoothed', grafica:
    - Puntos originales en gris semitransparente
    - Linea de curva suavizada en azul
    - Marcadores de minimo (triangulo verde) y maximo (triangulo rojo)

    Args:
        df_stress: DataFrame con stress curves
        zone_id: ID de la zona a graficar
        ax: Axes de matplotlib (opcional)
        show_percentiles: Si mostrar lineas de percentiles de stress
        thresholds: Series con umbrales calculados para marcar en eje X
        title: Titulo personalizado
        min_orders: Minimo de ordenes para considerar un punto valido (default: MIN_ORDERS_PER_POINT)
        show_filtered: Si mostrar puntos filtrados en gris (default: True)
        show_peak: Si mostrar el punto de corte del pico de stress (default: True)
        peak_delay: Delay del pico de stress (si no se pasa, se calcula automaticamente)
        use_smoothed: Si True y existe 'stress_smoothed', grafica curva suavizada (default: True)

    Returns:
        Axes con el grafico
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    zone_data = df_stress[df_stress['zone_id'] == zone_id].copy()

    if len(zone_data) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Zone {zone_id} not found', ha='center', va='center')
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    zone_name = zone_data['zone_name'].iloc[0] if 'zone_name' in zone_data.columns else f'Zone {zone_id}'

    # Separar puntos validos de filtrados
    valid_data = zone_data[zone_data['total_orders_matched'] >= min_orders]
    filtered_data = zone_data[zone_data['total_orders_matched'] < min_orders]

    # Mostrar puntos filtrados en gris (si show_filtered=True)
    if show_filtered and len(filtered_data) > 0:
        ax.scatter(
            filtered_data['mean_delay_minute'],
            filtered_data['stress_score_normalized'],
            s=20,
            alpha=0.3,
            c='gray',
            marker='x',
            label=f'Filtrados (<{min_orders} ord): {len(filtered_data)}'
        )

    # Verificar si hay curva suavizada disponible
    has_smoothed = use_smoothed and 'stress_smoothed' in valid_data.columns

    # Scatter plot de puntos validos
    if len(valid_data) > 0:
        sizes = valid_data['total_orders_matched'] / valid_data['total_orders_matched'].max() * 150 + 20

        if has_smoothed:
            # Si hay curva suavizada: puntos originales en gris semitransparente
            scatter = ax.scatter(
                valid_data['mean_delay_minute'],
                valid_data['stress_score_normalized'],
                s=sizes,
                alpha=0.35,
                c='gray',
                edgecolors='black',
                linewidths=0.3,
                label='Datos originales'
            )
        else:
            # Sin suavizado: puntos coloreados por stress
            scatter = ax.scatter(
                valid_data['mean_delay_minute'],
                valid_data['stress_score_normalized'],
                s=sizes,
                alpha=0.7,
                c=valid_data['stress_score_normalized'],
                cmap='RdYlGn_r',
                edgecolors='black',
                linewidths=0.5
            )

        # Linea de curva suavizada (si existe)
        if has_smoothed and len(valid_data) >= 3:
            valid_sorted = valid_data.sort_values('mean_delay_minute')
            ax.plot(
                valid_sorted['mean_delay_minute'],
                valid_sorted['stress_smoothed'],
                'b-', linewidth=2.5, alpha=0.9, zorder=10,
                label='Curva suavizada'
            )

            # Marcar minimo y maximo de la curva suavizada
            if 'delay_at_min' in valid_data.columns:
                delay_min = valid_data['delay_at_min'].iloc[0]
                stress_min = valid_data['stress_smooth_min'].iloc[0]
                ax.scatter([delay_min], [stress_min],
                          color='green', s=200, marker='v', zorder=15,
                          edgecolors='black', linewidths=1.5,
                          label=f'Min: {delay_min:.0f} min')

            if 'delay_at_max' in valid_data.columns:
                delay_max = valid_data['delay_at_max'].iloc[0]
                stress_max = valid_data['stress_smooth_max'].iloc[0]
                ax.scatter([delay_max], [stress_max],
                          color='red', s=200, marker='^', zorder=15,
                          edgecolors='black', linewidths=1.5,
                          label=f'Max: {delay_max:.0f} min')

        elif len(valid_data) >= 3 and not has_smoothed:
            # Fallback: linea de tendencia polinomial si no hay suavizado
            weights = valid_data['total_orders_matched'].values
            x = valid_data['mean_delay_minute'].values
            y = valid_data['stress_score_normalized'].values

            coeffs = np.polyfit(x, y, 2, w=weights)
            p = np.poly1d(coeffs)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'b-', alpha=0.8, linewidth=2, label='Tendencia (ponderada)')

    # Lineas horizontales de percentiles de stress (usando solo datos validos)
    if show_percentiles and len(valid_data) > 0:
        stress = valid_data['stress_score_normalized']
        percentile_edges = _get_percentile_edges()
        internal_edges = [p for p in percentile_edges if 0 < p < 100]
        percentile_colors = [STAGE_COLORS[stage] for stage in STAGE_ORDER[1:1 + len(internal_edges)]]

        for i, p_val in enumerate(internal_edges):
            p_stress = np.percentile(stress, p_val)
            color = percentile_colors[i] if i < len(percentile_colors) else 'gray'
            ax.axhline(y=p_stress, color=color, linestyle=':', alpha=0.7,
                      label=f'P{int(p_val)}={p_stress:.1f}')

    # Lineas verticales para umbrales (si se pasan)
    if thresholds is not None:
        for stage, color in STAGE_COLORS.items():
            if stage in thresholds.index:
                ax.axvline(x=thresholds[stage], color=color, linestyle='--', alpha=0.8,
                          linewidth=2, label=f'{stage}={thresholds[stage]:.0f}')

    # Mostrar el punto de corte del pico de stress
    if show_peak and len(valid_data) > 0:
        # Calcular peak_delay si no se paso
        if peak_delay is None:
            max_idx = valid_data['stress_score_normalized'].idxmax()
            peak_delay_calc = valid_data.loc[max_idx, 'mean_delay_minute']
            peak_delay_calc = max(peak_delay_calc, MIN_DELAY_FOR_TRUNCATION)
        else:
            peak_delay_calc = peak_delay

        # Linea vertical roja punteada en el pico
        ax.axvline(x=peak_delay_calc, color='red', linestyle=':', alpha=0.9,
                  linewidth=2.5, label=f'Corte pico={peak_delay_calc:.0f}')

        # Sombrear area despues del pico (zona ignorada)
        ylims = ax.get_ylim()
        max_x = MAX_MEAN_DELAY_MINUTE if MAX_MEAN_DELAY_MINUTE is not None else ax.get_xlim()[1]
        max_x = max_x if max_x else 60
        ax.axvspan(peak_delay_calc, max_x,
                  alpha=0.1, color='gray', label='Ignorado (post-pico)')

    ax.set_xlabel('Mean Delay (minutos)', fontsize=11)
    ax.set_ylabel('Stress Score Normalizado (0-100)', fontsize=11)

    n_valid = len(valid_data)
    n_total = len(zone_data)
    subtitle = f'({n_valid}/{n_total} puntos con >={min_orders} ordenes)'

    if title:
        ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Curva de Stress: {zone_name}\n{subtitle}', fontsize=12, fontweight='bold')

    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    if MAX_MEAN_DELAY_MINUTE is not None:
        ax.set_xlim(0, MAX_MEAN_DELAY_MINUTE)
    else:
        ax.set_xlim(0, None)
    ax.set_ylim(-5, 105)

    return ax


def plot_zone_stress_distribution(
    df_stress: pd.DataFrame,
    zone_id: int,
    ax: Optional[plt.Axes] = None,
    plot_type: str = 'hist',
    min_orders: Optional[int] = None
) -> plt.Axes:
    """
    Histograma o KDE del stress_score_normalized para una zona.

    Args:
        df_stress: DataFrame con stress curves
        zone_id: ID de la zona
        ax: Axes de matplotlib (opcional)
        plot_type: 'hist', 'kde', o 'both'
        min_orders: Minimo de ordenes para filtrar puntos

    Returns:
        Axes con el grafico
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    zone_data = df_stress[df_stress['zone_id'] == zone_id].copy()

    if len(zone_data) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Zone {zone_id} not found', ha='center', va='center')
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Filtrar por ordenes
    valid_data = zone_data[zone_data['total_orders_matched'] >= min_orders]
    if len(valid_data) < 3:
        valid_data = zone_data  # Fallback

    zone_name = zone_data['zone_name'].iloc[0] if 'zone_name' in zone_data.columns else f'Zone {zone_id}'
    stress = valid_data['stress_score_normalized']

    # Plot segun tipo
    if plot_type == 'hist':
        ax.hist(stress, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    elif plot_type == 'kde':
        stress.plot.kde(ax=ax, color='steelblue', linewidth=2)
    else:  # both
        ax.hist(stress, bins=30, color='steelblue', alpha=0.5, edgecolor='black', density=True)
        if len(stress) > 1:
            stress.plot.kde(ax=ax, color='darkblue', linewidth=2)

    # Lineas verticales en los percentiles (segun config)
    percentile_edges = _get_percentile_edges()
    internal_edges = [p for p in percentile_edges if 0 < p < 100]
    percentile_colors = [STAGE_COLORS[stage] for stage in STAGE_ORDER[1:1 + len(internal_edges)]]

    for i, p_val in enumerate(internal_edges):
        p_stress = np.percentile(stress, p_val)
        color = percentile_colors[i] if i < len(percentile_colors) else 'gray'
        ax.axvline(x=p_stress, color=color, linestyle='--', linewidth=2,
                  label=f'P{int(p_val)}={p_stress:.1f}')

    ax.set_xlabel('Stress Score Normalizado', fontsize=11)
    ax.set_ylabel('Frecuencia' if plot_type == 'hist' else 'Densidad', fontsize=11)
    ax.set_title(f'Distribucion de Stress: {zone_name}\n({len(valid_data)} puntos con >={min_orders} ord)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_delay_by_stress_bucket(
    df_stress: pd.DataFrame,
    zone_id: int,
    ax: Optional[plt.Axes] = None,
    min_orders: Optional[int] = None
) -> plt.Axes:
    """
    Barras de delay promedio por bucket de stress.

    Args:
        df_stress: DataFrame con stress curves
        zone_id: ID de la zona
        ax: Axes de matplotlib (opcional)
        min_orders: Minimo de ordenes para filtrar puntos

    Returns:
        Axes con el grafico
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    zone_data = df_stress[df_stress['zone_id'] == zone_id].copy()

    if len(zone_data) == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Zone {zone_id} not found', ha='center', va='center')
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Filtrar por ordenes
    valid_data = zone_data[zone_data['total_orders_matched'] >= min_orders]
    if len(valid_data) < 3:
        valid_data = zone_data  # Fallback

    zone_name = zone_data['zone_name'].iloc[0] if 'zone_name' in zone_data.columns else f'Zone {zone_id}'
    stress = valid_data['stress_score_normalized']

    # Calcular percentiles usando datos validos (segun config)
    percentile_edges = _get_percentile_edges()
    percentiles = {p: np.percentile(stress, p) for p in percentile_edges}
    stage_bounds = _get_stage_bounds()

    label_map = {
        "low_delay": "low_delay",
        "preventive": "preventive",
        "containment_I": "cont_I",
        "containment_II": "cont_II",
        "inoperable": "inoperable"
    }

    # Definir buckets segun config
    bucket_ranges = [
        (f"P{int(low)}-P{int(high)}\n({label_map.get(stage, stage)})", low, high)
        for stage, low, high in stage_bounds
    ]

    bucket_delays = []
    bucket_counts = []
    bucket_orders = []
    colors = [STAGE_COLORS[stage] for stage, _, _ in stage_bounds]

    for label, p_low, p_high in bucket_ranges:
        stress_min = percentiles[p_low]
        stress_max = percentiles[p_high]

        bucket_data = valid_data[
            (valid_data['stress_score_normalized'] >= stress_min) &
            (valid_data['stress_score_normalized'] < stress_max)
        ]

        if len(bucket_data) > 0 and 'total_orders_matched' in bucket_data.columns:
            total_orders = bucket_data['total_orders_matched'].sum()
            if total_orders > 0:
                weighted_delay = (bucket_data['mean_delay_minute'] * bucket_data['total_orders_matched']).sum() / total_orders
            else:
                weighted_delay = bucket_data['mean_delay_minute'].mean()
        else:
            weighted_delay = bucket_data['mean_delay_minute'].mean() if len(bucket_data) > 0 else 0
            total_orders = 0

        bucket_delays.append(weighted_delay)
        bucket_counts.append(len(bucket_data))
        bucket_orders.append(total_orders)

    # Grafico de barras
    x = range(len(bucket_ranges))
    bars = ax.bar(x, bucket_delays, color=colors, edgecolor='black', alpha=0.8)

    # Etiquetas en las barras
    for i, (bar, count, orders) in enumerate(zip(bars, bucket_counts, bucket_orders)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}\n({count} pts, {orders:,} ord)',
               ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in bucket_ranges], fontsize=9)
    ax.set_xlabel('Bucket de Stress', fontsize=11)
    ax.set_ylabel('Delay Promedio Ponderado (min)', fontsize=11)
    ax.set_title(f'Delay por Bucket: {zone_name}\n(solo puntos con >={min_orders} ord)',
                fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def plot_zone_diagnostic_panel(
    df_stress: pd.DataFrame,
    zone_id: int,
    thresholds: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 10),
    min_orders: Optional[int] = None
) -> plt.Figure:
    """
    Panel completo de diagnostico para UNA zona (4 subplots).

    Layout 2x2:
    - [0,0] Curva stress vs delay
    - [0,1] Distribucion del stress
    - [1,0] Barras de delay por bucket
    - [1,1] Info diagnostica

    Args:
        df_stress: DataFrame con stress curves
        zone_id: ID de la zona
        thresholds: Series con umbrales (opcional)
        figsize: Tamano de la figura
        min_orders: Minimo de ordenes para filtrar puntos

    Returns:
        Figure de matplotlib
    """
    if min_orders is None:
        min_orders = MIN_ORDERS_PER_POINT

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Obtener info diagnostica
    diag_info = get_zone_diagnostic_info(df_stress, zone_id, min_orders=min_orders)

    if 'error' in diag_info:
        fig.text(0.5, 0.5, diag_info['error'], ha='center', va='center', fontsize=14)
        return fig

    # [0,0] Curva stress vs delay
    plot_zone_stress_curve(df_stress, zone_id, ax=axes[0, 0], thresholds=thresholds, min_orders=min_orders)

    # [0,1] Distribucion del stress
    plot_zone_stress_distribution(df_stress, zone_id, ax=axes[0, 1], plot_type='both', min_orders=min_orders)

    # [1,0] Barras de delay por bucket
    plot_delay_by_stress_bucket(df_stress, zone_id, ax=axes[1, 0], min_orders=min_orders)

    # [1,1] Info diagnostica (texto)
    ax_info = axes[1, 1]
    ax_info.axis('off')

    # Construir texto de diagnostico
    status_emoji = "PROBLEMATICA" if diag_info['is_problematic'] else "OK"

    percentile_edges = sorted(p for p in diag_info['percentiles'].keys() if 0 < p < 100)
    percentile_lines = []
    for i in range(0, len(percentile_edges), 2):
        left = percentile_edges[i]
        left_val = diag_info['percentiles'][left]
        if i + 1 < len(percentile_edges):
            right = percentile_edges[i + 1]
            right_val = diag_info['percentiles'][right]
            percentile_lines.append(f"  P{int(left)}: {left_val:.1f}  |  P{int(right)}: {right_val:.1f}")
        else:
            percentile_lines.append(f"  P{int(left)}: {left_val:.1f}")
    percentile_text = "\n".join(percentile_lines) if percentile_lines else "  (sin percentiles internos)"

    info_text = f"""
DIAGNOSTICO: {diag_info['zone_name']} ({diag_info['city_name']})
{'='*50}

ESTADO: {status_emoji}
{f"Problemas: {', '.join(diag_info['problems'])}" if diag_info['problems'] else "Sin problemas detectados"}

FILTRO DE ORDENES (min={min_orders})
  Puntos validos: {diag_info['n_valid_points']} de {diag_info['n_records']}
  Puntos filtrados: {diag_info['n_filtered_points']}
  Ordenes en puntos validos: {diag_info['valid_orders']:,}

STRESS SCORE (solo puntos validos)
  Rango: {diag_info['stress_min']:.1f} - {diag_info['stress_max']:.1f} (span: {diag_info['stress_range']:.1f})
  Media: {diag_info['stress_mean']:.1f}  |  Std: {diag_info['stress_std']:.1f}

PERCENTILES DE STRESS (segun config)
{percentile_text}

DELAY
  Rango: {diag_info['delay_min']:.1f} - {diag_info['delay_max']:.1f} min

TRUNCADO (pico de stress)
  Pico real en delay: {diag_info['peak_delay_raw']:.1f} min
  Corte usado: {diag_info['peak_delay']:.1f} min
  Puntos antes del corte: {diag_info['n_points_before_peak']}
  Puntos ignorados: {diag_info['n_points_after_peak']}
"""

    if thresholds is not None:
        info_text += f"""
UMBRALES CALCULADOS
  low_delay: {thresholds.get('low_delay', 'N/A')}
  preventive: {thresholds.get('preventive', 'N/A')}
  containment_I: {thresholds.get('containment_I', 'N/A')}
  containment_II: {thresholds.get('containment_II', 'N/A')}
  inoperable: {thresholds.get('inoperable', 'N/A')}
"""

    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.suptitle(f'Panel de Diagnostico: {diag_info["zone_name"]}',
                 fontsize=14, fontweight='bold', y=1.02)

    return fig


def identify_problematic_zones(
    df_stress: pd.DataFrame,
    df_thresholds: pd.DataFrame,
    min_stress_range: float = 10.0,
    min_records: int = 50
) -> pd.DataFrame:
    """
    Identifica zonas problematicas y clasifica el tipo de problema.

    Criterios:
    - stress_range < min_stress_range (baja varianza)
    - umbrales no monotonicos
    - umbrales todos iguales
    - pocos registros

    Args:
        df_stress: DataFrame con stress curves
        df_thresholds: DataFrame con umbrales calculados
        min_stress_range: Minimo rango de stress para considerar OK
        min_records: Minimo de registros para considerar OK

    Returns:
        DataFrame con zone_id, zone_name, problem_type, details
    """
    problems = []

    for zone_id in df_stress['zone_id'].unique():
        diag = get_zone_diagnostic_info(df_stress, zone_id)

        if 'error' in diag:
            continue

        zone_problems = []
        details = []

        # Check 1: Baja varianza de stress
        if diag['stress_range'] < min_stress_range:
            zone_problems.append('low_stress_variance')
            details.append(f"stress_range={diag['stress_range']:.1f}")

        # Check 2: Pocos registros
        if diag['n_records'] < min_records:
            zone_problems.append('insufficient_data')
            details.append(f"n_records={diag['n_records']}")

        # Check 3: Umbrales del df_thresholds
        if df_thresholds is not None and len(df_thresholds) > 0:
            zone_thresh = df_thresholds[df_thresholds['zone_id'] == zone_id]

            if len(zone_thresh) > 0:
                thresh_row = zone_thresh.iloc[0]
                stages = ['low_delay', 'preventive', 'containment_I', 'containment_II', 'inoperable']

                # Check: todos iguales
                thresh_values = [thresh_row.get(s, 0) for s in stages if s in thresh_row.index]
                if len(set(thresh_values)) == 1:
                    zone_problems.append('all_thresholds_equal')
                    details.append(f"all={thresh_values[0]}")

                # Check: no monotonicos
                for i in range(1, len(stages)):
                    if stages[i] in thresh_row.index and stages[i-1] in thresh_row.index:
                        if thresh_row[stages[i]] < thresh_row[stages[i-1]]:
                            zone_problems.append('non_monotonic')
                            details.append(f"{stages[i-1]}>{stages[i]}")
                            break

        if zone_problems:
            problems.append({
                'zone_id': zone_id,
                'zone_name': diag['zone_name'],
                'city_name': diag['city_name'],
                'problems': ', '.join(zone_problems),
                'details': '; '.join(details),
                'stress_range': diag['stress_range'],
                'n_records': diag['n_records']
            })

    df_problems = pd.DataFrame(problems)

    if len(df_problems) > 0:
        df_problems = df_problems.sort_values('stress_range', ascending=True)

    return df_problems


def plot_multiple_zones_comparison(
    df_stress: pd.DataFrame,
    zone_ids: List[int],
    ncols: int = 3,
    plot_type: str = 'stress_curve',
    figsize_per_plot: Tuple[int, int] = (5, 4)
) -> plt.Figure:
    """
    Grid de graficos comparando multiples zonas.

    Args:
        df_stress: DataFrame con stress curves
        zone_ids: Lista de zone_id a comparar
        ncols: Numero de columnas en el grid
        plot_type: 'stress_curve', 'distribution', o 'delay_buckets'
        figsize_per_plot: Tamano de cada subplot

    Returns:
        Figure de matplotlib
    """
    n_zones = len(zone_ids)
    nrows = (n_zones + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    # Flatten axes para iterar facilmente
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    for i, zone_id in enumerate(zone_ids):
        ax = axes_flat[i]

        if plot_type == 'stress_curve':
            plot_zone_stress_curve(df_stress, zone_id, ax=ax, show_percentiles=False)
        elif plot_type == 'distribution':
            plot_zone_stress_distribution(df_stress, zone_id, ax=ax, plot_type='hist')
        elif plot_type == 'delay_buckets':
            plot_delay_by_stress_bucket(df_stress, zone_id, ax=ax)

    # Ocultar axes vacios
    for i in range(n_zones, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    fig.suptitle(f'Comparacion de {n_zones} Zonas', fontsize=14, fontweight='bold', y=1.02)

    return fig


def plot_problematic_zones_summary(
    df_problems: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Grafico resumen de zonas problematicas.

    Args:
        df_problems: DataFrame de identify_problematic_zones()
        figsize: Tamano de la figura

    Returns:
        Figure de matplotlib
    """
    if len(df_problems) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No hay zonas problematicas detectadas',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # [0] Barras por tipo de problema
    problem_counts = df_problems['problems'].str.split(', ').explode().value_counts()

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db']
    axes[0].barh(range(len(problem_counts)), problem_counts.values, color=colors[:len(problem_counts)])
    axes[0].set_yticks(range(len(problem_counts)))
    axes[0].set_yticklabels(problem_counts.index)
    axes[0].set_xlabel('Cantidad de Zonas')
    axes[0].set_title('Zonas por Tipo de Problema')
    axes[0].invert_yaxis()

    # Agregar conteos en las barras
    for i, v in enumerate(problem_counts.values):
        axes[0].text(v + 0.1, i, str(v), va='center')

    # [1] Scatter stress_range vs n_records
    scatter = axes[1].scatter(
        df_problems['n_records'],
        df_problems['stress_range'],
        s=100,
        c='red',
        alpha=0.6,
        edgecolors='black'
    )

    # Etiquetas para algunas zonas
    for idx, row in df_problems.head(5).iterrows():
        axes[1].annotate(
            row['zone_name'][:15],
            (row['n_records'], row['stress_range']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Min stress_range')
    axes[1].axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Min records')
    axes[1].set_xlabel('Cantidad de Registros')
    axes[1].set_ylabel('Rango de Stress')
    axes[1].set_title('Zonas Problematicas: Records vs Stress Range')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.suptitle(f'Resumen: {len(df_problems)} Zonas Problematicas',
                fontsize=14, fontweight='bold', y=1.02)

    return fig
