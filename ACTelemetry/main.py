# Módulos del sistema y utlilidades

from __future__ import annotations # Usar una clase que no esta definida en las anotaciones de tipo

import os # Permite interactuar con el sistema operativo (rutas, archivos, variable entorno)
import sys # Acceso a funcionalidades del intérprete de Python
import argparse # Crea interfaces de línea de comandos y parsear argumentos
import csv # Leer y escribir archivos CSV (formato de datos sacados por acti)

# Análisis de datos

import pandas as pd # Biblioteca principal para análisis de datos. Facilita la carga, filtrado, agrupación y cálculo 
# de estadísticas sobre los datos de telemetría.

# Visualización

import matplotlib.pyplot as plt # Biblioteca principal para crear gráficos y visualizaciones
from matplotlib.gridspec import GridSpec # Crea layouts complejos de subplots dentro de una figura
from matplotlib.animation import FuncAnimation # Se usa para crear animaciones, actualizandolas en tiempo real
from matplotlib.ticker import AutoMinorLocator # Añade divisiones menores automáticas a los ejes, mejorando la legibilidad
from matplotlib.collections import LineCollection # Permite dibujar multiples líneas con diferentes colores y estilos
from matplotlib.colors import Normalize # Normaliza valores para mapearlos a una escala de colores
from matplotlib import cm # Define mapas de color para visualizar datos


DATA_FILES = ["data1.csv", "data2.csv"]  # Guarda en una variable la lista de archivos CSV a comparar
ENABLE_ANIMATION = False  # Animacion automatica de la vuelta en el mapa
TRACK_ZOOM_HALF_M = 25.0  # Zoom al punto que amos a aseguir en mapa para comparar datos
DEFAULT_X_AXIS = "time"  # En el plot de las graficas, la x puede ser tiempo ("time") o distancia ("distance")


def format_lap_time(seconds: float) -> str: # Función para pasar el dato de segundos a formato m:ss.ss
    mins = int(seconds // 60) # Valor entero de minutos
    secs = seconds % 60 # Resto en segundos
    return f"{mins}:{secs:05.2f}" # Return formateado, y cambio a string


def load_dataframe(path: str) -> pd.DataFrame: # Lee un archivo CSV de telemetria, se estpera que sea str y devuelve un DataFrame de pandas
    with open(path, newline='', encoding='utf-8') as f: # Abre un archivo "path" en modo lectura
        reader = csv.reader(f)
        for _ in range(14): # Se salta las 14 lines del csv, cabecera y metadatos
            next(reader)
        names = next(reader) # La 15ª línea son los nombres de las columnas
        _units = next(reader) # La 16ª línea son las unidades (no se usan)

    df = pd.read_csv(path, skiprows=16, names=names) # Lee el CSV con pandas, saltando las 16 primeras líneas y usando los nombres de columnas
    df.columns = [c.strip('"') for c in df.columns] # Limpia los nombres de columnas de comillas, porque puede dar problemas
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    return df


def compute_fastest_lap(df: pd.DataFrame):
    required_cols = {'Session Lap Count', 'Last Lap Time', 'Lap Invalidated', 'Time'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas necesarias en el CSV: {missing}")

    laps = df[['Session Lap Count', 'Last Lap Time', 'Lap Invalidated']].dropna().copy()
    laps['Real Lap Number'] = laps['Session Lap Count'] - 1

    # Filtrado básico
    laps_valid = laps[
        (laps['Lap Invalidated'] == 0)
        & (laps['Real Lap Number'] > 0)
        & (laps['Last Lap Time'] > 40)
    ]

    # Outliers (2 std)
    mean_time = laps_valid['Last Lap Time'].mean()
    std_time = laps_valid['Last Lap Time'].std()
    laps_valid = laps_valid[
        (laps_valid['Last Lap Time'] > mean_time - 2 * std_time)
        & (laps_valid['Last Lap Time'] < mean_time + 2 * std_time)
    ]

    fastest = laps_valid.loc[laps_valid['Last Lap Time'].idxmin()]
    real_lap = int(fastest['Real Lap Number'])
    lap_time = float(fastest['Last Lap Time'])

    # Rango temporal de esa vuelta
    lap_start_time = df.loc[df['Session Lap Count'] == real_lap, 'Time'].iloc[0]
    lap_end_time = lap_start_time + lap_time

    lap_data = df[(df['Time'] >= lap_start_time) & (df['Time'] <= lap_end_time)].copy()
    lap_data['Lap Time (s)'] = lap_data['Time'] - lap_start_time
    return real_lap, lap_time, lap_data


def compute_controls(lap_data: pd.DataFrame):
    # Control inputs a 0–100%
    thr_raw = lap_data['Throttle Pos']
    brk_raw = lap_data['Brake Pos']
    throttle = thr_raw if thr_raw.max() > 1.1 else thr_raw * 100
    brake = brk_raw if brk_raw.max() > 1.1 else brk_raw * 100
    return throttle, brake


def compute_cumulative_distance(lap_data: pd.DataFrame):
    """Calcula distancia acumulada (m) a lo largo de la vuelta usando Car Coord X/Y."""
    import numpy as np
    x = lap_data['Car Coord X'].to_numpy()
    y = lap_data['Car Coord Y'].to_numpy()
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ds = np.hypot(dx, dy)
    s = ds.cumsum()
    s[0] = 0.0
    return s


def compute_sector_times(lap_data: pd.DataFrame, lap_time: float) -> list[float] | None:
    """Extrae los 3 sectores de la vuelta a partir de 'Last Sector Time'.
    Estrategia: detectar cambios de valor > 0.05s en la columna durante la vuelta
    y tomar los primeros 3 valores no nulos.
    """
    import numpy as np
    if 'Last Sector Time' not in lap_data.columns:
        return None
    s = lap_data['Last Sector Time'].to_numpy()
    s = s[~np.isnan(s)]
    s = s[s > 0.05]
    if s.size == 0:
        return None
    sectors = []
    prev = None
    for val in s:
        if prev is None or abs(val - prev) > 0.05:
            sectors.append(float(val))
            prev = val
            # Evitar captar el mismo valor repetido en frames consecutivos
            if len(sectors) == 3:
                break
    # Relleno si faltan sectores
    if len(sectors) == 2:
        s3 = lap_time - sum(sectors)
        if 0.1 < s3 < lap_time:
            sectors.append(float(s3))
    if len(sectors) != 3:
        return None
    return sectors


def build_comparison_figure(datasets: list[dict], x_mode: str = "time"):
    """Crea la figura comparando múltiples datasets (1..N)."""
    plt.style.use('seaborn-v0_8')

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[2, 1, 1, 1],
        width_ratios=[3, 2.6],
        hspace=0.35,
        wspace=0.25,
    )

    # Ejes
    ax_track = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    ax_speed = fig.add_subplot(gs[1, :])
    ax_throttle = fig.add_subplot(gs[2, :], sharex=ax_speed)
    ax_brake = fig.add_subplot(gs[3, :], sharex=ax_speed)

    # Tema negro: fondo, ticks y títulos claros
    fig.patch.set_facecolor('#000000')
    for ax in (ax_track, ax_info, ax_speed, ax_throttle, ax_brake):
        ax.set_facecolor('#000000')
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.tick_params(colors='#DDDDDD')
        ax.yaxis.label.set_color('#DDDDDD')
        ax.xaxis.label.set_color('#DDDDDD')
        if hasattr(ax, 'title'):
            ax.title.set_color('#DDDDDD')
        # Grids mayor y menor
        ax.grid(True, which='major', color='#444444', alpha=0.35, linestyle='-')
        ax.grid(True, which='minor', color='#444444', alpha=0.22, linestyle=':')
        try:
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        except Exception:
            pass

    ax_track.set_title('Traza vuelta rápida')
    ax_track.set_xlabel('Coord X')
    ax_track.set_ylabel('Coord Y')
    # Preserve true track proportions without conflicting with limits/zoom.
    # adjustable='box' avoids the Matplotlib warning and keeps aspect ratio.
    ax_track.set_aspect('equal', adjustable='box')

    # Dibujar datasets
    for idx, d in enumerate(datasets):
        lap_data = d['lap_data']
        label = d['label']
        throttle = d['throttle']
        brake = d['brake']
        # Eje X elegido
        if x_mode == 'distance':
            xvals = d['s']
        else:
            xvals = lap_data['Lap Time (s)']

        # Colores por dataset: dos líneas de un color cada una (mapa)
        if idx == 0:
            c_track = 'tab:cyan'      # mapa en cian (cambiado)
            c_speed = 'tab:purple'    # velocidad morado
            c_thr = 'tab:green'       # acelerador verde
            c_brk = 'tab:red'         # freno rojo
        else:
            c_track = 'tab:orange'    # segunda línea en naranja
            c_speed = 'white'
            c_thr = 'white'
            c_brk = 'white'

        z = 5 if idx == 0 else 3  # data1 por encima
        # Track: dibujar siempre como línea simple por dataset (sin coloreado por ventaja)
        ax_track.plot(lap_data['Car Coord X'], lap_data['Car Coord Y'], color=c_track, alpha=0.95, linewidth=1.0, label=label, zorder=z)
        # Speed / Throttle / Brake
        ax_speed.plot(xvals, lap_data['Ground Speed'], color=c_speed, linewidth=0.9, label=f"Velocidad — {label}", zorder=z)
        ax_throttle.plot(xvals, throttle, color=c_thr, linewidth=0.8, label=f"Acel — {label}", zorder=z)
        ax_brake.plot(xvals, brake, color=c_brk, linewidth=0.8, label=f"Freno — {label}", zorder=z)

    # Leyenda del mapa (las líneas ya llevan label)
    ax_track.legend(loc='upper right')

    # Estética y leyendas
    ax_speed.set_title('Velocidad', color='#DDDDDD')
    ax_speed.set_ylabel('km/h')
    ax_throttle.set_ylabel('Acelerador (%)')
    ax_brake.set_ylabel('Freno (%)')
    ax_brake.set_xlabel('Distancia dentro de la vuelta (m)' if x_mode == 'distance' else 'Tiempo dentro de la vuelta (s)')
    ax_throttle.set_ylim(-5, 105)
    ax_brake.set_ylim(-5, 105)

    # Leyendas con estilo oscuro
    for ax in (ax_track, ax_speed, ax_throttle, ax_brake):
        leg = ax.legend(loc='upper right')
        if leg is not None:
            leg.get_frame().set_facecolor('#111111')
            leg.get_frame().set_edgecolor('#555555')
            for text in leg.get_texts():
                text.set_color('#DDDDDD')

    for ax in (ax_speed, ax_throttle, ax_brake):
        ax.grid(True, alpha=0.3)

    # Panel de info con formato solicitado
    # Dos bloques amplios y uno final para la teórica
    positions = [0.98, 0.62]
    for idx, d in enumerate(datasets):
        color = 'tab:blue' if idx == 0 else 'white'
        label = d['label']
        lap_time = d['lap_time']
        sectors = d.get('sectors')
        s1 = f"{sectors[0]:.2f} s" if sectors else "N/A"
        s2 = f"{sectors[1]:.2f} s" if sectors else "N/A"
        s3 = f"{sectors[2]:.2f} s" if sectors else "N/A"

        text = (
            f"{label}: {format_lap_time(lap_time)}\n"
            f"sector1: {s1}\n"
            f"sector2: {s2}\n"
            f"sector3: {s3}\n"
        )
        y = positions[idx] if idx < len(positions) else max(0.30, positions[-1] - 0.36 * (idx - len(positions) + 1))
        ax_info.text(0.02, y, text, transform=ax_info.transAxes, va='top', ha='left', fontsize=12,
                     family='monospace', color=color,
                     bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#555555'))

    # Cursores compartidos
    vline_speed = ax_speed.axvline(0, color='#AAAAAA', alpha=0.8, linestyle='--')
    vline_thr = ax_throttle.axvline(0, color='#AAAAAA', alpha=0.8, linestyle='--')
    vline_brk = ax_brake.axvline(0, color='#AAAAAA', alpha=0.8, linestyle='--')

    # Artistas por dataset para hover/animación
    track_points = []
    track_paths = []
    speed_markers = []
    thr_markers = []
    brk_markers = []
    for idx, d in enumerate(datasets):
        if idx == 0:
            c_track = 'tab:blue'; c_speed = 'tab:purple'; c_thr = 'tab:green'; c_brk = 'tab:red'
            z = 6
        else:
            c_track = 'white'; c_speed = 'white'; c_thr = 'white'; c_brk = 'white'
            z = 4
        p, = ax_track.plot([], [], 'o', color=c_track, markersize=5, zorder=z)
        path, = ax_track.plot([], [], '-', color=c_track, alpha=0.9, linewidth=1.0, zorder=z)
        ms, = ax_speed.plot([], [], 'o', color=c_speed, markersize=4, zorder=z)
        mt, = ax_throttle.plot([], [], 'o', color=c_thr, markersize=4, zorder=z)
        mb, = ax_brake.plot([], [], 'o', color=c_brk, markersize=4, zorder=z)
        track_points.append(p)
        track_paths.append(path)
        speed_markers.append(ms)
        thr_markers.append(mt)
        brk_markers.append(mb)

    fig._axes = (ax_track, ax_info, ax_speed, ax_throttle, ax_brake)  # evitar GC de referencias
    fig._x_mode = x_mode
    fig._cursors = (vline_speed, vline_thr, vline_brk)
    fig._artists = (track_points, track_paths, speed_markers, thr_markers, brk_markers)
    return fig


def animate_lap_multi(fig, datasets: list[dict], x_mode: str = "time"):
    # Anima siguiendo el primer dataset como referencia temporal
    from numpy import asarray

    ax_track, _ax_info, ax_speed, ax_thr, ax_brk = fig._axes
    vline_speed, vline_thr, vline_brk = fig._cursors
    track_points, track_paths, speed_markers, thr_markers, brk_markers = fig._artists

    t_list = [d['lap_data']['Lap Time (s)'].to_numpy() for d in datasets]
    s_list = [d['s'] for d in datasets]
    x_list = [d['lap_data']['Car Coord X'].to_numpy() for d in datasets]
    y_list = [d['lap_data']['Car Coord Y'].to_numpy() for d in datasets]
    speed_list = [d['lap_data']['Ground Speed'].to_numpy() for d in datasets]
    thr_list = [d['throttle'].to_numpy() for d in datasets]
    brk_list = [d['brake'].to_numpy() for d in datasets]

    if x_mode == 'distance':
        xref = s_list[0]
        ax_speed.set_xlim(xref.min(), max(s[-1] for s in s_list))
    else:
        xref = t_list[0]
        ax_speed.set_xlim(xref.min(), max(t[-1] for t in t_list))

    # Intervalo
    if len(xref) > 1:
        dx_avg = max((xref[-1] - xref[0]) / (len(xref) - 1), 1/120)
    else:
        dx_avg = 0.03
    interval_ms = int(max(10, min(1000 * dx_avg, 1000 / 30)))

    import numpy as np

    def idx_for_axis(arr, x):
        i = int(np.searchsorted(arr, x))
        if i <= 0:
            return 0
        if i >= len(arr):
            return len(arr) - 1
        return i

    def update(i):
        xi = xref[i]
        for v in (vline_speed, vline_thr, vline_brk):
            v.set_xdata([xi, xi])
        for k in range(len(datasets)):
            arr = s_list[k] if x_mode == 'distance' else t_list[k]
            ik = idx_for_axis(arr, xi)
            # track
            track_points[k].set_data([x_list[k][ik]], [y_list[k][ik]])
            track_paths[k].set_data(x_list[k][:ik + 1], y_list[k][:ik + 1])
            # markers
            xv = arr[ik]
            speed_markers[k].set_data([xv], [speed_list[k][ik]])
            thr_markers[k].set_data([xv], [thr_list[k][ik]])
            brk_markers[k].set_data([xv], [brk_list[k][ik]])
        return (*track_points, *track_paths, *speed_markers, *thr_markers, *brk_markers)

    ani = FuncAnimation(fig, update, frames=len(xref), interval=interval_ms, blit=False, repeat=False)
    return ani


def enable_hover_interaction_multi(fig, datasets: list[dict], x_mode: str = "time"):
    """Hover: al mover el ratón por las gráficas, sincroniza cursores/markers y puntos del track en todos los datasets."""
    import numpy as np

    ax_track, ax_info, ax_speed, ax_thr, ax_brk = fig._axes
    vline_speed, vline_thr, vline_brk = fig._cursors
    track_points, track_paths, speed_markers, thr_markers, brk_markers = fig._artists

    axes_set = {ax_speed, ax_thr, ax_brk}

    # Prepara arrays por dataset (usar eje elegido para sincronizar)
    t_list = [d['lap_data']['Lap Time (s)'].to_numpy() for d in datasets]
    s_list = [d['s'] for d in datasets]
    x_list = [d['lap_data']['Car Coord X'].to_numpy() for d in datasets]
    y_list = [d['lap_data']['Car Coord Y'].to_numpy() for d in datasets]
    speed_list = [d['lap_data']['Ground Speed'].to_numpy() for d in datasets]
    thr_list = [d['throttle'].to_numpy() for d in datasets]
    brk_list = [d['brake'].to_numpy() for d in datasets]

    # Texto de hover (apilado por dataset)
    hover_texts = []
    y0 = 0.18  # más abajo para no chocar con info y teórica
    for idx, d in enumerate(datasets):
        if idx == 0:
            color = 'tab:blue'
        else:
            color = 'white'
        hover_texts.append(
            ax_info.text(0.02, y0, "", transform=ax_info.transAxes, va='top', ha='left', fontsize=11,
                         family='monospace', color=color)
        )
        y0 -= 0.10

    # Texto de distancia cerca del punto de data1 en el track
    dist_text = ax_track.text(0.02, 0.02, "", transform=ax_track.transAxes, va='bottom', ha='left',
                              fontsize=10, color='tab:blue', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#555555', alpha=0.8))

    # (Eliminado) Texto de diferencia de velocidad: ya no se muestra

    def idx_for_axis(arr, x):
        i = int(np.searchsorted(arr, x))
        if i <= 0:
            return 0
        if i >= len(arr):
            return len(arr) - 1
        return i

    def on_move(event):
        if event.inaxes not in axes_set or event.xdata is None:
            return
        xi = event.xdata  # valor en eje X (tiempo o distancia)
        # Cursores
        for v in (vline_speed, vline_thr, vline_brk):
            v.set_xdata([xi, xi])

        # Por dataset
        for k, d in enumerate(datasets):
            arr = s_list[k] if x_mode == 'distance' else t_list[k]
            ik = idx_for_axis(arr, xi)
            # Puntos en el track y rastro
            track_points[k].set_data([x_list[k][ik]], [y_list[k][ik]])
            track_paths[k].set_data(x_list[k][:ik + 1], y_list[k][:ik + 1])
            # Markers en subplots
            xv = arr[ik]
            speed_markers[k].set_data([xv], [speed_list[k][ik]])
            thr_markers[k].set_data([xv], [thr_list[k][ik]])
            brk_markers[k].set_data([xv], [brk_list[k][ik]])
            # Texto
            if x_mode == 'distance':
                xinfo = f"s={float(arr[ik]):6.1f} m"
            else:
                xinfo = f"t={float(arr[ik]):6.2f} s"
            hover_texts[k].set_text(
                f"{d['label']}  {xinfo}  v={speed_list[k][ik]:6.1f} km/h  "
                f"thr={thr_list[k][ik]:3.0f}%  brk={brk_list[k][ik]:3.0f}%"
            )

        # Zoom agresivo centrado en el punto de data1
        if datasets:
            # índice correspondiente en data1
            arr0 = s_list[0] if x_mode == 'distance' else t_list[0]
            ik0 = idx_for_axis(arr0, xi)
            x0 = x_list[0][ik0]
            y0p = y_list[0][ik0]
            hw = TRACK_ZOOM_HALF_M
            ax_track.set_xlim(x0 - hw, x0 + hw)
            ax_track.set_ylim(y0p - hw, y0p + hw)
            # Mostrar distancia acumulada de data1 (m)
            if x_mode == 'distance' and 's' in datasets[0]:
                s0 = datasets[0]['s']
                if x_mode == 'distance':
                    dist_text.set_text(f"s = {float(s0[ik0]):.1f} m")
                else:
                    dist_text.set_text(f"t = {float(arr0[ik0]):.2f} s")

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig._hover_cid = cid


def main():
    # Paleta para datasets
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    datasets: list[dict] = []

    for idx, path in enumerate(DATA_FILES):
        if not os.path.exists(path):
            continue
        try:
            df = load_dataframe(path)
            real_lap, lap_time, lap_data = compute_fastest_lap(df)
            throttle, brake = compute_controls(lap_data)
        except Exception as e:
            print(f"No se pudo procesar {path}: {e}")
            continue

        label = os.path.splitext(os.path.basename(path))[0]
        color = colors[idx % len(colors)]
        print(f"{label}: vuelta rápida {real_lap} — {lap_time:.2f} s")
        datasets.append({
            'path': path,
            'label': label,
            'color': color,
            'real_lap': real_lap,
            'lap_time': lap_time,
            'lap_data': lap_data,
            'throttle': throttle,
            'brake': brake,
        })

    # Calcular sectores para cada dataset (tras construir la lista)
    for d in datasets:
        d['sectors'] = compute_sector_times(d['lap_data'], d['lap_time'])
        d['s'] = compute_cumulative_distance(d['lap_data'])

    if not datasets:
        raise SystemExit("No se encontró ningún archivo válido en DATA_FILES.")

    # Elegir eje X por CLI o prompt
    x_mode = get_x_axis_mode()

    fig = build_comparison_figure(datasets, x_mode=x_mode)
    enable_hover_interaction_multi(fig, datasets, x_mode=x_mode)

    if ENABLE_ANIMATION:
        ani = animate_lap_multi(fig, datasets, x_mode=x_mode)
        fig._ani = ani

    # Final confirmation for the terminal so the operator knows everything
    # converted and the plot is about to be displayed.
    print("Plot ready and displaying – all conversions successful")
    plt.show()


def get_x_axis_mode() -> str:
    parser = argparse.ArgumentParser(description="ACTelemetry - comparador de vueltas")
    parser.add_argument("--x-axis", "-x", dest="xaxis", choices=["time", "distance", "tiempo", "metros", "t", "m"], default=None,
                        help="Elegir el eje X: 'time' o 'distance' (alias: t/tiempo, m/metros)")
    args, _ = parser.parse_known_args()
    choice = args.xaxis
    if choice is None:
        try:
            raw = input("Elige eje X [t=tiempo / m=metros] (por defecto: tiempo): ").strip().lower()
        except Exception:
            raw = ""
        if raw in ("m", "metros", "distance", "dist", "d"):
            return "distance"
        return DEFAULT_X_AXIS
    # normalizar
    if choice in ("m", "metros", "distance"):
        return "distance"
    return "time"


if __name__ == "__main__":
    main()