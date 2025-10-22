from __future__ import annotations

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import AutoMinorLocator


DATA_FILES = ["data1.csv", "data2.csv"]  # añade aquí tus archivos a comparar
ENABLE_ANIMATION = False  # Pon a True si quieres la animación automática además del hover
TRACK_ZOOM_HALF_M = 25.0  # mitad de la ventana de zoom en el mapa (metros) al seguir el punto de data1


def format_lap_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def load_dataframe(path: str) -> pd.DataFrame:
    """Carga el CSV de ACT con nombres/unidades leyendo encabezados en líneas 15 y 16."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for _ in range(14):
            next(reader)
        names = next(reader)
        _units = next(reader)

    df = pd.read_csv(path, skiprows=16, names=names)
    df.columns = [c.strip('"') for c in df.columns]
    # Convertimos columnas numéricas cuando sea posible
    df = df.apply(pd.to_numeric, errors='ignore')
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


def build_comparison_figure(datasets: list[dict]):
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
    ax_track.axis('equal')

    # Dibujar datasets
    for idx, d in enumerate(datasets):
        lap_data = d['lap_data']
        label = d['label']
        throttle = d['throttle']
        brake = d['brake']

        # Colores por dataset: data1 colores "normales", data2 en blanco
        if idx == 0:
            c_track = 'tab:blue'      # mapa en azul
            c_speed = 'tab:purple'    # velocidad morado
            c_thr = 'tab:green'       # acelerador verde
            c_brk = 'tab:red'         # freno rojo
        else:
            c_track = 'white'
            c_speed = 'white'
            c_thr = 'white'
            c_brk = 'white'

        z = 5 if idx == 0 else 3  # data1 por encima
    # Track
        ax_track.plot(lap_data['Car Coord X'], lap_data['Car Coord Y'], color=c_track, alpha=0.95, linewidth=1.0, label=label, zorder=z)
    # Speed
        ax_speed.plot(lap_data['Lap Time (s)'], lap_data['Ground Speed'], color=c_speed, linewidth=0.9, label=f"Velocidad — {label}", zorder=z)
    # Throttle / Brake
        ax_throttle.plot(lap_data['Lap Time (s)'], throttle, color=c_thr, linewidth=0.8, label=f"Acel — {label}", zorder=z)
        ax_brake.plot(lap_data['Lap Time (s)'], brake, color=c_brk, linewidth=0.8, label=f"Freno — {label}", zorder=z)

    # Estética y leyendas
    ax_speed.set_title('Velocidad', color='#DDDDDD')
    ax_speed.set_ylabel('km/h')
    ax_throttle.set_ylabel('Acelerador (%)')
    ax_brake.set_ylabel('Freno (%)')
    ax_brake.set_xlabel('Tiempo dentro de la vuelta (s)')
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

    # Vuelta teórica combinada (mejores sectores de todos los datasets)
    all_sectors = [d['sectors'] for d in datasets if d.get('sectors') and len(d['sectors']) == 3]
    if len(all_sectors) >= 1:
        import numpy as np
        # Matriz N x 3
        arr = np.array(all_sectors)
        best = arr.min(axis=0)
        theo = best.sum()
        ax_info.text(0.02, 0.32, f"vuelta teorica: {format_lap_time(float(theo))}",
                     transform=ax_info.transAxes, va='top', ha='left', fontsize=12,
                     family='monospace', color='#DDDDDD',
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
    fig._cursors = (vline_speed, vline_thr, vline_brk)
    fig._artists = (track_points, track_paths, speed_markers, thr_markers, brk_markers)
    return fig


def animate_lap_multi(fig, datasets: list[dict]):
    # Anima siguiendo el primer dataset como referencia temporal
    from numpy import asarray

    ax_track, _ax_info, ax_speed, ax_thr, ax_brk = fig._axes
    vline_speed, vline_thr, vline_brk = fig._cursors
    track_points, track_paths, speed_markers, thr_markers, brk_markers = fig._artists

    t_list = [d['lap_data']['Lap Time (s)'].to_numpy() for d in datasets]
    x_list = [d['lap_data']['Car Coord X'].to_numpy() for d in datasets]
    y_list = [d['lap_data']['Car Coord Y'].to_numpy() for d in datasets]
    speed_list = [d['lap_data']['Ground Speed'].to_numpy() for d in datasets]
    thr_list = [d['throttle'].to_numpy() for d in datasets]
    brk_list = [d['brake'].to_numpy() for d in datasets]

    tref = t_list[0]
    ax_speed.set_xlim(tref.min(), max(t[-1] for t in t_list))

    # Intervalo
    if len(tref) > 1:
        dt_avg = max((tref[-1] - tref[0]) / (len(tref) - 1), 1/120)
    else:
        dt_avg = 0.03
    interval_ms = int(max(10, min(1000 * dt_avg, 1000 / 30)))

    import numpy as np

    def idx_for_time(tarr, t):
        i = int(np.searchsorted(tarr, t))
        if i <= 0:
            return 0
        if i >= len(tarr):
            return len(tarr) - 1
        return i

    def update(i):
        ti = tref[i]
        for v in (vline_speed, vline_thr, vline_brk):
            v.set_xdata([ti, ti])
        for k in range(len(datasets)):
            tk = t_list[k]
            ik = idx_for_time(tk, ti)
            # track
            track_points[k].set_data([x_list[k][ik]], [y_list[k][ik]])
            track_paths[k].set_data(x_list[k][:ik + 1], y_list[k][:ik + 1])
            # markers
            speed_markers[k].set_data([tk[ik]], [speed_list[k][ik]])
            thr_markers[k].set_data([tk[ik]], [thr_list[k][ik]])
            brk_markers[k].set_data([tk[ik]], [brk_list[k][ik]])
        return (*track_points, *track_paths, *speed_markers, *thr_markers, *brk_markers)

    ani = FuncAnimation(fig, update, frames=len(tref), interval=interval_ms, blit=False, repeat=False)
    return ani


def enable_hover_interaction_multi(fig, datasets: list[dict]):
    """Hover: al mover el ratón por las gráficas, sincroniza cursores/markers y puntos del track en todos los datasets."""
    import numpy as np

    ax_track, ax_info, ax_speed, ax_thr, ax_brk = fig._axes
    vline_speed, vline_thr, vline_brk = fig._cursors
    track_points, track_paths, speed_markers, thr_markers, brk_markers = fig._artists

    axes_set = {ax_speed, ax_thr, ax_brk}

    # Prepara arrays por dataset
    t_list = [d['lap_data']['Lap Time (s)'].to_numpy() for d in datasets]
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

    def idx_for_time(tarr, t):
        i = int(np.searchsorted(tarr, t))
        if i <= 0:
            return 0
        if i >= len(tarr):
            return len(tarr) - 1
        return i

    def on_move(event):
        if event.inaxes not in axes_set or event.xdata is None:
            return
        ti = event.xdata
        # Cursores
        for v in (vline_speed, vline_thr, vline_brk):
            v.set_xdata([ti, ti])

        # Por dataset
        for k, d in enumerate(datasets):
            tk = t_list[k]
            ik = idx_for_time(tk, ti)
            # Puntos en el track y rastro
            track_points[k].set_data([x_list[k][ik]], [y_list[k][ik]])
            track_paths[k].set_data(x_list[k][:ik + 1], y_list[k][:ik + 1])
            # Markers en subplots
            speed_markers[k].set_data([tk[ik]], [speed_list[k][ik]])
            thr_markers[k].set_data([tk[ik]], [thr_list[k][ik]])
            brk_markers[k].set_data([tk[ik]], [brk_list[k][ik]])
            # Texto
            hover_texts[k].set_text(
                f"{d['label']}  t={tk[ik]:6.2f} s  v={speed_list[k][ik]:6.1f} km/h  "
                f"thr={thr_list[k][ik]:3.0f}%  brk={brk_list[k][ik]:3.0f}%"
            )

        # Zoom agresivo centrado en el punto de data1
        if datasets:
            # índice correspondiente en data1
            tk0 = t_list[0]
            ik0 = idx_for_time(tk0, ti)
            x0 = x_list[0][ik0]
            y0p = y_list[0][ik0]
            hw = TRACK_ZOOM_HALF_M
            ax_track.set_xlim(x0 - hw, x0 + hw)
            ax_track.set_ylim(y0p - hw, y0p + hw)
            # Mostrar distancia acumulada de data1 (m)
            if 's' in datasets[0]:
                s0 = datasets[0]['s']
                dist_text.set_text(f"s = {float(s0[ik0]):.1f} m")

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

    fig = build_comparison_figure(datasets)
    enable_hover_interaction_multi(fig, datasets)

    if ENABLE_ANIMATION:
        ani = animate_lap_multi(fig, datasets)
        fig._ani = ani

    plt.show()


if __name__ == "__main__":
    main()