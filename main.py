import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# Debido a que los datos del CSV de vueltas generado por Assetto Corsa tiene ; al final de las líneas de datos, 
# esta función lee el archivo y se adapta automáticamente al formato correcto.

def read_csv_smart(path, sep=';'): #path (parametro obligarorio) y sep (parametro opcional con valor por defecto ';')
    # Leemos solo las primeras líneas para inspeccionar
    with open(path, 'r', encoding='utf-8') as f:
        lines = [next(f) for _ in range(5)]  # primeras 5 líneas
    
    header_cols = len(lines[0].strip().split(sep))
    data_cols = [len(l.strip().split(sep)) for l in lines[1:] if l.strip()]

    # Detectar si las filas de datos tienen una columna extra
    if all(dc == header_cols + 1 for dc in data_cols):
        with open(path, 'r', encoding='utf-8') as f:
            clean_lines = [line.rstrip(';\n') + '\n' for line in f]

        # Leemos desde memoria, sin crear un archivo nuevo
        from io import StringIO
        return pd.read_csv(StringIO(''.join(clean_lines)), sep=sep)
    
    # Si no hay problema, lo leemos normalmente
    return pd.read_csv(path, sep=sep)

def realtime(series):
    """
    Convierte una serie de tiempos (en milésimas de segundo)
    a formato mm:ss.mmm (string legible).
    """
    def format_time(ms_value):
        # Aseguramos que el valor sea numérico
        try:
            ms_value = float(ms_value)
        except ValueError:
            return None
        
        seconds_total = ms_value / 1000.0
        minutes = int(seconds_total // 60)
        seconds = int(seconds_total % 60)
        milliseconds = int((seconds_total - int(seconds_total)) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    return series.apply(format_time)


def vuelta_rapida(laps_validas):
    return laps_validas.loc[laps_validas['laptime'].idxmin()]

def rapida_teorica(laps_validas):
    return laps_validas['split1'].min() + laps_validas['split2'].min() + laps_validas['split3'].min()

# MAIN
# Leer CSV con separador ;
laps = read_csv_smart("laps.csv", sep=';')

# Normalizar nombres de columnas
laps.columns = laps.columns.str.strip().str.lower()

# Filtrar vueltas válidas
laps_validas = laps[laps['validity'] == 1].copy()
# Calcular vuelta teórica
vuelta_teorica = rapida_teorica(laps_validas)
# Calcular el ritmo medio
ritmo_medio = laps_validas['laptime'].mean()

# Preparar formato de datos para mostrar
laps_validas['laptime'] = realtime(laps_validas['laptime'])
laps_validas['split1'] = realtime(laps_validas['split1'])
laps_validas['split2'] = realtime(laps_validas['split2'])
laps_validas['split3'] = realtime(laps_validas['split3'])

# Mostrar datos
print("Datos cargados:")
print(laps_validas)

# Obtener vuelta rápida
vuelta_rapida = vuelta_rapida(laps_validas)
print("\nVuelta rápida:")
print("Lap", vuelta_rapida['lap'], ":", vuelta_rapida['laptime'], "\n")

# Obtener vuelta teórica
print("\nVuelta teórica:")
print(realtime(pd.Series([vuelta_teorica]))[0], "\n")

# Mostrar ritmo medio
print("Ritmo medio (válidas):")
print(realtime(pd.Series([ritmo_medio]))[0], "\n")

#Grafica de tiempos de vuelta
x = laps['lap'][laps['validity'] == 1]
y = laps['laptime'][laps['validity'] == 1]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', color='blue')
plt.xlabel('Vuelta')
plt.ylabel('Tiempo de Vuelta')
plt.title('Tiempos de Vuelta Válidas')
plt.grid(True)
plt.show()