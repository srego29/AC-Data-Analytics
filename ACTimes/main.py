import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from typing import Iterable


# === SAFE CSV READER ===
def read_csv_safely(path: str, sep: str = ';') -> pd.DataFrame:
    """Read a CSV file robustly.

    This attempts to detect and fix a common issue where rows end with an
    extra separator (e.g. a trailing ';') that creates an extra empty column.
    If the first few data lines have one more column than the header, the
    function strips trailing separators and retries reading.
    """
    # Try a quick sniff of the first few lines; fall back to pandas if anything fails.
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # read up to 5 lines or until EOF
            lines = [next(f) for _ in range(5)]
    except Exception:
        return pd.read_csv(path, sep=sep)

    header_cols = len(lines[0].strip().split(sep))
    data_cols = [len(l.strip().split(sep)) for l in lines[1:] if l.strip()]

    # If every sample data line has exactly one more column than the header,
    # strip trailing separator characters and re-parse.
    if data_cols and all(dc == header_cols + 1 for dc in data_cols):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                clean_lines = [line.rstrip(sep + '\n') + '\n' for line in f]
            return pd.read_csv(StringIO(''.join(clean_lines)), sep=sep)
        except Exception:
            # Fall back to normal read if cleaning fails
            return pd.read_csv(path, sep=sep)

    return pd.read_csv(path, sep=sep)


# === TIME CONVERSIONS ===
def ms_to_seconds(series: Iterable) -> pd.Series:
    """Convert a pandas Series (milliseconds) to seconds (float).

    Non-numeric values are coerced to NaN.
    """
    return pd.to_numeric(series, errors='coerce').astype(float) / 1000.0


def seconds_to_mmssmmm(seconds: float) -> str:
    """Format seconds (float) as mm:ss.mmm.

    Rounds milliseconds to the nearest integer.
    """
    if pd.isna(seconds):
        return "--:--.---"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_series_realtime(series: pd.Series) -> pd.Series:
    """Map a numeric seconds Series to human-readable time strings."""
    return series.apply(seconds_to_mmssmmm)


# === ANALYSIS HELPERS ===
def fastest_lap(laps_valid: pd.DataFrame) -> pd.Series:
    """Return the row corresponding to the fastest lap (min laptime)."""
    return laps_valid.loc[laps_valid['laptime'].idxmin()]


def theoretical_fastest_lap(laps_valid: pd.DataFrame) -> float:
    """Calculate theoretical fastest lap by summing best splits."""
    return (
        laps_valid['split1'].min()
        + laps_valid['split2'].min()
        + laps_valid['split3'].min()
    )


# === GENERIC SECTOR PLOTTING ===
def plot_sector(laps_valid: pd.DataFrame, sector: str) -> None:
    """Plot sector times across laps with mean and best lines.

    The function expects numeric seconds in the sector column.
    """
    if sector not in laps_valid.columns:
        raise KeyError(f"Sector '{sector}' not found in data")

    x = laps_valid['lap']
    y = laps_valid[sector]

    mean_sector = y.mean()
    best_sector = y.min()

    labels = format_series_realtime(y)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='blue')
    plt.axhline(y=mean_sector, color='red', linestyle='--', label='Mean Pace')
    plt.axhline(y=best_sector, color='purple', linestyle='--', label='Best Sector')
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
    plt.xlabel('Lap')
    plt.ylabel('Sector Time (s)')
    plt.title(f'Sector {sector}')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # === MAIN ===
    laps = read_csv_safely("laps.csv", sep=';')
    # normalize column names
    laps.columns = laps.columns.str.strip().str.lower()

    # Convert time columns to seconds for calculations
    for col in ['laptime', 'split1', 'split2', 'split3']:
        if col in laps.columns:
            laps[col] = ms_to_seconds(laps[col])

    # Keep only valid laps (assumes validity==0 means valid)
    if 'validity' in laps.columns:
        laps_valid = laps[laps['validity'] == 0].copy()
    else:
        laps_valid = laps.copy()

    if laps_valid.empty:
        print('No valid laps found.')
        raise SystemExit(1)

    # Calculations
    theoretical = theoretical_fastest_lap(laps_valid)
    mean_pace = laps_valid['laptime'].mean()
    fastest_row = fastest_lap(laps_valid)

    # Display results
    print('Loaded data:')
    print(laps_valid.head())

    print('\nFastest lap:')
    print(f"Lap {int(fastest_row['lap'])} : {seconds_to_mmssmmm(fastest_row['laptime'])}")

    print('\nTheoretical fastest lap:')
    print(seconds_to_mmssmmm(theoretical))

    print('\nMean pace (valid laps):')
    print(seconds_to_mmssmmm(mean_pace))

    # === FULL LAP PLOT ===
    x = laps_valid['lap']
    y = laps_valid['laptime']

    labels = format_series_realtime(y)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='blue')
    plt.axhline(y=mean_pace, color='red', linestyle='--', label='Mean Pace')
    plt.axhline(y=theoretical, color='purple', linestyle='--', label='Theoretical Lap')
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
    plt.xlabel('Lap')
    plt.ylabel('Lap Time (s)')
    plt.title('Valid Lap Times')
    plt.grid(True)
    plt.legend()
    plt.show()

    # === SECTOR PLOTS ===
    for sector in ['split1', 'split2', 'split3']:
        if sector in laps_valid.columns:
            plot_sector(laps_valid, sector)
