#!/usr/bin/env python3
import serial
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from serial.tools import list_ports

# --- SETTINGS ---
PORT = '/dev/ttyACM0'  # Your confirmed port
START_FREQ = 1000000  # 1 MHz
STOP_FREQ = 900000000  # 900 MHz
POINTS = 101  # Standard NanoVNA sweep points


def connect():
    try:
        ser = serial.Serial(PORT, baudrate=115200, timeout=1)
        ser.read_all()  # Clear buffer
        return ser
    except Exception as e:
        print(f"Error opening {PORT}: {e}")
        sys.exit(1)


def get_trace_data(ser):
    print(f"Requesting sweep from {START_FREQ / 1e6} MHz to {STOP_FREQ / 1e6} MHz...")

    # 1. Setup Sweep Range
    cmd_sweep = f"sweep {START_FREQ} {STOP_FREQ}\r"
    ser.write(cmd_sweep.encode())

    # 2. Request Data (data 0 = S11/Port 1)
    # The NanoVNA returns data as "Real Imaginary" pairs, one per line
    ser.write(b"data 0\r")

    # 3. Read the stream
    raw_lines = ser.readlines()

    data = []
    for line in raw_lines:
        line = line.decode('utf-8', errors='ignore').strip()

        # Filter out command echoes (like "data 0" or "ch>")
        if "data" in line or "ch>" in line or not line:
            continue

        parts = line.split(' ')
        if len(parts) == 2:
            try:
                # Parse "Real Imag" into a complex number
                re = float(parts[0])
                im = float(parts[1])
                data.append(complex(re, im))
            except ValueError:
                pass

    return np.array(data)


def plot_data(data):
    # Create Frequency Axis
    freqs = np.linspace(START_FREQ, STOP_FREQ, len(data)) / 1e6  # in MHz

    # Calculate Log Magnitude (dB) = 20 * log10(|Gamma|)
    log_mag = 20 * np.log10(np.abs(data))

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Log Magnitude (Bode-like)
    ax1.plot(freqs, log_mag, 'b-')
    ax1.set_title("S11 Log Magnitude")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)

    # Plot 2: Polar Plot (Poor Man's Smith Chart)
    # Note: This is just a raw polar plot of Gamma.
    # Real Smith Charts have curved grid lines, but this shows the spiral!
    ax2.plot(np.real(data), np.imag(data), 'r-')
    ax2.set_title("Reflection Coefficient (Gamma)")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imaginary")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True)
    ax2.set_aspect('equal')

    # Draw unit circle (Smith Chart boundary)
    unit_circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
    ax2.add_artist(unit_circle)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ser = connect()
    data = get_trace_data(ser)
    ser.close()

    print(f"Received {len(data)} data points.")
    if len(data) > 0:
        plot_data(data)
    else:
        print("No data received. Check connection.")