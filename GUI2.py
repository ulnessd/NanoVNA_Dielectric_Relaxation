#!/usr/bin/env python3
import sys
import serial
import time
import numpy as np
from serial.tools import list_ports

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QGroupBox, QMessageBox, QLineEdit, QFormLayout)
from PySide6.QtCore import Qt

# Matplotlib for PyQt/PySide
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --- HARDWARE INTERFACE CLASS ---
class NanoVNA:
    def __init__(self):
        self.ser = None
        self.connected = False

    def connect(self, port):
        try:
            self.ser = serial.Serial(port, baudrate=115200, timeout=1)
            self.ser.read_all()
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection Error: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False

    def _run_single_chunk(self, start_hz, stop_hz):
        """Helper to run one hardware sweep (max 101 points usually)."""
        cmd = f"sweep {int(start_hz)} {int(stop_hz)}\r"
        self.ser.write(cmd.encode())
        self.ser.write(b"data 0\r")

        raw_lines = self.ser.readlines()
        data = []
        for line in raw_lines:
            line = line.decode('utf-8', errors='ignore').strip()
            if "data" in line or "ch>" in line or not line: continue
            parts = line.split(' ')
            if len(parts) == 2:
                try:
                    re = float(parts[0])
                    im = float(parts[1])
                    data.append(complex(re, im))
                except:
                    pass
        return data

    def sweep(self, start_hz, stop_hz, points=101):
        """
        Performs a sweep. If points > 101, it segments the sweep
        into multiple hardware commands and stitches the results.
        """
        if not self.connected: return None, None

        try:
            # NanoVNA standard is 101 points per sweep chunk
            # We will perform overlapping segments if points > 101

            # Simple segmentation strategy:
            # If user asks for 201 points, we can do 2 chunks of 101?
            # Or just interpolate?
            # Better: Calculate standard segments.
            # For simplicity in this V1, if points != 101, we will do a basic segmentation
            # approach assuming the hardware returns ~101 points per 'sweep' command.

            # Note: Many NanoVNA firmwares strictly return 101 points per sweep command.
            # To get 201, we need to carefully calculate the frequency steps.

            if points <= 101:
                # Simple case: One hardware sweep
                data = self._run_single_chunk(start_hz, stop_hz)
                freqs = np.linspace(start_hz, stop_hz, len(data))
                return freqs, np.array(data)

            else:
                # Segmented sweep (Stitching)
                # We need (points) total.
                # Hardware gives fixed 101 points.
                # We can request chunks that overlap or butt together.

                # Logic: Calculate exact frequency step needed
                span = stop_hz - start_hz
                step = span / (points - 1)

                # Calculate how many 101-point chunks we need
                # Each chunk covers 100 steps (101 points)
                chunk_points = 101
                chunk_span_steps = chunk_points - 1

                num_chunks = int(np.ceil((points - 1) / chunk_span_steps))

                all_data = []

                for i in range(num_chunks):
                    # Calculate start/stop for this chunk
                    c_start = start_hz + (i * chunk_span_steps * step)

                    # For the last chunk, we might overshoot, but let's just aim for the target
                    # or fill the remaining.
                    # Simplified approach: Overlap the last point of prev and first of next
                    c_stop = c_start + (chunk_span_steps * step)

                    # If this goes past global stop, clamp it?
                    # Actually, better to just run the calculated window to keep spacing linear

                    chunk_data = self._run_single_chunk(c_start, c_stop)

                    # Stitching:
                    # If it's not the first chunk, discard the first point (overlap)
                    if i > 0:
                        all_data.extend(chunk_data[1:])
                    else:
                        all_data.extend(chunk_data)

                # Trim to requested points if we got slightly more
                all_data = all_data[:points]

                freqs = np.linspace(start_hz, stop_hz, len(all_data))
                return freqs, np.array(all_data)

        except Exception as e:
            print(f"Sweep Error: {e}")
            return None, None


# --- GUI MAIN WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PChem Dielectric Spectrometer")
        self.resize(1200, 800)

        self.vna = NanoVNA()

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT SIDEBAR (Controls) ---
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignTop)
        main_layout.addLayout(sidebar, stretch=1)

        # 1. Connection Group
        grp_conn = QGroupBox("Connection")
        layout_conn = QVBoxLayout()

        self.combo_ports = QComboBox()
        self.btn_refresh = QPushButton("Refresh Ports")
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setCheckable(True)

        layout_conn.addWidget(self.combo_ports)
        layout_conn.addWidget(self.btn_refresh)
        layout_conn.addWidget(self.btn_connect)
        grp_conn.setLayout(layout_conn)
        sidebar.addWidget(grp_conn)

        # 2. Sweep Settings Group (NEW)
        grp_set = QGroupBox("Sweep Settings")
        layout_set = QFormLayout()

        self.input_start = QLineEdit("1")
        self.input_stop = QLineEdit("900")
        self.combo_points = QComboBox()
        self.combo_points.addItems(["101", "201", "301", "401"])

        layout_set.addRow("Start (MHz):", self.input_start)
        layout_set.addRow("Stop (MHz):", self.input_stop)
        layout_set.addRow("Points:", self.combo_points)

        grp_set.setLayout(layout_set)
        sidebar.addWidget(grp_set)

        # 3. Acquisition Group
        grp_acq = QGroupBox("Acquisition")
        layout_acq = QVBoxLayout()

        self.btn_sweep = QPushButton("Run Sweep")
        self.btn_sweep.setFixedHeight(40)
        self.btn_sweep.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #e1e1e1;")

        layout_acq.addWidget(self.btn_sweep)
        grp_acq.setLayout(layout_acq)
        sidebar.addWidget(grp_acq)

        # 4. Status
        self.lbl_status = QLabel("Status: Disconnected")
        sidebar.addWidget(self.lbl_status)

        sidebar.addStretch()  # Push everything up

        # --- RIGHT SIDE (Plots) ---
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=4)

        self.ax_mag = self.fig.add_subplot(121)
        self.ax_smith = self.fig.add_subplot(122)
        self.init_plots()

        # --- SIGNALS & SLOTS ---
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_sweep.clicked.connect(self.run_measurement)

        # Initial setup
        self.refresh_ports()

    def init_plots(self):
        self.ax_mag.set_title("Log Magnitude (S11)")
        self.ax_mag.set_xlabel("Frequency (MHz)")
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True)

        self.ax_smith.set_title("Smith Chart")
        self.ax_smith.set_xlabel("Real")
        self.ax_smith.set_ylabel("Imaginary")
        self.ax_smith.grid(True)
        self.ax_smith.set_aspect('equal')

        # Draw Unit Circle
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_smith.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = list_ports.comports()

        # Try to find a good default
        default_index = 0

        for i, p in enumerate(ports):
            # Display format: "/dev/ttyACM0 - NanoVNA..."
            name = f"{p.device} - {p.description}"
            self.combo_ports.addItem(name, userData=p.device)

            # Heuristic to find NanoVNA
            if "NanoVNA" in p.description or "ChibiOS" in p.description or "STM" in p.description:
                default_index = i

        if self.combo_ports.count() > 0:
            self.combo_ports.setCurrentIndex(default_index)

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            # Get the actual device path from UserData
            port_device = self.combo_ports.currentData()
            if not port_device:
                # Fallback if manual entry or issue
                port_device = self.combo_ports.currentText().split(' ')[0]

            if self.vna.connect(port_device):
                self.btn_connect.setText("Disconnect")
                self.lbl_status.setText(f"Status: Connected to {port_device}")
                self.lbl_status.setStyleSheet("color: green")
            else:
                self.btn_connect.setChecked(False)
                QMessageBox.critical(self, "Error", "Could not connect to VNA.")
        else:
            self.vna.disconnect()
            self.btn_connect.setText("Connect")
            self.lbl_status.setText("Status: Disconnected")
            self.lbl_status.setStyleSheet("color: black")

    def run_measurement(self):
        if not self.vna.connected:
            QMessageBox.warning(self, "Warning", "Please connect first.")
            return

        # Get Settings
        try:
            start_mhz = float(self.input_start.text())
            stop_mhz = float(self.input_stop.text())
            points = int(self.combo_points.currentText())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid numeric input for frequency.")
            return

        start_hz = start_mhz * 1e6
        stop_hz = stop_mhz * 1e6

        self.lbl_status.setText("Status: Sweeping...")
        self.btn_sweep.setEnabled(False)
        QApplication.processEvents()

        # Run Sweep
        freqs, data = self.vna.sweep(start_hz, stop_hz, points)

        if data is not None:
            self.update_plots(freqs, data)
            self.lbl_status.setText("Status: Ready")
        else:
            self.lbl_status.setText("Status: Error Reading Data")

        self.btn_sweep.setEnabled(True)

    def update_plots(self, freqs, data):
        self.ax_mag.clear()
        self.ax_smith.clear()
        self.init_plots()

        # Log Mag
        log_mag = 20 * np.log10(np.abs(data))
        self.ax_mag.plot(freqs / 1e6, log_mag, 'b-')

        # Smith
        self.ax_smith.plot(np.real(data), np.imag(data), 'r-')

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())