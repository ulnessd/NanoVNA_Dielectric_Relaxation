#!/usr/bin/env python3
import sys
import serial
import time
import numpy as np
from serial.tools import list_ports

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QGroupBox, QMessageBox)
from PySide6.QtCore import QTimer, Slot, Qt

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

    def sweep(self, start_hz, stop_hz):
        if not self.connected: return None, None

        try:
            # 1. Send Sweep Command
            cmd = f"sweep {int(start_hz)} {int(stop_hz)}\r"
            self.ser.write(cmd.encode())

            # 2. Request Data
            self.ser.write(b"data 0\r")

            # 3. Read
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

            freqs = np.linspace(start_hz, stop_hz, len(data))
            return freqs, np.array(data)

        except Exception as e:
            print(f"Sweep Error: {e}")
            return None, None


# --- GUI MAIN WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PChem Dielectric Spectrometer")
        self.resize(1100, 700)

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

        # 2. Acquisition Group
        grp_acq = QGroupBox("Acquisition")
        layout_acq = QVBoxLayout()

        self.btn_sweep = QPushButton("Run Single Sweep")
        self.btn_sweep.setFixedHeight(40)
        self.btn_sweep.setStyleSheet("font-weight: bold; font-size: 14px;")

        layout_acq.addWidget(self.btn_sweep)
        grp_acq.setLayout(layout_acq)
        sidebar.addWidget(grp_acq)

        # 3. Status
        self.lbl_status = QLabel("Status: Disconnected")
        sidebar.addWidget(self.lbl_status)

        sidebar.addStretch()  # Push everything up

        # --- RIGHT SIDE (Plots) ---
        # We create a Matplotlib Figure and wrap it in a Canvas widget
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=4)

        # Setup Subplots (LogMag and Smith)
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
        # Initialize empty plots with labels
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
        for p in ports:
            # Add device path (e.g. /dev/ttyACM0)
            self.combo_ports.addItem(p.device)

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            # Connect
            port = self.combo_ports.currentText()
            if not port: return

            if self.vna.connect(port):
                self.btn_connect.setText("Disconnect")
                self.lbl_status.setText(f"Status: Connected to {port}")
                self.lbl_status.setStyleSheet("color: green")
            else:
                self.btn_connect.setChecked(False)
                QMessageBox.critical(self, "Error", "Could not connect to VNA.")
        else:
            # Disconnect
            self.vna.disconnect()
            self.btn_connect.setText("Connect")
            self.lbl_status.setText("Status: Disconnected")
            self.lbl_status.setStyleSheet("color: black")

    def run_measurement(self):
        if not self.vna.connected:
            QMessageBox.warning(self, "Warning", "Please connect first.")
            return

        self.lbl_status.setText("Status: Sweeping...")
        QApplication.processEvents()  # Force UI update

        # Run Sweep (1 MHz to 900 MHz)
        freqs, data = self.vna.sweep(1000000, 900000000)

        if data is not None:
            self.update_plots(freqs, data)
            self.lbl_status.setText("Status: Ready")
        else:
            self.lbl_status.setText("Status: Error Reading Data")

    def update_plots(self, freqs, data):
        # Clear old data
        self.ax_mag.clear()
        self.ax_smith.clear()

        # Re-draw grids/labels (clearing wipes them)
        self.init_plots()

        # 1. Log Mag Plot
        log_mag = 20 * np.log10(np.abs(data))
        self.ax_mag.plot(freqs / 1e6, log_mag, 'b-')

        # 2. Smith Chart Plot
        self.ax_smith.plot(np.real(data), np.imag(data), 'r-')

        # Refresh Canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())