#!/usr/bin/env python3
import sys
import serial
import time
import numpy as np
from serial.tools import list_ports

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QGroupBox, QMessageBox, QLineEdit, QFormLayout,
                               QCheckBox)
from PySide6.QtCore import Qt

# Matplotlib for PyQt/PySide
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --- CALIBRATION LOGIC (One-Port SOL) ---
class SOLTCalibrator:
    def __init__(self):
        self.calibrated = False
        # Raw measurements of standards
        self.short_raw = None
        self.open_raw = None
        self.load_raw = None
        # Calculated Error Terms
        self.E_d = None  # Directivity
        self.E_s = None  # Source Match
        self.E_r = None  # Reflection Tracking

    def measure_standard(self, standard_type, data):
        """Store raw data for a specific standard."""
        if standard_type == 'short':
            self.short_raw = np.array(data)
        elif standard_type == 'open':
            self.open_raw = np.array(data)
        elif standard_type == 'load':
            self.load_raw = np.array(data)

    def compute_coefficients(self):
        """
        Solves the 3-term error model using the measured standards.
        Assumes ideal standards: Gamma_Short = -1, Gamma_Open = 1, Gamma_Load = 0
        """
        if self.short_raw is None or self.open_raw is None or self.load_raw is None:
            return False

        # 1. Directivity (E_d) is simply the Load measurement (assuming perfect load)
        self.E_d = self.load_raw

        # 2. To find E_s and E_r, we solve the system for Short and Open
        # M_open = E_d + E_r / (1 - E_s)
        # M_short = E_d - E_r / (1 + E_s)

        # Let M' = M - E_d
        M_open_prime = self.open_raw - self.E_d
        M_short_prime = self.short_raw - self.E_d

        # Algebra allows us to solve for E_s (Source Match):
        # E_s = (M_open_prime + M_short_prime) / (M_open_prime - M_short_prime)
        self.E_s = (M_open_prime + M_short_prime) / (M_open_prime - M_short_prime)

        # Solve for E_r (Reflection Tracking):
        # E_r = M_open_prime * (1 - E_s)
        self.E_r = M_open_prime * (1 - self.E_s)

        self.calibrated = True
        return True

    def apply_correction(self, raw_dut):
        """
        Applies error correction to DUT data.
        Gamma_Actual = (Gamma_Meas - E_d) / (E_r + E_s * (Gamma_Meas - E_d))
        """
        if not self.calibrated:
            return raw_dut

        M_prime = raw_dut - self.E_d
        numerator = M_prime
        denominator = self.E_r + (self.E_s * M_prime)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_actual = numerator / denominator
            gamma_actual = np.nan_to_num(gamma_actual)  # Handle singularities

        return gamma_actual


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
        if not self.connected: return None, None
        try:
            if points <= 101:
                data = self._run_single_chunk(start_hz, stop_hz)
                freqs = np.linspace(start_hz, stop_hz, len(data))
                return freqs, np.array(data)
            else:
                span = stop_hz - start_hz
                step = span / (points - 1)
                chunk_points = 101
                chunk_span_steps = chunk_points - 1
                num_chunks = int(np.ceil((points - 1) / chunk_span_steps))

                all_data = []
                for i in range(num_chunks):
                    c_start = start_hz + (i * chunk_span_steps * step)
                    c_stop = c_start + (chunk_span_steps * step)
                    chunk_data = self._run_single_chunk(c_start, c_stop)
                    if i > 0:
                        all_data.extend(chunk_data[1:])
                    else:
                        all_data.extend(chunk_data)

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
        self.setWindowTitle("PChem Dielectric Spectrometer (Calibrated)")
        self.resize(1300, 800)

        self.vna = NanoVNA()
        self.calibrator = SOLTCalibrator()

        # Store last sweep data for toggling calibration view
        self.last_freqs = None
        self.last_raw_data = None

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT SIDEBAR (Controls) ---
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignTop)
        main_layout.addLayout(sidebar, stretch=1)

        # 1. Connection
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

        # 2. Sweep Settings
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

        # 3. Calibration Panel (NEW)
        grp_cal = QGroupBox("Calibration (SOLT)")
        layout_cal = QVBoxLayout()

        self.btn_cal_short = QPushButton("Measure SHORT")
        self.btn_cal_open = QPushButton("Measure OPEN")
        self.btn_cal_load = QPushButton("Measure LOAD")
        self.chk_apply_cal = QCheckBox("Apply Calibration")
        self.chk_apply_cal.setEnabled(False)  # Only enable after cal is done

        layout_cal.addWidget(self.btn_cal_short)
        layout_cal.addWidget(self.btn_cal_open)
        layout_cal.addWidget(self.btn_cal_load)
        layout_cal.addWidget(self.chk_apply_cal)
        grp_cal.setLayout(layout_cal)
        sidebar.addWidget(grp_cal)

        # 4. Acquisition
        grp_acq = QGroupBox("Acquisition")
        layout_acq = QVBoxLayout()
        self.btn_sweep = QPushButton("Run Sweep")
        self.btn_sweep.setFixedHeight(40)
        self.btn_sweep.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #e1e1e1;")
        layout_acq.addWidget(self.btn_sweep)
        grp_acq.setLayout(layout_acq)
        sidebar.addWidget(grp_acq)

        # 5. Status
        self.lbl_status = QLabel("Status: Disconnected")
        sidebar.addWidget(self.lbl_status)

        sidebar.addStretch()

        # --- RIGHT SIDE (Plots) ---
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=4)

        self.ax_mag = self.fig.add_subplot(121)
        self.ax_smith = self.fig.add_subplot(122)
        self.init_plots()

        # --- SIGNALS ---
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_sweep.clicked.connect(self.run_measurement)

        # Calibration Signals
        self.btn_cal_short.clicked.connect(lambda: self.run_cal_step('short'))
        self.btn_cal_open.clicked.connect(lambda: self.run_cal_step('open'))
        self.btn_cal_load.clicked.connect(lambda: self.run_cal_step('load'))
        self.chk_apply_cal.toggled.connect(self.refresh_plot_view)

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
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_smith.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = list_ports.comports()
        default_index = 0
        for i, p in enumerate(ports):
            name = f"{p.device} - {p.description}"
            self.combo_ports.addItem(name, userData=p.device)
            if "NanoVNA" in p.description or "ChibiOS" in p.description or "STM" in p.description:
                default_index = i
        if self.combo_ports.count() > 0:
            self.combo_ports.setCurrentIndex(default_index)

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            port_device = self.combo_ports.currentData()
            if not port_device: port_device = self.combo_ports.currentText().split(' ')[0]
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

    def get_sweep_params(self):
        try:
            start = float(self.input_start.text()) * 1e6
            stop = float(self.input_stop.text()) * 1e6
            points = int(self.combo_points.currentText())
            return start, stop, points
        except ValueError:
            return None

    def run_cal_step(self, std_type):
        """Runs a sweep and stores it as a calibration standard."""
        if not self.vna.connected: return

        params = self.get_sweep_params()
        if not params: return

        self.lbl_status.setText(f"Status: Measuring {std_type.upper()}...")
        QApplication.processEvents()

        freqs, data = self.vna.sweep(*params)

        if data is not None:
            self.calibrator.measure_standard(std_type, data)
            # Update Button Style to show it's done
            btn = getattr(self, f"btn_cal_{std_type}")
            btn.setStyleSheet("background-color: #aaffaa;")  # Green
            self.lbl_status.setText(f"Status: {std_type.upper()} Captured.")

            # Check if all 3 are done
            if self.calibrator.compute_coefficients():
                self.chk_apply_cal.setEnabled(True)
                self.chk_apply_cal.setChecked(True)
                QMessageBox.information(self, "Calibration", "SOLT Calibration Complete!")
        else:
            self.lbl_status.setText("Status: Error")

    def run_measurement(self):
        if not self.vna.connected: return

        params = self.get_sweep_params()
        if not params: return

        self.lbl_status.setText("Status: Sweeping...")
        self.btn_sweep.setEnabled(False)
        QApplication.processEvents()

        freqs, data = self.vna.sweep(*params)

        if data is not None:
            self.last_freqs = freqs
            self.last_raw_data = data
            self.refresh_plot_view()
            self.lbl_status.setText("Status: Ready")
        else:
            self.lbl_status.setText("Status: Error")

        self.btn_sweep.setEnabled(True)

    def refresh_plot_view(self):
        """Updates plots, applying calibration if the checkbox is checked."""
        if self.last_raw_data is None: return

        # Decide whether to use raw or calibrated data
        if self.chk_apply_cal.isChecked() and self.calibrator.calibrated:
            data_to_plot = self.calibrator.apply_correction(self.last_raw_data)
            plot_color = 'g-'  # Green line for calibrated
        else:
            data_to_plot = self.last_raw_data
            plot_color = 'b-'  # Blue line for raw

        self.ax_mag.clear()
        self.ax_smith.clear()
        self.init_plots()

        # Log Mag
        log_mag = 20 * np.log10(np.abs(data_to_plot))
        self.ax_mag.plot(self.last_freqs / 1e6, log_mag, plot_color)

        # Smith
        self.ax_smith.plot(np.real(data_to_plot), np.imag(data_to_plot), 'r-')

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())