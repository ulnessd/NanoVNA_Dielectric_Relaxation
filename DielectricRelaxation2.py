#!/usr/bin/env python3
import sys
import serial
import time
import numpy as np
import pandas as pd
import datetime
from serial.tools import list_ports

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QGroupBox, QMessageBox, QLineEdit, QFormLayout,
                               QCheckBox, QFileDialog)
from PySide6.QtCore import Qt

# Matplotlib for PyQt/PySide
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# --- CALIBRATION LOGIC (One-Port SOL) ---
class SOLTCalibrator:
    def __init__(self):
        self.calibrated = False
        self.short_raw = None
        self.open_raw = None
        self.load_raw = None
        # Error Terms
        self.E_d = None  # Directivity
        self.E_s = None  # Source Match
        self.E_r = None  # Reflection Tracking

    def measure_standard(self, standard_type, data):
        if standard_type == 'short':
            self.short_raw = np.array(data)
        elif standard_type == 'open':
            self.open_raw = np.array(data)
        elif standard_type == 'load':
            self.load_raw = np.array(data)

    def compute_coefficients(self):
        if self.short_raw is None or self.open_raw is None or self.load_raw is None:
            return False

        # 1. Directivity (E_d) is the Load measurement (assuming perfect load)
        self.E_d = self.load_raw

        # 2. Compute Source Match (E_s) and Tracking (E_r)
        # Standard definitions: Gamma_Short = -1, Gamma_Open = 1
        M_open_prime = self.open_raw - self.E_d
        M_short_prime = self.short_raw - self.E_d

        # E_s = (M_open' + M_short') / (M_open' - M_short')
        numerator = M_open_prime + M_short_prime
        denominator = M_open_prime - M_short_prime

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self.E_s = np.nan_to_num(numerator / denominator)

        # E_r = M_open' * (1 - E_s)
        self.E_r = M_open_prime * (1 - self.E_s)

        self.calibrated = True
        return True

    def apply_correction(self, raw_dut):
        if not self.calibrated: return raw_dut

        # Formula: Gamma_Actual = (Gamma_Meas - E_d) / (E_r + E_s * (Gamma_Meas - E_d))
        M_prime = raw_dut - self.E_d
        numerator = M_prime
        denominator = self.E_r + (self.E_s * M_prime)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_actual = np.nan_to_num(numerator / denominator)

        return gamma_actual

    def save_calibration(self, filename):
        """Saves current error terms to a .npz file."""
        if not self.calibrated:
            return False
        try:
            np.savez(filename, E_d=self.E_d, E_s=self.E_s, E_r=self.E_r)
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filename):
        """Loads error terms from a .npz file."""
        try:
            data = np.load(filename)
            self.E_d = data['E_d']
            self.E_s = data['E_s']
            self.E_r = data['E_r']
            self.calibrated = True
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


# --- HARDWARE INTERFACE ---
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
        if self.ser and self.ser.is_open: self.ser.close()
        self.connected = False

    def _run_single_chunk(self, start_hz, stop_hz):
        """Runs one sweep command and parses output."""
        self.ser.read_all()  # Flush input buffer to prevent data mixing

        cmd = f"sweep {int(start_hz)} {int(stop_hz)}\r"
        self.ser.write(cmd.encode())
        self.ser.write(b"data 0\r")

        # Read lines until we see the prompt 'ch>' or run out
        # We assume standard NanoVNA response speed
        time.sleep(0.05)

        raw_lines = self.ser.readlines()
        data = []
        for line in raw_lines:
            line = line.decode('utf-8', errors='ignore').strip()
            # Skip echoes and prompts
            if "data" in line or "ch>" in line or "sweep" in line or not line: continue

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
            # Basic segmentation logic
            if points <= 101:
                data = self._run_single_chunk(start_hz, stop_hz)
                # Fallback if device returns fewer points
                if len(data) == 0: return None, None
                freqs = np.linspace(start_hz, stop_hz, len(data))
                return freqs, np.array(data)
            else:
                # Segmented Sweep
                span = stop_hz - start_hz
                step = span / (points - 1)
                chunk_points = 101
                chunk_span_steps = chunk_points - 1
                num_chunks = int(np.ceil((points - 1) / chunk_span_steps))

                all_data = []

                for i in range(num_chunks):
                    # Calculate segment range
                    c_start = start_hz + (i * chunk_span_steps * step)
                    c_stop = c_start + (chunk_span_steps * step)

                    # Run chunk
                    chunk_data = self._run_single_chunk(c_start, c_stop)

                    # Stitching: skip first point of subsequent chunks to avoid duplicates
                    if i > 0:
                        all_data.extend(chunk_data[1:])
                    else:
                        all_data.extend(chunk_data)

                    # Short pause to let VNA buffer clear
                    time.sleep(0.05)

                # Trim to exact requested length
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
        self.setWindowTitle("Dielectric Spectrometer (Calibrated)")
        self.resize(1600, 900)  # Increased width for 3rd column

        self.vna = NanoVNA()
        self.calibrator = SOLTCalibrator()

        # Data State
        self.last_freqs = None
        self.last_raw_data = None
        self.current_data_calibrated = None

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar
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

        # 2. Sweep
        grp_set = QGroupBox("Sweep Settings")
        layout_set = QFormLayout()
        self.input_start = QLineEdit("1")  # MHz
        self.input_stop = QLineEdit("900")  # MHz
        self.combo_points = QComboBox()
        self.combo_points.addItems(["101", "201", "301", "401"])
        layout_set.addRow("Start (MHz):", self.input_start)
        layout_set.addRow("Stop (MHz):", self.input_stop)
        layout_set.addRow("Points:", self.combo_points)
        grp_set.setLayout(layout_set)
        sidebar.addWidget(grp_set)

        # 3. Calibration
        grp_cal = QGroupBox("Calibration (SOLT)")
        layout_cal = QVBoxLayout()
        self.btn_cal_short = QPushButton("Measure SHORT")
        self.btn_cal_open = QPushButton("Measure OPEN")
        self.btn_cal_load = QPushButton("Measure LOAD")

        # --- NEW: Save/Load Calibration Buttons ---
        hbox_cal_io = QHBoxLayout()
        self.btn_save_cal = QPushButton("Save Cal")
        self.btn_load_cal = QPushButton("Load Cal")
        hbox_cal_io.addWidget(self.btn_save_cal)
        hbox_cal_io.addWidget(self.btn_load_cal)

        self.chk_apply_cal = QCheckBox("Apply Calibration")
        self.chk_apply_cal.setEnabled(False)
        layout_cal.addWidget(self.btn_cal_short)
        layout_cal.addWidget(self.btn_cal_open)
        layout_cal.addWidget(self.btn_cal_load)
        layout_cal.addLayout(hbox_cal_io)
        layout_cal.addWidget(self.chk_apply_cal)
        grp_cal.setLayout(layout_cal)
        sidebar.addWidget(grp_cal)

        # 4. Action
        grp_acq = QGroupBox("Acquisition")
        layout_acq = QVBoxLayout()
        self.btn_sweep = QPushButton("Run Sweep")
        self.btn_sweep.setFixedHeight(40)
        self.btn_sweep.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #e1e1e1;")
        self.btn_save = QPushButton("Save Data (.csv)")
        self.btn_save.setEnabled(False)
        layout_acq.addWidget(self.btn_sweep)
        layout_acq.addWidget(self.btn_save)
        grp_acq.setLayout(layout_acq)
        sidebar.addWidget(grp_acq)

        self.lbl_status = QLabel("Status: Disconnected")
        sidebar.addWidget(self.lbl_status)
        sidebar.addStretch()

        # Plots
        plot_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=4)

        self.fig = Figure(figsize=(14, 8))  # Wider figure

        # Adjust subplot spacing: left margin, width space between cols, height space between rows
        self.fig.subplots_adjust(left=0.08, right=0.95, wspace=0.35, hspace=0.35)

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Grid Spec for 3 columns (using add_subplot gridspec logic)
        # Layout:
        # Col 1: Log Mag (Full Height)
        # Col 2: Smith Chart (Full Height)
        # Col 3: Bode Phase (Top), Nyquist (Bottom)

        gs = self.fig.add_gridspec(2, 3)

        self.ax_mag = self.fig.add_subplot(gs[:, 0])  # Col 1, All Rows
        self.ax_smith = self.fig.add_subplot(gs[:, 1])  # Col 2, All Rows
        self.ax_phase = self.fig.add_subplot(gs[0, 2])  # Col 3, Row 1
        self.ax_nyquist = self.fig.add_subplot(gs[1, 2])  # Col 3, Row 2

        self.init_plots()

        # Events
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_sweep.clicked.connect(self.run_measurement)
        self.btn_save.clicked.connect(self.save_data)

        self.btn_cal_short.clicked.connect(lambda: self.run_cal_step('short'))
        self.btn_cal_open.clicked.connect(lambda: self.run_cal_step('open'))
        self.btn_cal_load.clicked.connect(lambda: self.run_cal_step('load'))

        self.btn_save_cal.clicked.connect(self.save_calibration)
        self.btn_load_cal.clicked.connect(self.load_calibration)

        self.chk_apply_cal.toggled.connect(self.refresh_plot_view)

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        self.refresh_ports()

    def init_plots(self):
        # 1. Log Magnitude
        self.ax_mag.set_title("Log Magnitude (S11)")
        self.ax_mag.set_xlabel("Frequency (MHz)")
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True)

        # 2. Smith Chart
        self.ax_smith.set_title("Smith Chart")
        self.ax_smith.set_xlabel("Real")
        self.ax_smith.set_ylabel("Imaginary")
        self.ax_smith.grid(True)
        self.ax_smith.set_aspect('equal')

        # Reference Circle (Gray, Dashed)
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_smith.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.5)
        # Reference Line (Real Axis)
        self.ax_smith.plot([-1, 1], [0, 0], color='gray', linestyle='-', alpha=0.3)

        # 3. Phase (Bode)
        self.ax_phase.set_title("Phase (S11)")
        self.ax_phase.set_xlabel("Frequency (MHz)")
        self.ax_phase.set_ylabel("Phase (Degrees)")
        self.ax_phase.set_ylim(-180, 180)
        self.ax_phase.grid(True)

        # 4. Nyquist Plot (-Im(Z) vs Re(Z))
        self.ax_nyquist.set_title("Nyquist Plot (Impedance)")
        self.ax_nyquist.set_xlabel("Z' (Real Ohm)")
        self.ax_nyquist.set_ylabel("-Z'' (Neg Imag Ohm)")
        self.ax_nyquist.grid(True)
        # Aspect ratio can be tricky for Nyquist if R >> X, keeping it auto for now
        # self.ax_nyquist.set_aspect('equal', adjustable='datalim')

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = list_ports.comports()
        idx = 0
        for i, p in enumerate(ports):
            name = f"{p.device} - {p.description}"
            self.combo_ports.addItem(name, userData=p.device)
            if "NanoVNA" in p.description or "STM" in p.description: idx = i
        if self.combo_ports.count() > 0: self.combo_ports.setCurrentIndex(idx)

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            p = self.combo_ports.currentData() or self.combo_ports.currentText().split(' ')[0]
            if self.vna.connect(p):
                self.btn_connect.setText("Disconnect")
                self.lbl_status.setText(f"Connected: {p}")
                self.lbl_status.setStyleSheet("color: green")
            else:
                self.btn_connect.setChecked(False)
                QMessageBox.critical(self, "Error", "Connection Failed")
        else:
            self.vna.disconnect()
            self.btn_connect.setText("Connect")
            self.lbl_status.setText("Disconnected")
            self.lbl_status.setStyleSheet("color: black")

    def get_sweep_params(self):
        try:
            start = float(self.input_start.text()) * 1e6
            stop = float(self.input_stop.text()) * 1e6
            points = int(self.combo_points.currentText())
            return start, stop, points
        except:
            return None

    def run_cal_step(self, std):
        if not self.vna.connected: return
        p = self.get_sweep_params()
        if not p: return

        self.lbl_status.setText(f"Calibrating {std.upper()}...")
        QApplication.processEvents()
        f, d = self.vna.sweep(*p)

        if d is not None:
            self.calibrator.measure_standard(std, d)
            getattr(self, f"btn_cal_{std}").setStyleSheet("background-color: #aaffaa;")
            self.lbl_status.setText(f"{std.upper()} Saved.")
            if self.calibrator.compute_coefficients():
                self.chk_apply_cal.setEnabled(True)
                self.chk_apply_cal.setChecked(True)
                QMessageBox.information(self, "Done", "Calibration Applied!")

    # --- NEW: Save/Load Calibration Handlers ---
    def save_calibration(self):
        if not self.calibrator.calibrated:
            QMessageBox.warning(self, "Warning", "Calibration not ready to save.")
            return

        name, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "nanovna.cal.npz", "NumPy Zip (*.npz)")
        if name:
            if self.calibrator.save_calibration(name):
                self.lbl_status.setText("Calibration Saved.")
            else:
                self.lbl_status.setText("Error Saving Cal.")

    def load_calibration(self):
        name, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "NumPy Zip (*.npz)")
        if name:
            if self.calibrator.load_calibration(name):
                self.lbl_status.setText("Calibration Loaded.")
                self.chk_apply_cal.setEnabled(True)
                self.chk_apply_cal.setChecked(True)
                # Visual feedback: Turn buttons green
                self.btn_cal_short.setStyleSheet("background-color: #aaffaa;")
                self.btn_cal_open.setStyleSheet("background-color: #aaffaa;")
                self.btn_cal_load.setStyleSheet("background-color: #aaffaa;")
                self.refresh_plot_view()
            else:
                QMessageBox.critical(self, "Error", "Failed to load calibration file.")

    def run_measurement(self):
        if not self.vna.connected: return
        p = self.get_sweep_params()
        if not p: return

        self.lbl_status.setText("Sweeping...")
        self.btn_sweep.setEnabled(False)
        QApplication.processEvents()

        f, d = self.vna.sweep(*p)
        if d is not None:
            self.last_freqs = f
            self.last_raw_data = d
            self.refresh_plot_view()
            self.lbl_status.setText("Ready")
            self.btn_save.setEnabled(True)
        else:
            self.lbl_status.setText("Read Error")
        self.btn_sweep.setEnabled(True)

    def refresh_plot_view(self):
        if self.last_raw_data is None: return

        if self.chk_apply_cal.isChecked() and self.calibrator.calibrated:
            data = self.calibrator.apply_correction(self.last_raw_data)
            col, lbl = 'g-', "Calibrated"
        else:
            data = self.last_raw_data
            col, lbl = 'b-', "Raw"

        self.current_data_calibrated = data

        # Clear plots
        self.ax_mag.clear()
        self.ax_smith.clear()
        self.ax_phase.clear()
        self.ax_nyquist.clear()

        self.init_plots()

        # 1. Log Mag
        log_mag = 20 * np.log10(np.abs(data))
        self.ax_mag.plot(self.last_freqs / 1e6, log_mag, col, label=lbl)
        self.ax_mag.legend()

        # 2. Smith
        self.ax_smith.plot(np.real(data), np.imag(data), 'r-', label=lbl)

        # Calculate Impedance Z = 50 * (1+G)/(1-G)
        z = 50 * (1 + data) / (1 - data)

        # 3. Phase (Bode)
        phase = np.angle(data, deg=True)
        self.ax_phase.plot(self.last_freqs / 1e6, phase, 'purple', label="Phase")

        # 4. Nyquist (-Im(Z) vs Re(Z))
        # Convention: -Im(Z) on Y-axis (so capacitive is UP)
        self.ax_nyquist.plot(np.real(z), -np.imag(z), 'orange', label="Nyquist")

        self.canvas.draw()

    def on_plot_click(self, event):
        if event.inaxes is None or self.last_freqs is None: return

        idx = 0

        # Common lookup: Find nearest frequency index
        # For Mag/Phase plots (x-axis is Freq)
        if event.inaxes in [self.ax_mag, self.ax_phase]:
            f_mhz = self.last_freqs / 1e6
            idx = (np.abs(f_mhz - event.xdata)).argmin()

        # For Smith/Nyquist plots (x-axis is Real) - we need a different search strategy
        # or we just search the complex data array for closest match
        elif event.inaxes in [self.ax_smith, self.ax_nyquist]:
            # We search based on distance in the complex plane appropriate for that plot
            pass

        # Handle specific plot annotations
        if event.inaxes == self.ax_mag:
            val = 20 * np.log10(np.abs(self.current_data_calibrated[idx]))
            self.ax_mag.plot(self.last_freqs[idx] / 1e6, val, 'ro')
            self.ax_mag.annotate(f"{self.last_freqs[idx] / 1e6:.1f} MHz\n{val:.1f} dB",
                                 (self.last_freqs[idx] / 1e6, val),
                                 xytext=(0, 10), textcoords='offset points',
                                 bbox=dict(boxstyle="round", fc="yellow"), ha='center')

        elif event.inaxes == self.ax_phase:
            val = np.angle(self.current_data_calibrated[idx], deg=True)
            self.ax_phase.plot(self.last_freqs[idx] / 1e6, val, 'mo')
            self.ax_phase.annotate(f"{self.last_freqs[idx] / 1e6:.1f} MHz\n{val:.1f} deg",
                                   (self.last_freqs[idx] / 1e6, val),
                                   xytext=(0, 10), textcoords='offset points',
                                   bbox=dict(boxstyle="round", fc="violet"), ha='center')

        elif event.inaxes == self.ax_smith:
            # Re-calculate index based on visual distance on Smith Chart (Gamma plane)
            click = complex(event.xdata, event.ydata)
            idx = np.abs(self.current_data_calibrated - click).argmin()

            pt = self.current_data_calibrated[idx]
            z = 50 * (1 + pt) / (1 - pt)  # Impedance conversion
            self.ax_smith.plot(np.real(pt), np.imag(pt), 'bo')
            self.ax_smith.annotate(f"Z={z.real:.1f}{z.imag:+.1f}j\n{self.last_freqs[idx] / 1e6:.1f} MHz",
                                   (np.real(pt), np.imag(pt)),
                                   xytext=(0, 10), textcoords='offset points',
                                   bbox=dict(boxstyle="round", fc="cyan"), ha='center')

        elif event.inaxes == self.ax_nyquist:
            # Re-calculate index based on visual distance on Nyquist Plot (Impedance plane)
            # Nyquist plots Re(Z) on X and -Im(Z) on Y
            z_data = 50 * (1 + self.current_data_calibrated) / (1 - self.current_data_calibrated)
            z_real = np.real(z_data)
            z_neg_imag = -np.imag(z_data)

            # Find closest point in (Real, -Imag) space
            dist = np.sqrt((z_real - event.xdata) ** 2 + (z_neg_imag - event.ydata) ** 2)
            idx = dist.argmin()

            pt_real = z_real[idx]
            pt_imag = z_neg_imag[idx]  # This is -Imag(Z)
            actual_z = z_data[idx]

            self.ax_nyquist.plot(pt_real, pt_imag, 'ko')  # Black dot
            self.ax_nyquist.annotate(
                f"{self.last_freqs[idx] / 1e6:.1f} MHz\nZ'={actual_z.real:.1f}, -Z''={-actual_z.imag:.1f}",
                (pt_real, pt_imag),
                xytext=(0, 10), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="orange"), ha='center')

        self.canvas.draw()

    def save_data(self):
        if self.last_freqs is None: return
        name, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"scan_{int(time.time())}.csv", "CSV (*.csv)")
        if name:
            z = 50 * (1 + self.current_data_calibrated) / (1 - self.current_data_calibrated)
            df = pd.DataFrame({
                'Freq_Hz': self.last_freqs,
                'Gamma_Re': np.real(self.current_data_calibrated),
                'Gamma_Im': np.imag(self.current_data_calibrated),
                'Z_Re': np.real(z), 'Z_Im': np.imag(z)
            })
            df.to_csv(name, index=False)
            self.lbl_status.setText(f"Saved: {name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())