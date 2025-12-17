"""
Interface Examiner (Patched for Scaling)
Hardware: NanoVNA (Serial) + Thorlabs TDC001 (Kinesis .NET)
Framework: PySide6
"""
import sys
import time
import os
import serial
import numpy as np
import pandas as pd
from serial.tools import list_ports

# --- 1. PYTHONNET / KINESIS SETUP ---
import clr  # pip install pythonnet

# Default Kinesis 64-bit path
kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
if not os.path.exists(kinesis_path):
    # Fallback to 32-bit path just in case
    kinesis_path = r"C:\Program Files (x86)\Thorlabs\Kinesis"

if os.path.exists(kinesis_path):
    sys.path.append(kinesis_path)
    try:
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.TCube.DCServoCLI import TCubeDCServo
        from System import Decimal

        KINESIS_AVAILABLE = True
    except Exception as e:
        print(f"Kinesis DLL Load Error: {e}")
        KINESIS_AVAILABLE = False
else:
    print("Warning: Kinesis software not found. Motor features disabled.")
    KINESIS_AVAILABLE = False

# --- GUI IMPORTS ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QGroupBox, QMessageBox, QLineEdit, QFormLayout,
                               QCheckBox, QFileDialog)
from PySide6.QtCore import Qt

# Matplotlib for PyQt/PySide
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# --- THORLABS MOTOR CLASS (UPDATED) ---
class ThorlabsMotor:
    def __init__(self):
        self.device = None
        self.serial_no = None
        self.connected = False
        self.is_simulated = not KINESIS_AVAILABLE
        self.scale_factor = 1.0  # Default 1.0 (Assume DLL handles it)

    def find_devices(self):
        if self.is_simulated: return []
        try:
            DeviceManagerCLI.BuildDeviceList()
            return list(DeviceManagerCLI.GetDeviceList())
        except Exception as e:
            print(f"Device Scan Error: {e}")
            return []

    def connect(self, serial_no):
        if self.is_simulated:
            self.connected = True
            return True

        try:
            self.serial_no = str(serial_no)
            self.device = TCubeDCServo.CreateTCubeDCServo(self.serial_no)
            self.device.Connect(self.serial_no)

            if not self.device.IsSettingsInitialized():
                self.device.WaitForSettingsInitialized(5000)

            # --- FIX 1: FASTER POLLING (50ms instead of 250ms) ---
            self.device.StartPolling(50)
            time.sleep(0.5)
            self.device.EnableDevice()

            try:
                self.device.LoadMotorConfiguration(self.serial_no)
            except Exception as conf_err:
                print(f"Config Load Warning: {conf_err}")

            self.connected = True
            return True
        except Exception as e:
            print(f"Motor Connect Error: {e}")
            self.connected = False
            return False

    def move_to(self, position_val):
        if not self.connected: return
        try:
            scaled_target = float(position_val) * self.scale_factor
            target = Decimal(scaled_target)
            self.device.MoveTo(target, 0)
        except Exception as e:
            print(f"Move Error: {e}")

    def disconnect(self):
        if self.device and self.connected:
            try:
                self.device.StopPolling()
                self.device.Disconnect()
            except:
                pass
        self.connected = False

    def home(self):
        if not self.connected: return
        try:
            self.device.Home(60000)
        except Exception as e:
            print(f"Homing Error: {e}")

    def move_to(self, position_val):
        """
        Moves to position * scale_factor.
        If DLL thinks 1 unit = 1 encoder count, user sets scale to 34304.
        If DLL thinks 1 unit = 1 mm, user sets scale to 1.0.
        """
        if not self.connected: return
        try:
            # Apply scaling
            scaled_target = float(position_val) * self.scale_factor
            target = Decimal(scaled_target)
            self.device.MoveTo(target, 0)
        except Exception as e:
            print(f"Move Error: {e}")

    def is_moving(self):
        if not self.connected: return False
        return self.device.Status.IsInMotion

    def get_position(self):
        if not self.connected: return 0.0
        # Return raw position / scale to show "User Units"
        raw = float(str(self.device.Position))
        return raw / self.scale_factor if self.scale_factor != 0 else 0


# --- CALIBRATION LOGIC (One-Port SOL) ---
class SOLTCalibrator:
    def __init__(self):
        self.calibrated = False
        self.short_raw = None
        self.open_raw = None
        self.load_raw = None
        self.E_d, self.E_s, self.E_r = None, None, None

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

        # 1. Directivity
        self.E_d = self.load_raw

        # 2. Source Match & Tracking
        M_open_prime = self.open_raw - self.E_d
        M_short_prime = self.short_raw - self.E_d

        numerator = M_open_prime + M_short_prime
        denominator = M_open_prime - M_short_prime

        with np.errstate(divide='ignore', invalid='ignore'):
            self.E_s = np.nan_to_num(numerator / denominator)

        self.E_r = M_open_prime * (1 - self.E_s)
        self.calibrated = True
        return True

    def apply_correction(self, raw_dut):
        if not self.calibrated: return raw_dut
        M_prime = raw_dut - self.E_d
        numerator = M_prime
        denominator = self.E_r + (self.E_s * M_prime)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_actual = np.nan_to_num(numerator / denominator)
        return gamma_actual

    def save_calibration(self, filename):
        if not self.calibrated: return False
        try:
            np.savez(filename, E_d=self.E_d, E_s=self.E_s, E_r=self.E_r)
            return True
        except:
            return False

    def load_calibration(self, filename):
        try:
            data = np.load(filename)
            self.E_d, self.E_s, self.E_r = data['E_d'], data['E_s'], data['E_r']
            self.calibrated = True
            return True
        except:
            return False


# --- HARDWARE INTERFACE: NanoVNA ---
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
            print(f"VNA Connection Error: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open: self.ser.close()
        self.connected = False

    def _run_single_chunk(self, start_hz, stop_hz):
        self.ser.read_all()
        cmd = f"sweep {int(start_hz)} {int(stop_hz)}\r"
        self.ser.write(cmd.encode())
        time.sleep(0.4)
        self.ser.write(b"data 0\r")

        raw_lines = self.ser.readlines()
        data = []
        for line in raw_lines:
            line = line.decode('utf-8', errors='ignore').strip()
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
            if points <= 101:
                data = self._run_single_chunk(start_hz, stop_hz)
                if len(data) == 0: return None, None
                freqs = np.linspace(start_hz, stop_hz, len(data))
                return freqs, np.array(data)
            else:
                return None, None
        except Exception as e:
            print(f"Sweep Error: {e}")
            return None, None


# --- GUI MAIN WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Examiner (Scaled)")
        self.resize(1600, 950)

        self.vna = NanoVNA()
        self.motor = ThorlabsMotor()
        self.calibrator = SOLTCalibrator()

        self.last_freqs = None
        self.last_raw_data = None
        self.current_data_calibrated = None
        self.master_data_buffer = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignTop)
        main_layout.addLayout(sidebar, stretch=1)

        # 1. VNA
        grp_conn = QGroupBox("VNA Connection")
        layout_conn = QVBoxLayout()
        self.combo_ports = QComboBox()
        self.btn_refresh = QPushButton("Refresh Ports")
        self.btn_connect = QPushButton("Connect VNA")
        self.btn_connect.setCheckable(True)
        layout_conn.addWidget(self.combo_ports)
        layout_conn.addWidget(self.btn_refresh)
        layout_conn.addWidget(self.btn_connect)
        grp_conn.setLayout(layout_conn)
        sidebar.addWidget(grp_conn)

        # 2. Motor
        grp_motor = QGroupBox("Thorlabs Motor")
        layout_motor = QVBoxLayout()
        self.combo_motor = QComboBox()
        self.btn_refresh_motor = QPushButton("Scan Devices")
        self.btn_connect_motor = QPushButton("Connect Motor")
        self.btn_connect_motor.setCheckable(True)
        self.lbl_motor_pos = QLabel("Pos: ---")
        layout_motor.addWidget(self.combo_motor)
        layout_motor.addWidget(self.btn_refresh_motor)
        layout_motor.addWidget(self.btn_connect_motor)
        layout_motor.addWidget(self.lbl_motor_pos)
        grp_motor.setLayout(layout_motor)
        sidebar.addWidget(grp_motor)

        # 3. Frequency
        grp_set = QGroupBox("Frequency Settings")
        layout_set = QFormLayout()
        self.input_start = QLineEdit("1")
        self.input_stop = QLineEdit("900")
        layout_set.addRow("Start (MHz):", self.input_start)
        layout_set.addRow("Stop (MHz):", self.input_stop)
        grp_set.setLayout(layout_set)
        sidebar.addWidget(grp_set)

        # 4. Calibration
        grp_cal = QGroupBox("Calibration (SOLT)")
        layout_cal = QVBoxLayout()
        hbox_cal_btns = QHBoxLayout()
        self.btn_cal_short = QPushButton("SHORT")
        self.btn_cal_open = QPushButton("OPEN")
        self.btn_cal_load = QPushButton("LOAD")
        hbox_cal_btns.addWidget(self.btn_cal_short)
        hbox_cal_btns.addWidget(self.btn_cal_open)
        hbox_cal_btns.addWidget(self.btn_cal_load)

        hbox_cal_io = QHBoxLayout()
        self.btn_save_cal = QPushButton("Save Cal")
        self.btn_load_cal = QPushButton("Load Cal")
        hbox_cal_io.addWidget(self.btn_save_cal)
        hbox_cal_io.addWidget(self.btn_load_cal)

        self.chk_apply_cal = QCheckBox("Apply Calibration")
        self.chk_apply_cal.setEnabled(False)
        layout_cal.addLayout(hbox_cal_btns)
        layout_cal.addLayout(hbox_cal_io)
        layout_cal.addWidget(self.chk_apply_cal)
        grp_cal.setLayout(layout_cal)
        sidebar.addWidget(grp_cal)

        # 5. Vertical Scan (UPDATED WITH SCALE)
        grp_scan = QGroupBox("Vertical Scan")
        layout_scan = QFormLayout()
        self.input_z_start = QLineEdit("0")
        self.input_z_stop = QLineEdit("10")
        self.input_z_step = QLineEdit("1.0")

        # --- NEW SCALING INPUT ---
        self.input_scale = QLineEdit("125")
        self.input_scale.setToolTip("Set to 34304 if motor moves tiny steps. Set to 1.0 if using Real World Units.")

        self.chk_round_trip = QCheckBox("Round Trip (Hysteresis)")
        self.btn_test_scan = QPushButton("Test Scan (Current Pos)")
        self.btn_test_scan.setStyleSheet("background-color: #f0f0f0;")
        self.btn_run_scan = QPushButton("RUN VERTICAL SCAN")
        self.btn_run_scan.setFixedHeight(50)
        self.btn_run_scan.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        layout_scan.addRow("Start Height (mm):", self.input_z_start)
        layout_scan.addRow("Stop Height (mm):", self.input_z_stop)
        layout_scan.addRow("Step Size (mm):", self.input_z_step)
        layout_scan.addRow("Scale Factor:", self.input_scale)  # Added
        layout_scan.addRow(self.chk_round_trip)
        layout_scan.addRow(self.btn_test_scan)
        layout_scan.addRow(self.btn_run_scan)
        grp_scan.setLayout(layout_scan)
        sidebar.addWidget(grp_scan)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("font-weight: bold; color: blue;")
        sidebar.addWidget(self.lbl_status)
        sidebar.addStretch()

        # --- PLOTS ---
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addLayout(plot_layout, stretch=4)

        gs = self.fig.add_gridspec(2, 2)
        self.ax_mag = self.fig.add_subplot(gs[0, 0])
        self.ax_smith = self.fig.add_subplot(gs[0, 1])
        self.ax_phase = self.fig.add_subplot(gs[1, 0])
        self.ax_nyquist = self.fig.add_subplot(gs[1, 1])
        self.init_plots()

        # --- SIGNALS ---
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_vna)
        self.btn_refresh_motor.clicked.connect(self.refresh_motor_list)
        self.btn_connect_motor.clicked.connect(self.toggle_motor)

        self.btn_cal_short.clicked.connect(lambda: self.run_cal_step('short'))
        self.btn_cal_open.clicked.connect(lambda: self.run_cal_step('open'))
        self.btn_cal_load.clicked.connect(lambda: self.run_cal_step('load'))
        self.btn_save_cal.clicked.connect(self.save_calibration)
        self.btn_load_cal.clicked.connect(self.load_calibration)
        self.chk_apply_cal.toggled.connect(self.refresh_plot_view)

        self.btn_test_scan.clicked.connect(self.run_test_scan)
        self.btn_run_scan.clicked.connect(self.run_vertical_scan)

        self.refresh_ports()
        self.refresh_motor_list()

    def init_plots(self):
        for ax in [self.ax_mag, self.ax_smith, self.ax_phase, self.ax_nyquist]:
            ax.clear()
            ax.grid(True)
        self.ax_mag.set_title("Log Mag (dB)")
        self.ax_smith.set_title("Smith Chart")
        self.ax_phase.set_title("Phase (Deg)")
        self.ax_nyquist.set_title("Nyquist")

    # --- HARDWARE CONNECT ---
    def refresh_ports(self):
        self.combo_ports.clear()
        for p in list_ports.comports():
            self.combo_ports.addItem(f"{p.device} - {p.description}", userData=p.device)

    def toggle_vna(self):
        if self.btn_connect.isChecked():
            if self.vna.connect(self.combo_ports.currentData()):
                self.btn_connect.setText("Disconnect VNA")
                self.lbl_status.setText("VNA Connected")
            else:
                self.btn_connect.setChecked(False)
        else:
            self.vna.disconnect()
            self.btn_connect.setText("Connect VNA")

    def refresh_motor_list(self):
        self.combo_motor.clear()
        for d in self.motor.find_devices():
            self.combo_motor.addItem(d)

    def toggle_motor(self):
        if self.btn_connect_motor.isChecked():
            s_no = self.combo_motor.currentText()
            if not s_no: return
            self.lbl_status.setText(f"Connecting Motor {s_no}...")
            QApplication.processEvents()

            if self.motor.connect(s_no):
                self.btn_connect_motor.setText("Disconnect Motor")
                self.lbl_status.setText("Motor Connected. Homing...")
                QApplication.processEvents()
                self.motor.home()
                self.lbl_status.setText("Motor Ready")
                self.lbl_motor_pos.setText(f"Pos: {self.motor.get_position():.3f}")
            else:
                self.btn_connect_motor.setChecked(False)
                self.lbl_status.setText("Connection Failed")
        else:
            self.motor.disconnect()
            self.btn_connect_motor.setText("Connect Motor")

    # --- CALIBRATION ---
    def run_cal_step(self, std):
        if not self.vna.connected: return
        try:
            start = float(self.input_start.text()) * 1e6
            stop = float(self.input_stop.text()) * 1e6
        except:
            return

        self.lbl_status.setText(f"Measuring {std.upper()}...")
        QApplication.processEvents()

        f, d = self.vna.sweep(start, stop)
        if d is not None:
            self.calibrator.measure_standard(std, d)
            getattr(self, f"btn_cal_{std}").setStyleSheet("background-color: #aaffaa;")
            self.lbl_status.setText(f"{std.upper()} Captured.")
            if self.calibrator.compute_coefficients():
                self.chk_apply_cal.setEnabled(True)
                self.chk_apply_cal.setChecked(True)

    def save_calibration(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Cal", "nano.cal.npz", "NPZ (*.npz)")
        if fname: self.calibrator.save_calibration(fname)

    def load_calibration(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Cal", "", "NPZ (*.npz)")
        if fname:
            if self.calibrator.load_calibration(fname):
                self.chk_apply_cal.setEnabled(True)
                self.chk_apply_cal.setChecked(True)
                for btn in [self.btn_cal_short, self.btn_cal_open, self.btn_cal_load]:
                    btn.setStyleSheet("background-color: #aaffaa;")

    # --- TEST SCAN ---
    def run_test_scan(self):
        if not self.vna.connected: return
        try:
            start = float(self.input_start.text()) * 1e6
            stop = float(self.input_stop.text()) * 1e6
        except:
            return

        self.lbl_status.setText("Running Test Scan...")
        self.btn_test_scan.setEnabled(False)
        QApplication.processEvents()

        freqs, raw_data = self.vna.sweep(start, stop)
        if raw_data is not None:
            self.last_freqs = freqs
            self.last_raw_data = raw_data
            self.refresh_plot_view()
            self.lbl_status.setText("Test Scan Complete")
        self.btn_test_scan.setEnabled(True)

    # --- VERTICAL SCAN ---
    def run_vertical_scan(self):
        if not self.vna.connected or not self.motor.connected:
            QMessageBox.critical(self, "Error", "Hardware not connected")
            return

        fname, _ = QFileDialog.getSaveFileName(self, "Save Scan", f"scan_{int(time.time())}.csv", "CSV (*.csv)")
        if not fname: return

        try:
            z_start = float(self.input_z_start.text())
            z_stop = float(self.input_z_stop.text())
            z_step = float(self.input_z_step.text())
            scale = float(self.input_scale.text())
        except:
            return

        # Apply Scale to Motor Class
        self.motor.scale_factor = scale

        freq_start = float(self.input_start.text()) * 1e6
        freq_stop = float(self.input_stop.text()) * 1e6

        num_steps = int(abs(z_stop - z_start) / z_step) + 1
        heights = np.linspace(z_start, z_stop, num_steps)
        scan_points = [(h, "Forward") for h in heights]
        if self.chk_round_trip.isChecked():
            scan_points += [(h, "Reverse") for h in heights[::-1][1:]]

        self.master_data_buffer = []
        self.btn_run_scan.setEnabled(False)

        for i, (target_z, direction) in enumerate(scan_points):
            self.lbl_status.setText(f"Moving to {target_z:.2f} ({direction})...")
            QApplication.processEvents()

            self.motor.move_to(target_z)
            time.sleep(0.1)

            while self.motor.is_moving():
                QApplication.processEvents()
                time.sleep(0.02)
            current_pos = self.motor.get_position()
            # Update position label (unscaled for user readability)
            self.lbl_motor_pos.setText(f"Pos: {self.motor.get_position():.3f}")

            self.lbl_status.setText(f"Sweeping at {target_z:.2f}...")
            QApplication.processEvents()
            time.sleep(0.3)

            freqs, raw_data = self.vna.sweep(freq_start, freq_stop)
            if raw_data is not None:
                self.last_freqs = freqs
                self.last_raw_data = raw_data
                self.refresh_plot_view()

                if self.chk_apply_cal.isChecked() and self.calibrator.calibrated:
                    final_data = self.calibrator.apply_correction(raw_data)
                else:
                    final_data = raw_data

                for f, s11 in zip(freqs, final_data):
                    z = 50 * (1 + s11) / (1 - s11)
                    self.master_data_buffer.append({
                        "Frequency_Hz": f,
                        "Height_mm": target_z,
                        "Actual_Pos_Scaled": self.motor.get_position(),
                        "Direction": direction,
                        "S11_Real": np.real(s11),
                        "S11_Imag": np.imag(s11),
                        "Z_Real": np.real(z),
                        "Z_Imag": np.imag(z)
                    })

        df = pd.DataFrame(self.master_data_buffer)
        df.to_csv(fname, index=False)
        self.lbl_status.setText("Scan Complete")
        self.btn_run_scan.setEnabled(True)
        QMessageBox.information(self, "Done", f"Saved {len(df)} rows.")

    def refresh_plot_view(self):
        if self.last_raw_data is None: return

        # Decide which data to plot (Calibrated or Raw)
        if self.chk_apply_cal.isChecked() and self.calibrator.calibrated:
            data = self.calibrator.apply_correction(self.last_raw_data)
            col = 'g-'
        else:
            data = self.last_raw_data
            col = 'b-'

        # Clear
        for ax in [self.ax_mag, self.ax_smith, self.ax_phase, self.ax_nyquist]:
            ax.clear()
            ax.grid(True)

        # Re-title
        self.ax_mag.set_title("Log Mag (dB)")
        self.ax_smith.set_title("Smith Chart")

        # Plot
        self.ax_mag.plot(self.last_freqs / 1e6, 20 * np.log10(np.abs(data)), col)
        self.ax_smith.plot(np.real(data), np.imag(data), 'r-')
        self.ax_phase.plot(self.last_freqs / 1e6, np.angle(data, deg=True), 'purple')
        z = 50 * (1 + data) / (1 - data)
        self.ax_nyquist.plot(np.real(z), -np.imag(z), 'orange')

        self.canvas.draw()

    def closeEvent(self, event):
        if self.motor.connected: self.motor.disconnect()
        if self.vna.connected: self.vna.disconnect()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
