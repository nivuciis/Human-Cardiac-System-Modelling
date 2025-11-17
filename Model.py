import sys
import time
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
from collections import deque

#Pyqt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl

import pyqtgraph as pg

# Sound File 
BEEP_FILE = 'beep.wav'

def create_beep_file():
    """ Creates a simple 'beep.wav' file if it doesn't exist. """
    if os.path.exists(BEEP_FILE):
        return
    print(f"Creating sound file '{BEEP_FILE}'...")
    try:
        sample_rate = 44100
        duration_ms = 100
        frequency = 1000
        t = np.linspace(0., duration_ms / 1000., int(sample_rate * duration_ms / 1000.), endpoint=False)
        waveform = np.sin(frequency * 2 * np.pi * t)
        waveform_normalized = np.int16(waveform * 32767)
        write_wav(BEEP_FILE, sample_rate, waveform_normalized)
        print("File 'beep.wav' created.")
    except Exception as e:
        print(f"Error creating sound file: {e}")


class SimulationWorker(QObject):
    """
    Runs the coupled simulation in a separate thread.
    """
    data_updated = pyqtSignal(dict)
    beep_signal = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, fs=2000, gui_update_ms=50):
        super().__init__()
        self.is_running = False
        self.hr_bpm = 75.0
        
        self.fs = fs
        self.dt = 1.0 / self.fs
        
        self.gui_update_interval_sec = gui_update_ms / 1000.0
        self.steps_per_update = int(self.fs * self.gui_update_interval_sec)
        
        self.beep_triggered = False

    def set_hr(self, hr_bpm):
        self.hr_bpm = hr_bpm

    def stop(self):
        self.is_running = False
        
    
    # --- McSharry (ECG) Model ---
    def _mcsharry_derivs(self, state, omega):
        x, y, z = state
        params = {
            'P': [-np.pi/3, 1.2, 0.25], 'Q': [-np.pi/12, -5.0, 0.1],
            'R': [0.0, 30.0, 0.1], 'S': [np.pi/12, -7.5, 0.1],
            'T': [np.pi/2, 0.75, 0.4],
        }
        theta = np.arctan2(y, x)
        alpha = 1.0 - np.sqrt(x**2 + y**2)
        dxdt = alpha * x - omega * y
        dydt = alpha * y + omega * x
        
        dzdt_sum = 0
        for i in params:
            theta_i, a_i, b_i = params[i]
            d_theta = (theta - theta_i + np.pi) % (2 * np.pi) - np.pi
            dzdt_sum += a_i * d_theta * np.exp(-d_theta**2 / (2 * b_i**2))
        
        dzdt = -dzdt_sum - z
        return np.array([dxdt, dydt, dzdt])

    # Hemodynamics Model 
    def _ramp(self, xi):
        return xi if xi >= 0 else 0.0

    def _elastance(self, t_local, t_c, p):
        T_max = 0.2 + 0.15 * t_c
        if T_max <= 0: T_max = 0.8 # Failsafe
        
        t_n = t_local / T_max
        tn_07 = max(0, t_n / 0.7)
        tn_117 = max(0, t_n / 1.17)
        
        term1 = (tn_07 ** 1.9) / (1 + tn_07 ** 1.9) if tn_07 > 0 else 0.0
        term2 = 1 / (1 + tn_117 ** 21.9) if tn_117 > 0 else 1.0
            
        E_n = 1.55 * term1 * term2
        return p["E_min"] + (p["E_max"] - p["E_min"]) * E_n

    def _simaan_derivs_volume_state(self, x, t_local, t_c, p):
        Vve, Pae, Pas, Pao, Q = x
        
        E_t = self._elastance(t_local, t_c, p)
        Pve = E_t * (Vve - p["V0"])

        Q_m = (1/p["R_m"]) * self._ramp(Pae - Pve)
        Q_a = (1/p["R_a"]) * self._ramp(Pve - Pao)
        
        dVve_dt = Q_m - Q_a
        dPae_dt = (1/p["C_r"]) * ( ((Pas - Pae)/p["R_s"]) - Q_m )
        dPas_dt = (1/p["C_s"]) * ( ((Pae - Pas)/p["R_s"]) + Q )
        dPao_dt = (1/p["C_a"]) * ( Q_a - Q )
        dQ_dt = (1/p["L_s"]) * ( Pao - Pas - p["R_c"]*Q )

        return np.array([dVve_dt, dPae_dt, dPas_dt, dPao_dt, dQ_dt]), Pve, Q_a, E_t


    def _rk4_step_ecg(self, x, dt, deriv_func, *args):
        """
        Correct RK4 integrator for ECG model.
        It does NOT unpack the return value from the derivative function.
        """
        k1 = deriv_func(x, *args) 
        k2 = deriv_func(x + 0.5 * dt * k1, *args)
        k3 = deriv_func(x + 0.5 * dt * k2, *args)
        k4 = deriv_func(x + dt * k3, *args)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    @pyqtSlot()
    def run(self):
        """ The main simulation loop. """
        print("Simulation thread started.")
        self.is_running = True
        
        CVS_PARAMS = {
            "R_s": 1.0, "R_m": 0.005, "R_a": 0.001, "R_c": 0.0398,
            "C_r": 4.4, "C_s": 1.33, "C_a": 0.08, "L_s": 0.0005,
            "E_max": 2.0, "E_min": 0.06, "V0": 10.0,
        }
        
        ecg_state = np.array([1.0, 0.0, 0.0])
        cvs_state = np.array([140.0, 5.0, 80.0, 80.0, 0.0])
        
        theta_prev = np.arctan2(ecg_state[1], ecg_state[0])
        t_local = 0.0
        current_time = 0.0
        
        t_c = 60.0 / self.hr_bpm
        
        while self.is_running:
            start_chunk_time = time.perf_counter()
            
            mean_rr = 60.0 / self.hr_bpm
            omega = (2 * np.pi) / mean_rr
            
            data_out = {}
            
            for _ in range(self.steps_per_update):
                
                k1, Pve1, Qa1, E1 = self._simaan_derivs_volume_state(cvs_state, t_local, t_c, CVS_PARAMS)
                k2, *_ = self._simaan_derivs_volume_state(cvs_state + 0.5 * self.dt * k1, t_local + 0.5*self.dt, t_c, CVS_PARAMS)
                k3, *_ = self._simaan_derivs_volume_state(cvs_state + 0.5 * self.dt * k2, t_local + 0.5*self.dt, t_c, CVS_PARAMS)
                k4, Pve_final, Qa_final, E_final = self._simaan_derivs_volume_state(cvs_state + self.dt * k3, t_local + self.dt, t_c, CVS_PARAMS)
                
                cvs_state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
            
                
                ecg_state = self._rk4_step_ecg(ecg_state, self.dt, self._mcsharry_derivs, omega)
                

                
                theta_new = np.arctan2(ecg_state[1], ecg_state[0])
                if theta_prev < 0 and theta_new >= 0:
                    t_local = 0.0
                    t_c = mean_rr
                
                theta_prev = theta_new
                t_local += self.dt
                current_time += self.dt
                
                
                z_ecg = ecg_state[2]
                if z_ecg > 10.0 and not self.beep_triggered:
                    self.beep_signal.emit()
                    self.beep_triggered = True
                elif z_ecg < 5.0:
                    self.beep_triggered = False

            #Prep data
            data_out['time'] = current_time
            data_out['ecg'] = ecg_state[2]
            data_out['Vve'] = cvs_state[0]
            data_out['Pae'] = cvs_state[1]
            data_out['Pas'] = cvs_state[2]
            data_out['Pao'] = cvs_state[3]
            data_out['Pve'] = Pve_final
            data_out['Q_a'] = Qa_final
            data_out['E_t'] = E_final
            
            self.data_updated.emit(data_out)
            
            # Sync to real-time 
            end_chunk_time = time.perf_counter()
            time_to_sleep = self.gui_update_interval_sec - (end_chunk_time - start_chunk_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        print("Simulation thread stopped.")
        self.finished.emit()


#GUI Application

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Cardiac Simulator (PyQt6)")
        self.setGeometry(100, 100, 1200, 900)
        
        self.plot_window_size = 10
        self.fs_gui = 20
        self.buffer_size = int(self.plot_window_size * self.fs_gui)
        
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.ecg_data = deque(maxlen=self.buffer_size)
        self.elastance_data = deque(maxlen=self.buffer_size)
        self.pve_data = deque(maxlen=self.buffer_size)
        self.pao_data = deque(maxlen=self.buffer_size)
        self.pae_data = deque(maxlen=self.buffer_size)
        self.vve_data = deque(maxlen=self.buffer_size)
        self.qa_data = deque(maxlen=self.buffer_size)
        
        self.thread = None
        self.worker = None
        
        create_beep_file()
        self.init_sound()
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Heart Rate (BPM):"))
        
        self.hr_input = QSpinBox()
        self.hr_input.setRange(40, 150)
        self.hr_input.setValue(75)
        control_layout.addWidget(self.hr_input)
        
        self.start_button = QPushButton("Start")
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(control_layout)
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)
        self.init_plots()
        
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.hr_input.valueChanged.connect(self.update_hr)

    def init_plots(self):
        self.plot1 = self.plot_widget.addPlot(title="ECG (McSharry)")
        self.plot1.setLabel('left', "Signal (mV)")
        self.ecg_curve = self.plot1.plot(pen='g')

        self.plot_widget.nextRow()

        self.plot2 = self.plot_widget.addPlot(title="Ventricular Elastance E(t)")
        self.plot2.setLabel('left', "E(t) (mmHg/mL)")
        self.elastance_curve = self.plot2.plot(pen='y')

        self.plot_widget.nextRow()
        
        self.plot3 = self.plot_widget.addPlot(title="Pressures (Simaan)")
        self.plot3.setLabel('left', "Pressure (mmHg)")
        self.plot3.addLegend()
        self.pve_curve = self.plot3.plot(pen='r', name="P_Ventricular (Pve)")
        self.pao_curve = self.plot3.plot(pen='c', name="P_Aortic (Pao)")
        self.pae_curve = self.plot3.plot(pen=(100, 100, 100), name="P_Atrial (Pae)")
        
        self.plot_widget.nextRow()
        
        self.plot4 = self.plot_widget.addPlot(title="Volume & Flow (Simaan)")
        self.plot4.setLabel('left', "Volume VE (mL)")
        self.vve_curve = self.plot4.plot(pen='w', name="Volume (Vve)")
        
        self.plot4_ax2 = pg.ViewBox()
        self.plot4.showAxis('right')
        self.plot4.scene().addItem(self.plot4_ax2)
        self.plot4.getAxis('right').linkToView(self.plot4_ax2)
        self.plot4_ax2.setXLink(self.plot4)
        self.plot4_ax2.setYRange(-100, 700)
        self.plot4.getAxis('right').setLabel("Flow (mL/s)", color='#00F')

        self.qa_curve = pg.PlotDataItem(pen='b', name="Flow (Q_a)")
        self.plot4_ax2.addItem(self.qa_curve)
        
        def update_plot4_ax2():
            self.plot4_ax2.setGeometry(self.plot4.vb.sceneBoundingRect())
            self.plot4_ax2.linkedViewChanged(self.plot4.vb, self.plot4_ax2.XAxis)
        self.plot4.vb.sigResized.connect(update_plot4_ax2)

    def init_sound(self):
        self.beep_sound = QSoundEffect()
        beep_path = os.path.join(os.getcwd(), BEEP_FILE)
        self.beep_sound.setSource(QUrl.fromLocalFile(beep_path))
        self.beep_sound.setVolume(0.5)

    def start_simulation(self):
        print("Starting simulation...")
        
        # Clear all data buffers on restart
        self.time_buffer.clear()
        self.ecg_data.clear()
        self.elastance_data.clear()
        self.pve_data.clear()
        self.pao_data.clear()
        self.pae_data.clear()
        self.vve_data.clear()
        self.qa_data.clear()
        
        self.worker = SimulationWorker()
        self.worker.set_hr(self.hr_input.value())
        
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        self.worker.data_updated.connect(self.update_plots)
        self.worker.beep_signal.connect(self.play_beep)
        self.worker.finished.connect(self.on_simulation_finished)
        
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_simulation(self):
        print("Stopping simulation...")
        if self.worker:
            self.worker.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_simulation_finished(self):
        print("Cleaning up thread.")
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    @pyqtSlot(dict)
    def update_plots(self, data):
        """
        Slot that receives data from the worker and updates the plots.
        """
        try:
            self.time_buffer.append(data['time'])
            self.ecg_data.append(data['ecg'])
            self.elastance_data.append(data['E_t'])
            self.pve_data.append(data['Pve'])
            self.pao_data.append(data['Pao'])
            self.pae_data.append(data['Pae'])
            self.vve_data.append(data['Vve'])
            self.qa_data.append(data['Q_a'])
            
            # This check is a safeguard
            if len(self.time_buffer) != len(self.elastance_data):
                return

            self.ecg_curve.setData(self.time_buffer, self.ecg_data)
            self.elastance_curve.setData(self.time_buffer, self.elastance_data)
            self.pve_curve.setData(self.time_buffer, self.pve_data)
            self.pao_curve.setData(self.time_buffer, self.pao_data)
            self.pae_curve.setData(self.time_buffer, self.pae_data)
            self.vve_curve.setData(self.time_buffer, self.vve_data)
            self.qa_curve.setData(self.time_buffer, self.qa_data)
        except Exception as e:
            print(f"Error updating plots: {e}")

    @pyqtSlot()
    def play_beep(self):
        self.beep_sound.play()
        
    @pyqtSlot(int)
    def update_hr(self, value):
        """Called when the spinbox is changed."""
        if self.worker:
            print(f"Updating HR to: {value} BPM")
            self.worker.set_hr(value)

    def closeEvent(self, event):
        """EnsRowsures the simulation thread is stopped when closing the window."""
        self.stop_simulation()
        event.accept()

# --- Run the App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())