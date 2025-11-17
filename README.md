# Real-Time Cardiac Simulator

A real-time cardiac simulator built with PyQt6 and pyqtgraph. It visualizes a coupled simulation of a synthetic ECG (McSharry model) and a cardiovascular hemodynamic model (Simaan-style model).


## Preview

![Real-Time Cardiac Simulator Demo](assets/ECG_and_Simaan_MODEL.gif)

# Features

- Real-Time Visualization: Plots multiple physiological signals simultaneously using the high-performance pyqtgraph library.

- Coupled Models: A synthetic ECG generator (McSharry) is coupled with a cardiovascular hemodynamic model (Simaan), where the ECG's R-peak triggers the cardiac cycle.

- Dynamic Controls: The heart rate (BPM) can be adjusted in real-time, and the simulation parameters will update accordingly.

- (unstable) Audible Feedback: Plays a "beep" sound synchronized with the ECG R-peak, similar to a real patient monitor.

- Responsive GUI: The simulation runs in a separate QThread to ensure the PyQt6 user interface remains fast and responsive.

# Signals Plotted:

1. Synthetic ECG

2. Ventricular Elastance ($E(t)$)

3. Ventricular Pressure ($Pve$)

4. Aortic Pressure ($Pao$)

5. Atrial Pressure ($Pae$)

6. Ventricular Volume ($Vve$)

7. Aortic Flow ($Q_a$)

# Scientific Models

This simulator couples two different dynamical models:

ECG (McSharry, 2003): The synthetic electrocardiogram is generated using the McSharry dynamical model. This model uses a set of three ordinary differential equations (ODEs) to create a realistic, noise-free ECG waveform. The position of the trajectory in its phase space determines the P, Q, R, S, and T waves.

Hemodynamics (Simaan-style): The cardiovascular system (CVS) is simulated using a lumped-parameter model. It models the left ventricle, aorta, and systemic circulation using a system of ODEs describing pressure, volume, and flow. The ventricular pressure is determined by a time-varying elastance (E(t)) function, which represents the contractility of the heart muscle.

Model Coupling: The simulation is coupled by using the phase of the McSharry model. When the McSharry model's trajectory crosses a specific angle (representing the R-peak), the local timer for the hemodynamic model is reset, triggering the start of a new systolic phase.

# Requirements

This project requires Python 3 and the following libraries:

- PyQt6 (for the GUI and multimedia)

- pyqtgraph (for the real-time plots)

- NumPy (for numerical operations)

- SciPy (for sound file generation)

# Installation

Clone the repository: 
``` bash
git clone https://github.com/nivuciis/Human-Cardiac-System-Modelling
cd Human-Cardiac-System-Modelling
```

Install the required packages:

* A requirements.txt file is included, but you can easily install the dependencies using pip:
```bash
pip install PyQt6 pyqtgraph numpy scipy
```
or 
```bash
    pip install -r requirements.txt
```
* ! Its highly recommended to use a virtual environment. 



# Usage

To run the simulator, execute the main Python script Model.py:
```bash
python main.py
```

Click "Start" to begin the simulation.

Use the "Heart Rate (BPM)" spinbox to change the heart rate in real-time.

Click "Stop" to end the simulation.

The script will automatically generate a beep.wav file in the same directory if one does not already exist.

# Code Overview

SimulationWorker(QObject): This class runs in a separate QThread. It contains the core logic for:

The McSharry and Simaan-style ODE models.

The RK4 (Runge-Kutta) integrators for solving the ODEs.

The main simulation loop and real-time synchronization.

Emitting data to the GUI via pyqtSignal.

MainWindow(QMainWindow): This class sets up the entire PyQt6 GUI, including all pyqtgraph plots. It handles user input (buttons, spinbox) and has slots to receive data from the SimulationWorker to update the plots.

create_beep_file(): A utility function that generates a simple .wav file for the cardiac beep on the first run.

# License
This project is licensed under the MIT License.
Which means : 
* Use it for free: For personal, educational, or commercial projects.

* Modify it: You can change the code to fit your needs.

* Distribute it: you can share the original code or yours modified versions.

* Sublicense it: you can incorporate this code into your own project and release it under a different license (even a paid, commercial one).