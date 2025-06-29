from .fastflight2_utils import FastFlight64

import sys
import time
import numpy as np
import matplotlib
from typing import Tuple
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QLabel, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox
)

class ScopeView(QMainWindow):
    """Main window of the ScopeView Application"""

    def __init__(self, ff2: FastFlight64):
        super().__init__()
        self.ff2 = ff2
        self.connected = self.ff2.is_connected()
        if self.connected:
            self.acq_running = self.ff2.is_acq_running()
        else:
            self.acq_running = False
        
        # Configuration parameters
        self.refresh_rate = 50  # mainloop refresh rate (ms)
        self.min_plot_update_interval = 100  # minimum plot update interval (ms)
        
        # State tracking
        self.settings_changed = False
        self.last_plot_update = 0
        self.updating_settings = False  # Flag to prevent recursive updates
        
        self._make_gui()

        # If we are connected already, disconnect
        if self.connected:
            self._disconnect_ff2()

        # Timer for main loop
        self.timer = QTimer()
        self.timer.setInterval(self.refresh_rate)
        self.timer.timeout.connect(self._mainloop)
        self.timer.start()

    def _make_gui(self):
        """Set up the GUI components"""
        WIDTH = 100
        BUTTON_SPACING = 20
        MIN_SIZE = (900, 500)

        # Time Resolution
        self.time_res_label = QLabel("Time Resolution:")
        self.time_res_select = QComboBox()
        self.time_res_select.setFixedWidth(WIDTH)
        self.time_res_select.addItems([
            "250 ps (Interlaced)",
            "250 ps (Interpolated)",
            "500 ps",
            "1 ns",
            "2 ns"
        ])

        # Record Length
        self.record_len_label = QLabel("Record Length (us):")
        self.record_len_input = QDoubleSpinBox()
        self.record_len_input.setFixedWidth(WIDTH)
        self._update_record_len_spinbox()

        # Connect time resolution changes to update record length range
        self.time_res_select.currentIndexChanged.connect(self._update_record_len_spinbox)
        self.time_res_select.currentIndexChanged.connect(self._mark_settings_changed)

        # Averages
        self.averages_label = QLabel("Num Averages:")
        self.averages_input = QSpinBox()
        self.averages_input.setFixedWidth(WIDTH)
        self.averages_input.setRange(1, 65535)
        self.averages_input.setSingleStep(1)

        # Time Offset
        self.time_offset_label = QLabel("Time Offset (us):")
        self.time_offset_input = QDoubleSpinBox()
        self.time_offset_input.setFixedWidth(WIDTH)
        self.time_offset_input.setRange(0, 1000)
        self.time_offset_input.setSingleStep(0.016)

        # Input Offset
        self.input_offset_label = QLabel("Input Offset (V):")
        self.input_offset_input = QDoubleSpinBox()
        self.input_offset_input.setFixedWidth(WIDTH)
        self.input_offset_input.setRange(-0.25, 0.25)
        self.input_offset_input.setSingleStep(30e-6)

        # Correlated Noise Subtraction
        self.enable_cns_label = QLabel("Enable CNS:")
        self.enable_cns_select = QComboBox()
        self.enable_cns_select.setFixedWidth(WIDTH)
        self.enable_cns_select.addItems(["OFF", "ON"])

        # Precision Enhance Enable
        self.prec_enh_label = QLabel("Precision Enhance:")
        self.prec_enh_select = QComboBox()
        self.prec_enh_select.setFixedWidth(WIDTH)
        self.prec_enh_select.addItems(["OFF", "ON"])

        # Trigger Edge
        self.trigger_edge_label = QLabel("Trigger Edge:")
        self.trigger_edge_select = QComboBox()
        self.trigger_edge_select.setFixedWidth(WIDTH)
        self.trigger_edge_select.addItems(["Rising", "Falling"])

        # Trigger Input Threshold
        self.trigger_threshold_label = QLabel("Trigger Threshold (V):")
        self.trigger_threshold_input = QDoubleSpinBox()
        self.trigger_threshold_input.setFixedWidth(WIDTH)
        self.trigger_threshold_input.setRange(-2.5, 2.5)
        self.trigger_threshold_input.setSingleStep(0.01)

        # Connect all inputs to settings change handler
        self.record_len_input.valueChanged.connect(self._mark_settings_changed)
        self.averages_input.valueChanged.connect(self._mark_settings_changed)
        self.time_offset_input.valueChanged.connect(self._mark_settings_changed)
        self.input_offset_input.valueChanged.connect(self._mark_settings_changed)
        self.enable_cns_select.currentIndexChanged.connect(self._mark_settings_changed)
        self.prec_enh_select.currentIndexChanged.connect(self._mark_settings_changed)
        self.trigger_edge_select.currentIndexChanged.connect(self._mark_settings_changed)
        self.trigger_threshold_input.valueChanged.connect(self._mark_settings_changed)

        # Scope Settings Layout
        self.scope_settings_layout = QVBoxLayout()
        self.labels = [
            self.time_res_label,
            self.record_len_label,
            self.averages_label,
            self.time_offset_label,
            self.input_offset_label,
            self.enable_cns_label,
            self.prec_enh_label,
            self.trigger_edge_label,
            self.trigger_threshold_label
        ]
        self.inputs = [
            self.time_res_select,
            self.record_len_input,
            self.averages_input,
            self.time_offset_input,
            self.input_offset_input,
            self.enable_cns_select,
            self.prec_enh_select,
            self.trigger_edge_select,
            self.trigger_threshold_input
        ]
        for i in range(len(self.labels)):
            self.scope_settings_layout.addWidget(self.labels[i])
            self.scope_settings_layout.addWidget(self.inputs[i])

        self.scope_settings_layout.addStretch(1)

        # Start/Stop Acquisition Button
        self.start_stop_acq_button = QPushButton()
        self._update_acq_button_appearance()

        # Connect/Disconnect Buttons only
        self.connect_button = QPushButton("Connect")
        self.connect_button.setFixedWidth(WIDTH)
        
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setFixedWidth(WIDTH)

        # Status label to show when settings are being applied
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green;")

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addSpacing(BUTTON_SPACING)
        self.buttons_layout.addWidget(self.status_label)
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.start_stop_acq_button)
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.connect_button)
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.disconnect_button)
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addSpacing(BUTTON_SPACING)

        # Matplotlib Figure Canvas
        self.canvas = ScopeCanvas()

        # Window Layout
        self.canvas_layout = QHBoxLayout()
        self.canvas_layout.addLayout(self.scope_settings_layout)
        self.canvas_layout.addWidget(self.canvas)
        
        self.control_layout = QVBoxLayout()
        self.control_layout.addLayout(self.canvas_layout)
        self.control_layout.addLayout(self.buttons_layout)

        self.container = QWidget()
        self.container.setLayout(self.control_layout)
        self.setCentralWidget(self.container)
        self.setMinimumSize(*MIN_SIZE)
        self.setWindowTitle("FASTFLIGHT-2 ScopeView")

        # Connect Buttons
        self.connect_button.clicked.connect(self._connect_ff2)
        self.disconnect_button.clicked.connect(self._disconnect_ff2)
        self.start_stop_acq_button.clicked.connect(self._toggle_acquisition)

        # Get everything up to date
        self._fetch_settings()
        self._update_connection_status()
    
    def _update_acq_button_appearance(self):
        """Update the acquisition button appearance based on current state"""
        if not self.connected:
            self.start_stop_acq_button.setText("Start Acquisition")
            self.start_stop_acq_button.setStyleSheet("background-color: gray; color: white;")
            self.start_stop_acq_button.setEnabled(False)
        elif self.acq_running:
            self.start_stop_acq_button.setText("Stop Acquisition")
            self.start_stop_acq_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            self.start_stop_acq_button.setEnabled(True)
        else:
            self.start_stop_acq_button.setText("Start Acquisition")
            self.start_stop_acq_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.start_stop_acq_button.setEnabled(True)

    @pyqtSlot()
    def _toggle_acquisition(self):
        """Start or stop data acquisition"""
        if not self.connected:
            return
            
        try:
            if self.acq_running:
                self.ff2.stop_acq()
                self.acq_running = False
                self.status_label.setText("Acquisition stopped")
                self.status_label.setStyleSheet("color: orange;")
            else:
                self.ff2.start_acq()
                self.acq_running = True
                self.status_label.setText("Acquisition started")
                self.status_label.setStyleSheet("color: green;")
            
            self._update_acq_button_appearance()
            
        except Exception as e:
            self.status_label.setText(f"Acquisition error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    @pyqtSlot()        
    def _mark_settings_changed(self):
        """Mark that settings have changed and need to be applied"""
        if not self.updating_settings:  # Prevent recursive calls
            self.settings_changed = True
            self.status_label.setText("Settings pending...")
            self.status_label.setStyleSheet("color: orange;")
            
    @pyqtSlot()
    def _update_record_len_spinbox(self):
        """Update record length spinbox range based on time resolution"""
        res_selection = self.time_res_select.currentIndex()
        record_len_range = {
            0: [0, 375, 375/1.5e6],
            1: [0, 375, 375/1.5e6],
            2: [0, 750, 750/1.5e6],
            3: [0, 1.5e3, 1.5e3/1.5e6],
            4: [0, 3.0e3, 3.0e3/1.5e6]
        }

        current_val = self.record_len_input.value()
        new_min, new_max, new_step = record_len_range[res_selection]
        new_val = min(max(current_val, new_min), new_max)

        # Temporarily disable change tracking to prevent recursive updates
        self.updating_settings = True
        self.record_len_input.setRange(new_min, new_max)
        self.record_len_input.setSingleStep(new_step)
        self.record_len_input.setValue(new_val)
        self.updating_settings = False

    def _fetch_settings(self):
        """Fetch current settings from the instrument and update GUI"""
        if not self.connected:
            return
            
        try:
            self.updating_settings = True  # Prevent change tracking
            prot = self.ff2.get_protocol(0)
            gs = self.ff2.get_general_settings()
            self.time_res_select.setCurrentIndex(prot['SamplingInterval'])
            self.record_len_input.setValue(prot['RecordLength'])
            self.averages_input.setValue(prot['RecordsPerSpectrum'])
            self.time_offset_input.setValue(prot['TimeOffset'])
            self.input_offset_input.setValue(prot['VerticalOffset'])
            self.enable_cns_select.setCurrentIndex(prot['EnableCNS'])
            self.prec_enh_select.setCurrentIndex(prot['PrecEnhEnable'])
            self.trigger_edge_select.setCurrentIndex(gs['ExtTriggerInputEdge'])
            self.trigger_threshold_input.setValue(gs['ExtTriggerInputThreshold'])
            self.updating_settings = False
            
            self.status_label.setText("Settings synced")
            self.status_label.setStyleSheet("color: green;")
        except Exception as e:
            self.updating_settings = False
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def _prot_kwarg_settings(self):
        """Get current protocol settings as kwargs for the instrument"""
        kwargs = {
            'SamplingInterval'  : self.time_res_select.currentIndex(),
            'RecordLength'      : self.record_len_input.value(),
            'RecordsPerSpectrum': self.averages_input.value(),
            'TimeOffset'        : self.time_offset_input.value(),
            'VerticalOffset'    : self.input_offset_input.value(),
            'EnableCNS'         : self.enable_cns_select.currentIndex(),
            'PrecEnhEnable'     : self.prec_enh_select.currentIndex()
        }
        return kwargs
    
    def _gs_kwarg_settings(self):
        """Get current general settings as kwargs for the instrument"""
        kwargs = {
            'ExtTriggerInputEdge'     : self.trigger_edge_select.currentIndex(),
            'ExtTriggerInputThreshold': self.trigger_threshold_input.value()
        }

        return kwargs

    def _update_settings(self):
        """Apply current GUI settings to the instrument"""
        if not self.connected:
            return
            
        try:
            self.status_label.setText("Applying settings...")
            self.status_label.setStyleSheet("color: blue;")
            
            self.ff2.set_protocol(0, **self._prot_kwarg_settings())
            self.ff2.set_general_settings(**self._gs_kwarg_settings())
            self.settings_changed = False
            
            self.status_label.setText("Settings applied")
            self.status_label.setStyleSheet("color: green;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    @pyqtSlot()
    def _connect_ff2(self):
        """Connect to the FastFlight instrument"""
        try:
            self.ff2.open()
            self._fetch_settings()  # Sync settings after connecting
            self._init_ff2_as_scope()
            self.connected = True
            self._update_connection_status()

        except Exception as e:
            self.status_label.setText(f"Connection failed: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    @pyqtSlot()
    def _disconnect_ff2(self):
        """Disconnect from the FastFlight instrument"""
        try:
            self.connected = False
            self.ff2.close()
            self._update_connection_status()
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: gray;")
        except Exception as e:
            self.status_label.setText(f"Disconnect error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def _update_connection_status(self):
        """Update GUI elements based on connection status"""
        # Only connect/disconnect buttons are affected
        self.connect_button.setDisabled(self.connected)
        self.disconnect_button.setDisabled(not self.connected)
        self._update_acq_button_appearance()

    def _init_ff2_as_scope(self):
        """Set up the FastFlight to act as a scope"""
        self.ff2.set_general_settings(
            ActiveProtoNumber = 0,
            ExtTriggerInputEdge = 0,
            ExtTriggerInputEnable = 1,
            ExtTriggerInputThreshold = 0.0
        )
        self.acq_running = False
        if self.ff2.is_acq_running():
            self.ff2.stop_acq()
        
    def _mainloop(self):
        """Main loop that handles settings updates and plot refreshing"""

        if not self.connected:
            return

        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Handle settings changes
        if self.settings_changed:
            self.ff2.stop_acq()
            self._update_settings()
            self.ff2.start_acq()
            self.last_plot_update = current_time
            return
        
        # If there's no acquisition going on, we are done
        if not self.acq_running:
            return
        
        # Handle plot updates (only if enough time has passed)
        plot_update_interval = max(self.min_plot_update_interval, 
            1e-3 * self.record_len_input.value() * self.averages_input.value())
        if (current_time - self.last_plot_update) >= plot_update_interval:
            try:
                x, y, d = self.ff2.get_data()
                
                # Translate y to volts properly
                y *= (0.5 / 255.0) / self.averages_input.value() 

                self.canvas.plot_waveform(x, y)
                self.last_plot_update = current_time
            except Exception as e:
                # Silently handle data acquisition errors to avoid spam
                pass

    def closeEvent(self, event):
        """Properly cleanup when the window is closed"""
        
        # Stop the timer
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        # Stop acquisition if running
        if hasattr(self, 'connected') and self.connected:
            try:
                if hasattr(self, 'acq_running') and self.acq_running:
                    self.ff2.stop_acq()
            except:
                pass  # Ignore errors during cleanup
            
            # Disconnect from instrument
            try:
                self.ff2.close()
            except:
                pass  # Ignore errors during cleanup

        # Clean up matplotlib canvas
        if hasattr(self, 'canvas'):
            try:
                self.canvas.fig.clear()
                self.canvas.close()
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()

        event.accept()


class ScopeCanvas(FigureCanvas):
    """
    Matplotlib Canvas to embed in the scope view window.
    """

    def __init__(self):
        FIGSIZE = (5, 4)
        DPI = 100
        FACE_COLOR = '#fafafa'

        # Set up figure
        self.fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=FACE_COLOR)
        self.axes = self.fig.add_subplot(111)
        self.decorate_axes()
        super().__init__(self.fig)

    def plot_waveform(self, x: np.ndarray, y: np.ndarray):
        """Plot the waveform on the canvas"""
        COLOR = 'g'
        LINEWIDTH = 1.0

        self.axes.clear()
        self.axes.plot(x, y, color=COLOR, linewidth=LINEWIDTH)
        self.decorate_axes()
        self.fig.tight_layout()
        self.draw()
        
    def decorate_axes(self):
        """Style the plot axes"""
        BACKGROUND_COLOR = '#000000'
        AXIS_FONT = 12
        TICK_LABEL_SIZE = 10

        self.axes.set_facecolor(BACKGROUND_COLOR)
        self.axes.grid(True, which='major', linestyle='-', 
                       linewidth=0.75, color='gray')
        self.axes.minorticks_on()
        self.axes.grid(True, which='minor', linestyle=':', 
                       linewidth=0.45, color='lightgray')
        self.axes.set_xlabel("Time [ns]", fontsize=AXIS_FONT)
        self.axes.set_ylabel("Analog In [V]", fontsize=AXIS_FONT)
        self.axes.tick_params(axis='both', which='major', 
                              labelsize=TICK_LABEL_SIZE)
        
    def closeEvent(self, event):
        """Clean up the canvas"""
        self.fig.clear()
        super().close()
        event.accept()

def FFScopeView(ff2: FastFlight64):
    """Launch the ScopeView application"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app_created = True
    else:
        app_created = False

    window = ScopeView(ff2)
    window.show()

    if app_created:
        app.exec_()
    else:
        while window.isVisible():
            app.processEvents()
