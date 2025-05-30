import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass
from typing import List
import numpy as np
import math
import time, random

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams

from concurrent.futures import ThreadPoolExecutor
import os
import csv
import webbrowser
import pyvisa

class Test:
    """Object for holding all the data associated with a Test."""
    def __init__(self):  
        self.name = tk.StringVar()
        self.material = tk.StringVar()
        self.freq = tk.StringVar()
        self.freq_log = []
        self.notes = tk.StringVar()
        self.gauge_length = tk.StringVar(value="1.4")
        self.xmin = tk.StringVar(value="0")
        self.bin_val = tk.StringVar(value="1")
        self.data_file_name = ""
        self.info_file_name = ""
        self.last_written_index = 0


class TestInfoEntry(tk.Frame):
    """A widget for inputting Test information."""
    def __init__(self, parent: tk.Widget, handler: "TestHandler"):
        super().__init__(parent)
        self.handler: TestHandler = handler 
        self.build()
    
    def build(self):
        self.grid_columnconfigure(1, weight=1)

        # row 0 ---------------------------------------------
        name_lbl = tk.Label(self, text="Name:", anchor="e")
        name_lbl.grid(row=0, column=0, sticky="ew")
        self.name_ent = tk.Entry(self, textvariable=self.handler.test.name)
        self.name_ent.grid(row=0, column=1, sticky="ew")
        
        # row 1 ---------------------------------------------
        matr_lbl = tk.Label(self, text="Material:", anchor="e")
        matr_lbl.grid(row=1, column=0, sticky="ew")
        self.matr_ent = tk.Entry(self, textvariable=self.handler.test.material)
        self.matr_ent.grid(row=1, column=1, sticky="ew")

        # row 2 ---------------------------------------------
        freq_lbl = tk.Label(self, text="Period (s):", anchor="e")
        freq_lbl.grid(row=2, column=0, sticky="ew")
        self.freq_ent = tk.Entry(self, textvariable=self.handler.test.freq)
        self.freq_ent.grid(row=2, column=1, sticky="ew")

        # row 3 ---------------------------------------------
        gauge_length_lbl = tk.Label(self, text="Gauge Length (in):", anchor="e")
        gauge_length_lbl.grid(row=3, column=0, sticky="ew")
        self.gauge_length_ent = tk.Entry(self, textvariable=self.handler.test.gauge_length)
        self.gauge_length_ent.grid(row=3, column=1, sticky="ew")

        # row 4 ---------------------------------------------
        notes_lbl = tk.Label(self, text="Notes:", anchor="e")
        notes_lbl.grid(row=4, column=0, sticky="ew")
        self.notes_ent = tk.Entry(self, textvariable=self.handler.test.notes)
        self.notes_ent.grid(row=4, column=1, sticky="ew")

        # row 5 ---------------------------------------------
        xmin_lbl = tk.Label(self, text="x-min:", anchor="e")
        xmin_lbl.grid(row=5, column=0, sticky="ew")
        self.xmin_ent = tk.Entry(self, textvariable=self.handler.test.xmin)
        self.xmin_ent.grid(row=5, column=1, sticky="ew")
        self.xmin_ent.config(state="disabled")

        # row 6 ---------------------------------------------
        bin_lbl = tk.Label(self, text="Bin Size:", anchor="e")
        bin_lbl.grid(row=6, column=0, sticky="ew")
        self.bin_ent = tk.Entry(self, textvariable=self.handler.test.bin_val)
        self.bin_ent.grid(row=6, column=1, sticky="ew")
        self.bin_ent.config(state="disabled")


class StrainPlot(tk.Frame):
    """Renders data from a TestHandler as it is collected."""
    def __init__(self, parent: tk.Frame, handler: "TestHandler"):
        super().__init__(parent)
        self.handler = handler
        self.build()

    def build(self):
        self.fig, (self.strainplt, self.strainrateplt, self.temperatureplt) = plt.subplots(3,
            figsize=(6,6),
            dpi=100,
            constrained_layout=True,
            subplotpars=SubplotParams(left=0.5, bottom=0.1, right=0.95, top=0.95),
            sharex=True # x-axes are shared
        )

        self.strainplt.set_xlabel("Time (s)")
        self.strainplt.set_ylabel("True Strain")
        self.strainplt.grid(color="darkgrey", alpha=0.65, linestyle='dashed')
        self.strainplt.set_facecolor("w")
        self.strainplt.margins(0, tight=True)

        self.strainrateplt.set_xlabel("Time (s)")
        self.strainrateplt.set_ylabel("True Strain Rate")
        self.strainrateplt.grid(color="darkgrey", alpha=0.65, linestyle='dashed')
        self.strainrateplt.set_facecolor("w")
        self.strainrateplt.margins(0, tight=True)

        self.temperatureplt.set_xlabel("Time (s)")
        self.temperatureplt.set_ylabel("Temperature (C)")
        self.temperatureplt.grid(color="darkgrey", alpha=0.65, linestyle='dashed')
        self.temperatureplt.set_facecolor("w")
        self.temperatureplt.margins(0, tight=True)

        self.line1, = self.strainplt.plot([], [])
        self.line2, = self.strainrateplt.plot([], [])
        self.line3, = self.temperatureplt.plot([], [])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(sticky="nsew", pady=(40,0))
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.ani = FuncAnimation(self.fig, self.animate, interval=500, cache_frame_data=False) # Animation period
        self.handler.ani = self.ani

    def animate(self, interval):
        if self.handler.is_running:
            if self.handler.idx == 0:
                return



            # Number of points collected so far
            n = self.handler.idx

            # Slice into existing arrays (views, no copies)
            x_full    = self.handler.elapsed[:n]
            ts_full   = self.handler.trueStrain[:n]
            sr_full   = self.handler.strainRate[:n]
            temp_full = self.handler.temperature[:n]

            def bin_mean(arr, bin_size):
                length = arr.size
                # No binning if bin_size <=1 or too few points
                if bin_size <= 1 or length <= 1:
                    return arr
                m = length // bin_size
                # Full bins: reshape view and mean over axis=1
                full = arr[:m * bin_size].reshape(m, bin_size).mean(axis=1)
                # Handle leftover partial bin
                rem = length - m * bin_size
                if rem:
                    tail = np.array([arr[m * bin_size :].mean()])
                    return np.concatenate((full, tail))
                return full

            # Choose raw vs binned data
            bin_val = int(self.handler.test.bin_val)
            if bin_val >= 1 and bin_val < n:
                x   = bin_mean(x_full,    bin_val)
                ts  = bin_mean(ts_full,   bin_val)
                sr  = bin_mean(sr_full,   bin_val)
                temp= bin_mean(temp_full, bin_val)
            else:
                x, ts, sr, temp = x_full, ts_full, sr_full, temp_full


            # Set X-axis limits using first and last elements (efficient for ordered data)
            x_min = float(self.handler.test.xmin)
            x_max = self.handler.elapsed[self.handler.idx - 1]
            self.strainplt.set_xlim(x_min, x_max)  # Shared x-axis

            # Update each line’s data
            self.line1.set_data(x, ts)
            self.line2.set_data(x, sr)
            self.line3.set_data(x, temp)

            # Auto-scale Y axes
            def get_ylim(arr, padding=0.1):
                if arr.size == 0 or np.isnan(arr).all():  # Handle empty or all-NaN arrays
                    return (-0.1, 0.1)  # Default y-axis range
                arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
                if arr_min == arr_max:  # Prevent zero range
                    return (arr_min - 0.1, arr_max + 0.1)
                data_range = arr_max - arr_min
                pad = padding * data_range  # padding as a percentage of data range
                return (arr_min - pad, arr_max + pad)
    
            self.strainplt.set_ylim(*get_ylim(ts))
            self.strainrateplt.set_ylim(*get_ylim(sr))
            self.temperatureplt.set_ylim(*get_ylim(temp))


class TestControls(tk.Frame):
    """A widget for Test Controls."""
    def __init__(self, parent: tk.Widget, handler: "TestHandler"):
        super().__init__(parent)
        self.handler: TestHandler = handler 
        self.build()

    def build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # row 0 col 0 --------------------------------------
        self.start_btn = tk.Button(self)
        self.start_btn.configure(text="Start", state="disabled", command=self.handler.start_test)
        self.start_btn.grid(row=0, column=0, sticky="ew")

        # row 0 col 1 --------------------------------------
        self.stop_btn = tk.Button(self) 
        self.stop_btn.configure(text="Stop", state="disabled", command=self.handler.stop_test)
        self.stop_btn.grid(row=0, column=1, sticky="ew")

        # row 0 col 2 --------------------------------------
        self.pause_btn = tk.Button(self)
        self.pause_btn.configure(text="Pause", state="disabled", command=self.handler.toggle_pause)
        self.pause_btn.grid(row=0, column=2, sticky="ew")

        # row 5 col 0 --------------------------------------
        self.log_text = ScrolledText(
            self, background="white", height=25, width=44, state="disabled"
        )
        self.log_text.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.display("Welcome!")

        self.text_box = tk.Text(self, height=1, width=3)
        self.text_box.insert("1.0", "Calibration")
        self.text_box.tag_add("hyperlink", "1.0", "1.11")
        self.text_box.tag_config("hyperlink", foreground="blue", underline=True)
        self.text_box.tag_bind("hyperlink", "<Button-1>",lambda e: webbrowser.open("https://docs.google.com/document/d/1zfNkIwj9hVPKOLqsJrkodcGsiBNH8iFY/edit?usp=sharing&ouid=107579670681160493805&rtpof=true&sd=true"))
        self.text_box.config(state="disabled")
        self.text_box.grid(row=5, column=0, sticky="ew")

        # row 5 col 1 --------------------------------------
        self.connect_btn = tk.Button(self)
        self.connect_btn.configure(text="Connect I/O", state="normal", command=self.handler.connect_IO)
        self.connect_btn.grid(row=5, column=2, sticky="ew")

        

    def display(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "".join((msg, "\n")))
        self.log_text.configure(state="disabled")
        self.log_text.yview("end")


class TestHandler:
    def __init__(self, test_controls: TestControls = None, strainplot: StrainPlot = None, test_info_entry: TestInfoEntry = None, toolbar = None):
        self.root: tk.Tk = strainApp.ROOT
        self.test = Test()

        # Store instances
        self.test_controls = test_controls
        self.strainplot = strainplot
        self.test_info_entry = test_info_entry
        self.toolbar = toolbar

        # Initialize flags
        self.is_running = False
        self.request_stop = False
        self.paused = False
        self.views: List[tk.Widget] = []
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.ani = None
        self.last_save_time = time.time()
        self.last_read_time = time.time()
        self.last_check_time = time.time()

        self.firstStrain = 0
        self.testStarted = False
        
    def start_test(self):
        # Read the text entries (except notes)
        self.test.name = self.test_info_entry.name_ent.get()
        self.test.material = self.test_info_entry.matr_ent.get()
        self.test.freq = self.test_info_entry.freq_ent.get()
        self.test.gauge_length = self.test_info_entry.gauge_length_ent.get()
        self.test.xmin = self.test_info_entry.xmin_ent.get()
        self.test.bin_val = self.test_info_entry.bin_ent.get()

        # Try to convert frequency and gauge length into floats
        try:
            freq = float(self.test.freq)
        except ValueError:
            freq = 0
        try:
            gauge_length = float(self.test.gauge_length)
        except ValueError:
            gauge_length = 0
        
        # Require user to enter valid input before starting test
        if (
            (self.test.name and self.test.material)
            and (freq > 0) and (gauge_length > 0)
        ):
            #check status signal
            self.daq.write("VC3") # send 1 mA current output
            status = float(self.daq.query("AI0")) # channel 0
            print(status)
            if (abs(status) <= 0.001):
                self.test_controls.display("Please prepare machine for test.")
                return
            
            self.start_time = time.time()
            self.testStarted = True
            self.is_running = True
            self.paused = False

            # Disable the text boxes
            self.test_info_entry.name_ent.config(state="disabled")
            self.test_info_entry.matr_ent.config(state="disabled")
            self.test_info_entry.gauge_length_ent.config(state="disabled")

            # Enable editing of xmin and bin value
            self.test_info_entry.xmin_ent.config(state="normal")
            self.test_info_entry.bin_ent.config(state="normal")

            # Disable the start button and enable the stop and pause buttons
            if self.test_controls:
                self.test_controls.start_btn.configure(state="disabled")
                self.test_controls.stop_btn.configure(state="normal")
                self.test_controls.pause_btn.configure(state="normal")

            initial_capacity = 100
            self.capacity = initial_capacity # initial capacity
            self.idx = 0  # Current number of valid readings
            
            # Initialize arrays with NaN (to distinguish empty slots)
            self.timestamps = np.full(initial_capacity, np.nan, dtype=np.float64)
            self.elapsed = np.full(initial_capacity, np.nan, dtype=np.float32)
            self.displacement = np.full(initial_capacity, np.nan, dtype=np.float32)
            self.strain = np.full(initial_capacity, np.nan, dtype=np.float32)
            self.trueStrain = np.full(initial_capacity, np.nan, dtype=np.float32)
            self.strainRate = np.full(initial_capacity, np.nan, dtype=np.float32)
            self.temperature = np.full(initial_capacity, np.nan, dtype=np.float32)

            self.test.freq_log.append({"Period (s)": self.test.freq, "Timestamp (s)": 0})

            self.test.data_file_name = f"{self.test.name}_data.csv"
            self.test.info_file_name = f"{self.test.name}_info.csv"

            # for first strain readings before test
            displacementVoltage = float(self.daq.query("AI2")) # channel 2
            print(f"Displacement Voltage for First Strain: {displacementVoltage}")
            displacement = (0.04897 * displacementVoltage) + 0.53505
            strain = displacement / float(self.test.gauge_length)
            self.firstStrain = strain

            self.test_controls.display("Test started.")
            print("Started the test.")
            self.pool.submit(self.cont_test)
        else:
            self.test_controls.display("Please enter valid input.")

    def cont_test(self):
        self.is_running = True
        self.request_stop = False
        self.take_readings()

    def stop_test(self):
        self.request_stop = True
        self.test_controls.pause_btn.configure(state="disabled")
        self.test_controls.stop_btn.configure(state="disabled")
        self.ani.event_source.stop()

        print("Stopped the test.")
        self.test_controls.display("Test stopped.")
        self.test_info_entry.freq_ent.config(state="disabled")
        self.test_info_entry.gauge_length_ent.config(state="disabled")
        self.test_info_entry.notes_ent.config(state="disabled")
        self.test_info_entry.xmin_ent.config(state="disabled")
        self.test_info_entry.bin_ent.config(state="disabled")

        self.daq.close()
        print("DAQ closed")
        self.voltmeter.close()
        print("Voltmeter closed")

    def toggle_pause(self):
        """Toggle the pause/resume state."""
        self.paused = not self.paused
        if self.paused:
            print("Paused the test.")
            self.test_controls.display("Test paused.")
            self.test_controls.pause_btn.configure(text="Resume")
            self.ani.event_source.stop()

        else:
            print("Resumed the test.")
            self.test_controls.display("Test resumed.")
            self.test_controls.pause_btn.configure(text="Pause")
            self.ani.event_source.start()

    def connect_IO(self):
            try:
                self.rm = pyvisa.ResourceManager()
            except Exception as e:
                print(f"Error creating VISA Resource Manager: {e}")
                return
            
            try:
                resources = self.rm.list_resources()
                print(resources)
            except Exception as e:
                print(f"Error listing available resources: {e}")
                return

            # Open DAQ
            try:
                self.daq = self.rm.open_resource('GPIB0::9::INSTR')
                print("DAQ open")
            except Exception as e:
                print(f"Failed to open DAQ instrument: {e}")
                return

            # Open Voltmeter
            try:
                self.voltmeter = self.rm.open_resource('GPIB0::8::INSTR')
                print("Voltmeter open")
            except Exception as e:
                print(f"Failed to open voltmeter: {e}")
                return
            
            self.test_controls.connect_btn.configure(state="disabled")
            self.test_controls.start_btn.configure(state="normal")

            self.pool.submit(self.wait_for_start)

    def wait_for_start(self):
        while not self.testStarted:
            current_time = time.time()
            if (current_time - self.last_read_time) >= 3:
                '''status = float(self.daq.query("AI0")) # channel 0
                print(f"Status Voltage: {status}")''' #causing error when starting test
                displacementVoltage = float(self.daq.query("AI2")) # channel 2
                print(f"Displacement Voltage: {displacementVoltage}")
                temperatureVoltage = 1000 * float(self.voltmeter.query("?")) # channel 1
                print(f"Temperature Voltage: {temperatureVoltage}")
                self.last_read_time = current_time
        self.last_read_time = self.last_read_time - 1000 # to ensure a reading is taken once started

    def _resize_arrays(self, new_capacity):
        """Double array size while preserving existing data (amortized O(1) time)."""
        def resize(arr):
            new_arr = np.full(new_capacity, np.nan, dtype=arr.dtype)
            new_arr[:self.idx] = arr[:self.idx]
            return new_arr
        
        self.timestamps = resize(self.timestamps)
        self.elapsed = resize(self.elapsed)
        self.displacement = resize(self.displacement)
        self.strain = resize(self.strain)
        self.trueStrain = resize(self.trueStrain)
        self.strainRate = resize(self.strainRate)
        self.temperature = resize(self.temperature)
        self.capacity = new_capacity

    def take_readings(self):
        while self.is_running and not self.request_stop:
            current_time = time.time()
            if current_time - self.last_read_time >= float(self.test.freq): # Data acquisition period
                if self.idx >= self.capacity:
                    self._resize_arrays(2 * self.capacity)
                    print("Doubled size of arrays")
                
                # Take the readings
                self.timestamps[self.idx] = current_time
                self.elapsed[self.idx] = self.get_time(current_time)
                self.displacement[self.idx] = self.get_displacement()
                self.strain[self.idx] = self.get_strain(self.displacement[self.idx])
                self.trueStrain[self.idx] = self.get_true_strain(self.strain[self.idx])
                if self.idx < 1:
                    self.strainRate[self.idx] = 0
                else:
                    # Calculate the window of data points to use
                    start_idx = max(0, self.idx - 9)
                    t_window = self.elapsed[start_idx:self.idx + 1]
                    s_window = self.trueStrain[start_idx:self.idx + 1]

                    if len(t_window) > 1:
                        # If more than 1 point, fit a line to the data and compute strain rate (slope)
                        slope, _ = np.polyfit(t_window, s_window, 1)
                        self.strainRate[self.idx] = slope
                    else:
                        # If only 1 point, strain rate is undefined, set to 0 or any default value
                        self.strainRate[self.idx] = 0
                self.temperature[self.idx] = self.get_temperature()
                self.idx += 1

                self.last_read_time = current_time

            if current_time - self.last_check_time >= 3:
                # CHECK FOR STATUS SIGNAL
                status = float(self.daq.query("AI0")) # channel 0
                if (abs(status) <= 0.001):
                    self.test_controls.display("Test over")
                    self.save_to_csv() # update file up to end of test
                    self.stop_test()
                    return
                
                # CHECK FOR FREQUENCY/PERIOD
                temp = self.test_info_entry.freq_ent.get()
                # Try to convert frequency into float
                try:
                    freq = float(temp)
                    if freq != float(self.test.freq):
                        if (freq > 0):
                            self.test.freq = freq
                            self.test.freq_log.append({"Period (s)": self.test.freq, "Timestamp (s)": self.elapsed[self.idx - 1]})
                            self.test_controls.display(f"Period changed to {self.test.freq}s.")
                        else:
                            self.test_controls.display("Period must be a positive number.")
                except ValueError:
                    self.test_controls.display("Please enter valid period.")

                # CHECK FOR X-MIN
                temp = self.test_info_entry.xmin_ent.get()
                # Try to convert xmin into float
                try:
                    xmin = float(temp)
                    if xmin != float(self.test.xmin):
                        if (xmin >= 0 and xmin < self.elapsed[self.idx - 1]):
                            self.test.xmin = temp
                            self.test_controls.display(f"x-min changed to {xmin}s.")
                        else:
                            self.test_controls.display("x-min out of range.")
                except ValueError:
                    self.test_controls.display("Please enter valid x-min.")

                # CHECK FOR BIN_VAL
                temp = self.test_info_entry.bin_ent.get()
                # Try to convert bin value into int
                try:
                    bin_val = int(temp)
                    if bin_val != int(self.test.bin_val):
                        if (bin_val >= 1 and bin_val < self.idx):
                            self.test.bin_val = temp
                            self.test_controls.display(f"Bin Value changed to {bin_val}.")
                        else:
                            self.test_controls.display("Bin Value out of range.")
                except ValueError:
                    self.test_controls.display("Please enter valid Bin Value.")

                self.last_check_time = current_time

            if current_time - self.last_save_time >= max(5, float(self.test.freq)):
                # Print saved data to log (only elapsed time, true strain, true strain rate, and temperature)
                self.test_controls.display(f"Elapsed Time (s): {self.elapsed[self.idx - 1]:.2f}\nTrue Strain: {self.trueStrain[self.idx - 1]:.2f}\nTrue Strain Rate: {self.strainRate[self.idx - 1]:.2f}\nTemperature (C): {self.temperature[self.idx - 1]:.2f}")
                self.test_controls.display("="*44)

                # Save data to csv file
                self.save_to_csv()
                # Update last save time
                self.last_save_time = current_time

    def save_to_csv(self):
        # Get current data index and last saved index
        current_idx = self.idx
        start_idx = self.test.last_written_index
        
        # Only proceed if there's new data
        if start_idx < current_idx:
            # Extract new data slices directly from numpy arrays
            new_timestamps = self.timestamps[start_idx:current_idx]
            new_elapsed = self.elapsed[start_idx:current_idx]
            new_displacement = self.displacement[start_idx:current_idx]
            new_strain = self.strain[start_idx:current_idx]
            new_true_strain = self.trueStrain[start_idx:current_idx]
            new_strain_rate = self.strainRate[start_idx:current_idx]
            new_temp = self.temperature[start_idx:current_idx]

            fieldnames = ['Epoch Time (s)', 'Elapsed Time (s)', 'Displacement (in)', 'Engineering Strain', 'True Strain', 
                        'True Strain Rate (1/s)', 'Temperature (C)']
            
            with open(self.test.data_file_name, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # Write header only for empty files
                if file.tell() == 0:
                    writer.writeheader()

                # Write all new entries
                for i in range(len(new_elapsed)):
                    writer.writerow({
                        'Epoch Time (s)': new_timestamps[i],  
                        'Elapsed Time (s)': new_elapsed[i],
                        'Displacement (in)': new_displacement[i],
                        'Engineering Strain': new_strain[i],
                        'True Strain': new_true_strain[i],
                        'True Strain Rate (1/s)': new_strain_rate[i],
                        'Temperature (C)': new_temp[i]
                    })
            self.test.last_written_index = current_idx

        self.test.notes = self.test_info_entry.notes_ent.get()
        with open(self.test.info_file_name, mode='w', newline='') as file:
            file.write("="*50 + "\n")
            file.write(f"Name: {self.test.name}\n")
            file.write(f"Material: {self.test.material}\n")
            file.write(f"Gauge Length (in): {self.test.gauge_length}\n")
            file.write(f"Notes: {self.test.notes}\n")
            for entry in self.test.freq_log:
                file.write(f"Period Log: {entry['Period (s)']} at {entry['Timestamp (s)']}\n")
            file.write("="*50 + "\n")

    # -200°C to 0°C
    NEGATIVE_COEFFICIENTS = [
        0.000000e+00,   # c0
        2.5173462e+01,  # c1
        -1.1662878e+00, # c2
        -1.0833638e+00, # c3
        -8.9773540e-01, # c4
        -3.7342377e-01, # c5
        -8.6632643e-02, # c6
        -1.0450598e-02, # c7
        -5.1920577e-04  # c8
    ]

    # 0°C to 500°C
    POSITIVE_COEFFICIENTS = [
        0.0000000e+00,   # c0
        2.508355e+01,    # c1
        7.860106e-02,    # c2
        -2.503131e-01,   # c3
        8.315270e-02,    # c4
        -1.228034e-02,   # c5
        9.804036e-04,    # c6
        -4.413030e-05,   # c7
        1.057734e-06,    # c8
        -1.052755e-08    # c9
    ]

    # 500°C to 1372°C
    HIGH_COEFFICIENTS = [
    -1.318058e+02,  # c0
     4.830222e+01,  # c1
    -1.646031e+00,  # c2
     5.464731e-02,  # c3
    -9.650715e-04,  # c4
     8.802193e-06,  # c5
    -3.110810e-08   # c6
    ]

    def get_time(self, time):
        return time - self.start_time

    def get_displacement(self):
        displacementVoltage = float(self.daq.query("AI2")) # channel 2
        print(f"Displacement Voltage: {displacementVoltage}")
        displacement = (0.04897 * displacementVoltage) + 0.53505
        return displacement

    def get_strain(self, displacement):
        strain = displacement / float(self.test.gauge_length) - self.firstStrain
        return strain
    
    def get_true_strain(self, strain):
        true_strain = np.log(1 + strain)
        return true_strain
    
    def get_strain_rate(self, strainDiff, timeDiff):
        strain_rate = strainDiff / timeDiff
        return strain_rate

    def get_temperature(self):
        #temperatureVoltage = 1000 * float(self.daq.query("AI1")) # channel 1
        temperatureVoltage = 1000 * float(self.voltmeter.query("?")) # channel 1
        print(f"Temperature Voltage: {temperatureVoltage}")
        
        if temperatureVoltage < 0.0:
            coeffs = self.NEGATIVE_COEFFICIENTS
        elif 0.0 <= temperatureVoltage <= 20.644:
            coeffs = self.POSITIVE_COEFFICIENTS
        else:
            coeffs = self.HIGH_COEFFICIENTS
        
        if temperatureVoltage > 54.886 or temperatureVoltage < -5.891:
            print("WARNING: Temperature voltage out of range")

        temperature = 0.0
        for i, coeff in enumerate(coeffs):
            temperature += coeff * (temperatureVoltage ** i)
        return temperature


class MainFrame(tk.Frame):
    """Main Frame for the application."""
    def __init__(self, parent: tk.Frame):
        super().__init__(parent)
        self.parent: tk.Frame = parent
        self.handler = TestHandler(test_controls=None, test_info_entry=None, toolbar=None)
        self.strainplot: StrainPlot = None
        self.build()

    def build(self):
        """Builds the UI"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # row 0 ---------------------------------------------
        test_info = TestInfoEntry(self, self.handler)
        test_info.grid(row=0, column=0, sticky="new")
        self.handler.test_info_entry = test_info

        # row 0 ---------------------------------------------
        plt_frm = tk.Frame(self)
        self.strainplot = StrainPlot(plt_frm, self.handler)
        self.strainplot.grid(row=0, column=0, sticky="nsew")
        plt_frm.grid(row=0, column=1, rowspan=2)
        self.handler.strainplot = self.strainplot

        # above strainplot
        self.toolbar = NavigationToolbar2Tk(self.strainplot.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.place(x=600, y=0, relwidth=1)
        self.handler.toolbar = self.toolbar

        # row 1 ---------------------------------------------
        test_controls = TestControls(self, self.handler)
        test_controls.grid(row=1, column=0, sticky="nsew")
        self.handler.test_controls = test_controls


class strainApp(tk.Frame):
    """Core object for the application.
    Used to define widget styles."""
    def __init__(self, parent):
        super().__init__(parent)
        parent.resizable(True, True)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.tk_setPalette(background="#FAFAFA")
        
        self.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.close)

        MainFrame(self).grid(sticky="nsew")

    def close(self) -> None:
        '''self.handler.daq.close()
        print("DAQ closed")
        self.handler.voltmeter.close()
        print("Voltmeter closed")'''
        self.quit()


def main():
    """The Tkinter entry point of the program; enters mainloop."""
    root = tk.Tk()
    root.title("Creep Test")
    strainApp.ROOT = root
    root.geometry("1000x650") # window size
    root.withdraw() # temporarily hide window

    def get_load():
        while True:
            appliedLoad = simpledialog.askstring("Applied Load (g)", "Enter applied load (g):")
            if not appliedLoad:
                messagebox.showwarning("Invalid Input", "Please enter a valid value for the applied load.")
                continue
            try:
                appliedLoad = float(appliedLoad)
                if appliedLoad <= 0:
                    messagebox.showwarning("Invalid Input", "Applied load must be greater than 0.")
                    continue
                return appliedLoad
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid value for the applied load.")
                continue

    def get_area():
        while True:
            area = simpledialog.askstring("Cross Sectional Area (m^2)", "Enter cross sectional area (m^2):")
            if not area:
                messagebox.showwarning("Invalid Input", "Please enter a valid value for the cross sectional area.")
                continue
            try:
                area = float(area)
                if area <= 0:
                    messagebox.showwarning("Invalid Input", "Cross sectional area must be greater than 0.")
                    continue
                return area
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid value for the cross sectional area.")
                continue

    while True:
        appliedLoad = get_load()
        area = get_area()

        intendedLoad = (appliedLoad + 274) / 1000 * 3 # pre-load: 274 g, convert to kg, 3:1 load
        intendedForce = intendedLoad * 9.80665 # F = mg
        intendedStress = (intendedForce / area) / (10 ** 6) # P = F/A, convert to MPa

        response = messagebox.askquestion("Confirmation of Intended Stress", f"Verify intended stress of {intendedStress:.2f} MPa.")

        if response == "no":
            continue  # Continue loop to ask input again
        else:
            break

    root.deiconify() # unhide window
    strainApp(root).grid(sticky="nsew")
    root.mainloop()

if __name__ == "__main__":
    main()