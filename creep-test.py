import tkinter as tk
from tkinter import *
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

@dataclass
class Reading:
    elapsedMin: float
    strain: float 
    strainRate: float
    temperature: float


class Test:
    """Object for holding all the data associated with a Test."""
    def __init__(self):  
        self.name = tk.StringVar()
        self.material = tk.StringVar()
        self.freq = tk.StringVar()
        self.freq_log = []
        self.notes = tk.StringVar()
        self.gauge_length = tk.StringVar()
        self.readings: List[Reading] = []
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
        freq_lbl = tk.Label(self, text="Frequency:", anchor="e")
        freq_lbl.grid(row=2, column=0, sticky="ew")  
        self.freq_ent = tk.Entry(self, textvariable=self.handler.test.freq)
        self.freq_ent.grid(row=2, column=1, sticky="ew")    

        # row 3 ---------------------------------------------  
        gauge_length_lbl = tk.Label(self, text="Gauge Length:", anchor="e")
        gauge_length_lbl.grid(row=3, column=0, sticky="ew")  
        self.gauge_length_ent = tk.Entry(self, textvariable=self.handler.test.gauge_length)
        self.gauge_length_ent.grid(row=3, column=1, sticky="ew")

        # row 4 ---------------------------------------------  
        notes_lbl = tk.Label(self, text="Notes:", anchor="e")
        notes_lbl.grid(row=4, column=0, sticky="ew")  
        self.notes_ent = tk.Entry(self, textvariable=self.handler.test.notes)
        self.notes_ent.grid(row=4, column=1, sticky="ew")


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
        self.strainplt.set_ylabel("Strain")
        self.strainplt.grid(color="darkgrey", alpha=0.65, linestyle='dashed')
        self.strainplt.set_facecolor("w")
        self.strainplt.margins(0, tight=True)

        self.strainrateplt.set_xlabel("Time (s)")
        self.strainrateplt.set_ylabel("Strain Rate")
        self.strainrateplt.grid(color="darkgrey", alpha=0.65, linestyle='dashed')
        self.strainrateplt.set_facecolor("w")
        self.strainrateplt.margins(0, tight=True)

        self.temperatureplt.set_xlabel("Time (s)")
        self.temperatureplt.set_ylabel("Temperature")
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
            elapsedMin = []
            strain = []
            strainRate = []
            temperature = []
            readings = tuple(self.handler.readings)

            for reading in readings:
                elapsedMin.append(reading.elapsedMin)
                strain.append(reading.strain)
                strainRate.append(reading.strainRate)
                temperature.append(reading.temperature)

            self.line1.set_data(elapsedMin, strain)
            self.strainplt.set_xlim(0, elapsedMin[-1]) # x-axes are shared
            self.strainplt.set_ylim(0, max(strain))

            self.line2.set_data(elapsedMin, strainRate)
            if (min(strainRate) == max(strainRate)):
                self.strainrateplt.set_ylim(min(strainRate), max(strainRate) + 0.1)
            else:
                self.strainrateplt.set_ylim(min(strainRate), max(strainRate))

            self.line3.set_data(elapsedMin, temperature)
            if (min(temperature) == max(temperature)):
                self.temperatureplt.set_ylim(min(temperature), max(temperature) + 0.1)
            else:
                self.temperatureplt.set_ylim(min(temperature), max(temperature))


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
        self.start_btn.configure(text="Start", state="normal", command=self.handler.start_test)
        self.start_btn.grid(row=0, column=0, sticky="ew")

        # row 0 col 1 --------------------------------------
        self.stop_btn = tk.Button(self) 
        self.stop_btn.configure(text="Stop", state="disabled", command=self.handler.stop_test)
        self.stop_btn.grid(row=0, column=1, sticky="ew")

        # row 0 col 2 --------------------------------------
        self.pause_btn = tk.Button(self)
        self.pause_btn.configure(text="Pause", state="disabled", command=self.handler.toggle_pause)
        self.pause_btn.grid(row=0, column=2, sticky="ew")

        # row 1 col 0 --------------------------------------
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

        self.readings: List[Reading] = []
        self.elapsed_min: float = float()
        self.strain: float = float()
        self.strain_rate: float = float()

        # Initialize flags
        self.is_running = False
        self.request_stop = False
        self.paused = False
        self.views: List[tk.Widget] = []
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.ani = None
        self.last_save_time = time.time()
        self.last_read_time = time.time()

    def start_test(self):
        # Read the text entries (except notes)
        self.test.name = self.test_info_entry.name_ent.get()
        self.test.material = self.test_info_entry.matr_ent.get()
        self.test.freq = self.test_info_entry.freq_ent.get()
        self.test.gauge_length = self.test_info_entry.gauge_length_ent.get()

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
            self.start_time = time.time() 
            self.is_running = True
            self.paused = False

            # Disable the text boxes
            self.test_info_entry.name_ent.config(state="disabled")
            self.test_info_entry.matr_ent.config(state="disabled")
            self.test_info_entry.gauge_length_ent.config(state="disabled")

            # Disable the start button and enable the stop and pause buttons
            if self.test_controls:
                self.test_controls.start_btn.configure(state="disabled")
                self.test_controls.stop_btn.configure(state="normal")
                self.test_controls.pause_btn.configure(state="normal")

            # Initialize the first reading FIXME
            self.elapsed_min = 0.0
            self.strain = 0.0
            self.strain_rate = 0.0
            self.temperature = 0.0
            first_reading = Reading(
                elapsedMin=self.elapsed_min,
                strain=self.strain,
                strainRate=self.strain_rate,
                temperature=self.strain
            )
            self.readings.append(first_reading)

            self.test.freq_log.append({"Frequency": self.test.freq, "Timestamp": self.elapsed_min})

            self.test.data_file_name = f"{self.test.name}_data.csv"
            self.test.info_file_name = f"{self.test.name}_info.csv"

            # I/O
            self.rm = pyvisa.ResourceManager()
            resources = self.rm.list_resources()
            print(resources)

            self.daq = self.rm.open_resource('GPIB0::9::INSTR')

            available_channels = self.daq.query('INST:LIST?')
            print(f"Available Channels: {available_channels}")
            # END

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
        self.test_controls.start_btn.configure(state="normal")
        self.test_controls.pause_btn.configure(state="disabled")
        self.test_controls.stop_btn.configure(state="disabled")
        self.ani.event_source.stop()

        print("Stopped the test.")
        self.test_controls.display("Test stopped.")
        self.test_info_entry.notes_ent.config(state="disabled")

        self.daq.close()

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

    def take_readings(self):
        while self.is_running and not self.request_stop:
            current_time = time.time()
            if current_time - self.last_read_time >= 1/(float(self.test.freq)): # Data acquisition period
                self.pool.submit(self.get_strain)
                self.pool.submit(self.get_time)
                self.pool.submit(self.get_strain_rate)
                self.pool.submit(self.get_temperature)
                reading = Reading(
                    elapsedMin=self.elapsed_min, strain=self.strain, strainRate=self.strain_rate, temperature=self.temperature
                )
                self.readings.append(reading)
                self.last_read_time = current_time

            if current_time - self.last_save_time >= 5:  # Update files and frequency (if there is a change) period
                # Print saved data to log
                self.test_controls.display(f"Elapsed Time: {self.readings[-1].elapsedMin:.2f}\nStrain: {self.readings[-1].strain:.2f}\nStrain Rate: {self.readings[-1].strainRate:.2f}\nTemperature: {self.readings[-1].temperature:.2f}")
                self.test_controls.display("="*44)

                # Save data to csv file
                self.save_to_csv()
                # Update last save time
                self.last_save_time = current_time

                temp = self.test_info_entry.freq_ent.get()
                # Try to convert frequency into float
                try:
                    freq = float(temp)
                except ValueError:
                    freq = 0
                if freq != float(self.test.freq):
                    if (freq > 0):
                        self.test.freq = freq
                        self.test.freq_log.append({"Frequency": self.test.freq, "Timestamp": self.elapsed_min})
                        self.test_controls.display(f"Frequency changed to {self.test.freq}.")
                    else:
                        self.test_controls.display("Please enter valid input.")

    def save_to_csv(self):
        new_readings = self.readings[self.test.last_written_index:]
        if new_readings:
            fieldnames = ['Epoch Time', 'Elapsed Time (min)', 'Strain', 'Strain Rate', 'Temperature']
            with open(self.test.data_file_name, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:  # If the file is empty, write header
                    writer.writeheader()
                for reading in new_readings:
                    epoch_time = time.time()
                    writer.writerow({
                        'Epoch Time': epoch_time,
                        'Elapsed Time (min)': reading.elapsedMin,
                        'Strain': reading.strain,
                        'Strain Rate': reading.strainRate,
                        'Temperature': reading.temperature
                    })
            self.test.last_written_index = len(self.readings)

        self.test.notes = self.test_info_entry.notes_ent.get()
        with open(self.test.info_file_name, mode='w', newline='') as file:
            file.write("="*50 + "\n")
            file.write(f"Name: {self.test.name}\n")
            file.write(f"Material: {self.test.material}\n")
            file.write(f"Gauge Length: {self.test.gauge_length}\n")
            file.write(f"Notes: {self.test.notes}\n")
            for entry in self.test.freq_log:
                file.write(f"Frequency Log: {entry['Frequency']} at {entry['Timestamp']}\n")
            file.write("="*50 + "\n")

    # NIST Type K Thermocouple coefficients for 0°C to 1372°C
    coefficients = [
        0.0000000e+00,
        2.508355e+01,
        7.860106e-02,
        -2.503131e-01,
        8.315270e-02,
        -1.228034e-02,
        9.804036e-04,
        -4.413030e-05,
        1.057734e-06,
        -1.052755e-08
    ]

    def get_strain(self):
        # calibration?
        self.voltage = math.sqrt(self.elapsed_min * 10000) #FIXME
        self.displacement = (0.04897 * self.voltage) + 0.53505
        self.strain = self.displacement / float(self.test.gauge_length) # check if need (0.0637167 / self.test.gauge_length)

    def get_time(self):
        self.elapsed_min = (time.time() - self.start_time) #FIXME
    
    def get_strain_rate(self): 
        if len(self.readings) > 1:
            strain_diff = self.strain - self.readings[-1].strain
            time_diff = self.elapsed_min - self.readings[-1].elapsedMin

            if time_diff > 0: # Avoid division by zero
                self.strain_rate = strain_diff / time_diff
            else:
                self.strain_rate = 0

        else:
            self.strain_rate = 0

    def get_temperature(self):
        self.voltage = (self.elapsed_min % 3) #FIXME
        self.temperature = 0
        for i, coeff in enumerate(self.coefficients):
            self.temperature += coeff * (self.voltage ** i)


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
        self.quit()


def main():
    """The Tkinter entry point of the program; enters mainloop."""
    root = tk.Tk()
    root.title("Creep Test")
    strainApp.ROOT = root
    root.geometry("1000x650") # window size
    strainApp(root).grid(sticky="nsew")
    root.mainloop()

if __name__ == "__main__":
    main()