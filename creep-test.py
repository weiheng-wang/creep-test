import tkinter as tk
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass
from typing import List
import numpy as np
import time, random

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams

from concurrent.futures import ThreadPoolExecutor
import os
import csv


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
        self.freq: tk.StringVar() = "100"
        self.notes = tk.StringVar()
        self.readings: List[Reading] = []


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
        notes_lbl = tk.Label(self, text="Notes:", anchor="e")
        notes_lbl.grid(row=3, column=0, sticky="ew")  
        self.notes_ent = tk.Entry(self, textvariable=self.handler.test.notes)
        self.notes_ent.grid(row=3, column=1, sticky="ew")


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
            sharex=True
        )

        self.strainplt.set_xlabel("Time (s)")
        self.strainplt.set_ylabel("Strain")
        self.strainplt.grid(color="darkgrey", alpha=0.65, linestyle='')
        self.strainplt.set_facecolor("w")

        self.strainrateplt.set_xlabel("Time (s)")
        self.strainrateplt.set_ylabel("Strain Rate")
        self.strainrateplt.grid(color="darkgrey", alpha=0.65, linestyle='')
        self.strainrateplt.set_facecolor("w")

        self.temperatureplt.set_xlabel("Time (s)")
        self.temperatureplt.set_ylabel("Temperature")
        self.temperatureplt.grid(color="darkgrey", alpha=0.65, linestyle='')
        self.temperatureplt.set_facecolor("w")

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().grid(sticky="nsew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.ani = FuncAnimation(self.fig, self.animate, interval=500, cache_frame_data=False)
        self.handler.ani = self.ani

        self.fig.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

    def on_zoom(self, event):
        """Zoom in or out on the plot when the mouse wheel is scrolled."""
        if self.handler.paused:  # Only allow zooming when paused
            zoom_factor = 1.01  # The factor by which the zoom happens
            if event.button == 'down':
                scale = 1 / zoom_factor
            elif event.button == 'up':
                scale = zoom_factor
            else:
                return

        # Get current axes limits
        for ax in [self.strainplt, self.strainrateplt, self.temperatureplt]:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()

            # Get mouse position
            mouse_x, mouse_y = ax.transData.inverted().transform([event.x, event.y])
            x_center = mouse_x
            y_center = mouse_y

            # Calculate the new limits based on zooming around the mouse position
            new_xlim = [x_center - ((x_center - xlim[0]) * scale), x_center + ((xlim[1] - x_center) * scale)]
            new_ylim = [y_center - ((y_center - ylim[0]) * scale), y_center + ((ylim[1] - y_center) * scale)]

            # Apply the new zoomed limits to the axis
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

            self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        """Capture mouse press for panning."""
        print("Mouse press detected.")  # Debug line
        if self.handler.paused and event.button == 1:
            self.is_dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        """Capture mouse drag for panning."""
        if self.handler.paused and self.is_dragging:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y

            # Get canvas width and height
            canvas_width, canvas_height = self.fig.canvas.get_width_height()

            # Get current limits
            for ax in [self.strainplt, self.strainrateplt, self.temperatureplt]:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()

                # Adjust limits based on mouse drag movement
                new_xlim = [x - dx * (xlim[1] - xlim[0]) / canvas_width for x in xlim]
                new_ylim = [y - dy * (ylim[1] - ylim[0]) / canvas_height for y in ylim]

                # Ensure the x-axis (time) doesn't move into the negative range
                if new_xlim[0] < 0:
                    new_xlim[0] = 0
                    new_xlim[1] = xlim[1]  # Preserve the upper bound of xlim

                # Ensure the y-axis doesn't move into the negative range
                if new_ylim[0] < 0:
                    new_ylim[0] = 0
                    new_ylim[1] = ylim[1]  # Preserve the upper bound of ylim

                # Apply the new limits to axes
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)

            # Update starting mouse position for next drag
            self.drag_start_x = event.x
            self.drag_start_y = event.y

            # Redraw the plot
            self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Stop dragging when mouse is released."""
        self.is_dragging = False

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

            self.strainplt.clear()
            self.strainrateplt.clear()

            # Strain Plot
            self.strainplt.set_xlabel("Time (s)")
            self.strainplt.set_ylabel("Strain")
            self.strainplt.grid(color="darkgrey", linestyle='dashed')
            self.strainplt.set_facecolor("w")
            self.strainplt.margins(0, tight=True) 
            self.strainplt.plot(elapsedMin, strain)

            # Strain Rate Plot
            self.strainrateplt.set_xlabel("Time (s)")
            self.strainrateplt.set_ylabel("Strain Rate")
            self.strainrateplt.grid(color="darkgrey", linestyle='dashed')
            self.strainrateplt.set_facecolor("w")
            self.strainrateplt.margins(0, tight=True) 
            self.strainrateplt.plot(elapsedMin, strainRate)

            # Temperature Plot
            self.temperatureplt.clear()
            self.temperatureplt.set_xlabel("Time (s)")
            self.temperatureplt.set_ylabel("Temperature")
            self.temperatureplt.grid(color="darkgrey", linestyle='dashed')
            self.temperatureplt.set_facecolor("w")
            self.temperatureplt.margins(0, tight=True) 
            self.temperatureplt.plot(elapsedMin, temperature)


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

    def display(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "".join((msg, "\n")))
        self.log_text.configure(state="disabled")
        self.log_text.yview("end")


class TestHandler:
    def __init__(self, name: str = "Test 1", test_controls: TestControls = None, strainplot: StrainPlot = None, test_info_entry: TestInfoEntry = None):
        self.name = name
        self.root: tk.Tk = strainApp.ROOT
        self.test = Test()

        # Store instances
        self.test_controls = test_controls
        self.strainplot = strainplot
        self.test_info_entry = test_info_entry
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
        self.start_time = time.time() 
        self.is_running = True
        self.paused = False

        # Read the text entries (except notes)
        self.test.name = self.test_info_entry.name_ent.get()
        self.test.material = self.test_info_entry.matr_ent.get()
        self.test.freq = self.test_info_entry.freq_ent.get()

        # Disable the text boxes
        self.test_info_entry.name_ent.config(state="disabled")
        self.test_info_entry.matr_ent.config(state="disabled")
        self.test_info_entry.freq_ent.config(state="disabled")

        # Disable the start button and enable the stop and pause buttons
        if self.test_controls:
            self.test_controls.start_btn.configure(state="disabled")
            self.test_controls.stop_btn.configure(state="normal")
            self.test_controls.pause_btn.configure(state="normal")
            self.ani.event_source.start()

        # Initialize the first reading
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
        self.readings.clear()
        self.readings.append(first_reading)

        self.test_controls.display("Test started.")
        self.pool.submit(self.cont_test)

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

        self.test_controls.display("Test stopped.")

        self.test_info_entry.notes_ent.config(state="disabled")
        self.test.notes = self.test_info_entry.notes_ent.get()
        filename = f"{self.test.name}_data.csv"
        with open(filename, mode='a', newline='') as file:
            file.write(f"Notes: {self.test.notes}\n")
            file.write("="*50 + "\n")

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
            if current_time - self.last_read_time >= 0.25:
                self.pool.submit(self.get_strain) 
                self.pool.submit(self.get_time) 
                self.pool.submit(self.get_strain_rate) 
                self.pool.submit(self.get_temperature)
                reading = Reading(
                    elapsedMin=self.elapsed_min, strain=self.strain, strainRate=self.strain_rate, temperature=self.temperature
                )
                self.readings.append(reading)
                self.last_read_time = current_time

            if current_time - self.last_save_time >= 5:  # Can change to user entered frequency
                # Print saved data to log
                self.test_controls.display(f"Elapsed Time: {self.readings[-1].elapsedMin:.2f}\nStrain: {self.readings[-1].strain:.2f}\nStrain Rate: {self.readings[-1].strainRate:.2f}\nTemperature: {self.readings[-1].temperature:.2f}")
                # Save data to csv file
                self.save_to_csv()
                # Update last save time
                self.last_save_time = current_time

    def save_to_csv(self):
        filename = f"{self.test.name}_data.csv"
        fieldnames = ['Epoch Time', 'Elapsed Time (min)', 'Strain', 'Strain Rate', 'Temperature']
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader() # Write header only if file doesn't exist
            for reading in self.readings:
                epoch_time = time.time()
                writer.writerow({
                    'Epoch Time': epoch_time,
                    'Elapsed Time (min)': reading.elapsedMin,
                    'Strain': reading.strain,
                    'Strain Rate': reading.strainRate,
                    'Temperature': reading.temperature
                })
        # Clear the readings after saving
        self.readings.clear()

    def get_strain(self):
        self.strain = np.sqrt(self.elapsed_min) 

    def get_time(self):
        self.elapsed_min = (time.time() - self.start_time) 
    
    def get_strain_rate(self): 
        if len(self.readings) > 0:
            strain_diff = self.strain - self.readings[-1].strain 
            time_diff = self.elapsed_min - self.readings[-1].elapsedMin

            if time_diff > 0: # Avoid division by zero
                self.strain_rate = strain_diff / time_diff
            else:
                self.strain_rate = 0

        else:
            self.strain_rate = 0

    def get_temperature(self):
        self.temperature= (self.elapsed_min) ** 2

    def rebuild_views(self): 
        for widget in self.views: 
            if widget.winfo_exists():
                self.root.after_idle(widget.build, {"reload": True})
            else:
                self.views.remove(widget)


class MainFrame(tk.Frame):
    """Main Frame for the application."""
    def __init__(self, parent: tk.Frame):
        super().__init__(parent)
        self.parent: tk.Frame = parent
        self.handler = TestHandler(test_controls=None, test_info_entry=None)
        self.strainplot: StrainPlot = None
        self.build()

    def build(self):
        """Builds the UI"""
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=2)

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
    root.geometry("1000x600") # window size
    strainApp(root).grid(sticky="nsew")
    root.mainloop()

if __name__ == "__main__":
    main()