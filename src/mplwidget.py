"""
File contains the classes for embedding matplotlib into PyQt5.
One class sets up the canvas and figure and the other creates the matplotlib widget.
https://stackoverflow.com/questions/43947318/plotting-matplotlib-figure-inside-qwidget-using-qt-designer-form-and-pyqt5
"""
# Imports

from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')


class MplCanvas(Canvas):
    """
    Matplotlib canvas class to create figure.
    """
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


class MplWidget(QtWidgets.QWidget):
    """
    MplWidget is a Matplotlib widget where and matplotlib figure will be plotted.
    """
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self.canvas.fig.set_tight_layout(True)
        self.canvas.ax.axis('off')