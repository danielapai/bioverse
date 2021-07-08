from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QInputDialog,\
                            QFileDialog, QListWidgetItem, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt
import numpy as np
import os
import signal
import sys

# Bioverse imports
from bioverse.generator import Generator
from bioverse.survey import ImagingSurvey, TransitSurvey
from bioverse.survey import reset_imaging_survey, reset_transit_survey
from bioverse.constants import OBJECTS_DIR, UI_DIR
from bioverse.util import get_type

# Key values that allow float or percent values
allows_pct = ['precision']

# Key values that allow float or None values
allows_none = ['t_total', 't_ref']

class startDialog(QDialog):
    """ Initial dialog window. Prompts the user to select a class to edit. """
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi(UI_DIR+'/startDialog.ui', self)
        self.window = None
        self.show()

    def open_generatorWindow(self):
        """ Opens a window to edit Generator objects. """
        self.window = generatorWindow(self)
        self.hide()

    def open_imagingWindow(self):
        """ Opens a window to edit ImagingSurvey objects. """
        self.window = surveyWindow(self, imaging=True)
        self.hide()
    
    def open_transitWindow(self):
        """ Opens a window to edit TransitSurvey objects. """
        self.window = surveyWindow(self, imaging=False)
        self.hide()

class generatorWindow(QMainWindow):
    """ Window for editing Generator objects. """
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)
        uic.loadUi(UI_DIR+'/generatorWindow.ui', self)

        # Class for managing the Generator and load/save functions
        self.mgr = FileManager(Generator, self)

        self.args_loading = False
        self.show()

    def load(self):
        self.mgr.load()
        if self.mgr.obj is not None:
            self.load_steps()
            self.generatorLabel.setText('Current Generator: {:s}'.format(self.mgr.obj.label))
            self.stepsListWidget.setCurrentRow(0)
        self.update_buttons()
    
    def saveas(self):
        self.mgr.saveas()
        if self.mgr.obj is not None:
            self.generatorLabel.setText('Current Generator: {:s}'.format(self.mgr.obj.label))

    def update_buttons(self):
        """ Enables or disables QPushButtons based on certain conditions. """
        # Enable the Add/Remove step buttons if a Generator is loaded
        enable = self.mgr.obj is not None
        self.addButton.setEnabled(enable)
        self.removeButton.setEnabled(enable)
        self.upButton.setEnabled(enable)
        self.downButton.setEnabled(enable)

    def add_step(self):
        """ Prompts the user for a new step to insert at the end of the list. """
        function_name, ok = QInputDialog.getText(self, 'Add step', 'Enter the function name (in custom.py or functions.py):')
        if ok:
            nsteps = len(self.mgr.obj.steps)
            try:
                self.mgr.obj.insert_step(function_name)
            except ValueError as err:
                print("Error adding step: {:s}".format(str(err)))
                return

            # Flag the Generator as changed
            self.mgr.changed = True

            # Refresh lists/tables
            self.load_steps()
            self.stepsListWidget.setCurrentRow(nsteps)
    
    def remove_step(self):
        """ Deletes the currently-selected step. """
        idx = self.stepsListWidget.currentRow()
        if len(self.mgr.obj.steps) > 0:
            del self.mgr.obj.steps[idx]

            # Flag the Generator as changed
            self.mgr.changed = True

            # Refresh lists/tables
            self.load_steps()
            nsteps = len(self.mgr.obj.steps)
            if nsteps > 0:
                self.stepsListWidget.setCurrentRow(min(idx, nsteps-1))

    def select_step(self, idx):
        """ Selects a new step in the list and displays its description. """
        steps = self.mgr.obj.steps
        if len(steps) > 0:
            step = self.mgr.obj.steps[idx]
            self.descriptionTextBrowser.setText(step.description)
            self.stepLabel.setText('Current Step: {:s} (found in {:s})'.format(step.function_name, step.filename))
            self.load_args()
        else:
            self.descriptionTextBrowser.setText('')
            self.stepLabel.setText('Current Step:')

    def load_steps(self):
        """ Loads the list of steps into stepsListWidget. """
        # Clear stepsListWidget
        self.stepsListWidget.clear()

        steps = self.mgr.obj.steps
        for i, step in enumerate(steps):
            item = QListWidgetItem('Step {:d}: {:s}'.format(i, step.function_name))
            self.stepsListWidget.addItem(item)
    
    def move_up(self):
        """ Moves a step up in the sequence. """
        self.move_step(-1)

    def move_down(self):
        """ Moves a step down in the sequence. """
        self.move_step(1)

    def move_step(self, move):
        """ Moves the currently-selected step up or down by `move` places. """
        # Check that the move is valid
        steps = self.mgr.obj.steps
        if len(steps) == 0:
            return
        idx = self.stepsListWidget.currentRow()
        idx_max = len(steps) - 1
        if (idx+move < 0) or (idx+move > idx_max):
            return
        
        # Insert the step at its new location, then delete it at the old location
        steps.insert(idx+move+(move>0), steps[idx])
        del steps[idx if move>0 else idx+1]
        
        self.load_steps()
        self.stepsListWidget.setCurrentRow(idx+move)
        self.mgr.changed = True

    def load_args(self):
        """ Loads the arguments and saved values into argsTableWidget. """
        self.args_loading = True

        idx = self.stepsListWidget.currentRow()
        args = self.mgr.obj.steps[idx].args

        load_table_from_dict(args, self.argsTableWidget)

        self.args_loading = False
    
    def set_arg(self, idx):
        """ Sets new keyword argument values when modified. """
        # Skip if the argsTable is being loaded
        if self.args_loading:
            return

        idx_step = self.stepsListWidget.currentRow()
        key, val = get_changed_table_val(self.mgr.obj.steps[idx_step].args, self.argsTableWidget, idx)
        
        if key is not None:
            # Set key = val for all steps that accept this keyword
            self.mgr.obj.set_arg(key, val)
            
            # Flag the Generator as changed
            self.mgr.changed = True

        # Refresh the arguments table
        self.load_args()

    def change_editors(self):
        """ Close this editor and re-open the editor selection GUI. """
        if self.mgr.exit_check():
            self.mgr.changed = False
            self.close()
            self.parent().show()

    def closeEvent(self, event):
        """ Checks that the current Generator has been saved, else warns the user before closing. """
        if self.mgr.exit_check():
            event.accept()
        else:
            event.ignore()

class surveyWindow(QMainWindow):
    """ Window for editing Survey objects. """
    def __init__(self, parent, imaging=True):
        QMainWindow.__init__(self, parent)
        uic.loadUi(UI_DIR+'/surveyWindow.ui', self)

        self.clas = ImagingSurvey if imaging else TransitSurvey
        self.loading = False
        self.surveyLabel.setText('Current {:s}:'.format(self.clas.__name__))

        # Class for managing the Survey and load/save functions
        self.mgr = FileManager(self.clas, self)

        self.show()

    def load(self):
        self.mgr.load()
        if self.mgr.obj is not None:
            self.load_params()
            self.load_measurements()
            self.surveyLabel.setText('Current {:s}: {:s}'.format(self.clas.__name__, self.mgr.obj.label))
        self.update_buttons()
    
    def saveas(self):
        self.mgr.saveas()
        if self.mgr.obj is not None:
            self.surveyLabel.setText('Current {:s}: {:s}'.format(self.clas.__name__, self.mgr.obj.label))
    
    def update_buttons(self):
        """ Enables or disables QPushButtons based on certain conditions. """
        # Enable the Add/Remove/Up/Down measurements buttons if a Survey is loaded
        enable = self.mgr.obj is not None
        self.addButton.setEnabled(enable)
        self.removeButton.setEnabled(enable)
        self.upButton.setEnabled(enable)
        self.downButton.setEnabled(enable)
        
        # Enable the Add/Remove condition buttons if a Measurement is selected
        #enable = len(list(self.mgr.obj.measurements)) > 0
        enable = self.measurementTableWidget.rowCount() > 0
        self.addConditionButton.setEnabled(enable)
        self.removeConditionButton.setEnabled(enable)

    def load_params(self):
        self.loading = True
        params = self.mgr.obj.__dict__
        load_table_from_dict(params, self.paramsTableWidget, skip=['mode', 'measurements', 'filename', 'label'])
        self.loading = False

    def set_param(self, idx):
        # Skip if the paramsTable is being loaded
        if self.loading:
            return

        key, val = get_changed_table_val(self.mgr.obj.__dict__, self.paramsTableWidget, idx)
        
        if key is not None:
            # Set the new parameter value
            self.mgr.obj.__dict__[key] = val
            
            # Flag the Survey as changed
            self.mgr.changed = True

        # Refresh the params table
        self.load_params()
    
    def load_measurements(self):
        """ Loads the list of measurements in measurementsListWidget. """
        # Clear measurementsListWidget
        self.measurementsListWidget.clear()

        measurements = self.mgr.obj.measurements
        for i, measurement in enumerate(measurements):
            item = QListWidgetItem('Measurement {:d}: {:s}'.format(i, measurement))
            self.measurementsListWidget.addItem(item)
    
    def select_measurement(self, idx):
        """ Loads a measurement's properties into measurementTableWidget and priorityTableWidget. """
        self.loading = True

        if len(list(self.mgr.obj.measurements)) > 0:
            key = list(self.mgr.obj.measurements)[idx]
            params = self.mgr.obj.measurements[key].__dict__
            load_table_from_dict(params, self.measurementTableWidget, skip=['survey', 'conditions', 'priority', 'key'])
            self.load_conditions()
            self.measurementLabel.setText('Current Measurement: {:s}'.format(key))

        else:
            self.measurementTableWidget.setRowCount(0)
            self.priorityTableWidget.setRowCount(0)
            self.measurementLabel.setText('Current Measurement:')

        self.update_buttons()
        self.loading = False
    
    def set_measurement_param(self, idx):
        # Skip if the measurementTableWidget is being loaded
        if self.loading:
            return

        midx = self.measurementsListWidget.currentRow()
        measurement = self.mgr.obj.measurements[list(self.mgr.obj.measurements)[midx]]
        key, val = get_changed_table_val(measurement.__dict__, self.measurementTableWidget, idx)
        
        if key is not None:
            # Set the new parameter value
            measurement.__dict__[key] = val
            
            # Flag the Survey as changed
            self.mgr.changed = True

        # Refresh the measurement parms table
        self.select_measurement(midx)
    
    def add_measurement(self):
        """ Adds a measurement after the currently-selected measurement. """
        key, ok = QInputDialog.getText(self, 'Add measurement', 'Enter the parameter to measure:')
        if ok:
            if key in self.mgr.obj.measurements:
                print("parameter {:s} is already measured by the survey".format(key))
                return
            idx = self.measurementsListWidget.currentRow()+1
            self.mgr.obj.add_measurement(key, idx=idx)
            self.load_measurements()
            self.measurementsListWidget.setCurrentRow(idx)

    def remove_measurement(self):
        """ Deletes the currently-selected measurement. """
        idx = self.measurementsListWidget.currentRow()
        if len(self.mgr.obj.measurements) > 0:
            key = list(self.mgr.obj.measurements)[idx]
            del self.mgr.obj.measurements[key]

            # Flag the Survey as changed
            self.mgr.changed = True

            # Refresh lists/tables
            self.load_measurements()
            nmeas = len(self.mgr.obj.measurements)
            if nmeas > 0:
                self.measurementsListWidget.setCurrentRow(min(idx, nmeas-1))
    
    def move_up(self):
        """ Moves the currently-selected measurement up by one. """
        self.move_measurement(-1)

    def move_down(self):
        """ Moves the currently-selected measurement down by one. """
        self.move_measurement(1)

    def move_measurement(self, move):
        """ Moves the currently-selected measurement up or down by `move` places. """
        if len(self.mgr.obj.measurements) == 0:
            return
        idx = self.measurementsListWidget.currentRow()
        key = list(self.mgr.obj.measurements)[idx]
        idx_max = self.measurementsListWidget.count() - 1
        if (idx+move < 0) or (idx+move > idx_max):
            return
        self.mgr.obj.move_measurement(key, idx+move)

        self.load_measurements()
        self.measurementsListWidget.setCurrentRow(idx+move)
        self.mgr.changed = True

    def get_current_measurement(self):
        """ Returns the currently-selected measurement. """
        idx = self.measurementsListWidget.currentRow()
        key = list(self.mgr.obj.measurements)[idx]
        return self.mgr.obj.measurements[key]

    def format_condition(self, key, val1, val2):
        """ Formats (key, val1, val2) into a conditional statement e.g. val1 < key < val2. """
        if val1 is not None and val2 is not None:
            condition = '{:.2f} < {:s} < {:.2f}'.format(val1, key, val2)
        elif val2 is None:
            condition = '{:s} == {:s}'.format(key, str(val1))
        return condition

    def load_conditions(self):
        """ Loads the prioritization scheme for the currently-selected measurement. """
        self.loading = True
        m = self.get_current_measurement()
        d = {}
        for key in m.priority:
            arr1 = m.priority[key]
            for arr2 in arr1:
                val1, val2, weight = arr2
                condition = self.format_condition(key, val1, val2)
                d[condition] = weight
        
        load_table_from_dict(d, self.priorityTableWidget)
        self.loading = False

    def add_condition(self):
        """ Add a priority condition to the list. """
        m = self.get_current_measurement()
        result = PriorityDialog()
        if result.exec_():
            # Update Survey.priority based on the input
            key, val1, val2, weight = result.key, result.val1, result.val2, result.weight
            
            # If the condition is x == val1, determine whether val1 is str or int
            if result.type == 'value':
                val1 = get_type(val1)(val1)

            # Add the condition to Survey.priority
            arr = np.array([[val1, val2, weight]])
            if key not in m.priority:
                m.priority[key] = np.zeros(shape=(0, 3))
            m.priority[key] = np.append(m.priority[key], arr, axis=0)
            
            self.mgr.changed = True
        
        self.load_conditions()
    
    def remove_condition(self):
        """ Removes a priority condition from the list. """
        # Skip if the table is empty
        if self.priorityTableWidget.rowCount() == 0:
            return

        # Get the currently-selected row values from priorityTableWidget
        idx = self.priorityTableWidget.currentRow()
        col1 = self.priorityTableWidget.item(idx, 0).text()
        col2 = float(self.priorityTableWidget.item(idx, 1).text())

        # Determine which entry in Measurement.priority matches and remove it
        m = self.get_current_measurement()
        for key in m.priority:
            arr1 = m.priority[key]
            for i in range(arr1.shape[0]):
                val1, val2, weight = arr1[i]
                condition = self.format_condition(key, val1, val2)
                if condition == col1 and weight == col2:
                    m.priority[key] = np.delete(m.priority[key], i, axis=0)
                    if len(m.priority[key]) == 0:
                        del m.priority[key]        
                    self.load_conditions()
                    return

        print("Error: could not find condition to delete.")

    def restore_default(self):
        """ Restores the default TransitSurvey or ImagingSurvey as default.pkl. """
        msg = "Restore the default {:s} and save under the label 'default'?".format(self.clas.__name__)
        msg += " Note: this will overwrite any changes made to the 'default' {:s}!".format(self.clas.__name__)
        reply = QMessageBox.question(self, 'Message', msg, QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            if self.clas == ImagingSurvey:
                reset_imaging_survey()
            elif self.clas == TransitSurvey:
                reset_transit_survey()

    def change_editors(self):
        """ Close this editor and re-open the editor selection GUI. """
        if self.mgr.exit_check():
            self.mgr.changed = False
            self.close()
            self.parent().show()

    def closeEvent(self, event):
        """ Checks that the current Survey has been saved, else warns the user before closing. """
        if self.mgr.exit_check():
            event.accept()
        else:
            event.ignore()

class PriorityDialog(QDialog):
    """ Window for setting measurement priority. """
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi(UI_DIR+'/priorityDialog.ui', self)
        self.type, self.key, self.weight = None, None, None
        self.val1, self.val2 = None, None
        self.show()
    
    def accept(self):
        """ Checks and sets the return value(s), then hides the dialog. """
        
        self.key = self.keyEdit.text()
        if self.key == '':
            print("Must set parameter name!")
            QDialog.reject(self)
            return

        try:
            self.weight = float(self.weightEdit.text())
        except ValueError:
            print("Must set weight!")
            QDialog.reject(self)
            return

        if self.rangeRadioButton.isChecked():
            # Return min and max as float values
            self.type = 'range'
            self.val1, self.val2 = self.minEdit.text(), self.maxEdit.text()

            # If either is blank, use + or - inf
            self.val1 = -np.inf if self.val1 == '' else float(self.val1)
            self.val2 = np.inf if self.val2 == '' else float(self.val2)
            if self.val1 >= self.val2:
                print("Max must be greater than min!")
                QDialog.reject(self)
                return
            if np.isinf(self.val1) and np.isinf(self.val2):
                print("Must set either min or max!")
                QDialog.reject(self)
                return
        
        elif self.valueRadioButton.isChecked():
            # Return the value as-is
            self.type = 'value'
            self.val1 = self.valueEdit.text()
            if self.val1 == '':
                print("Must set comparison value!")
                QDialog.reject(self)
                return

        elif self.boolRadioButton.isChecked():
            # Return True or False as a bool value
            self.type = 'bool'
            self.val1 = self.boolComboBox.currentText().upper() == 'TRUE'

        QDialog.accept(self)

class FileManager():
    """ Class for loading and saving Generator or Survey files. """
    def __init__(self, cls, parent):
        self.cls = cls
        self.parent = parent
        self.filename = None
        self.dir = OBJECTS_DIR+'/{:s}s/'.format(cls.__name__)

        # Configure the file dialog
        self.fileDialog = QFileDialog(self.parent)
        self.fileDialog.setDefaultSuffix('.pkl')
        self.fileDialog.setNameFilter("(*.pkl)")
        self.fileDialog.setDirectory(self.dir)
        self.fileDialog.directoryEntered.connect(self.restrict_dir)

        self.obj = None
        self.changed = False
    
    def restrict_dir(self, dir):
        """ Prevents the user from leaving the correct directory (might not work perfectly). """
        if dir != self.dir:
            self.fileDialog.setDirectory(self.dir)

    def load(self):
        """ Opens a dialog to load an object. """
        self.fileDialog.setAcceptMode(QFileDialog.AcceptOpen)
        if self.fileDialog.exec_():
            filename = self.fileDialog.selectedFiles()[0]
        else:
            return

        if filename != '' and os.path.exists(filename):
            label = filename.split('/')[-1].split('.')[-2]
            self.obj = self.cls(label=label)
            self.filename = filename
        
        # Flag the object as unchanged
        self.changed = False
    
    def saveas(self):
        """ Opens a dialog to save an object. """
        self.fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        if self.fileDialog.exec_():
            filename = self.fileDialog.selectedFiles()[0]
        else:
            return

        if filename != '' and self.obj is not None:
            label = filename.split('/')[-1].split('.')[-2]
            self.obj.save(label=label)
            self.filename = filename

        # Flag the object as unchanged
        self.changed = False
    
    def exit_check(self):
        """ Prompts the user to save the Survey or Generator if needed. """
        if self.changed:
            msg = "The current object has not been saved - would you like to exit?"
            reply = QMessageBox.question(self.parent, 'Message', msg, QMessageBox.Yes, QMessageBox.No)
            return reply == QMessageBox.Yes
        else:
            return True

def load_table_from_dict(d, tableWidget, skip=[], disable_first=True):
    """ Loads a two-column tableWidget from a dictionary, with column1 = keys and column2 = values. Skips keys in `skip` and disables the first column
    from editing if `disable_first` is True. """
    N_rows = len(d)-len(skip)
    tableWidget.setRowCount(N_rows)
    i = 0
    for key, val in d.items():
        if key in skip:
            continue
        item1, item2 = QTableWidgetItem(key), QTableWidgetItem(str(val))
        if disable_first:
            item1.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        tableWidget.setItem(i, 0, item1)
        tableWidget.setItem(i, 1, item2)
        i += 1
    
    #tableWidget.resizeColumnsToContents()
    tableWidget.resizeRowsToContents()

def get_changed_table_val(d, tableWidget, idx):
    """ Returns a new value for d[key] as updated in the tableWidget. Determines the data type
    of d[key] based on its current value."""

    # Get the parameter keyword and updated value
    key = tableWidget.item(idx, 0).text()
    val = tableWidget.item(idx, 1).text().strip()

    # Attempt to cast `val` to the correct type
    typ = type(d[key])
    try:
        if typ is bool:
            # Exception for type = bool
            if val.upper() == 'TRUE':
                val = True
            elif val.upper() == 'FALSE':
                val = False
            else:
                raise ValueError("must be 'True' or 'False'")

        elif key in allows_pct:
            # Exception for keys that allow absolute or percent values
            if val[-1] == '%':
                float(val[:-1])
                val = str(val)
            else:
                val = float(val)

        elif key in allows_none:
            # Exception for keys that allow float or None values
            if val.upper() == 'NONE':
                val = None
            else:
                val = float(val)

        else:
            val = typ(val)
    except ValueError as err:
        print("Failed to change keyword '{:s}':\n{:s}".format(key, str(err)))
        return None, None
    
    return key, val

def show():
    app = QApplication(sys.argv)

    # Prompts the user for which GUI to open
    window = startDialog()

    # Ensures the GUI closes with Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Workaround to prevent segmentation fault when running from IPython
    os._exit(app.exec())

if __name__ == "__main__":
    sys.exit(show())