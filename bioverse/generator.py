""" This module defines the Generator class and demonstrates how to create new Generators. """

# System imports
from copy import deepcopy
import inspect
import numpy as np
import traceback
from warnings import warn

# Bioverse modules
from .classes import Object, Table, Stopwatch
from .constants import ROOT_DIR
from . import util

class Generator(Object):
    """ This class executes a series of functions to generate a set of nearby systems.
    
    Parameters
    ----------
    label : str, optional
        Name of the Generator. Leave as None to create a new Generator.
    
    Attributes
    ----------
    steps : list of Step
        List of Steps to be performed sequentially by the Generator.
    """

    def __init__(self, label=None):
        self.steps = []
        Object.__init__(self, label)
        # Reloads each step to catch new keyword arguments
        self.update_steps()
    
    def __repr__(self):
        s = "Generator with {:d} steps:".format(len(self.steps))
        for i in range(len(self.steps)):
            s += "\n{:2d}: {:s}".format(i,self.steps[i].__repr__(long_version=False))
        return s

    def copy(self):
        """ Returns a deep copy of the Generator. """
        return deepcopy(self)

    def initialize(self, filename):
        """ Creates a new program from a CSV file containing the function names and descriptions. 
        
        Parameters
        ----------
        filename : str
            Name of the CSV file from which to initialize the Generator.
        """
        self.steps = []
        with open(filename,'r') as f:
            for line in f.readlines():
                if line[0].strip() == '#': continue
                function_name, path = line.split('#')[0].split(',')[:2]
                description = ','.join(line.split('#')[0].split(',')[2:])
                self.steps.append(Step(function_name, path))
    
    def update_steps(self, reload=False):
        """ Loads or re-loads the keyword arguments and description of each step in the Generator.
        
        Parameters
        ----------
        reload : bool
            Whether or not to reload the default values for the arguments.
        """
        for i,step in enumerate(self.steps):
            try:
                step.load_function(reload=reload)
            except KeyError:
                warn("function not found for step '{:s}' - removing it from the sequence!"\
                      .format(step.function_name))
                del self.steps[i]
    
    def insert_step(self, function, idx=None, filename=None):
        """ Inserts a step into the program sequence at the specified index.

        Parameters
        ----------
        function : str or function
            Name of the function to be run by this step *or* the function itself.
        idx : int, optional
            Position in the program at which to insert the step. Default is -1 i.e. at the end.
        filename : str, optional
            Filename containing the function.
        """
        step = Step(function=function, filename=filename)
        idx = len(self.steps) if idx is None else idx
        self.steps.insert(idx,step)
        
    def replace_step(self, new_function, idx, new_filename=None):
        """ Replaces a step into the program sequence at the specified index.

        Parameters
        ----------
        new_function : str or function
            Name of the function to be run by this step *or* the function itself.
        idx : int
            Position in the program at which to replace the step.
        new_filename : str, optional
            Filename containing the function.
        """
        # Remove the old step at the specified index
        self.steps.pop(idx)
        # Insert the new step at the specified index
        step = Step(function=new_function, filename=new_filename)
        idx = len(self.steps) if idx is None else idx
        self.steps.insert(idx,step)

    def get_arg(self, key):
        """ Gets the default value of a keyword argument, and warns if there are multiple values. """
        vals = [step.get_arg(key) for step in self.steps if key in step.args]
        if len(np.unique(vals)) > 1:
            warn("multiple values for argument {:s}".format(key))
        return vals[0]

    def set_arg(self, key, value):
        """ Sets the default value of a keyword argument for every step it applies to. """
        for step in self.steps:
            if key in step.args:
                step.set_arg(key, value)
                #print("set {0} = {1} for step '{2}'".format(key, value, step.function_name))

    def generate(self, d=None, timed=False, idx_start=0, idx_stop=None, **kwargs):
        """ Runs the generator with the current program and returns a simulated set of stars and planets.
        
        Parameters
        ----------
        d : Table, optional
            Pre-existing table of simulated planets to be passed as input. If not specified, an empty table is created.
        timed : bool, optional
            If True, times each step in the program and prints the results.
        idx_start : int, optional
            Specifies at which step in the program the Generator should start.
        idx_stop : int, optional
            Specifies at which step in the program the Generator should stop.
        **kwargs
            Keyword argument(s) to be passed to the individual steps, e.g. d_max=20. Can have unintended
            consequences if the keyword argument appears in more than one step.
        
        Returns
        -------
        d : Table
            Table of simulated planets (plus host star parameters).
        """
        # Update the generator to make sure all steps are up to date
        self.update_steps()

        if d is None:
            d = Table()    
        if timed:
            timer = Stopwatch()
        idx_stop = len(self.steps) if idx_stop is None else idx_stop
        for i in range(idx_start, idx_stop):
            try:
                d = self.steps[i].run(d,**kwargs)
                if timed:
                    timer.mark(self.steps[i].function_name)
            except:
                traceback.print_exc()
                print("\n!!! The program failed at step {:d}: {:s} !!!".format(i, self.steps[i].function_name))
                print("!!! Returning incomplete simulation !!!")
                return d
        if timed:
            print("Timing results:")
            timer.read()
        return d

class Step():
    """ This class runs one function for a Generator and saves its keyword argument values.

    Parameters
    ----------
    function : str or function
        Name of the function to be run *or* the function itself.
    filename : str, optional
        Name of the file containing the function for this step. If None, looks in custom.py and functions.py.

    Attributes
    ----------
    description : str
        Docstring for this step's function.
    """
    def __init__(self, function, filename=None):
        # Determine whether a function or function name was passed
        if type(function) is str:
            self.function_name = function
            self.function = None
        else:
            self.function = function
            self.function_name = function.__name__
            
        self.filename = filename
        self.args = {}

        # Load the function and its description / arguments
        self.load_function()

    def __repr__(self, long_version=True):
        s = "Function '{:s}' with {:s} keyword arguments.".format(self.function_name,
            str(len(self.args)) if len(self.args) else 'no')
        if long_version:
            if len(self.description) > 0:
                s += "\n\nDescription:\n    {:s}".format(self.description)
            if len(self.args) > 0:
                s += '\n\nArgument values:'
                for key,val in self.args.items():
                    s += '\n    {:14s} = {:10s}'.format(key,str(val))
        return s
    
    def run(self, d, **kwargs):
        """ Runs the function described by this step.
        
        Parameters
        ----------
        d : Table
            Table of simulated planets to be passed as the function's first argument.
        **kwargs
            Keyword argument(s) to override. Ignores arguments which don't apply to this step.
        
        Returns
        -------
        d : Table
            Updated version of the simulated planet table.
        """
        if self.function is None:
            func = util.import_function_from_file(self.function_name, ROOT_DIR+'/'+self.filename)
        else:
            func = self.function
        kwargs2 = {key:val for key,val in self.args.items()}
        for key,val in kwargs.items():
            if key in self.args:
                kwargs2[key] = val
        return func(d,**kwargs2)

    def get_arg(self, key):
        """ Returns the value of a keyword argument. """
        return self.args[key]
    
    def set_arg(self, key, value):
        """ Sets the value of a keyword argument.
        
        Parameters
        ----------
        key : str
            Name of the argument whose value will be set.
        val
            New value of the argument.
        """
        if key not in self.args.keys():
            raise ValueError("keyword argument {:s} not found for step {:s}".format(key, self.function_name))
        self.args[key] = value

    def find_filename(self):
        """ If the filename is not specified, look in custom.py followed by functions.py. """
        if self.filename is not None:
            return self.filename
        else:
            for path in [ROOT_DIR+'/custom.py', ROOT_DIR+'/functions.py']:
                try:
                    util.import_function_from_file(self.function_name, path)
                    filename = path.split('/')[-1]
                    return filename
                except KeyError:
                    continue

        raise ValueError("specify the filename for function '{:s}' (not found in custom.py or functions.py)"\
                            .format(self.function_name))

    def load_function(self, reload=False):
        """ Loads or re-loads the function's description and keyword arguments.
        
        Parameters
        ----------
        reload : bool, optional
            Whether or not to reload the default values for the arguments.
        """

        # Determine which file contains the function
        if self.function is None:
            self.filename = self.find_filename()
            func = util.import_function_from_file(self.function_name, ROOT_DIR+'/'+self.filename)
        else:
            func = self.function

        params = inspect.signature(func).parameters
        for k,v in params.items():
            if v.default == inspect._empty:
                continue
            if k not in self.args or reload:
                self.args[k] = v.default

        # Delete arguments which have been removed from the function
        for k in list(self.args.keys()):
            if k not in params:
                del self.args[k]

        # Save the function docstring for easy reference
        self.description = func.__doc__.strip() if func.__doc__ is not None else '(no description)'

def reset_imaging_generator():
    """ Re-creates the default Generator for imaging surveys. """
    g_imaging = Generator(label=None)
    g_imaging.insert_step('read_stellar_catalog')
    g_imaging.insert_step('create_planets_bergsten')
    g_imaging.insert_step('assign_orbital_elements')
    g_imaging.insert_step('impact_parameter')
    g_imaging.insert_step('assign_mass')
    g_imaging.insert_step('compute_habitable_zone_boundaries')
    g_imaging.insert_step('classify_planets')
    g_imaging.insert_step('geometric_albedo')
    g_imaging.insert_step('effective_values')
    g_imaging.insert_step('Example1_water')
    g_imaging.insert_step('Example2_oxygen')

    g_imaging.save(label='imaging')

def reset_transit_generator():
    """ Re-creates the default Generator for transit surveys. """
    g_transit = Generator(label=None)
    g_transit.insert_step('create_stars_Gaia')
    g_transit.insert_step('create_planets_bergsten')
    g_transit.insert_step('assign_orbital_elements')
    g_transit.insert_step('geometric_albedo')
    g_transit.insert_step('impact_parameter')
    g_transit.insert_step('assign_mass')
    g_transit.insert_step('compute_habitable_zone_boundaries')
    g_transit.insert_step('compute_transit_params')
    g_transit.insert_step('classify_planets')
    g_transit.insert_step('scale_height')
    g_transit.insert_step('Example1_water')
    g_transit.insert_step('Example2_oxygen')

    g_transit.set_arg('transit_mode', True)
    g_transit.save(label='transit')
