# System modules
from astropy.nddata import NDDataArray, NDDataRef, StdDevUncertainty
from copy import deepcopy
import inspect
import joblib
import json
import multiprocessing as mp
import numpy as np
import os
import pickle
import shutil
import time
import traceback
from warnings import warn

# Bioverse modules
import util
from util import ROOT_DIR,ATMOSPHERE_TEMPLATES_DIR,MODELS_DIR,INSTRUMENTS_DIR,OBJECTS_DIR
from util import STR_TYPES,LIST_TYPES,ARRAY_TYPES,INT_TYPES,BOOL_TYPES
from util import CONST
#from PySG import ExoplanetTemplate,Params,run_multiple
import plots
from pdfs import normal

# HELA
#import rfretrieval as hela

class Object():
    """ This class manages I/O for some of the objects below. Objects can be initialized from a CSV file or loaded
    directly from a template under ./Objects/OBJECT_NAMEs/. The actual parsing of the CSV file is done by the child
    class with an initialize() method.

    Parameters
    ----------
    label : str, optional
        Label to assign to this object. If `filename` is not given, the object will be loaded from
        <label>.pkl under the directory ./<object type>s/, for example ./Generators/default.pkl.
    filename : str, optional
        Name of the CSV file from which to initialize the Object.
    """
    def __init__(self,label=None,filename=None):
        # Save the label if given
        self.label = label

        # If a label is given, synthesize a .pkl filename from it
        if filename is None and label is not None:
            filename = OBJECTS_DIR+'/'+type(self).__name__+'s/{:s}.pkl'.format(label)
        
        if filename is not None:
            # First try to load the file as a .pkl file and copy over the keys and attributes
            try:
                pkl = pickle.load(open(filename,'rb'))
                for key in pkl.__dict__.keys():
                    self.__dict__[key] = pkl.__dict__[key]
                # If the sub-class inherits from dict, then add the dict keys as well
                if isinstance(pkl,dict):
                    for key in pkl.keys():
                        self[key] = pkl[key]
            
            # If this fails, initialize the object using a CSV file and the initialize() method of the child program
            except pickle.UnpicklingError:
                self.initialize(filename)
    
    def save(self,label=None,filename=None):
        """ Saves the Object as a template in a .pkl file under ./<object type>s/.
        
        Parameters
        ----------
        label : str, optional
            Label under which to save the template. If not given, use self.label.
        filename : str, optional
            Filename to which to save the template. Overrides `label` if given.
        """
        if label is None:
            label = self.label
            if label is None:
                raise ValueError("no label specified")
        else:
            self.label = label
        if label is not None and filename is None:
            filename = OBJECTS_DIR+'/'+type(self).__name__+'s/{:s}.pkl'.format(label)
        pickle.dump(self,open(filename,'wb'))

class Generator(Object):
    """ This class loads a program and uses it to generate a set of nearby systems.
    
    Parameters
    ----------
    label : str, optional
        Label to assign to this Generator. See the Object class description for details.
    filename : str, optional
        Name of the CSV file from which to initialize the Object. See the Object class description for details.
    """
    def __init__(self,label='IMF_SAG13',filename=None):
        self.steps = []
        Object.__init__(self,label,filename)
        # Reloads each step to catch new keyword arguments
        self.update_steps()
    
    def __repr__(self):
        s = "Generator '{:s}' with {:d} steps:".format(self.label,len(self.steps))
        for i in range(len(self.steps)):
            s += "\n{:2d}: {:s}".format(i,self.steps[i].__repr__(long_version=False))
        return s

    class Step():
        """ This class describes each step in the Generator, including its keyword argument descriptions and values.

        Parameters
        ----------
        function_name : str
            Name of the function to be run, as found within the file specified by `path`.
        path : str, optional
            Points to the filename containing the function for this step. Defaults to ./functions.py.
        description : str, optional
            Human-readable description of this step.
        """
        def __init__(self,function_name,path='functions.py'):
            self.function_name = function_name
            self.path = path
            self.args = {}
            self.load_function()

        def __repr__(self,long_version=True):
            s = "Function '{:s}' with {:s} keyword arguments.".format(self.function_name,
                str(len(self.args)) if len(self.args) else 'no')
            if long_version:
                if len(self.description) > 0:
                    s += "\nDescription:\n    {:s}".format(self.description)
                if len(self.args) > 0:
                    s += '\nKeyword arguments:'
                    for key,val in self.args.items():
                        descr = 'no description' if val[1] is None else val[1]
                        s += '\n    {:14s} = {:10s} ({:s})'.format(key,str(val[0]),descr)
            return s
        
        def run(self,d,**kwargs):
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
            func = util.import_function_from_file(self.function_name,ROOT_DIR+'/'+self.path)
            kwargs2 = {key:val[0] for key,val in self.args.items()}
            for key,val in kwargs.items():
                if key in self.args:
                    kwargs2[key] = val
            return func(d,**kwargs2)

        def get_arg(self, key):
            """ Returns the value of a keyword argument. """
            return self.args[key][0]
        
        def set_arg(self,key,value):
            """ Sets the value of a keyword argument.
            
            Parameters
            ----------
            key : str
                Name of the argument whose value will be set.
            val
                New value of the argument.
            """
            if key not in self.args.keys():
                raise ValueError("keyword argument {:s} not found for step {:s}".format(key,self.function_name))
            self.args[key][0] = value
        
        def set_arg_description(self,key,description):
            """ Sets the description of a keyword argument.
            
            Parameters
            ----------
            key : str
                Name of the argument whose description will be set.
            description : str
                Human-readable description of the keyword argument.
            """
            if key not in self.args.keys():
                raise ValueError("keyword argument {:s} not found for step {:s}".format(key,self.function_name))
            self.args[key][1] = description

        def load_function(self,reload=False):
            """ Loads or re-loads the function's description and keyword arguments.
            
            Parameters
            ----------
            reload : bool
                Whether or not to reload the default values for the arguments.
            """
            func = util.import_function_from_file(self.function_name,ROOT_DIR+'/'+self.path)
            params = inspect.signature(func).parameters
            for k,v in params.items():
                if v.default == inspect._empty:
                    continue
                if k not in self.args or reload:
                    self.args[k] = [v.default,None]

            # Delete arguments which have been removed from the function
            for k in list(self.args.keys()):
                if k not in params:
                    del self.args[k]

            self.description = util.get_description_from_function(func)

    def initialize(self,filename):
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
                function_name,path = line.split('#')[0].split(',')[:2]
                description = ','.join(line.split('#')[0].split(',')[2:])
                self.steps.append(Generator.Step(function_name,path))
    
    def update_steps(self,reload=False):
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
    
    def insert_step(self,function_name,idx=None,path='functions.py'):
        """ Inserts a step into the program sequence at the specified index.

        Parameters
        ----------
        function_name : str
            Name of the function to be run by this step.
        idx : int, optional
            Position in the program at which to insert the step. Default is -1 i.e. at the end.
        path : str, optional
            Filename containing the function.
        """
        step = Generator.Step(function_name=function_name,path=path)
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
                print("set {0} = {1} for step '{2}'".format(key, value, step.function_name))

    def generate(self,d=None,timed=False,idx_start=0,idx_stop=None,**kwargs):
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
        for i in range(idx_start,idx_stop):
            try:
                d = self.steps[i].run(d,**kwargs)
                if timed:
                    timer.mark(self.steps[i].function_name)
            except:
                traceback.print_exc()
                print("\n!!! The program failed at step {:d}: {:s} !!!".format(i,self.steps[i].function_name))
                print("!!! Returning incomplete simulation !!!")
                return d
        if timed:
            print("Timing results:")
            timer.read()
        return d

class Table(dict):
    """ Class for storing numpy arrays in a table-like format. Inherits dict.

    Each key is treated as a separate table column. Columns can also contain dict-like objects, in which case
    their keys can be referenced using a semi-colon, e.g. table['Atmosphere:O2'].
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def __repr__(self,show_table=True):
        s1 = 'Table of {:d} objects with {:d} parameters'.format(len(self),len(self.keys()))
        return s1

    def __len__(self):
        """ Returns the number of rows in the table rather than the number of keys (default dict behavior). """
        if len(self.keys()) == 0:
            return 0
        else:
            for key,val in self.items():
                if isinstance(val,(dict,Table)):
                    continue
                else:
                    return 1 if np.isscalar(val) else len(val)
        return 0

    def __getitem__(self,key):
        """ Expands on dict.__getitem__. Passing a string key will return values for the corresponding column. Passing an
        int, int array, bool array, or slice will return values for the corresponding table rows.
        """
        # If `key` is a string then return the column it refers to
        if isinstance(key,STR_TYPES):
            # If a semi-colon is used, retrieve the column from a sub-table, e.g. Planets:Atmosphere:O2
            if ':' in key:
                key1,key2 = self.split_key(key)
                return self[key1][key2]
            return super().__getitem__(key)
        
        # If `key` is an int, int array, bool array, or slice then return the corresponding row(s)
        if isinstance(key,(np.ndarray,slice)) or isinstance(key,INT_TYPES):
            out = type(self)()
            for k in self.keys():
                out[k] = self[k][key]
            return out

        return super().__getitem__(key)

    def __setitem__(self,key,value,force=False):
        """ Expands on dict.__setitem__. Adds a new column to the table.

        Parameters
        ----------
        force : bool, optional
            If True, ignores the requirement that table column lengths match. Used for Table.append.
        """
        # Ensure that the new entry matches the size of existing entries
        if len(self) > 0 and not force:
            if (np.isscalar(value) or isinstance(value,(dict,type(None),Spectrum))) and len(self) == 1:
                pass
            elif ~np.isscalar(value) and len(self) == len(value):
                pass
            else:
                raise ValueError("size of new column does not match other columns")

        # If a semi-colon is used, set the value inside a nested dict, e.g. Planets:Atmosphere:O2
        if ':' in key:
            key1,key2 = self.split_key(key)
            self[key1].__setitem__(key2,value)
        else:
            # Convert lists to NumPy arrays before adding the column))
            super().__setitem__(key,value if (isinstance(value,np.ndarray) or np.size(value)==1) else np.array(value))

    def split_key(self,key):
        """ Splits a key such as 'Planets:Atmosphere:O2' into ('Planets','Atmosphere:O2') and ensures that the first
        key refers to a dict-like object. """
        split = key.split(':')
        key1,key2 = split[0],':'.join(split[1:])
        if isinstance(self[key1],dict):
            return key1,key2
        else:
            raise KeyError("invalid key {:s}; '{:s}' does not refer to a dict-like object".format(key,key1))

    def keys(self):
        """ Returns an array of keys instead of a dict_keys object, because I prefer it this way. """
        return np.array(list(super().keys()))

    def sort_by(self,key,inplace=False,ascending=True):
        """ Sorts the table by the values in one column.

        Parameters
        ----------
        key : str
            Name of the column by which to sort the table.
        inplace : bool, optional
            If True, sort the table in-place. Otherwise return a new sorted Table.
        ascending : bool, optional
            If True, sort from least to greatest.

        Returns
        -------
        sortd : Table
            A sorted copy of this table. Only returned if `inplace` is True.
        """
        order = np.argsort(self[key])[::1 if ascending else -1]
        sortd = self if inplace else self.copy()
        for key in self.keys():
            sortd[key] = self[key][order]
        return None if inplace else sortd
    
    def get_stars(self):
        """ Returns just the first entry for each star in the Table. """
        idxes = np.array([np.where(self['starID']==idx)[0][0] for idx in np.unique(self['starID'])])
        return self[idxes]

    def legend(self,keys=None,filename=ROOT_DIR+'/legend.dat'):
        """ Prints the description of parameter(s) in the Table.

        Parameters
        ----------
        keys : str or str array, optional
            Key or list of keys to describe. If not specified, every key is described.
        filename : str, optional
            CSV file containing the list of parameter descriptions. Default is ./legend.dat.
        """
        # Loop through each line in the legend file and read the key, data type, and description into a dictionary
        leg = {}
        for line in open(filename,'r').readlines():
            if line.strip() == ''  or line.strip()[0] == '#':
                continue
            key,dtype = line.split('#')[0].split(',')[:2]
            description = ','.join(line.split('#')[0].split(',')[2:])
            # Print a warning if duplicate keys are found
            if key.strip() in leg.keys():
                print("WARNING: Duplicate entries found in the legend for parameter {:s}!".format(key))
            leg[key.strip()] = [dtype.strip(),description.strip()]
        
        # Loop through each key and print the data type and description (if available)
        if keys is None:
            keys = self.keys()
        elif np.ndim(keys) == 0:
            keys = [keys]
        for key in keys:
            if key not in self.keys():
                continue
            if key in leg.keys():
                print("{:20s} {:10s} {:s}".format(key,'('+leg[key][0]+')',leg[key][1]))
            else:
                print("{:20s} (not found in legend)".format(key))

    def copy(self):
        """ Returns a deep copy of the Table instead of a shallow copy (as in dict.copy). This way, if a column
        is filled by objects (such as Atmosphere objects), a copy of those is returned instead of a reference. """
        return deepcopy(self)
        
    def append(self, table, inplace=True):
        """ Appends another table onto this table in-place. The second table must have the same
        columns, unless this table is empty, in which case the columns are copied over.
        
        Parameters
        ----------
        table : Table
            Table to be appended onto this one.
        inplace : bool, optional
            If True, append inplace and return None. If False, return a new Table.
        """
        # Append in-place?
        if not inplace:
            out_table = self.copy()
            out_table.append(table, inplace=True)
            return out_table

        # If this table has no columns, then just copy over the columns from `table`
        if len(self.keys()) == 0:
            super().__init__(table)
        
        # Otherwise ensure the columns match then append the rows to the end
        elif np.array_equal(np.sort(self.keys()),np.sort(table.keys())):
            for key in self.keys():
                self.__setitem__(key,np.append(self[key],table[key]),force=True)
        else:
            raise ValueError("attempting to append a Table with different columns")

    def compute(self,key,force=False):
        """ Computes the value of `key` using other values in the dictionary and a pre-defined formula. Useful for
        translating measured values (e.g. 'a', 'L_st') into secondary data products (e.g., 'S'). """
        if key in self and not force:
            # Unless force == True, don't re-compute existing values
            return
        if key[:4] == 'log(':
            k = key[4:-1]
            self.compute(k)
            self[key] = np.log10(self[k])
        elif key[:3] == 'ln(':
            k = key[3:-1]
            self.compute(k)
            self[key] = np.log(self[k])
        elif key == 'S':
            self.compute('a')
            self['S'] = self['L_st']/self['a']**2
        elif key == 'R':
            self['R'] = (109.2**2)*self['depth']*self['R_st']**2
        elif key == 'a':
            self['a'] = (self['M_st']*(self['P']/CONST['yr_to_day'])**2)**(1/3)
        elif key == 'a_eff':
            self.compute('S')
            self['a_eff'] = self['S']**-0.5
        elif key == 'R_eff':
            self['R_eff'] = (self['contrast'] * np.pi / 0.29)**0.5 / (4.258756e-5 / self['a'])
        elif key == 'h_eff':
            self.compute('S')
            self['h_eff'] = CONST['h_Earth']*self['S']/self['R']
        else:
            raise ValueError("no formula defined for {:s}".format(key))
    
    def shuffle(self, inplace=True):
        """ Re-orders rows in the Table. If `inplace` is False, return a new re-ordered Table instead. """
        if not inplace:
            out_table = self.copy()
            out_table.shuffle(inplace=True)
            return out_table

        order = np.random.choice(range(len(self)), replace=False, size=len(self))
        for key, val in self.items():
            self[key] = val[order]

class Atmosphere(dict):
    """
    This class describes a planet's atmosphere. The keys 'T', 'P', and 'z' correspond to
    temperature (in Kelvin), pressure (in bars), and altitude (in km). All other keys describe the fractional abundance
    of species (e.g., 'H2O' or 'CH4').

    The key values can be arrays of matching length, with each entry describing a layer in the atmosphere. 
    Alternatively, if scalars are given, the atmosphere is assumed isothermal, with 'P' describing the surface pressure.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.layered = None
        self.layers = 0
        self.template = ''
        self.surface_albedo = 0.3

    def copy(self,N=1):
        """ Deep-copies the atmosphere N times, returning a list of Atmospheres if N > 1.

        Parameters
        ----------
        N : int, optional
            Number of copies to create. Default is 1.

        Returns
        -------
        copies : Atmosphere or Atmosphere array
            Returns a single copy or an array of copies with length `N`.
        """
        if N == 1:
            return deepcopy(self)
        else:
            return [deepcopy(self) for i in range(N)]

    def init_profile(self,x):
        """ Initializes the atmospheric profile with a certain number of layers. The number of layers is equal to the
        length of the argument, which can be a scalar or array (e.g. surface pressure or vertical pressure profile).
        
        Parameters
        ----------
        x : float or float array
            If an array is passed, then create a layered atmosphere with len(x) layers. Otherwise, create an isothermal atmosphere.
        """
        self.layered = np.ndim(x) == 1
        self.layers = len(x) if self.layered else 0

    def check_input_dims(self,x):
        """ Checks the dimensions of an array or scalar to ensure that they match the current layers, raising an error if not.
        
        Parameters
        ----------
        x : float or float array
            Argument whose dimensions are to be checked.
        """
        if self.layered is not None:
            if self.layered and np.ndim(x) == 0:
                raise ValueError("tried to add a scalar value to an atmosphere with {:d} layers".format(self.layers))
            if self.layered and len(x) != self.layers:
                raise ValueError("tried to add a profile with {:d} layers to an atmosphere with {:d} layers"\
                    .format(len(x),self.layers))
            if (not self.layered) and np.ndim(x) == 1:
                raise ValueError("tried to add a profile with {:d} layers to an isothermal atmosphere".format(len(x)))
        if np.ndim(x) > 1:
            raise ValueError("tried to add a {:d}-dimensional array to an atmosphere".format(np.ndim(x)))

    def add_value(self,key,values):
        """
        Add a parameter to the atmosphere, such as a temperature or gas abundance.

        Parameters
        ----------
        key : str
            Name of the value to be added, e.g. 'T' for temperature or 'H2O' for H2O abundance.
        values : float or float array
            Value or array of values for the parameter in each layer.
        """
        # Initialize the atmospheric profile (if needed) and check that the input dimensions are correct
        if self.layered is None:
            self.init_profile(values)
        self.check_input_dims(values)

        # Add the value to the atmosphere
        self[key] = values

    def remove(self,key):
        """ Removes a parameter from the atmosphere, if it exists.

        Parameters
        ----------
        key : str
            Name of the parameter to be removed.
        """
        if key in self.keys():
            del self[key]
    
    def load(self,filename=None,template='Earth'):
        """ Loads the atmosphere from a template or input file. The input file should consist of at least
        two columns, with the first row listing the parameter names, e.g. '# P T H2O'. The remaining row
        or rows should list the parameter values in each layer of the atmosphere.
        
        Parameters
        ----------
        filename : str, optional
            Name of a file describing the atmospheric profile. Overrides `template`.
        template : str, optional
            Label of the atmospheric template under ./Templates/Atmospheres/. Default is `Earth`.
        
        """
        if filename is None:
            filename = ATMOSPHERE_TEMPLATES_DIR+'/{:s}.dat'.format(template)
        
        # Parse the keys and values
        for line in open(filename,'r').readlines():
            if '#' in line:
                keys = line.strip('#').split()
                break
        values = np.genfromtxt(filename,unpack=True,dtype=float)
        
        # Add them to the atmosphere profile
        for i in range(len(keys)):
            self.add_value(keys[i],values[i])

        # Save the template used
        self.template = filename.split('/')[-1].split('.')[0]
    
    def get_layer(self,idx):
        """ Returns the value of each key in the specified layer.
        
        Parameters
        ----------
        idx : int
            Index of the layer to return.
        """
        if self.layered:
            return np.array([self[key][idx] for key in self.keys()])
        else:
            return np.array([self[key] for key in self.keys()])

    def get_species_list(self):
        """ Returns a list of chemical species in the atmosphere. """
        return np.array([key for key in self.keys() if key not in ['P','T','z']])

    def set_abundance(self,species,abundance):
        """ Sets the mixing ratio of a species already present in the atmosphere. If the atmosphere is layered,
        then the species profile is modified to match the pressure-averaged abundance to this value.
        
        Parameters
        ----------
        species : str
            Species to change the abundance of.
        abundance : float
            New abundance or pressure-averaged abundance.
        """
        if species not in self.keys():
            raise ValueError("species {:s} not found in atmosphere".format(species))

        if self.layered:
            self[species] *= abundance/np.average(self[species],weights=self['P'])
        else:
            self[species] = abundance

    def flatten(self,inplace=False):
        """ Converts a layered atmosphere into a single layer, isothermal atmosphere. The maximum pressure is assumed
        to be the surface pressure. The temperature and abundances are computed from a pressure-weighted average.
        
        TODO: Fix the average temperature, abundances.
        
        Parameters
        ----------
        inplace : bool, optional
            If True, flattens the atmosphere "in place". Otherwise, returns a new Atmosphere object.
            
        Returns
        -------    
        atm : Atmosphere
            Flattened version of the Atmosphere - only returned if `inplace` is False.
        """
        atm = self if inplace else deepcopy(self)
        if 'P' in atm.keys():
            atm.layered = False
            atm.layers = 0
            for key in atm.keys():
                if key == 'P': continue
                atm[key] = np.average(atm[key],weights=atm['P'])
            atm['P'] = np.amax(atm['P'])
        return atm if not inplace else None

    def save(self,filename):
        """ Saves the atmosphere to an output file. See the description of Atmosphere.load for the output format.
        
        Parameters
        ----------
        filename : str
            Filename of the file to which to save the atmosphere.
        """
        with open(filename,'w') as f:
            f.write('# {:s}\n'.format(' '.join(self.keys())))
            if self.layered:
                for i in range(self.layers):
                    sout = ' '.join(['{:.3E}'.format(val) for val in self.get_layer(i)])
                    f.write(sout+'\n')
            else:
                f.write(' '.join(['{:.3E}'.format(val) for val in self.get_layer(0)])+'\n')

class Survey(dict,Object):
    """
    Describes an exoplanet survey, including properties and methods for calculating detection yield estimates
    and simulated data sets.

    Parameters
    ----------
    label : str, optional
        Label to assign to this Survey. See the Object class description for details.
    filename : str, optional
        Name of the plain text file from which to initialize the Survey. See the Object class description for details.
    """
    def __init__(self,label='imaging',filename=None):
        super().__init__()
        # Always initialize from a filename
        if filename is not None:
            label = filename.split('/')[-1].split('.')[-2]
        else:
            filename = OBJECTS_DIR+'/Surveys/{:s}.dat'.format(label)
        Object.__init__(self,label,filename)

        # Max survey time and max number of observations to allow
        self.t_max = 10 * 365.25
        self.N_obs_max = 1000
    
    def __repr__(self):
        s = "Survey simulator with the following properties:"
        s += "\n   Name: {:s}".format(self.label)
        s += "\n   Survey type: {:s}".format(self.mode)
        for i,m in enumerate(self.measurements):
            if i == 0:
                s += "\n\nPerforms {:d} measurements:".format(len(self.measurements))
            s += "\n({:d}) {:s}".format(i,m.__repr__())
        return s

    def initialize(self,filename):
        """ Creates a new Survey object from a plain text input file.
        
        Parameters
        ----------
        filename : str
            Filename of a plain text file describing the survey. See README for details on the input format.
        """
        # Create a list of measurements
        self.measurements = []
        # Discard blank lines and comments
        lines = open(filename,'r').readlines()
        lines = [line.strip().split('#')[0] for line in lines if line.strip() != '' and line.strip()[0] != '#']
        # Discard blank space around conditional operators (e.g. x > 1 turns into x>1)
        lines = [line.replace(' > ','>').replace(' < ','').replace(' == ','==') for line in lines]
        # Loop through each line and parse the input
        measurement = False
        for i in range(len(lines)):
            if lines[i].strip() == '' or lines[i].strip()[0] == '#': continue
            vals = lines[i].split('#')[0].split()
            # Process single value lines
            if len(vals) == 1:
                val = vals[0]
                # This line begins a new Measurement definition
                if val.strip() == '%measurement':
                    # If a previous Measurement is done being read in, then add it to the Survey
                    if measurement:
                        self.measurements.append(Measurement(**measurement_kwargs))
                    measurement = True
                    measurement_kwargs = {'key':None,'survey':self,'precision':0.,'condition':[],'t_total':np.inf,
                                          't_ref':None,'t_min':0.,'priority':{},'wl_eff':0.5,'debias':True,'bounds':None}
                    continue
            # Process multiple value lines
            elif len(vals) >= 2:
                key,val = vals[0],[util.get_type(v)(v) for v in vals[1:]]
                # Pre-defined Survey keywords
                survey_keys = ['yield_file','instrument','inner_working_angle','outer_working_angle','diameter',
                               'contrast_limit','atmosphere_depth_limit','t_slew','R_st_ref','T_st_ref','D_ref','d_ref']
                if key == 'mode':
                    self.mode = val[0]
                    continue
                elif key in survey_keys:
                    self[key] = val[0]
                    continue
                # Measurement parameters (if %measurement has been read)
                elif measurement and key in measurement_kwargs:
                    v = val[0] if len(val) == 1 else val
                    if isinstance(measurement_kwargs[key], list):
                        measurement_kwargs[key].append(v)
                    elif isinstance(measurement_kwargs[key], dict):
                        if v[0] not in measurement_kwargs[key]:
                            measurement_kwargs[key][v[0]] = []
                        measurement_kwargs[key][v[0]].append(v[1:])
                    else:
                        measurement_kwargs[key] = v
                    continue

            print("Not sure what to do with this line: {:s}".format(lines[i]))
        
        # Save any remaining Measurements
        if measurement:
            self.measurements.append(Measurement(**measurement_kwargs))

    def quickrun(self, generator, t_total=None, N_sim=1, *args, **kwargs):
        """ Convenience function which generates a sample, computes the yield, and returns a simulated data set.
        
        Parameters
        ----------
        generator : Generator
            Generator object which is used to generate the planet population.
        t_total : float, optional
            Total amount of observing time for any measurements with a limited observing time.
        N_sim : int, optional
            If > 1, simulate the survey this many times and return the combined result.
        *args, **kwargs
            Arguments and keyword arguments to be passed to the generator.

        Returns
        -------
        sample : Table
            Table of all simulated planets.
        detected : Table
            Table of planets detected by the survey.
        data : Table
            Simulated data set produced by the survey.
        """
        
        # For transit surveys, set transit_mode=True unless otherwise specified
        if self.mode == 'transit' and 'transit_mode' not in kwargs:
            kwargs['transit_mode'] = True

        if N_sim > 1:
            sample, detected, data = Table(), Table(), Table()
            for i in range(int(N_sim)):
                res = self.quickrun(generator, t_total=t_total, N_sim=1, *args, **kwargs)
                sample.append(res[0])
                detected.append(res[1])
                data.append(res[2])
        else:
            sample = generator.generate(*args, **kwargs)
            detected = self.compute_yield(sample)
            data = self.observe(detected, t_total=t_total)

        return sample, detected, data

    def compute_yield(self,d,simple=True,sort_key=None):
        """ Computes the detection yield from a simulated population of stars and planets. Different methods are used
        to calculate detection yields for 'transit' and 'imaging' surveys.
        
        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.
        simple : bool, optional
            If True, compute a simple yield estimate, otherwise apply a pre-calculated yield estimate from the 
            ./Yields directory.
        sort_key : str, optional
            If True, sort the sample by this key value (e.g., 'd') before returning.

        Returns
        -------
        detected : Table
            Copy of the input table containing only planets which were detected by the survey.
        """
        if simple:
            if self.mode == 'imaging':
                detected = self.compute_imaging_yield(d)
            elif self.mode == 'transit':
                detected = self.compute_transit_yield(d)
        else:
            detected = self.apply_precalculated_yield(d)
        return detected[np.argsort(detected[sort_key])] if sort_key is not None else detected
    
    def compute_imaging_yield(self, d, wl_eff=0.5, A_g=0.3):
        """ Computes a simple estimate of the detection yield for an imaging survey. Compares the contrast ratio and
        projected separation of each planet when observed at quadrature to the contrast limit and inner/outer working
        angles of the survey. Planets which satisfy these criteria are considered to be detected.
        
        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.
        wl_eff : float, optional
            Effective wavelength of observation in microns (used for calculating the IWA/OWA).
        A_g : float, optional
            Geometric albedo of each planet, ignored if 'A_g' is already assigned.

        Returns
        -------
        yield : Table
            Copy of the input table containing only planets which were detected by the survey.
        """

        # Compute the angular separation at quadrature (milli-arcseconds)
        separation = d['a'] / d['d'] * 1000

        # If no albedo is given, assume one
        if 'A_g' not in d:
            d['A_g'] = np.full(len(d), A_g)

        # Determine which planets are brighter than the contrast limit
        if 'contrast' not in d:
            d['contrast'] = d['A_g'] * (4.258756e-5*d['R']/d['a'])**2 / np.pi
        mask1 = d['contrast'] > 10**self['contrast_limit']

        # Determine which planets are within the field of view
        iwa = self['inner_working_angle'] * (1e-6*wl_eff)/ self['diameter']*206265*1000
        owa = self['outer_working_angle'] * (1e-6*wl_eff) / self['diameter']*206265*1000
        mask2 = (separation > iwa) & (separation < owa)

        # Return the output table
        return d[mask1 & mask2]
    
    def compute_transit_yield(self,d):
        """ Computes a simple estimate of the detection yield for an imaging survey. Compares the transit depth induced
        by one atmospheric scale height of the planet's atmosphere to a minimum threshold for the survey. Planets which
        satisfy this criteria are considered to be detected.

        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.

        Returns
        -------
        yield : Table
            Copy of the input table containing only planets which were detected by the survey.
        """
        # Determine which transiting planets have transit depths above the limit
        mask1 = d['transiting']
        #mask2 = d['atmosphere_depth'] > self['atmosphere_depth_limit']

        # Return the output table
        return d[mask1]

    def apply_precalculated_yield(self,d,multiple=1.):
        """ Applies the yield estimates specified in self['yield_file'] to a simulated population of planets. First, the
        simple method is used to determine which planets lie within the loose detection thresholds of the Survey. Then,
        planets are selected in order of distance until the pre-computed yield estimates have been met.
        
        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.
        multiple : float, optional
            Multiply the number of planets detected in each category by this amount.

        Returns
        -------
        y : Table
            Copy of the input table containing only planets which were detected by the survey.
        """
        # Apply the simple yield estimate to prune the sample
        y = self.compute_yield(d,simple=True)
        y_sort = y.sort_by('d')

        # Loop through each line in the yield estimate file
        filename = ROOT_DIR+'/Yields/{:s}'.format(self['yield_file'])
        include,incomplete = [],False
        for line in np.loadtxt(filename,dtype='str'):
            # Size class, temperature class, stellar spectral type, and yield estimate for each line
            class2,class1,SpT,N = line
            N = (N.astype(int)*multiple).astype(int)
            
            # Include the first (i.e. nearest) N planets which match these criteria
            bin_mask = (y_sort['class1']==class1)&(y_sort['class2']==class2)&(y_sort['SpT']==SpT)
            if bin_mask.sum() < N:
                incomplete = True
                N = bin_mask.sum()
                if N == 0:
                    continue
            include = np.append(include,y_sort['planetID'][bin_mask][:N]).astype(int)

        # Print a warning if the yield is incomplete due to missing planets
        if incomplete:
            print("Not enough planets were found for every bin in the yield estimate")

        # Return a new table with just the detected planets
        return y[np.in1d(y['planetID'],include)].copy()
        
    def observe(self, y, t_total=None, data=None):
        """ Returns a simulated data set for a table of simulated planets. Each measurement specified in the
        survey configuration file is performed on every planet in the table.
        
        Parameters
        ----------
        y : Table
            Table containing the set of planets to be observed, usually the detection yield of the survey.
        t_total : float, optional
            Total observing time for any measurements with limited observing time.
        data : Table, optional
            Table in which to store the measurements.

        Returns
        -------
        data : Table
            Table of measurements made by the survey, with one row for each planet observed.
        """
        # Create an output Table to store measurements
        data = data if data is not None else Table()

        # Perform each measurement in order
        for m in self.measurements:
            if m.key not in y.keys():
                print("Could not measure property: {:s}".format(m.key))
                continue

            # Determine the amount of time for this measurement (if it has any exposure time constraints)
            t = np.inf
            if t_total is not None and m.t_ref is not None:
                if isinstance(t_total, dict) and m.key in t_total:
                    t = t_total[m.key]
                elif not isinstance(t_total, dict):
                    t = t_total
                    
            data = m.measure(y, data, t_total=t)
        return data
    
    def simulate_spectra(self,y,do_all=True,scale_exptime=True,apply_noise=True,local=True):
        """ Simulates spectra for a databse of simulated planets using PySG and PSG. Only planets with defined Atmospheres
        are observed. This only works for 'imaging' mode right now.
        
        Parameters
        ----------
        y : Table
            Table containing the set of planets to be observed, usually the detection yield of the survey.
        do_all : bool, optional
            If False, just simulate the spectrum of the first planet. Useful for testing purposes.
        scale_exptime : bool, optional
            If True, scale the exposure time for distance.
        apply_noise : bool, optional
            Whether noise should be simulated in the spectra. The uncertainties will be estimated either way.
        local : bool, optional
            Run PSG on the local machine, or through the web API (not recommended for multiple spectra).
        """
        # Blank entries for planets without atmospheres
        atmosphere_mask = ~np.in1d(y['Atmosphere'],[{},None])
        y['Spectrum'] = [None for i in range(len(y))]
        
        # Load the PSG configuration file of the Survey instrument
        instrument = Params(filename=INSTRUMENTS_DIR+'/{:s}.cfg'.format(self['instrument']),category='GENERATOR')

        # Loop through planets with Atmospheres
        templates = []
        for idx in np.arange(len(y))[atmosphere_mask]:
            # Create a PySG template for this object and edit some relevant parameters
            pl0 = y[idx]
            template = ExoplanetTemplate(template='LUVOIR_ExoEarth')
            template.set_planet_properties(R_pl=pl0['R'],M_pl=pl0['M'],a=pl0['a'],e=pl0['e'],
                                           inc=np.arccos(pl0['cos(i)'])*180./np.pi)
            spt = pl0['SpT'] if pl0['SpT'] in ['A','G','K','M'] else 'U'
            template.set_star_properties(R_st=pl0['R_st'],T_eff=pl0['T_eff_st'],spt=spt,d=pl0['d'])

            # Specify the isothermal or vertical atmosphere composition
            atm = pl0['Atmosphere']
            species = atm.get_species_list()
            abundances = [atm[key] for key in species]
            if np.size(atm['P']) == 1:
                template.new_isothermal_atmosphere(atm['P'],atm['T'],species=species,abundances=abundances,Punit='bar')
            else:
                template.new_atmospheric_profile(atm['P'],atm['T'],species=species,abundances=abundances,Punit='bar')

            # Set the surface albedo (TODO: better implementation of surfaces in PySG)
            template['SURFACE-ALBEDO'] = atm.surface_albedo

            # Specify the instrument parameters
            template.apply(instrument)
            template['GENERATOR-RADUNITS'] = 'rif' if self.mode == 'imaging' else 'rel'

            # Imaging mode: Determine the total exposure time based on distance (scaling from 5 pc)
            if self.mode == 'imaging' and 'ref_exptime' in self.keys():
                    exptime = self['ref_exptime'] * (pl0['d']/5.)**2

                    # Enforce minimum/maximum total exposure times, if supplied
                    if 'max_exptime' in self.keys():
                        exptime = min(exptime,self['max_exptime'])
                    if 'min_exptime' in self.keys():
                        exptime = max(exptime,self['min_exptime'])

                    # Implement the total exposure time by modifying the number of exposures and/or single exposure time
                    if exptime < template['GENERATOR-NOISETIME']:
                        template['GENERATOR-NOISETIME'] = exptime
                        template['GENERATOR-NOISEFRAMES'] = 1
                    else:
                        template['GENERATOR-NOISEFRAMES'] = int(exptime/template['GENERATOR-NOISETIME'])
                        template['GENERATOR-NOISETIME'] = exptime/template['GENERATOR-NOISEFRAMES']

            # Append to the list of templates
            templates.append(template)

            if not do_all:
                result = template.run_imaging_spectrum(phase=90.,apply_noise=apply_noise,local=local)
                y['Spectrum'][idx] = Spectrum(*result)
                return

        # Run all of the templates using the PySG multi-processing mode and append the results to the sample
        results = run_multiple(templates,function=ExoplanetTemplate.run_imaging_spectrum,
                               apply_noise=apply_noise,local=local)
        for i,idx in enumerate(np.arange(len(y))[atmosphere_mask]):
            y['Spectrum'][idx] = Spectrum(*results[i])

    def get_measurement(self, parameter):
        """ Returns the first measurement of `parameter` performed by this survey. """
        return self.get_measurements(parameter)[0]

    def get_measurements(self, parameter):
        """ Returns all measurements of `parameter` performed by this survey. """
        measurements = tuple([m for m in self.measurements if m.key == parameter])
        if len(measurements) == 0:
            return ValueError("survey '{:s}' does not measure parameter '{:s}'".format(self.label, parameter))
        return measurements

class Measurement():
    """
    Class which describes a simple measurement to be applied to a table of planets detected by a Survey.

    Parameters
    ----------
    key : str
        Name of the parameter will be measured (e.g., 'M' for planet mass)
    precision : str or float, optional
        Precision of measurement, e.g. '10%' or 0.10 units. If zero, a perfect measureent is made.
    condition : str array, optional
        Conditional statements specifying which planets the measurement applies to (e.g., '0.5 < R < 1.5').
        Defaults to None, in which case the measurement applies to all planets.
    t : float, optional
        Amount of time required per measurement, arbitrary unit. If `t_scale` is specified, then `t` is the time
        required for a reference case i.e. where all of the scale factors are 1.
    t_total : float, optional
        Total amount of time allocated for this measurement, same unit as `t`. The measurement will be made for as 
        many targets as possible within this time allocation.
    priority : dict, optional
        Each entry describes how targets should be prioritized based on a planet property. For example,
        {'S':(0.3,1.,2)} indicates that planets with 0.3 < S < 1 should have 2x priority (i.e. the same priority
        as a target requiring half as much observing time).
    """
    def __init__(self,key,survey,precision=0.,condition=[],t_total=None,t_ref=None,t_min=0.,priority={},wl_eff=0.5,debias=True,bounds=None):
        super().__init__()

        # Save the keyword values
        self.key = key
        self.survey = survey
        self.precision = precision
        self.conditions = condition
        self.t_total = t_total
        self.t_ref = t_ref
        self.t_min = t_min
        self.priority = {key:np.array(val) for key, val in priority.items()}
        self.wl_eff = wl_eff
        self.debias = debias
        self.bounds = bounds if bounds is not None else [-np.inf, np.inf]

    def __repr__(self):
        s = "Measures parameter '{:s}'".format(self.key)
        if self.precision != 0.:
            s += " with {:s} precision".format(str(self.precision))
        for i,cdtn in enumerate(self.conditions):
            s += "\n    Conditions: {:s}".format(cdtn) if i == 0 else ' AND {:s}'.format(cdtn)
        if not np.isnan(self.t_total):
            s += "\n    Total allocated time: {:.1f}".format(self.t_total)
            
        return s

    def measure(self, detected, data=None, t_total=None):
        """ Produces the measurement for planets in a table and places them into a data table.
        
        Parameters
        ----------
        detected : Table
            Table containing detected planets.
        data : Table, optional
            Table in which to store the measured values for each planet. If not given,
            then a new table is created.
        
        Returns
        -------
        data : Table
            Table containing the measured values for each planet.
        """
        # Create a new output table if necessary
        if data is None:
            data = Table()
        data['planetID'] = detected['planetID']
        data['starID'] = detected['starID']
        
        # Determine which planets are valid targets, and which of those can be observed in the allotted time
        valid = self.compute_valid_targets(data)
        observable = self.compute_observable_targets(data, valid, t_total)

        # Simulate a measurement for each observable planet
        x = self.perform_measurement(detected[self.key][observable])

        # Place this measurement into the measurements database
        if self.key not in data.keys():
            data[self.key] = np.full(len(detected),np.nan)
        data[self.key][observable] = x

        return data
    
    def compute_valid_targets(self, data):
        """ Determines which planets are valid targets based on whether their previously measured parameters
        meet the measurement conditions.

        Parameters
        ----------
        data : Table
            Table of data values already measured for these planets.
        
        Returns
        -------
        valid : bool array
            Specifies which planets in the table are valid targets.
        """
        valid = np.ones(len(data),dtype=bool)
        
        # Supported comparison strings
        comparisons = {'<': np.less, '>': np.greater, '==': np.in1d}
        
        cdtns = self.conditions
        if cdtns is not None:
            if np.ndim(cdtns) == 0: cdtns = [cdtns]
            for cdn in cdtns:
                # Determine which parameter is being compared to which value, and which comparison to perform
                cmp_str = [cs for cs in comparisons if cs in cdn][0]
                key, val = cdn.split(cmp_str)
                key, val = key.strip(), val.strip()

                # Ensure that the parameter has already been measured, else try to compute it
                if key not in data:
                    data.compute(key)
                
                # Make the comparison, assuming the comparison value to be the same type as in `d`
                try:
                    dtype = type(data[key][~np.isnan(data[key])][0])
                    valid = valid & comparisons[cmp_str](data[key], dtype(val))
                except IndexError:
                    valid = valid & np.zeros(len(valid), dtype=bool)
        
        return valid
        
    def compute_observable_targets(self, data, valid, t_total=None):
        """ Determines which planets are observable based on the total allotted observing time.

        Parameters
        ----------
        data : Table
            Table of data values already measured for these planets.
        valid : bool array
            Mask specifying which targets in `data` are valid targets.
        t_total : float, optional
            Total observing time for this measurement. If None, use self.t_total.
        
        Returns
        -------
        observable : bool array
            Specifies which planets in the table are observable within the allotted time.
        """

        # If t_total is infinite, then all valid targets are observable
        if t_total is None:
            t_total = self.t_total
        if np.isinf(t_total):
            return valid

        # Compute the required exposure and overhead time for each target
        t_exp, N_obs = self.compute_exposure_time(data)
        t_over = self.compute_overhead_time(data)

        # (Transit mode) Targets are invalid if they cannot be observed within 10 yr
        if self.survey.mode == 'transit':
            valid[(N_obs*data['P']) > self.survey.t_max] = False
            valid[N_obs > self.survey.N_obs_max] = False

        # Compute the priority of each target; invalid targets have zero priority
        weights = self.compute_weights(data)
        weights *= self.compute_debias(data)
        priority = weights/(t_exp+t_over)
        priority[~valid] = 0.

        # Observe planets in order of priority, until we run out of time or valid targets
        t_sum, observable = 0., np.zeros(len(data), dtype=bool)
        for i in range(valid.sum()):
            # Determine the current highest priority planet and how much time is required to observe it
            t_obs = t_exp + t_over
            idx = np.argmax(priority)

            # Add this amount of time to the budget - if it's too much, then stop observing immediately
            t_sum += t_obs[idx]
            if t_sum > t_total:
                break

            # (Imaging mode) Observe the target along with other planets in the same system
            if self.survey.mode == 'imaging':
                t_exp[data['starID']==data['starID'][idx]] -= t_exp[idx]
                
            # (Transit mode) Only observe the target
            elif self.survey.mode == 'transit':
                t_exp[idx] -= t_exp[idx]

            # Mark planets with t_exp <= 0 as "observable" and remove them from the line-up
            finished = t_exp <= 0
            observable[finished&valid], priority[finished] = True, 0.

        return observable

    def compute_exposure_time(self, d):
        """ Computes the exposure time and number of observations required to characterize each planet in `d`. """
        wl = self.wl_eff / 10000 # convert from microns -> cm
        h, c, k, T_eff_sol = 6.6261e-27, 2.9979e10, 1.3807e-16, 5780.

        # Reference case parameters
        T_ref, R_ref, D_ref, d_ref = self.survey['T_st_ref'], self.survey['R_st_ref'], self.survey['D_ref'], self.survey['d_ref']

        # Scaling factors for each survey mode
        if self.survey.mode == 'imaging':
            f = (d['contrast']/1e-10)**-1
        if self.survey.mode == 'transit':
            d.compute('S')
            f = (d['H']/9)**-2 * d['R']**-2 * (d['R_st']/R_ref)**4

        # Calculate the exposure time required for each target, based on the reference time
        flux_ratio = (np.exp(h*c/(wl*k*T_ref)) - 1) / (np.exp(h*c/(wl*k*d['T_eff_st'])) - 1)
        t_exp = f * self.t_ref * (self.survey['diameter']/D_ref)**-2 * (d['d']/d_ref)**2 * (d['R_st']/R_ref)**-2 * flux_ratio**-1

        # Number of observations (1 for imaging mode, integer multiple of transit duration for transit mode)
        if self.survey.mode == 'imaging':
            N_obs = np.ones(len(d))
        if self.survey.mode == 'transit':
            N_obs = np.ceil(t_exp/d['T_dur'])
            N_act = t_exp/d['T_dur']
            t_exp = d['T_dur']*N_obs

        return t_exp, N_obs

    def compute_overhead_time(self, d, N_obs=1):
        """ Computes the overheads associated with each observation. """

        # Imaging mode: 1x overhead for each target
        if self.survey.mode == 'imaging':
            t_over = np.full(len(d), self.survey['t_slew'])

        # Transit mode: N_obs x (overhead + T_dur) for each target
        if self.survey.mode == 'transit':
            t_over = N_obs * (self.survey['t_slew'] + d['T_dur'])

        return t_over

    def compute_weights(self,d):
        """ Computes the priority weight of each planet in `d`. """
        weights = np.ones(len(d))
        for key, val1 in self.priority.items():
            # Try to compute d[key] if it hasn't already been measured
            d.compute(key)
            for val2 in val1:
                weights[(d[key] >= val2[0]) & (d[key] <= val2[1])] *= val2[2]
        return weights

    def compute_debias(self, d):
        """ Removes detection biases from the data set (transit mode only). """
        weights = np.ones(len(d))
        if not self.debias:
            return weights

        d.compute('a')
        d.compute('a_eff')
        if 'a' in d and self.survey.mode == 'transit':
            weights = d['a']/d['R_st']
        return weights

    def compute_priority(self, data):
        """ Computes the target priority based on its weight and required exposure time. """
        # Determine which targets are valid targets; if no targets are valid, return zeros
        valid = self.compute_valid_targets(data)
        if valid.sum() == 0:
            return np.zeros(len(valid), dtype=bool)

        # Compute the exposure times and overheads for each target
        t_exp, N_obs = self.compute_exposure_time(data)
        t_over = self.compute_overhead_time(data, N_obs)

        # (Transit mode) Targets are invalid if N_obs * P > 10 yr
        if self.survey.mode == 'transit':
            valid[(N_obs*data['P']) > self.survey.t_max] = False
            valid[N_obs > self.survey.N_obs_max] = False

        # Compute target weights (invalid targets have no weight)
        #weights = self.compute_weights(data)
        weights = self.compute_weights_new(data)
        weights[~valid] = 0.

        return weights/(t_exp+t_over)

    def perform_measurement(self,x):
        """ Simulates measurements of the parameter from a set of true values.
        
        Parameters
        ----------
        x : array
            Array of true values on which to perform the measurement.

        Returns
        -------
        xm : array
            Array of measured values with the same length and type as `x`.
        """
        # Skip empty values
        if len(x) == 0:
            return x

        # If the value is type str or bool then the measurement has to be "exact"
        if type(x[0]) in list(np.append(STR_TYPES,BOOL_TYPES)):
            return x

        # Percentage-based precision
        if type(self.precision) is str and '%' in self.precision:
            sig = x*float(self.precision.strip('%'))/100.
            
        # Absolute precision
        else:
            sig = float(self.precision)

        # Return draw from bounded normal distribution
        return normal(x, sig, xmin=self.bounds[0], xmax=self.bounds[1], size=len(x))

# class Data(np.ndarray):
#     """ Sub-class of np.ndarray to describe a set of data points with measurement uncertainties.
    
#     Parameters
#     ----------
#     input_array : float or float array
#         Array of data values.
#     error_array : float or float array, optional
#         Errors or uncertainties associated with each data value. If one value is given, assume the same
#         uncertainty for each data point.

#     Attributes
#     ----------
#     error : array
#         Error associated with each data point.
#     """
#     def __new__(cls, input_array, error_array=None):
#         arr = np.asarray(input_array).view(cls)
#         if error_array is None:
#             arr.error = None
#         elif np.size(error_array) == arr.size:
#             arr.error = np.asarray(error_array)
#         elif np.ndim(error_array) == 0:
#             arr.error = np.full_like(arr, error_array)
#         else:
#             raise ValueError("error_array must either be a scalar or have the same size as input_array")
#         return arr

#     def __getitem__(self, item):
#         """ Returns the same as ndarray.__getitem__, but keeps the associated errors. """
#         input_array = super().__getitem__(item)
#         error_array = self.error.__getitem__(item) if self.error is not None else None
#         return Data(input_array, error_array)

#     def __array_finalize__(self, obj):
#         """ Ensures that self.error exists. """
#         if 'error' not in self.__dict__:
#             self.error = None

# class Error(StdDevUncertainty):
#     """ Sub-class of StdDevUncertainty which can propagate error through pow(). """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def propagate(self, operation, other_nddata, result_data, correlation):
#             if operation.__name__ == 'pow':
#                 other_uncert = self._convert_uncertainty(other_nddata.uncertainty)
#                 result = self._propagate_pow(other_uncert, result_data, correlation)
#                 return self.__class__(result, copy=False)
#             else:
#                 super().propagate(operation, other_nddata, result_data, correlation)


class Data(NDDataArray):
    """ Sub-class of NDDataArray for measurements with normal uncertainties. Implements some unary operators which
    aren't available in the parent class, and automatically converts `uncertainty` to the correct format. """

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            args = list(args)
            args[1] = StdDevUncertainty(args[1])
        elif 'uncertainty' in kwargs:
            kwargs['uncertainty'] = StdDevUncertainty(kwargs['uncertainty'])

        super().__init__(*args, **kwargs)

    def __pow__(self, p):
        """ Supports pow() for scalar values of `p`, and propagates uncertainty correctly. """
        if np.ndim(p) > 0:
            raise ValueError("{:s}.__pow__(p) only supports scalar values for `p`".format(type(self).__name__))
        val = self.data.__pow__(p)
        if self.uncertainty is not None:
            err = np.abs(p*self.data.__pow__(p-1)*self.uncertainty.array)
        else:
            err = None
        return type(self)(val, uncertainty=err)

    def __add__(self, other):
        return self.add(other)
    def __sub__(self, other):
        return self.subtract(other)
    def __mul__(self, other):
        return self.multiply(other)
    def __truediv__(self, other):
        return self.divide(other)
    def __matmul__(self, other):
        return self.data.__matmul__(other)
    def __gt__(self, other):
        return self.data.__gt__(other)
    def __ge__(self, other):
        return self.data.__ge__(other)
    def __lt__(self, other):
        return self.data.__lt__(other)
    def __le__(self, other):
        return self.data.__le__(other)
    def __eq__(self, other):
        return self.data.__eq__(other)
    def __ne__(self, other):
        return self.data.__ne__(other)
    

class Spectrum():
    """ Class describing a planet's spectrum.

    Parameters
    ----------
    x : float array
        Wavelength grid of the spectrum.
    y : float array
        Value of the spectrum at each wavelength.
    dy : float array
        Uncertainties on `y`.
    xunit : {'micron'}, optional
        Unit of the wavelength grid.
    yunit : {'albedo'}, optional
        Unit of the spectrum and uncertainties.
    """
    def __init__(self,x,y,dy,xunit='micron',yunit='albedo'):
        self.update(x,y,dy)
        self.xunit,self.yunit = xunit,yunit
    
    def update(self,x,y,dy):
        """ Updates the wavelengths and spectrum values."""
        self.x,self.y,self.dy = x,y,dy

    def plot(self,errorbars=True):
        """ Plots the spectrum.
        
        Parameters
        ----------
        errorbars : bool, optional
            If True, plot errorbars with the spectrum.
        """
        plots.plot_spectrum(self.x,self.y,self.dy if errorbars else None,xunit=self.xunit,yunit=self.yunit)

    def simple_retrieval(self,wl_min,wl_max,window=0.5,threshold=3.):
        """ Uses a simple SNR-based algorithm to determine whether an absorption band is detectable.

        Parameters
        ----------
        wl_min : float
            Minimum wavelength of the absorption band, in same units as self.xunit.
        wl_max : float
            Maximum wavelength of the absorption band, in same units as self.xunit.
        window : float, optional
            Wavelength range used to estimate the continuum, default is 0.1.
        threshold : float, optional
            Sigma detection threshold, default is 3.

        Returns
        -------
        detected : bool
            If True, the species is detected in the spectrum.
        """
        # Identify points in the spectrum within the absorption band
        mask1 = (self.x>=wl_min)&(self.x<=wl_max)
        x1,y1,dy1 = self.x[mask1],self.y[mask1],self.dy[mask1]

        # Identify the continuum data points within window/2. of each edge of the absorption band.
        mask2 = (self.x<wl_min)&(self.x>(wl_min-window/2.))
        mask2 = mask2|(self.x>wl_max)&(self.x<(wl_max+window/2.))
        x2,y2,dy2 = self.x[mask2],self.y[mask2],self.dy[mask2]

        # Fit a line to the continuum and determine the difference between the continuum and absorption points
        params = np.polyfit(x2,y2,deg=1,w=1/dy2)
        diff = np.abs(np.polyval(params,x1)-y1)

        # Get the 80th percentile value of the difference - helps with band edges (e.g. ozone in the UV)
        diff = np.percentile(diff,80)

        # Determine the significance
        sigma = np.mean(dy1)/len(dy1)**0.5
        
        return (diff/sigma)>threshold

class Model(Object):
    """ Class describing a theoretical model for planetary atmospheres. A model translates each unique set of input
    values into either a unique set of output values or a probability distribution for the same.
    """
    def __init__(self,label=None,filename=None,inp=None):
        Object.__init__(self,label=label)
    
    def apply(self,sample):
        """ Applies the model to a sample of planets.

        Parameters
        ----------
        sample : Table
            Table containing the simulated planets.
        """
        # Create a mask identifying which planets the model should be applied to
        mask = True
        for key,val in self.subset.items():
            if np.size(val) == 2:
                mask = mask & ((sample[key]>val[0])&(sample[key]<val[1]))
            else:
                mask = mask & (sample[key]==val)
        
class Forest(Object):
    """ Class describing a set of random forest models for atmospheric retrieval as trained with the HELA code. A different
    Forest should be trained for each atmosphere model and observing instrument.

    Parameters
    ----------
    label : str, optional
        Label of the Forest as found under ./Objects/Forests/.
    spectra : Spectrum array, optional
        Initial set of spectra to add to the training set.
    atmospheres : Atmospheres array, optional
        Initial set of atmospheres to add to the training set.
    noise_levels : int, optional
        Initial number of noise levels to compute. Default is 10.
    """

    def __init__(self,label=None,spectra=None,atmospheres=None,noise_levels=10):
        self.x,self.y,self.dy = [],[],[]
        self.species,self.abundances = [],[]
        self.noise_levels,self.models = [],[]
        Object.__init__(self,label=label)
        if spectra is not None:
            self.add_spectra(spectra,atmospheres)
            self.compute_noise_levels(noise_levels)

    def add_spectra(self,spectra,atmospheres,interpolate=False):
        """ Adds a set of simulated spectra to the training set.

        Parameters
        ----------
        spectra : Spectrum array, optional
            Set of spectra to add to the training set.
        atmospheres : Atmospheres array, optional
            Set of atmospheres to add to the training set.
        interpolate : bool, optional
            If True, interpolate the input spectra if their wavelength values don't match.
        """
        for spec,atm in zip(spectra,atmospheres):
            self.add_spectrum(spec,atm)

    def add_spectrum(self,spectrum,atmosphere,interpolate=False):
        """ Adds a new simulated spectrum to the training set. The y-values of the spectrum should be noiseless, but
        the uncertainties should still be calculated.

        Parameters
        ----------
        spectra : Spectrum array, optional
            Spectrum to add to the training set.
        atmospheres : Atmospheres array, optional
            Atmosphere to add to the training set.
        interpolate : bool, optional
            If True, interpolate the input spectrum if its wavelength values don't match.
        """
        # Ensure the x-axis of the spectrum matches the rest of the training set
        if np.size(self.x) > 0 and not np.array_equal(self.x,spectrum.x):
            if interpolate:
                y = np.interp(self.x,spectrum.x,spectrum.y)
                dy = np.interp(self.x,spectrum.x,spectrum.dy)
            else:
                raise ValueError("spectrum wavelengths do not match the rest of the training set")
        else:
            y,dy = spectrum.y,spectrum.dy
            if np.size(self.x) == 0:
                self.x = spectrum.x

        # Check that the atmospheric species match
        if np.size(self.species) == 0:
            self.species = atmosphere.get_species_list()
        elif not np.array_equal(self.species,atmosphere.get_species_list()):
            raise ValueError("atmospheric species do not match the rest of the training set")

        # Add the y-values and uncertainties to the data set
        self.y.append(y)
        self.dy.append(dy)

        # Add the atmospheric species and log-abundances, averaging over the pressure profile if needed
        P,abundances = atmosphere['P'],[]
        for key in self.species:
            abun = atmosphere[key]
            if np.size(abun) > 1:
                abun = np.average(abun,weights=P)
            abundances.append(abun)
        self.abundances.append(list(np.log10(abundances)))

    def compute_noise_levels(self,N=10):
        """ (Re-)computes the noise levels for the random forest models.

        Parameters
        ----------
        N : int, optional
            Number of noise levels to compute.
        """
        # Calculate N evenly spaced percentiles and the median noise for each spectrum
        percentiles = (np.linspace(0,100,N+1)[1:]+np.linspace(0,100,N+1)[:-1])/2.
        med_noise = np.median(self.dy,axis=1)
        self.noise_levels = np.percentile(med_noise,percentiles)
        
    def get_training_testing_sets(self,noise_level,f_train=0.8):
        """ Assembles training and testing sets with a specified noise level from the set of spectra in the Forest.

        Parameters
        ----------
        noise_level : float
            Median level of noise to inject into each spectrum.
        f_train : float, optional
            Fraction of the spectra to put into the training set, placing the rest into the testing set.

        Returns
        -------
        training : float array
            Training set.
        testing : float array
            Testing set.
        """
        # Apply noise to the spectra by modulating each spectrum's simulated uncertainties to match the median noise level
        y,dy = np.array(self.y),np.array(self.dy)
        dy = dy/np.median(dy,axis=1)[:,None]*noise_level
        specs = np.random.normal(y,dy)

        # Combine the spectra and log-abundances into the HELA-compatible format [y0,y1,y2,...,abun0,abun1,abun2]
        abuns = self.abundances
        sets = np.append(specs,abuns,axis=1)

        # Split these into training and testing sets
        idx_split = int(f_train*sets.shape[0])
        return sets[:idx_split],sets[idx_split:]
    
    def train(self,f_train=0.8,num_trees=1000,num_jobs=5):
        """ Train the random forest model for each noise level.
        
        Parameters
        f_train : float, optional
            Fraction of simulated spectra to use for training each model.
        num_trees : int, optional
            Number of trees in the random forest model.
        num_jobs : int, optional
            Number of jobs to run.
        """
        # Create a temporary working directory for HELA
        workdir = ROOT_DIR+'/HELAworkdir_temp/'
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # Create the JSON file in the working directory
        j = {'training_data':'training.npy','testing_data':'testing.npy','metadata':{}}
        j['metadata']['names'] = list(self.species)
        mins,maxs = np.min(self.abundances,axis=0),np.max(self.abundances,axis=0)
        j['metadata']['ranges'] = [[round(mn,1),round(mx,1)] for mn,mx in zip(mins,maxs)]
        j['metadata']['colors'] = ['#F14532']*len(mins) # These don't really matter
        j['metadata']['num_features'] = len(self.y[0])
        json.dump(j,open(workdir+'/dataset.json','w'))

        # Loop through noise levels
        print("Training models...")
        models,r2s = [],{key:[] for key in self.species}
        for i in util.bar(range(len(self.noise_levels))):
            # Get training and testing sets for this noise level and save into the working directory
            training,testing = self.get_training_testing_sets(self.noise_levels[i],f_train=f_train)
            np.save(workdir+'/training.npy',training)
            np.save(workdir+'/testing.npy',testing)

            # Train a model with HELA
            dataset = hela.load_dataset(workdir+'/dataset.json')
            model = hela.train_model(dataset,num_trees,num_jobs,False)
            
            # Test the model with HELA (i.e. calculate r^2 factor for each retrieved species)
            prediction = model.predict(dataset.testing_x)
            for j in range(len(self.species)):
                r2 = hela.metrics.r2_score(dataset.testing_y.T[j],prediction.T[j])
                r2s[self.species[j]].append(r2)

            # Add the model to the list
            models.append(model)
        self.models = models
        self.r2s = r2s

        # Remove the temporary working directory
        shutil.rmtree(workdir)

    def predict(self,spectrum):
        """ Predicts the abundances of a spectrum using a model trained to a dataset of similar noise.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum for which to predict the values.

        Returns
        -------
        abundances : dict
            Dict containing the name and predicted value of each species.
        """
        # Ensure that the spectrum matches this Forest's wavelength range
        if not np.array_equal(self.x,spectrum.x):
            raise ValueError("spectrum wavelengths do not match those of the random forest model")

        # Determine the appropriate noise level for this spectrum and use that model to predict the abundances
        noise_idx = np.argmin(np.abs(self.noise_levels-np.median(spectrum.dy)))
        prediction = self.models[noise_idx].predict([spectrum.y])[0]
        
        return {self.species[i]:prediction[i] for i in range(len(self.species))}

    def autotrain(self,sample,s,bounds,N_sample=1000,**kwargs):
        """ Uses a simulated sample of planets to build a large training set with which to train the Forest.

        Parameters
        ----------
        sample : Table
            Table of simulated planets.
        s : Survey
            Survey object which is used to simulate the planetary spectra.
        bounds : dict
            dict containing the atmospheric parameters to vary and the boundaries to vary them between.
            example: {'O2':[1e-6,0.5],'CO2':[1e-6,3e-3]}
        N_sample : int, optional
            Size of the training/testing set.
        **kwargs
            keyword arguments for Forest.train().
        """
        # Initialize the large training set using members of `d` as templates
        idxes = np.random.choice(range(len(sample)),size=N_sample)
        sample = sample.copy()[idxes]

        # Vary the atmospheric parameters listed in `bounds`
        vals, atm = {}, sample['Atmosphere']
        for key in bounds.keys():
            mn,mx = bounds[key]
            if key in ['P','T']:
                vals[key] = np.random.uniform(mn,mx,size=N_sample)
            else:
                vals[key] = 10**np.random.uniform(np.log10(mn),np.log10(mx),size=N_sample)

        for idx in range(len(atm)):
            for key in bounds.keys():
                sample['Atmosphere'][idx].set_abundance(key,vals[key][idx])

        # Simulate spectra for these planets
        print("Simulating spectra for the training set...")
        s.simulate_spectra(sample,apply_noise=False,**kwargs)

        # Add the training set to this Forest, compute the noise levels, and train the models
        print("Training the models...")
        self.add_spectra(sample['Spectrum'],sample['Atmosphere'])
        self.compute_noise_levels()
        self.train()

    def test(self,sample,species=None):
        """ Tests the accuracy of the RF retrieval over a sample of planets with simulated spectra.

        Parameters
        ----------
        sample : Table
            Table of simulated planets for which spectra have been simulated.
        """
        # Predict the abundances for each planets
        species = self.species if species is None else species
        predicted = {key:[] for key in species}
        actual = {key:[] for key in species}
        for i in util.bar(range(len(sample))):
            pred = self.predict(sample['Spectrum'][i])
            for key in predicted.keys():
                predicted[key] = np.append(predicted[key],pred[key])
                actual[key] = np.append(actual[key],sample['Atmosphere'][i][key])

        # Plot predicted vs actual abundances
        plots.rf_performance(actual,predicted)
        return actual,predicted

class Stopwatch():
    """ This class uses the time module to profile chunks of code. Use the start() and stop() methods to start and
    stop the Stopwatch, and the mark() methods to record a time. stop() and read() will report the total time elapsed
    as well as the time between each step.
    """
    def __init__(self):
        self.clear()
        self.t0 = time.time()
    def clear(self):
        self.t0 = None
        self.t = []
        self.flags = []
    def mark(self,flag=None):
        self.t.append(time.time()-self.t0)
        self.flags.append(str(len(self.t)-1) if flag is None else flag)
    def stop(self,flag=None):
        self.mark('stop' if flag is None else flag)
        self.read()
    def read(self):
        print("{:40s}\t{:20s}\t{:20s}".format("Mark","Time step (s)","Total elapsed time (s)"))
        for i in range(len(self.t)):
            print("({:3d}) {:30s}\t{:20.8f}\t{:20.8f}".format(i,self.flags[i],self.t[i]-(self.t[i-1] if i>0 else 0.),self.t[i]))
        idx = np.argmax(np.array(self.t[1:])-np.array(self.t[:-1]))
        print("\nLongest step: ({0}) {1}".format(idx,self.flags[idx]))
        


    