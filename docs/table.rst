##############################################
The :class:`~bioverse.classes.Table` class
##############################################

Bioverse uses the :class:`~bioverse.classes.Table` class to manage large simulated datasets across the code. Each row in the Table corresponds to a different simulated planet, while each column corresponds to a planetary parameter. Generally, rows correspond to indices while columns correspond to string keys. Some examples for selecting data in a table:

.. code-block:: python

    # Returns semi-major axis for every planet
    table['a']

    # Returns the mass of the tenth planet
    table['M'][9]

    # Returns all parameters for the first 50 planets
    table[:50]

A Table is somewhat similar to a Pandas DataFrame.
Indeed, if Pandas is installed, the Table will be displayed as one.
To export a Table as a Pandas DataFrame, we can use the :meth:`~bioverse.classes.Table.to_pandas` method:

.. code-block:: python

    table.to_pandas()

              d      M_st      R_st      L_st   T_eff_st SpT  ...  N_pl  order         R            P         a           S
    0    13.779  0.891276  0.912031  0.668409  5469.6313   G  ...     1      0  2.279597   680.071360  1.456531    0.315067
    1    17.990  1.368140  1.285004  2.995380  6704.4496   F  ...     1      0  1.603131  4461.320947  5.887965    0.086402
    2    21.648  1.384050  1.296945  3.119100  6741.3827   F  ...     1      0  3.114289    13.869061  0.125901  196.776529
    3    17.565  0.892344  0.912906  0.671218  5472.7466   G  ...     1      0  1.246218   303.569497  0.851064    0.926700
    4     3.563  0.397945  0.478474  0.039754  3729.2254   M  ...     4      0  2.560260   645.564506  1.075260    0.034384
    ..      ...       ...       ...       ...        ...  ..  ...   ...    ...       ...          ...       ...         ...
    517   5.905  0.353798  0.435516  0.026342  3526.6437   M  ...     5      2  2.587812   128.221803  0.351970    0.212635
    518   5.905  0.353798  0.435516  0.026342  3526.6437   M  ...     5      3  0.802393   165.421708  0.417119    0.151400
    519   5.905  0.353798  0.435516  0.026342  3526.6437   M  ...     5      4  5.704183   311.476887  0.636036    0.065115
    520  10.908  0.763220  0.805601  0.388396  5081.1400   K  ...     2      0  1.343379  1518.340450  2.362701    0.069576
    521  10.908  0.763220  0.805601  0.388396  5081.1400   K  ...     2      1  0.513268    30.733272  0.175483   12.612595

    [522 rows x 22 columns]
    
Each planet or stellar property is referred to throughout Bioverse by a unique string key. This formalism allows properties to be easily accessed across the code. The keys are not formally defined anywhere in the code, so creating a new property is as simple as adding it to a Table of planets:

.. code-block:: python
    
    # Assigns a random ocean covering fraction to every planet in the Table
    table['f_ocean'] = np.random.uniform(0, 1, len(t))
    
This new column must have the same length as others in the Table. Some other examples of Table usage:

.. code-block:: python

    # Change the value of `f_ocean` to zero for planets that are not exo-Earth candidates
    EEC = table['EEC'] # boolean array
    table['f_ocean'][~EEC] = 0.
    
    # Calculate planet densities in g/cm3
    table['rho'] = 5.51 * table['M'] / table['R']**3

    # List the definition of all keys in the table (found in legend.dat)
    table.legend()

    # Append one table to another in-place
    table.append(table2, inplace=True)

See the :class:`~bioverse.classes.Table` documentation for a full list of its methods.

List of properties
******************
The following table lists all keys currently used in Bioverse and the properties they correspond to:

.. csv-table::
    :header: "Key", "Data type", "Description"
    :file: ../bioverse/legend.dat
    :widths: auto





