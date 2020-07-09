import numpy as np
import pandas as pd
import vaex

from pymove.core import MoveDataFrameAbstractModel
from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import (
    DATE,
    DATETIME,
    DAY,
    DAY_PERIODS,
    DIST_PREV_TO_NEXT,
    DIST_TO_NEXT,
    DIST_TO_PREV,
    HOUR,
    HOUR_COS,
    HOUR_SIN,
    LATITUDE,
    LONGITUDE,
    MOVE,
    PERIOD,
    SITUATION,
    SPEED_PREV_TO_NEXT,
    SPEED_TO_NEXT,
    SPEED_TO_PREV,
    STOP,
    TID,
    TIME_PREV_TO_NEXT,
    TIME_TO_NEXT,
    TIME_TO_PREV,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
    TYPE_VAEX,
    UID,
    WEEK_DAYS,
    WEEK_END,
)
from pymove.utils.mem import begin_operation, end_operation


class VaexMoveDataFrame(vaex.dataframe.DataFrame, MoveDataFrameAbstractModel):
    def __init__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    ):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' columns,
        and renames it with the PyMove lib standard. After starts the
        attributes of the class.

        - self._data : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self.last_operation : Represents the last operation performed.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame.
            Input trajectory data.
        latitude : str, optional, default 'lat'.
            Represents column name latitude.
        longitude : str, optional, default 'lon'.
            Represents column name longitude.
        datetime : str, optional, default 'datetime'.
            Represents column name datetime.
        traj_id : str, optional, default 'id'.
            Represents column name trajectory id.

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        if isinstance(data, dict):
            data = vaex.from_dict(data)
        elif (
            (isinstance(data, list) or isinstance(data, np.ndarray))
            and len(data) >= 4
        ):
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)

            list_data = dict(zip(zip_list, np.transpose(data)))
            data = vaex.from_arrays(**list_data)
        elif(isinstance(data, pd.DataFrame)):
            data = vaex.from_pandas(data)

        data.rename(latitude, LATITUDE)
        data.rename(longitude, LONGITUDE)
        data.rename(datetime, DATETIME)
        if traj_id in data:
            data.rename(traj_id, TRAJ_ID)

        tdf = data

        if MoveDataFrame.has_columns(tdf):
            MoveDataFrame.validate_move_data_frame(tdf)
            self._data = tdf
            self._type = TYPE_VAEX
            self.last_operation = None
        else:
            raise AttributeError(
                'Couldn\'t instantiate MoveDataFrame because data has missing columns.'
            )

    @property
    def lat(self):
        """Checks for the 'lat' column and returns its value."""
        if LATITUDE not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LATITUDE
            )
        return self._data[LATITUDE]

    @property
    def lng(self):
        """Checks for the 'lon' column and returns its value."""
        if LONGITUDE not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LONGITUDE
            )
        return self._data[LONGITUDE]

    @property
    def datetime(self):
        """Checks for the 'datetime' column and returns its value."""
        if DATETIME not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % DATETIME
            )
        return self._data[DATETIME]

    def __setitem__(self, attr, value):
        """Modifies and item in this object."""
        self.__dict__['_data'][attr] = value

    def __getitem__(self, name):
        """Retrieves and item from this object."""
        try:
            item = self.__dict__['_data'][name]
            if (
                isinstance(item, vaex.dataframe.DataFrame)
                and MoveDataFrame.has_columns(item)
            ):
                return VaexMoveDataFrame(item)
            return item
        except Exception as e:
            raise e

    @property
    def loc(self):
        """Access a group of rows and columns by label(srs) or a boolean array."""
        raise NotImplementedError('To be implemented')

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position."""
        raise NotImplementedError('To be implemented')

    @property
    def at(self):
        """Access a single value for a row/column label pair."""
        raise NotImplementedError('To be implemented')

    @property
    def values(self, col):
        """Return a Numpy representation of the DataFrame."""
        raise NotImplementedError('To be implemented')

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        return self._data.get_column_names()

    @property
    def index(self):
        """The row labels of the DataFrame."""
        raise NotImplementedError('To be implemented')

    @property
    def dtypes(self):
        '''
        Return the dtypes in the DataFrame. This returns a Series with
        the data type of each column. The result'srs index is the original
        DataFrame'srs columns. Columns with mixed types are stored with the
        object dtype. See the User Guide for more.

        Returns
        -------
        pandas.Series
            The data type of each column.
        '''

        operation = begin_operation('dtypes')
        dtypes_ = self._data.dtypes
        self.last_operation = end_operation(operation)
        return dtypes_

    @property
    def shape(self):
        """Return a tuple representing the dimensionality of the DataFrame."""
        operation = begin_operation('shape')
        shape_ = self._data.shape
        self.last_operation = end_operation(operation)
        return shape_

    def rename(self):
        """Alter axes labels.."""
        raise NotImplementedError('To be implemented')

    def len(self):
        """
        Returns the length/row numbers in trajectory data.

        Returns
        -------
        int
            Represents the trajectory data length.

        """
        operation = begin_operation('len')
        len_ = self._data.shape[0]
        self.last_operation = end_operation(operation)

        return len_

    def unique(self, values):
        """
        Return unique values of Series object. Uniques are returned
        in order of appearance.
        Hash table-based unique, therefore does NOT sort.

        Parameters
        ----------
        values : array, list, series or dataframe.
            The set of values to identify unique occurrences.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html

        """
        operation = begin_operation('unique')
        unique_ = self._data.unique(values)
        self.last_operation = end_operation(operation)

        return unique_

    def head(self, n=5):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, optional, default 5
            Number of rows to select.
        npartitions : int, optional, default 1.
            Represents the number partitions.
        compute : bool, optional, default True.
            ?

        Returns
        -------
        same type as caller
            The first n rows of the caller object.

        """
        operation = begin_operation('head')
        head_ = self._data.head(n)
        self.last_operation = end_operation(operation)

        return head_

    def tail(self, n=5):
        """
        Return the last n rows.

        This function returns the last n rows for the object based on position.
        It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, optional, default 5
            Number of rows to select.
        npartitions : int, optional, default 1.
            Represents the number partitions.
        compute : bool, optional, default True.
            ?

        Returns
        -------
        same type as caller
            The last n rows of the caller object.

        """
        operation = begin_operation('tail')
        tail_ = self._data.tail(n)
        self.last_operation = end_operation(operation)

        return tail_

    def get_users_number(self):
        """Check and return number of users in trajectory data."""
        operation = begin_operation('get_users_numbers')

        if UID in self._data:
            number_ = self._data[UID].nunique()
        else:
            number_ = 1
        self.last_operation = end_operation(operation)

        return number_

    def to_numpy(self):
        """Converts trajectory data to numpy array format."""
        raise NotImplementedError('To be implemented')

    def to_dict(self,
                column_names=None,
                selection=None,
                strings=True,
                virtual=True,
                parallel=True,
                chunk_size=None,
                array_type=None):
        """Converts trajectory data to dict format."""

        operation = begin_operation('to_dict')
        dict_ = self._data.to_dict(column_names,
                                   selection,
                                   strings,
                                   virtual,
                                   parallel,
                                   chunk_size,
                                   array_type)
        self.last_operation = end_operation(operation)

        return dict_

    def to_grid(self):
        """Converts trajectory data to grid format."""
        raise NotImplementedError('To be implemented')

    def to_data_frame(self,
                      column_names=None,
                      selection=None,
                      strings=True,
                      virtual=True,
                      index_name=None,
                      parallel=True,
                      chunk_size=None):
        """
        Converts trajectory data to DataFrame format.

        Returns
        -------
        dask.dataframe.DataFrame
            Represents the trajectory in DataFrame format.

        """

        operation = begin_operation('to_data_frame')
        data_ = self._data.to_pandas_df(column_names,
                                        selection,
                                        strings,
                                        virtual,
                                        index_name,
                                        parallel,
                                        chunk_size)
        self.last_operation = end_operation(operation)

        return data_

    def info(self):
        """Print a concise summary of a DataFrame."""
        raise NotImplementedError('To be implemented')

    def describe(self, strings=True, virtual=True, selection=None):
        """Generate descriptive statistics."""
        operation = begin_operation('describe')
        describe_ = self._data.describe(strings, virtual, selection)
        self.last_operation = end_operation(operation)
        return describe_

    def memory_usage(self):
        """Return the memory usage of each column in bytes."""
        raise NotImplementedError('To be implemented')

    def copy(self):
        """Make a copy of this object’srs indices and data."""
        operation = begin_operation('copy')
        copy_ = VaexMoveDataFrame(self._data.copy())
        self.last_operation = end_operation(operation)
        return copy_

    def generate_tid_based_on_id_datetime(self):
        """Create or update trajectory id based on id e datetime."""
        raise NotImplementedError('To be implemented')

    def generate_date_features(self):
        """Create or update date feature."""
        raise NotImplementedError('To be implemented')

    def generate_hour_features(self):
        """Create or update hour feature."""
        raise NotImplementedError('To be implemented')

    def generate_day_of_the_week_features(self):
        """Create or update a feature day of the week from datatime."""
        raise NotImplementedError('To be implemented')

    def generate_weekend_features(self):
        """Create or update the feature weekend to the dataframe."""
        raise NotImplementedError('To be implemented')

    def generate_time_of_day_features(self):
        """Create a feature time of day or period from datatime."""
        raise NotImplementedError('To be implemented')

    def generate_datetime_in_format_cyclical(self):
        """Create or update column with cyclical datetime feature."""
        raise NotImplementedError('To be implemented')

    def generate_dist_time_speed_features(self):
        """Creates features of distance, time and speed between points."""
        raise NotImplementedError('To be implemented')

    def generate_dist_features(self):
        """Create the three distance in meters to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_time_features(self):
        """Create the three time in seconds to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_speed_features(self):
        """Create the three speed in meters by seconds to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_move_and_stop_by_radius(self):
        """Create or update column with move and stop points by radius."""
        raise NotImplementedError('To be implemented')

    def time_interval(self):
        """Get time difference between max and min datetime in trajectory."""

        operation = begin_operation('time_interval')
        time_diff = self._data[DATETIME].max() - self._data[DATETIME].min()
        self.last_operation = end_operation(operation)

        return time_diff

    def get_bbox(self):
        """Creates the bounding box of the trajectories."""
        operation = begin_operation('get_bbox')

        try:
            bbox_ = (
                self._data[LATITUDE].min().item(),
                self._data[LONGITUDE].min().item(),
                self._data[LATITUDE].max().item(),
                self._data[LONGITUDE].max().item(),
            )

            self.last_operation = end_operation(operation)

            return bbox_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def plot_all_features(self):
        """Generate a visualization for each column that type is equal dtype."""
        raise NotImplementedError('To be implemented')

    def plot_trajs(self):
        """Generate a visualization that show trajectories."""
        raise NotImplementedError('To be implemented')

    def plot_traj_id(self):
        """Generate a visualization for a trajectory with the specified tid."""
        raise NotImplementedError('To be implemented')

    def show_trajectories_info(self):
        """Show dataset information from dataframe."""
        raise NotImplementedError('To be implemented')

    def min(self,
            expression,
            binby=[],
            limits=None,
            shape=128,
            selection=False,
            delay=False,
            progress=None,
            edges=False,
            array_type=None):
        """
        Return the minimum of the values for the requested expression

        """

        operation = begin_operation('min')
        _min = self._data.min(expression,
                              binby,
                              limits,
                              shape,
                              selection,
                              delay,
                              progress,
                              edges,
                              array_type)
        self.last_operation = end_operation(operation)

        return _min

    def max(self,
            expression,
            binby=[],
            limits=None,
            shape=128,
            selection=False,
            delay=False,
            progress=None):
        """
        Return the maximum of the values for the requested expression

        """

        operation = begin_operation('max')
        _min = self._data.max(expression,
                              binby,
                              limits,
                              shape,
                              selection,
                              delay,
                              progress)
        self.last_operation = end_operation(operation)

        return _min

    def count(self,
              expression=None,
              binby=[],
              limits=None,
              shape=128,
              selection=False,
              delay=False,
              edges=False,
              progress=None,
              array_type=None):
        """Counts the non-NA cells for each column or row."""

        operation = begin_operation('count')
        _count = self._data.count(expression,
                                  binby,
                                  limits,
                                  shape,
                                  selection,
                                  delay,
                                  edges,
                                  progress,
                                  array_type)
        self.last_operation = end_operation(operation)

        return _count

    def groupby(self, by=None, agg=None):
        """
        Groups
        """

        operation = begin_operation('groupby')
        _groupby = self._data.groupby(by, agg)
        self.last_operation = end_operation(operation)

        return _groupby

    def plot(self):
        """Plot the data of the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def select_dtypes(self):
        """Returns a subset of the columns based on the column dtypes."""
        raise NotImplementedError('To be implemented')

    def astype(self):
        """Casts a dask object to a specified dtype."""
        raise NotImplementedError('To be implemented')

    def sort_values(self):
        """Sorts the values of the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def reset_index(self):
        """Resets the dask DataFrame'srs index, and use the default one."""
        raise NotImplementedError('To be implemented')

    def set_index(self):
        """Set of row labels using one or more existing columns or arrays."""
        raise NotImplementedError('To be implemented')

    def drop(self):
        """Drops specified rows or columns of the dask Dataframe."""
        raise NotImplementedError('To be implemented')

    def duplicated(self):
        """Returns boolean Series denoting duplicate rows."""
        raise NotImplementedError('To be implemented')

    def drop_duplicates(self):
        """Removes duplicated rows from the data."""
        raise NotImplementedError('To be implemented')

    def shift(self):
        """Shifts by desired number of periods with an optional time freq."""
        raise NotImplementedError('To be implemented')

    def all(self):
        """Indicates if all elements are True, potentially over an axis."""
        raise NotImplementedError('To be implemented')

    def any(self):
        """Indicates if any element is True, potentially over an axis."""
        raise NotImplementedError('To be implemented')

    def isna(self):
        """Detect missing values."""
        raise NotImplementedError('To be implemented')

    def fillna(self):
        """Fills missing data in the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def dropna(self):
        """Removes missing data from dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def sample(self):
        """Samples data from the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def isin(self):
        """Determines whether each element is contained in values."""
        raise NotImplementedError('To be implemented')

    def append(self):
        """Append rows of other to the end of caller, returning a new object."""
        raise NotImplementedError('To be implemented')

    def join(self):
        """Join columns of another DataFrame."""
        raise NotImplementedError('To be implemented')

    def merge(self):
        """Merge columns of another DataFrame."""
        raise NotImplementedError('To be implemented')

    def nunique(self):
        """Count distinct observations over requested axis."""
        raise NotImplementedError('To be implemented')

    def write_file(self):
        """Write trajectory data to a new file."""
        raise NotImplementedError('To be implemented')

    def to_csv(self):
        """Write object to a comma-separated values (csv) file."""
        raise NotImplementedError('To be implemented')

    def convert_to(self, new_type):
        """
        Convert an object from one type to another specified by the user.

        Parameters
        ----------
        new_type: 'pandas' or 'dask'
            The type for which the object will be converted.

        Returns
        -------
        A subclass of MoveDataFrameAbstractModel
            The converted object.

        """

        raise NotImplementedError('To be implemented')

    def get_type(self):
        """
        Returns the type of the object.

        Returns
        -------
        str
            A string representing the type of the object.

        """

        return self._type
