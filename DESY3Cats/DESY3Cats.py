import numpy as np
import fitsio as fio
import h5py
import pickle as pickle
import yaml
import os
import sys
import time

# from mpi4py import MPI
try:
    from sharedNumpyMemManager import SharedNumpyMemManager as snmm 
    use_snmm = True
except:
    use_snmm = False

if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,

def get_array( array ):
    if use_snmm:
        return snmm.getArray( array )
    else:
        return array

def save_obj( obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj( name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def file_path( params, subdir, name, var=None, var2=None, var3=None, ftype='txt' ):
    """
    Set up a file path, and create the path if it doesn't exist.
    """

    if var is not None:
        name += '_' + var
    if var2 is not None:
        name += '_' + var2
    if var3 is not None:
        name += '_' + var3
    name += '.' + ftype

    fpath = os.path.join(params['output'],params['param_file'][:params['param_file'].index('.')],subdir)

    if os.path.exists(fpath):
        if not params['output_exists']:
            raise IOError('Output directory already exists. Set output_exists to True to use existing output directory at your own peril.')
    else:
        if not os.path.exists(os.path.join(params['output'],params['param_file'][:params['param_file'].index('.')])):
            os.mkdir(os.path.join(params['output'],params['param_file'][:params['param_file'].index('.')]))
        try:
            os.mkdir(fpath)
        except:
            pass
        params['output_exists'] = True

    return os.path.join(fpath,name)


def write_table( params, table, subdir, name, var=None, var2=None, var3=None ):
    """
    Save a text table to file. Table must be a numpy-compatible array.
    """

    fpath = file_path(params,subdir,name,var=var,var2=var2,var3=var3)
    np.savetxt(fpath,table)


def child_testsuite( calc ):

    params, selector, calibrator = calc

    Testsuite( params, selector=selector, calibrator=calibrator, child=True )


def scalar_sum(x,length):
    # catches scalar weights, responses and multiplies by the vector length for the mean
    if np.isscalar(x):
        return x*length
    return np.sum(x)


class SourceParser(object):
    """
    A class to manage the actual reading or downloading of data from external sources. 
    Initiate with a testsuite param dictionary.
    To use later: source_parser.read(...). All the messy details are hidden.
    """

    def __init__( self, params ):
        self.params = params
        self.open()

    def open( self ):
        raise NotImplementedError('Subclass '+self.__class__.__name__+' should have method open().')

    def read( self ):
        raise NotImplementedError('Subclass '+self.__class__.__name__+' should have method read().')

    def close( self ):
        raise NotImplementedError('Subclass '+self.__class__.__name__+' should have method close().')


class H5Source(SourceParser):
    """
    A class to manage the actual reading or downloading of data from HDF5 sources. 
    """

    def __init__( self, params ):

        super(H5Source,self).__init__(params)

        if 'filename' not in self.params.keys():
            raise NameError('Must provide a filename for hdf5 source.')
        if 'table' not in self.params.keys():
            raise NameError('Must specify table name for hdf5 file.')
        if type(self.params['table']) is not list:
            raise TypeError('Table must be provided as a list of names (even a list of one).')

        if 'group' in self.params.keys():

            self.hdf = h5py.File(self.params['filename'], mode = 'r')
            # save all column names
            self.cols = list(self.hdf[self.params['group']][self.params['table'][0]].keys())
            # save length of tables
            self.size = self.hdf[self.params['group']][self.params['table'][0]][self.cols[0]].shape[0]

            # Loop over tables and save convenience information
            for t in self.params['table']:
                keys = list(self.hdf[self.params['group']][t].keys())
                if self.hdf[self.params['group']][t][keys[0]].shape[0] != self.size:
                    raise TypeError('Length of sheared tables in hdf5 file must match length of unsheared table.')

            if len(self.params['table'])>1:
                # save sheared column names
                self.sheared_cols = list(self.hdf[self.params['group']][self.params['table'][1]].keys())
                print(self.sheared_cols)

        else:

            raise NameError('Need group name for hdf5 file.')

        self.close()

    def open( self ):

            self.hdf = h5py.File(self.params['filename'], mode = 'r')

    def read_direct( self, group, table, col):
        # print('READING FROM HDF5 FILE: group = ',group,' table = ',table,' col = ',col)
        self.open() #attempting this
        return self.hdf[group][table][col][:] 
        self.close()
    def read( self, col=None, rows=None, nosheared=False, full_path = None ):

        self.open()

        def add_out( table, rows, col ):
            """
            Extract a portion of a column from the file.
            """

            if rows is not None:
                if hasattr(rows,'__len__'):
                    if len(rows==2):
                        out = self.hdf[self.params['group']][table][col][rows[0]:rows[1]] 
                else:
                    out = self.hdf[self.params['group']][table][col][rows] 
            else:
                out = self.hdf[self.params['group']][table][col][:] 

            return out

        if full_path is not None:
            return self.hdf[full_path][:]

        if col is None:
            raise NameError('Must specify column.')

        out = []
        # For metacal file, loop over tables and return list of 5 unsheared+sheared values for column (or just unsheraed if 'nosheared' is true or there doesn't exist sheared values for this column) 
        # For classic file, get single column.
        # For both metacal and classic files, output is a list of columns (possible of length 1)
        for i,t in enumerate(self.params['table']):
            if i==0:
                if col not in list(self.hdf[self.params['group']][t].keys()):
                    print(self.params['group'],t,col,list(self.hdf[self.params['group']][t].keys()))
                    raise NameError('Col '+col+' not found in hdf5 file.')
            else:
                if nosheared:
                    print('skipping sheared columns for',col)
                    continue
                if col not in self.sheared_cols:
                    print(col,'not in sheared cols')
                    continue
                if col not in list(self.hdf[self.params['group']][t].keys()):
                    print(col,'not in table keys')
                    raise NameError('Col '+col+' not found in sheared table '+t+' of hdf5 file.')

            if rows is not None:
                out.append( add_out(t,rows,col) )
            else:
                out.append( add_out(t,None,col) )

        self.close()

        return out

    def close( self ):

        if hasattr(self,'hdf'):
            self.hdf.close()


class FITSSource(SourceParser):
    """
    A class to manage the actual reading or downloading of data from FITS sources. 
    """

    def __init__( self, params ):

        super(FITSSource,self).__init__(params)

        if 'filename' not in self.params.keys():
            raise NameError('Must provide a filename for fits source.')
        if 'table' not in self.params.keys():
            raise NameError('Must specify table name for fits file.')
        if type(self.params['table']) is not list:
            raise TypeError('Table must be provided as a list of names (even a list of one).')

        self.fits = fio.FITS(self.params['filename'])
        # save all column names
        self.cols = self.fits[self.params['table'][0]][0].dtype.names
        # save length of tables
        self.size = self.fits[self.params['table'][0]].read_header()['NAXIS2']

        # No metacal sheared capability currently

        self.close()

    def open( self ):

            self.fits = fio.FITS(self.params['filename'])

    def read_direct( self, group, table, col):
        # print('READING FROM HDF5 FILE: group = ',group,' table = ',table,' col = ',col)
        self.open() #attempting this
        return self.fits[table][col][:] 
        self.close()

    def read( self, col=None, rows=None, nosheared=False ):

        self.open()

        def add_out( table, rows, col ):
            """
            Extract a portion of a column from the file.
            """

            if rows is not None:
                if hasattr(rows,'__len__'):
                    if len(rows==2):
                        out = self.fits[table][col][rows[0]:rows[1]] 
                else:
                    out = self.fits[table][col][rows] 
            else:
                out = self.fits[table][col][:] 

            return out

        if col is None:
            raise NameError('Must specify column.')

        out = []
        # For metacal file, loop over tables and return list of 5 unsheared+sheared values for column (or just unsheraed if 'nosheared' is true or there doesn't exist sheared values for this column) 
        # For classic file, get single column.
        # For both metacal and classic files, output is a list of columns (possible of length 1)
        for i,t in enumerate(self.params['table']):
            if i==0:
                if col not in self.cols:
                    raise NameError('Col '+col+' not found in fits file.')
            else:
                if nosheared:
                    print('skipping sheared columns for',col)
                    continue
                if col not in self.sheared_cols:
                    print(col,'not in sheared cols')
                    continue
                if col not in self.fits[t][0].dtype.names:
                    print(col,'not in table keys')
                    raise NameError('Col '+col+' not found in sheared table '+t+' of fits file.')

            if rows is not None:
                out.append( add_out(t,rows,col) )
            else:
                out.append( add_out(t,None,col) )

        self.close()

        return out

    def close( self ):

        if hasattr(self,'fits'):
            self.fits.close()

class DESDMSource(SourceParser):
    """
    A class to manage the actual reading or downloading of data from DESDM sources. 
    """

    def __init__( self ):
        raise NotImplementedError('You should write this.')


class LSSTDMSource(SourceParser):
    """
    A class to manage the actual reading or downloading of data from LSSTDM sources. 
    """

    def __init__( self ):
        raise NotImplementedError('You should write this.')


class Selector(object):
    """
    A class to manage masking and selections of the data.
    Initiate with a testsuite object.
    Initiation will parse the 'select_cols' conditions in the yaml file and create a limiting mask 'mask_', ie, an 'or' of the individual unsheared and sheared metacal masks. The individual masks (list of 1 or 5 masks) are 'mask'.
    """

    def __init__( self, params, source, inherit = None ):
        self.params    = params
        self.source    = source
        if inherit is None:
            self.build_limiting_mask()
        else:
            self.mask = inherit.mask
            self.mask_ = inherit.mask_

    def kill_source( self ):
        self.source = None

    def build_source ( self, source ):
        self.source = source

    def build_limiting_mask( self ):
        """
        Build the limiting mask for use in discarding any data that will never be used.
        """

        mask = None

        # Setup mask file cache path.
        mask_file = file_path(self.params,'cache',self.params['name']+'mask',ftype='pickle')
        if self.params['load_cache']:
            # if mask cache exists, read mask from pickle and skip parsing yaml selection conditions.

            if os.path.exists(mask_file):
                mask, mask_ = load_obj(mask_file)
                print('loaded mask cache')

        if mask is None:

            if 'select_path' in self.params:
                print('using select_path for mask')

                if (self.params['select_path'] is None)|(self.params['select_path'].lower() == 'none'):
                    print('None select path - ignoring selection')
                    mask = [np.ones(self.source.size,dtype=bool)]
                    mask_ = np.where(mask[0])[0]

                else:

                    mask = []

                    tmp = np.zeros(self.source.size,dtype=bool)
                    select = self.source.read(full_path=self.params['select_path'])
                    print('destest',self.params['filename'],self.params['select_path'],len(tmp),len(select))
                    tmp[select]=True
                    mask.append( tmp )
                    try:
                        tmp = np.zeros(self.source.size,dtype=bool)
                        select = self.source.read(full_path=self.params['select_path']+'_1p')
                        tmp[select]=True
                        mask.append( tmp )
                        tmp = np.zeros(self.source.size,dtype=bool)
                        select = self.source.read(full_path=self.params['select_path']+'_1m')
                        tmp[select]=True
                        mask.append( tmp )
                        tmp = np.zeros(self.source.size,dtype=bool)
                        select = self.source.read(full_path=self.params['select_path']+'_2p')
                        tmp[select]=True
                        mask.append( tmp )
                        tmp = np.zeros(self.source.size,dtype=bool)
                        select = self.source.read(full_path=self.params['select_path']+'_2m')
                        tmp[select]=True
                        mask.append( tmp )
                    except:
                        print('No sheared select_path ',self.params['select_path'])

                    mask_ = np.zeros(self.source.size, dtype=bool)
                    for imask in mask:
                        mask_ = mask_ | imask
                    mask_ = np.where(mask_)[0]

                    # Cut down masks to the limiting mask
                    # Its important to note that all operations will assume that data has been trimmed to satisfy selector.mask_ from now on
                    for i in range(len(mask)):
                        mask[i] = mask[i][mask_]

                    print('end mask', mask_, mask[0], np.sum(mask_), np.sum(mask[0]))

            else:

                # mask cache doesn't exist, or you chose to ignore it, so masks are built from yaml selection conditions
                # set up 'empty' mask
                mask = [np.ones(self.source.size, dtype=bool)]
                if self.params['cal_type']=='mcal':
                    mask = mask * 5

                # For each of 'select_cols' in yaml file, read in the data and iteratively apply the appropriate mask
                for i,select_col in enumerate(self.params['select_cols']):
                    cols = self.source.read(col=select_col)
                    for j,col in enumerate(cols):
                        mask[j] = mask[j] & eval(self.params['select_exp'][i])

                # Loop over unsheared and sheared mask arrays and build limiting mask
                mask_ = np.zeros(self.source.size, dtype=bool)
                for imask in mask:
                    mask_ = mask_ | imask

                # Cut down masks to the limiting mask
                # Its important to note that all operations will assume that data has been trimmed to satisfy selector.mask_ from now on
                for i in range(len(mask)):
                    mask[i] = mask[i][mask_]
                mask_ = np.where(mask_)[0]

            # save cache of masks to speed up reruns
            save_obj( [mask, mask_], mask_file )

        if use_snmm:
            # print('using snmm')
            self.mask_ = snmm.createArray((len(mask_),), dtype=np.int64)
            snmm.getArray(self.mask_)[:] = mask_[:]
            mask_ = None
        else:
            self.mask_ = mask_

        self.mask = []
        for i in range(len(mask)):
            if use_snmm:
                self.mask.append( snmm.createArray((len(mask[i]),), dtype=np.bool) )
                snmm.getArray(self.mask[i])[:] = mask[i][:]
                mask[i] = None
            else:
                self.mask.append( mask[i] )

    def get_col( self, col, nosheared=False, uncut=False ):
        """
        Wrapper to retrieve a column of data from the source and trim to the limiting mask (mask_)
        """

        # x at this point is the full column
        x = self.source.read(col=col, nosheared=nosheared)
        # print('get_col length',len(x))

        # if col=='zmean_sof':
        #     print('inside get_col')
        #     print(x[0])

        # trim and return
        for i in range(len(x)):
            x[i] = x[i][get_array(self.mask_)]
            # print('get_col length i',len(x[i]))
        if col=='zmean_sof':
            print(x[0])
        if uncut:
            return x

        for i in range(len(x)):
            x[i] = x[i][get_array(self.mask[i])]
            # print('get_col length2 i',len(x[i])
        if col=='zmean_sof':
            print(x[0])
        return x

    def get_masked( self, x, mask ):
        """
        Accept a mask and column(s), apply the mask jointly with selector.mask (mask from yaml selection) and return masked array.
        """

        if mask is None:
            mask = [np.s_[:]]*5

        if type(mask) is not list:
            mask = [ mask ]

        if type(x) is not list:
            if np.isscalar(x):
                return x
            else:
                return x[get_array(self.mask[0])][mask[0]]

        if np.isscalar(x[0]):
            return x

        return [ x_[get_array(self.mask[i])][mask[i]] for i,x_ in enumerate(x) ]

    def get_mask( self, mask=None ):
        """
        Same as get_masked, but only return the mask.
        """

        if mask is None:
            return [ np.where(get_array(self.mask[i]))[0] for i in range(len(self.mask)) ]

        return [ np.where(get_array(self.mask[i]))[0][mask_] for i,mask_ in enumerate(mask) ]

    def get_match( self ):
        """
        Get matching to parent catalog.
        """

        return self.source.read_direct( self.params['group'].replace('catalog','index'), self.params['table'][0], 'match_gold')

    def get_tuple_col( self, col ):
        """
        Force a tuple return of sheared selections of an unsheared quantity (like coadd_object_id).
        """

        # x at this point is the full column
        x = self.source.read(col=col, nosheared=True)

        x = x*5

        # trim and return
        for i in range(len(x)):
            x[i] = x[i][get_array(self.mask_)]

        for i in range(len(x)):
            x[i] = x[i][get_array(self.mask[i])]

        return x



class Calibrator(object):
    """
    A class to manage calculating and returning calibration factors and weights for the catalog.
    Initiate with a testsuite params object.
    When initiated, will read in the shear response (or m), additive corrections (or c), and weights as requested in the yaml file. These are a necessary overhead that will be stored in memory, but truncated to the limiting mask (selector.mask_), so not that bad.
    """

    def __init__( self, params, selector ):

        self.params = params
        self.selector = selector

    def calibrate(self,col,mask=None,return_full_w=False,weight_only=False,return_wRg=False,return_wRgS=False,return_full=False):
        """
        Return the calibration factor and weights, given potentially an ellipticity and selection.
        """

        # Get the weights
        w  = self.selector.get_masked(self.w,mask)
        if return_full_w:
            w_ = w
        else:
            w_ = w[0]
        if weight_only:
            return w_

        if col == self.params['e'][0]:
            Rg = self.selector.get_masked(get_array(self.Rg1),mask)
            c = self.selector.get_masked(self.c1,mask)
        if col == self.params['e'][1]:
            Rg = self.selector.get_masked(get_array(self.Rg2),mask)
            c = self.selector.get_masked(self.c2,mask)

        print('-----',col, self.params['e'])

        if col in self.params['e']:
            ws = [ scalar_sum(wx_,len(Rg)) for i,wx_ in enumerate(w)]
            # Get a selection response
            Rs = self.select_resp(col,mask,w,ws)
            print('Rs',col,np.mean(Rg),Rs)
            R  = Rg + Rs
            if return_wRg:
                Rg1 = self.selector.get_masked(get_array(self.Rg1),mask)
                Rg2 = self.selector.get_masked(get_array(self.Rg2),mask)
                return ((Rg1+Rg2)/2.)*w[0]
            if return_wRgS:
                Rg1 = self.selector.get_masked(get_array(self.Rg1),mask)
                Rg2 = self.selector.get_masked(get_array(self.Rg2),mask)
                Rs1 = self.select_resp(self.params['e'][0],mask,w,ws)
                Rs2 = self.select_resp(self.params['e'][1],mask,w,ws)
                return ((Rg1+Rg2)/2.+(Rs1+Rs2)/2.)*w[0]
            elif return_full:
                return R,c,w_
            else:
                R = np.sum((Rg+Rs)*w[0],)/ws[0]
                return R,c,w_

        else:
            
            return None,None,w_

    def select_resp(self,col,mask,w,ws):
        """
        Return a zero selection response (default).
        """
        return 0.


class NoCalib(Calibrator):
    """
    A class to manage calculating and returning calibration factors and weights for a general catalog without shear corrections.
    """

    def __init__( self, params, selector ):

        super(NoCalib,self).__init__(params,selector)

        self.Rg1 = self.Rg2 = None
        self.c1 = self.c2 = None
        self.w = [1]
        if 'w' in self.params:
            self.w = self.selector.get_col(self.params['w'])


class MetaCalib(Calibrator):
    """
    A class to manage calculating and returning calibration factors and weights for a metacal catalog.
    """

    def __init__( self, params, selector ):

        super(MetaCalib,self).__init__(params,selector)

        self.Rg1 = self.Rg2 = 1.
        if 'Rg' in self.params:
            Rg1 = self.selector.get_col(self.params['Rg'][0],uncut=True)[0]
            Rg2 = self.selector.get_col(self.params['Rg'][1],uncut=True)[0]
            e1  = self.selector.get_col(self.params['e'][0],nosheared=True,uncut=True)[0]
            e2  = self.selector.get_col(self.params['e'][1],nosheared=True,uncut=True)[0]
            if use_snmm:
                self.Rg1 = snmm.createArray((len(Rg1),), dtype=np.float64)
                snmm.getArray(self.Rg1)[:] = Rg1[:]
                Rg1 = None
                self.Rg2 = snmm.createArray((len(Rg2),), dtype=np.float64)
                snmm.getArray(self.Rg2)[:] = Rg2[:]
                Rg2 = None
                self.e1 = snmm.createArray((len(e1),), dtype=np.float64)
                snmm.getArray(self.e1)[:] = e1[:]
                e1 = None
                self.e2 = snmm.createArray((len(e2),), dtype=np.float64)
                snmm.getArray(self.e2)[:] = e2[:]
                e2 = None
            else:
                self.Rg1 = Rg1
                self.Rg2 = Rg2
                self.e1 = e1
                self.e2 = e2
        self.c1 = self.c2 = 0.
        if 'c' in self.params:
            self.c1 = self.selector.get_col(self.params['c'][0],uncut=True)
            self.c2 = self.selector.get_col(self.params['c'][1],uncut=True)
        self.w = [1] * 5
        if 'w' in self.params:
            self.w = self.selector.get_col(self.params['w'],uncut=True)

    def select_resp(self,col,mask,w,ws):
        """
        Get the selection response.
        """

        # if an ellipticity column, calculate and return the selection response and weight
        if col in self.params['e']:
            if mask is not None:
                if len(mask)==1: # exit for non-sheared column selections
                    return 0.
            mask_ = [ get_array(imask) for imask in self.selector.mask ]

        if col == self.params['e'][0]:
            if mask is not None:
                eSp = np.sum((get_array(self.e1)[mask_[1]])[mask[1]]*w[1])
                eSm = np.sum(get_array(self.e1)[mask_[2]][mask[2]]*w[2])
            else:
                eSp = np.sum(get_array(self.e1)[mask_[1]]*w[1])
                eSm = np.sum(get_array(self.e1)[mask_[2]]*w[2])
            Rs = eSp/ws[1] - eSm/ws[2]
        elif col == self.params['e'][1]:
            if mask is not None:
                eSp = np.sum(get_array(self.e2)[mask_[3]][mask[3]]*w[3])
                eSm = np.sum(get_array(self.e2)[mask_[4]][mask[4]]*w[4])
            else:
                eSp = np.sum(get_array(self.e2)[mask_[3]]*w[3])
                eSm = np.sum(get_array(self.e2)[mask_[4]]*w[4])
            Rs = eSp/ws[3] - eSm/ws[4]
        else:
            return 0.

        Rs /= 2.*self.params['dg']
        # print('Rs',Rs,Rs*2.*self.params['dg'] 
        # print('check what dg is used....'

        return Rs


class ClassicCalib(Calibrator):
    """
    A class to manage calculating and returning calibration factors and weights for a metacal catalog.
    """

    def __init__( self, params, selector ):

        super(ClassCalib,self).__init__(params,selector)

        self.Rg1 = self.Rg2 = 1.
        if 'Rg' in self.params:
            self.Rg1 = self.selector.get_col(self.params['Rg'][0])
            self.Rg2 = self.selector.get_col(self.params['Rg'][1])

        self.c1 = self.c2 = 0.
        if 'c' in self.params:
            self.c1 = self.selector.get_col(self.params['c'][0])
            self.c2 = self.selector.get_col(self.params['c'][1])

        self.w  = [1]
        if 'w' in self.params:
            self.w = self.selector.get_col(self.params['w'])



