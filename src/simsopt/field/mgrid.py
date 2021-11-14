import numpy as np
import netCDF4 as nc
import sys

### File I/O ###


class MGRID():

    
    def __init__(self,fname='temp',binary=False,
                    nr=51,nz=51,nphi=24,nfp=2,
                    rmin=0.20,rmax=0.40,zmin=-0.10,zmax=0.10,
                    nextcur=0):

        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.nfp = nfp

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.n_ext_cur  = 0
        self.cur_labels = []

        self.br_arr = [] 
        self.bz_arr = [] 
        self.bp_arr = [] 

        # this option was disabled upon request (can be easily added back)
#        if (binary):
#            self.read_binary(fname,_debug=False)

        print('Initialized mgrid file: (nr,nphi,nz,nfp) = ({}, {}, {}, {})'.format(nr,nphi,nz,nfp))

    # takes 3D vector field (N,3) as well as coil group label (up to 30 char)
    def add_field(self,B,name='default'):

        
        # structure Bfield data into arrays (phi,z,r) arrays
        bx,by,bz = B.T
        shape = (self.nr, self.nz, self.nphi) 
        bx_arr = np.reshape(bx  , shape).T
        by_arr = np.reshape(by  , shape).T
        bz_arr = np.reshape(bz  , shape).T

        # pass from cartesian to cylindrical coordinates
        phi = self.export_phi()
        cos = np.cos(phi)[:,np.newaxis,np.newaxis]
        sin = np.sin(phi)[:,np.newaxis,np.newaxis]
        br_arr =  cos*bx_arr + sin*by_arr
        bp_arr = -sin*bx_arr + cos*by_arr

        self.br_arr.append( br_arr )
        self.bz_arr.append( bz_arr )
        self.bp_arr.append( bp_arr )

        # add coil label
        if (name == 'default'):
            label = pad_string('magnet_%i' % self.n_ext_cur)
        else:
            label = pad_string(name)
        self.cur_labels.append(label)
        self.n_ext_cur = self.n_ext_cur + 1


    def read_netCDF(self,fname):
        
        print(' reading:', fname)
        # overwrites existing class information

        f = nc.Dataset(fname, mode='r')
        self.nr   = int( get(f, 'ir')  )
        self.nz   = int( get(f, 'jz')  )
        self.nphi = int( get(f, 'kp')  )
        self.nfp  = int( get(f, 'nfp') )
        self.n_ext_cur  = int( get(f, 'nextcur') )

        self.rmin = float( get(f, 'rmin') )
        self.rmax = float( get(f, 'rmax') )
        self.zmin = float( get(f, 'zmin') )
        self.zmax = float( get(f, 'zmax') )

        self.cur_labels = get(f, 'coil_group')

        # implement read fields in for loop
        #self.br_arr = [] 
        #self.bz_arr = [] 
        #self.bp_arr = [] 

        print(' overwriting  mgrid coordinates: (nr,nphi,nz,nfp) = ({}, {}, {}, {})'.format(self.nr,self.nphi,self.nz,self.nfp))


    def write(self,fout):

        ### Write
        print('Writing mgrid file')
        ds = nc.Dataset(fout, 'w', format='NETCDF4')
        
        # set dimensions
        ds.createDimension('stringsize', 30)
        ds.createDimension('dim_00001', 1)
        ds.createDimension('external_coil_groups', self.n_ext_cur)
        ds.createDimension('external_coils', self.n_ext_cur)
        ds.createDimension('rad', self.nr)
        ds.createDimension('zee', self.nz)
        ds.createDimension('phi', self.nphi)
        
        # declare variables
        var_ir = ds.createVariable('ir', 'i4')
        var_jz = ds.createVariable('jz', 'i4')
        var_kp = ds.createVariable('kp', 'i4')
        var_nfp     = ds.createVariable('nfp', 'i4')
        var_nextcur = ds.createVariable('nextcur', 'i4')
        
        var_rmin = ds.createVariable('rmin','f8')
        var_zmin = ds.createVariable('zmin','f8')
        var_rmax = ds.createVariable('rmax','f8')
        var_zmax = ds.createVariable('zmax','f8')
        
        var_coil_group = ds.createVariable('coil_group', 'c', ('external_coil_groups', 'stringsize',))
        var_mgrid_mode = ds.createVariable('mgrid_mode', 'c', ('dim_00001',))
        var_raw_coil_cur = ds.createVariable('raw_coil_cur', 'f8', ('external_coils',))
       
        
        # assign values
        var_ir[:] = self.nr
        var_jz[:] = self.nz
        var_kp[:] = self.nphi
        var_nfp[:] = self.nfp
        var_nextcur[:] = self.n_ext_cur
        
        var_rmin[:] = self.rmin
        var_zmin[:] = self.zmin
        var_rmax[:] = self.rmax
        var_zmax[:] = self.zmax
        
        var_coil_group[:] = self.cur_labels
        var_mgrid_mode[:] = 'R' # R - Raw, S - scaled, N - none (old version)
        var_raw_coil_cur[:] = np.ones(self.n_ext_cur)
        
        
        
        # go to rectangular arrays
        #cos_arr = np.cos(phi)[np.newaxis,np.newaxis,:]
        #sin_arr = np.sin(phi)[np.newaxis,np.newaxis,:]
        #
        #bx = np.ravel( br_arr*cos_arr - bphi_arr*sin_arr )
        #by = np.ravel( br_arr*sin_arr + bphi_arr*cos_arr )
        
        # transpose because binary is read (r,z,phi)
        # but netCDF is written (phi,zee,rad)

        # add fields
        for j in np.arange(self.n_ext_cur):
            
            tag = '_%.3i' % (j+1)
            var_br_001 = ds.createVariable('br'+tag, 'f8', ('phi','zee','rad') )
            var_bp_001 = ds.createVariable('bp'+tag, 'f8', ('phi','zee','rad') )
            var_bz_001 = ds.createVariable('bz'+tag, 'f8', ('phi','zee','rad') )

            var_br_001[:,:,:] = self.br_arr[j]
            var_bz_001[:,:,:] = self.bz_arr[j]
            var_bp_001[:,:,:] = self.bp_arr[j]
        
        ds.close()

        print('  Wrote to file:', fout)

    
    def init_targets(self):

        raxis = np.linspace(self.rmin,self.rmax,self.nr)
        zaxis = np.linspace(self.zmin,self.zmax,self.nz)
        
        phi   = np.linspace(0,2*np.pi/self.nfp,self.nphi)
        
        xyz = []
        for r in raxis:
            for z in zaxis:
                for p in phi:
                    x = r*np.cos(p)
                    y = r*np.sin(p)
                    xyz.append([x,y,z])
        return np.array(xyz)


    def export_phi(self):
        phi   = np.linspace(0,2*np.pi/self.nfp,self.nphi)
        return phi

    def export_grid_spacing(self):
        return self.nr, self.nz, self.nphi


def pad_string(string):
    return '{:^30}'.format(string).replace(' ','_')


# function for reading netCDF files
def get(f,key):
    return f.variables[key][:]