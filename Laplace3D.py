__all__ = ['Laplace3D']

import numpy as np
import numpy.linalg as alg
from mesh3D import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class Laplace3D:

    def __init__(self,h,H1,sigma, save = True, **kwgs):        
        self.__save = save
        self.__sigma = sigma
        self.__H1 = H1

        if 'f' in kwgs.keys():
            self.__f = kwgs['f']
            self.__h = h
            self.__gridX, self.__gridY,self.__gridZ, self.__lim_fun, self.__features, self.__mesh,general = mesh3D(h,H1,kwgs['f'])
        else:
            self.__mesh = np.load(kwgs['path'],allow_pickle = True)['data']
            self.__gridX = np.load(kwgs['path'],allow_pickle = True)['gridX']
            self.__gridY = np.load(kwgs['path'],allow_pickle = True)['gridY']
            self.__gridZ = np.load(kwgs['path'],allow_pickle = True)['gridZ']
            self.__gridV = np.load(kwgs['path'],allow_pickle = True)['gridV']
            self.__gridJx = np.load(kwgs['path'],allow_pickle = True)['gridJx']
            self.__gridJy = np.load(kwgs['path'],allow_pickle = True)['gridJy']
            self.__gridJz = np.load(kwgs['path'],allow_pickle = True)['gridJz']
            self.__lim_fun = np.load(kwgs['path'],allow_pickle = True)['lim_fun']
            self.__features = np.load(kwgs['path'],allow_pickle = True)['features']
            general = np.load(kwgs['path'],allow_pickle = True)['general_data']
            self.__h = general[-1]
            
        
        self.__r0 = general[0]
        self.__r1 = general[1]
        
        self.__idx_V = np.where(self.__features == 'potential')[0]
        self.__idx_Jx = np.where(self.__features == 'Jx')[0]
        self.__idx_Jy = np.where(self.__features == 'Jy')[0]
        self.__idx_Jz = np.where(self.__features == 'Jz')[0]
        self.__idx_up = np.where(self.__features == 'up_neighbor')[0]
        self.__idx_down = np.where(self.__features == 'down_neighbor')[0]
        self.__idx_left = np.where(self.__features == 'left_neighbor')[0]
        self.__idx_right = np.where(self.__features == 'right_neighbor')[0]
        self.__idx_front = np.where(self.__features == 'front_neighbor')[0]
        self.__idx_back = np.where(self.__features == 'back_neighbor')[0]
            
            

    def compile_pot(self):
        
        def set_address(element,mesh, i,idx_1,idx_0):
            ''' Function that looks for the idx of the neighbors of a certain element
                IN: Element (array),array of Ids, idx element, Id neighbor 1,Id neighbor 0
                OUT: idx neighbor 1, idx neighbor 0, denominator '''
            if element[idx_1] == '-1' and element[idx_0] == '-1':
                return i,i,1
            
            elif element[idx_1] != '-1' and element[idx_0] == '-1':
                return np.where(mesh==element[idx_1])[0][0], i,1
            
            elif element[idx_1] == '-1' and element[idx_0] != '-1':
                return i,np.where(mesh==element[idx_0])[0][0],1
            
            return np.where(mesh==element[idx_1])[0][0],np.where(mesh==element[idx_0])[0][0],2
        
        mesh = self.get_mesh()
        h = self.get_h()
        matrix = np.zeros((mesh.shape[0],mesh.shape[0]), dtype = 'float')
        charge_vector = np.zeros(mesh.shape[0], dtype = 'float')

        idx_up = self.get_idx_up()
        idx_down = self.get_idx_down()
        idx_left = self.get_idx_left()
        idx_right = self.get_idx_right()
        idx_front = self.get_idx_front()
        idx_back = self.get_idx_back()
        idx_V = self.get_idx_V()

        for i,element in enumerate(mesh):
            
            if element[3] == 0 or element[3] == self.get_H1(): #top or bottom
                charge_vector[i] = element[idx_V]
                matrix[i][i] = h**2
                count[0] = count[0]+1
                
            elif '-1' in element[idx_front:]: #sides
                ## Obtaining normal vector
                count[1] = count[1]+1
                
                n_c = np.array([1,-1*(self.get_f()(element[3])[1])])
                #n_c = np.array([1,0.5])
                #n_c =  n_c/alg.norm(n_c,2)
                
                cos = np.cos(np.arctan(element[2]/element[1]))
                sin = np.sin(np.arctan(element[2]/element[1]))
                n = np.array([n_c[0]*cos,n_c[0]*sin,n_c[1]])
                
                ## Computing Grad
                list_idx = [[idx_front,idx_back],
                            [idx_right,idx_left],
                            [idx_up,idx_down]]       
                    
                for g, [idx_1, idx_0] in enumerate(list_idx):

                    
                    neighbor_1, neighbor_0, den = set_address(element,mesh[:,0],i,idx_1,idx_0)
                 
                    matrix[i][neighbor_1] = matrix[i][neighbor_1] + (h*n[g]/den)
                    matrix[i][neighbor_0] = matrix[i][neighbor_0] - (h*n[g]/den)
         
            else:
                
                up_neighbor = np.where(mesh[:,0]==element[idx_up])[0][0]
                down_neighbor = np.where(mesh[:,0]==element[idx_down])[0][0]
                left_neighbor = np.where(mesh[:,0]==element[idx_left])[0][0]
                right_neighbor = np.where(mesh[:,0]==element[idx_right])[0][0]
                front_neighbor = np.where(mesh[:,0]==element[idx_front])[0][0]
                back_neighbor = np.where(mesh[:,0]==element[idx_back])[0][0]

                matrix[i][up_neighbor] = 1
                matrix[i][down_neighbor] = 1
                matrix[i][left_neighbor] = 1
                matrix[i][right_neighbor] = 1
                matrix[i][front_neighbor] = 1
                matrix[i][back_neighbor] = 1
                matrix[i][i] = -6
       
        inv_matrix = alg.inv(matrix)
        potentials = (inv_matrix@charge_vector)*(h**2)
        mesh[:,idx_V] = potentials
         
        self.set_mesh(mesh)
        self.compile_grad()
        #self.plot()

    def compile_grad(self):

        def search_V(mesh,element,idx, idx_V =4):
            loc = np.where(mesh[:,0]==element[idx])[0][0]
            return mesh[loc][idx_V]

        mesh = self.get_mesh()

        idx_up = self.get_idx_up()
        idx_down = self.get_idx_down()
        idx_left = self.get_idx_left()
        idx_right = self.get_idx_right()
        idx_front = self.get_idx_front()
        idx_back = self.get_idx_back()

        idx_Jx =  self.get_idx_Jx()
        idx_Jy = self.get_idx_Jy()
        idx_Jz = self.get_idx_Jz()
        idx_V = self.get_idx_V()
        h = self.get_h()
        sigma = self.get_sigma()

        for i,element in enumerate(mesh):

            list_idx = [idx_front,idx_back,idx_right,idx_left,idx_up,idx_down]
                
            list_temp_V = []
            CoefDiff = []
            
            for idx in list_idx:
                if '-1' == element[idx]:
                    list_temp_V.append(element[idx_V])
                    CoefDiff.append(1)                    
                else:
                    list_temp_V.append(search_V(mesh,element,idx,idx_V))
                    CoefDiff.append(2)

            mesh[i][idx_Jy] = -1*sigma*(list_temp_V[3] - list_temp_V[2])/(h*int((CoefDiff[3]*CoefDiff[2])**.5))
            mesh[i][idx_Jx] = -1*sigma*(list_temp_V[0] - list_temp_V[1])/(h*int((CoefDiff[0]*CoefDiff[1])**.5))
            mesh[i][idx_Jz] = -1*sigma*((list_temp_V[4] - list_temp_V[5]))/(h*int((CoefDiff[4]*CoefDiff[5])**.5))

        self.set_mesh(mesh)

        new_mesh = self.get_mesh()
        data = np.hstack((np.reshape(mesh[:,0],(new_mesh.shape[0],1)),new_mesh[:,idx_V:idx_Jz+1]))
        self.__gridV, self.__gridJx,self.__gridJy, self.__gridJz = self.build_grid(data, self.get_gridX().shape, np.nan,0,0,0)

        self.__I = self.calc_I()
        print(self.get_I())

        #compute current
        
        name = 'mesh'
        if self.__save: self.save_npz(name)
                
        self.plot(name)

    def calc_I(self):
        ''' calculates I over all cross sections '''

        Jz = self.get_gridJz()
        
        h = self.get_h()
        
        #dI = Jz*(h**2)
        I = []
        for section in range(Jz.shape[2]):
            #c = dI[:,:,section]
            c = Jz[:,:,section]
            I.append((h**2)*np.sum(c[~np.isnan(c)]))

        return I
            
    def build_grid(self,array_values, shape,*args):
        ''' [array,tuple] -> list(array(shape(tuple))

            where array_values:
                    -> [[id_0,value_0_0, value_0_1,...,value_0_n],
                       [id_1,value_1_0, value_1_1,...,value_1_n],
                                        ...
                       [id_m,value_m_0, value_m_1,...,value_m_n]

            args -> value of empty cells. If theargument is not passed,
            a list of zeros is considered

            shape -> shape of grid

            args2 -> list of the grid's name to save them '''

        
        if args == ():
            args = np.zeros(array_values.shape[1]-1)


        list_grids = []
        for i, non_value in enumerate(args):
            aux = np.empty(shape)
            aux[:] = non_value
            list_grids.append(np.copy(aux))
            
        for element in array_values:
            i = int(element[0].split('.')[0])
            j = int(element[0].split('.')[1])
            k = int(element[0].split('.')[2])
            
            for e in range(len(list_grids)):
                list_grids[e][i][j][k] = element[e+1]    

        return list_grids       

    def plot(self, *args):
        
        X = self.get_gridX()
        Y  = self.get_gridY()
        Z = self.get_gridZ()
        Jx =  self.get_gridJx()
        Jy = self.get_gridJy()
        Jz = self.get_gridJz()
        V = self.get_gridV()
        h = self.get_h()
        if args == ():
            name = 'mesh'
        else: name = args[0]
        mesh = self.get_mesh()
            
        cmap = 'jet'
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
        fig = plt.figure(figsize=plt.figaspect(0.44),dpi = 200)   
        
        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        #plt.zticks(fontsize=10)

        sctt = ax.scatter3D(np.asarray(mesh[::2,1],dtype='float'), np.asarray(mesh[::2,2],dtype='float'), np.asarray(mesh[::2,3],dtype='float'),
                    alpha = 0.4,
                    c = np.asarray(mesh[::2,4],dtype='float'),
                    cmap = cmap,
                    marker ='s')
        ax.set_title('Pontencial (V)')
        
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        
        
        ax.quiver(np.asarray(X[::3,::3,1::3],dtype='float'),np.asarray(Y[::3,::3,1::3],dtype='float'),np.asarray(Z[::3,::3,1::3],dtype='float'),
                  np.asarray(Jx[::3,::3,1::3],dtype='float'),np.asarray(Jy[::3,::3,1::3],dtype='float'),np.asarray(Jz[::3,::3,1::3],dtype='float'),
                  color = 'k',alpha = 0.3,length=h,normalize = True)
        ax.set_title('Densidades de Corrente ($\\frac{A}{cm^2}$)')

        plt.savefig(name + '.png')

    def generate_pid(self,i,j,k):
        return str(i) + '.' + str(j) + '.' + str(k)

    def plot2d(self,filename = 'plot2D', list_cuts1 = [0,1,2], list_cuts2 = [0,1,2]):

        H1 = 1
        h = self.get_h()
        r = self.get_r0()
        X = self.get_gridX()
        Y = self.get_gridY()
        Z = self.get_gridZ()
        V = self.get_gridV()
        Jx = self.get_gridJx()
        Jy = self.get_gridJy()
        Jz = self.get_gridJz()

        print(list_cuts1,list_cuts2)

        x_1 = list_cuts1

        fig, ax = plt.subplots(ncols=3,constrained_layout=True, dpi = 200, figsize = (12,4))

        for e,val in enumerate(x_1):

            print(val)
            try:
                y1 = np.reshape(Y[np.where(X == val)],(X.shape[1],X.shape[2]))
                z1 = np.reshape(Z[np.where(X == val)],(X.shape[1],X.shape[2]))
                v1 = np.reshape(V[np.where(X == val)],(X.shape[1],X.shape[2]))

                cmap = 'jet'
                #plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
                #shading='gouraud'
                c = ax[e].pcolormesh(y1, z1, v1, cmap=cmap,vmin=0, vmax=10,shading='gouraud', snap = True)
                fig.colorbar(c, ax=ax[e], shrink = 0.7, label = 'Potencial (V)', aspect = 20)
            except: None
            ax[e].set_title('d = %.2f cm' %(val))

            ax[e].spines["top"].set_visible(False)
            ax[e].spines["right"].set_visible(False)
            

        plt.savefig(filename + '_longitudinal.png')

        z_1 = list_cuts2

        fig, ax = plt.subplots(ncols=3,constrained_layout=True, dpi = 200, figsize = (12,4))

        for e,val in enumerate(z_1):

            print(val)
            try:
                y1 = np.reshape(Y[np.where(Z == val)],(X.shape[0],X.shape[1]))
                x1 = np.reshape(X[np.where(Z == val)],(X.shape[0],X.shape[1]))
                v1 = np.reshape(V[np.where(Z == val)],(X.shape[0],X.shape[1]))

                cmap = 'jet'
                #plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
                #vmin=np.min(v1), vmax=np.max(v1)
                #, shading='gouraud'
                c = ax[e].pcolormesh(x1, y1, v1, cmap=cmap,vmin=0, vmax=10,shading='gouraud',snap = True)
                fig.colorbar(c, ax=ax[e], label = 'Potencial (V)')
            except: None
            ax[e].set_title('d = %.2f cm' %(val))
            
                
            ax[e].spines["top"].set_visible(False)
            ax[e].spines["right"].set_visible(False)

        plt.savefig(filename + '_transversal.png')   
        
        
    def save_npz(self,filename = 'mesh'):

        data = self.get_mesh()
        features = self.features()
        X = self.get_gridX()
        Y = self.get_gridY()
        Z = self.get_gridZ()
        V = self.get_gridV()
        Jx = self.get_gridJx()
        Jy = self.get_gridJy()
        Jz = self.get_gridJz()
        I = self.get_I()
        lim_fun = self.get_limfun()
        general = np.array([self.get_r0(),self.get_r1(),self.get_h()])

        np.savez_compressed(filename + '.npz',data = data, features = features,gridX = X, gridY = Y,
                            gridZ = Z, gridV = V, gridJx = Jx, gridJy = Jy, gridJz = Jz, I = np.array(I),
                            lim_fun = lim_fun, general_data = general)       

    def get_f(self):
        return self.__f
    
    def get_r0(self):
        return self.__r0

    def get_r1(self):
        return self.__r1

    def get_H1(self):
        return self.__H1
    
    def get_h(self):
        return self.__h

    def features(self):
        return self.__features

    def get_mesh(self):
        return np.copy(self.__mesh)

    def get_gridX(self):
        return self.__gridX

    def get_gridY(self):
        return self.__gridY

    def get_gridZ(self):
        return self.__gridZ

    def get_gridV(self):
        return self.__gridV

    def get_gridJx(self):
        return self.__gridJx

    def get_gridJy(self):
        return self.__gridJy

    def get_gridJz(self):
        return self.__gridJz

    def set_mesh(self,new_mesh):
        if self.__mesh.shape == new_mesh.shape:
            self.__mesh = np.copy(new_mesh)
        else: raise ValueError('Could not set mesh with different shape.')

    def get_limfun(self):
        return self.__lim_fun[0],self.__lim_fun[1]

    def get_sigma(self):
        return self.__sigma
    def set_sigma(self, v):
        self.__sigma = v

    def get_idx_V(self):
        return int(self.__idx_V)

    def get_idx_Jx(self):
        return int(self.__idx_Jx)

    def get_idx_Jy(self):
        return int(self.__idx_Jy)

    def get_idx_Jz(self):
        return int(self.__idx_Jz)

    def get_idx_up(self):
            return int(self.__idx_up)

    def get_idx_down(self):
            return int(self.__idx_down)

    def get_idx_left(self):
            return int(self.__idx_left)

    def get_idx_right(self):
            return int(self.__idx_right)

    def get_idx_front(self):
            return int(self.__idx_front)
        
    def get_idx_back(self):
            return int(self.__idx_back)

    def get_I(self):
        return self.__I
