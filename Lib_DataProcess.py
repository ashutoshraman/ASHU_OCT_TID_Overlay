
'''
1. Load ID data
2. build ID dataset
3. Post-process ID data.

'''
import time
import numpy as np
from datetime import datetime
import pyvisa
from tkinter import *
import matplotlib.pyplot as plt
import os, sys, clr
from System import *
import pandas as pd
import thorlabs_apt as apt 

class ControlStage:
    def __init__(self):
        apt.list_available_devices()
        self.x_motor = apt.Motor(45199964)
        self.y_motor = apt.Motor(45199944)
        self.x_motor.move_home()
        self.y_motor.move_home()
        time.sleep(2)
        print("Stages Homed and Init")

    def home(self):
        self.x_motor.move_home()
        self.y_motor.move_home()
        self.xvec = 0.0
        self.yvec = 0.0
        print("Stages Homed: ({},{})".format(self.xvec,self.yvec))

    def move_to(self,xloc,yloc):
        self.x_motor.move_to(xloc)
        self.y_motor.move_to(yloc)
        self.xvec = xloc
        self.yvec = yloc
        print("Location: ({},{})".format(self.xvec,self.yvec))
        

    def move_by(self,xd,yd):
        self.x_motor.move_by(xd)
        self.y_motor.move_by(yd)
        self.xvec += xd
        self.yvec += yd
        print("Location: ({},{})".format(self.xvec,self.yvec))

    def get_position(self):
        return self.xvec, self.yvec

class ControlLD:
    def __init__(self):
        self.rm = pyvisa.ResourceManager();
        print(self.rm.list_resources())
        self.LD_name = 'USB0::0x1313::0x804F::M00470766::INSTR';
        self.my_instrument = self.rm.open_resource(self.LD_name);
        print('LD INIT Succesful: ' + self.my_instrument.query('*IDN?'))
        self.max_laser_current =325; #ma

    def turn_laser_ON(self, laser_current):
        if (laser_current > 250):
            laser_current = 150;

        msg = "source:current:level:amplitude " + str(laser_current)
        self.send_msg(msg);

        msg = "OUTPUT2:STATE 1"; #turn on TEC protection
        self.send_msg(msg);

        msg = "OUTPUT:STATE 1"; #turn on laser
        self.send_msg(msg)

    def turn_laser_OFF(self):
        msg = "OUTPUT:STATE 0";
        self.send_msg(msg)

    #Sends a VISA message to the LD and returns the response
    def send_msg(self,uinput):
        try:
            print(uinput)
            self.my_instrument.write(uinput);
        except:
            print ("It didn't work. Try something else.")

    def clear_error(self):
        msg = "*CLS"
        self.send_msg(msg);

class ControlSpec:
    def __init__(self):
        sys.path.append(r"C:\Program Files (x86)\Microsoft.NET\Primary Interop Assemblies\\")
        clr.AddReference("Thorlabs.ccs.interop64")
        import Thorlabs.ccs.interop64
        b1 = Boolean(True)
        self.spec = Thorlabs.ccs.interop64.TLCCS("USB0::0x1313::0x8089::M00458735::RAW",b1,b1)
        self.sessionData = pd.DataFrame(columns=["Name","wv","Data","Laser Current","Integration Time", "Comments"])
        print("SPEC INIT Sucessful")
        
    def setIntegration(self,integration_time):
        integrationTime = Double(integration_time)
        self.spec.setIntegrationTime(integrationTime)
        print("Integration Time Set: {:.2f}".format(integration_time))

    def getWavelengths(self):
        i = Int16(0)
        nullable_double = Nullable[Double](0) 
        wvdata = Array.CreateInstance(Double,3648)
        self.spec.getWavelengthData(i,wvdata,nullable_double,nullable_double)
        wavelengths = list(wvdata)
        return wavelengths
        
    def preCapture(self,integrationTime):
        self.setIntegration(integrationTime)
        wv = self.getWavelengths()
        return wv
    
    def capture(self):
        scan = Array.CreateInstance(Double,3648)
        self.spec.startScan()
        self.spec.getScanData(scan)
        scandata = list(scan)
        print("Capture Complete")
        return scandata
    
    def record(self,rtime,integrationTime,num):
        now = time.perf_counter()
        scan = Array.CreateInstance(Double,3648)
        self.setIntegration(integrationTime)
        data = np.zeros((3648,num))
        for i in range(num):
            self.spec.startScan()
            self.spec.getScanData(scan)
            scandata = np.asarray(list(scan))
            data[:,i] = scandata
        print("Function Time: {:.2f}s".format(time.perf_counter()-now))
        return data
    
    def plot(self,wv,data,num,root):
        fig = plt.figure(num=1,clear=True,figsize = (4,4),dpi=100)
        ax = fig.add_subplot(1,1,1)
        if num == 1:
            ax.plot(wv,data,label="Spectra")
        else:
            x = np.linspace(0,5,100)
            for i in range(num):
                ax.plot(wv,data[:,i],label="Trial {}".format(i))
            plt.legend()
        
    #def save(self,data,filename,laser_current,integration_time,comments,wv):
     #   full_filename = r"C:\Users\tjz5.DHE\Desktop\data\\" +  filename + ".csv"
      #  t = time.localtime()
       # current_time = time.strftime("%H:%M:%S", t)
        #data = pd.Series([current_time,wv,data,laser_current,integration_time,comments],index=self.sessionData.columns)
        #self.sessionData = self.sessionData.append(data,ignore_index=True)
        #self.sessionData.to_csv(full_filename,index=False)
        #self.sessionData.to_pickle(r"C:\Users\tjz5.DHE\Desktop\data\\"+filename+'.pkl')
        
    def save(self,data,wavelengths,filename,num):
        full_filename = r"C:\Users\tjz5.DHE\Desktop\data\\" +  filename + ".csv";
        if num == 1:
            final = np.array([wavelengths,data]).transpose()
            with open(full_filename, "w+") as myfile:
                np.savetxt(myfile, final, delimiter=',')
                myfile.close()
            print("File Save Success")
        else:
            wv = np.reshape(wavelengths,(3648,1))
            final = np.hstack((wv,data))
            with open(full_filename, "w+") as myfile:
                np.savetxt(myfile, final, delimiter=',')
                myfile.close()
            print("File Save Success")


class TumorID:

    def __init__(self, integration_time =.25):
        self.ld = ControlLD()
        self.spec = ControlSpec()
        self.stage = ControlStage()
        self.wv = self.spec.preCapture(integration_time)
    
    def measurment(self, laser_current = 185):
        self.ld.turn_laser_ON(laser_current)
        data = self.spec.capture()
        self.ld.turn_laser_OFF()
        return np.array(data), np.array(self.wv)

    def calibration(self, xguess, yguess, r = 5, n = 10 ):

        '''
        1. r: range of the local ROI, default = 5.0 mm
        2. n: number of step size, default = 10
        3. x_guess: initial position of x
        4. y_guess: initial position of y
        '''

        # tight grid search in the ROI defined by a center point and a radius
        # scanning region
        lowwv = 500
        highwv = 525
        s = r/np.sqrt(2)
        xroi = np.linspace(xguess - s, xguess + s, n)
        yroi = np.linspace(yguess - s, yguess + s, n)
        xx, yy = np.meshgrid(xroi, yroi)

        # vector of intensity
        z = np.zeros([n, n])
        i = 0

        # scan a local region
        for i, x in enumerate(xroi):
            for j, y in enumerate(yroi):

                # step-1 move the stage to the first position
                self.stage.move_to(x, y)

                # step-2 get the data + max-filter
                data, data_wavelength = self.measurment()
                idx_use = np.where((data_wavelength > 425) & (data_wavelength < 750))
                data_valid = data[idx_use]
                wv_valid = data_wavelength[idx_use]
                f_vec_idx = np.argmax(data_valid)
                f_vec = wv_valid[f_vec_idx]

                # step-3 update fiducial feature (max-intensity)
                if f_vec >= lowwv and f_vec <= highwv:
                    z[i,j] = 1
                else:
                    z[i,j] = 0

        # report the final fiducial positions
        x_coord = np.argmax(np.mean(z, axis = 1))
        y_coord = np.argmax(np.mean(z, axis = 0))

        # z now represents a 2D grid of all max intensity values
        # check surface
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(xx,yy,z)
        # value of 1 represents inside fiducuial, 0 represents outside, therefore average z to get mean coordinate

        return x_coord, y_coord

if __name__ == "__main__":
    tumor = TumorID()





