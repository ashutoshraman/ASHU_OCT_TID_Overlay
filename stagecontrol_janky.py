import Lib_DataProcess
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

tid = Lib_DataProcess.TumorID() #use arg to specify int time
ldcurrent = 80 #change to 250 when ready to actually scan

data, wv = tid.measurment(50)
#fname = 'white_spectra.csv' 
#tid.spec.save(data, wv, fname, 1) #uncomment when saving multiple spectra

plt.figure(1)
plt.plot(wv, data, 'red')
plt.xlabel('wavelength')
plt.ylabel('Intensity')
plt.title('Spectral Capture')
plt.show()


xpos = 0
ypos = 0
tid.stage.move_to(xpos,ypos)
time.sleep(3)
tid.ld.turn_laser_ON(ldcurrent/1000)
input('Sample Positioned well?: ')
tid.ld.turn_laser_OFF()


plt.figure(2)
integrationtime = .25 #change to increase laser on time
#define ROI as function
def create_step_positions(x_start, y_start, x_size, y_size, step_size):
    x_steps = np.linspace(x_start, x_start+x_size, x_size/step_size +1)
    y_steps = np.linspace(y_start, y_start+y_size, y_size/step_size +1)
    return x_steps, y_steps

def normalize_and_smooth(data, w=10): #perhaps add functionality to cut wavelengths with 405 peak for real tissue later
    spectra = data
    target = np.max(spectra)
    data = data/target
    data = np.convolve(data,np.ones(w)/w,'same')
    return data

x_steps, y_steps = create_step_positions(xpos, ypos, 0, 30, .05) #600x600 will take forever if it takes this long for one acquisition

#once laser repositions at new location, it takes forever to scan when it shouldn't
#if this time cannot be reduced, think about increasing precision of step only at boundaries
#pre-measure spectra of each fiducial, tumor and nontumor surety. at every spectral capture, compare to reference and classify
#based on RMSE or some other similar simple technique, if error is high then go to rapid large steps
#only do this when exiting feature to non tumor so you can compare previous one that satisfied error threshold to one that
#doesn't then proceed
#also try boundary estimation and approx mehods online, gihub and articles
# to reduce time in beginning, take large steps and when finally getting to new boundary, shorten step size and move backward
#cross boundary again, then shorten step size again and move forward to cross boundary, then restart large step sizes until next
#boundary and do same thing in reverse this time when exiting boundary

black_spectra = normalize_and_smooth(np.genfromtxt("C:\Users\tjz5.DHE\Desktop\data\black_spectra.csv", delimiter = ',')[:, 1])
white_spectra = normalize_and_smooth(np.genfromtxt("C:\Users\tjz5.DHE\Desktop\data\white_spectra.csv", delimiter = ',')[:, 1])



def compare_to_reference(in_spec, ref_spec0, ref_spec1):
    #output both RMSE value and also classify based on defined threshold, maybe an arg as well as reference file to compare to
    data = normalize_and_smooth(in_spec)
    RMS_one = np.sqrt((np.sum(np.square(ref_spec1 - data))) / ref_spec1.size)
    RMS_zero = np.sqrt((np.sum(np.square(ref_spec0 - data))) / ref_spec0.size)
    if RMS_one > RMS_zero:
        classify = 0
    else:
        classify = 1
    return RMS_one, RMS_zero, classify

def rms_plot_class_errors():
    pass




for i in x_steps:
    for y, j in enumerate(y_steps): #this is raster path number points to collect, but change step size too
        tid.stage.move_to(i,j)
        if j == 0:
            time.sleep(3)
        time.sleep(.15)
        tid.ld.turn_laser_ON(ldcurrent/1000)
        data = tid.spec.capture()
        tid.ld.turn_laser_OFF()
        fname  = "normal-smallbox-{:.3f}-{:.3f}-{}-{}".format(i,j,ldcurrent,integrationtime)
        tid.spec.save(data,wv,fname,1)
        #if j % 50 == 0:
        if y+1 < y_steps.size/4:
            plt.plot(wv,data, label = '{}, {}'.format(i, j)) #figure out how to do with commented out Tanner save method shown in data process classes
        else:
            plt.plot(wv,data)
        #time.sleep(.1) #.1 s adds 10 hrs for step size .05 range 3
plt.legend()
plt.show()

sys.exit()
### while loop to evaluate broad boundary, move back in smaller steps, then move forward again in even smaller step
### then restart large step size without aliased step size to feature size


x=0; y=0
x_new=0; y_new=0
scan_region = 30 #3cm since full range is 300mm or 30cm
step_size = .05
x_steps, y_steps = create_step_positions(x, y, scan_region, scan_region, step_size) # remember to change x to scan region as well when running real thing
tumor_map = np.zeros(scan_region/step_size, scan_region/step_size) #pay attention to coordinates, stages move from bottom right corner, 0,0 here is top left, rotate this 180 degrees to be like reality or cam picture



tid.stage.move_to(x, y)
tid.ld.turn_laser_ON(ldcurrent/1000)
input('Sample Positioned well?: ')
in_spec = np.array(tid.spec.capture())
tid.ld.turn_laser_OFF()
RMS_one, RMS_zero, classify = compare_to_reference(in_spec, white_spectra, black_spectra)
tumor_map[0,0] = classify
print(RMS_one, RMS_zero)


def boundary_find_based_on_rms(RMS_a, RMS_b, tumor_map_xind, tumor_map_yind, step_sizes, direction='forward'):
    while RMS_b > RMS_a: #this means you're not on tumor since error for class 1 is greater than that of class 0 (healthy)
        
        tid.stage.move_to(x_new, y_new)
        if y_new == 0:
            time.sleep(3)
        time.sleep(.25)
        tid.ld.turn_laser_ON(ldcurrent/1000)
        in_spec = np.array(tid.spec.capture())
        tid.ld.turn_laser_OFF()
        RMS_b, RMS_a, classify = compare_to_reference(in_spec, white_spectra, black_spectra)
        tumor_map[tumor_map_xind, 100*(tumor_map_yind+1)-1] = classify #fix this, this is very wrong, but it should build classified map of 1s and 0s
        
        if direction == 'backward':
            y_new-=step_sizes #global variable of x and y
        else:
            y_new+=step_sizes #global variable of x and y
        if y_new >= y+scan_region or y_new <= y:
             return RMS_a, RMS_b
    return RMS_a, RMS_b


for i, x in enumerate(x_steps):
    
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_zero, RMS_one, i, y_new, step_size*100)
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_one, RMS_zero, i, y_new, step_size*10, direction='backward')
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_zero, RMS_one, i, y_new, step_size, direction='forward')
    time.sleep(3) #you can delete later, just for demonstration purpose
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_one, RMS_zero, i, y_new, step_size*100, direction='forward')
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_zero, RMS_one, i, y_new, step_size*10, direction='backward')
    RMS_zero, RMS_one = boundary_find_based_on_rms(RMS_one, RMS_zero, i, y_new, step_size, direction='forward')
    
    x_new=x;y_new = 0


    

    
        
        
    
    
    
    

tumor_map = np.rot90(tumor_map, 2)