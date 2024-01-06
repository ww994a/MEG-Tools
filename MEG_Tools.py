from scipy import io
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import networkx as nx
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy import signal
import copy
from scipy.stats import kurtosis, skew


class MEG:
    def __init__(self, filename):
        #init variables
        self.data_raw = None
        self.fif_data_info = None
        self.sampling_freq = None
        self.highpass = None
        self.lowpass = None
        self._mat_data = None
        self._fif_file = None
        self.channels = []
        self._sd = {}
        
        #set variables
        self.filename = filename

    def add_marker(self, time, id=1):
        self.markers.append((time,id))
        
    def copy(self):
        new_object = copy.deepcopy(self)
        return new_object
        
    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        print ('Creating MEG object...',end="\r")
        self._filename = value
        self._mat_data = io.loadmat(value)
        self.data_raw = self._mat_data['data_raw']
        self.markers = [] #can mark events with tuple (marker_time in seconds, id)
        self.time_raw = self._mat_data['time_raw']
        self.fif_data_info = self._mat_data['fif_data_info']
        self.sampling_freq =   self.fif_data_info[0][0][1][0][0][4][0][0]
        self.highpass = self.fif_data_info[0][0][1][0][0][5][0][0] 
        self.lowpass = self.fif_data_info[0][0][1][0][0][6][0][0]
        #channels
        self.channels_list = self.fif_data_info[0][0][1][0][0][7][0]
        keys = ['scanno', 'logno', 'kind', 'range', 'cal', 'coil_type', 'loc', 'coil_trans', 'eeg_loc', 'coord_frame', 'unit', 'unit_mul', 'ch_name']
        
        for n in range(len(self.channels_list)):

            values =  self.channels_list[n]
            # Creating the dictionary
          
            data_dict = dict(zip(keys, values))
            self.channels.append(data_dict)
            
        #Unnest variables - 

        
        for n, channel in enumerate(self.channels):
            channel['coord_frame'] = channel['coord_frame'][0][0]
            new_loc_array = []
            for item in channel['loc']:
                new_loc_array.append(item[0])
            channel['loc'] = np.array(new_loc_array)
            if type(channel['ch_name']) != np.str_:
                channel['ch_name'] = channel['ch_name'][0]
            if type(channel['range']) != np.float64:
                channel['range'] = channel['range'][0][0]
            if type(channel['scanno']) != np.int32:
                channel['scanno'] = channel['scanno'][0][0] 
            if type(channel['logno']) != np.int32:
                channel['logno'] = channel['logno'][0][0] 
            if type(channel['coil_type']) != np.int32:
                channel['coil_type'] = channel['coil_type'][0][0] 
            if type(channel['cal']) != np.float64:
                channel['cal'] = channel['cal'][0][0]
            if type(channel['kind']) != np.int32:
                channel['kind'] = channel['kind'][0][0]
            if type(channel['unit']) != np.int32:
                channel['unit'] = channel['unit'][0][0]
            if type(channel['unit_mul']) != np.int32:
                channel['unit_mul'] = channel['unit_mul'][0][0]

        print('Complete!                         ')
        #Other data?
        #dipole_sleep_2
        self.info = {}
        for key, value in self._mat_data.items():
            if key not in ('__header__', '__version__', '__globals__','data_raw', 'fif_data_info', 'time_raw'):
                self.info[key] = value
                
        #Spikes
        self.spikes = []
        self.set_spikes()
        self.spike_peaks = []
        self.set_spike_peaks()
    


        
    def filter_data(self, lowcut=1.0, highcut=100.0, type='band'): #use bandstop to see what was removed
        # Define the filter parameters
        fs = self.sampling_freq  # Sampling frequency in Hz 
        order = 4  # Filter order (adjust as needed)
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype=type)
        print("Filtering...",end="\r")
        self.data_raw = signal.lfilter(b, a, self.data_raw)
        print(f"Data filtered from {lowcut} Hz to {highcut} Hz")
        
    
        
    def get_channel_number(self, name):
        for n in range(len(self.channels)):
            if self.channels[n]['ch_name'] == name:
                return n
        raise Exception("Channel "+name+" not found.")
    
    def get_channel_data(self, channel_id):
        if isinstance(channel_id, str):
            channel_id = self.get_channel_number(channel_id)
        return self.data_raw[channel_id]

    def get_channel_slice (self, channel_id, start, finish, segments=False): #in seconds unless segments = true
        if isinstance(channel_id, str):
            channel_id = self.get_channel_number(channel_id)
        if start == 0 and finish == 0: #return all
                return self.data_raw[channel_id]
        multiplier = self.sampling_freq
        
        if segments: multiplier = 1 #for samples
            
        #print(channel_id,int(np.floor(start * multiplier)),":",int(np.floor(finish * multiplier)))
        
        return self.data_raw[channel_id][int(np.floor(start * multiplier)): int(np.floor(finish * multiplier))]


    def get_features_at_location(self,channel,location,margin=200, smoothing = 0,  plot=False, center_line = False):
 
        start = location - margin / self.sampling_freq #100 ms in either direction
        end = location  + margin / self.sampling_freq #100 ms in either direction

        peaks = self.get_peak_features(channel,start,end,sd_threshold = 0.0, smoothing = smoothing ,plot=False)
        #which is closest to center?
        distance = margin
        peak_data = None
        
        for peak in peaks:
            if peak['value'] > 0 and abs(peak['time'] - margin) < abs(distance): #should be closest POSITIVE to margin
                distance = peak['time'] - margin
                peak_data = peak

        if plot: #to center the graph and make line match up
            self.get_peak_features(channel,start,end,sd_threshold = 0.0, smoothing = smoothing ,plot=True,lines = [peak_data['time']])
        return peak_data
    
    def get_gaussian_smoothed(self,sensor,start=0,end=0, smoothing=7):
        # Apply Gaussian smoothing
        sig = self.get_channel_slice(sensor,start,end)
        if smoothing > 0:
            sigma = smoothing  # Standard deviation for Gaussian kernel
            sig_smoothed = gaussian_filter(sig, sigma=sigma)
        else:
            sig_smoothed = sig
        return sig_smoothed

    def get_mne(self, preload=True, use_cached=False):
        if preload == False:
            print('Events will not be loaded when preload=False')
         
        self._fif_file = self.filename+'.raw.fif'
        if not use_cached or not os.path.exists(self._fif_file):
            #delete old temporary file
            if os.path.exists(self._fif_file):
                os.remove(self._fif_file)
            #Let's get Channel names and type
            ch_names = []
            ch_types = []
    
            for chan in self.channels:
                ch_names.append(chan['ch_name'])
                if (chan['kind']) == 1:
                    ch_types.append('mag')
                else :
                    ch_types.append('eeg')
            # Mne Object
            info = mne.create_info(ch_names, ch_types=ch_types, sfreq=self.sampling_freq) 
            MEG_raw = mne.io.RawArray(self.data_raw, info)
            #save fif file
            MEG_raw.save(self._fif_file, overwrite=True)
        else:
            print ('Using Cached File...')

        raw = mne.io.read_raw(self._fif_file, preload=preload)
        #format and set info
        for n in range(len(self.channels)):
            raw.info['chs'][n]['logno'] = self.channels[n]['logno']
            raw.info['chs'][n]['range'] = self.channels[n]['range']
            raw.info['chs'][n]['cal'] = self.channels[n]['cal']
            raw.info['chs'][n]['coil_type'] = self.channels[n]['coil_type']
            raw.info['chs'][n]['loc'] = self.channels[n]['loc']
            raw.info['chs'][n]['unit'] = self.channels[n]['unit']

        #add spikes as events if they exist
        if len(self.spikes) > 0 and preload == True:
            #get spike times
            spike_times = []
            spike_duration = []
            for spike in self.spikes:
                spike_times.append(spike['begin'])
                spike_duration.append(spike['end']-spike['begin'])

            spike_list = ['Spike' for _ in spike_times]
            spike_ch = [[] for _ in spike_times]
            
            #create annotations
            annotations = mne.Annotations(onset=spike_times, duration=spike_duration,
            description=spike_list,
            ch_names=spike_ch)

            raw.set_annotations(annotations)

            events_from_annot, event_dict = mne.events_from_annotations(raw, event_id={'Spike':1}, use_rounding=True, chunk_duration=None, verbose=None)

            raw.add_events(events_from_annot) #, stim_channel
        return raw

    

            
    def get_peaks(self,sensor,start=0,end=0,threshold=1e-13, smoothing=0, plot=False, lines=None):
  
        sig_smoothed = self.get_gaussian_smoothed(sensor,start,end, smoothing)
        # Define a prominence threshold
        prominence_threshold = threshold  

        # Find Peaks and filter by prominence
        sig_peaks, properties_peaks = signal.find_peaks(sig_smoothed)
        peak_promin = signal.peak_prominences(sig_smoothed, sig_peaks)[0]
        
        high_promin_peaks = sig_peaks[peak_promin > prominence_threshold]
        high_promin_peak_values = sig_smoothed[high_promin_peaks]
        
        # Find Troughs and filter by prominence
        sig_inverted = -sig_smoothed
        sig_troughs, properties_troughs = signal.find_peaks(sig_inverted)
        trough_promin = signal.peak_prominences(sig_inverted, sig_troughs)[0]
        high_promin_troughs = sig_troughs[trough_promin > prominence_threshold]
        high_promin_trough_values = sig_smoothed[high_promin_troughs]
    
        changes = []

        if len(high_promin_troughs) != 0 and  len(high_promin_peaks) != 0:
            
            #which comes first?
            if high_promin_troughs[0] > high_promin_peaks[0]:
                list_one = high_promin_peaks
                list_one_values = high_promin_peak_values
                list_two = high_promin_troughs
                list_two_values = high_promin_trough_values
            else: 
                list_two = high_promin_peaks
                list_two_values = high_promin_peak_values
                list_one = high_promin_troughs
                list_one_values = high_promin_trough_values
    
            #now go through and calculate
            
            changes.append(list_one_values[0]) #assumes change from 0
    
            count = 0
            while count < len(list_one)-1:
                changes.append(list_two_values[count]-list_one_values[count])
                changes.append(list_one_values[count+1]-list_two_values[count])
                count += 1
    
            if (len(list_two_values) == len(list_one_values)):
                changes.append(list_two_values[count]-list_one_values[count])
        else:
            #there were not enough peaks to cmpute add 0
            changes.append(0)
            

        
        
        # Plotting
        if plot:
            plt.plot(sig_smoothed)
            plt.plot(high_promin_peaks, high_promin_peak_values, "r*", label='Prominent Peak')
            plt.plot(high_promin_troughs, high_promin_trough_values, "g*", label='Prominent Trough')
            if lines != None:
                for line in lines:
                    plt.axvline(line, color='red')
            plt.legend()
            plt.show()

        #pks = np.concatenate([high_promin_peaks, high_promin_troughs])
        cng = np.array(changes)

        # Combining peaks and troughs and keeping track of their origin
        combined = [(value, 'peak') for value in high_promin_peaks] + [(value, 'trough') for value in high_promin_troughs]
    
        # Sorting the combined list and keeping track of the indices
        sorted_combined = sorted(enumerate(combined), key=lambda x: x[1][0])
    
        # Preparing the final sorted lists
        sorted_peaks_troughs = []
        sorted_values = []
    
        for index, (value, tag) in sorted_combined:
            sorted_peaks_troughs.append(value)
            if tag == 'peak':
                sorted_values.append(high_promin_peak_values[index])
            else:  # tag == 'trough'
                sorted_values.append(high_promin_trough_values[index - len(high_promin_peaks)])
        pks = np.array(sorted_peaks_troughs)
        pk_val = np.array(sorted_values)

      
       
        # Calculate widths
        wdth = np.array([pks[i+1] - (pks[i-1] if i > 0 else 0) if i < len(pks) - 1 else len(sig_smoothed) - (pks[i-1] if i > 0 else 0) for i in range(len(pks))])
        
        # Calculate left widths
        wdth_left = np.array([pks[i] - (pks[i-1] if i > 0 else 0) for i in range(len(pks))])
        
        # Calculate right widths
        wdth_right = np.array([(pks[i+1] if i < len(pks) - 1 else len(sig_smoothed)) - pks[i] for i in range(len(pks))])

        
        return (pks,pk_val,cng, wdth, wdth_left, wdth_right) #returns location, height, change, widths, etc.
    


    #score each peak based on it fitting criteria 1 point for each

    
    
    def get_peak_features (self, channel, start=0, end=0, sd_threshold = 1.0, 
                           smoothing=12,plot=False, lines = None):
        
        peaks = self.get_peaks(channel,start,end,smoothing=smoothing,plot=plot, lines=lines)
        sd_peaks = self.sd(channel, smoothing=smoothing)
        data = self.get_gaussian_smoothed(channel,start,end, smoothing)
        
        peak_features = []
        
        for n, peak in enumerate(peaks[1]):
            if abs(peak) > sd_peaks*sd_threshold: #eliminates smaller peaks
                peak_width = (peaks[3][n]/self.sampling_freq)*1000#ms regardless of sample freq
                peak_width_left = (peaks[4][n]/self.sampling_freq)*1000#ms regardless of sample freq
                peak_width_right = (peaks[5][n]/self.sampling_freq)*1000#ms regardless of sample freq
               
        
                #changes
                change_left = None
                change_right = None
               
                if n != 0:
                    change_left = peaks[2][n]
                else:
                    change_left = peak #amount from 0
        
                if n < len(peaks[2])-1:
                    change_right = peaks[2][n+1]
                else:
                    change_right = peak #amount from 0
        
        
    
                    
                #gradients - calculate from slope of mid point

                midpoint_left = round(peaks[0][n] - peak_width_left / 2)
                midpoint_right = round(peaks[0][n] + peak_width_right / 2)

                # Compute the gradient

                x = np.arange(len(data)) 
                y = data
                
                dy_dx = np.gradient(y, x)

                # Get the slope at the point of interest
                slope_l = dy_dx[midpoint_left]
                slope_r = dy_dx[midpoint_right]

                # Compute the sharpness - second derivative
                second_derivative = np.gradient(dy_dx, x)
                
                # Find the index of the peak
                peak_index = peaks[0][n]
                
                # Evaluate the sharpness at the peak (second derivative at the peak)
                sharpness_at_peak = second_derivative[peak_index]

                #kurtosis and skewness
                _left = math.ceil(peaks[0][n] - peak_width_left)
                _right = math.floor(peaks[0][n] + peak_width_right)
                peak_data = data[_left:_right]
                
                # Calculate skewness and kurtosis
                skewness = skew(peak_data)
                kurt = kurtosis(peak_data, fisher=False)  # Set fisher=False for standard kurtosis
                
                            
                peak_features.append({'id':n,'time':peaks[0][n],'value':peak,
                                      'width':peak_width,'width_left':peak_width_left,'width_right':peak_width_right,
                                      'change_left':change_left,'change_right':change_right,'slope_left':slope_l,'slope_right':slope_r,
                                     'sharpness':sharpness_at_peak,'skewness':skewness,'kurtosis':kurt}) 
        return peak_features

    
        
    
    
        

    def max_peak_channel_at_location(self, location, threshold=1e-13, margin=50, smoothing=0):
        return self.max_peak_channel(start=location-margin/self.sampling_freq, finish=location+margin/self.sampling_freq, threshold=threshold, smoothing=smoothing)
        
        
    def max_change(self,sensor,start=0,finish=0,threshold=1e-13, smoothing=0):

            the_peaks = (self.get_peaks(sensor,start,finish,threshold,smoothing))
            if (len(the_peaks[0]) == 0):
                        return (None, None)
            
            biggest_change_index = 0
            biggest_change = None
            
            for n, change in enumerate(the_peaks[2]):
                if biggest_change == None:
                    biggest_change = change
                biggest_change = max(biggest_change, change)
                if biggest_change == change:
                    biggest_change_index = n
            
            biggest_change_location = the_peaks[0][biggest_change_index]
            
            return (biggest_change_location, biggest_change)
        
    
    
    
    def max_peak(self,sensor,start=0,finish=0,threshold=1e-13, smoothing=0):

            the_peaks = (self.get_peaks(sensor,start,finish,threshold,smoothing))
            if (len(the_peaks[0]) == 0):
                    return (None, None, None)

            biggest_peak_index = 0
            biggest_peak = None

            for n, peak in enumerate(the_peaks[1]):
                if biggest_peak == None:
                    biggest_peak = peak
                biggest_peak = max(biggest_peak, peak)
                if biggest_peak == peak:
                    biggest_peak_index = n
            
            biggest_peak_location = the_peaks[0][biggest_peak_index]
            biggest_peak_width = the_peaks[3][biggest_peak_index]
            #biggest_peak_sd = the_peaks[4][biggest_peak_index]
            return (biggest_peak_location, biggest_peak, biggest_peak_width)

        
    def max_peak_channel(self,start=0,finish=0,threshold=1e-13, smoothing=0):
            channel_ = None
            max = 0
            
            for channel in self.channels:
                if channel['kind'] == 1:
                    current_max = self.max_peak(channel['ch_name'],start,finish,threshold,smoothing)[1]
                    if current_max != None and current_max > max:
                        channel_ = channel['ch_name']
                        max = current_max
            return channel_


    def median_filter_data(self):
        print("Filtering...",end="\r")

        # Apply median filter
        kernel_size = 5
        for n in range(len(self.data_raw)):
            self.data_raw[n] = medfilt(self.data_raw[n], kernel_size=kernel_size)
        print("Data Median Filtered")


    def plot_channel_slice(self, channel_id, start, finish, spikes=True, markers=True, smoothing=0, score_resolution=1000, lines=None):
    
        plt.plot(self.get_gaussian_smoothed(channel_id, start, finish, smoothing=smoothing))
        if spikes:
            #look for spikes in slice
            for spike in self.spikes:
                if spike['begin'] >= start and spike['begin'] <= finish:
                    plt.axvline((spike['begin']*self.sampling_freq)-start*self.sampling_freq, color='red', linestyle='--')
    
        if markers:
            #look for spikes in slice
            marker_pal = {1:'g', 2:'c', 3:'m', 4:'y', 5:'k'}
            colors = ["navy", "blue", "lightblue", "white", "lightcoral", "red", "darkred"]
    
            for marker in self.markers:
                if marker[0] >= start and marker[1] <= finish:
                    plt.axvline((marker[0]*self.sampling_freq)-start*self.sampling_freq, color=marker_pal[marker[1]], linestyle='--')
    
        if lines != None:
            for line in lines:
                
                plt.axvline(line, color='red')
                    
        if isinstance(channel_id, str):
            channel_id = self.get_channel_number(channel_id)
        if smoothing == 0:
            smoothed_text = ''
        else:
            smoothed_text = ' Smoothed'
             
        plt.title('Channel '+self.channels[channel_id]['ch_name']+smoothed_text)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.show()
    
    def sd(self,channel_name, smoothing=0):
            if channel_name in self._sd:
                return self._sd[channel_name]
            sd_peaks = np.std(self.get_peaks(channel_name,0,0,smoothing=smoothing)[1])
            self._sd[channel_name] = sd_peaks
            return sd_peaks

    def set_spikes(self):
            spike_location = ''
            for key in self.info.keys():
                if key.startswith("Dipole") or key.startswith("dipole"):
                    spike_location = key
                    break 
                
            if spike_location in self.info:
                self.spikes = []
                
                for n in range(len(self.info[spike_location][0])):
                    spike = {}
                    spike['dipole'] = self.info[spike_location][0][n][0][0][0]
                    spike['begin'] = self.info[spike_location][0][n][1][0][0]
                    spike['end'] = self.info[spike_location][0][n][2][0][0]
                    spike['r0'] = self.info[spike_location][0][n][3]
                    spike['rd'] = self.info[spike_location][0][n][4]
                    spike['Q'] = self.info[spike_location][0][n][5]
                    spike['goodness'] = self.info[spike_location][0][n][6][0][0]
                    spike['errors_computed'] = self.info[spike_location][0][n][7][0][0]
                    spike['noise_level'] = self.info[spike_location][0][n][8][0][0]
                    spike['single_errors'] = self.info[spike_location][0][n][9]
                    spike['error_matrix'] = self.info[spike_location][0][n][10]
                    spike['conf_volume'] = self.info[spike_location][0][n][11][0][0]
                    spike['khi2'] = self.info[spike_location][0][n][12][0][0]
                    spike['prob'] = self.info[spike_location][0][n][13][0][0]
                    spike['noise_est'] = self.info[spike_location][0][n][14][0][0]
                    self.spikes.append(spike)

    def set_spike_peaks(self):
        
        spike_begins = []
        for spike in self.spikes:
            spike_begins.append(spike['begin'])

            
        # find first peak after
        for n in range(len(spike_begins)):
            channel = self.max_peak_channel(spike_begins[n], spike_begins[n]+200/self.sampling_freq, smoothing=12)
            peaks = self.get_peaks(channel,spike_begins[n],spike_begins[n]+self.sampling_freq, smoothing=12)
            for m in range(len(peaks)):
                if peaks[1][m] > 0:
                    self.spike_peaks.append(peaks[0][m]/self.sampling_freq+spike_begins[n])
                    break


# This class represents a directed graph using
# adjacency list representation
class Graph:

    # Constructor
    def __init__(self):
        # Default dictionary to store graph
        self.graph = defaultdict(list)
        self.vertices = defaultdict(list)
    
    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Function to add a vertex
    def add_vertex(self, name, x, y, id, info=None):
        self.vertices[name] = (x, y, id, info)

    def set_info(self, dict): #dict key=channel_name 
        for key, value in dict.items():
            if key in self.vertices:
                self.vertices[key] = (self.vertices[key][0],self.vertices[key][1],self.vertices[key][2],value)

        
    def distance(self, vertex1, vertex2):
        x1, y1 = self.vertices[vertex1][:2]
        x2, y2 = self.vertices[vertex2][:2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def create_edges_based_on_distance(self, threshold=0.1):
        for v1 in self.vertices:
            for v2 in self.vertices:
                if v1 != v2 and self.distance(v1, v2) < threshold:
                    self.addEdge(v1, v2)
    
    # A function used by DFS
    def DFSUtil(self, v, visited):
        # Mark the current node as visited and print it
        visited.add(v)
        print(v, end=' ')

        # Recur for all the vertices adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses recursive DFSUtil()
    def DFS(self, v):
        # Create a set to store visited vertices
        visited = set()

        # Call the recursive helper function to print DFS traversal
        self.DFSUtil(v, visited)



    def draw_graph(self, node_size=300, fig_size=(12, 8), font_size=8, font_color='red', edge_color='gray'):  # Added edge_color parameter with default value 'gray'
        G = nx.DiGraph()
        for vertex, pos in self.vertices.items():
            G.add_node(vertex, pos=(pos[0], pos[1]))
        
        for vertex, edges in self.graph.items():
            for edge in edges:
                G.add_edge(vertex, edge)
        
        pos = nx.get_node_attributes(G, 'pos')
    
        plt.figure(figsize=fig_size)  # Set the figure size
        # Updated to include edge_color in the nx.draw function
        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='skyblue', font_size=font_size, font_color=font_color, edge_color=edge_color)
        plt.show()

#####UTILITIES

import functools
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer






            
        
            