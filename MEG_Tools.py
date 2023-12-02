from scipy import io
import os
import mne
import numpy as np

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
        
        #set variables
        self.filename = filename
        

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        self._mat_data = io.loadmat(value)
        self.data_raw = self._mat_data['data_raw']
        
        self.time_raw = self._mat_data['time_raw']
        self.fif_data_info = self._mat_data['fif_data_info']
        self.sampling_freq =   self.fif_data_info[0][0][1][0][0][4][0][0]
        self.highpass = self.fif_data_info[0][0][1][0][0][5][0][0] 
        self.lowpass = self.fif_data_info[0][0][1][0][0][6][0][0]
        #channels
        channels_list = self.fif_data_info[0][0][1][0][0][7][0]
        self.channels = []
        keys = ['scanno', 'logno', 'kind', 'range', 'cal', 'coil_type', 'loc', 'coil_trans', 'eeg_loc', 'coord_frame', 'unit', 'unit_mul', 'ch_name']
        
        for n in range(len(channels_list)):
            values =  channels_list[n]
            # Creating the dictionary
          
            data_dict = dict(zip(keys, values))
            self.channels.append(data_dict)
            
        #Unnest variables - 

            
        for channel in self.channels:
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
        #Other data?
        #dipole_sleep_2
        self.info = {}
        for key, value in self._mat_data.items():
            if key not in ('__header__', '__version__', '__globals__','data_raw', 'fif_data_info', 'time_raw'):
                self.info[key] = value
                
        #Spikes
        self.spikes = []
        self.get_spikes()
                
        
                
    def get_mne(self):
        #delete old temporary file
        if self._fif_file != None:
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
        self._fif_file = self.filename+'.tmp.raw.fif'
        MEG_raw.save(self._fif_file, overwrite=True)
        raw = mne.io.read_raw(self._fif_file)
        #format and set info
        for n in range(len(self.channels)):
            raw.info['chs'][n]['logno'] = self.channels[n]['logno']
            raw.info['chs'][n]['range'] = self.channels[n]['range']
            raw.info['chs'][n]['cal'] = self.channels[n]['cal']
            raw.info['chs'][n]['coil_type'] = self.channels[n]['coil_type']
            raw.info['chs'][n]['loc'] = self.channels[n]['loc']
            raw.info['chs'][n]['unit'] = self.channels[n]['unit']
        return raw
    
    def get_spikes(self):
        if 'Dipole_sleep_2' in self.info:
            self.spikes = []
            
            for n in range(len(self.info['Dipole_sleep_2'][0])):
                spike = {}
                spike['dipole'] = self.info['Dipole_sleep_2'][0][n][0][0][0]
                spike['begin'] = self.info['Dipole_sleep_2'][0][n][1][0][0]
                spike['end'] = self.info['Dipole_sleep_2'][0][n][2][0][0]
                spike['r0'] = self.info['Dipole_sleep_2'][0][n][3]
                spike['rd'] = self.info['Dipole_sleep_2'][0][n][4]
                spike['Q'] = self.info['Dipole_sleep_2'][0][n][5]
                spike['goodness'] = self.info['Dipole_sleep_2'][0][n][6][0][0]
                spike['errors_computed'] = self.info['Dipole_sleep_2'][0][n][7][0][0]
                spike['noise_level'] = self.info['Dipole_sleep_2'][0][n][8][0][0]
                spike['single_errors'] = self.info['Dipole_sleep_2'][0][n][9]
                spike['error_matrix'] = self.info['Dipole_sleep_2'][0][n][10]
                spike['conf_volume'] = self.info['Dipole_sleep_2'][0][n][11][0][0]
                spike['khi2'] = self.info['Dipole_sleep_2'][0][n][12][0][0]
                spike['prob'] = self.info['Dipole_sleep_2'][0][n][13][0][0]
                spike['noise_est'] = self.info['Dipole_sleep_2'][0][n][14][0][0]
                self.spikes.append(spike)
        
        
    def __del__(self):
        #erase temporary file
        os.remove(self._fif_file)
            

            
        
            