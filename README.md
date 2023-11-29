# MEG-Tools
Tools for accessing, visualizing, and analyzing MEG 

## Requirements
MEG-Tools requires the installation of the MNE library. Visit [mne.tools](https://mne.tools) to download and install.

## Using the Tools
1. Download or clone this repository.
2. Copy a MEG file to the directory, such as "case_2225_with_spike_dipoles_sleep_2.mat"
3. Open the MEG-Example Notebook and change the name in `meg = MEG('case_2225_with_spike_dipoles_sleep_2.mat')`
4. Run all the code blocks.
5. You should see something like the following image for `mne_fif.plot(scalings={'mag': 3.0e-11})`
<img src="images/MEG-Pop-Up.png" width="350">

