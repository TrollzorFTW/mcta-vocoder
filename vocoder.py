import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter,buttord, lfilter, freqz,find_peaks,square
from scipy.fftpack import fft


class vocoder():


    # setting up default values for vocoder
    def __init__(self):

        self.M = 16
        self.wav = './vocale.wav'
        self.frames_no = 200
        self.duration = 20*pow(10,-3)



    # reads audio with help from soundfile library (https://pypi.org/project/PySoundFile/)
    def read_audio(self):

        # extracts data from audio file
        data, fs = sf.read('vocale.wav')
        self.fs = fs
        self.nyq = self.fs/2
        self.data = data

        # compute useful values as total number of samples, number of samples per frame, etc
        samples_no = len(data)
        self.frame_samples = math.floor(samples_no/self.frames_no) + 1
        diff = abs(self.frames_no*self.frame_samples-samples_no)

        # padding until the division has an integer result
        self.data = np.append(self.data,[0]*diff)
        self.samples_no = len(self.data)




    # Asks user input for variable M
    def set_M(self):

        response = input("Current M value is {}. Do you want to change it? [Y/N]  ".format(self.M))

        if response.upper() == 'Y':

            value = int(input("Enter value for M:"))
            self.M = value

        else:
            return




    # sets bandwidth for BPF filters
    def set_bandwidth(self):

        # creates a bandwidth variable for the vocoder object
        self.band = 4000/self.M
        



    # extracts and returns b and a parameters for Butterworth BPF
    def bandpass(self,low,high,order,analog=False):
        
        # computes parameters b and a using butter function from scipy.signal
        b,a = butter(order,[low,high],btype='band')

        # returns parameters
        return b,a




    # filters the signal with a Butterworth BPF and returns the output
    def bandpass_filter(self,low,high,frame,order=5):

        # computes parameters b and a with the help of bandpass function
        b,a = self.bandpass(low,high,order=order)

        # filters the signal using lfilter function from scipy.signal library
        y = lfilter(b,a,frame)

        # returns filtered signal
        return y       




    # splitting audio file in 200 frames
    def set_frames(self):

        # initialize a list to store the frames
        frames = [] 

        # parses through each frame
        for i in range(self.frames_no):

            # setting the number of samples limits
            low = i*self.frame_samples
            high = (i+1)*self.frame_samples - 1
            
            # extracted a frame from the initial data
            frame = self.data[low:high+1]

            # store the frame in the list
            frames.append(frame)

        # creates a list variable for the vocoder object to store the frames
        self.frames = frames




    # computing powers for every frame
    def set_powers(self):

        # initialize a list to store the power for each frame
        powers = []

        # parses through each frame
        for frame in self.frames:

            # computes sum of every element squared
            sum_frame = sum([el**2 for el in frame])

            # divides by the number of elements
            power = sum_frame/len(frame)

            # store the result in a list
            powers.append(power)

        # creates a list variable for the vocoder object to store the powers 
        self.powers = powers





    # using power, detects if each frame is voiced or unvoiced and returns 1 or 0 respectively,stored in a list
    def voice_recognition(self):

        # initialize a list to store the status of the frame (voiced/unvoiced)
        is_voice = []

        # initialize delay variable
        counter = 0

        # parse through each frame
        for i in range(self.frames_no):

            # initialize limits
            if i == 0:
                Emax = self.powers[i]
                Emin = 0

            # update limits
            if Emax < self.powers[i]:
                Emax = self.powers[i]

            elif Emin > self.powers[i] and self.powers[i] > 0:
                Emin = self.powers[i]

            # compute H and TH
            H = (Emax - Emin)/Emax 
            TH = H*Emin + (1-H)*Emax
            
            # check if the frame is voiced or not by comparing with the threshold. Note: counter was added to generate a little delay so the recognition is more accurate
            if self.powers[i] > TH:
                voice = 1
                counter = 3
            
            if counter == 0 and self.powers[i] < TH:
                voice = 0
            elif counter > 0 and self.powers[i] < TH:
                voice = 1
                counter = counter - 1
            
            # stored the result in a list
            is_voice.append(voice)

            # update lower limit
            Emin = Emax - Emax*0.8
        
        # create a list variable for the vocoder object to store the status of the frame (voiced/unvoiced)
        self.is_voice = is_voice




    # computes periods for voiced frames
    def set_periods(self):

        # initialize a list to store the periods for each frame
        period = []

        # parse through each frame
        for i in range(self.frames_no):

            # compute the period only for the voiced frames.
            if self.is_voice[i] == 1:

                # filter out false-positives (frames that don't have a max peak over 0.4 are not voiced - figured out through testing)
                if max(self.frames[i]) > 0.4:

                    # find peaks with help from find_peaks function from scipy.signal library
                    peaks, _ = find_peaks(self.frames[i])

                    # calculate the sample difference between 2 adjacent peaks 
                    diff = [peaks[i + 1] - peaks[i] for i in range(len(peaks)-1) if peaks[i+1] - peaks[i] > 10]

                    # if all differences between 2 adjacent peaks are less than 10 samples, then it's unvoiced (another false-positive filtering)
                    if len(diff) == 0:
                        period.append(0)

                    # finally, if the frame is indeed voiced, we calculate the period of the frame and store it in a list 
                    else:
                        frame_period = sum(diff)/len(diff)
                        period.append(frame_period)
                

                # if it's false positive, then it's unvoiced, so we set period to be 0
                else:
                    period.append(0)


            # if it's not voiced, then we set period to be 0 (we don't need it in this case anyway)
            else:
                period.append(0)

        # the period list we computed above, it's measured in samples, so we have to convert it in seconds
        period = [samples/self.fs for samples in period]


        # create a list variable for the vocoder object to store the periods for each frame 
        self.periods = period



    # filters each frame through M different BPF channels
    def filter(self):

        # initialize a list to store frames passed through each channel (will be a 200x20 array)
        filtered_frames = []

        # parse through each frame
        for i in range(self.frames_no):
            
            # initialize/reset a list to store the frame through each channel
            filtered_frame = []

            # parse through each channel
            for j in range(self.M):   

                # set bandwidth limits
                if j == 0:
                    low = 0.01
                    high = self.band/self.nyq

                elif j == self.M-1:
                    low = j*self.band/self.nyq
                    high = (j+1)*self.band/self.nyq - 0.01

                else:
                    low = j*self.band/self.nyq
                    high = (j+1)*self.band/self.nyq

                # filters the signal
                y = self.bandpass_filter(low,high,self.frames[i])

                # append the frame filtered from the current BPF filter
                filtered_frame.append(y)
        
            # store the frame passed through all M channels 
            filtered_frames.append(filtered_frame)


        # create a list variable for the vocoder object to store the filtered frames for each frame
        self.filtered_frames = filtered_frames



    # compute energies for each filtered frames
    def set_energies(self):

        # initialize list to store the energy for each filtered frame (will be a 200x20 array)
        energies = []

        # parse through each frame
        for frame in self.filtered_frames:

            # initialize/reset list to store the energy from each BPF channel
            frame_energy = []

            # parse through each channel
            for i in range(self.M):     

                # computes energy       
                y_energy = sum([el**2 for el in frame[i]])

                # stores value in a list
                frame_energy.append(y_energy)

            # stores all energies from each channel for a signle frame
            energies.append(frame_energy)

        # create a list variable for the vocoder object to store the energy for each filtered frame
        self.energies = energies



    def multiply_signal(self):

        # initialize list that stores a list of each multiplied frame
        altered_frames = []

        # parse through each frame
        for i in range(self.frames_no):
            
            # initialize/reset list that stores each frame after the multiplication
            altered_frame = []

            # initialize a variable containing the current filtered frame
            filtered_frame = self.filtered_frames[i]

            # generate PWM signal only if the frame is voiced
            if self.is_voice[i]:

                # generate pulse train signal with period self.periods[i] and add to the frame of the coded signal
                t = np.linspace(0,self.duration,self.samples_no,endpoint=False)
                sig = np.sin(2*np.pi*t)

                # if the frame is voiced, calculate the frequency of the frame
                if self.periods[i]:
                    freq = 1/self.periods[i]

                # if the frame is voiced but it has period = 0 (untreated false-positives), we set it to a default value
                else:
                    freq = self.fs/5
                
                # generate pwm signal
                pwm = square(2*np.pi*freq*t,duty=(sig+1)/2)

                # parse through each channel
                for j in range(self.M):

                    # multiply element-wise the current filtered frame
                    altered_samples = [a*b for a,b in zip(filtered_frame[j],pwm)]

                    # store the result in a list
                    altered_frame.append(altered_samples)

            # if the signal is not voiced, generate a random noise signal
            else:

                # generate noise signal and add to the frame of the coded signal
                noise_signal = np.random.random(size=self.samples_no)
                
                # parse through each channel
                for j in range(self.M):

                    # multiply element-wise the current filtered frame
                    altered_samples = [a*b for a,b in zip(filtered_frame[j],noise_signal)]
                    
                    # store the result in a list
                    altered_frame.append(altered_samples)

            # store the altered frame in a list
            altered_frames.append(altered_frame)

            # create a list variable for the vocoder object to store all the altered frames
            self.altered_frames = altered_frames



    # filters the altered signal (multiplied with pwm/noise) through all M channels again
    def decoder_filter(self):

        ###################################################
        # It's the same procedure as in the coder section #
        ###################################################
        
        decoded_filtered_frames = []

        for i in range(self.frames_no):

            filtered_altered_frames = []

            for j in range(self.M):

                if j == 0:
                    low = 0.01
                    high = self.band/self.nyq

                elif j == self.M-1:
                    low = j*self.band/self.nyq
                    high = (j+1)*self.band/self.nyq - 0.01

                else:
                    low = j*self.band/self.nyq
                    high = (j+1)*self.band/self.nyq

                filtered_altered_frame = self.bandpass_filter(low,high,self.altered_frames[i][j])

                filtered_altered_frames.append(filtered_altered_frame)

            decoded_filtered_frames.append(filtered_altered_frames)

        self.decoded_filtered_frames = decoded_filtered_frames


    

    # merge all the frames from all channel outputs
    def merge_frames(self):

        # initialize a list to store the information from all the frames
        full_audio = []

        # parse through each channel
        for i in range(self.M):

            # initialize a temporary list to store the concatenation of all the frames
            temp = []

            # parse through each frame
            for j in range(self.frames_no):

                # merge all the elements in one list
                temp = np.concatenate((temp, self.decoded_filtered_frames[j][i]), axis=None)

            # store the merged frames in a list
            full_audio.append(temp)


        # create a list variable for the vocoder object to store all the audio signals from all channel outputs
        self.full_audio = full_audio
                
    


    def add_decoded_signals(self):
        
        # initialize a list to store the processed signal through the vocoder
        output_audio = []
        
        # initialize a zeros list with number of samples as length
        for i in range(self.samples_no):
            output_audio.append(0)

        # parse through each channel
        for i in range(self.M):
            
            # sum element-wise each channel
            output_audio = [a+b for a,b in zip(self.full_audio[i],output_audio)]


        # create a list variable for the vocoder object to store the processed signal through the vocoder
        self.output_audio = output_audio





    def coder(self):

        # reads audio file
        self.read_audio()

        # Asks if you want to change variable M
        self.set_M()

        # calculates bandwidth according to variable M
        self.set_bandwidth()

        # split signal in frames
        self.set_frames()

        # compute powers for each channel
        self.set_powers()

        # estimate voiced/unvoiced
        self.voice_recognition()

        # compute main period for voiced
        self.set_periods()

        # filter all the frames through all the banks
        self.filter()

        # compute energies after processing each frame in bandpass filter banks
        self.set_energies()





    def decoder(self):

        # multiplies the signal from the coder output with PWM/AWGN according to voiced status
        self.multiply_signal()

        # filters all the altered frames through all the banks 
        self.decoder_filter()

        # merge all the frames from each channel output
        self.merge_frames()

        # sums element-wise all the signals from each channel output
        self.add_decoded_signals()

    
    def export_output(self,filename):
        
        print("Exporting processed audio file...")
        sf.write(filename+'_M={}.wav'.format(self.M),self.output_audio,self.fs)


    def run(self):

        print("Processing audio file...")
        self.coder()
        self.decoder()
