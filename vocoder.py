import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter,buttord, lfilter, freqz,find_peaks,square
from scipy.fftpack import fft


class vocoder():


    def __init__(self):
        self.M = 16
        self.wav = './vocale.wav'
        self.frames_no = 200



    def read_audio(self):
        data, fs = sf.read('vocale.wav')
        self.fs = fs
        self.nyq = self.fs/2
        self.data = data


    def set_bandwidth(self):
        self.band = 4000/self.M
        


    def bandpass(self,low,high,order,analog=False):
        b,a = butter(order,[low,high],btype='band')
        return b,a



    def bandpass_filter(self,low,high,frame,order=5):
        b,a = self.bandpass(low,high,order=order)
        y = lfilter(b,a,frame)
        return y       


    def set_frames(self):
        self.read_audio()
        self.set_bandwidth()
        frames_no = 200
        samples_no = len(self.data)
        frame_samples = math.floor(samples_no/frames_no) + 1

        frames = []
        diff = abs(frames_no*frame_samples-samples_no)
        self.data = np.append(self.data,[0]*diff)

        for i in range(frames_no):
            low = i*frame_samples
            high = (i+1)*frame_samples - 1
            frame = self.data[low:high+1]
            frames.append(frame)

        self.frames = frames



    def set_powers(self):

        powers = []
        for frame in self.frames:

            sum_frame = sum([el**2 for el in frame])
            power = sum_frame/len(frame)
            powers.append(power)

        self.powers = powers



    def voice_recognition(self):

        is_voice = []
        counter = 0

        for i in range(self.frames_no):

            if i == 0:
                Emax = self.powers[i]
                Emin = 0

            if Emax < self.powers[i]:
                Emax = self.powers[i]

            elif Emin > self.powers[i] and self.powers[i] > 0:
                Emin = self.powers[i]

            H = (Emax - Emin)/Emax 
            TH = H*Emin + (1-H)*Emax
            
   
            if self.powers[i] > TH:
                voice = 1
                counter = 3
            
            if counter == 0 and self.powers[i] < TH:
                voice = 0
            elif counter > 0 and self.powers[i] < TH:
                voice = 1
                counter = counter - 1

            is_voice.append(voice)
            Emin = Emax - Emax*0.8
        
        self.is_voice = is_voice



    def set_periods(self):

        period = []
        for i in range(self.frames_no):
            if self.is_voice[i] == 1:
                if max(self.frames[i]) > 0.4:
                    peaks, _ = find_peaks(self.frames[i])
                    diff = [peaks[i + 1] - peaks[i] for i in range(len(peaks)-1) if peaks[i+1] - peaks[i] > 10]

                    if len(diff) == 0:
                        period.append(0)
                    else:
                        frame_period = sum(diff)/len(diff)
                        period.append(frame_period)
                else:
                    period.append(0)

            else:
                period.append(0)


        period = [samples/self.fs for samples in period]
        self.periods = period


    def filter(self):

        self.nyq = self.fs/2
        filtered_frames = []

        for i in range(self.frames_no):
            
            filtered_frame = []

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

                y = self.bandpass_filter(low,high,self.frames[i])

                filtered_frame.append(y)
        
            filtered_frames.append(filtered_frame)

        self.filtered_frames = filtered_frames


    def set_energies(self):

        self.nyq = self.fs/2

        energies = []

        for frame in self.filtered_frames:

            frame_energy = []

            for i in range(self.M):            
                y_energy = sum([el**2 for el in frame[i]])
                frame_energy.append(y_energy)

            energies.append(frame_energy)

        self.energies = energies


    def coder(self):
        self.read_audio()
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


    
    def add_signal(self):

        # set number of samples
        samples = 160

        # set the duration to 20 ms
        duration = 20*pow(10,-3)


        altered_frames = []

        for i in range(len(self.is_voice)):
        
            altered_frame = []

            # initialize a variable containing the current filtered frame
            filtered_frame = self.filtered_frames[i]

            if self.is_voice[i]:

                # generate pulse train signal with period self.periods[i] and add to the frame of the coded signal
                t = np.linspace(0,duration,samples,endpoint=False)
                sig = np.sin(2*np.pi*t)

                if self.periods[i]:
                    freq = 1/self.periods[i]
                else:
                    freq = self.fs/5

                pwm = square(2*np.pi*freq*t,duty=(sig+1)/2)

                for j in range(self.M):
                    altered_samples = [a*b for a,b in zip(filtered_frame[j],pwm)]
                    altered_frame.append(altered_samples)

            else:

                # generate noise signal and add to the frame of the coded signal
                noise_signal = np.random.random(size=samples)

                for j in range(self.M):
                    altered_samples = [a*b for a,b in zip(filtered_frame[j],noise_signal)]
                    altered_frame.append(altered_samples)


            altered_frames.append(altered_frame)
            self.altered_frames = altered_frames


    def decoder_filter(self):

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


    
    def merge_frames(self):

        full_audio = []

        # parsing each bpf channel
        for i in range(self.M):

            temp = []

            # merge all the elements in one list
            for j in range(self.frames_no):

                temp = np.concatenate((temp, self.decoded_filtered_frames[j][i]), axis=None)

            
            full_audio.append(temp)

        self.full_audio = full_audio
                
    

    def add_decoded_signals(self):
        
        output_audio = []
        
        for i in range(32000):
            output_audio.append(0)


        for i in range(self.M):

            output_audio = [a+b for a,b in zip(self.full_audio[i],output_audio)]

        self.output_audio = output_audio


    def decoder(self):
        self.add_signal()
        self.decoder_filter()
        self.merge_frames()
        self.add_decoded_signals()



    def run(self):
        self.coder()
        self.decoder()
        sf.write('output3.wav', self.output_audio, self.fs)
        






       
