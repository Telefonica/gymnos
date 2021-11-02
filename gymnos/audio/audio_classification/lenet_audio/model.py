#
#
#   Model
#
#

import math
import torch
import torch.nn as nn
import torchaudio


class SerializableModule(nn.Module):

    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.zip')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class LeNetAudio(SerializableModule):

    def __init__(self, num_classes, window_size=24000):
        super().__init__()

        # Mel-Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=320,
            hop_length=160,
            n_mels=40
        )

        self.features = nn.Sequential(
            nn.InstanceNorm2d(1),
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        f_bins = 40
        t_bins = int(window_size/160) + 1
        f_r = self.__get_size(f_bins)
        t_r = self.__get_size(t_bins)
    
        self.classifier = nn.Sequential(     
            nn.Linear(32 * f_r * t_r, 100),
            nn.Dropout(0.5),
            nn.ReLU(),  
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # [b, ch, t]
        x = self.mel_spectrogram(x) # [b, ch, f_b, t_b]
        x = self.features(x) 
        x = x.view(x.shape[0], -1) # [b, ch*f_b*t_b]
        x = torch.sigmoid(self.classifier(x)) # [b, 10]
        return x

    def __get_size(self, in_dim):
        return int(math.floor((((in_dim-4)/2)-4)/2))