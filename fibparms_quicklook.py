#fibparms_quicklook.py

import numpy as np

#fibparms = np.load('D:\\cloudstor\\codestore\\dataredux\\mq_spectrograph\\data\\fibparms_by_ord.npy', encoding= 'bytes').item()
fibparms = np.load('D:\\cloudstor\\codestore\\dataredux\\mq_spectrograph\\data\\fibparms_by_ord.npy', encoding= 'latin1').item()

print(fibparms)

