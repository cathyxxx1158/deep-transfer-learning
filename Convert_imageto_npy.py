import os
from matplotlib import pyplot as plt
import cv2
import pydicom
import numpy as np

def turn_covid_to_npy(split):
    root_for_covid = 'F:/MLdata/Covid/'+split+'/'
    print(root_for_covid)
    for root, dirs, files in os.walk(root_for_covid):
        for f in sorted(files):
            fs= os.path.splitext(f)
            fn= fs[0]
            # if f=='3935110146332427995.dcm':
            #     print("d")
            #     eachpath = os.path.join(root, f)
            #     ds = pydicom.read_file(eachpath)
            #     plt.imshow(ds.pixel_array, cmap='gray')
            #     plt.show()
            eachpath = os.path.join(root, f)
            ds = pydicom.read_file(eachpath)
            pix = np.stack((cv2.resize(ds.pixel_array, (224,224)),) * 3, axis=-1)
            np.save('F:/MLdata/Covid_npy/'+split+'/'+fn+'.npy', pix / 255)
            # cv2.imwrite('D:/dataset/Covidjpg/test/'+f+'.png',ds.pixel_array*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            print(f)
def turn_other_to_npy(split):
    root_for_PNEUMONIA = 'F:/MLdata\Other/'+split+'/PNEUMONIA/'
    root_for_NORMAL = 'F:/MLdata/Other/'+split+'/NORMAL/'
    root_for_VIRUS = 'F:/MLdata/Other/' + split + '/VIRUS/'
    i=0
    for eachpath in [root_for_PNEUMONIA,root_for_NORMAL,root_for_VIRUS]:
        i=i+1
        for root, dirs, files in os.walk(eachpath):
            for f in sorted(files):
                fs = os.path.splitext(f)
                fn = fs[0]
                # if f=='3935110146332427995.dcm':
                #     print("d")
                #     eachpath = os.path.join(root, f)
                #     ds = pydicom.read_file(eachpath)
                #     plt.imshow(ds.pixel_array, cmap='gray')
                #     plt.show()
                eachpath = os.path.join(root, f)
                jpg = cv2.imread(eachpath, 0)
                pix = np.stack((cv2.resize(jpg, (224, 224)),) * 3, axis=-1)
                if i==1:
                    np.save('F:/MLdata/Other_npy/'+split+'/PNEUMONIA/'+fn+'.npy',pix / 255)
                elif i==2:
                    np.save('F:/MLdata/Other_npy/' + split + '/NORMAL/' + fn + '.npy', pix / 255)
                else:
                    np.save('F:/MLdata/Other_npy/' + split + '/VIRUS/' + fn + '.npy', pix / 255)
                # cv2.imwrite('D:/dataset/Covidjpg/test/'+f+'.png',ds.pixel_array*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                print(f)

if __name__=='__main__':
    #turn_covid_to_npy("test")
    # turn_covid_to_npy("train")
    # turn_covid_to_npy("val")
    turn_other_to_npy("train")
    turn_other_to_npy("test")
    turn_other_to_npy("val")
