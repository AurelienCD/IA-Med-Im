from math import *
import numpy as np
#import SimpleITK as sitk
#import sitkUtils as su
import pydicom
import sys, time, os

############################si il manque une biblio exemple scikit #################################
#slicer.util.pip_install("scikit-image")
#########################################################

data_directory = "//s-usr/usr/M105411/Tampon/Preprocessing_trie"
adresse_save_result="//s-usr/usr/M105411/Tampon/Preprocessing_nii"

def main(data_directory, adresse_save_result):
    timeInit        = time.time()
    Nimageouverte   = 0
    Nimagetraitees  = 0
    
   ########################################iterate trough subfolder##############
   
    directory_list  = []
    i               = 0 #number of sub folder
    
    for root, dirs, files in os.walk(data_directory):
        for subdirname in dirs:
            directory_list.append(os.path.join(root,subdirname))
            
    ################################partie principale du code
    
    for i in range(len(directory_list)):
        data_directory = directory_list[i].replace('\\','/')
        series_IDs     = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
        if not series_IDs:
            print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        else:
            for i,series_ID in enumerate(series_IDs):   
                Nimageouverte     = Nimageouverte+1
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID,useSeriesDetails=False) #useSeriesDetails ?
                try:
                    img_metadata = pydicom.read_file(series_file_names[0])  # Importation des metadata lié à l'image
                    if True : # img_metadata.Modality=='MR':#"PET TAP AC HD (AC)": #"[DetailWB_CTAC_2i-10s-PSF] Body"
                        try:
                            timeRMR1       = time.time()
                            Nimagetraitees = Nimagetraitees+1
                            series_reader  = sitk.ImageSeriesReader()
                            series_reader.SetFileNames(series_file_names)
                            img            = series_reader.Execute()  #importation de l'image
                            series_file_names  = series_file_names[0].split("/")
                            series_file_names1 = series_file_names[-1].split("\\")
                            name               = series_file_names[-3]+"_"+series_file_names[-2]+"_"+series_file_names1[-2]+".nii"
                            nameID             = series_file_names[-3]
                            #save_path          = adresse_save_result+"/"+series_file_names[-3]
                            save_path = os.path.join(adresse_save_result,name)
                            sitk.WriteImage(img,save_path) # Attention vérifier les dicom de l'image sinon .CopyInformation
                            timeRMR2               = time.time()
                            TimeForrunFunctionRMR2 = timeRMR2 - timeRMR1
                            print(u"La fonction de traitement s'est executée en " + str(TimeForrunFunctionRMR2) +" secondes")
                            print("\n")
                        except RuntimeError:
                            print ("--> Problème avec l'importation et/ou le traitement d'image")
                except RuntimeError:
                    print ("--> Probleme avec la lecture des metadata")
    print("\n")
    print("Nombre d'image total lue:"+str(Nimageouverte)+"\n")
    print("Nombre d'image total traité:"+str(Nimagetraitees)+"\n" )
    timefinal = time.time()
    TimeTotal = timefinal - timeInit
    print(u"Le traitement de l'ensemble des données c'est executé en " + str(TimeTotal) +" secondes")

##################################Execution du code############################################
###############################################################################################  
main(data_directory, adresse_save_result)
