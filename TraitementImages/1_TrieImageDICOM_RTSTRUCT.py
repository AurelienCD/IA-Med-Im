import os
import pydicom
from pydicom import dcmread

def clean_text(string):
    # clean and standardize text descriptions, which makes searching files easier
    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_") # replace everything with an underscore
    return string.lower()

def clean_folder_path(chemin_sortie):
    chemin_sortie=chemin_sortie.replace('\\','/')
    chemin_sortie=chemin_sortie.replace('<','_')
    chemin_sortie=chemin_sortie.replace('>','_')
    return chemin_sortie

def clean_filename(string):
    # clean and standardize text descriptions, which makes searching files easier
    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_") # replace everything with an underscore
    return string.lower()

def list_file_adresse(src):
    print('reading file list...')
    unsortedList = []
    for root, dirs, files in os.walk(src):
        for file in files:
            #if ".dcm" in file:# exclude non-dicoms, good for messy folders
            try:
                chemin_entre=os.path.join(root, file)
                #img_metadata=pydicom.read_file(chemin_entre,force=True) #test si dicom :)
                unsortedList.append(chemin_entre)
            except:
                pass
    print('%s files found.' % len(unsortedList))
    return unsortedList

def trie_fichiers_dicom(unsortedList,dst):
    i=0
    for dicom_loc in unsortedList:
        # read the file
        ds = pydicom.read_file(dicom_loc, force=True)
        i=i+1
        patientID = clean_text(ds.get("PatientID", "NA"))
        studyDate = clean_text(ds.get("StudyDate", "NA"))
        seriesDescription = clean_text(ds.get("SeriesDescription", "NA"))
        modality = ds.get("Modality","NA")
        studyInstanceUID = ds.get("StudyInstanceUID","NA")
        seriesInstanceUID = ds.get("SeriesInstanceUID","NA")
        instanceNumber = str(ds.get("InstanceNumber","0"))
        fileName = modality + "." + seriesInstanceUID + "." + instanceNumber
        fileName=clean_filename(fileName)
        fileName=fileName+ ".dcm"
        
        try:
            if not os.path.exists(os.path.join(dst, patientID)):
                chemin_sortie=os.path.join(dst, patientID)
                chemin_sortie=clean_folder_path(chemin_sortie)
                os.makedirs(chemin_sortie)

            if not os.path.exists(os.path.join(dst, patientID, studyDate)):
                chemin_sortie=os.path.join(dst, patientID, studyDate)
                chemin_sortie=clean_folder_path(chemin_sortie)
                os.makedirs(chemin_sortie)

            
            if not os.path.exists(os.path.join(dst, patientID, studyDate, seriesDescription)):
                chemin_sortie=os.path.join(dst, patientID, studyDate, seriesDescription)
                chemin_sortie=clean_folder_path(chemin_sortie)
                os.makedirs(chemin_sortie)

            try :
                print('Saving out file: %s - %s - %s.' % (patientID, studyDate, seriesDescription ))
                chemin_sortie=os.path.join(dst, patientID, studyDate, seriesDescription, fileName)
                chemin_sortie=clean_folder_path(chemin_sortie)
                os.renames(dicom_loc, chemin_sortie)
            except:
                print(str(dicom_loc))
                print('Erreur ?')
        except:
            print('Erreur lors de la creation du fichiers')

print('done.')


# user specified parameters
src ="//s-usr/usr/M105411/Tampon/Preprocessing"  ##dossier importation
dst = "//s-usr/usr/M105411/Tampon/Preprocessing_trie" #dossier exportation
###############################
unsortedList=list_file_adresse(src)
########################
trie_fichiers_dicom(unsortedList,dst)