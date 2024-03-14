import os
import re

def replace_date(file_path):
    files = os.listdir(file_path)

    for file_name in files:
        
        if "fusion" in file_name:
            match = re.match(r'(\d+)_(\d+)_fusion', file_name)
            
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_IRMpre.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for "fusion"): {file_name}')
        
        # Remplace la date par 'IRMpost_4mois' si 'sag_t1_mprage_3d_gado_mpr_tra' est dans le nom du fichier
        elif "sag_t1_mprage_3d_gado_mpr_tra" in file_name:
            match = re.match(r'(\d+)_(\d+)_sag_t1_mprage_3d_gado_mpr_tra', file_name)
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_IRMpost_4mois.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for "XXX"): {file_name}')
        
        elif 'tomotherapy_planned_dose' in file_name:
            match = re.match(r'(\d+)_(\d+)_tomotherapy_planned_dose', file_name)
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_RTDOSE.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for date replacement): {file_name}')
        
        # Remplace 'doses_eclipse" par RTDOSE
        elif "doses_eclipse" in file_name:
            match = re.match(r'(\d+)_(\d+)_doses_eclipse', file_name)
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_RTDOSE.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for date replacement): {file_name}')
                
        elif "kvct" in file_name:
            match = re.match(r'(\d+)_(\d+)_kvct', file_name)
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_CT.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for date replacement): {file_name}')
        
        elif "crane_2mm" in file_name:
            match = re.match(r'(\d+)_(\d+)_crane_2mm', file_name)
            if match:
                patient_id, date = match.groups()
                new_file_name = f'{patient_id}_CT.nii'
                perform_rename(file_path, file_name, new_file_name)
            else:
                print(f'Skipped (no match for date replacement): {file_name}')
                
        

def perform_rename(file_path, old_file_name, new_file_name):
    old_file_path = os.path.join(file_path, old_file_name)
    new_file_path = os.path.join(file_path, new_file_name)
    
    os.rename(old_file_path, new_file_path)
    print(f'Renamed: {old_file_path} -> {new_file_path}')

# Specify the path to the folder containing the files
folder_path = "//s-usr/usr/M105411/Tampon/Preprocessing_nii"

# Call the function to replace dates based on conditions
replace_date(folder_path)