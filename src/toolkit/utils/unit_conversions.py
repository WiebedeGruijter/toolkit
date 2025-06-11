import numpy as np
from pathlib import Path

def convert_file(filepath: Path, savepath: Path, conversion_factor=1e-3):
    '''Converts flux to specific intensity, and applied a conversion factor for unit conversion. 
    Saves the new file to current directory.
    
    Default conversion factor is from ergs/cm^2/s/nm to J/m^2/s/nm, which is 1e-3.
    '''
    data = np.genfromtxt(filepath, comments='#')

    flux_ergs_cm_2_s_nm= data[:,1]

    flux_j_m_2_s_nm = flux_ergs_cm_2_s_nm*conversion_factor

    # Convert to specific intensity
    flux_j_m_2_s_nm_sr = flux_j_m_2_s_nm / np.pi

    new_data = data.copy()
    new_data[:,1] = flux_j_m_2_s_nm_sr

    header_text = 'wavelength[nm]    specific_intensity[J/s/m^2/nm/sr]'

    np.savetxt(savepath / 'HD_189_SI.txt', new_data, delimiter='\t', fmt='%.2f', header=header_text)