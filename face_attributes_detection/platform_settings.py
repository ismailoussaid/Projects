import platform

host = platform.node()
aligned = False
root_linux = "/dev/shm/data/celeba_files/"
root_windows = "C:/Users/Ismail/Documents/Projects/celeba_files/"
root_scaleway = '/root/data/celeba_files/'

if host == 'castor' or host == 'altair':  # Enrico's PCs
    root_path = root_linux
elif host == 'DESKTOP-AS5V6C3':  # Ismail's PC
    root_path = root_windows
elif host == 'scw-zealous-ramanujan' or host == 'scw-cranky-jang':
    root_path = root_scaleway
else:
    raise RuntimeError('Unknown host')

global_path = root_path

if aligned:
    images_path = root_path + "well_cropped_images/"
    attributes_path = root_path + "celeba_csv/list_attr_celeba_aligned.csv"
else:
    images_path = root_path + "cropped_images/"
    attributes_path = root_path + "celeba_csv/list_attr_celeba.csv"


