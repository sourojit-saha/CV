import os 
import imageio 
import glob


# path = os.getcwd() #+ "/results-log5-zmax-0-08" #where images are stored
# path = os.getcwd() + "/data/results/" #where images are stored
path = "/home/p43s/Downloads/cmu/cv/cv-hw/hw2/data/results/"
num_files = 510
i = 0

writer = imageio.get_writer('cv-hw2.mp4', fps=20)

while i<(num_files+1):
    if i!=435 and i!=436 and i!=437:
        # actualpath = ('{}/{:04d}.jpg'.format(path,i))#naming each image
        actualpath =path + str(i) + ".jpg"
        print(actualpath)
        im = imageio.imread(actualpath)
        writer.append_data(im)
        i=i+1
    else:
        i=i+1
writer.close()