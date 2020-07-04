import glob, os
import imageio

filenames = []
for file in glob.glob("*.png"):
    filenames.append(file)

filenames.sort()

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
    
imageio.mimsave('ex01_polynomial_regression.gif', images, duration=0.25)