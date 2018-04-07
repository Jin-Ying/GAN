import matplotlib
import matplotlib.pyplot as plt

def visualize(path, samples, pic_index):
    plt.imsave(path+'/{}.png'.format(str(pic_index).zfill(3)), samples)
