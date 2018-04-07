import matplotlib
import matplotlib.pyplot as plt

def visualize(samples, pic_index):
    plt.imsave('out/{}.png'.format(str(pic_index).zfill(3)), samples)
