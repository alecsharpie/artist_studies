import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow.image import resize
from tensorflow.keras.utils import img_to_array
from tqdm import tqdm
import numpy as np


def show_xy_images(x_coords, y_coords, img_list, image_zoom=1):
    fig, ax = plt.subplots(1,1, figsize=(16, 16))
    artists = []
    for x_coord, y_coord, img in tqdm(zip(x_coords, y_coords, img_list)):
        c_img = resize(img, [50,50])
        img = OffsetImage(c_img, zoom=image_zoom)
        ab = AnnotationBbox(img, (x_coord, y_coord),
                            xycoords='data',
                            frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.stack([x_coords, y_coords], axis=1))
    ax.autoscale()
