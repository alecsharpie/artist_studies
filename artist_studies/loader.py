import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.image import resize
from PIL import Image
from tqdm import tqdm


class ArtLoader:
    """Load in art from artists as an array, batched and ready for tf"""

    def __init__(self):
        pass

    def get_image_single(self, path_to_image):
        """read in a single image, as an np array with dataset dim"""
        img = load_img(path_to_image).convert('RGB')
        return np.expand_dims(img, 0)

    def get_all_from_artist(self, artist_name):
        """read in images from the prompts:
        'A beautiful painting of a waterlily pond, {artist}, Trending on artstation',
        'A beautiful painting of a building in a serene landscape, {artist}, Trending on artstation'
        for a particluar artist"""
        all_img_paths = os.listdir(f'{self.data_path}/{artist_name}')
        pond_img_list = [
            load_img(f'{self.data_path}/{artist_name}/{img_path}').convert(
                'RGB') for img_path in all_img_paths
            if img_path.endswith('png')
        ]
        return np.array(pond_img_list)

    def get_image_folders(self,
                          data_path,
                          exclude_list,
                          num_artists=None,
                          preprocessor=None,
                          shape=(224, 224),
                          scale = None):
        """set the path to the data directory
        loop through all artists and load images
        optionally process, eg for clip"""
        self.data_path = data_path
        self.preprocessor = preprocessor

        if scale:
            # invert scale
            scale = 1 / scale

        self.art_list = []
        self.array_art_list = []
        self.preprocessed_art_list = []
        self.artist_list = []

        if self.data_path[-1] == '/':
            self.data_path = self.data_path[:-1]

        exclude_list.append('.DS_Store')

        if 'batch' in self.data_path:
            self.all_artists_paths = [
                artist for artist in os.listdir(self.data_path)
                if artist not in exclude_list
            ]
        else:
            self.all_artists_paths = []
            batches = os.listdir(self.data_path)
            for batch in batches:
                for artist_path in os.listdir(f'{self.data_path}/{batch}'):
                    if artist_path not in exclude_list:
                        self.all_artists_paths.append(f'{batch}/{artist_path}')

        if num_artists:
            self.all_artists_paths = self.all_artists_paths[:num_artists]
        for artist in tqdm(self.all_artists_paths):

            for art in self.get_all_from_artist(artist):
                if scale:
                    shape[0] = int(art.size[0] / scale)
                    shape[1] = int(art.size[1] / scale)

                # which is faster resize array or resize image

                self.artist_list.append(artist)
                self.array_art_list.append(
                    resize(img_to_array(art), [shape[0], shape[1]]))
                if preprocessor:
                    img = self.preprocessor(
                        art.resize([shape[0], shape[1]], Image.ANTIALIAS))
                    self.preprocessed_art_list.append(img)
                else:
                    self.art_list.append(art.resize([shape[0], shape[1]], Image.ANTIALIAS))

        if preprocessor:
            self.preprocessed_art_list = np.stack(self.preprocessed_art_list)
        else:
            self.art_list = np.array(self.art_list, dtype=object)
        self.artist_list = np.array(self.artist_list, dtype=object)
        self.array_art_list = np.array(self.array_art_list)
