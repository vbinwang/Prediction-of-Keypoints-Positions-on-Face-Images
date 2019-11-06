#!/usr/bin/env python
'''Preprocessing scripts
'''
import abc
import random

import numpy as np
from skimage import exposure, feature
import skimage.filters


class ImagePreprocessor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def _process_one_image(self, image, coords):
        pass

    def process(self, all_images, all_coords):
        assert len(all_images) == len(all_coords)
        assert all_images.shape[1] == 1

        new_images = np.empty(all_images.shape, dtype=np.float32)
        new_coords = np.empty(all_coords.shape, dtype=np.float32)
        for i in range(len(all_images)):
            new_image, new_coord = self._process_one_image(
                all_images[i][0], all_coords[i])

            assert new_image is not None and new_coords is not None

            new_images[i][0], new_coords[i] = new_image, new_coord

        # Merge the new and old images
        return new_images, new_coords


class RotateFlip(ImagePreprocessor):
    '''Rotates and flips images randomly
    '''
    def __init__(self):
        super(RotateFlip, self).__init__()

    @staticmethod
    def __split(coords):
        x_coords = coords[0:len(coords):2]
        y_coords = coords[1:len(coords):2]
        return x_coords, y_coords

    @staticmethod
    def __merge(x_coords, y_coords):
        return [j for i in zip(x_coords, y_coords) for j in i]

    @staticmethod
    def __flip(image, coords, axis):
        flip_horizontal = {
            "h": True,
            "v": False
        }[axis]

        x_coords, y_coords = RotateFlip.__split(coords)

        new_image = image
        if flip_horizontal:
            new_image = np.fliplr(image)
            x_coords = image.shape[0] - x_coords
            # y_coords unaffacted
        else:
            new_image = np.flipud(image)
            # x_coords unaffected
            y_coords = image.shape[1] - y_coords

        return new_image, RotateFlip.__merge(x_coords, y_coords)

    @staticmethod
    def __rotate90(image, coords, direction):
        rotate_clockwise = {
            "cw": True,
            "ccw": False
        }[direction]

        x_coords, y_coords = RotateFlip.__split(coords)

        new_image = image
        if rotate_clockwise:
            new_image = np.rot90(image, 3)
            tmp_x_coords = x_coords
            x_coords = image.shape[0] - y_coords
            y_coords = tmp_x_coords
        else:
            new_image = np.rot90(image, 1)
            tmp_y_coords = y_coords
            y_coords = image.shape[1] - x_coords
            x_coords = tmp_y_coords

        return new_image, RotateFlip.__merge(x_coords, y_coords)

    def _process_one_image(self, image, coords):
        assert image.shape == (96, 96)

        draw = random.randint(0, 1)
        if draw == 0:
            return RotateFlip.__flip(image, coords, "h")
        else:
            return image, coords

        # draw = random.uniform(0, 1)
        # if draw < 1./8.:
        #     new_image, new_coords = RotateFlip.__rotate90(
        #         image, coords, "cw")
        # elif draw >= 1./8. and draw < 1./4.:
        #     new_image, new_coords = RotateFlip.__rotate90(
        #         image, coords, "ccw")
        # elif draw >= 1./4. and draw < 3./8.:
        #     new_image, new_coords = RotateFlip.__flip(
        #         image, coords, "h")
        # elif draw >= 3./8. and draw < 1./2.:
        #     new_image, new_coords = RotateFlip.__flip(
        #         image, coords, "v")
        # else:
        #     new_image = image
        #     new_coords = coords
        # return new_image, new_coords


class ContrastEnhancer(ImagePreprocessor):
    '''Enhances contrast in the image
    '''
    def __init__(self, enhancement_type, placement='add_channel'):
        assert placement in ['add_channel', 'replace']
        super(ContrastEnhancer, self).__init__(placement)
        self.__filter_fn = {
            "equalize": ContrastEnhancer.__equalize_hist,
            "roberts": skimage.filters.roberts,
            "sobel": skimage.filters.sobel,
            "canny": ContrastEnhancer.__canny,
        }[enhancement_type]

    @staticmethod
    def __equalize_hist(image):
        return skimage.exposure.equalize_hist(image) * 255

    @staticmethod
    def __canny(image):
        return skimage.feature.canny(image, sigma=1.5)

    def _process_one_image(self, image, coords, channel_list):
        new_image = image
        for i in channel_list:
            new_image[i] = self.__filter_fn(image[i])
        return(new_image, coords)


# def rotate(face, kp, clockwise=True):
#     if clockwise:
#         face_rot = np.rot90(face, 3)
#         x_kp, y_kp = split_kp(kp)
#         return face_rot, merge_kp(face.shape[0] - y_kp, x_kp)
#     else:
#         face_rot = np.rot90(face, 1)
#         x_kp, y_kp = split_kp(kp)
#         return face_rot, merge_kp(y_kp, face.shape[1] - x_kp)


# def enhance_contrast(face):
#     # hist = np.histogram(face, bins=np.arange(0, 256))
#     glob = exposure.equalize_hist(face) * 255
#     return glob

if __name__ == "__main__":
    import fileio
    faces = fileio.FaceReader(
        "../data/training.csv", "../data/training.pkl.gz",
        fast_nrows=10)
    data = faces.load_file()
    rf = RotateFlip()
    res_X, res_Y = rf.process(data['X'][0:1], data['Y'][0:1])

    import code
    code.interact(local=locals())



### TEST ###

# faces = fileio.FaceReader("../data/training.csv", "../data/training.pkl.gz", 
#   fast_nrows=10)
# faces.load_file()
# raw_data = faces.get_data()
# to_keep = ~(np.isnan(raw_data['Y']).any(1))
# X = raw_data['X'][to_keep]
# Y = raw_data['Y'][to_keep]

# face = X[9][0]
# kp = Y[9]
# x_kp, y_kp = split_kp(kp)

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

# ax1.imshow(face, cmap=plt.cm.Greys_r)
# ax1.plot(x_kp, y_kp, 'rx')
# ax1.axis('off')
# ax1.set_title('Original', fontsize=10)

# ax2.imshow(roberts(face), cmap=plt.cm.Greys_r)
# ax2.plot(x_kp, y_kp, 'rx')
# ax2.axis('off')
# ax2.set_title('Roberts', fontsize=10)

# ax3.imshow(sobel(face), cmap=plt.cm.Greys_r)
# ax3.plot(x_kp, y_kp, 'rx')
# ax3.axis('off')
# ax3.set_title('Sobel', fontsize=10)

# ax4.imshow(feature.canny(face, sigma=1.5), cmap=plt.cm.Greys_r)
# ax4.plot(x_kp, y_kp, 'rx')
# ax4.axis('off')
# ax4.set_title('Canny', fontsize=10)

# plt.show()

# code.interact(local=locals())

# face_flip1, x_kp_flip1, y_kp_flip1 = flip(face, kp, True)
# face_flip2, x_kp_flip2, y_kp_flip2 = flip(face, kp, False)
# face_rot1, x_kp_rot1, y_kp_rot1 = rotate(face, kp, True)
# face_rot2, x_kp_rot2, y_kp_rot2 = rotate(face, kp, False)

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), 
#   (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

# ax1.imshow(face, cmap=plt.cm.Greys_r)
# ax1.plot(x_kp, y_kp, 'rx')
# ax1.axis('off')
# ax1.set_title('Original', fontsize=20)

# ax2.imshow(face_flip1, cmap=plt.cm.gray)
# ax2.plot(x_kp_flip1, y_kp_flip1, 'rx')
# ax2.axis('off')
# ax2.set_title('Flipped Horizontal', fontsize=20)

# ax3.imshow(face_flip2, cmap=plt.cm.gray)
# ax3.plot(x_kp_flip2, y_kp_flip2, 'rx')
# ax3.axis('off')
# ax3.set_title('Flipped Vertical', fontsize=20)

# ax4.imshow(face_rot1, cmap=plt.cm.gray)
# ax4.plot(x_kp_rot1, y_kp_rot1, 'rx')
# ax4.axis('off')
# ax4.set_title('Rotated Clockwise', fontsize=20)

# ax5.imshow(face_rot2, cmap=plt.cm.gray)
# ax5.plot(x_kp_rot2, y_kp_rot2, 'rx')
# ax5.axis('off')
# ax5.set_title('Rotated Counter Clockwise', fontsize=20)

# ax6.imshow(skim.random_noise(face, mode='gaussian'), cmap=plt.cm.gray)
# ax6.plot(x_kp, y_kp, 'rx')
# ax6.axis('off')
# ax6.set_title('Random Noise', fontsize=20)

# ax7.imshow(skim.random_noise(face, mode='s&p'), cmap=plt.cm.gray)
# ax7.plot(x_kp, y_kp, 'rx')
# ax7.axis('off')
# ax7.set_title('Salt and Pepper', fontsize=20)

# ax8.imshow(enhance_contrast(face), cmap=plt.cm.gray)
# ax8.plot(x_kp, y_kp, 'rx')
# ax8.axis('off')
# ax8.set_title('High Contrast', fontsize=20)

# plt.show()