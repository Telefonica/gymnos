#
#
#   Gaussian distorsion
#
#

import math
import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor


class GaussianDistortion(DataAugmentor):
    """
    This class performs randomised, elastic gaussian distortions on images.
    """

    def __init__(self, probability, grid_width, grid_height, magnitude, corner, method, mex, mey, sdx, sdy):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float

        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:

        .. math::

         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        super().__init__(probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True
        self.corner = corner
        self.method = method
        self.mex = mex
        self.mey = mey
        self.sdx = sdx
        self.sdy = sdy

    def transform(self, image):
        """
        Distorts the passed image(s) according to the parameters supplied
        during instantiation, returning the newly distorted image.

        :param image: The image(s) to be distorted.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        w, h = image.size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(math.floor(w / float(horizontal_tiles)))
        height_of_square = int(math.floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def sigmoidf(x, y, sdx=0.05, sdy=0.05, mex=0.5, mey=0.5, const=1):
            def sigmoid(x1, y1):
                return (const * (math.exp(-(((x1 - mex)**2) / sdx + ((y1 - mey)**2) / sdy))) + max(0, -const) -
                        max(0, const))

            xl = np.linspace(0, 1)
            yl = np.linspace(0, 1)
            X, Y = np.meshgrid(xl, yl)

            Z = np.vectorize(sigmoid)(X, Y)
            mino = np.amin(Z)
            maxo = np.amax(Z)
            res = sigmoid(x, y)
            res = max(((((res - mino) * (1 - 0)) / (maxo - mino)) + 0), 0.01) * self.magnitude
            return res

        def corner(x, y, corner="ul", method="out", sdx=0.05, sdy=0.05, mex=0.5, mey=0.5):
            ll = {'dr': (0, 0.5, 0, 0.5), 'dl': (0.5, 1, 0, 0.5), 'ur': (0, 0.5, 0.5, 1), 'ul': (0.5, 1, 0.5, 1),
                  'bell': (0, 1, 0, 1)}
            new_c = ll[corner]
            new_x = (((x - 0) * (new_c[1] - new_c[0])) / (1 - 0)) + new_c[0]
            new_y = (((y - 0) * (new_c[3] - new_c[2])) / (1 - 0)) + new_c[2]
            if method == "in":
                const = 1
            else:
                if method == "out":
                    const = -1
                else:
                    const = 1
            res = sigmoidf(x=new_x, y=new_y, sdx=sdx, sdy=sdy, mex=mex, mey=mey, const=const)

            return res

        for a, b, c, d in polygon_indices:
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]

            sigmax = corner(x=x3 / w, y=y3 / h, corner=self.corner, method=self.method, sdx=self.sdx, sdy=self.sdy,
                            mex=self.mex, mey=self.mey)
            dx = np.random.normal(0, sigmax, 1)[0]
            dy = np.random.normal(0, sigmax, 1)[0]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        image = image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        return np.array(image)
