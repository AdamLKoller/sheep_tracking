import numpy as np


# v5
class distance_calculation:
    def __init__(
        self,
        x=0,
        y=6.9,
        h=5.69,
        sheep_length=1.1375,
        sheep_height=0.775,
        sheep_thick=0.4125,
    ):
        # sheep dimensions: https://www.dimensions.com/element/domestic-sheep-ovis-aries
        """
        this function is to set constant values of the problem
        the parameters are:
          x, y, h is the (physical) coordinates of the anchor point
          px, py is the pixel location of the anchor point
          sheep_length, sheep_height, sheep_thick are the (physical) measurements of an average sheep
        """
        self.x_anchor = x
        self.y_anchor = y
        self.h_anchor = h
        self.sheep_length = sheep_length
        self.sheep_height = sheep_height
        self.sheep_thick = sheep_thick
        self.x_multiplier = 1
        self.y_multiplier = 1
        self.sheep_pixel_width = None
        self.sheep_pixel_height = None

    # def find_orientation(self, px_top, py_top, px_bot, py_bot):
    def find_orientation(self, px_top, py_top, px_bot, py_bot):
        """
        this function is to determine the orientation of the sheep (horizontal, vertical, diagonal)
        and calculate the x_multiplier & y_multiplier
        the parameters are:
          px_top, py_top, px_bot, py_bot are the pixel locations of the top & bottom corners of the object-detection bounding box
        """
        sheep_pixel_width = abs(px_bot - px_top)
        sheep_pixel_height = abs(py_top - py_bot)

        self.sheep_pixel_width = sheep_pixel_width
        self.sheep_pixel_height = sheep_pixel_height

        if (sheep_pixel_width / sheep_pixel_height) > 1.8:
            # ori = "horizontal"
            # print("ori hor")
            self.x_multiplier = self.sheep_length
            self.y_multiplier = self.sheep_height
        if (sheep_pixel_height / sheep_pixel_width) > 1.8:
            # ori = "vertical"
            # print("ori vert")
            self.x_multiplier = self.sheep_thick
            self.y_multiplier = self.sheep_length
        else:
            # ori = "diagonal"
            # print("ori diagonal")
            # self.x_multiplier = self.sheep_length * np.cos(np.arctan(pixel_width/pixel_height))
            self.x_multiplier = (
                self.sheep_length
                * sheep_pixel_width
                / np.sqrt(sheep_pixel_height**2 + sheep_pixel_width**2)
            )

            # self.y_multiplier = self.sheep_length * np.sin(np.arctan(pixel_width/pixel_height))
            self.y_multiplier = (
                self.sheep_length
                * sheep_pixel_height
                / np.sqrt(sheep_pixel_height**2 + sheep_pixel_width**2)
            )

    def delta_x(self, delta_px):
        """
        find the distance the object has moved in the x-direction
        """
        delta_x = delta_px / self.sheep_pixel_width * self.x_multiplier
        return delta_x

    def delta_y(self, delta_py):
        """
        find the distance the object has moved in the y-direction
        """
        cosb = self.h_anchor / np.sqrt(self.h_anchor**2 + self.y_anchor**2)
        delta_y = cosb * (delta_py / self.sheep_pixel_height) * self.y_multiplier
        return delta_y

    # def cal_coordinates(self, px_top, py_top, px_bot, py_bot):
    def find_center(self, px_top, py_top, px_bot, py_bot):
        """
        return the coordinates (x,y) of the center of the bounding box
        """
        px_center = (px_top + px_bot) / 2
        py_center = (py_top + py_bot) / 2
        return px_center, py_center

    def cal_distance(self, p_rep1: tuple, p_rep2: tuple):
        """
        calculate the distance the object has moved from frame1 to frame2
        p_rep1 and p_rep2 follow this format:
          (x_top, y_top, x_bot, y_bot)
        whether top-left corner or bottom-left corner is the origin point does not matter.
        the algorithm works either case
        """
        # find the center of each frame
        x1, y1 = self.find_center(p_rep1[0], p_rep1[1], p_rep1[2], p_rep1[3])
        x2, y2 = self.find_center(p_rep2[0], p_rep2[1], p_rep2[2], p_rep2[3])

        # how much real (PHYSICAL) distance the sheep is perceived to have moved vertically & horizontally, with respect to the avg size of a sheep
        self.find_orientation(p_rep2[0], p_rep2[1], p_rep2[2], p_rep2[3])

        dx = self.delta_x(abs(x1 - x2))
        dy = self.delta_y(abs(y1 - y2))

        # print("\nx_mul\n", self.x_multiplier)
        # print("\ny_mul\n", self.y_multiplier)

        # return the (physical) distance object has moved between frame1 and frame2

        return np.sqrt(dx**2 + dy**2)
