import numpy as np


class distance_calculation:
    def __init__(
        self,
        x=0,
        y=6.9,
        h=5.69,
        px=2560,
        py=1438,
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
        self.pixel_anchor_x = px
        self.pixel_anchor_y = py
        self.sheep_length = sheep_length
        self.sheep_height = sheep_height
        self.sheep_thick = sheep_thick
        self.x_multiplier = 1
        self.y_multiplier = 1

    def find_orientation(self, px_top, py_top, px_bot, py_bot):
        """
        this function is to determine the orientation of the sheep (horizontal, vertical, diagonal)
        and calculate the x_multiplier & y_multiplier
        the parameters are:
          px_top, py_top, px_bot, py_bot are the pixel locations of the top & bottom corners of the object-detection bounding box
        """
        pixel_width = px_bot - px_top
        pixel_height = py_top - py_bot

        if (pixel_width / pixel_height) > 1.8:
            # ori = "horizontal"
            print("ori hor")
            self.x_multiplier = self.sheep_length
            self.y_multiplier = self.sheep_height
        if (pixel_height / pixel_width) > 1.8:
            # ori = "vertical"
            print("ori vert")
            self.x_multiplier = self.sheep_thick
            self.y_multiplier = self.sheep_length
        else:
            # ori = "diagonal"
            print("ori diagonal")
            # self.x_multiplier = self.sheep_length * np.cos(np.arctan(pixel_width/pixel_height))
            self.x_multiplier = (
                self.sheep_length
                * pixel_width
                / np.sqrt(pixel_height**2 + pixel_width**2)
            )

            # self.y_multiplier = self.sheep_length * np.sin(np.arctan(pixel_width/pixel_height))
            self.y_multiplier = (
                self.sheep_length
                * pixel_height
                / np.sqrt(pixel_height**2 + pixel_width**2)
            )

    def delta_x(self, delta_Rw):
        """
        this function returns the (physical) coordinate difference along the x_axis between the Anchor point and an obj
        condition for mock-test: beta<pi/2
        the parameters are:
          delta_Rw:
            the real (PHYSICAL) distance the sheep is perceived to have moved HORIZONTALLY, with respect to the avg size of a sheep
        """
        R = np.sqrt(self.x_anchor**2 + self.y_anchor**2 + self.h_anchor**2)
        beta = np.arccos(1 - 1 / 2 * ((delta_Rw / (R)) ** 2))
        # # beta = np.arccos(1 - ( (delta_pw/(2*R))**2 ) )
        # print("R", R)
        # print("delta_Rw", delta_Rw)
        # print("cos beta", 1 - 1/2 *( (delta_Rw/(R))**2 ))
        # # print("cos beta", 1 - ( (delta_pw/(2*R))**2 ))
        # print("beta", beta)
        if beta < np.pi / 2:
            if delta_Rw > 0:
                return R * np.tan(beta)
            return -R * np.tan(beta)
        raise ValueError("beta", beta, "> than pi/2 = ", np.pi / 2)

    def delta_y(self, delta_Rh):
        """
        this function returns the (physical) coordinate difference along the y_axis between the Anchor point and an obj
        condition for mock-test: (alpha + delta_alpha) < pi/2
        the parameters are:
          delta_Rh:
            the real (PHYSICAL) distance the sheep is perceived to have moved VERTICALLY, with respect to the avg size of a sheep
        """
        R = np.sqrt(self.x_anchor**2 + self.y_anchor**2 + self.h_anchor**2)
        alpha = np.arccos(self.h_anchor / R)
        delta_alpha = np.arccos(1 - 1 / 2 * ((delta_Rh / (R)) ** 2))

        if delta_Rh < 0:
            delta_alpha = -delta_alpha

        if alpha + delta_alpha < np.pi / 2:
            # return R*np.sin(delta_alpha)/np.sin(np.pi/2 - alpha - delta_alpha)
            return self.h_anchor * np.tan(alpha - delta_alpha) - self.y_anchor

        print("alpha", alpha)
        print("delta", delta_alpha)
        raise ValueError(
            "(alpha + delta_alpha)", alpha + delta_alpha, "> than pi/2 = ", np.pi / 2
        )

    def cal_coordinates(self, px_top, py_top, px_bot, py_bot):
        """
        return the (physical) coordinates of obj in the (n)th frame, given the pixel representation of the obj
        """
        px_center = (px_top + px_bot) / 2
        py_center = (py_top + py_bot) / 2

        # print("px_center", px_center)
        # print("py_center", py_center)

        # how many pixels the sheep has moved horizontally & vertically
        del_pw = px_center - self.pixel_anchor_x
        del_ph = py_center - self.pixel_anchor_y

        # how much real (PHYSICAL) distance the sheep is perceived to have moved vertically & horizontally, with respect to the avg size of a sheep
        self.find_orientation(px_top, py_top, px_bot, py_bot)
        del_Rw = del_pw / (px_bot - px_top) * self.x_multiplier
        del_Rh = del_ph / (py_top - py_bot) * self.y_multiplier

        # print("x_mul", self.x_multiplier)
        # print("y_mul", self.y_multiplier)

        del_x = self.delta_x(del_Rw)
        del_y = self.delta_y(del_Rh)
        x_n = self.x_anchor + del_x
        y_n = self.y_anchor + del_y
        return x_n, y_n

    def cal_distance(self, p_rep1: tuple, p_rep2: tuple):
        """
        this function takes in the pixel representations (p_rep) of 2 bounding boxes and return the (physical) distance between them
        the parameters are:
          pixel representations are (1,4) tuples in xyxy format
          p_rep1: bounding box of object in the (n)th frame
          p_rep2: bounding box of the same object in the (n+1)th frame
              note: an object is considered the same object if it has the same id in obj tracking output
        """
        # calculate the (physical) coordinates of obj1 & obj2
        x1, y1 = self.cal_coordinates(p_rep1[0], p_rep1[1], p_rep1[2], p_rep1[3])
        x2, y2 = self.cal_coordinates(p_rep2[0], p_rep2[1], p_rep2[2], p_rep2[3])

        # return the (physical) distance between obj1 & obj2
        return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
