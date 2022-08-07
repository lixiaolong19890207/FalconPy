import numpy as np

from core.defines import RGBA


class TransferFunc:
    def __init__(self):
        self.min_pos = 0
        self.max_pos = 99

        self.pos_to_rgba = {}
        self.pos_to_alpha = {}

    def set_control_pts(self, val: dict):
        self.pos_to_rgba = val

    def get_transfer_func(self):
        pos_list = self.pos_to_rgba.keys()
        pos_list = sorted(pos_list)

        if len(pos_list) < 2:
            return None

        pos_first, pos_last = pos_list[0], pos_list[-1]
        if pos_first > self.min_pos:
            self.pos_to_rgba[self.min_pos] = self.pos_to_rgba[pos_first]

        if pos_last < self.max_pos:
            self.pos_to_rgba[self.max_pos] = self.pos_to_rgba[pos_last]

        array = np.zeros((self.max_pos - self.min_pos + 1, 4), dtype=np.float32)

        pos_list = sorted(self.pos_to_rgba.keys())
        for i in range(1, len(pos_list)):
            pos_left = pos_list[i - 1]
            pos_right = pos_list[i]
            rgba_left = self.pos_to_rgba[pos_left]
            rgba_right = self.pos_to_rgba[pos_right]

            for p in range(pos_left, pos_right + 1):
                radio = (p - pos_left) / (pos_right - pos_left + 1)

                self.pos_to_rgba[p] = RGBA(
                    red=rgba_left.red + (rgba_right.red - rgba_left.red) * radio,
                    green=rgba_left.green + (rgba_right.green - rgba_left.green) * radio,
                    blue=rgba_left.blue + (rgba_right.blue - rgba_left.blue) * radio,
                    alpha=rgba_left.alpha + (rgba_right.alpha - rgba_left.alpha) * radio,
                )
                array[p, ...] = [self.pos_to_rgba[p].red, self.pos_to_rgba[p].green, self.pos_to_rgba[p].blue, self.pos_to_rgba[p].alpha]

            i += 1

        return array


if __name__ == '__main__':
    pos_to_rgba = {
        0: RGBA(red=0.8, green=0, blue=0, alpha=0),
        10: RGBA(red=0.8, green=0, blue=0, alpha=0.3),
        40: RGBA(red=0.8, green=0.8, blue=0, alpha=0),
        99: RGBA(red=1.0, green=0.8, blue=1.0, alpha=1.0)
    }

    tf = TransferFunc()
    tf.set_control_pts(pos_to_rgba)
    fun = tf.get_transfer_func()
    pass