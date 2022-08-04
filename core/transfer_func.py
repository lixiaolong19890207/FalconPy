import numpy as np


class TransferFunc:
    def __init__(self):
        self.min_pos = 0
        self.max_pox = 99

        self.pos_to_rgba = {}
        self.pos_to_alpha = {}

    def set_control_pts(self, val: dict):
        self.pos_to_rgba = val

    def get_transfer_func(self):
        pos_list = self.pos_to_rgba.keys()
        pos_list = sorted(pos_list)

        if len(pos_list) > 2:
            return None

        pos_first, pos_last = pos_list[0], pos_list[-1]
        if pos_first > self.min_pos:
            self.pos_to_rgba[self.min_pos] = self.pos_to_rgba[pos_first]

        if pos_last < self.max_pox:
            self.pos_to_rgba[self.max_pox] = self.pos_to_rgba[pos_last]

        array = np.zeros((self.max_pox - self.max_pox + 1, 4), dtype=np.float32)
        for pos in self.pos_to_rgba.keys():
