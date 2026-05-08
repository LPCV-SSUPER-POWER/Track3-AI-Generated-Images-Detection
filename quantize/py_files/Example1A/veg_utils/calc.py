#!/usr/local/bin/python3
#-*- encoding=utf-8 -*-

import numpy as np


class Calculator(object):

    @staticmethod
    def cosineSim(data1, data2):
        data1, data2 = data1.flatten(), data2.flatten()
        num = data1 @ data2
        denom = np.linalg.norm(data1) * np.linalg.norm(data2)
        return num / denom

    @staticmethod
    def mse(data1, data2):
        data1, data2 = data1.flatten(), data2.flatten()
        if len(data1) == len(data2):
            return np.mean((data1 - data2)**2)
        else:
            raise ('target length is different with prediction')

    @staticmethod
    def sqnr(data, golden):
        data, golden = data.flatten(), golden.flatten()
        if len(data) != len(golden):
            raise ValueError("length mismatch.")
        mse = Calculator.mse(data, golden)
        mss = np.mean(golden**2)

        if 0.0 == mse:
            sqnr = float("inf")
        else:
            sqnr = 10 * np.log10(mss / mse)

        return sqnr

    @staticmethod
    def scaledDiff(data, golden, min_val, max_val):
        diff = np.abs(data - golden)
        scaled_diff_mean = (diff.mean() / (max_val - min_val)) * 255

        return scaled_diff_mean
