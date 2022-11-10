import numpy as np
from scipy import spatial
from sklearn.metrics import mean_squared_error
import random


class df:

    @staticmethod
    def get_formatted_training(split, max_precursor):
        t_seq = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_train_seq.npy" % (split, split))
        t_pre = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_train_pre.npy" % (split, split))
        t_int = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_train_int.npy" % (split, split))

        return df.format_data(t_seq, t_pre, t_int, max_precursor)

    @staticmethod
    def get_formatted_validation(split, max_precursor):
        t_seq = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_valid_seq.npy" % (split, split))
        t_pre = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_valid_pre.npy" % (split, split))
        t_int = np.load("E:\\PBL Dataset\\splits\\split%s\\s%s_valid_int.npy" % (split, split))

        return df.format_data(t_seq, t_pre, t_int, max_precursor)

    @staticmethod
    def format_data(seq, pre, int, max_precursor):
        to_remove = []
        for i in range(len(pre)):
            if pre[i] == 0 or pre[i] > max_precursor:
                to_remove.append(i)

        seq = np.delete(seq, to_remove, axis=0)
        pre = np.delete(pre, to_remove, axis=0)
        int = np.delete(int, to_remove, axis=0)

        pre = df.reformat_precursor(pre, max_precursor)
        return [seq, pre, int]

    @staticmethod
    def reformat_precursor(pre, max_precursor):
        one_hot = []
        for i in pre:
            one_hot.append(np.zeros(max_precursor))
            if i <= max_precursor:
                one_hot[-1][i.astype(int)-1] = 1
        return np.array(one_hot)

    @staticmethod
    def cosine_similarity(x, y):
        return 1-spatial.distance.cosine(x,y)

    @staticmethod
    def mse(x, y):
        return mean_squared_error(x, y)

    @staticmethod
    def spectral_angle(x, y):
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        prod = np.dot(x/x_norm, y/y_norm)
        if prod > 1.0:
            prod = 1
        return 1-2*(np.arccos(prod)/np.pi)

        

    @staticmethod
    def generate_baseline(y):
        cos = []
        pcc = []
        spec_angle = []
        for elem in y:
            random_generated = np.zeros(56)
            for i in range(56):
                if elem[i] != 0:
                    random_generated[i] = random.random()
            #print(elem)
            #print(random_generated)
            cos.append(df.cosine_similarity(elem, random_generated))
            pcc.append(np.corrcoef(elem, random_generated))
            spec_angle.append(df.spectral_angle(elem, random_generated))
        return [sum(cos)/len(cos), sum(pcc)/len(pcc), sum(spec_angle)/len(spec_angle)]
