import pandas as pd


"""
文件加载 450000 * 17
    column:
        0: time
        1: 经度
        2: 纬度
        3: 高度
        4: 北向速度
        5: 东向速度
        6: 天向速度
        7: x轴加速度
        8: y轴加速度
        9: z轴加速度
        10: 俯仰角
        11: 滚转角
        12: 偏航角
        13: x轴角速度
        14: y轴角速度
        15: z轴角速度
"""
def load_data(path):
    temp = load_file(path)
    temp[:, 5] = -temp[:, 5]
    temp[:, 12] = -temp[:, 12]
    temp[:, 9] = -temp[:, 9]
    temp[:, 15] = -temp[:, 15]
    return temp

def load_file(path):
    return pd.read_csv(path, sep=" ", header=None).values