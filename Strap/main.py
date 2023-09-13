import time
from Strapdown import *
from Util.Fileloader import load_data


if __name__ == '__main__':
    print("Start INS solution.")
    start_time = time.time()
    print("The data file starts to load! Please keep waiting...")
    data = load_data(r"E:\Code\test1.txt")
    print("File load complete!")
    inertia = np.concatenate(data[:, 7:10], data[:, 13:16], axis=1)
    steps = int(input("请设置仿真步数："))
    print("Loading...")
    navigation = Strapdown(steps, 0.01)
    navigation.para_init(data)

    for i in range(1, steps):
        sigma_inc_nb = navigation.cal_angle_increment(navigation.data[i - 1][3], navigation.data[i - 1][5], navigation.data[i - 1][1],
                                                      navigation.data[i - 1][2], inertia[i][3], inertia[i][4], inertia[i][5])
        navigation.update_q(sigma_inc_nb)
        navigation.cal_matrix_attitude()
        v_n, v_u, v_e = navigation.cal_velocity(inertia[i][0], inertia[i][1], inertia[i][2], navigation.data[i - 1][1],
                                                navigation.data[i - 1][2], navigation.data[i - 1][3], navigation.data[i - 1][4],
                                                navigation.data[i - 1][5])
        lamb, l, h = navigation.cal_location(navigation.data[i - 1][0], navigation.data[i - 1][1], navigation.data[i - 1][2],
                                             v_n, v_u, v_e, navigation.data[i - 1][3], navigation.data[i - 1][4], navigation.data[i - 1][5])
        sigma, gamma, psi = navigation.cal_angle()
        para = [lamb, l, h, v_n, v_u, v_e, sigma, gamma, psi]
        for j in range(9):
            navigation.data[i][j] = para[j]

    print("Done.")
    print("Saving data...")
    np.savetxt("strap_solved.csv", navigation.data, delimiter=",")
    print("Done.")
    end_time = time.time() - start_time
    print("The program runs time is " + str(end_time) + "s.")