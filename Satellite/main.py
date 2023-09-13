import time
from Util.Fileloader import load_data, load_file
from Strap.Strapdown import *
from Satellite.Satellite import *


if __name__ == '__main__':
    print("Start GINS solution.")
    start_time = time.time()
    print("The data file starts to load! Please keep waiting...")
    data = load_data(r"E:\Code\test1.txt")
    data_sat = load_file("satellite's data")
    print("File load complete!")
    inertia = np.concatenate(data[:, 7:10], data[:, 13:16], axis=1)
    steps = int(input("请设置仿真步数："))
    print("Loading...")
    T_ins = 0.01
    navigation = Strapdown(steps, T_ins)
    navigation.para_init(data)
    sat = Satellite(T_ins)
    T_sat = 0.04
    row = 0
    velocity_inc = data[:, 7:10]*T_ins

    try:
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

            if not i % 4:
                G = data_sat[row]
                order = [5, 3, 4, 1, 0, 2, 8, 6, 7]
                I = [para[i] for i in order]
                sat.update_X(T_sat, I, G, velocity_inc[i])
                sat.update_P(T_sat, I)
                ans = sat.solver(I, sat.X, navigation.C_n_b)
                para[0] = ans[1]
                para[1] = ans[0]
                para[2] = ans[2]
                para[3] = ans[4]
                para[4] = ans[5]
                para[5] = ans[3]
                navigation.q_0 = ans[6]
                navigation.q_1 = ans[7]
                navigation.q_2 = ans[8]
                navigation.q_3 = ans[9]

            for j in range(9):
                navigation.data[i][j] = para[j]
                
    except IndexError as e:
        print("Insufficient satellite data: ", e)
    finally:
        print("Done.")
        print("Saving data...")
        np.savetxt("strap_solved.csv", navigation.data, delimiter=",")
        print("Done.")
        end_time = time.time() - start_time
        print("The program runs time is " + str(end_time) + "s.")