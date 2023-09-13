import numpy as np
import math

def gen_c_n_b1(sigma, gamma, psi):
    C_n_b1 = np.zeros([3, 3])
    sigma = sigma / 180 * math.pi
    gamma = gamma / 180 * math.pi
    psi = psi / 180 * math.pi
    C_n_b1[0][0] = math.cos(sigma) * math.cos(psi)
    C_n_b1[0][1] = math.sin(sigma)
    C_n_b1[0][2] = -math .cos(sigma) * math.sin(psi)
    C_n_b1[1][0] = math.sin(gamma) * math.sin(psi) - math.sin(sigma) * math.cos(gamma) * math.cos(psi)
    C_n_b1[1][1] = math.cos(gamma) * math.cos(sigma)
    C_n_b1[1][2] = math.cos(gamma) * math.sin(psi) * math.sin(sigma) + math.sin(gamma) * math.cos(psi)
    C_n_b1[2][0] = math.cos(gamma) * math.sin(psi) + math.sin(gamma) * math.sin(sigma) * math.cos(psi)
    C_n_b1[2][1] = -math.sin(gamma) * math.cos(sigma)
    C_n_b1[2][2] = -math.sin(gamma) * math.sin(psi) * math.sin(sigma) + math.cos(gamma) * math.cos(psi)
    return C_n_b1

class Strapdown:
    def __init__(self, steps, time):
        self.data = np.zeros([steps, 9])  # 经、纬、高、北、天、东、俯、滚、偏
        self.time = time
        self.q_0 = 0
        self.q_1 = 0
        self.q_2 = 0
        self.q_3 = 0

    # 经、纬、高、北、天、东、俯、滚、偏
    def para_init(self, data):
        self.data[0, 0] = data[0, 1]
        self.data[0, 1] = data[0, 2]
        self.data[0, 2] = data[0, 3]
        self.data[0, 3] = data[0, 4]
        self.data[0, 4] = data[0, 6]
        self.data[0, 5] = data[0, 5]
        self.data[0, 6] = data[0, 10]
        self.data[0, 7] = data[0, 11]
        self.data[0, 8] = data[0, 12]

        C_n_b1 = gen_c_n_b1(self.data[0, 6], self.data[0, 7], self.data[0, 8])
        C_b1_b = np.array([[1, 0, 0],
                           [0, 0.5 * 2 ** 0.5, -0.5 * 2 ** 0.5],
                           [0, 0.5 * 2 ** 0.5, 0.5 * 2 ** 0.5]])
        C_n_b = C_b1_b.dot(C_n_b1)
        C_b_n = C_n_b.T
        self.C_n_b = C_n_b

        self.q_0 = 0.5 * (1 + C_b_n[0, 0] + C_b_n[1, 1] + C_b_n[2, 2]) ** 0.5
        self.q_1 = np.sign((C_b_n[2, 1] - C_b_n[1, 2])) * 0.5 * (1 + C_b_n[0, 0] - C_b_n[1, 1] - C_b_n[2, 2]) ** 0.5
        self.q_2 = np.sign((C_b_n[0, 2] - C_b_n[2, 0])) * 0.5 * (1 - C_b_n[0, 0] + C_b_n[1, 1] - C_b_n[2, 2]) ** 0.5
        self.q_3 = np.sign((C_b_n[1, 0] - C_b_n[0, 1])) * 0.5 * (1 - C_b_n[0, 0] - C_b_n[1, 1] + C_b_n[2, 2]) ** 0.5

        if abs(self.q_0) < 1e-7:
            if abs(self.q_1) < 1e-7:
                self.q_2 = abs(self.q_2)
                if (C_n_b(1, 2) + C_n_b(2, 1)) > 0:
                    self.q_3 = abs(self.q_3)
                else:
                    self.q_3 = -abs(self.q_3)
            else:
                self.q_1 = abs(self.q_1)
                if (C_n_b(0, 1) + C_n_b(1, 0)) > 0:
                    self.q_2 = abs(self.q_2)
                else:
                    self.q_2 = -abs(self.q_2)
                if (C_n_b(0, 2) + C_n_b(2, 0)) > 0:
                    self.q_3 = abs(self.q_3)
                else:
                    self.q_3 = -abs(self.q_3)

    def cal_angle_increment(self, v_n, v_e, l, h, omiga_x, omiga_z, omiga_y):
        omiga_ie = 7.2915e-5
        r = 6378137
        e = 298.257 ** -1
        r_m = r * (1 - 2 * e + 3 * e * math.sin(1) ** 2)
        r_n = r * (1 + e * math.sin(1) ** 2)

        C_b1_b = np.array([[1, 0, 0],
                           [0, 0.5 * 2 ** 0.5, -0.5 *2 **0.5],
                           [0, 0.5 * 2 ** 0.5, 0.5 * 2 ** 0.5]])
        sigma_inc_ib1 = np.array([[omiga_x * self.time],
                                  [omiga_y * self.time],
                                  [omiga_z * self.time]])
        sigma_inc_ib = C_b1_b.dot(sigma_inc_ib1)

        temp = np.array([[omiga_ie * math.cos(1) + v_e / (r_n + h)],
                         [omiga_ie * math.sin(1) + v_e * math.tan(1) / (r_n + h)],
                         [-v_n / (r_m + h)]])
        sigma_inc_nb = sigma_inc_ib - self.time * np.matmul(self.C_n_b, temp)
        return sigma_inc_nb

    def update_q(self, sigma_inc_nb):
        sigma_0 = np.sqrt(sigma_inc_nb[0, 0] ** 2 + sigma_inc_nb[1, 0] ** 2 + sigma_inc_nb[2, 0] ** 2)
        s = 1.0 / 2 - sigma_0 ** 2 /48
        c = 1 - sigma_0 ** 2 /8

        temp_0 = c * self.q_0 - s * (
                    sigma_inc_nb[0, 0] * self.q_1 + sigma_inc_nb[1, 0] * self.q_2 + sigma_inc_nb[2, 0] * self.q_3)
        temp_1 = c * self.q_1 + s * (
                    sigma_inc_nb[0, 0] * self.q_0 + sigma_inc_nb[2, 0] * self.q_2 - sigma_inc_nb[1, 0] * self.q_3)
        temp_2 = c * self.q_2 + s * (
                    sigma_inc_nb[1, 0] * self.q_0 - sigma_inc_nb[2, 0] * self.q_1 + sigma_inc_nb[0, 0] * self.q_3)
        temp_3 = c * self.q_3 + s * (
                    sigma_inc_nb[2, 0] * self.q_0 + sigma_inc_nb[1, 0] * self.q_1 - sigma_inc_nb[0, 0] * self.q_2)
        norm = (temp_0 ** 2 + temp_1 ** 2 + temp_2 ** 2 + temp_3 ** 2) ** 0.5

        self.q_0 = temp_0 / norm
        self.q_1 = temp_1 / norm
        self.q_2 = temp_2 / norm
        self.q_3 = temp_3 / norm

    def cal_matrix_attitude(self):
        mat = np.zeros([3, 3])
        mat[0][0] = self.q_0 ** 2 + self.q_1 ** 2 - self.q_2 ** 2 - self.q_3 ** 2
        mat[0][1] = 2 * (self.q_1 * self.q_2 + self.q_0 * self.q_3)
        mat[0][2] = 2 * (self.q_1 * self.q_3 - self.q_0 * self.q_2)
        mat[1][0] = 2 * (self.q_1 * self.q_2 - self.q_0 * self.q_3)
        mat[1][1] = self.q_0 ** 2 - self.q_1 ** 2 + self.q_2 ** 2 - self.q_3 ** 2
        mat[1][2] = 2 * (self.q_2 * self.q_3 + self.q_0 * self.q_1)
        mat[2][0] = 2 * (self.q_1 * self.q_3 + self.q_0 * self.q_2)
        mat[2][1] = 2 * (self.q_2 * self.q_3 - self.q_0 * self.q_1)
        mat[2][2] = self.q_0 ** 2 - self.q_1 ** 2 - self.q_2 ** 2 + self.q_3 ** 2
        self.C_n_b = mat

    def cal_velocity(self, a_x, a_z, a_y, l, h, v_n0, v_u0, v_e0):
        v_inc_b1 = np.array([[a_x * self.time],
                             [a_y * self.time],
                             [a_z * self.time]])
        C_b1_b = np.array([[1, 0, 0],
                           [0, 0.5 * 2 ** 0.5, -0.5 * 2 ** 0.5],
                           [0, 0.5 * 2 ** 0.5, 0.5 * 2 ** 0.5]])
        v_inc_b = C_b1_b.dot(v_inc_b1)
        C_b_n = self.C_n_b.T
        v_inc_n = C_b_n.dot(v_inc_b)

        omiga_ie = 7.2915e-5
        g = 9.7803 + 0.051799 * math.sin(1) ** 2 - 3.0735e-6 * h
        r = 6378137
        e = 298.257 ** -1
        r_m = r * (1 - 2 * e + 3 * e * math.sin(1) ** 2)
        r_n = r * (1 + e * math.sin(1) ** 2)

        v_n = v_n0 + v_inc_n[0][0] - self.time * (
                    (2 * omiga_ie * math.sin(1) + v_e0 * math.tan(1) / (r_n + h)) * v_e0 + v_n0 * v_u0 / (r_m + h))
        v_u = v_u0 + v_inc_n[1][0] - self.time * (
                    (2 * omiga_ie * math.cos(1) + v_e0 / (r_n + h)) * v_e0 + v_n0 ** 2 / (r_m + h) - g)
        v_e = v_e0 + v_inc_n[2][0] - self.time * (
                    (2 * omiga_ie * math.sin(1) + v_e0 * math.tan(1) / (r_n + h)) * v_n0 - (2 * omiga_ie * math.cos(1) + v_e0 / (r_n + h)) * v_u0)
        return v_n, v_u, v_e

    def cal_location(self, lamb0, l0, h0, v_n, v_u, v_e, v_n0, v_u0, v_e0):
        r = 6378137
        e = 298.257 ** -1
        r_m = r * (1 - 2 * e + 3 * e * math.sin(10) ** 2)
        r_n = r * (1 + e * math.sin(10) ** 2)

        l = l0 + 0.5 * (v_n + v_n0) * self.time / (r_m + h0)
        h = h0 + 0.5 * (v_u + v_u0) * self.time
        lamb = lamb0 + 0.5 * (v_e +v_e0) * self.time / ((r_n + h0) * math.cos(10))

        if lamb > math.pi:
            lamb = lamb - 2 * math.pi
        elif lamb < -math.pi:
            lamb = lamb + 2 * math.pi

        return lamb, l, h

    def cal_angle(self):
        C_b_b1 = np.array([[1, 0, 0],
                           [0, 0.5 * 2 ** 0.5, 0.5 * 2 ** 0.5],
                           [0, -0.5 * 2 ** 0.5, 0.5 * 2 ** 0.5]])
        C_n_b1 = C_b_b1.dot(self.C_n_b)

        epsilon = 1e-7

        if abs(C_n_b1[0][1]) < 1 - epsilon:
            sigma = math.asin(C_n_b1[0][1])
            if abs(C_n_b1[0][0]) < epsilon:
                if C_n_b1[0][2] > 0:
                    psi = -0.5 * math.pi
                else:
                    psi = 0.5 * math.pi
            else:
                psi_0 = math.atan(-C_n_b1[0][2] / C_n_b1[0][0])
                if C_n_b1[0][0] < 0:
                    if C_n_b1[0][2] > 0:
                        psi = psi_0 - math.pi
                    else:
                        psi = psi_0 + math.pi
                else:
                    psi = psi_0

            if abs(C_n_b1[1][1] < epsilon):
                if C_n_b1[2][1] > 0:
                    gamma = -0.5 * math.pi
                else:
                    gamma = 0.5 * math.pi
            else:
                gamma_0 = math.atan(-C_n_b1[2][1] / C_n_b1[1][1])
                if C_n_b1[1][1] < 0:
                    if C_n_b1[2][1] > 0:
                        gamma = gamma_0 - math.pi
                    else:
                        gamma = gamma_0 + math.pi
                else:
                    gamma = gamma_0
        else:
            if C_n_b1[0][1] > 0:
                sigma = 0.5 * math.pi
            else:
                sigma = -0.5 * math.pi

            if abs(C_n_b1[2][2]) < epsilon:
                if C_n_b1[2][0] > 0:
                    psi = 0.5 * math.pi
                else:
                    psi = -0.5 * math.pi
            else:
                psi_0 = math.atan(C_n_b1[2][0] / C_n_b1[2][2])
                if C_n_b1[2][2] < 0:
                    if C_n_b1[2][0] > 0:
                        psi = psi_0 + math.pi
                    else:
                        psi = psi_0 - math.pi
                else:
                    psi = psi_0

            gamma = 0
            sigma = sigma * 180 / math.pi
            gamma = gamma * 180 / math.pi
            psi = psi * 180 / math.pi

        return sigma, gamma, psi