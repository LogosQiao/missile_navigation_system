import numpy as np
import math
import autograd.numpy as anp
from autograd import jacobian


class Satellite:
    def __init__(self, time):
        self.X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.P = np.diag([(5 * math.pi / 180) ** 2, (5 * math.pi / 180) ** 2, (5 * math.pi / 180) ** 2, 1, 1, 1,
                          (1e-4 * math.pi / 180) ** 2, (1e-4 * math.pi / 180) ** 2, 400])
        self.Q = np.diag(
            [(2 * math.pi / 180 / 3600) ** 2, (2 * math.pi / 180 / 3600) ** 2, (2 * math.pi / 180 / 3600) ** 2,
             (1e-3 * 9.8) ** 2, (1e-3 * 9.8) ** 2, (1e-3 * 9.8) ** 2, 0, 0, 0])
        self.R_1 = np.diag([0.04, 0.04, 0.04, 100, 100, 100])
        self.R_2 = np.diag([0.04 * 1e12, 0.04 * 1e12, 0.04 * 1e12, 1e14, 1e14, 1e14])
        self.T_ins = time

    """
        I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
        delta_v_ib: [delta_vx, delta_vy, delta_vz]
    """
    def get_fx(self, X, I, delta_v_ib):
        M = self.get_M()
        C = self.get_Cmm()
        omiga_ie = self.get_omiga_ie(I)
        omiga_em = self.get_omiga_em(I)
        omiga_im = omiga_ie + omiga_em
        delta_ie = self.get_delta_ie(I)
        delta_em = self.get_delta_em(I)
        delta_im = delta_ie + delta_em
        f_m = self.get_fm(delta_v_ib, I)
        delta_vm = np.array([X[3], X[4], X[5]])
        vm = np.array([I[0], I[1], I[2]])
        RM, RN, R, e, w_ie = self.cal_constPara(I)
        return anp.array([M.dot((np.diag([1, 1, 1]) - C).dot(omiga_im) + C.dot(delta_im))[0],
                          M.dot((np.diag([1, 1, 1]) - C).dot(omiga_im) + C.dot(delta_im))[1],
                          M.dot((np.diag([1, 1, 1]) - C).dot(omiga_im) + C.dot(delta_im))[2], (
                                      (np.diag([1, 1, 1]) - C).dot(f_m) - (2 * omiga_ie + delta_em) * vm + (
                                          2 * omiga_ie + delta_em) * delta_vm)[0],
                          ((np.diag([1, 1, 1]) - C).dot(f_m) - (2 * omiga_ie + delta_em) * vm + (
                                      2 * omiga_ie + delta_em) * delta_vm)[1], (
                                      (np.diag([1, 1, 1]) - C).dot(f_m) - (2 * omiga_ie + delta_em) * vm + (
                                          2 * omiga_ie + delta_em) * delta_vm)[2],
                          X[4] / (RM + I[5]),
                          X[3] / (RN + I[5]) * math.acos(I[3]) + X[6] * I[0] / (RN + I[5]) * math.tan(I[3]) * math.acos(
                              I[3]),
                          X[5]])

    def get_M(self):
        return np.array([[math.cos(self.X[1]), 0, math.sin(self.X[1])],
                         [math.sin(self.X[1]) * math.sin(self.X[0]) / math.cos(self.X[0]), 1,
                          -math.cos(self.X[1]) * math.sin(self.X[0]) / math.cos(self.X[0])],
                         [-math.sin(self.X[1]) / math.cos(self.X[0]), 0, math.cos(self.X[1]) / math.cos(self.X[0])]])

    def get_Cmm(self):
        return np.array([[math.cos(self.X[1]) * math.cos(self.X[2]) - math.sin(self.X[0]) * math.sin(
            self.X[1]) * math.sin(self.X[2]),
                          math.cos(self.X[1]) * math.sin(self.X[2]) + math.sin(self.X[0]) * math.sin(
                              self.X[1]) * math.cos(self.X[2]),
                          -math.sin(self.X[1]) * math.cos(self.X[0])],
                         [-math.cos(self.X[0]) * math.sin(self.X[2]), math.cos(self.X[0]) * math.cos(self.X[2]),
                          math.sin(self.X[0])],
                         [math.sin(self.X[1]) * math.cos(self.X[2]) + math.cos(self.X[1]) * math.sin(
                             self.X[0]) * math.sin(self.X[2]),
                          math.sin(self.X[1]) * math.sin(self.X[2]) - math.cos(self.X[1]) * math.sin(
                              self.X[0]) * math.cos(self.X[2]),
                          math.cos(self.X[1]) * math.cos(self.X[0])]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def get_omiga_ie(self, I):
        _, _, _, _, w_ie = self.cal_constPara(I)
        return np.array([[0], [w_ie * math.cos(I[3])], [w_ie * math.sin(I[3])]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def get_omiga_em(self, I):
        RM, RN, _, _, _ = self.cal_constPara(I)
        return np.array([[-I[1] / (RM + I[5])], [I[0] / (RN + I[5])], [I[0] / (RN + I[5]) * math.tan(I[3])]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def get_delta_ie(self, I):
        _, _, _, _, w_ie = self.cal_constPara(I)
        return np.array([[0], [-self.X[6] * w_ie * math.sin(I[3])], [self.X[6] * w_ie * math.cos(I[3])]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def get_delta_em(self, I):
        RM, RN, _, _, _ = self.cal_constPara(I)
        return np.array([[-self.X[4] / (RM + I[5])], [self.X[3] / (RN + I[5])], [
            self.X[3] / (RN + I[5]) * math.tan(I[3]) + self.X[6] * I[0] / (RN + I[5]) * math.acos(I[3]) ** 2]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def cal_constPara(self, I):
        R = 6378137
        e = 1 / 298.257
        w_ie = 7.2915 * 1e-5
        RM = R * (1 - 2 * e + 3 * e * math.sin(I[3]) ** 2)
        RN = R * (1 + e * math.sin(I[3]) ** 2)
        return RM, RN, R, e, w_ie

    # delta_v_ib: [delta_vx, delta_vy, delta_vz]
    def get_fm(self, delta_v_ib, I):
        C_ms = self.get_Cms(I)
        return C_ms.dot([[delta_v_ib[2]], [delta_v_ib[0]], [delta_v_ib[1]]]) / self.T_ins

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def get_Cms(self, I):
        return np.array([[math.cos(I[8]) * math.cos(I[6]) - math.sin(I[8]) * math.sin(I[6]) * math.sin(I[7]),
                          -math.sin(I[6]) * math.cos(I[7]),
                          math.sin(I[8]) * math.cos(I[6]) + math.cos(I[8]) * math.sin(I[6]) * math.sin(I[7])],
                         [math.cos(I[8]) * math.sin(I[6]) + math.sin(I[8]) * math.sin(I[7]) * math.cos(I[6]),
                          math.cos(I[6]) * math.cos(I[7]),
                          math.sin(I[8]) * math.sin(I[6]) - math.cos(I[8]) * math.cos(I[6]) * math.sin(I[7])],
                         [-math.sin(I[8]) * math.cos(I[7]), math.sin(I[7]), math.cos(I[8]) * math.cos(I[7])]])

    def get_Jacobian(self):
        Jacobian_matrix = jacobian(self.get_fx)
        return Jacobian_matrix(self.X)

    def get_PHI(self, T):
        F = self.get_Jacobian()
        x = [1 for _ in range(9)]
        return np.diag(x) + F * T + 0.5 * F.dot(F) * T * T + 1 / 6.0 * F.dot(F.dot(F)) * T ** 3

    def get_Qk1(self, T):
        Q = self.Q
        F = self.get_Jacobian()
        return Q.dot(T) + 0.5 * T ** 2 * (Q.T.dot(F.T) + F.dot(Q)) + 1 / 6.0 * T ** 3 * (Q.T.dot(F.T) + F.dot(Q))

    """
    I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    G: [ve, vn, vu, L, Lambda, h] of satellite
    return: [Z_ve, Z_vn, Z_vu, Z_L, Z_Lambda, Z_h]
    """

    def update_Obs(self, I, G):
        RM, RN, _, _, _ = self.cal_constPara(I)
        return np.array([I[0] - G[0], I[1] - G[1], I[2] - G[2], (RM + I[5]) * (I[3] - G[3]),
                         (RN + I[5]) * (I[4] - G[4]) * math.cos(I[3]), I[5] - G[5]])

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def update_hx(self, x, I):
        RM, RN, _, _, _ = self.cal_constPara(I)
        return np.array([x[3], x[4], x[5], x[6] * (RM + I[5]), x[7] * (RN + I[5]) * math.cos(I[3]), x[8]]).T

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def update_H(self, I):
        RM, RN, _, _, _ = self.cal_constPara(I)
        H1 = np.array([[RM + I[5], 0, 0], [0, (RN + I[5]) * math.cos(I[3]), 0], [0, 0, 1]])
        return np.block([np.zeros([3, 3]), np.eye(3), np.zeros([3, 3])], [np.zeros([3, 3]), np.zeros([3, 3]), H1])

    def get_X_k1_k(self, X, T, I, delta_v_ib):
        return X + self.get_fx(X, I, delta_v_ib) * T

    def get_B(self, I):
        M_00 = -self.get_M().dot(self.get_Cms(I))
        M_11 = self.get_Cmm().dot(self.get_Cms(I))
        return np.block([[M_00, np.zeros([3, 3]), np.zeros([3, 3])],
                         [np.zeros([3, 3]), M_11, np.zeros([3, 3])],
                         [np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])]])

    def get_P_k1_k(self, T, I):
        P_k = self.P
        B = self.get_B(I)
        PHI = self.get_PHI(T)
        Q = self.get_Qk1(T)
        return PHI.dot(P_k).dot(PHI.T) + B.dot(Q).dot(B.T)

    def get_K(self, T, I):
        P = self.get_P_k1_k(T, I)
        H = self.update_H(I)
        R = self.R_1
        return P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))

    def update_X(self, T, I, G, delta_v_ib):
        X_k1_k = self.get_X_k1_k(self.X, T, I, delta_v_ib)
        K = self.get_K(T, I)
        Z = self.update_Obs(I, G)
        h = self.update_hx(X_k1_k, I)
        self.X = X_k1_k + K.dot(Z - h)

    def update_P(self, T, I):
        K = self.get_K(T, I)
        H = self.update_H(I)
        P = self.get_P_k1_k(T, I)
        R = self.R_1
        self.P = (np.eye(9) - K.dot(H)).dot(P).dot((np.eye(9) - K.dot(H)).T) + K.dot(R).dot(K.T)

    # I: [ve, vn, vu, L, Lambda, h, psi,sigma, gamma] of INS
    def solver(self, I, X, Cbn):
        Lz = I[3] - X[6]
        Lambdaz = I[4] - X[7]
        hz = I[5] - X[8]

        if Lambdaz > math.pi:
            Lambdaz -= 2 * math.pi
        elif Lambdaz < -math.pi:
            Lambdaz += 2 * math.pi

        Vze = I[0] - X[3]
        Vzn = I[1] - X[4]
        Vzu = I[2] - X[5]

        Cnn = self.get_Cnn(X[0], X[1], X[2])
        Cnb = Cnn.dot(Cbn.T)

        Zq_0 = 0.5 * (1 + Cnb[0, 0] + Cnb[1, 1] + Cnb[2, 2]) ** 0.5
        Zq_1 = np.sign((Cnb[2, 1] - Cnb[1, 2])) * 0.5 * (1 + Cnb[0, 0] - Cnb[1, 1] - Cnb[2, 2]) ** 0.5
        Zq_2 = np.sign((Cnb[0, 2] - Cnb[2, 0])) * 0.5 * (1 - Cnb[0, 0] + Cnb[1, 1] - Cnb[2, 2]) ** 0.5
        Zq_3 = np.sign((Cnb[1, 0] - Cnb[0, 1])) * 0.5 * (1 - Cnb[0, 0] - Cnb[1, 1] + Cnb[2, 2]) ** 0.5

        if abs(Zq_0) < 1e-7:
            if abs(Zq_1) < 1e-7:
                Zq_2 = abs(Zq_2)
                if (Cnb(1, 2) + Cnb(2, 1)) > 0:
                    Zq_3 = abs(Zq_3)
                else:
                    Zq_3 = -abs(Zq_3)
            else:
                Zq_1 = abs(Zq_1)
                if (Cnb(0, 1) + Cnb(1, 0)) > 0:
                    Zq_2 = abs(Zq_2)
                else:
                    Zq_2 = -abs(Zq_2)
                if (Cnb(0, 2) + Cnb(2, 0)) > 0:
                    Zq_3 = abs(Zq_3)
                else:
                    Zq_3 = -abs(Zq_3)

        return Lz, Lambdaz, hz, Vze, Vzn, Vzu, Zq_0, Zq_1, Zq_2, Zq_3

    def get_Cnn(self, Phi_e, Phi_n, Phi_u):
        return np.array([[math.cos(Phi_e) * math.cos(Phi_u),
                          math.sin(Phi_n) * math.sin(Phi_u) - math.sin(Phi_e) * math.cos(Phi_n) * math.cos(Phi_u),
                          math.cos(Phi_n) * math.sin(Phi_u) + math.sin(Phi_n) * math.sin(Phi_e) * math.cos(Phi_u)],
                         [math.sin(Phi_e), math.cos(Phi_n) * math.cos(Phi_e), -math.sin(Phi_n) * math.cos(Phi_e)],
                         [-math.cos(Phi_e) * math.sin(Phi_u),
                          math.cos(Phi_n) * math.sin(Phi_u) * math.sin(Phi_e) + math.sin(Phi_n) * math.cos(Phi_u),
                          -math.sin(Phi_n) * math.sin(Phi_u) * math.sin(Phi_e) + math.cos(Phi_n) * math.cos(Phi_u)]])



