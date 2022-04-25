import numpy as np


def get_m_k(N_f, s_rate, octave_fraction):
    f_bin = s_rate / N_f

    k = np.arange(N_f // 2 + 1)
    first_half = np.floor(0.5 * calculate_P_f(k*f_bin, octave_fraction)/f_bin)

    k = np.arange(N_f // 2 - 1) + N_f // 2 + 1
    second_half = np.floor(0.5 * calculate_P_f((N_f-k)*f_bin, octave_fraction)/f_bin)

    return np.hstack((first_half, second_half)).astype(np.int)


def calculate_P_f(f, octave_fraction):
    f_U = 2 ** (0.5 * octave_fraction) * f
    f_L = 0.5 ** (0.5 * octave_fraction) * f
    return f_U - f_L


def smooth_complex(N_f, s_rate, octave_fraction, H):

    # get ²H after Equation (12)
    # n x n matrix
    H2 = list()
    for k in range(N_f):
        H2.append(np.roll(H, k))
    H2 = np.vstack(H2)

    # get m(k) for ideal lowpass (rectagular window) after equation (9)
    # 1 x n Vector
    m_k = get_m_k(N_f, s_rate, octave_fraction)

    # get the smoothing matrix ²W_sm after equation (8)
    # m x n Matrix
    W_sm_k = get_W_sm_k(N_f, np.max(m_k))

    # calculate ²H_cs after Equation (13)
    # m x n Matrix
    H2_cs = np.matmul(W_sm_k, H2)

    # calculate/get ¹H_cs after Equation (14)
    # 1 x n Vektor
    H1_cs = np.array([H2_cs[m_k[k], k] for k in range(N_f)])

    return H1_cs


def get_W_sm_k(N_f, m_max):
    # return smoothing function for rectangular window
    # m = is the smoothing index corresponding to the length of the half-window
    W_sm_list = list()
    for m in range(m_max+1):
        if m == 0:
            W_sm_list.append(np.hstack( (np.ones(1),
                                        np.zeros(N_f - 1 ))))
        else:
            W_sm_list.append(np.hstack((1 / (2 * m + 1) * np.ones(m+1),
                                        np.zeros(N_f - 2 * (m + 1) + 1),
                                        1 / (2 * m + 1) * np.ones(m))))
    return np.vstack(W_sm_list)
