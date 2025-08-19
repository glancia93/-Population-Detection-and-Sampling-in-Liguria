# p10.py
# Generazione della popolazione
import numpy as np
from parametri import get_parametri

def genera_popolazione():
    p = get_parametri()
    np.random.seed(p['random_seed'])
    Q = p['Q']
    N = p['N']
    mean = np.array(p['mean'])
    cov = np.array(p['cov'])
    Nd_pct = np.array(p['Nd_pct'])
    D = 2**Q - 1
    d = np.arange(1, D+1)
    q = np.arange(1, Q+1)
    # dBinary: matrice D x Q con rappresentazione binaria
    dBinary = np.array([list(map(int, np.binary_repr(i, Q)[::-1])) for i in range(1, D+1)])
    md = dBinary.sum(axis=1)
    # Generazione popolazione
    yx = []
    while len(yx) < N:
        vals = np.random.multivariate_normal(mean, cov)
        if np.all(vals > 0):
            yx.append(vals)
    yx = np.array(yx) / 100.0
    y = yx[:,0]
    x = yx[:,1]
    # Controllo media
    assert np.isclose(y.mean()*100, p['my'], atol=2), f"Media y fuori range: {y.mean()*100}" 
    assert np.isclose(x.mean()*100, p['mx'], atol=8), f"Media x fuori range: {x.mean()*100}" 
    # Domini
    Nd = np.round(Nd_pct * N).astype(int)
    diff = N - Nd.sum()
    if diff != 0:
        Nd[0] += diff
    assert Nd.sum() == N
    # Shuffle e assegnazione domini
    labels = np.arange(N)
    np.random.shuffle(labels)
    domains = []
    start = 0
    for nd in Nd:
        domains.append(labels[start:start+nd].tolist())
        start += nd
    # Calcolo Nq
    Nq = []
    for iq in range(Q):
        mask = dBinary[:, iq] == 1
        Nq.append(sum([Nd[idom] for idom in range(D) if mask[idom]]))
    # Frames
    frames = []
    for iq in range(Q):
        idxs = []
        for idom in range(D):
            if dBinary[idom, iq] == 1:
                idxs.extend(domains[idom])
        frames.append(sorted(idxs))
    # x_kq, y_kq
    x_kq = [[x[i] for i in frame] for frame in frames]
    y_kq = [[y[i] for i in frame] for frame in frames]
    # Totali
    T_y = y.sum()
    T_x = x.sum()
    T_yd = [y[domains[idom]].sum() for idom in range(D)]
    T_xd = [x[domains[idom]].sum() for idom in range(D)]
    # m_kq
    m_kq = []
    for iq in range(Q):
        mkq = []
        for idom in range(D):
            if dBinary[idom, iq] == 1:
                mkq += [md[idom]] * len(domains[idom])
        m_kq.append(mkq)
    # domain_kd
    domain_kd = np.zeros((N, D), dtype=int)
    for idom, dom in enumerate(domains):
        domain_kd[dom, idom] = 1
    # print(domain_kd.shape)
    # g_list
    g_list = []
    g = 1
    for idom in range(D):
        for iq in range(Q):
            if dBinary[idom, iq] == 1:
                g_list.append((g, idom+1, iq+1))
                g += 1
    return {
        'Q': Q, 'D': D, 'N': N, 'd': d, 'q': q, 'dBinary': dBinary, 'md': md,
        'y': y, 'x': x, 'domains': domains, 'frames': frames,
        'x_kq': x_kq, 'y_kq': y_kq, 'T_y': T_y, 'T_x': T_x,
        'T_yd': T_yd, 'T_xd': T_xd, 'Nq': Nq, 'Nd': Nd,
        'm_kq': m_kq, 'domain_kd': domain_kd, 'g_list': g_list
    }

if __name__ == "__main__":
    pop = genera_popolazione()
    print(f"Popolazione generata: N={pop['N']}, Q={pop['Q']}, D={pop['D']}")
