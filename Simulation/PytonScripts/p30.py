# p30.py
# Calcolo delle stime avanzate (SM e PML) per Multi-Frame
import numpy as np
import scipy.linalg
from p20 import estrai_campione
from p10 import genera_popolazione

def calcola_stime():
    camp = estrai_campione()
    pop = camp['pop']
    Q = pop['Q']
    D = pop['D']
    dBinary = pop['dBinary']
    md = pop['md']
    nq = camp['nq']
    Nq = pop['Nq']
    deff = np.ones(Q)  # deff placeholder (efficienza disegno)
    s_q = camp['s_q']
    pi_kq = camp['pi_kq']
    m_kq = camp['m_kq']
    y_kq = pop['y_kq']
    # DESIGN WEIGHTS
    w_kq = [[1/pi for pi in pi_kq[q]] for q in range(Q)]
    # UNIT MULTIPLICITY (dal pop)
    m_kq_full = [[m for m in m_kq[q]] for q in range(Q)]
    # DOMAIN MULTIPLICITY
    md_vec = md.copy()  # D
    # ALPHA (SIMPLE MULTIPLICITY ADJUSTMENT)
    alpha_kq = [[1/m for m in m_kq[q]] for q in range(Q)]
    # SIMPLE MULTIPLICITY ESTIMATOR
    Y_SM = sum(
        y_kq[q][j] * alpha_kq[q][j] * w_kq[q][j]
        for q in range(Q)
        for j in range(len(s_q[q]))
    )
    # --- PML ESTIMATOR ---
    # Nhat_q: somma pesi per frame
    Nhat_q = [sum(w_kq[q]) for q in range(Q)]
    # Nhat_d|q: somma pesi per ogni dominio d in ogni frame q
    # g_list: (g, id, iq) per ogni dominio-frame
    g_list = pop['g_list']
    Nhat_dq = []
    y_dq = []
    wPML_kdq = []
    for g, id, iq in g_list:
        q = iq-1
        d = id-1
        # print(f"g={g}, d={1+d}, q={1+q}")
        # trova indici del campione s_q che sono nel dominio d
        dom_idx = [i for i, k in enumerate(s_q[q]) if pop['domain_kd'][pop['frames'][q][k], d] == 1]
        # print(dom_idx)
        Nhat = sum(w_kq[q][i] for i in dom_idx)
        Nhat_dq.append(Nhat)
        ysum = sum(y_kq[q][i] for i in dom_idx)
        y_dq.append(ysum)
        # placeholder per pesi PML (da aggiornare dopo NPML)
        wPML_kdq.append([0.0 for _ in dom_idx])
    # Hhat_d|q: matrice D x Q, copia di dBinary, poi aggiornata con Nhat_g
    Hhat_dq = dBinary.copy().astype(float)
    g_idx = 0
    for d in range(D):
        for q in range(Q):
            if dBinary[d, q] == 1:
                Hhat_dq[d, q] = Nhat_dq[g_idx]
                g_idx += 1
    # NPML0: per ogni d, prendi Hhat_dq[d, q] dove n_d|q massimo
    NPML0 = np.zeros(D)
    for d in range(D):
        idx_q = np.where(dBinary[d] == 1)[0]
        if len(idx_q) == 1:
            NPML0[d] = Hhat_dq[d, idx_q[0]]
        else:
            # prendi q con massimo campione
            n_dq = [sum([pop['domain_kd'][pop['frames'][q][k], d] for k in s_q[q]]) for q in idx_q]
            qmax = idx_q[np.argmax(n_dq)]
            NPML0[d] = Hhat_dq[d, qmax]
    # Iterative routine Lohr & Rao (JASA 2006)
    Mplus = scipy.linalg.pinv(dBinary)
    fdeff = [nq[q] / Nq[q] / deff[q] for q in range(Q)]
    def A(x):
        A1 = (np.identity(D) - dBinary @ Mplus) @ scipy.linalg.inv(np.diag(x)) @ np.diag(dBinary @ fdeff)
        A2 = dBinary.T
        return np.concatenate((A1, A2))
    def b(x):
        b1 = (np.identity(D) - dBinary @ Mplus) @ scipy.linalg.inv(np.diag(x)) @ Hhat_dq @ fdeff
        b2 = np.array(Nq)
        return np.concatenate((b1, b2))
    Ncheck_d = np.zeros(D)
    x_old = np.array(NPML0)
    for _ in range(25):
        try:
            A_mat = A(x_old)
            b_vec = b(x_old)
            x_new = np.linalg.pinv(A_mat) @ b_vec
        except Exception:
            x_new = x_old
        if np.allclose(x_old, x_new):
            break
        x_old = x_new
    NPML_d = x_new
    # PML weights per g (dominio-frame)
    wPML_kdq = []
    g_idx = 0
    for g, id, iq in g_list:
        q = iq-1
        d = id-1
        # print(f"g={g}, d={1+d}, q={1+q}")
        dom_idx = [i for i, k in enumerate(s_q[q]) if pop['domain_kd'][pop['frames'][q][k], d] == 1]
        Nhat = Nhat_dq[g_idx]
        wPML = [NPML_d[d] * w_kq[q][i] / Nhat if Nhat > 0 else 0.0 for i in dom_idx]
        wPML_kdq.append(wPML)
        g_idx += 1
    # t_ywPML_g: somma y_kq * wPML_kdq per g
    t_ywPML_g = []
    g_idx = 0
    for g, id, iq in g_list:
        q = iq-1
        d = id-1
        dom_idx = [i for i, k in enumerate(s_q[q]) if pop['domain_kd'][pop['frames'][q][k], d] == 1]
        t_yw = sum(y_kq[q][i] * wPML_kdq[g_idx][j] for j, i in enumerate(dom_idx))
        t_ywPML_g.append(t_yw)
        g_idx += 1
    # PML alpha (pPML_g)
    pPML_g = []
    for g, id, iq in g_list:
        d = id-1
        idx_q = np.where(dBinary[d] == 1)[0]
        if md[d] == 1:
            pPML_g.append(1.0)
        elif md[d] == 2:
            q1, q2 = idx_q
            f1 = fdeff[q1] * Hhat_dq[d, q1]
            f2 = fdeff[q2] * Hhat_dq[d, q2]
            if iq-1 == q1:
                pPML_g.append(f1 / (f1 + f2) if (f1+f2)>0 else 0.5)
            else:
                pPML_g.append(f2 / (f1 + f2) if (f1+f2)>0 else 0.5)
        elif md[d] == 3:
            q1, q2, q3 = idx_q
            f1 = fdeff[q1] * Hhat_dq[d, q1]
            f2 = fdeff[q2] * Hhat_dq[d, q2]
            f3 = fdeff[q3] * Hhat_dq[d, q3]
            tot = f1 + f2 + f3
            if iq-1 == q1:
                pPML_g.append(f1 / tot if tot>0 else 1/3)
            elif iq-1 == q2:
                pPML_g.append(f2 / tot if tot>0 else 1/3)
            else:
                pPML_g.append(f3 / tot if tot>0 else 1/3)
        else:
            pPML_g.append(1.0/md[d])
    # PML ESTIMATE
    Y_PML = sum(pPML_g[g] * t_ywPML_g[g] for g in range(len(g_list)))
    return {
        'Y_SM': Y_SM,
        'w_kq': w_kq,
        'm_kq': m_kq_full,
        'md': md_vec,
        'alpha_kq': alpha_kq,
        'Nhat_q': Nhat_q,
        'Nhat_dq': Nhat_dq,
        'Hhat_dq': Hhat_dq,
        'NPML0': NPML0,
        'NPML_d': NPML_d,
        'wPML_kdq': wPML_kdq,
        't_ywPML_g': t_ywPML_g,
        'pPML_g': pPML_g,
        'Y_PML': Y_PML
    }

if __name__ == "__main__":
    stime = calcola_stime()
    print(f"Stima semplice Y_SM: {stime['Y_SM']:.3f}")
    print(f"Stima PML Y_PML: {stime['Y_PML']:.3f}")
