# p20.py
# Estrazione del campione secondo le istruzioni dettagliate
import numpy as np
from parametri import get_parametri
from p10 import genera_popolazione
from sampling import Sampling

def estrai_campione():
    p = get_parametri()
    pop = genera_popolazione()
    Q = pop['Q']
    Nq = pop['Nq']
    sample_designs = p['sample_design']
    x_kq = pop['x_kq']
    y_kq = pop['y_kq']
    frames = pop['frames']
    m_kq_pop = pop['m_kq']
    nq = [int(np.ceil(Nq[q] * p['fq'][q])) for q in range(Q)]
    random_seed = p['random_seed']
    Nrun = p.get('Nrun', 1)

    # 1. Estrazione campioni e calcolo probabilità di inclusione
    s_q = []  # lista di liste Q x nq
    pi_kq_pop = []  # lista di liste Q x Nq
    pi_kq = []      # lista di liste Q x nq
    pi_klq = []     # lista di matrici Q x (nq x nq)
    m_kq = []       # lista di liste Q x nq
    rejects_q = []  # lista di rifiuti per ogni frame

    for q in range(Q):
        design = sample_designs[q] if isinstance(sample_designs, list) else sample_designs
        N = len(frames[q])
        n = nq[q]
        x = x_kq[q] if design in ("pareto", "sampford") else None
        # Crea oggetto Sampling
        sampling = Sampling.Make(design, N=N, n=n, x=x)
        # Estrai campione (seed 42 + q per ripetibilità)
        sample, rejects = sampling.get_sample(seed=random_seed + q)
        s_q.append(sample)
        rejects_q.append(rejects)
        # Probabilità di inclusione I ordine (popolazione)
        pi_kq_pop.append(list(sampling.get_πi()))
        # Probabilità di inclusione I ordine (solo campione)
        pi_kq.append(list(sampling.get_πi_sample(sample)))
        # Probabilità di inclusione II ordine (solo campione)
        pi_klq.append(sampling.get_πij_sample(sample))
        # m_kq per i campioni
        m_kq.append([m_kq_pop[q][i] for i in sample])

    # 2. Classificazione s_q secondo la relazione d|q (domini per frame)
    dBinary = pop['dBinary']
    D = pop['D']
    domains = pop['domains']
    s_dq = []  # lista di liste per ogni g corrispondente a d|q
    d_q = []   # lista di tuple (d, q) per ogni g
    for idom in range(D):
        for iq in range(Q):
            if dBinary[idom, iq] == 1:
                # Trova gli indici del campione di frame iq che appartengono al dominio idom
                frame_idxs = frames[iq]
                dom_set = set(domains[idom])
                # sample contiene indici relativi a frame iq
                sample = s_q[iq]
                # Trova gli indici nel sample che corrispondono a unità del dominio idom
                idxs_in_dom = [i for i, idx in enumerate(sample) if frame_idxs[idx] in dom_set]
                s_dq.append(idxs_in_dom)
                d_q.append((idom, iq))
    # 3. Calcola le dimensioni dei sottocampioni dominio
    n_dq = [len(idxs) for idxs in s_dq]

    return {
        's_q': s_q,  # campioni estratti per frame
        'pi_kq_pop': pi_kq_pop,  # prob. inclusione I ordine popolazione
        'pi_kq': pi_kq,          # prob. inclusione I ordine campione
        'pi_klq': pi_klq,        # prob. inclusione II ordine campione
        'm_kq': m_kq,            # molteplicità per campione
        'nq': nq,                # dimensione campione per frame
        'frames': frames,        # frame di partenza
        'x_kq': x_kq,            # x per frame
        'y_kq': y_kq,            # y per frame
        'pop': pop,              # popolazione completa
        's_dq': s_dq,            # sottocampioni dominio
        'd_q': d_q,              # lista (d, q) per ogni g
        'n_dq': n_dq,            # dimensioni sottocampioni dominio
        'rejects_q': rejects_q   # rifiuti per ogni frame
    }

if __name__ == "__main__":
    camp = estrai_campione()
    print(f"Campioni estratti: {[len(s) for s in camp['s_q']]}")
    print(f"Dimensioni sottocampioni dominio: {camp['n_dq']}")
