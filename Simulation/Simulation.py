import numpy as np
import scipy.linalg
from scipy.special import softmax
from sampling import Sampling
import optimal_cost_allocation as oca
import sys

class Simulation:
    """
    Simulation framework for multi-frame survey sampling.
    
    Includes:
    - Population generation
    - Sample extraction from frames
    - Estimation (SM, PML, NPML) following Lohr & Rao (2006, JASA).
    """

    def __init__(self, params: dict = None, reproduce= True, allocation= None):
        """
        Initialize the simulation with default or user-specified parameters.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of simulation parameters. If None, defaults are used.
        """

        if allocation is None:
            self.allocation = None
        elif allocation == "equal":
            self.allocation = "equal"
        elif allocation == "optimal_cost":
            self.allocation = "optimal_cost"
        else:
            ValueError("allocation must be either 'equal' or 'optimal_cost'")
        
        self.params = params if params is not None else self.get_parameters()
        self.reproduce= reproduce
        
    def get_parameters(self):
        """
        Returns the default simulation parameters.
        
        Returns
        -------
        dict
        """

        dict_of_params = {
            'Q': 3,
            'N': 1000,
            'Nrun': 2,
            'my': 100,
            'mx': 400,
            'sy': 9,
            'sx': 4,
            'royx': 0.85,
            'mean': [100, 400],
            'cov': [
                [9**2, 0.85*9*4],
                [0.85*9*4, 4**2],
            ],
            'random_seed': 6,
            'Nd_pct': [0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
            'sample_design': ["srs", "srs", "srs"],
            'fq': [0.05, 0.05, 0.05],
        }

        if self.allocation == "equal":
            D = 2**dict_of_params['Q']-1
            dict_of_params["Nd_pct"] = [1/D for kk in range(D)]
        elif self.allocation == "optimal_cost":            
            #Use the optimal MF estimator for a linear cost function
            cost = np.array([4, 4, 4, 4, 4, 4, 4]) # cost per domain
            domain_totals = np.array([24, 17, 15, 14, 13, 12, 5])
            alloc = (domain_totals*np.sqrt(cost))/(domain_totals*np.sqrt(cost)).sum()
            dict_of_params["Nd_pct"] = alloc/alloc.sum()
        else:
            dict_of_params["Nd_pct"] = dict_of_params["Nd_pct"]

        return dict_of_params

    def generate_population(self):
        """
        Generate a synthetic population with overlapping domains and frames.
        
        Returns
        -------
        dict
            Dictionary containing the simulated population structure and values.
        """
        p = self.params

        if self.reproduce:
            np.random.seed(p['random_seed'])

        Q, N = p['Q'], p['N']
        mean, cov = np.array(p['mean']), np.array(p['cov'])
        Nd_pct = np.array(p['Nd_pct'])

        D = 2**Q - 1
        d = np.arange(1, D+1)
        q = np.arange(1, Q+1)

        # Domain-frame incidence matrix
        dBinary = np.array([list(map(int, np.binary_repr(i, Q)[::-1])) for i in range(1, D+1)])
        md = dBinary.sum(axis=1)

        # Generate (y, x) > 0
        yx = []
        while len(yx) < N:
            vals = np.random.multivariate_normal(mean, cov)
            if np.all(vals > 0):
                yx.append(vals)
        yx = np.array(yx) / 100.0
        y, x = yx[:, 0], yx[:, 1]

        # Mean checks
        assert np.isclose(y.mean()*100, p['my'], atol=2)
        assert np.isclose(x.mean()*100, p['mx'], atol=8)

        # Domain sizes
        Nd = np.round(Nd_pct * N).astype(int)
        diff = N - Nd.sum()
        if diff != 0:
            Nd[0] += diff
        assert Nd.sum() == N

        # Assign units to domains
        labels = np.arange(N)
        np.random.shuffle(labels)
        domains, start = [], 0
        for nd in Nd:
            domains.append(labels[start:start+nd].tolist())
            start += nd

        # Frame sizes
        Nq = []
        for iq in range(Q):
            mask = dBinary[:, iq] == 1
            Nq.append(sum([Nd[idom] for idom in range(D) if mask[idom]]))

        # Frame membership
        frames = []
        for iq in range(Q):
            idxs = []
            for idom in range(D):
                if dBinary[idom, iq] == 1:
                    idxs.extend(domains[idom])
            frames.append(sorted(idxs))

        # Values by frame
        x_kq = [[x[i] for i in frame] for frame in frames]
        y_kq = [[y[i] for i in frame] for frame in frames]

        # Totals
        T_y, T_x = y.sum(), x.sum()
        T_yd = [y[domains[idom]].sum() for idom in range(D)]
        T_xd = [x[domains[idom]].sum() for idom in range(D)]

        # Multiplicities
        m_kq = []
        for iq in range(Q):
            mkq = []
            for idom in range(D):
                if dBinary[idom, iq] == 1:
                    mkq += [md[idom]] * len(domains[idom])
            m_kq.append(mkq)

        # Domain membership matrix
        domain_kd = np.zeros((N, D), dtype=int)
        for idom, dom in enumerate(domains):
            domain_kd[dom, idom] = 1

        # Group list g = (group index, domain, frame)
        g_list, g = [], 1
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

    def extract_sample(self, population):
        """
        Extracts samples from each frame according to the chosen designs.
        
        Parameters
        ----------
        population : dict
            The dictionary produced by `generate_population`.
        
        Returns
        -------
        dict
            Dictionary containing selected units, inclusion probabilities, and frame-domain splits.
        """
        p, Q = self.params, population['Q']
        Nq, frames, x_kq, y_kq, m_kq_pop = population['Nq'], population['frames'], population['x_kq'], population['y_kq'], population['m_kq']
        nq = [int(np.ceil(Nq[q] * p['fq'][q])) for q in range(Q)]
        random_seed = p['random_seed']
        sample_designs = p['sample_design']

        # Storage
        s_q, pi_kq_pop, pi_kq, pi_klq, m_kq, rejects_q = [], [], [], [], [], []

        for q in range(Q):
            design = sample_designs[q] if isinstance(sample_designs, list) else sample_designs
            N = len(frames[q])
            n = nq[q]
            x = x_kq[q] if design in ("pareto", "sampford") else None

            sampling = Sampling.Make(design, N=N, n=n, x=x)
            if self.reproduce:
                sample, rejects = sampling.get_sample(seed=random_seed + q)
            else:
                sample, rejects = sampling.get_sample()
            
            s_q.append(sample)
            rejects_q.append(rejects)
            pi_kq_pop.append(list(sampling.get_πi()))
            pi_kq.append(list(sampling.get_πi_sample(sample)))
            pi_klq.append(sampling.get_πij_sample(sample))
            m_kq.append([m_kq_pop[q][i] for i in sample])

        # Classify into domain-frame subsamples
        dBinary, D, domains = population['dBinary'], population['D'], population['domains']
        s_dq, d_q = [], []
        for idom in range(D):
            for iq in range(Q):
                if dBinary[idom, iq] == 1:
                    frame_idxs = frames[iq]
                    dom_set = set(domains[idom])
                    sample = s_q[iq]
                    idxs_in_dom = [i for i, idx in enumerate(sample) if frame_idxs[idx] in dom_set]
                    s_dq.append(idxs_in_dom)
                    d_q.append((idom, iq))

        n_dq = [len(idxs) for idxs in s_dq]

        return {
            's_q': s_q,
            'pi_kq_pop': pi_kq_pop,
            'pi_kq': pi_kq,
            'pi_klq': pi_klq,
            'm_kq': m_kq,
            'nq': nq,
            'frames': frames,
            'x_kq': x_kq,
            'y_kq': y_kq,
            's_dq': s_dq,
            'd_q': d_q,
            'n_dq': n_dq,
            'rejects_q': rejects_q,
            'pop': population
        }

    def compute_estimates(self, sample):
        """
        Computes SM, PML, and NPML estimates from the drawn sample.
        
        Parameters
        ----------
        sample : dict
            Output of `extract_sample`.
        
        Returns
        -------
        dict
            Dictionary of estimates and intermediate quantities.
        """

        
        pop = sample['pop']
        Q = pop['Q']
        D = pop['D']
        dBinary = pop['dBinary']
        md = pop['md']
        nq = sample['nq']
        Nq = pop['Nq']
        deff = np.ones(Q)  # deff placeholder (efficienza disegno)
        s_q = sample['s_q']
        pi_kq = sample['pi_kq']
        m_kq = sample['m_kq']
        y_kq = sample['y_kq']
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
        

    def run_pipeline(self):
        """
        Full simulation pipeline: population → sample → estimates.
        
        Returns
        -------
        dict
        """
        pop = self.generate_population()
        sample = self.extract_sample(pop)
        estimates = self.compute_estimates(sample)
        return {'population': pop, 'sample': sample, 'estimates': estimates}