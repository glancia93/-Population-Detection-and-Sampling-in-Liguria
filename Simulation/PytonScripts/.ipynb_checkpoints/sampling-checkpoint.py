#
import numpy as np

#
from functools import lru_cache
cache = lru_cache(maxsize=4096)

#
dbg = 0

# calcolo n ** t
@cache
def pow_n_alla_t(n, t):
    return pow(np.longdouble(n), np.longdouble(t))

#
class Sampling():
    """
    Classe base per esecuzione del campionamento.
    """
    #
    def __init__(self, N=None, n=None, x=None):
        if dbg: print("Sampling // CTOR")
        self.N = N
        self.n = n 
        self.x = x
        if self.N is None: self.N = len(self.x)
        if dbg: print("N:", N)
        if dbg: print("n:", n)
        if dbg: print("x:", x)
    #
    @staticmethod
    def Make(design, N=None, n=None, x=None):
        """
        Ritorna un oggetto di campionamento in base al valore "design"
        """
        if dbg: print(f"Sampling.Make({design})")
        match design:
            case "srs": return SRS(N=N, n=n)
            case "pareto": return Pareto(x=x, n=n)
            case "sampford": return Sampford(x=x, n=n)
    #
    @cache
    def get_πi(self):
        """
        Ritorna il vettore delle probabilità di inclusione di I ordine, dimensione N
        """
        πi = np.zeros((self.N,))
        for i in range(self.N):
            πi[i] = self.calc_πi(i)
        return πi
    #
    def get_πi_sample(self, sample):
        """
        Ritorna le probabilità di inclusione di I ordine, per i soli elementi del campione, dimensione n
        """
        πi_sample = np.zeros( (len(sample),) )
        for i in range(len(sample)):
            πi_sample[i] = self.calc_πi(sample[i])
        return πi_sample
    #
    @cache
    def get_πij(self):
        """
        Ritorna la matrice delle probabilità di inclusione di II ordine, dimensione (N,N)
        """
        πij = np.zeros( (self.N, self.N) )
        for i in range(self.N):
            πij[i][i] = self.calc_πi(i)
        for i in range(self.N):
            for j in range(i+1, self.N):
                πij[i][j] = self.calc_πij(i,j)
                πij[j][i] = πij[i][j]
        return πij
    #
    def get_πij_sample(self, sample):
        """
        Ritorna le probabilità di inclusione di II ordine, per i soli elementi del campione, dimensione (n,n)
        """
        πij_sample = np.zeros( (len(sample), len(sample)) )
        for i in range(len(sample)):
            πij_sample[i][i] = self.calc_πi(sample[i])
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                πij_sample[i][j] = self.calc_πij(sample[i],sample[j])
                πij_sample[j][i] = πij_sample[i][j]
        return πij_sample
    #
    def get_sample(self, seed=None):
        """
        Estrae un campione e ritorna gli indici relativi (da 0 a N-1) e l'eventuale numero di rifiuti in sede di campionamento
        """
        if seed is not None: np.random.seed(seed)
        return self.sample()
    
#
class SRS(Sampling):
    """
    Classe per esecuzione del campionamento con disegno SRS.
    """
    #
    def __init__(self, N, n):
        """
        Parametri del costruttore:
        N = dimensione popolazione
        n = dimensione campione
        """
        if dbg: print("SRS // CTOR")
        super().__init__(N=N, n=n)
    #
    def design(self): return "srs"
    #
    @cache
    def calc_πi(self, i):
        return self.n / self.N
    #
    @cache
    def calc_πij(self, i, j):
        if i == j: return self.calc_πi(i) 
        return ( self.n * ( self.n-1 ) ) / ( self.N * ( self.N-1 ) )
    #
    def sample(self):
        ret = np.random.choice(size=self.n, a=range(self.N), replace=False)
        return (sorted(ret.tolist()), 0)

#
class Pareto(Sampling):
    """
    Classe per esecuzione del campionamento con disegno Pareto.
    """
    #
    def __init__(self, x, n):
        """
        Parametri del costruttore:
        x = variabile x per calcolo probabilità di estrazione
        n = dimensione campione
        """
        if dbg: print("Pareto // CTOR")
        super().__init__(x=x, n=n)
        self.totx = np.sum(x)
        self.den = np.sum([ self.calc_πi(k) * (1-self.calc_πi(k)) for k in range(self.N)])
    #
    def design(self): return "pareto"
    #
    @cache
    def calc_πi(self, i):
        return self.n * self.x[i] / self.totx
    #
    @cache
    def calc_πij(self, i, j):
        if i == j: return self.calc_πi(i) 
        return self.calc_πi(i) * self.calc_πi(j) * ( 1 - (1-self.calc_πi(i)) * (1-self.calc_πi(i)) / self.den )
    #
    def sample(self):
        val = []
        for j in range(self.N):
            uni = np.random.uniform()
            factor = 1/(self.den**2) * self.calc_πi(j) * (1-self.calc_πi(j)) * (self.calc_πi(j)-1/2)
            # print(q,j,factor,np.exp(-factor))
            val.append( (uni/(1-uni)) / ( np.exp(-factor) * self.calc_πi(j) / (1-self.calc_πi(j)) ) )
        # print(q, val)
        idx = np.argsort(val)[:self.n] # primi N più piccoli
        # print(q, idx)
        # print(q, qry.iframe_labels(q))
        ret = np.take(range(self.N), idx)
        return (sorted(ret.tolist()), 0)

#
class Sampford(Sampling):
    """
    Classe per esecuzione del campionamento con disegno Sampford.
    """
    #
    def __init__(self, x, n):
        """
        Parametri del costruttore:
        x = variabile x per calcolo probabilità di estrazione
        n = dimensione campione
        """
        if dbg: print("Sampford // CTOR")
        super().__init__(x=x, n=n)
        self.totx = np.sum(x)
        self.πi = self.get_πi()
        self.pi = self.πi / self.n
        self.λi = self.pi / (1 - self.πi)
        # calcolo di kn (normalizzazione)
        self.kn:np.longdouble = 0.0
        for t in range(1, self.n + 1):
            self.kn += t * self.L(self.n - t) / pow_n_alla_t(self.n, t)
        self.kn = 1.0 / self.kn
        if dbg: print("kn", self.kn)

    #
    def design(self): return "sampford"
    #
    @cache
    def calc_πi(self, i):
        return self.n * self.x[i] / self.totx
    #
    @cache
    def calc_πij(self, i, j):
        if i == j: return self.calc_πi(i)
        return self.kn * self.λi[i] * self.λi[j] * self.calc_Φij(i, j)
    #
    def sample(self):
        cum_x = np.cumsum(self.x)
        cum_λ = np.cumsum(self.λi)
        rifiuti = 0
        while True:
            u = np.random.uniform(0, self.totx)
            i = np.searchsorted(cum_x, u)
            labels = set([i])
            for _ in range(self.n - 1):
                u = np.random.uniform(0, cum_λ[-1])
                j = np.searchsorted(cum_λ, u)
                if j in labels:
                    labels = set()
                    rifiuti += 1
                    break
                labels.add(j)
            if len(labels) == self.n:
                labels = list(labels)
                return (sorted(labels), rifiuti)

    # L[m]
    @cache
    def L(self, m):
        if m == 0: return 1.0
        sum_L = 0.0
        for r in range(1, m + 1):
            sum_L += ((-1) ** (r - 1)) * self.sum_λi_alla_r(r) * self.L(m - r)
        val = sum_L / m
        return val

    # Lnoni[m, i]
    @cache
    def Lnoni(self, m, i):
        if m == 0: return 1.0 # self.L(0)
        val = self.L(m) - self.λi[i] * self.Lnoni(m - 1, i)
        return val

    # Lnoninonj[m, i, j]
    @cache
    def Lnoninonj(self, m, i, j):
        if m == 0: return 1.0 # self.L(0)
        val = self.Lnoni(m, i) - self.λi[j] * self.Lnoninonj(m - 1, i, j)
        return val

    # calcolo di Φij
    @cache
    def calc_Φij(self, i, j):
        sum_pii_pij = self.pi[i] + self.pi[j]
        s:np.longdouble = 0.0
        for t in range(2, self.n+1):
            s += (t - self.n * sum_pii_pij) * self.Lnoninonj(self.n-t, i, j) / pow_n_alla_t(self.n, t-2)
        return s

    # calcolo np.sum(self.λi ** r)
    @cache
    def sum_λi_alla_r(self, r):
        return np.sum(self.λi ** r)
    