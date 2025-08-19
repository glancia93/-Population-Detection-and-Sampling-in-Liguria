# parametri.py
# Modulo per la gestione dei parametri globali del progetto

def get_parametri():
    """Restituisce i parametri principali come dizionario."""
    parametri = {
        'Q': 3,
        'N': 10000,
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
        'sample_design': ["srs", "sampford", "sampford"],
        'fq': [0.05, 0.05, 0.05],
    }
    return parametri
