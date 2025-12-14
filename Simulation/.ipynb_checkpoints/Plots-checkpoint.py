import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
import numpy as np
from typing import Dict, Iterable, Optional
from datetime import datetime
from Simulation import Simulation 

# -------- Global style (journal look) --------
def _set_paper_style():
    sns.set_theme(style="whitegrid", font_scale=1.35)
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.title_fontsize": 12,
        "legend.fontsize": 11
    })

# -------- Helpers --------
def _to_long_df(estimates: Dict[str, np.ndarray],
                y_true: float,
                allocations: Iterable[str], 
                kind = "MF") -> pd.DataFrame:
    """
    Build a tidy DataFrame from arrays of estimates and allocation labels.
    `estimates` must contain keys 'Y_SM' and 'Y_PML' mapping to 1D arrays of equal length.
    `allocations` is a list/array of same length with allocation strategy per run.
    """
    if kind == "MF":
        Y_SM = np.asarray(estimates["Y_SM"])
        Y_PML = np.asarray(estimates["Y_PML"])
        alloc = np.asarray(list(allocations))
        assert Y_SM.shape == Y_PML.shape == alloc.shape, "Arrays must have same length."
    
        df = pd.DataFrame({
            "Y_SM": Y_SM,
            "Y_PML": Y_PML,
            "allocation": alloc
        })

        # long format for plotting
        df_long = df.melt(id_vars="allocation", value_vars=["Y_SM", "Y_PML"],
                          var_name="Estimator", value_name="Estimate")
        df_long["Estimator"] = df_long["Estimator"].str.replace("Y_", "", regex=False)
        df_long["Y_true"] = y_true
        df_long["RelError"] = (df_long["Estimate"] - y_true) / y_true
        
    elif kind == "STS":
        
        Y_STS = np.asarray(estimates["Y_STS"])
        alloc = np.asarray(list(allocations))
        assert Y_STS.shape == alloc.shape, "Arrays must have same length."
    
        df = pd.DataFrame({
            "Y_STS": Y_STS,
            "allocation": alloc
        })

        # long format for plotting
        df_long = df.melt(id_vars="allocation", value_vars=["Y_STS"],
                          var_name="Estimator", value_name="Estimate")
        df_long["Estimator"] = df_long["Estimator"].str.replace("Y_", "", regex=False)
        df_long["Y_true"] = y_true
        df_long["RelError"] = (df_long["Estimate"] - y_true) / y_true
        
    return df_long

def _mse_group(df_long: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """
    Compute MSE per (allocation, Estimator) and return 95% CI via bootstrap.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Must contain columns ["allocation", "Estimator", "Estimate", "Y_true"].
    n_boot : int
        Number of bootstrap resamples.
    
    Returns
    -------
    pd.DataFrame
        Columns: ["allocation", "Estimator", "MSE", "CI_low", "CI_high", "n"]
    """
    tmp = df_long.assign(SE=lambda d: (d["Estimate"] - d["Y_true"])**2)
    out_rows = []

    for (alloc, est), group in tmp.groupby(["allocation", "Estimator"]):
        se_values = group["SE"].values
        n = len(se_values)
        
        # Bootstrap the mean of squared errors
        res = bootstrap((se_values,), np.mean, confidence_level=0.95, n_resamples=n_boot, method='percentile')
        mse = se_values.mean()
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        
        out_rows.append({
            "allocation": alloc,
            "Estimator": est,
            "MSE": mse,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "n": n
        })
    
    return pd.DataFrame(out_rows)



class Plots:

    """
    Some functions to make some comparisions
    """

    @staticmethod
    def plot_estimator_distribution(estimates: Dict[str, np.ndarray],
                                    y_true: float,
                                    allocations: Iterable[str],
                                    kind: str = "violin",
                                    save: Optional[str] = None):
        """
        Distribution of Y_SM vs Y_PML across runs, split by allocation.
        kind: 'violin' or 'box'
        """
        _set_paper_style()
        df = _to_long_df(estimates, y_true, allocations)
    
        plt.figure(figsize=(8.2, 4.8))
        if kind == "violin":
            ax = sns.violinplot(data=df, x="allocation", y="Estimate", hue="Estimator",
                                inner="quartile", cut=0)
        else:
            ax = sns.boxplot(data=df, x="allocation", y="Estimate", hue="Estimator",
                             width=0.55, linewidth=1.2)
        # True line
        ax.axhline(y_true, color="black", linestyle="--", linewidth=1.2, label="True value")
        ax.set_xlabel("Allocation strategy")
        ax.set_ylabel("Estimator value")
        ax.set_title("Distribution of Estimators across runs")
        # combine legend entries (avoid duplicate line handle)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left", title="Estimator")
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches="tight")
        plt.show()


    @staticmethod
    def plot_relative_error(estimates: Dict[str, np.ndarray],
                            y_true: float,
                            allocations: Iterable[str],
                            save: Optional[str] = None, 
                            kind = "MF"):
        """
        Relative error = (estimate - Y_true)/Y_true for SM vs PML, per allocation.
        """
        _set_paper_style()
        df = _to_long_df(estimates, y_true, allocations, kind = kind)

        roma_friends_rgb3 = [
                        (85/256, 107/256, 47/256),    # Olive
                        (27/256, 42/256, 73/256),     # Navy
                        (217/256, 198/256, 165/256)   # Sand Beige
                            ]
        roma_friends_rgb2 = [(27/256, 42/256, 73/256),     # Navy
                    (217/256, 198/256, 165/256)   # Sand Beige
                            ]

        
        plt.figure(figsize=(8.2, 4.8))
        ax = sns.boxplot(
            data=df,
            x="allocation",
            y="RelError",
            hue="Estimator",
            palette= roma_friends_rgb2 if kind=="MF" else roma_friends_rgb3,
            showcaps=True,
            fliersize=2,
            boxprops={"alpha": 0.7}
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Allocation strategy")
        ax.set_ylabel("Relative Bias")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Estimator")
    
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_mse(estimates: Dict[str, np.ndarray],
                 y_true: float,
                 allocations: Iterable[str],
                 save: Optional[str] = None, 
                 kind = "MF"):
        """
        Plot MSE per estimator and allocation using pre-computed bootstrap CI.
        """
        _set_paper_style()
        df = _to_long_df(estimates, y_true, allocations, kind)
        mse_df = _mse_group(df)

        roma_friends_rgb3 = [
                        (85/256, 107/256, 47/256),    # Olive
                        (27/256, 42/256, 73/256),     # Navy
                        (217/256, 198/256, 165/256)   # Sand Beige
                            ]
        roma_friends_rgb2 = [(27/256, 42/256, 73/256),     # Navy
                    (217/256, 198/256, 165/256) ]  # Sand Beige
        
        print(mse_df)
        
        plt.figure(figsize=(8.2, 4.8))
        
        # Get unique estimators and allocations for positioning
        estimators = mse_df["Estimator"].unique()
        allocations_list = mse_df["allocation"].unique()
        n_alloc = len(allocations_list)
        width = 0.8 / len(estimators)  # width of each point

        kolors = roma_friends_rgb2 if kind=="MF" else roma_friends_rgb3
        for i, est in enumerate(estimators):
            est_df = mse_df[mse_df["Estimator"] == est]
            x = np.arange(n_alloc) - 0.4 + i * width + width/2
            y = est_df["MSE"]
            yerr = [y - est_df["CI_low"], est_df["CI_high"] - y]  # asymmetric error bars
            plt.errorbar(
                x, y, yerr=yerr, fmt='o', label=est, capsize=5, color = kolors[i],
            )
        
        plt.xticks(np.arange(n_alloc), allocations_list)
        plt.xlabel("Allocation strategy")
        plt.ylabel("Mean Squared Error")
        plt.legend(title="Estimator", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches="tight")
        plt.show()

    @staticmethod
    def analyze_population_mean(
        estimates, 
        R, 
        true_value,
        M_total=17, 
        alpha=0.05, 
        n_resamples=10000,
        save_path=None,
        kind = "STS"
    ):
        """
        Compute bootstrap CIs for proportional vs optimal allocation estimates
        and plot comparison against true population mean.
        
        Parameters
        ----------
        estimates : dict
            Dictionary containing Monte Carlo estimates.
            Should include "mu_STS" with 2*R elements (first R = proportional, next R = optimal).
        R : int
            Maximum number of runs for each allocation scheme.
        true_value : float
            True population mean.
        M_total : int, default=8
            Number of Monte Carlo batch sizes (log-spaced).
        alpha : float, default=0.05
            Significance level for bootstrap confidence intervals.
        n_resamples : int, default=10000
            Number of bootstrap resamples.
        save_path : str, optional
            Path to save plot. If None, no file is saved.
        
        Returns
        -------
        dict
            Dictionary with keys "batch_sizes", "prop", "opt" containing results arrays.
        """
    
        def compute_mean_and_ci(data):
            """Return mean, lower, upper CI using bootstrap."""
            mean_val = np.mean(data)
            if len(data) > 1:
                res = bootstrap(
                    (data,), 
                    statistic=np.mean, 
                    confidence_level=1 - alpha,
                    n_resamples=n_resamples, 
                    method="basic"
                )
                return mean_val, res.confidence_interval.low, res.confidence_interval.high
            else:
                return mean_val, mean_val, mean_val
    
        # Split proportional vs optimal allocation
        if kind == "STS":
            estimates_prop = estimates["mu_STS"][:R]/true_value
            estimates_opt  = estimates["mu_STS"][R:]/true_value
            true_value *= 1/true_value

        if kind == "SM":
            estimates_prop = estimates["Y_SM"][:R]/true_value
            estimates_opt  = estimates["Y_SM"][R:]/true_value
            true_value *= 1/true_value

        if kind == "PML":
            estimates_prop = estimates["Y_PML"][:R]/true_value
            estimates_opt  = estimates["Y_PML"][R:]/true_value
            true_value *= 1/true_value
        
        # Batch sizes (log-spaced integers)
        batch_sizes = np.geomspace(2, R, M_total).astype(int)
    
        results = {"prop": [], "opt": []}
    
        # Compute results
        for batch in batch_sizes:
            results["prop"].append(compute_mean_and_ci(estimates_prop[:batch]))
            results["opt"].append(compute_mean_and_ci(estimates_opt[:batch]))
    
        # Convert to arrays
        prop = np.array(results["prop"])
        opt  = np.array(results["opt"])
    
        # --- Plot ---
        plt.figure(figsize=(8, 5))
    
        # Proportional allocation
        plt.errorbar(batch_sizes, prop[:,0],
                     yerr=[prop[:,0]-prop[:,1], prop[:,2]-prop[:,0]],
                     fmt='o', capsize=5, label='Proportional Allocation', color="#8E001C")
    
        # Optimal allocation
        plt.errorbar(batch_sizes, opt[:,0],
                     yerr=[opt[:,0]-opt[:,1], opt[:,2]-opt[:,0]],
                     fmt='o', capsize=5, label='Optimal Cost Allocation', color="#FFB300")
    
        # True population mean
        plt.axhline(true_value, color='black', linestyle='--', label='True Population Mean')
    
        plt.xscale('log')
        plt.xlabel('Number of Monte Carlo Runs')
        plt.ylabel('Population Mean Estimate')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.yticks([.975, .98, .99, 1, 1.01, 1.02, 1.03], 
                  [.975, .98, .99, 1, 1.01, 1.02, 1.03]
                  )
    
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
        return {"batch_sizes": batch_sizes, "prop": prop, "opt": opt}






    