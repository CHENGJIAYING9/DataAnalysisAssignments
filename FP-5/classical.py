import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

def cohens_d(x1, x2):
    """Calculate Cohen's d for two independent samples."""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(x1) - np.mean(x2)) / s_pooled
    return d


def run_hypothesis_analyses(df):
    """
    Perform hypothesis tests on student performance.
    Analyze how absence frequency affects grades
    under different performance groups (low vs high achievers).
    """

    df = df.copy()  # operate on a copy to avoid modifying original df

    # Binary flags for high/low absences and high/low grades
    df['high_absence'] = df['absences'] > df['absences'].median()

    df['high_grade'] = df['grade'] >= 10
    df['low_grade'] = df['grade'] < 10

    # ----------------------------
    # Group 1: Low-grade students
    # ----------------------------
    L = df[df['low_grade']]
    L_low_abs = L[L['high_absence'] == False]['grade']
    L_high_abs = L[L['high_absence'] == True]['grade']
    U_low = mannwhitneyu(L_low_abs, L_high_abs, alternative='two-sided')

    # ----------------------------
    # Group 2: High-grade students
    # ----------------------------
    H = df[df['high_grade']]
    H_low_abs = H[H['high_absence'] == False]['grade']
    H_high_abs = H[H['high_absence'] == True]['grade']
    U_high = mannwhitneyu(H_low_abs, H_high_abs, alternative='two-sided')

    print("Low-grade group: Mann–Whitney U-test p-value (low vs high absences):", U_low.pvalue)
    print("High-grade group: Mann–Whitney U-test p-value (low vs high absences):", U_high.pvalue)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

    sns.boxplot(ax=axs[0], x='high_absence', y='grade', data=L)
    axs[0].set_title('Effect of Absences on Grades (Low-grade Group)')
    axs[0].set_xlabel('Absence Level (False: Low, True: High)')
    axs[0].set_ylabel('Grade')

    sns.boxplot(ax=axs[1], x='high_absence', y='grade', data=H)
    axs[1].set_title('Effect of Absences on Grades (High-grade Group)')
    axs[1].set_xlabel('Absence Level (False: Low, True: High)')
    axs[1].set_ylabel('Grade')

    plt.show()
