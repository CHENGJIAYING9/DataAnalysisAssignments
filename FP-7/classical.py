import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    sp = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / sp


def run_hypothesis_analyses(df):
    df = df.copy()
    df['high_absence'] = df['absences'] > df['absences'].median()
    df['high_grade'] = df['grade'] >= 10
    df['low_grade'] = df['grade'] < 10

    # low-grade
    L = df[df['low_grade']]
    L_low = L[L['high_absence'] == False]['grade']
    L_high = L[L['high_absence'] == True]['grade']
    U_low = mannwhitneyu(L_low, L_high)
    d_low = cohens_d(L_low, L_high)

    # high-grade
    H = df[df['high_grade']]
    H_low = H[H['high_absence'] == False]['grade']
    H_high = H[H['high_absence'] == True]['grade']
    U_high = mannwhitneyu(H_low, H_high)
    d_high = cohens_d(H_low, H_high)

    return {
        "low_p": U_low.pvalue,
        "high_p": U_high.pvalue,
        "low_d": d_low,
        "high_d": d_high
    }


def plot_absence_effect(df):
    df = df.copy()
    df['high_absence'] = df['absences'] > df['absences'].median()
    df['high_grade'] = df['grade'] >= 10
    df['low_grade'] = df['grade'] < 10

    L = df[df['low_grade']]
    H = df[df['high_grade']]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(ax=axs[0], x='high_absence', y='grade', data=L)
    axs[0].set_title('Low-grade Group')

    sns.boxplot(ax=axs[1], x='high_absence', y='grade', data=H)
    axs[1].set_title('High-grade Group')

    plt.tight_layout()
    plt.show()


def plot_absence_trend(df):
    plt.figure(figsize=(6,4))
    sns.regplot(x='absences', y='grade', data=df, scatter_kws={'alpha':0.5})
    plt.title("Absences vs Grade")
    plt.tight_layout()
    plt.show()
