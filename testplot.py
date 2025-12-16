
import numpy as np
import matplotlib.pyplot as plt
import time
from data_processing import load_influnet_data
from guesses import best_SEIRHET, best_SEIR




colors_model1 = ['#aec7e8', '#1f77b4', '#0c4b8e']
colors_model2 = ['#ff9896', '#d62728', '#8c1c13']
color_model3 = '#2ca02c'
ur = 1
N = 59_500_000


year = ["2014", "2015", "2016", "2017", "2018" , "2019", "2023"]
#


for y in year:
    start_time = time.time()
    data = load_influnet_data(y, -1)
    weeks = data["weeks"]
    dates = data["data_dates"]
    dates_test = data["data_dates_test"]
    cases = data["data_cases_v2"]
    cases_test = data["data_cases_test_v2"]
    init_beta = data["init_beta"]
    s0_range = np.random.uniform(1e-6, 1e-2, size=3)
    b_range = np.random.uniform(0.5, 5, size=3)
    a_range = np.random.uniform(0.1, 5, size=3)
    E0_range = np.random.randint(1, 20001, size=3)
    R0_range = np.random.randint(0.2 * N, 0.9 * N, size=3)
    p_range = np.random.uniform(0.1, 10, size=3)
    best_seir = best_SEIR(dates, dates_test, cases, cases_test, init_beta, N, E0_range, R0_range)
    best_het = best_SEIRHET(dates, dates_test, cases, cases_test, init_beta, N, E0_range, p_range)
    plt.figure(figsize=(9, 6))
    plt.plot(dates, cases, 'bo', label='Observed Cases')
    plt.plot(dates, best_het["best_fitted"], 'b', linestyle='--', linewidth=2.4, label=f'$\phi(x)$ ~ Gamma({"%.2f" % best_het["best_p"]}), R_0 = {"%.2f" % (best_het["best_beta"] / 4)}')
    plt.plot(dates, best_seir["best_fitted"], 'r', linestyle='--', linewidth=2.4, label=f'SEIR-hom. R_0 = {"%.2f" % (best_seir["best_beta"][0] * (1 - best_seir["best_R0"] / N - 2 * best_seir["best_E0"] / N)/ 4)}, R(0) = {"%.2f" % (best_seir["best_R0"] / N)}')
    plt.title(f" {y}-{int(y)+1} winter season. $R^2$-scores: SEIR: {"%.4f" % best_seir["best_r2_score"]}, SEIR-HET: {"%.4f" % best_het["best_r2_score"]}")
    plt.xticks(dates, labels=weeks, rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Cases")
    plt.legend()
    plt.savefig(f"fit{y}_test3")
    plt.show()




