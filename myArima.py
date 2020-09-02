import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm


class Arima_Class:
    def __init__(self, arima_para, seasonal_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        P = arima_para['P']
        D = arima_para['D']
        Q = arima_para['Q']
        self.params = list(itertools.product(p, d, q, P, D, Q))
        self.seasonal = seasonal_para
        # Generate all different combinations of p, q and q triplets
        # self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                             for x in list(itertools.product(p, d, q))]

    def fit(self, ts):
        warnings.filterwarnings("ignore")
        results_list = []
        for param in tqdm(self.params):
            if (param[0] == 0 and param[2] == 0) or (param[3] == 0 and param[5] == 0):
                continue
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=(param[0], param[1], param[2]),
                                                seasonal_order=(param[3], param[4], param[5], self.seasonal),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=0)

                print('ARIMA{}x{}seasonal - AIC:{}'.format((param[0], param[1], param[2]),
                                                           (param[3], param[4], param[5], self.seasonal),
                                                           results.aic))
                results_list.append([(param[0], param[1], param[2]), (param[3], param[4], param[5], self.seasonal), results.aic])
            except:
                continue
        results_list = np.array(results_list)
        lowest_AIC = np.argmin(results_list[:, 2])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('ARIMA{}x{}seasonal with lowest_AIC:{}'.format(
            results_list[lowest_AIC, 0], results_list[lowest_AIC, 1], results_list[lowest_AIC, 2]))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        mod = sm.tsa.statespace.SARIMAX(ts,
                                        order=results_list[lowest_AIC, 0],
                                        seasonal_order=results_list[lowest_AIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.final_result = mod.fit()
        print('Final model summary:')
        print(self.final_result.summary().tables[1])
        print('Final model diagnostics:')
        self.final_result.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        plt.savefig('model_diagnostics.png', dpi=300)
        plt.show()

    def pred(self, train_data, test_data, dynamic, ts_label):

        pred_dynamic = self.final_result.get_prediction(
            start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=dynamic, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()
        ax = train_data.plot(label='observed', figsize=(15, 10))

        if dynamic == False:
            pred_dynamic.predicted_mean.plot(label='One-step ahead Forecast', ax=ax)
        else:
            pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

        ax.fill_between(pred_dynamic_ci.index,
                        pred_dynamic_ci.iloc[:, 0],
                        pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        ax.fill_betweenx(ax.get_ylim(), train_data.index[0], train_data.index[-1],
                         alpha=.1, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.legend()
        plt.tight_layout()
        if dynamic == False:
            plt.savefig(ts_label + '_one_step_pred.png', dpi=300)
        else:
            plt.savefig(ts_label + '_dynamic_pred.png', dpi=300)
        plt.show()

    def forcast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        ax = ts.plot(label='observed', figsize=(15, 10))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast in Future')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Time')
        ax.set_ylabel(ts_label)
        plt.tight_layout()
        plt.savefig(ts_label + '_forcast.png', dpi=300)
        plt.legend()
        plt.show()
