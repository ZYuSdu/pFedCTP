import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def metric_func(Pred, Y):
    pred = Pred.detach().cpu().numpy()
    y = Y.detach().cpu().numpy()
    result = {}
    times=pred.shape[-1]
    result['RMSE'], result['MAE'] = np.zeros(times), np.zeros(times)
    for i in range(times):
        y_i = y[:, :, i]
        pred_i = pred[:, :, i]
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        result['MAE'][i] += MAE
        result['RMSE'][i] += RMSE
    result['MAE']=np.round(result['MAE'],3)
    result['RMSE'] = np.round(result['RMSE'],3)
    print("predict the next "+str(times)+" steps:")
    return result

