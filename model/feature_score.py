import pandas as pd
from pycox.evaluation import EvalSurv
import numpy as np
def feature_score(model,x_test,durations_test,events_test,first_cindex,time_length):
    testDf = pd.DataFrame(x_test)
    testNo = testDf.shape[0]
    testCol = testDf.shape[1]
    time_interval=6
    score = []
    for i in range(testCol):
        print(i)
        errors = []
        for count in range(100):
            testTmp = testDf.copy()
            if i >=0 and i <=12 :#continuous
                sigma = np.std(testDf[i])  #
                epsilon = 10
                testTmp[i] = testTmp[i] + np.random.normal(0, epsilon * sigma, testNo)
            else:  # categorical
                s = np.random.binomial(1, 0.5, testNo)  #
                testTmp[i] = testTmp[i] * (1 - s) + (1 - testTmp[i]) * s
            Y_test_pred_tmp = model.predict(np.asarray(testTmp))
            pred_tmp = np.array(Y_test_pred_tmp[0:time_length])
            pred_dead_tmp = pred_tmp[:, :, 1]
            cif1 = pd.DataFrame(pred_dead_tmp, np.arange(time_length) + 1)
            ev1 = EvalSurv(1 - cif1, durations_test // time_interval, events_test == 1, censor_surv='km')
            c_index = ev1.concordance_td('antolini')
            error = first_cindex - c_index
            errors.append(error)
        score.append(np.mean(errors))