def PredictTest(data_train, data_test, varX, varY, C_svr_cv=None, Model = "lr" ):
    
    import numpy as np
    import pandas as pd 
    from sklearn.model_selection import KFold
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.stats import rankdata
    from IPython.display import clear_output
    from KfoldLinearModels import KfoldSVRLinear
    
    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    
    if Model == "lr":
        RegressionModel  = LinearRegression()   
    
    elif Model == "svr":
        rrmseLs = []
        for c in C_svr_cv:
            rrmseLs.append(KfoldSVRLinear(data_train, varY, varX, c=c)["rrmse"])
        
        Best_C = C_svr_cv[np.argmin(rrmseLs)]
        RegressionModel = SVR(C=Best_C)
 
    X_train = X_scaler.fit_transform(data_train[varX])
    Y_train = Y_scaler.fit_transform(data_train[[varY]]).ravel()
    X_test = X_scaler.transform(data_test[varX])
    RegressionModel.fit(X_train, Y_train)
    Y_pred = Y_scaler.inverse_transform(RegressionModel.predict(X_test).reshape(-1, 1)).ravel()
    
    return Y_pred