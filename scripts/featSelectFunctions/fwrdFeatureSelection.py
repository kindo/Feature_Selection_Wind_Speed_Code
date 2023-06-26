def FwrdFeatureSelection(data, varX, varY):

    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X = pd.DataFrame(np.ones(data.shape[0]), columns=["Const"])
    Y = Y_scaler.fit_transform(data[[varY]]).ravel()

    model = sm.OLS(Y, X)
    ModelR = model.fit()
    oldscore = ModelR.rsquared_adj

    newscore = 99999

    flag = 0
    while newscore > oldscore:
        flag += 1
        
        if flag == 2:
            X.drop(labels=['Const'], axis=1, inplace=True)
            
        if flag >=2:
            
            oldscore = newscore
        results = []


        for var in varX:
            X[var] = X_scaler.fit_transform(data[[var]])
            model = sm.OLS(Y, X)
            resModel = model.fit()
            results.append(resModel.rsquared_adj)
            X.drop(labels=var, axis=1, inplace=True)



        bestvar = varX[np.argmax(results)]
        varX.remove(bestvar)
        X[bestvar] = X_scaler.fit_transform(data[[bestvar]])
        newscore = np.max(results)

    X.drop(labels=bestvar, axis=1, inplace=True)

    return X.columns.values