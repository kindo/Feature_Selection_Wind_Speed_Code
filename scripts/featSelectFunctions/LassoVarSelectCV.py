def LassoVarSelectCV(df, varY, varX, alphaCV=[0.23]):

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score, explained_variance_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold 
    from IPython.display import clear_output


    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    Lasso_rrmse = []
    kfold = KFold(10, shuffle=True)
    i = 0
    for a in alphaCV:
        print("Progress ..." + str(i) + " / "+ str(len(alphaCV)))
        clear_output(wait=True)
        Model = Lasso(alpha=a, max_iter=5000, fit_intercept=False)
        
        LSRMSE = []
       
        
        for train_ix, test_ix in kfold.split(df):
            df_train = df.loc[train_ix, ]
            df_test = df.loc[test_ix, ]
            
            X_train = Xscaler.fit_transform(df_train[varX])
            X_test = Xscaler.transform(df_test[varX])
            Y_train = Yscaler.fit_transform(df_train[[varY]]).ravel()
        
            Model.fit(X_train, Y_train)
            pred = Yscaler.inverse_transform(Model.predict(X_test).reshape(-1, 1)).ravel()

            rrmse_val = rrmse(df_test[varY], pred)
            LSRMSE.append(rrmse_val)

        i += 1

        Lasso_rrmse.append(np.mean(LSRMSE))
        
    
    Model = Lasso(alpha=alphaCV[np.argmin(Lasso_rrmse)], max_iter=2000)
    Lasso_Coef = []
    
    for train_ix, test_ix in kfold.split(df):
        df_train = df.loc[train_ix, ]
        df_test = df.loc[test_ix, ]

        X_train = Xscaler.fit_transform(df_train[varX])
        X_test = Xscaler.transform(df_test[varX])
        Y_train = Yscaler.fit_transform(df_train[[varY]]).ravel()

        Model.fit(X_train, Y_train)
        Lasso_Coef.append(Model.coef_)
        
    LassoFeatures = varX[np.array(Lasso_Coef).sum(axis=0) != 0]
    return LassoFeatures