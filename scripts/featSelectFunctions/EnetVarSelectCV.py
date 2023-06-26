def EnetVarSelectCV(df, varY, varX, alphaCV, L1ratioCV):

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error 
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold 
    from IPython.display import clear_output

    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))

    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    kfold = KFold(10, shuffle=True)
    
    ParaEnet = []
    ENet_score = []
    #Select the best parameters of the model by cross-validation 
    i = 0
    for alpha in alphaCV:
        for l1ratio in L1ratioCV:
            print("Progress ... " + str(i + 1) + " / " + str(len(alphaCV)*len(L1ratioCV)))
            clear_output(wait=True)
            ParaEnet.append((alpha, l1ratio))
            Model = ElasticNet(alpha=alpha, l1_ratio=l1ratio, max_iter=50000, fit_intercept=False)
            scoreLS = []
            i += 1
            for train_ix, test_ix in kfold.split(df):
                df_train = df.loc[train_ix, ]
                df_test = df.loc[test_ix, ]

                X_train = Xscaler.fit_transform(df_train[varX])
                X_test = Xscaler.transform(df_test[varX])
                Y_train = Yscaler.fit_transform(df_train[[varY]]).ravel()

                Model.fit(X_train, Y_train)
                pred = Yscaler.inverse_transform(Model.predict(X_test).reshape(-1, 1)).ravel()

                score = r2_score(df_test[varY], pred)
                scoreLS.append(score)


            ENet_score.append(np.mean(scoreLS))

    BestParaEnet = ParaEnet[np.argmax(ENet_score)]
    
    Model = ElasticNet(alpha=BestParaEnet[0], l1_ratio=BestParaEnet[1], fit_intercept=False)
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    EnetCoef = []
    for train_ix, test_ix in kfold.split(df):
        df_train = df.loc[train_ix, ]
        df_test = df.loc[test_ix, ]

        X_train = Xscaler.fit_transform(df_train[varX])
        X_test = Xscaler.transform(df_test[varX])
        Y_train = Yscaler.fit_transform(df_train[[varY]]).ravel()

        Model.fit(X_train, Y_train)
        EnetCoef.append(Model.coef_)
    
    EnetFeatures = varX[np.array(EnetCoef).min(axis=0) != 0]
    return EnetFeatures, BestParaEnet