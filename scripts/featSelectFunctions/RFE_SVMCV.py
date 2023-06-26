def RFE_SVM_v1(data, VarY, VarX, SVRParaSearchC, k=10, min_features_to_select=2, RFE_Scoring = "neg_mean_squared_error"):


    """"

    *Recursive Feature Elimination with SVR and parameter search* 

    The function returns the best selected variables, the RFE steps and the Mean test score for each steps 

    returns (c, VarX[rfecv.support_], rfecv_steps, rfecv.cv_results_["mean_test_score"]

    freddy.houndekindo@inrs.ca 
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.svm import SVR
    from sklearn.feature_selection import RFECV
    from IPython.display import clear_output

    
    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    def KfoldSVR(df, CParaCV, VarY, VarX, k=k, scoring=rrmse):

        
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
        from IPython.display import clear_output
        from sklearn.svm import SVR

        Model = SVR(kernel="linear", C=CParaCV)
        df = df.reset_index(drop=True).copy()
        Xscaler = StandardScaler()
        Yscaler = StandardScaler()

        kfold = KFold(k, shuffle=True)
        rrmsecv = []

        for trainix, testix in kfold.split(df):
            df_train, df_test = df.loc[trainix, :], df.loc[testix, :]    


            X_train = Xscaler.fit_transform(df_train[VarX])
            X_test = Xscaler.transform(df_test[VarX])
            Y_train = Yscaler.fit_transform(df_train[[VarY]]).ravel()
            Y_test = df_test[VarY]

            Model.fit(X_train, Y_train)

            Y_pred = Yscaler.inverse_transform(Model.predict(X_test).reshape(1, -1)).ravel()

            rrmsecv.append(scoring(Y_test, Y_pred))


        return np.mean(rrmsecv) 
    
    Result = []
    i = 0
    for c in SVRParaSearchC:
        print("Searching for SVR best parameter (C): " + str(i+1) + " / " + str(len(SVRParaSearchC)))
        clear_output(wait=True)
        Result.append(KfoldSVR(data, c, VarY, VarX))
        i += 1
    C_paraCV = SVRParaSearchC[np.argmin(Result)]
    
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    
    Model = SVR(kernel="linear", C=C_paraCV)
    
    X_train = Xscaler.fit_transform(data[VarX])
    Y_train = Yscaler.fit_transform(data[[VarY]]).ravel()
    min_features_to_select = 2
    
    rfecv = RFECV(estimator=Model, 
                  step=1, min_features_to_select= min_features_to_select, 
                  cv=KFold(k, shuffle=True), scoring=RFE_Scoring)
    
    print("Performing the RFE ...")
    rfecv.fit(X_train, Y_train)
    rfecv_steps = np.arange(min_features_to_select, len(VarX)+1)
    Mean_test_score = rfecv.cv_results_["mean_test_score"]
    
    return (VarX[rfecv.support_], rfecv_steps, Mean_test_score)
    
 
 
def RFE_SVM_v2(data, VarY, VarX, SVRParaSearchC, k=10, min_features_to_select=2, RFE_Scoring = "neg_mean_squared_error"):

    """"

    *Recursive Feature Elimination with SVR and parameter search* 

    The function returns the best C and selected variables, the RFE steps and the Mean test score for each steps 

    returns (VarX[rfecv.support_], rfecv_steps, Mean_test_score)

    freddy.houndekindo@inrs.ca 
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.svm import SVR
    from sklearn.feature_selection import RFECV
    from IPython.display import clear_output
    

    Xscaler = StandardScaler()
    Yscaler = StandardScaler()

    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    rfecv_steps = np.arange(min_features_to_select, len(VarX)+1)
    
    i = 0
    BestResults = None
    BestResultsMetric = 99999
 
    for c in SVRParaSearchC:
        print("Progress ... " + str(i+1) + " / " + str(len(SVRParaSearchC)))
        clear_output(wait=True)
        i += 1
        
        Model = SVR(kernel="linear", C=c)
        X_train = Xscaler.fit_transform(data[VarX])
        Y_train = Yscaler.fit_transform(data[[VarY]]).ravel()
        
        rfecv = RFECV(estimator=Model, 
                  step=1, min_features_to_select= min_features_to_select, 
                  cv=KFold(k, shuffle=True), scoring=RFE_Scoring)
        
        rfecv.fit(X_train, Y_train)
        Mean_test_score = np.abs(rfecv.cv_results_["mean_test_score"].mean())
        
        if Mean_test_score < BestResultsMetric:
            BestResultsMetric = Mean_test_score
            
            BestResults = (c, VarX[rfecv.support_], rfecv_steps, rfecv.cv_results_["mean_test_score"])
        
    return BestResults