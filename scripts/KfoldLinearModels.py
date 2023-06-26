def KfoldLinearRegression(df, VarY, VarX, k=10):
    
    """
    
    ***LinearRegression K fold cross=validation 
    
    Inputs: df: data; VarY: Observed values; VarX: Feature Matrix; k: number pf folds
    
    returns: dictionary: rrmse; r2; explained variance
    
    freddy.houndekindo@inrs.ca
    
    """
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd
    from sklearn.metrics import explained_variance_score
    
    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    Model = LinearRegression()
    df = df.reset_index(drop=True).copy()
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    kfold = KFold(n_splits=k, shuffle=True)
    r2cv = []
    rrmsecv = []
    EXVarcv = []
    for trainix, testix in kfold.split(df):
        df_train, df_test = df.loc[trainix, :], df.loc[testix, :]    
        
        X_train = Xscaler.fit_transform(df_train[VarX])
        X_test = Xscaler.transform(df_test[VarX])
        Y_train = Yscaler.fit_transform(df_train[[VarY]]).ravel()     
        Y_test = df_test[VarY]
        
        Model.fit(X_train, Y_train)
        
        Y_pred = Yscaler.inverse_transform(Model.predict(X_test).reshape(1, -1)).ravel()
        
        r2cv.append(r2_score(Y_test, Y_pred))
        rrmsecv.append(rrmse(Y_test, Y_pred))
        EXVarcv.append(explained_variance_score(Y_test, Y_pred))
        
    return {"rrmse":np.mean(rrmsecv), "r2":np.mean(r2cv), "Explain_Variance":np.mean(EXVarcv)}     
    
    
def KfoldSVRLinear(df, VarY, VarX, k=10, c=1):

 
    """
    
    ***SVR Linear kernel regression with  K fold cross=validation 
    
    Inputs: df: data; VarY: Observed values; VarX: Feature Matrix; k: number pf folds
    
    returns: dictionary: rrmse; r2; explained variance
    
    freddy.houndekindo@inrs.ca
    
    """
    
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, explained_variance_score
    import numpy as np
    import pandas as pd
    
    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    Model = SVR(kernel = "linear", C=c)
    df = df.reset_index(drop=True).copy()
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    kfold = KFold(n_splits=k, shuffle=True)
    r2cv = []
    rrmsecv = []
    EXVarcv = []
    for trainix, testix in kfold.split(df):
        df_train, df_test = df.loc[trainix, :], df.loc[testix, :]    
        
        X_train = Xscaler.fit_transform(df_train[VarX])
        X_test = Xscaler.transform(df_test[VarX])
        Y_train = Yscaler.fit_transform(df_train[[VarY]]).ravel()     
        Y_test = df_test[VarY]
        
        Model.fit(X_train, Y_train)
        
        Y_pred = Yscaler.inverse_transform(Model.predict(X_test).reshape(1, -1)).ravel()
        
        r2cv.append(r2_score(Y_test, Y_pred))
        rrmsecv.append(rrmse(Y_test, Y_pred))
        EXVarcv.append(explained_variance_score(Y_test, Y_pred))
        
    return {"rrmse":np.mean(rrmsecv), "r2":np.mean(r2cv), "Explain_Variance":np.mean(EXVarcv)}     