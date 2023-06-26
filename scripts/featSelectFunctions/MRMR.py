def mRMR_VarSelect(df, VarX, VarY, k):
    
    import numpy as np
    from sklearn.feature_selection import r_regression, f_regression
    import pandas as pd
    from IPython.display import clear_output
    k = k - 1
    
    Fscore = f_regression(df[VarX], df[VarY])

    selected = [VarX[np.argmax(Fscore)]]
    NotSelected = list(set(VarX).difference(selected))   

    for i in range(k):
        print("Progress ... "+str(i)+" / "+str(k))
        clear_output(wait=True)
        corr_not_selected = np.array([np.abs(r_regression(df[NotSelected], df[selected[j]])) for j in range(len(selected))]).mean(axis=0)
        mRMRScore = f_regression(df[NotSelected], df[VarY])[0]/corr_not_selected
        selecVar = NotSelected[np.argmax(mRMRScore)]
        selected.append(selecVar)
        NotSelected.remove(selecVar)
    
    return selected 