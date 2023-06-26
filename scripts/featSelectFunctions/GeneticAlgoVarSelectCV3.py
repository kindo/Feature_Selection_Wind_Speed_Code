def GeneticAlgoVarSelectCV(data, npop, varExp, varPred, niter, FitnessWratio=0.7, CrossOverProba = 0.9, MutationProba = 0.1):
    
    
    """
       
    *Genetic Algo with ranking*
    
    The function returns the Best solution found and the minimum and the maximum solution at each generation
    return BestSolu, MinCost, MaxCost
    
    freddy.houndekindo@inrs.ca 
    
    """
    
    
    print("GeneticAlgoVarSelectCV3...")
    import numpy as np
    import pandas as pd 
    from sklearn.model_selection import KFold
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.stats import rankdata
    from IPython.display import clear_output
    def rrmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred)/obs)**2))
    
    def rmse(obs, Pred):
        return np.sqrt(np.mean(((obs - Pred))**2))

    def costLinearRegress(individual, data, varExp, varPred, k=10):
        rmseLS = []
        r2LS = []
        varExp = varExp[individual]
        kfold = KFold(k, shuffle=True)


        RegressionModel  = LinearRegression(fit_intercept=False)
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()


        for trainix, testix in kfold.split(data):
   
            df_train = data.loc[trainix, ]
            df_test = data.loc[testix, ]

            X_train = X_scaler.fit_transform(df_train[varExp].values)
            Y_train = Y_scaler.fit_transform(df_train[[varPred]]).ravel()

            X_test = X_scaler.transform(df_test[varExp].values)
            Y_test = df_test[varPred]
            RegressionModel.fit(X_train, Y_train)
            pred = Y_scaler.inverse_transform((RegressionModel.predict(X_test)).reshape(-1, 1)).ravel()

            rrmsescore = rmse(Y_test, pred)
            rmseLS.append(rrmsescore)
            
        return np.mean(rmseLS)

    def InitialRandomPoP(npop, varExp):
        return 1*(np.random.randn(npop, len(varExp)) > 0.5)

    def CrossOverUniform(Parent, nfeature):
    
        alpha = 1*(np.random.rand(nfeature) < 0.5)
        alpha_comp = 1 - alpha
        children = np.empty_like(Parent)
        children[0, :] = (Parent[0, :] * alpha) + (Parent[1, :] * alpha_comp)
        children[1, :] = (Parent[1, :] * alpha) + (Parent[0, :] * alpha_comp)
        return children

    def Mutation(Children, NbrGeneMutate, nfeature):
        Children_Mutate = Children.copy()
        genePos = np.random.choice(np.arange(nfeature), size=NbrGeneMutate, replace=False)
        Children_Mutate[0, genePos] = 1 - Children_Mutate[0, genePos]
        Children_Mutate[1, genePos] = 1 - Children_Mutate[1, genePos]
        return Children_Mutate


    def rank_selection(popcost, Population, FitnessWratio):
        print('FitnessWratio:', FitnessWratio)
        w1 = FitnessWratio
        w2 = 1 - w1
        
        popcost = 1/(popcost + 1)
        likelihood =  popcost / np.sum(popcost)
        ngenesMeasure = Population.sum(axis=1)
        ngenesMeasure = 1/(ngenesMeasure + 1)
        sparcity = ngenesMeasure / np.sum(ngenesMeasure) 
        
  
        return w1*likelihood + w2*sparcity


    def Parent_selection(Population, Probability, nparents = 2):
        ixParentSelect = np.random.choice(np.arange(Population.shape[0]), size=nparents, replace=False, p=Probability)
        return Population[ixParentSelect, :]
        
    def reproduction(Parent, nfeature, CrossOverProba, MutationProba):
        
        if np.random.rand() < CrossOverProba:
            children = CrossOverUniform(Parent, nfeature)
  
        else:
            children = Parent.copy()
            
        if np.random.rand() < MutationProba:
            NbrGeneMutate=np.random.randint(1, 5, 1)
            children = Mutation(children, NbrGeneMutate, nfeature)
        return children
    
    Population = InitialRandomPoP(npop, varExp)
    nfeature = len(varExp)
    MinCost = []
    MaxCost = []
    MeanCost = []
    BestSolu = Population[0, ].copy()
    Flag = False 
    
    for it in range(niter):
        print("Progress: "+str(it)+" iterations / "+str(niter) + " iterations")
        clear_output(wait=True)
        PopCost = np.array([costLinearRegress(np.bool8(individual),data,varExp, varPred, k=10) for individual in Population])
        
        Population_temp = Population.copy()
        
        Population_temp[0, :] = Population[np.argmin(PopCost), ]
                                                 
                                                 
        MinCost.append(PopCost.min())
        MaxCost.append(PopCost.max())
        MeanCost.append(PopCost.mean())

        #save best result 
        if Flag == False:
        
            if PopCost.min() < costLinearRegress(np.bool8(BestSolu),data,varExp, varPred, k=10):
                BestSolu = Population[np.argmin(PopCost), ].copy()
                Bestrmse = PopCost.min()
                Flag = True
        else:
            if PopCost.min() < Bestrmse:
                BestSolu = Population[np.argmin(PopCost), ].copy()
                Bestrmse = PopCost.min()

        if np.any(PopCost <= 0.10):
            break
            
        PopProba = rank_selection(PopCost, Population, FitnessWratio)
        
        if npop%2 == 1:
        
            for i in range(1, Population.shape[0] - 1, 2):
                Parent = Parent_selection(Population, PopProba)
                Population_temp[i:i+2, :] = reproduction(Parent, nfeature, CrossOverProba, MutationProba).copy()
        else:
            for i in range(1, Population.shape[0], 2):
                if i == Population.shape[0] - 1:
                    
                    Parent = Parent_selection(Population, PopProba)
                    Population_temp[i:i+2, :] = reproduction(Parent, nfeature, CrossOverProba, MutationProba)[0, :].copy()
                else:
                    Parent = Parent_selection(Population, PopProba)
                    Population_temp[i:i+2, :] = reproduction(Parent, nfeature, CrossOverProba, MutationProba).copy()
            
        Population = Population_temp.copy()
    return varExp[np.bool8(BestSolu)], np.array(MinCost), np.array(MeanCost)

