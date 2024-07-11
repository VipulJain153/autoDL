from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
coder=False
def preprocess(df,target):
    attribs =  []
    strAttribs = []
    for c in df.columns:
        global coder
        if c!=target:
            dtype = df.loc[:,c].dtype
            if dtype == object or dtype == pd.CategoricalDtype:
                attribs.append(c)
            elif dtype == pd.StringDtype:
                strAttribs.append(c)
        else:
            dtype = df.loc[:, c].dtype
            if dtype == object or dtype == pd.CategoricalDtype:
                coder=True
    if strAttribs:
        df.drop(strAttribs,axis=1,inplace=True)
    try:
        corrs = list(df.corr()[target].sort_values(ascending=False)[1:4].index)
        df["new1"] = df[corrs[0]]+df[corrs[1]]
        df["new2"] = df[corrs[2]]+df[corrs[0]]
    except:
        pass
    pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("std_scler", StandardScaler())
    ])
    otherAttribs=[]
    for i in df.columns:
        if i not in attribs and i!=target:
            otherAttribs.append(i)
    transformer = ColumnTransformer([
    ("attribs",OneHotEncoder(),attribs),
    ("other",pipeline,otherAttribs),
    ])
    encoder=OrdinalEncoder()

    try:
        x = df.drop(target,axis=1)
        X=transformer.fit_transform(x)
        y=df[target].to_numpy()
        if coder:
            y=encoder.fit_transform(y.reshape(-1, 1))
        pca = PCA(n_components=2)
        X=pca.fit_transform(X)
        return X,y
    except Exception as e:
        x = df.drop(target, axis=1)
        X = transformer.fit_transform(x)
        y = df[target].to_numpy()
        if coder:
            y=encoder.fit_transform(y.reshape(-1,1))
        return X, y
if __name__=="__main__":
    preprocess(pd.read_csv("source_data.csv"),"Sex")