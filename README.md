# k-nearest-neighbors
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import json

iris=load_iris()
X=iris['data']
y=iris['target']

df=pd.DataFrame(X,columns=iris['feature_names'])
df['target']=y
sampled_df=df.groupby('target').apply(lambda x:x.sample(n=5)).reset_index(drop=True)
X_sampled=sampled_df.drop('target',axis=1)
y_sampled=sampled_df['target']

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_sampled,y_sampled)

with open('knn_model.pkl','wb') as model_file:
    pickle.dump(knn_model,model_file)
    
def classify_iris_knn(sample_json):
    with open('knn_model.pkl','rb') as model_file:
        loaded_knn_model=pickle.load(model_file)
        
    sample_df=pd.DataFrame([json.loads(sample_json)])
    
    prediction=loaded_knn_model.predict(sample_df)
    
    return iris['target_names'][prediction][0]

example_input=json.dumps({
    "sepal length (cm)":5.1,
    "sepal width (cm)":3.5,
    "petal length (cm)":1.4,
    "petal width (cm)":0.2
    
})

predicted_class_knn=classify_iris_knn(example_input)
predicted_class_knn


