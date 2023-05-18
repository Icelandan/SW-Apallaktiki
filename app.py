import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


file=st.file_uploader("Insert file here")

df = pd.read_csv(file, sep = "\t")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Supervised
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# Fit classifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)

y_scores = neigh.predict_proba(x)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Unsupervised
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
labels = kmeans.fit_predict(x)
shil_score = silhouette_score(x, labels)
cali_score = calinski_harabasz_score(x, labels)

st.write("Supervised Algorithm: K Nearest Neighbors")
st.write("Unupervised Algorithm: K Means Clustering")

results = {
    "Type": ["Supervised","Supervised","Supervised","Unsupervised","Unsupervised", ],
    "Metric":["Accuracy", "Precision", "Recall", "Silhouette Score", "Calinski Harabasz Score"],
    "Result": [accuracy, precision, recall, shil_score, cali_score ]
}
st.table(data=results)
