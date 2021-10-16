## Importamos librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Para generar datos
from sklearn import datasets 
# Importamos librerias de Clustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.cluster import AgglomerativeClustering
# Importamos silhouette_score
from sklearn.metrics import silhouette_score , rand_score

def report_clustering_kmeans(dataset,dataset_name):
    # Listas vacias donde guardar las metricas
    dist_cent = []
    sil_list = []
    rand_list = []
    
    # Separamos en X e Y
    x = dataset[0]
    y = dataset[1]
    
    # Plot para tener de referencia
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title(dataset_name)
    plt.show()
    
    # Plot donde iremos mostrando los resultados
    fig,axs = plt.subplots(2,4,figsize=(15,7))
    axs = axs.ravel()
    for i,k in enumerate(range(2, 10)):
        # Creamos el objeto de clustering
        cluster = KMeans(n_clusters=k).fit(x)
        centers_i = cluster.cluster_centers_ # Centroide de cada cluster
        labels_i = cluster.labels_ # Labels de cada muestra
        # Silhouttte Score
        sil_score_i = silhouette_score(x,labels_i)
        sil_list.append(sil_score_i)        
        # Rand_Index
        rand_index_i = rand_score(y,labels_i)
        rand_list.append(rand_index_i)
        dist_cent.append(cluster.inertia_)
        # Plots de los clusters
        axs[i].scatter(x[:,0],x[:,1],c=labels_i)
        axs[i].scatter(centers_i[:,0], centers_i[:,1], marker="x", color='r',s=150)
        axs[i].set_title('Clusters: ' + str(k))
        
    plt.show()
    # Plot de metricas
    fig, axs = plt.subplots(1,3,figsize=(15,5))    
    axs[0].plot(range(2, 10), dist_cent, marker='s');
    axs[0].set_xlabel('N° K')
    axs[0].set_ylabel('Sum of squared distances')
    # Silhoute plot
    axs[1].plot(range(2, 10), sil_list, marker='s');
    axs[1].set_xlabel('N° K')
    axs[1].set_ylabel('Silhouette')
    # Rand Index plot
    axs[2].plot(range(2, 10), rand_list, marker='s');
    axs[2].set_xlabel('N° K')
    axs[2].set_ylabel('Rand Index')    
    plt.show()
    # Devolvemos la lista con las metricas
    return (sil_list,rand_list)


def report_clustering_hierarchical(dataset,dataset_name,dist_clusters,affinity_measure,linkage_name):
    """
    affinity_measure: Metrica de la distancia entre puntos

        # "cosine": Similaridad angular

        # "euclidean": L2

        # "cityblock"(Manhattan): L1


    linkage_name : Metodo para medir distancias entre clusters

        # 'single': 
        Utiliza la distancia de los dos puntoss más cercanos entre los clústeres para representar la distancia interclúster.

        # 'complete':
        Utiliza los dos puntos más distantes. Favorece la creacion de clusters esfericos.

        # 'average':
        Utiliza la distancia media. Menos afectado por outliers

        # 'ward': 
        Unimos clusters de manera tal que la suma de los errores cuadrados al centroide sea la menor.

    """
    # Listas vacias donde guardar las metricas/scores
    sil_list = []
    rand_list = []
    # Separamos en X e Y
    x = dataset[0]
    y = dataset[1]
    
    # Plot para tener de referencia
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title(dataset_name)
    plt.show()
    # Plot donde iremos mostrando los resultados
    fig,axs = plt.subplots(2,4,figsize=(15,7))
    axs = axs.ravel()
    for i , dist_i in enumerate(dist_clusters):
        
        # Creamos el objeto para clustering con sus parametros
        cluster = AgglomerativeClustering(n_clusters=None,
                                          distance_threshold =dist_i,
                                          affinity= affinity_measure,
                                          linkage=linkage_name)

        cluster.fit_predict(x)
        # Obtenemos los labels
        labels_i = cluster.labels_
        n_clusters = len(np.unique(labels_i))
        
        # Para evitar errores si la cantidad de clusters es 1 ...
        if not 1 < n_clusters:
            sil_list.append(0)
            rand_list.append(0) 
            continue
            
        # Silhouette
        sil_score = silhouette_score(x,labels_i)
        sil_list.append(sil_score)
        
        # Rand_Index
        rand_index_i = rand_score(y,labels_i)
        rand_list.append(rand_index_i)   
        
        # Plots de los clusters
        axs[i].scatter(x[:,0],x[:,1],c=labels_i)
        axs[i].set_title('Distance: ' + str(dist_i) + ' - N° Clusters: ' + str(n_clusters))

    plt.show()
    
    # Plot de metricas
    fig, axs = plt.subplots(1,3,figsize=(15,5))    
    axs[0].plot(dist_clusters, sil_list, marker='s');
    axs[0].set_xlabel('Threshold distances')
    axs[0].set_ylabel('Silhouette')
    # Ploteamos el dendomgram del mejor silhouette_score
    best_sil = np.argmax(sil_list)
    best_dist = dist_clusters[best_sil]
    Z = linkage(x, linkage_name)
    dendrogram(Z,color_threshold=best_dist,ax=axs[1])
    axs[1].axhline(c='k',linestyle='--', y=best_dist)
    axs[1].set_title('Dendogram')
    axs[2].plot(dist_clusters, rand_list, marker='s');
    axs[2].set_xlabel('Threshold distances')
    axs[2].set_ylabel('Rand Index')     
    plt.show()
    return (sil_list,rand_list)
