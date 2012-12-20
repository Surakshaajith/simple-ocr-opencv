import cv2 #opencv
import numpy

def kmeans( data, k_min=1, k_max=50):
    '''runs the kmeans algorithm with k from k_min to k_max.
    returns a a tuple. It's first element is the centroids (list of
    numpy arrays) and the second is a belonging vector'''
    data= data.astype( numpy.float32 )
    centroids_list= []
    belongings_list= []
    for k in range(k_min, k_max+1):
        compactness, belongings, centroids = cv2.kmeans( 
            data=data, 
            K=k, 
            bestLabels=None, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), 
            attempts=1, 
            flags=cv2.KMEANS_PP_CENTERS,
            )
        centroids_list.append( centroids )
        belongings_list.append( belongings )
    return centroids_list, belongings_list

def dunn_index(points, centroids, belongings):
    '''calculates Dunn-index, which evaluates clustering quality.
    larger is better''' 
    if points.shape[1]==1: #1D
        def intra_cluster_distance( c, points ):
            return numpy.average( numpy.abs(c-points))
        def inter_cluster_distance( c1, c2):
            return numpy.sqrt(numpy.sum((c1-c2)**2))
    else:
        def intra_cluster_distance( c, points ):
            return numpy.average( numpy.sqrt(numpy.sum((c-points)**2, axis=1)))
        def inter_cluster_distance( c1, c2):
            return numpy.sqrt(numpy.sum(numpy.abs(c1-c2)**2))
    n= len(centroids)
    max_intra_distance= max( [ intra_cluster_distance(centroids[i], points[belongings==i]) for i in range(n)] )
    tmp= []
    for i in range(n):
        for j in [x for x in range(i+1,n) if x!=i]:
            tmp.append( inter_cluster_distance( centroids[i], centroids[j] )  )
    if not len(tmp): #only one cluster
        tmp=[1] #ugly ugly hack!
    max_intra_distance+=1e-10   # divide by zero
    return min(tmp) / max_intra_distance

def even_distribution_index(points, centroids, belongings):
    '''a custom index that checks if the centroids are evenly 
    distributed, i.e.: arranged on a grid.
    larger is better'''
    assert centroids.shape[1]==1 #only implemented for 1D centroids
    if centroids.shape[0]<3:
        raise ValueError("this index doesn't make sense for k<3")
    spacing= numpy.diff(centroids, axis=0) .reshape(-1)
    return -numpy.sum( (spacing-numpy.mean(spacing))**2) #root mean square deviation, more sensitive than std
 
