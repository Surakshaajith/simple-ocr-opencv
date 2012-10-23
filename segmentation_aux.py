from processor import Processor, DisplayingProcessor
from opencv_utils import draw_lines, show_image_and_wait_for_key
import numpy
import cv2

class SegmentOrderer( Processor ):
    PARAMETERS= Processor.PARAMETERS + {"max_line_height":20, "max_line_width":10000}
    def _process( self, segments ):
        '''sort segments in read order - left to right, up to down'''
        #sort_f= lambda r: max_line_width*(r[1]/max_line_height)+r[0]
        #segments= sorted(segments, key=sort_f)
        #segments= segments_to_numpy( segments )
        #return segments
        mlh, mlw= self.max_line_height, self.max_line_width
        s= segments.astype( numpy.uint32 ) #prevent overflows
        order= mlw*(s[:,1]/mlh)+s[:,0]
        sort_order= numpy.argsort( order )
        return segments[ sort_order ]

class SegmentOrdererFromLines( Processor ):
    def _process( self, segments ):
        if not hasattr(self, "lines_middles"):
            raise Exception("This orderer needs the lines middles attribute obtained somewhere else")
        segment_lines= guess_segments_lines( segments, self.lines_middles)
        segment_lines*=1000000
        segment_lines+= segments[:,0]
        return segments[ numpy.argsort(segment_lines) ]

class LineFinder( DisplayingProcessor ):
    @staticmethod
    def _guess_lines( ys, max_lines=50, confidence_minimum=5 ):
        '''guesses and returns text inter-line distance, number of lines, y_position of first line'''
        def dunn_index(centroids, points, belongings):
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
            return min(tmp) / max_intra_distance
        def equalspacing_index( centroids ):
            '''assumes centroids should be equally spaced'''
            spacing= numpy.diff(centroids) 
            return numpy.sum( (spacing-numpy.mean(spacing))**2) #root mean square deviation, more sensitive than std

        ys= ys.astype( numpy.float32 )
        assert len(ys.shape)==1
        ys= ys.reshape(len(ys),1)
        
        means_list=[]
        compactness_indexes, equalspacing_indexes, dunn_indexes= [], [], []
        start_n= 3
        for k in range(start_n,max_lines):
            compactness, belongings, means = cv2.kmeans( data=ys, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=2, flags=cv2.KMEANS_PP_CENTERS)
            means_list.append( means.reshape(-1) )
            if k>=3:
                equalspacing_indexes.append( equalspacing_index( numpy.sort(means.reshape(-1)) ) )
            compactness_indexes.append( compactness )
            dunn_indexes.append( dunn_index( means, ys, belongings.reshape(-1) ) )
        equalspacing_indexes= [max(equalspacing_indexes)]*(3-start_n)+equalspacing_indexes #fill impossible to calculate positions
        
        #perform mangling in order to assemble a meaningful composite metric
        compactness_indexes=    numpy.log( numpy.array(compactness_indexes)+0.01 ) #sum small amount to avoid log(0)
        compactness_indexes=    numpy.diff( compactness_indexes )
        compactness_indexes=    numpy.append(compactness_indexes[0], compactness_indexes)
        dunn_indexes=           -numpy.array( dunn_indexes )
        equalspacing_indexes=   numpy.array( equalspacing_indexes )
        for indexes in (compactness_indexes, equalspacing_indexes, dunn_indexes):
            indexes-=numpy.mean(indexes)    #center on 0
            indexes/=numpy.std(indexes)     #normalize
        aglomerated_metric= 0.1*compactness_indexes + 0.3*equalspacing_indexes + 0.6*dunn_indexes
        
        i= numpy.argmin(aglomerated_metric)
        lines= numpy.sort( means_list[i] )
        
        #calculate confidence
        betterness= numpy.sort(aglomerated_metric, axis=0)
        confidence= ( betterness[1] - betterness[0]) / ( betterness[2] - betterness[1])

        '''
        from pylab import plot, show
        plot(compactness_indexes,   'r')
        plot(equalspacing_indexes,  'g')
        plot(dunn_indexes,          'b')
        plot(aglomerated_metric,    'y')
        show()
        '''
        
        if confidence<confidence_minimum:
            raise Exception("low confidence")
        return lines #still floating points
        
    def _process( self, segments ):
        segment_tops=       segments[:,1]
        segment_bottoms=    segment_tops+segments[:,3]
        tops=               self._guess_lines( segment_tops )
        bottoms=            self._guess_lines( segment_bottoms )
        if len(tops)!=len(bottoms):
            raise Exception("different number of lines")
        middles=                    (tops+bottoms)/2            #middle of the line
        inter=                      (tops[1:]+bottoms[:-1])/2   #between lines
        topbottoms=                 numpy.sort( numpy.append( tops, bottoms ) )
        topmiddlebottoms=           numpy.sort( reduce(numpy.append, ( tops, middles, bottoms )) )
        self.lines_tops=             tops
        self.lines_bottoms=          bottoms
        self.lines_middles=          middles
        self.lines_inter=            inter
        self.lines_topbottoms=       topbottoms
        self.lines_topmiddlebottoms= topmiddlebottoms
        return segments
    
    def display(self, display_before=False):
        copy= self.image.copy()
        draw_lines( copy, self.lines_tops,    (0,0,255) )
        draw_lines( copy, self.lines_bottoms, (0,255,0) )
        show_image_and_wait_for_key( copy, "line starts and ends")



def guess_segments_lines( segments, lines, nearline_tolerance=999.0 ):
    '''given segments, outputs a array of line numbers, or -1 if it 
    doesn't belong to any'''
    ys= segments[:,1]
    closeness= numpy.abs( numpy.subtract.outer(ys,lines) ) #each row a y, each collumn a distance to each line 
    line_of_y= numpy.argmin( closeness, axis=1)
    distance= numpy.min(closeness, axis=1)
    bad= distance > numpy.mean(distance)+nearline_tolerance*numpy.std(distance)
    line_of_y[bad]= -1
    return line_of_y



def contained_segments_matrix( segments ):
    '''givens a n*n matrix m, n=len(segments), in which m[i,j] means
    segments[i] is contained inside segments[j]'''
    x1,y1= segments[:,0], segments[:,1]
    x2,y2= x1+segments[:,2], y1+segments[:,3]
    n=len(segments)
    
    x1so, x2so,y1so, y2so= map(numpy.argsort, (x1,x2,y1,y2))
    x1soi,x2soi, y1soi, y2soi= map(numpy.argsort, (x1so, x2so, y1so, y2so)) #inverse transformations
    o1= numpy.triu(numpy.ones( (n,n) ), k=1).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1<x2
    o2= numpy.tril(numpy.ones( (n,n) ), k=0).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1>x2
    
    a_inside_b_x= o2[x1soi][:,x1soi] * o1[x2soi][:,x2soi] #(x1[a]>x1[b] and x2[a]<x2[b])
    a_inside_b_y= o2[y1soi][:,y1soi] * o1[y2soi][:,y2soi] #(y1[a]>y1[b] and y2[a]<y2[b])
    a_inside_b= a_inside_b_x*a_inside_b_y
    return a_inside_b
