from processor import Processor, DisplayingProcessor
from opencv_utils import draw_lines, show_image_and_wait_for_key, draw_segments
import clustering
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

def region_from_segment( image, segment ):
    '''given a segment (rectangle) and an image, returns it's corresponding subimage'''
    x,y,w,h= segment
    return image[y:y+h,x:x+w]

class SegmentSplitter( DisplayingProcessor ):
    PARAMETERS= DisplayingProcessor.PARAMETERS+ {'splitter_max_char_width':15}
    '''splits segments that have more than one character'''
    def _process( self, segments):
        assert hasattr(self, 'image') #must be set from the outside
        image= self.image.astype( numpy.float )
        new_segments= []
        self.changed=[]
        for segment in segments:
            if segment[2] <= self.splitter_max_char_width:
                new_segments.append(segment)
            else:
                region= region_from_segment( image, segment )
                vertical_histogram= numpy.mean( region, axis=1).reshape(-1)
                #thresolded= vertical_histogram!=0
                #n_regions= (vertical_histogram[0]!=0) + (numpy.sum( numpy.diff(thresolded) ) / 2)
                #split_point= numpy.argmin(vertical_histogram)
                split_point= segment[2] / 2 #hack
                new_segments.append( (segment[0], segment[1], split_point, segment[3]) )
                new_segments.append( (segment[0]+split_point, segment[1], segment[2]-split_point, segment[3]) )   
                self.changed.append( segment )  
        from segmentation import segments_to_numpy
        self.changed= segments_to_numpy( self.changed )
        new_segments= segments_to_numpy( new_segments )
        return new_segments

    def display( self, display_before=False):
        '''shows the effect of this filter'''
        copy= self.image.copy()
        draw_segments( copy, self._output, (0,255,0) )
        draw_segments( copy, self.changed, (0,0,255) )
        show_image_and_wait_for_key( copy, "segments filtered by "+self.__class__.__name__)


class LineFinder( DisplayingProcessor ):
    @staticmethod
    def _guess_lines( ys, tops=True ):
        '''guesses and returns lines vertical coordinates'''
        ys= ys.reshape(-1)
        s=numpy.sort(ys)
        d= numpy.diff(s)
        _,belongings,centroids= clustering.kmeans_single(d.reshape(-1,1), 2)
        line_changes= (belongings==numpy.argmax(centroids)).reshape(-1)
        line_changes= numpy.insert( line_changes, 0, True )
        line_numbers= numpy.cumsum(line_changes.astype(int))-1
        line_coordinates= s[line_changes]
        return line_coordinates
    
    def _process( self, segments ):
        segment_tops=       segments[:,1].reshape( (-1,1) )
        segment_bottoms=    segment_tops+segments[:,3].reshape( (-1,1) )
        tops=               self._guess_lines( segment_tops )
        bottoms=            self._guess_lines( segment_bottoms )
        if len(tops)!=len(bottoms):
            raise Exception("different number of lines:" +str(len(tops))+" "+str(len(bottoms)))
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
