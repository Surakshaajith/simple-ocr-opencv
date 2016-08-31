'''
TODO:
* scan for letters only from left to right, not top to bottom
* add ability to filter out letters with a triangle above the letter
'''

from os.path import isfile, getsize
from os import remove
from files import ImageFile
from grounding import UserGrounder
import segmentation_filters
from segmentation import ContourSegmenter, draw_segments

# input image file and output box file
imgfile= 'captcha'
boxfile= 'data/'+imgfile+'.box'
new_image= ImageFile(imgfile)

# delete the box file if it's empty
if (isfile(boxfile)):
	if (getsize(boxfile) == 0):
		remove(boxfile)

# define what to focus on and ignore in the image
stack= [
	segmentation_filters.LargeFilter(),
	segmentation_filters.SmallFilter(),
	segmentation_filters.LargeAreaFilter(),
	segmentation_filters.ContainedFilter()
	]

# process image, defining useful-looking segments
segmenter=  ContourSegmenter( blur_y=5, blur_x=5, block_size=11, c=10, filters=stack)
segments= segmenter.process(new_image.image)

# uncomment to watch the segmenter in action
#segmenter.display()

grounder= UserGrounder()
grounder.ground(new_image, segments);
new_image.ground.write()
