from skimage.util import random_noise
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.color import rgb2gray
from PIL import Image
import skimage.io as io
from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv

import numpy as np
import matplotlib.pyplot as plt
from steganography import *

class ImageProcessor:

	def __init__(self, image_url):
		try:
			# Pristine image that we can reset to.
			self.original_url = image_url
			self.image = io.imread(image_url)
		except:
			print 'Could not find image in directory.'

		# Altered image that we will do any alterations to.
		self.altered_image = io.imread(image_url)

	def reset(self):
		self.altered_image = np.copy(self.image)

	## Denoising methods
	def noisy(self,noise_typ):

	   	if noise_typ == "gauss":
		  	row,col,ch= self.altered_image.shape
		  	mean = 0
		  	var = 0.1
		  	sigma = var**0.5
		 	gauss = np.random.normal(mean,sigma,(row,col,ch))
		 	gauss = gauss.reshape(row,col,ch)
		 	noisy = self.altered_image + gauss
		  	return noisy
		elif noise_typ == "s&p":
		  	row,col,ch = self.altered_image.shape
		  	s_vs_p = 0.5
		  	amount = 0.004
		  	out = np.copy(self.altered_image)
		  	# Salt mode
		  	num_salt = np.ceil(amount * self.altered_image.size * s_vs_p)
		  	coords = [np.random.randint(0, i - 1, int(num_salt))
				  for i in self.altered_image.shape]
		  	out[coords] = 1

		  	# Pepper mode
		  	num_pepper = np.ceil(amount* self.altered_image.size * (1. - s_vs_p))
		  	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.altered_image.shape]
		  	out[coords] = 0
		  	return out
	  	elif noise_typ == "poisson":
		  	vals = len(np.unique(self.altered_image))
		  	vals = 2 ** np.ceil(np.log2(vals))
		  	noisy = np.random.poisson(self.altered_image * vals) / float(vals)
		  	return noisy
		elif noise_typ =="speckle":
		  	row,col,ch = self.altered_image.shape
		  	gauss = np.random.randn(row,col,ch)
		 	gauss = gauss.reshape(row,col,ch)        
		  	noisy = self.altered_image + self.altered_image * gauss
		  	return noisy

	### Code to make an image noisy
	def makeNoisyImage(self, sig = 0.155, save = False, plot = False):
		sigma = sig
		noisy = random_noise(self.image, var=sigma**2)
		self.altered_image = noisy

		if plot:
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
		                       sharex=True, sharey=True)
			ax[0].imshow(noisy)
			ax[0].axis('off')
			ax[0].set_title('Noisy')
			ax[1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=True))
			ax[1].axis('off')
			ax[1].set_title('Denoised')
			if save:
				plt.savefig('noise_denoise.png', transparent = True)

	def imageDenoise(self, denoise_method = 'TV'):
		if denoise_method == 'TV':
			return denoise_tv_chambolle(self.altered_image)
		elif denoise_method == 'bilateral':
			return denoise_bilateral(self.altered_image, sigma_color = 0.05, sigma_spatial = 15, multichannel = True)
		elif denoise_method == 'wavelet':
			return denoise_wavelet(self.altered_image, multichannel = True)
		else:
			print denoise_method, ' is not a valid denoising method choice.'

	## The Steganography methods
	def hideMyImage(self, image_url = None, save = False, output_url = './merged.png'):
		# Full credit for this code goes to Kelvin S. do Prado. See the official Github at
		# https://github.com/kelvins/steganography
		merged_image = Steganography.merge(Image.open(self.original_url), Image.open(image_url))
		if save:
			merged_image.save(output_url)
		else:
			print 'Assigned a merged image'
			self.merged_image = merged_image
		return merged_image

	def revealMyImage(self, image_url = None, save = False, output_url='./unmerged.png'):
		if image_url == None:
			try:
				unmerged_image = Steganography.unmerge(self.merged_image)
			except:
				print 'The system has no valid merged image stored.'
		
		else:
			try:
				unmerged_image = Steganography.unmerge(Image.open(image_url))
			except:
				print 'There was an issue opening the image url.'
		if save:
			unmerged_image.save(output_url)
		return unmerged_image

	### Edge detection

	def edgeDetection(self, im_url = None):
		if im_url is None:
			im_url = self.original_url
		canny = Canny(im_url,1.4,50,10)
 		im = canny.grad
 		return im
class Canny:
    '''
        Create instances of this class to apply the Canny edge
        detection algorithm to an image.
 
        input: imagename(string),sigma for gaussian blur
        optional args: thresHigh,thresLow
 
        output: numpy ndarray.
 
        P.S: use canny.grad to access the image array        
 
        Note:
        1. Large images take a lot of time to process, Not yet optimised
        2. thresHigh will decide the number of edges to be detected. It
           does not affect the length of the edges being detected
        3. thresLow will decide the lenght of hte edges, will not affect
           the number of edges that will be detected.
 
        usage example:
        >>>canny = Canny('image.jpg',1.4,50,10)
        >>>im = canny.grad
        >>>Image.fromarray(im).show()
    '''
    def __init__(self,imname,sigma,thresHigh = 50,thresLow = 10):
        self.imin = imread(imname,flatten = True)
 
        # Create the gauss kernel for blurring the input image
        # It will be convolved with the image
        gausskernel = self.gaussFilter(sigma,5)
        # fx is the filter for vertical gradient
        # fy is the filter for horizontal gradient
        # Please not the vertical direction is positive X
         
        fx = self.createFilter([1, 1, 1,
                                0, 0, 0,
                               -1,-1,-1])
        fy = self.createFilter([-1,0,1,
                                -1,0,1,
                                -1,0,1])
 
        imout = conv(self.imin,gausskernel)[1:-1,1:-1]
        gradx = conv(imout,fx)[1:-1,1:-1]
        grady = conv(imout,fy)[1:-1,1:-1]
 
        # Net gradient is the square root of sum of square of the horizontal
        # and vertical gradients
 
        grad = hypot(gradx,grady)
        theta = arctan2(grady,gradx)
        theta = 180 + (180/pi)*theta
        # Only significant magnitudes are considered. All others are removed
        x,y = where(grad < 10)
        theta[x,y] = 0
        grad[x,y] = 0
 
        # The angles are quantized. This is the first step in non-maximum
        # supression. Since, any pixel will have only 4 approach directions.
        x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                       +(theta>337.5)) == True)
        x45,y45 = where( ((theta>22.5)*(theta<67.5)
                          +(theta>202.5)*(theta<247.5)) == True)
        x90,y90 = where( ((theta>67.5)*(theta<112.5)
                          +(theta>247.5)*(theta<292.5)) == True)
        x135,y135 = where( ((theta>112.5)*(theta<157.5)
                            +(theta>292.5)*(theta<337.5)) == True)
 
        self.theta = theta
        Image.fromarray(self.theta).convert('L').save('Angle map.jpg')
        self.theta[x0,y0] = 0
        self.theta[x45,y45] = 45
        self.theta[x90,y90] = 90
        self.theta[x135,y135] = 135
        x,y = self.theta.shape        
        temp = Image.new('RGB',(y,x),(255,255,255))
        for i in range(x):
            for j in range(y):
                if self.theta[i,j] == 0:
                    temp.putpixel((j,i),(0,0,255))
                elif self.theta[i,j] == 45:
                    temp.putpixel((j,i),(255,0,0))
                elif self.theta[i,j] == 90:
                    temp.putpixel((j,i),(255,255,0))
                elif self.theta[i,j] == 45:
                    temp.putpixel((j,i),(0,255,0))
        self.grad = grad.copy()
        x,y = self.grad.shape
 
        for i in range(x):
            for j in range(y):
                if self.theta[i,j] == 0:
                    test = self.nms_check(grad,i,j,1,0,-1,0)
                    if not test:
                        self.grad[i,j] = 0
 
                elif self.theta[i,j] == 45:
                    test = self.nms_check(grad,i,j,1,-1,-1,1)
                    if not test:
                        self.grad[i,j] = 0
 
                elif self.theta[i,j] == 90:
                    test = self.nms_check(grad,i,j,0,1,0,-1)
                    if not test:
                        self.grad[i,j] = 0
                elif self.theta[i,j] == 135:
                    test = self.nms_check(grad,i,j,1,1,-1,-1)
                    if not test:
                        self.grad[i,j] = 0
                     
        init_point = self.stop(self.grad, thresHigh)
        # Hysteresis tracking. Since we know that significant edges are
        # continuous contours, we will exploit the same.
        # thresHigh is used to track the starting point of edges and
        # thresLow is used to track the whole edge till end of the edge.
         
        while (init_point != -1):
            #Image.fromarray(self.grad).show()
            #I commented out the print line -LDCB
            #print 'next segment at',init_point
            self.grad[init_point[0],init_point[1]] = -1
            p2 = init_point
            p1 = init_point
            p0 = init_point
            p0 = self.nextNbd(self.grad,p0,p1,p2,thresLow)
             
            while (p0 != -1):
                #print p0
                p2 = p1
                p1 = p0
                self.grad[p0[0],p0[1]] = -1
                p0 = self.nextNbd(self.grad,p0,p1,p2,thresLow)
                 
            init_point = self.stop(self.grad,thresHigh)
 
        # Finally, convert the image into a binary image
        x,y = where(self.grad == -1)
        self.grad[:,:] = 0
        self.grad[x,y] = 255
 
    def createFilter(self,rawfilter):
        '''
            This method is used to create an NxN matrix to be used as a filter,
            given a N*N list
        '''
        order = pow(len(rawfilter),0.5)
        order = int(order)
        filt_array = array(rawfilter)
        outfilter = filt_array.reshape((order,order))
        return outfilter
     
    def gaussFilter(self,sigma,window = 3):
        '''
            This method is used to create a gaussian kernel to be used
            for the blurring purpose. inputs are sigma and the window size
        '''
        kernel = zeros((window,window))
        c0 = window // 2
 
        for x in range(window):
            for y in range(window):
                r = hypot((x-c0),(y-c0))
                val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
                kernel[x,y] = val
        return kernel / kernel.sum()
 
    def nms_check(self,grad,i,j,x1,y1,x2,y2):
        '''
            Method for non maximum supression check. A gradient point is an
            edge only if the gradient magnitude and the slope agree
 
            for example, consider a horizontal edge. if the angle of gradient
            is 0 degress, it is an edge point only if the value of gradient
            at that point is greater than its top and bottom neighbours.
        '''
        try:
            if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
                return 1
            else:
                return 0
        except IndexError:
            return -1
     
    def stop(self,im,thres):
        '''
            This method is used to find the starting point of an edge.
        '''
        X,Y = where(im > thres)
        try:
            y = Y.min()
        except:
            return -1
        X = X.tolist()
        Y = Y.tolist()
        index = Y.index(y)
        x = X[index]
        return [x,y]
   
    def nextNbd(self,im,p0,p1,p2,thres):
        '''
            This method is used to return the next point on the edge.
        '''
        kit = [-1,0,1]
        X,Y = im.shape
        for i in kit:
            for j in kit:
                if (i+j) == 0:
                    continue
                x = p0[0]+i
                y = p0[1]+j
                 
                if (x<0) or (y<0) or (x>=X) or (y>=Y):
                    continue
                if ([x,y] == p1) or ([x,y] == p2):
                    continue
                if (im[x,y] > thres): #and (im[i,j] < 256):
                    return [x,y]
        return -1
# End of module Canny



### Spectral Methods
def nm_to_rgb(nm):
	"""
	Accepts a wavelength value in either string format (such as those provided by 'input()') or direct numerical format (e.g. 584) and converts it to the related RGB values.  Based on the code given at:
		http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm

	---------------------------------------------------------------------------------------
	Arument Definitions:
	> nm:   (#)     A numerical value for the wavelength to convert
			("str") A string value for the wavelength to convert

	---------------------------------------------------------------------------------------
	Output:
	< rgb:  (r, g, b)   A byte tuple containing the red, green, and blue values for 'nm_str'
	"""

	## Adjustment function ##
	def Adjust(color, intensity):
		"""
		This does some fancy thing that they talk about on the website.  I'm not really sure what it is or does, but here it is.  At the very least we want to avoid 0^x = 1 cases, so there's that.
		"""
		# Constants #
		GAMMA = 0.80
		INTENSITY_MAX = 255

		# Don't want 0^x = 1 for x <> 0 #
		if(color == 0.0):
			return 0
		else:
			# This is adjusting/normalizing the color/intensity into a linear value.
			return round(INTENSITY_MAX * pow(color * intensity, GAMMA))

	## Initialization ##
	# Prime the tuple with the R G B values set to 0, we'll combine them later #
	r = 0
	g = 0
	b = 0

	# Also introduce the intensity factor for dimming towards the ends of the spectrum. #
	intensity = 0

	## Determine RGB Values ##
	# Determine if the input value is a string or a numerical value #
	if(type(nm) == str):
		# Parse the string and overwrite the value #
		nm = float(nm)

	# These values won't change, so we can hard-code them #
	if( (380 <= nm) and (nm < 440) ):
		# The color fades from purple to blue #
		r = -(nm - 440)/(440-380)
		b = 1
	elif( (440 <= nm) and (nm < 490) ):
		# The color fades from blue to cyan #
		g = (nm - 440)/(490-440)
		b = 1
	elif( (490 <= nm) and (nm < 510) ):
		# The color fades from cyan to green #
		g = 1
		b = -(nm - 510)/(510-490)
	elif( (510 <= nm) and (nm < 580) ):
		# The color fades from green to yellow #
		r = (nm - 510)/(580-510)
		g = 1
	elif( (580 <= nm) and (nm < 645) ):
		# The color fades from yellow to red #
		r = 1
		g = -(nm - 645)/(645-580)
	elif( (645 <= nm) and (nm < 780) ):
		# The color stays red for a while #
		r = 1

	## Intensity Adjustment ##
	# Let the intensity fall off near the vision limits #
	if( (380 <= nm) and (nm < 420) ):
		intensity = 0.3 + 0.7*(nm-380)/(420-380)
	elif( (420 <= nm) and (nm < 700) ):
		intensity = 1.0
	elif( (700 <= nm) and (nm < 780) ):
		intensity = 0.3 - 0.7*(nm-780)/(780-700)

	# Further adjust the intensity #
	rgb = ( Adjust(r, intensity), Adjust(g, intensity), Adjust(b, intensity) )

	## Output ##
	return rgb
import numpy as np
import os