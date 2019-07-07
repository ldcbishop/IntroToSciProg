from skimage.util import random_noise
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.color import rgb2gray
from PIL import Image
import skimage.io as io

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