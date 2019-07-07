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

def noisy(noise_typ,image):
   	if noise_typ == "gauss":
	  	row,col,ch= image.shape
	  	mean = 0
	  	var = 0.1
	  	sigma = var**0.5
	 	gauss = np.random.normal(mean,sigma,(row,col,ch))
	 	gauss = gauss.reshape(row,col,ch)
	 	noisy = image + gauss
	  	return noisy
	elif noise_typ == "s&p":
	  	row,col,ch = image.shape
	  	s_vs_p = 0.5
	  	amount = 0.004
	  	out = np.copy(image)
	  	# Salt mode
	  	num_salt = np.ceil(amount * image.size * s_vs_p)
	  	coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape]
	  	out[coords] = 1

	  	# Pepper mode
	  	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	  	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
	  	out[coords] = 0
	  	return out
  	elif noise_typ == "poisson":
	  	vals = len(np.unique(image))
	  	vals = 2 ** np.ceil(np.log2(vals))
	  	noisy = np.random.poisson(image * vals) / float(vals)
	  	return noisy
	elif noise_typ =="speckle":
	  	row,col,ch = image.shape
	  	gauss = np.random.randn(row,col,ch)
	 	gauss = gauss.reshape(row,col,ch)        
	  	noisy = image + image * gauss
	  	return noisy