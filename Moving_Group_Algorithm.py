
'''
This module contains code for our algorithm to detect moving grouos
This is an updated version of the method described in Craig et. al. 2021
Written by Peter Craig
5/4/23
'''

##Useful imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.interpolate as interp
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as patches
from tqdm import tqdm
from astroquery.gaia import Gaia
import astropy.units as u
import astropy.coordinates as coord
import os
def PolyArea(x,y):

	##Simple function to compute the area of a polygon given the
	##x and y positions of the vertices.
	##x and y are both 1d arrays containing the x,y coordinates for
	##the vertices, respectively.
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_test_data():

	'''
	generates sample data
	will include backround data plus some artificial moving groups
	designed for algorithm testing
	'''


	##Background first

	bx = np.random.normal(0 , 50 , 50000)
	by = np.random.normal(0,50,50000)

	##Add in Simulated groups
	
	gx = np.random.normal(25 , 5 , 350)
	gy = np.random.normal(25 , 5 , 350)

	x = np.append(bx , gx)
	y = np.append(by , gy)

	gx = np.random.normal(-5 , 5.25 , 300)
	gy = np.random.normal(-25 , 2.7 , 300)

	x = np.append(x , gx)
	y = np.append(y , gy)


	gx = np.random.normal(-40 , 2 ,100)
	gy = np.random.normal(0 , 2 , 100)

	x = np.append(x , gx)
	y = np.append(y , gy)

	gx = np.random.normal(-40 , 1 ,250)
	gy = np.random.normal(-40 , 6 , 250)

	x = np.append(x , gx)
	y = np.append(y , gy)


	return x,y

def get_gaussian_background(x , y , params):

	'''
	Returns the background pixel data for the given data set
	Assumes a Gaussian distribution
	x and y are our two velocity components
	'''

	bw = params["bin_width"]
	N = params["gauss_integration_N"]

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw / N)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw / N)
	
	gx1 = np.mean(x)
	gy1 = np.mean(y)
	std_x1 = np.std(x)
	std_y1 = np.std(y)
	
	xgrid , ygrid = np.meshgrid(x1,y1)
	
	pos = np.dstack((xgrid, ygrid))

	pdf = stats.multivariate_normal(mean = [gx1,gy1] , cov = [ [ std_x1 ** 2 , 0 ] , [0 , std_y1 ** 2] ] )

	##Res evaluates the pdf at our increased resolution
	res = pdf.pdf(pos)
	##Now we integrate down to our original bin sizes

	##Set bin edges
	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)

	x1 = np.append(x1 , x1[-1] + bw)
	y1 = np.append(y1 , y1[-1] + bw)

	background = stats.binned_statistic_2d(xgrid.flatten(),ygrid.flatten(),res.flatten(),bins=[x1,y1],statistic = "sum")
	background = np.transpose(background[0]) * len(x)  * (bw / N) ** 2


	return background

def get_smoothed_background(imd,xgrid,ygrid , params):

	'''
	Function to return a smoothed background distribution
	This is done using a gaussian smoothing kernel
	The size of this kernel is set with params["background_smoothing_sigma"]
	'''

	smooth_sigma = params["background_smoothing_sigma"]
	
	new_imd = gaussian_filter(imd , sigma = smooth_sigma)

	if params["show_smoothing_figures"]:
		plt.subplot(1,2,1)
		plt.title("New")
		plt.pcolormesh(xgrid , ygrid , new_imd)

		plt.subplot(1,2,2)
		plt.title("old")
		plt.pcolormesh(xgrid , ygrid , imd)
		plt.show()

	return new_imd

def get_DR3_sample(N = 50000 , vsun = 220):

	'''
	This function will query Gaia DR3 to get a solar neighborhood sample
	The query can be relatively slow, so we save the results to a file (gaia.npy) and then
	automatically retrieve from there on subsequent runs.

	Note: It may be worth updating this query to implement error cuts to improve data quality
	Some cuts on parralax/error, rv errors and proper motion errors or altered distance cuts may provide
	better results.

	N is the maximum number of search results to return.
	xsun sets the solar y velocity, which gets subtracted off of the y velocities.
	This should not effect the algorithm as it just shifts the data.

	Returns
	___________
	returns the x and y velocity components for a sample of nearby Gaia sources.

	'''

	if os.path.exists("gaia.npy"): ##Read from file if possible
		x , y = np.load("gaia.npy" , allow_pickle = True)
		return x , y


	##Runs our query with cuts on parallax, plus we require RVs to exist

	query = "select top {} * from gaiadr3.gaia_source where ".format(N)
	
	##Cuts go here:
	query += "parallax > 10 and radial_velocity is not NULL"

	##Run query
	job = Gaia.launch_job_async(query)
	r = job.get_results()
	print ("Number of stars returned {}".format(len(r)))

	##Convert to Galactocentric coordinates w/ astropy
	sc = coord.SkyCoord(ra = r["ra"] , dec = r["dec"] , distance = 1 * u.kpc / r["parallax"].value , pm_ra_cosdec = r["pmra"] , pm_dec = r["pmdec"], radial_velocity = r["radial_velocity"])


	gc = sc.transform_to(coord.Galactocentric)
	
	np.save("gaia.npy" , [gc.v_x.value , gc.v_y.value - 220])
	##These are cartesian velocity compontents, commonly used in moving group analysis
	##Some coordinate definitions flip the sign of the x velocities. Doesn't change anything
	##except that it will flip the distribution in x (and therefore the group positions)

	return gc.v_x.value , gc.v_y.value - 220


def find_moving_groups(x,y, sigma , params):

	
	'''
	This is the main algorithm
	x and y are our velocity components.
	This will typically be v_r and v_phi
	The algorithm should work given any two equal length arrays however
	sigma sets the minimum value considered to be above the noise
	params is a dictionary containing all other parameters, see below.
	
	Returns
	____________
	good_x : array with x positions of the detected groups
	good_y : array with y positions of the detected groups
	residual : 2d array with residual data
	paths : array containing contour data
	xgrid : x bin edges
	ygrid : y bin edges
	imd : 2d array containing binned velopcity data

	The last three arguments are primarilly intended for making plots.
	'''

	areacut = params["area_cutoff"]
	bw = params["bin_width"]
	verbose = params["verbose"]
	
	rescale = 5

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)

	##Add an extra bin to the end, we want to define bin edges and this is required to
	##get the right shape of the residual array

	x1 = np.append(x1 , x1[-1] + bw)
	y1 = np.append(y1 , y1[-1] + bw)

	xgrid , ygrid = np.meshgrid(x1,y1)

	statres = stats.binned_statistic_2d(x , y , x, bins = [x1,y1],statistic = "count")

	imd = np.transpose(statres[0])
	
	
	if params["background_type"] == "Smoothed":
		res = get_smoothed_background(imd,xgrid,ygrid,params)

	elif params["background_type"] == "Gaussian":
		res = get_gaussian_background(x,y,params)

	sigma_grid = np.ones(res.shape)

	ii = np.where(res >= 1.0)
	sigma_grid[ii] = np.sqrt(res[ii])
	
	
	residual = imd - res

	residual /= sigma_grid


	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	
	nx,ny = x1,y1
	nxgrid , nygrid = np.meshgrid(nx,ny)

	if params["display_residuals"]:
		##Plots residual functions

		plt.pcolormesh(nxgrid , nygrid , residual)
		plt.colorbar()
		plt.title("Residuals")
		plt.contour(nxgrid,nygrid , residual , levels = [2,3,4,5])
		
		plt.show()


	cs = plt.contour(nxgrid,nygrid , residual , levels = [sigma])
	
	good_x = []
	good_y = []
	paths = []
	
	
	points = np.array( [ nxgrid.flatten() , nygrid.flatten() ] )
	for p in cs.collections[0].get_paths():
	
		v = p.vertices
		x = v[:,0]
		y = v[:,1]
		area = PolyArea(x,y)
		
		ii = p.contains_points(np.transpose(points))
		ii = ii.reshape(residual.shape)
		in2 = np.where(ii == True)
		
		area = np.sum(residual[in2]) * bw / rescale
		
		if area > areacut:
			if verbose:
				print (area , np.mean(x) , np.mean(y))
			paths.append(p)
			good_x.append(np.mean(x))
			good_y.append(np.mean(y))



	return good_x , good_y , residual , paths , xgrid , ygrid , imd

def plot_groups(imd,paths,xgrid,ygrid,show=False):



	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()
	plt.scatter(good_x , good_y , marker = "x" , s = 10 , color = "red")
	for p in paths:
		patch = patches.PathPatch(p, facecolor='black', lw=2 , fill = False)
		ax.add_patch(patch)
	
	plt.xlim(params["plot_x_limits"])
	plt.ylim(params["plot_y_limits"])
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")

	plt.savefig("Moving_Groups.pdf")
	if show:
		plt.show()


	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()
	plt.scatter(good_x , good_y , color = "red" , marker = "x")
	plt.xlim(params["plot_x_limits"])
	plt.ylim(params["plot_y_limits"])
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")
	plt.savefig("Moving_Groups_Scatter.pdf")
	if show:
		plt.show()

	return good_x , good_y

if __name__ == "__main__":

	##First, set up parameter dictionary

	params = {}

	##Bin width used for binning velocity data
	params["bin_width"] = 3

	##Number of integration steps to use for Gaussian background (uses N^2 positions per bin)
	params["gauss_integration_N"] = 30

	##area_cutoff sets the detection threshold. The area is computed as a surface integral of 
	##the residual / sqrt(background) inside each contour. If this is less then area_cutoff, the 
	##group is rejected. Typical values are something like 5 * bin_width ^ 2
	params["area_cutoff"] = 5

	##if verbose, we print out some extra info, mostly for debugging purposes
	params["verbose"] = False

	##Set size of smoothing kernel for smoothed background. 
	params["background_smoothing_sigma"] = 20

	##If true, display the smoothinf before/after figures. Useful for testing out smoothing lengths
	params["show_smoothing_figures"] = False

	##Set to either 'Gaussian' or 'Smoothed'. Determines which kind of background to use
	params["background_type"] = "Gaussian"

	##if true, display residual figure
	params["display_residuals"] = True

	##if true, make figures in find_moving_groups. Otherwise we skip this step.
	params["make_group_figures"] = True


	x,y = get_test_data()

	good_x , good_y , residual , paths , xgrid , ygrid , imd = find_moving_groups(x,y,3,params)

