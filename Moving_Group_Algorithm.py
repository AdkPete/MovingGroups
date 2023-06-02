
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

def get_test_data(seed):

	'''
	generates sample data
	will include backround data plus some artificial moving groups
	designed for algorithm testing
	returns x,y (simulated velocity components)
	also returns true group locations (true_x , true_y)
	'''

	if seed is not None:
		np.random.seed(seed)
	##Background first


	bx = np.random.normal(0 , 50 , 50000)
	by = np.random.normal(0,50,50000)

	##Add in Simulated groups

	true_x = [25 , -5 , -40 , -40]
	true_y = [25 , -25 , 0 , -40]

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


	return x,y,true_x,true_y

def get_gaussian_background(x , y , params):

	'''
	Returns the background pixel data for the given data set
	Assumes a Gaussian distribution
	x and y are our two velocity components
	'''

	bw = params["bin_width"]
	N = params["gauss_integration_N"]

	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw / N)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw / N)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw / N)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw / N)
	
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
	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw)

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

	smooth_sigma = params["background_smoothing_sigma"] / params["bin_width"]
	
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
	
	np.save("gaia.npy" , [gc.v_x.value , gc.v_y.value - vsun])
	##These are cartesian velocity compontents, commonly used in moving group analysis
	##Some coordinate definitions flip the sign of the x velocities. Doesn't change anything
	##except that it will flip the distribution in x (and therefore the group positions)

	return gc.v_x.value , gc.v_y.value - vsun


def find_moving_groups(x,y , params):

	
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
	sigma = params["sigma"]
	

	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw)

	
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

	
	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw)
	
	nxgrid , nygrid = np.meshgrid(x1,y1)



	if params["display_residuals"]:
		##Plots residual functions

		plt.pcolormesh(nxgrid , nygrid , residual)
		plt.colorbar()
		plt.title("Residuals")
		plt.contour(nxgrid,nygrid , residual , levels = [2,3,4,5])
		
		plt.show()


	good_x , good_y , paths = peak_finder(nxgrid,nygrid,residual,sigma,params)

	

	return good_x , good_y , residual , paths , xgrid , ygrid , imd

def peak_finder(nxgrid , nygrid , residual , sigma , params , known_x = [] , known_y = []):

	'''
	This funcion finds the peaks in a given 2d residual array
	uses the areacutoff defined in params
	sigma determines the level of the contours
	nxgrid and nygrid are our bin postions

	known_x and known_y give the positions of groups already detected.
	The idea is that we won't return those groups. This is used in the iterative solver

	Returns
	________
	peak x and y values, along with relevant contours.
	'''


	bw = params["bin_width"]
	areacut = params["area_cutoff"]
	verbose = params["verbose"]

	cs = plt.contour(nxgrid,nygrid , residual , levels = [sigma])
	
	good_x = []
	good_y = []
	paths = []
	
	known_points = np.transpose(np.array([known_x , known_y]))

	points = np.array( [ nxgrid.flatten() , nygrid.flatten() ] )
	for p in cs.collections[0].get_paths():
	
		R = p.contains_points( known_points)
		if np.any(R): ##If the contour contains a known group, skip.
			continue


		v = p.vertices
		x = v[:,0]
		y = v[:,1]
		area = PolyArea(x,y)
		
		ii = p.contains_points(np.transpose(points))
		ii = ii.reshape(residual.shape)
		in2 = np.where(ii == True)
		
		area = np.sum(residual[in2]) * bw
		if area > areacut:
			if verbose:
				print (area , np.mean(x) , np.mean(y))
			paths.append(p)
			good_x.append(np.mean(x))
			good_y.append(np.mean(y))

	

	return good_x , good_y , paths


def plot_groups(imd,paths,xgrid,ygrid,good_x , good_y , true_x = None , true_y = None , show=False):

	'''
	Plotting code to plot detected groups.
	Makes 2 figures, one with contours displayed and one without them

	imd is a 2d array with velocity or residual data
	paths contains the contours to display
	xgrid and ygrid contain the bin edges to use for plotting
	good_x and good_y contain our group coordinates
	true_x and true_y are the true coordinates for groups (from simulated test data) 

	'''

	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()
	plt.scatter(good_x , good_y , marker = "x" , s = 15 , color = "red")
	if true_x is not None:
		plt.scatter(true_x , true_y , marker = "o" , s = 25 , edgecolors = "blue" , facecolors = 'none')
	for p in paths:
		patch = patches.PathPatch(p, facecolor='black', lw=2 , fill = False)
		ax.add_patch(patch)
	
	
	plt.xlim(-75,75)
	plt.ylim(-75,75)
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")

	plt.savefig("Moving_Groups.pdf")
	plt.close()


	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()
	plt.scatter(good_x , good_y , color = "red" , marker = "x" , s = 15)
	if true_x is not None:
		plt.scatter(true_x , true_y , marker = "o" , s = 15 , edgecolors = "blue" , facecolors = 'none')
	plt.xlim(-75,75)
	plt.ylim(-75,75)
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")
	plt.savefig("Moving_Groups_Scatter.pdf")
	plt.close()
	return good_x , good_y

def iterative_solver(x,y,params):

	'''
	Function to minimize effects of parameter selection
	This will prevent adjacent groups from merging together

	x,y are our velocity components, and params is our parameter dictionary
	'''

	areacut = params["area_cutoff"]
	bw = params["bin_width"]
	verbose = params["verbose"]
	sigma = params["sigma"]
	

	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw)

	
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

	sigma_start = params["sigma_start"]
	sigma_step = params["sigma_step"]
	min_sigma = params["min_sigma"]

	sigma = sigma_start


	
	if params["xbounds"] is None:
		x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	else:
		x1 =  np.arange( params["xbounds"][0] ,  params["xbounds"][1]  , bw)


	if params["ybounds"] is None:
		y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	else:
		y1 = np.arange( params["ybounds"][0] ,  params["ybounds"][1]  , bw)
	
	nxgrid , nygrid = np.meshgrid(x1,y1)

	known_x = []
	known_y = []
	known_paths = []

	sigmas = np.arange(sigma_start , min_sigma , -1 * sigma_step)
	for sigma in tqdm(sigmas , desc = "Running iterative solver"):

		good_x , good_y , paths = peak_finder(nxgrid,nygrid,residual,sigma,params,known_x , known_y)
		known_x += good_x
		known_y += good_y
		known_paths += paths

	return known_x , known_y , residual , known_paths , xgrid , ygrid , imd

def combine_nearby_groups(group_x , group_y , min_distance):

	'''

	This function is designed to combine nearby groups. Sometimes this is necessary
	if the algorithm is splitting groups into multiple detections.
	Any groups within min_distance of each other will be comgined to form a single group.

	'''

	group_x = np.array(group_x)
	group_y = np.array(group_y)

	done = False ##will switch to true when all groups are min_distance or greater appart

	for i in range(len(group_x)):
		
		distances = np.sqrt( (group_x[i] - group_x) ** 2 + (group_y[i] - group_y) ** 2)
		
		ii = np.where(distances < min_distance)
		group_x[ii] = np.mean(group_x[ii])
		group_y[ii] = np.mean(group_y[ii])
		
	return group_x , group_y

if __name__ == "__main__":

	##First, set up parameter dictionary

	params = {}

	##Bin width used for binning velocity data
	params["bin_width"] = 3

	##Number of integration steps to use for Gaussian background (uses N^2 positions per bin)
	params["gauss_integration_N"] = 5

	##area_cutoff sets the detection threshold. The area is computed as a surface integral of 
	##the residual / sqrt(background) inside each contour. If this is less then area_cutoff, the 
	##group is rejected. Typical values are something like  5 * bin_width ^ 2
	params["area_cutoff"] = 30

	##Determines cutoff in residual grid. Higher numbers require that pixels have larger overdensities.
	##High values reduce sensitivity, and may reduce false positives. This is the primary parameter
	##That will determine wether or not nearby groups are counted as seperate groups or one group. Values can
	##Range considerably based on the source data / background, but typically something like 3 is good.
	params["sigma"] = 3

	##Params for the iterative solver. It runs through a series of sigma values, from sigma_start to min_sigma
	##Each iteration reduces the sigma value by sigma_step'

	params["sigma_start"] = 25
	params["sigma_step"] = 1.0
	params["min_sigma"] = 2.0

	##if verbose, we print out some extra info, mostly for debugging purposes
	params["verbose"] = False

	##Set size of smoothing kernel for smoothed background. 
	params["background_smoothing_sigma"] = 20

	##If true, display the smoothinf before/after figures. Useful for testing out smoothing lengths
	params["show_smoothing_figures"] = False

	##Set to either 'Gaussian' or 'Smoothed'. Determines which kind of background to use
	params["background_type"] = "Smoothed"

	##if true, display residual figure in find_moving_groups
	##These figures can be useful for parameter selection and debugging
	params["display_residuals"] = True


	##set velocity limits. If None, it will set them automatically based on the data
	##Otherwise, we only consider stars with velocities inside these ranges

	params["xbounds"] = [-250 , 250]
	params["ybounds"] = [-250 , 250]

	##Get data:

	x,y , true_x , true_y = get_test_data(seed=42)

	#x,y = get_DR3_sample(N = 50000 , vsun = 220)
	
	##Find Groups:
	good_x , good_y , residual , paths , xgrid , ygrid , imd = find_moving_groups(x,y,params)

	good_x , good_y , residual , paths , xgrid , ygrid , imd = iterative_solver(x,y,params)

	good_x , good_y = combine_nearby_groups(good_x , good_y , 15)
	##Make plots:
	plot_groups(imd,paths,xgrid,ygrid,good_x,good_y , true_x , true_y)
	#plot_groups(imd,paths,xgrid,ygrid,good_x,good_y)