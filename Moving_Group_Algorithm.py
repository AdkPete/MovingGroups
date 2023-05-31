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

'''
This module contains code for my algorithm to detect moving grouos
This is an updated version of the method described in Craig et. al. 2021
Written by Peter Craig
5/4/23
'''


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def interpolate_img_data(img,xvals,yvals, rescale_factor):

	'''
	Uses bilinear spline interpolators to increase image resolution
	Takes in a 2xN numpy array
	xvals is an array containing the x positions for img
	yvals is an array containing the y positions for img
	rescale_factor sets the dimensions of the new image. New shape = old shape * rescale_factor
	'''

	interp_func = interp.RectBivariateSpline(xvals,yvals,img,s=0.1)

	new_im = np.zeros( (img.shape[0]  * rescale_factor , img.shape[1] * rescale_factor) )
	nx = np.arange(min(xvals) , max(xvals) , (  xvals[1] - xvals[0] ) / rescale_factor)
	ny = np.arange(min(yvals) , max(yvals) , (  yvals[1] - yvals[0] ) / rescale_factor)

	return np.transpose(interp_func(nx,ny)) , nx , ny

def get_test_data():

	'''
	generates sample data
	will include backround data plus some artificial moving groups
	designed for algorithm testing
	'''


	##Background first

	bx = np.random.normal(0 , 50 , 50000)
	by = np.random.normal(0,50,50000)

	##Groups
	
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

def get_background(x,y,bw = 3 , N = 20):

	'''
	returns the background pixel data for the given data set
	bw is the bin width
	N determines how accurate the integration is
	'''
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
	##Now we integrate down to our original image resolution

	##Set bin edges
	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)

	x1 = np.append(x1 , x1[-1] + bw)
	y1 = np.append(y1 , y1[-1] + bw)

	background = stats.binned_statistic_2d(xgrid.flatten(),ygrid.flatten(),res.flatten(),bins=[x1,y1],statistic = "sum")
	background = np.transpose(background[0]) * len(x)  * (bw / N) ** 2


	return background

def get_smoothed_background(imd,xgrid,ygrid):



	
	new_imd = gaussian_filter(imd , sigma = 5)
	plt.subplot(1,2,1)
	plt.title("New")
	plt.pcolormesh(xgrid , ygrid , new_imd)

	plt.subplot(1,2,2)
	plt.title("old")
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.show()
	return new_imd

def find_moving_groups(x,y,bw = 3,showplots = False,verbose=False,sigma = 2 , areacut = None):

	
	'''
	This is the main algorithm
	x and y are our velocity components.
	This will typically be v_r and v_phi
	The algorithm should work given any two equal length arrays however
	'''

	if areacut == None:
		areacut = 2 * sigma

	
	rescale = 5

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)

	x1 = np.append(x1 , x1[-1] + bw)
	y1 = np.append(y1 , y1[-1] + bw)

	xgrid , ygrid = np.meshgrid(x1,y1)

	statres = stats.binned_statistic_2d(x , y , x, bins = [x1,y1],statistic = "count")

	imd = np.transpose(statres[0])
	
	res = get_smoothed_background(imd,xgrid,ygrid)


	sigma_grid = np.ones(res.shape)

	ii = np.where(res >= 1.0)
	sigma_grid[ii] = np.sqrt(res[ii])
	
	
	residual = imd - res

	residual /= sigma_grid

	#ii = np.where( (abs(residual) > 1))
	
	

	#cutoff = np.std(residual[ii])
	#residual /= cutoff

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	#residual , nx , ny = interpolate_img_data(np.transpose(residual) , x1 , y1 , rescale)
	nx,ny = x1,y1
	nxgrid , nygrid = np.meshgrid(nx,ny)
	if showplots:
		



		plt.pcolormesh(nxgrid , nygrid , residual)
		plt.colorbar()
		plt.title("Residuals")
		plt.contour(nxgrid,nygrid , residual , levels = [2,3,4,5])
		
		plt.show()

	plt.pcolormesh(nxgrid , nygrid , residual)
	plt.colorbar()
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


	if not showplots:
		plt.close()
		return good_x,good_y
	plt.scatter(good_x , good_y , marker = "x" , s = 10 , color = "red")
	plt.show()

	fig, ax = plt.subplots()
	plt.pcolormesh(nxgrid , nygrid , residual)
	plt.colorbar()
	plt.scatter(good_x , good_y , marker = "x" , s = 10 , color = "red")
	for p in paths:
		patch = patches.PathPatch(p, facecolor='black', lw=2 , fill = False)
		ax.add_patch(patch)
	plt.show()

	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()
	plt.scatter(good_x , good_y , marker = "x" , s = 10 , color = "red")
	for p in paths:
		patch = patches.PathPatch(p, facecolor='black', lw=2 , fill = False)
		ax.add_patch(patch)
	plt.xlim(-50,50)
	plt.ylim(-50,50)
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")
	plt.savefig("Moving_Groups.png")
	plt.show()

	fig, ax = plt.subplots()
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.colorbar()

	plt.xlim(-50,50)
	plt.ylim(-50,50)
	plt.xlabel("U (km/s)")
	plt.ylabel("V (km/s)")
	plt.savefig("Moving_Groups_ng.png")
	plt.show()

	return good_x , good_y

def get_DR3_sample():

	if os.path.exists("gaia.npy"):
		x , y = np.load("gaia.npy" , allow_pickle = True)
		return x , y
	job = Gaia.launch_job_async("select top 50000 * from gaiadr3.gaia_source where parallax > 10 and radial_velocity is not NULL")
	r = job.get_results()
	print ("Number of stars returned {}".format(len(r)))
	sc = coord.SkyCoord(ra = r["ra"] , dec = r["dec"] , distance = 1 * u.kpc / r["parallax"].value , pm_ra_cosdec = r["pmra"] , pm_dec = r["pmdec"], radial_velocity = r["radial_velocity"])

	gc = sc.transform_to(coord.Galactocentric)
	
	np.save("gaia.npy" , [gc.v_x.value , gc.v_y.value - 220])
	return gc.v_x.value , gc.v_y.value - 220

def filter(x,y):
	ii = np.where( (abs(x) < 100) & (abs(y) < 100))
	x = x[ii]
	y = y[ii]
	return x,y

x,y = get_DR3_sample()
x,y = filter(x,y)

plt.hist2d(x,y , bins = 75)
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.xlabel("U (km/s)")
plt.ylabel("V (km/s)")
plt.savefig("GaiaVelocities.pdf")
plt.close()
#x,y = get_test_data()
find_moving_groups(x,y , showplots = True,verbose=True,sigma = 2.5 )

#find_moving_groups(x,y , showplots = True,verbose=True,sigma = 11 , areacut = 10)
#find_moving_groups(x,y , bw = e , showplots = True,verbose=True,sigma = 7.25 , areacut = 7.5)
'''
ng = []
for i in tqdm(range(50)):
	x,y = get_test_data()
	gx , gy = find_moving_groups(x,y)

	ng.append(len(gx))

if min(ng) < 4 or max(ng) > 4:
	plt.hist(ng)
	plt.show()
print (min(ng) , max(ng))
'''