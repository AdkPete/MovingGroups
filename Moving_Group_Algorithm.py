import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.interpolate as interp
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon

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

def get_backround(x,y,bw = 3 , N = 10):

	'''
	returns the background pixel data for the given data set
	bw is the bin width
	N determines how accurate the integration is
	'''
	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	
	gx1 = np.mean(x)
	gy1 = np.mean(y)
	std_x1 = np.std(x)
	std_y1 = np.std(y)
	
	xgrid , ygrid = np.meshgrid(x1,y1)

	xstep = x1[1] = x1[0]
	ystep = y1[1] - y1[0]

	
	pos = np.dstack((xgrid, ygrid))

	pdf = stats.multivariate_normal(mean = [gx1,gy1] , cov = [ [ std_x1 ** 2 , 0 ] , [0 , std_y1 ** 2] ] )

	res = pdf.pdf(pos) * len(x) * bw ** 2
	return res


def find_moving_groups(x,y,bw = 3):

	
	'''
	This is the main algorithm
	x and y are our velocity components.
	This will typically be v_r and v_phi
	The algorithm should work given any two equal length arrays however
	'''


	
	rescale = 5

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)

	x1 = np.append(x1 , x1[-1] + bw)
	y1 = np.append(y1 , y1[-1] + bw)

	xgrid , ygrid = np.meshgrid(x1,y1)

	statres = stats.binned_statistic_2d(x , y , x, bins = [x1,y1],statistic = "count")

	imd = np.transpose(statres[0])
	
	res = get_backround(x,y,bw)
	residual = imd - res
	ii = np.where( (abs(residual) > 1))
	
	
	cutoff = np.std(residual[ii])
	residual /= cutoff

	x1 = np.arange(int( np.min(x) )  ,  int( np.max(x) + 1 )  , bw)
	y1 = np.arange(int( np.min(y) )  ,  int( np.max(y) + 1 )  , bw)
	residual , nx , ny = interpolate_img_data(np.transpose(residual) , x1 , y1 , rescale)

	nxgrid , nygrid = np.meshgrid(nx,ny)
	plt.pcolormesh(xgrid , ygrid , imd)
	plt.show()
	plt.show()

	plt.pcolormesh(xgrid , ygrid , res)
	plt.colorbar()
	plt.show()

	plt.pcolormesh(nxgrid , nygrid , residual)
	plt.colorbar()
	plt.contour(nxgrid,nygrid , residual , levels = [2,3,4])
	
	plt.show()

	plt.pcolormesh(nxgrid , nygrid , residual)
	plt.colorbar()
	cs = plt.contour(nxgrid,nygrid , residual , levels = [2])
	print (cs.collections[0].get_paths()[0])
	good_x = []
	good_y = []
	paths = []
	
	print (residual.shape , nxgrid.shape)
	points = np.array( [ nxgrid.flatten() , nygrid.flatten() ] )
	for p in cs.collections[0].get_paths():
	
		v = p.vertices
		x = v[:,0]
		y = v[:,1]
		area = PolyArea(x,y)
		print (area , np.mean(x) , np.mean(y))
		ii = p.contains_points(np.transpose(points))
		ii = ii.reshape(residual.shape)
		in2 = np.where(ii == True)
		print (len(residual[in2]))
		area = np.sum(residual[in2]) * bw / rescale
		print (area , np.mean(x) , np.mean(y))
		if area > 90:
			paths.append(p)
			good_x.append(np.mean(x))
			good_y.append(np.mean(y))



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
	plt.show()



x,y = get_test_data()
find_moving_groups(x,y)