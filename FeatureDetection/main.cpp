/****************************************************************************
Author:    Bailin Li
Brief:     Load a point cloud from a pcd file or xyz file
		   DownSampling, Smoothing, Denoising the point cloud
		   And do feature extraction after
*****************************************************************************/

#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
//#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/io/vtk_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>

#include <iostream>
#include <windows.h>

using namespace std;

void help(char* argv)
{
	cout << "Usage:" << argv << " [XYZ file name to load]" << " [mode 0 for read pcd file, 1 for read xyz file]" << "[-mode]" << endl;
	cout << "mode could be p(plane), s(sphere), c(circle), l(cylinder)" << endl;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pRawCloud);

boost::shared_ptr<pcl::visualization::PCLVisualizer> VisWithColor (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr feature, int r, int g, int b);

pcl::PointCloud<pcl::Normal>::Ptr estimateNormal(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

int
main (int argc, char** argv)
{
	if(argc < 4)
	{
		help(argv[0]);
		return -1;
	}

	if(pcl::console::find_argument (argc, argv, "-h") >= 0)
	{
		help(argv[0]);
		return -1;
	}	


	// Load input file into a PointCloud<T> with an appropriate type
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr rawcloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  
	if(atoi(argv[2]) == 1)//Create PointCloud based on the data in xyz file
	{
		ifstream ifile(argv[1]);
		if(!ifile.is_open())
		{
			cout << "ERROR:fail to find xyz file!!" << endl;
			return -1;
		}

		double x, y, z;
		while(ifile)
		{

			ifile >> x >> y >> z;
			rawcloud->points.push_back(pcl::PointXYZ(x,y,z));
		}
	}
	else if(atoi(argv[2]) == 0)//load pcd file
	{
	if(pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *rawcloud) == -1)
	{
		cout << "ERROR:fail to find point cloud file!!" << endl;
		return -1;
	}
	}
	else
	{
		help(argv[0]);
		return -1;
	}
  
	//for computing running time
	double start = GetTickCount();
	double end(0);

	//preprocess the raw point cloud
	cloud = preprocessPointCloud(rawcloud);

	normals = estimateNormal(cloud);

	//feature extraction
	std::vector<int> inliers;

	cout << "Feature Extracting..." <<endl;

	// created RandomSampleConsensus object and compute the appropriated model
	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
	model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
	model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
	pcl::SampleConsensusModelCircle2D<pcl::PointXYZ>::Ptr
	model_c (new pcl::SampleConsensusModelCircle2D<pcl::PointXYZ> (cloud));
	pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal>::Ptr
	model_l (new pcl::SampleConsensusModelCylinder<pcl::PointXYZ, pcl::Normal> (cloud));
	model_l->setInputNormals(normals);

	int vis_feature_mode = 0; 

	if(pcl::console::find_argument (argc, argv, "-p") >= 0)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
		ransac.setDistanceThreshold (.5);
		ransac.computeModel();
		ransac.getInliers(inliers);
		vis_feature_mode = 1;
	}
	else if (pcl::console::find_argument (argc, argv, "-s") >= 0 )
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
		ransac.setDistanceThreshold (.5);
		ransac.computeModel();
		ransac.getInliers(inliers);
		Eigen::VectorXf model_coeff;
		ransac.getModelCoefficients(model_coeff);
		vis_feature_mode = 2;
	}
	else if (pcl::console::find_argument (argc, argv, "-c") >= 0 )
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_c);
		ransac.setDistanceThreshold (3.5f);
		ransac.computeModel();
		ransac.getInliers(inliers);
		vis_feature_mode = 3;
	}
	else if (pcl::console::find_argument (argc, argv, "-l") >= 0 )
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
		ransac.setDistanceThreshold (1.5f);
		ransac.computeModel();
		ransac.getInliers(inliers);
		vis_feature_mode = 4;
	}
	else
	{
		cout << "Need to specify Ransac mode!!" << endl;
		return -1;

	}


	cout << "RANSAC finished..." <<endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr feature (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud, inliers, *feature);

	//visualization
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

	switch(vis_feature_mode)
	{
	case 1:
		viewer = VisWithColor(cloud,feature,255,0,0);
		break;
	case 2:
		viewer = VisWithColor(cloud,feature, 0,255,0);
		break;
	case 3:
		viewer = VisWithColor(cloud,feature,0,0,255);
		break;
	case 4:
		viewer = VisWithColor(cloud,feature,0,255,255);
		break;
	default:
		cout << "Can't find correct visualization mode. Program will stop.Please run it again and give it the right parameters!" << endl;
		return -1;
		break;
	}


	cout << "Feature Extractin Completed!" <<endl;

	end = GetTickCount();
	cout << "the algorithm takes " << (end-start)/1000 << endl;
  
	while(!viewer->wasStopped())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}


	return 0;
}


//function: preprocessPointCloud
//brief: Downsampling, denoising, smoothing the raw point cloud
//output: the result PointCloud<pcl::PointXYZ> ready to do feature extraction
pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pRawCloud)
{

	//downsampling

	cout << "DownSampling..." <<endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr ds_cloud (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::VoxelGrid<pcl::PointXYZ> ds;
	ds.setInputCloud (pRawCloud);
	ds.setLeafSize (.5f, .5f, .5f);

	ds.filter (*ds_cloud);

	cout << "DownSampling finished!" <<endl;

	//Denoise process
	cout << "Donoising the point cloud..." <<endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr denoised_cloud (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> denoise;
	denoise.setInputCloud (ds_cloud);
	denoise.setMeanK (100);
	denoise.setStddevMulThresh (1);
	denoise.filter (*denoised_cloud);

	cout << "Donoising process finished!" <<endl;


	//smooth pointcloud and get cloud with normal info
	// Create a KD-Tree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  
	//output cloud of mls method
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed (new pcl::PointCloud<pcl::PointXYZ>);
  
	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
 
	mls.setComputeNormals (false);

	mls.setInputCloud (denoised_cloud);
	mls.setPolynomialFit (true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (2.5);

	// Smooth
  
	cout << "Moving Least Squares Smoothing..." <<endl;
  
	mls.process (*cloud_smoothed);
  
	cout << "Smoothing finished!" << endl;

	//end of mls method

	return cloud_smoothed;
}


//functio: estimateNormal
//brief: estimate normal of the input point cloud
//output: PointCloud<pcl::Normal>

pcl::PointCloud<pcl::Normal>::Ptr estimateNormal(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setKSearch (15);

	cout << "Normal Estimation..." << endl;

	n.compute (*normals);

	cout << "Normal Estimation finished!!" << endl;
	return normals;
}


//function:VisWithColor
//Brief: Set point cloud color and other parameter to initialize viewer
boost::shared_ptr<pcl::visualization::PCLVisualizer> 
VisWithColor (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr feature, int r, int g, int b)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbfeature (new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::copyPointCloud(*cloud, *rgbcloud);
	pcl::copyPointCloud(*feature, *rgbfeature);
  

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	//viewer->setBackgroundColor (0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> feature_color(rgbfeature, r,g,b);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_color(rgbcloud, 255,255,255);
	viewer->addPointCloud<pcl::PointXYZRGB> (rgbfeature, feature_color, "feature");
	viewer->addPointCloud<pcl::PointXYZRGB> (rgbcloud, cloud_color, "cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "feature");
	//viewer->addCoordinateSystem (1.0, "global");
	viewer->initCameraParameters ();
	return (viewer);
}