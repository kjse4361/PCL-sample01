#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

const double PLANE_WIDTH = 2.0;
const double POINT_STEP = 0.05;

const double CLUSTERING_TOLERANCE = 0.1;
const int CLUSTERING_MIN_POINTS = 1;
const int CLUSTERING_MAX_POINTS = 1000000;

pcl::PointCloud<pcl::PointXYZI>::Ptr mk_vert_plane_cloud(double basex, double basey, double basez) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (double z = -PLANE_WIDTH / 2.0; z <= PLANE_WIDTH / 2.0; z += POINT_STEP) {
        for (double x = -PLANE_WIDTH / 2.0; x <= PLANE_WIDTH / 2.0; x += POINT_STEP) {
            pcl::PointXYZI p(basex + x, basey, basez + z);
            cloud->push_back(p);
        }
    }
    return cloud;
}

int main(int argc, char **argv) {
    const std::string planesfile("sample01_planes.pcd");
    const std::string outfile("sample01.pcd");
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr planecloud;

    planecloud = mk_vert_plane_cloud(5.0, 0.0, 0.0);
    *cloud += *planecloud;

    for (double rot = 10.0; rot < 360.0; rot += 10.0) {
        std::cerr << "rot=" << rot << std::endl;
        Eigen::Quaterniond q = Eigen::AngleAxisd(rot / 180.0 * M_PI, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
        Eigen::Translation3d t(0.0, 0.0, 0.0);
        Eigen::Affine3d affine = t * q;
        pcl::PointCloud<pcl::PointXYZI> rotcloud;
        pcl::transformPointCloud(*planecloud, rotcloud, affine);
        *cloud += rotcloud;
    }
    Eigen::Quaterniond q = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    Eigen::Translation3d t(10.0, 10.0, 1.5);
    Eigen::Affine3d affine = t * q;
    pcl::PointCloud<pcl::PointXYZI>::Ptr wkcloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloud, *wkcloud, affine);
    cloud = wkcloud;

    // ================================================================================

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clouds;
    {
        pcl::search::KdTree<pcl::PointXYZI>::Ptr sch_meth(new pcl::search::KdTree<pcl::PointXYZI>);
        sch_meth->setInputCloud(cloud);

        std::vector<pcl::PointIndices> indices_list;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ece;
        ece.setSearchMethod(sch_meth);
        ece.setClusterTolerance(CLUSTERING_TOLERANCE);
        ece.setMaxClusterSize(CLUSTERING_MAX_POINTS);
        ece.setMinClusterSize(CLUSTERING_MIN_POINTS);
        ece.setInputCloud(cloud);
        ece.extract(indices_list);

        std::cerr << "found clouds=" << indices_list.size() << std::endl;

        for (pcl::PointIndices indices : indices_list) {
            pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
            pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::ExtractIndices<pcl::PointXYZI> extract;
            extract.setInputCloud(cloud);
            extract.setIndices(indices_ptr);
            extract.setNegative(false);
            extract.filter(*cluster_cloud);
            clouds.push_back(cluster_cloud);
        }
    }

    // ================================================================================

    {
        for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : clouds) {
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.1);
            seg.setInputCloud(cluster);
            pcl::PointIndices::Ptr plane_indices(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
            seg.segment(*plane_indices, *plane_coefficients);
            if (plane_indices->indices.size() > 0) {
                pcl::PointCloud<pcl::PointXYZI> plane_cloud;
                pcl::ExtractIndices<pcl::PointXYZI> extract;
                extract.setInputCloud(cluster);
                extract.setIndices(plane_indices);
                extract.setNegative(false);
                extract.filter(plane_cloud);
                // std::cerr << "out size = " << plane_cloud.points.size() << std::endl;
                plane_cloud.height = 1;
                plane_cloud.width = plane_cloud.points.size();

                for (pcl::PointXYZI &p : plane_cloud.points) {
                    pcl::PointXYZRGBA cp(p.x, p.y, p.z, 0, 0, 255, 255);
                    output_cloud->push_back(cp);
                }

                Eigen::Vector3d u(plane_coefficients->values[0], plane_coefficients->values[1],
                                  plane_coefficients->values[2]);
                std::cerr << "norma: " << u.x() << " " << u.y() << " " << u.z() << " " << atan2(u.y(), u.x()) << std::endl;
                double uscaler = u.norm();
                Eigen::Vector4d vec4;
                pcl::compute3DCentroid(plane_cloud, vec4);
                Eigen::Vector3d centroid(vec4.x(), vec4.y(), vec4.z());
                for (double l = 0.0; l < 0.5; l += uscaler / 50.0) {
                    Eigen::Vector3d v = centroid + l * u;
                    pcl::PointXYZRGBA cp(v.x(), v.y(), v.z(), 255, 0, 255, 255);
                    output_cloud->push_back(cp);
                }
            }
        }
        output_cloud->height = 1;
        output_cloud->width = output_cloud->points.size();
    }

    // ================================================================================

    if (pcl::io::savePCDFileASCII(planesfile, *cloud) != 0) {
        std::cerr << "file save error: " << planesfile << std::endl;
        return -1;
    }
    if (pcl::io::savePCDFileASCII(outfile, *output_cloud) != 0) {
        std::cerr << "file save error: " << outfile << std::endl;
        return -1;
    }

    return 0;
}
