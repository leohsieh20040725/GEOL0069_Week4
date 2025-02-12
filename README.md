# GEOL0069_Satellite Echo Classification
This repository classifies echoes in leads and sea ice using machine learning. It generates an average echo shape, computes the standard deviation for both categories, and compares results against the ESA official classification using a confusion matrix

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data">Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data</a>
      <ul>
        <li><a href="#Step 0: Read in Functions Needed">Step 0: Read in Functions Needed</a></li>
        <li><a href="#Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)">Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)</a></li>
        <ul>
          <li><a href="#Co-locate the data">Co-locate the data</a></li>
        <li><a href="#Proceeding with Sentinel-3 OLCI Download">Proceeding with Sentinel-3 OLCI Download</a></li>
           <li><a href="#Sentinel-3 SRAL">Sentinel-3 SRAL</a></li>
      </ul>
        </ul>
    </li>
    <li>
     <a href="#Unsupervised Learning">Unsupervised Learning</a>
      <ul>
        <li><a href="#Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]">Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
          <li><a href="#Introduction to K-means Clustering">Introduction to K-means Clustering</a></li>
        <li><a href="#Why K-means for Clustering?">Why K-means for Clustering?</a></li>
        <li><a href="#Key Components of K-means">Key Components of K-means</a></li>
        <li><a href="#The Iterative Process of K-means">The Iterative Process of K-means</a></li>
        <li><a href="#Advantages of K-means">Advantages of K-means</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
          </ul>
        <li><a href="#Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]">Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
        <li><a href="#Introduction to Gaussian Mixture Models">Introduction to Gaussian Mixture Models</a></li>
        <li><a href="#Why Gaussian Mixture Models for Clustering?">Why Gaussian Mixture Models for Clustering?</a></li>
        <li><a href="#Key Components of GMM">Key Components of GMM</a></li>
        <li><a href="#The EM Algorithm in GMM">The EM Algorithm in GMM</a></li>
        <li><a href="#Advantages of GMM">Advantages of GMM</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
           </ul>
        <li><a href="#Image Classification">Image Classification</a></li>
        <ul>
        <li><a href="#K-Means Implementation">K-Means Implementation</a></li>
        <li><a href="#GMM Implementation">GMM Implementation</a></li>
           </ul>
        <li><a href="#Altimetry Classification">Altimetry Classification</a></li>
        <ul>
        <li><a href="#Read in Functions Needed">Read in Functions Needed</a></li>
          </ul>
        <li><a href="#Scatter Plots of Clustered Data">Scatter Plots of Clustered Data</a></li>
        <li><a href="#Waveform Alignment Using Cross-Correlation">Waveform Alignment Using Cross-Correlation</a></li>
        <li><a href="#Compare with ESA data">Compare with ESA data</a></li>   
       
  
  </ol>
</details>





<!-- ABOUT THE PROJECT -->
## About The Project


This project is dedicated to colocating Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data while leveraging unsupervised learning techniques for classifying sea ice and leads. The goal is to develop an automated pipeline that improves Earth Observation (EO) analysis by fusing different satellite datasets and applying machine learning models to classify environmental features.

Why This Project?
* 🌍 Integrates Sentinel-2 and Sentinel-3 data to improve classification accuracy.
* 🛰️ Leverages machine learning (K-Means & GMM) to identify patterns in EO data.
* 📊 Applies clustering techniques for classifying sea ice and leads.
* 🔄 Automates colocation and processing, reducing manual effort.
* ✅ Compares model predictions with ESA’s official classification for validation.

Key Learning Outcomes
* Understanding the colocation process for multi-sensor satellite data.
* Applying unsupervised learning to classify environmental features.
* Using altimetry and optical data fusion for sea ice and lead classification.
* Evaluating classification performance through confusion matrices and accuracy metrics.

Of course, satellite data analysis is an evolving field, and different approaches may work for different datasets. This project serves as a practical guide to combining remote sensing and machine learning. Contributions and improvements are always welcome! 🚀

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project utilizes key Python libraries and geospatial tools to process, analyze, and classify Earth Observation (EO) data. Below are the major dependencies used:

* NumPy – Numerical computations and matrix operations
* Pandas – Data manipulation and tabular processing
* Matplotlib – Visualization of classification results
* Rasterio – Handling Sentinel-2 geospatial raster data
* netCDF4 – Processing Sentinel-3 altimetry data
* Scikit-Learn – Machine learning models (K-Means, GMM)
* Folium – Geospatial data visualization
* Shapely – Geometric operations for colocation analysis
* Requests – API calls for Sentinel-3 metadata retrieval

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data -->
## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data

This section of the process focuses on the colocation of Sentinel-2 and Sentinel-3 data, combining Sentinel-2’s high spatial resolution with Sentinel-3’s extensive coverage and altimetry insights to enhance Earth observation analysis. This fusion enables precise environmental monitoring, particularly for applications like sea ice and lead classification. The following steps will outline how to identify, align, and analyze these datasets effectively.

### Step 0: Read in Functions Needed

This part of the process begins by loading essential functions, following the approach used in Week 3, to retrieve metadata for the two satellites efficiently. Google Drive is mounted in the Google Colab environment, enabling seamless access to stored files. The process fetches, processes, and analyzes Sentinel-2 and Sentinel-3 Earth observation data from the Copernicus Data Space Ecosystem, using key libraries like requests, pandas, shapely, and folium for API requests, geospatial processing, and visualization. Authentication is handled through access tokens, and data is queried based on date range, location, and cloud cover percentage. Sentinel-3 OLCI and SRAL data, along with Sentinel-2 optical data, are retrieved, and products can be downloaded via unique IDs. Geospatial footprints are processed to match images based on geographic overlap, with results visualized using interactive maps. Time-handling functions ensure correct timestamp formatting for accurate data retrieval. This structured pipeline enables seamless integration with scientific research and Earth observation projects. 🚀

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
This part of the process co-locates Sentinel-2 and Sentinel-3 OLCI data by retrieving their metadata separately, following the Week 3 approach. The goal is to identify common locations observed by both satellites, creating sentinel3_olci_data and sentinel2_data as the foundation for further analysis. Authentication is required to obtain and refresh access tokens before defining a date range and file path for metadata retrieval. The process queries Sentinel-3 OLCI and Sentinel-2 optical data using query_sentinel3_olci_arctic_data() and query_sentinel2_arctic_data(), applying a 0–10% cloud cover filter for Sentinel-2 to ensure clearer observations. The metadata is then saved as sentinel3_olci_metadata.csv and sentinel2_metadata.csv for alignment and analysis. To enhance visualization, Sentinel-3 OLCI and Sentinel-2 metadata are displayed in structured table formats within Jupyter Notebook or Google Colab. Using display(s3_olci_metadata) and display(s2_metadata) from IPython, the datasets are rendered as clear, interactive tables, making it easier to inspect, analyze, and verify key details such as product IDs, acquisition times, geospatial footprints, and cloud cover percentages.

![422d13d80a9193db1c3d56e377ac803](https://github.com/user-attachments/assets/cfcaa103-e028-45e3-bcf9-b91b6957116c)

The table displays the metadata retrieved for Sentinel-3 OLCI images within the specified time range. It includes essential attributes such as unique product IDs, names, content types, origin dates, modification dates, and storage paths. This metadata is crucial for identifying and accessing relevant satellite data for further analysis and co-location with Sentinel-2.

![7a75d1a91255c7e1a9145b45bb2fb72](https://github.com/user-attachments/assets/b428d280-5b6a-4ce4-a588-f9d46936ab1d)
This table displays metadata retrieved for Sentinel-2 images using the Copernicus Data Space API. It includes details such as product IDs, content type, content length, acquisition dates, publication and modification timestamps, online availability, and storage paths. This dataset is essential for analyzing and identifying relevant Sentinel-2 imagery based on specific timeframes and geospatial locations.

#### Co-locate the data
This part of the process uses the generated metadata to identify co-location pairs by matching Sentinel-2 and Sentinel-3 OLCI data based on their geo_footprint. The metadata is processed by extracting and converting ContentDate timestamps into a standardized datetime format using eval(), pd.to_datetime(), and make_timezone_naive(), ensuring consistency in time comparisons. The check_collocation() function then identifies overlapping observations within a 10-minute time window, aligning Sentinel-2 and Sentinel-3 OLCI datasets for further geospatial analysis. The resulting results DataFrame contains matched records where both satellites observed the same location within the defined timeframe, enabling effective data fusion and comparison. To visualize the results, the plot_results() function maps the first five co-located observations using folium, plotting the satellite footprints. The display() function from IPython.display renders the interactive map within the notebook, allowing users to inspect overlapping geographic locations.

![image](https://github.com/user-attachments/assets/f136ac9d-181f-42e3-9079-af0e87e2fdbf)
The table displays the first five rows of the collocated dataset, showing matched Sentinel-2 and Sentinel-3 OLCI observations. Each row contains details about the two satellites, including their unique IDs, footprints (geographical coverage), and the time range during which their observations overlap within a 10-minute window. This output helps verify the successful identification of collocated satellite data for further analysis.

![image](https://github.com/user-attachments/assets/e0bb2933-f303-4a27-ace3-e0dab44e97eb)
This interactive map visualization displays the geographical footprints of the first five collocated satellite observations from Sentinel-2 and Sentinel-3 OLCI. The overlapping satellite data areas are highlighted, showing the regions where both satellites have captured observations within the specified time window.


<!-- Proceeding with Sentinel-3 OLCI Download -->
#### Proceeding with Sentinel-3 OLCI Download
Next, the focus shifts to retrieving Sentinel-3 OLCI data, following the same structured approach used for Sentinel-2 to ensure consistency. By applying the same filename conversion logic, the required datasets are systematically accessed and downloaded from the Copernicus Dataspace, ensuring seamless integration into the analysis pipeline. This step facilitates the download of a specific Sentinel-3 OLCI product. The download_dir variable defines the target directory, while product_id and file_name are extracted from the results DataFrame, selecting the first product for download. The download_single_product() function, along with an access_token, ensures secure retrieval of the satellite data, storing it in the designated directory for further analysis. Users can modify product_id, file_name, and download_dir to customize their downloads.

#### Sentinel-3 SRAL
This part of the process extends co-location analysis by integrating Sentinel-3 SRAL altimetry data alongside Sentinel-2 and Sentinel-3 OLCI observations. The approach remains the same, requiring only the retrieval of S3 SRAL metadata to enhance the dataset with valuable altimetry measurements, enabling a more comprehensive surface analysis. The query_sentinel3_sral_arctic_data() function retrieves Sentinel-3 SRAL metadata for a specified date range, using an access token for authentication. The retrieved metadata is stored in sentinel3_sral_data and saved as s3_sral_metadata.csv in the defined directory (path_to_save_data), ensuring easy access for further processing. For co-location, previously saved metadata files (s3_sral_metadata.csv and sentinel2_metadata.csv) are loaded into Pandas DataFrames using pd.read_csv(). The ContentDate field in both datasets is parsed using eval(), extracted into Start and End timestamps, and converted into timezone-naive datetime objects using pd.to_datetime() and make_timezone_naive() for consistent time representation. The check_collocation() function then identifies overlapping Sentinel-2 and Sentinel-3 SRAL observations within a 10-minute time window, storing the results in a results DataFrame. This ensures that only observations meeting the spatial and temporal criteria are considered co-located. To visualize the results, the plot_results() function maps the top five co-located Sentinel-2 and Sentinel-3 SRAL entries using GeoJSON footprints over an interactive world map. The display() function from IPython.display renders the visualization within Jupyter Notebook or Google Colab, allowing users to analyze spatial relationships and assess the accuracy of co-located observations.


![image](https://github.com/user-attachments/assets/908fe20f-02df-403e-9937-32f8b527bc1b)
This interactive map visualizes the collocation of Sentinel-2 and Sentinel-3 SRAL satellite data. The blue outlines represent the geographical footprints of the detected overlaps, illustrating how the two satellite datasets align over the Arctic region. This visualization helps assess spatial intersections and validate the effectiveness of the collocation process.


<!-- Unsupervised Learning -->
## Unsupervised Learning
This section introduces a crucial domain in machine learning and AI: unsupervised learning. Rather than diving deep into theoretical complexities, our focus is on providing a hands-on, practical guide. Our goal is to equip you with the knowledge and tools necessary to effectively apply unsupervised learning techniques to real-world Earth Observation (EO) scenarios.

While unsupervised learning has a vast range of applications, this discussion will primarily focus on classification tasks. These techniques are particularly powerful in detecting patterns and grouping data when predefined labels are unavailable. By leveraging these methods, you’ll develop an intuitive understanding of how to uncover hidden structures and relationships within your datasets, even in the absence of explicit categorizations.

In this notebook, we will tackle two key tasks:

1-Distinguishing sea ice from leads using image classification based on Sentinel-2 optical data.

2-Identifying sea ice and leads through altimetry data classification utilizing Sentinel-3 altimetry data.

By the end of this section, you will have a solid foundation in applying unsupervised learning for remote sensing and EO data analysis.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]
#### Introduction to K-means Clustering
K-means clustering is a widely used unsupervised learning algorithm designed to partition a dataset into k distinct groups (clusters). The parameter k, which represents the number of clusters, is predefined by the user. The algorithm classifies data points based on feature similarity, ensuring that data points within the same cluster are more alike than those in different clusters [MacQueen et al., 1967].

At its core, K-means clustering works by:

1-Defining k centroids, one for each cluster.

2-Assigning each data point to the nearest centroid based on similarity.

3-Updating the centroids by recalculating the cluster means.

4-Iterating this process until the centroids remain stable, minimizing the variation within clusters.

This method is particularly useful for pattern recognition, data segmentation, and exploratory data analysis, making it a fundamental tool in unsupervised learning.
#### Why K-means for Clustering?
K-means clustering is particularly effective in scenarios where:

1-Unknown Data Structure: The algorithm does not require prior knowledge of the data distribution or structure, making it an excellent choice for exploratory data analysis and pattern detection.

2-Efficiency and Scalability: K-means is computationally efficient, easy to implement, and capable of handling large datasets with minimal complexity, making it a preferred choice for various real-world applications.

#### Key Components of K-means
1-Choosing K: The number of clusters (k) must be predefined before running the algorithm, which can significantly impact the clustering results.

2-Centroid Initialization: The initial placement of cluster centroids plays a crucial role in determining the final clustering outcome.

3-Assignment Step: Each data point is assigned to the closest centroid based on the squared Euclidean distance, effectively grouping similar points together.

4-Update Step: The centroids are recalculated as the mean position of all data points assigned to their respective clusters.

#### The Iterative Process of K-means
K-means follows an iterative approach where the assignment and update steps repeat until the centroids stabilize, meaning they no longer change significantly. This process minimizes intra-cluster variation and ensures convergence to an optimal clustering solution, though it may sometimes settle on a local optimum.

#### Advantages of K-means
1-High Efficiency: K-means is computationally fast, making it well-suited for handling large datasets.

2-Ease of Interpretation: The clustering results are straightforward, making it easy to analyze and understand the underlying data patterns.

#### Basic Code Implementation
Below is a fundamental implementation of the K-means clustering algorithm. This example provides a solid starting point for understanding how the algorithm works and can be adapted for various data analysis tasks. By exploring this implementation, you can gain insights into clustering techniques and apply them effectively to real-world datasets.

```python
from google.colab import drive
drive.mount('/content/drive')
```

This code is used in Google Colab to mount Google Drive, allowing users to access files stored in their Drive directly from the Colab notebook. The from google.colab import drive statement imports the drive module from the google.colab package, which is specifically designed for working with Google Colab. The drive.mount('/content/drive') command then mounts the user’s Google Drive to the /content/drive directory, enabling seamless reading and writing of files. Once executed, users will be prompted to authenticate their Google account and grant necessary permissions. This is particularly useful for loading datasets, saving outputs, and managing files within Colab.

```python
pip install rasterio
```
The command pip install rasterio is used to install the Rasterio library, a powerful tool for reading, writing, and processing geospatial raster data. Rasterio provides a Pythonic interface to interact with raster datasets, such as satellite imagery and digital elevation models, using the GDAL (Geospatial Data Abstraction Library) backend. By running this command, users can install Rasterio and its dependencies, enabling them to work with raster data efficiently, perform geospatial analysis, and manipulate multi-band imagery in Python. This command is typically executed in a Jupyter Notebook, Google Colab, or a terminal to ensure that the library is available for use in geospatial projects.

```python
pip install netCDF4
```
The command pip install netCDF4 is used to install the netCDF4 library, which provides tools for reading, writing, and manipulating netCDF (Network Common Data Form) files. NetCDF is a widely used data format for storing large-scale scientific data, particularly in meteorology, oceanography, and climate research. This library enables users to efficiently handle multi-dimensional data arrays, access metadata, and perform various operations on netCDF files in Python. Running this command ensures that the necessary dependencies are installed, allowing seamless interaction with netCDF datasets in Jupyter Notebooks, Google Colab, or Python scripts.


```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
This Python script implements a basic K-means clustering algorithm using the scikit-learn library. It begins by importing necessary libraries, including KMeans for clustering, matplotlib.pyplot for visualization, and numpy for numerical operations. The script generates a sample dataset consisting of 100 random points in a two-dimensional space. A K-means model is then initialized with four clusters (n_clusters=4) and trained on this dataset using kmeans.fit(X). After training, the model predicts the cluster assignments for each data point, storing the results in y_kmeans. To visualize the clustering results, a scatter plot is created where each point is color-coded according to its assigned cluster. Additionally, the computed centroids of the clusters are extracted using kmeans.cluster_centers_ and displayed as black dots to indicate the center of each group. Finally, the plt.show() function renders the plot, providing a clear visualization of how the algorithm has grouped the data into distinct clusters. This implementation serves as a foundational example for understanding how K-means clustering can be applied to pattern recognition and data segmentation tasks.

![image](https://github.com/user-attachments/assets/e336776c-92d6-4d6a-b3fc-be0c1f41960d)

Visualization of K-means clustering results on a randomly generated dataset. The colored points represent individual data samples grouped into four clusters, while the black dots indicate the centroids of each cluster, calculated by the K-means algorithm.

<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]
#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) are a powerful probabilistic approach for modeling normally distributed subpopulations within a larger dataset. This technique assumes that data is generated from a combination of multiple Gaussian distributions, each characterized by its unique mean and variance [Reynolds and others, 2009]. GMMs are widely employed in clustering and density estimation, offering a structured way to represent intricate data distributions by merging simpler components.

#### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models are particularly advantageous in clustering applications where:

1-Soft Clustering Capabilities: Unlike K-means, which assigns each data point to a single cluster, GMM provides a probabilistic classification. This means each data point has a probability of belonging to multiple clusters, enabling a more nuanced and flexible clustering approach while accounting for uncertainty.

2-Adaptive Cluster Shapes: Unlike K-means, which assumes clusters are spherical, GMM accommodates clusters of varying shapes and sizes. By adjusting the covariance structure of each Gaussian component, GMM provides a more adaptable and precise clustering solution for complex datasets.

These features make GMM a robust choice for clustering scenarios where data exhibits overlapping distributions or varying density regions.

#### Key Components of GMM
1-Defining the Number of Components (Gaussians):
Similar to specifying the number of clusters in K-means, GMM requires determining the number of Gaussian components. This defines how many distinct distributions will be used to model the data.

2-Expectation-Maximization (EM) Algorithm:
GMM utilizes the EM algorithm to iteratively refine the model. It alternates between estimating the probability of each data point belonging to a Gaussian and updating the parameters (mean, variance, and weight) to maximize the model’s likelihood.

3-Covariance Structure:
The shape and orientation of clusters are influenced by the covariance type of the Gaussians. This allows for flexibility, supporting spherical, diagonal, tied, or fully adaptable cluster shapes.



#### The EM Algorithm in GMM
The EM algorithm follows an iterative two-step process to optimize clustering:

1-Expectation Step (E-step):
Assigns probabilities to each data point, estimating the likelihood that it belongs to a particular Gaussian component.

2-Maximization Step (M-step):
Updates the Gaussian parameters (mean, variance, and weight) to maximize the overall likelihood of the dataset given the current assignments.

This process repeats until convergence, meaning the parameters stabilize and the model achieves an optimal fit.


#### Advantages of GMM
1-Probabilistic Soft Clustering:
Unlike hard clustering methods like K-means, GMM provides a probability score for each data point’s cluster membership. This helps in capturing uncertainty and overlapping group structures.

2-Flexible Cluster Shapes:
GMM supports non-spherical cluster formations, making it well-suited for datasets where clusters have varying sizes, orientations, or densities.

By leveraging GMM’s adaptability and probabilistic framework, it becomes an excellent choice for clustering complex datasets with overlapping distributions.

#### Basic Code Implementation
Below is a fundamental implementation of the Gaussian Mixture Model (GMM). This example provides a foundational understanding of how GMM works and serves as a practical starting point for incorporating it into data analysis tasks.


```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```
This code demonstrates the implementation of the Gaussian Mixture Model (GMM) for clustering a set of randomly generated data points. It begins by importing necessary libraries: GaussianMixture from sklearn.mixture for clustering, matplotlib.pyplot for visualization, and numpy for numerical operations. The dataset consists of 100 randomly generated points in a two-dimensional space. A GMM model is then initialized with three components (clusters) and fitted to the dataset. The model predicts cluster assignments for each data point. To visualize the results, the data points are plotted using different colors to represent different clusters, and the computed cluster centers (means) are highlighted in black. This visualization helps illustrate how GMM effectively groups similar data points based on probabilistic distributions.

![image](https://github.com/user-attachments/assets/ba6c9c55-ed2b-43ef-82a8-543445e64478)
Visualization of clustering results using the Gaussian Mixture Model (GMM). The data points are grouped into three distinct clusters, each represented by a different color. The black points indicate the computed cluster centers (means), highlighting the probabilistic nature of GMM clustering.

### Image Classification
In this section, we delve into the application of unsupervised learning techniques for image classification. Our primary focus is distinguishing between sea ice and leads using Sentinel-2 imagery. By leveraging clustering algorithms, we can identify and classify patterns in the imagery without the need for labeled data, enhancing our ability to analyze and interpret remote sensing data efficiently.

#### K-Means Implementation
```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```
This Python script performs K-means clustering on a Sentinel-2 satellite image to classify surface features based on reflectance values from the red (B4) band. It begins by importing necessary libraries: rasterio for handling satellite imagery, numpy for numerical operations, sklearn.cluster.KMeans for clustering, and matplotlib.pyplot for visualization. The script specifies the file paths for the Sentinel-2 image stored in a designated directory and loads Band 4 (Red) using rasterio.open(), storing the pixel values in a NumPy array. To ensure meaningful processing, a valid data mask is created to filter out pixels with no-data values (zero pixels).

Once the image data is loaded, the script reshapes the band values into a 1D array while excluding invalid pixels. The K-means clustering algorithm is then applied with two clusters (n_clusters=2), aiming to distinguish different surface types, such as sea ice and open water. After fitting the model, the script assigns each pixel a cluster label, classifying the satellite image into distinct regions based on reflectance characteristics.

To reconstruct and visualize the results, the classified pixel labels are reshaped back into the original image dimensions. Any masked-out pixels are assigned a no-data value (-1) to preserve data integrity. The clustering results are displayed using plt.imshow(), where different clusters are represented in distinct colors, allowing for an intuitive visual analysis. Finally, the script clears memory by deleting unnecessary variables to optimize performance.

This approach enables an unsupervised classification of Sentinel-2 imagery, allowing researchers to analyze environmental features without requiring labeled training data. The generated classification map provides valuable insights into surface variations, making it a useful tool for climate monitoring, environmental research, and land cover classification.

![image](https://github.com/user-attachments/assets/4dbda879-080b-406b-b1f0-24b97a1d7aa8)
This image represents the result of K-means clustering applied to a Sentinel-2 optical band (B4). The clustering algorithm groups pixels into two distinct clusters, as shown by the different colors. The yellow regions likely correspond to sea ice or land, while the darker regions represent open water or other surface types. The color bar on the right indicates cluster labels, with a no-data value (-1) assigned to areas outside the valid data range. This classification helps in distinguishing different surface features in remote sensing imagery.

#### GMM Implementation


```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
This Python script performs Gaussian Mixture Model (GMM) clustering on a Sentinel-2 optical band (B4) image using the scikit-learn library. The code begins by defining the file paths for Sentinel-2 bands (B4, B3, and B2), with a specific base directory where the data is stored. It then reads the B4 band using the Rasterio library and stacks the data into a NumPy array. To ensure that only valid pixels are processed, a mask is created to exclude pixels with zero values.

Next, the script reshapes the valid band data into a format suitable for clustering and applies a Gaussian Mixture Model (GMM) with two components. GMM is a probabilistic clustering approach that models data as a mixture of multiple Gaussian distributions, allowing for soft clustering, meaning each pixel is assigned a probability of belonging to each cluster.

Once the model is trained, it assigns cluster labels to the valid pixels. The results are stored in a new array, with invalid areas being assigned a no-data value (-1). Finally, the script visualizes the clustering results using Matplotlib, where different clusters are color-coded, and a color bar is added to indicate cluster labels. This method is useful for analyzing sea ice, land cover, or other environmental patterns in Sentinel-2 imagery.

![2e7a5d5ebb2325493b5e9bdfa718c6d](https://github.com/user-attachments/assets/d6655404-0987-4593-8fc0-2bcc32270091)

Visualization of Gaussian Mixture Model (GMM) clustering applied to Sentinel-2 Band 4 imagery. The image showcases different clusters identified in the dataset, with the color scale representing distinct cluster labels. This method helps differentiate between various land cover types, such as sea ice, open water, and land surfaces, based on spectral reflectance patterns.


### Altimetry Classification
In this section, we explore the application of unsupervised learning techniques for classifying altimetry data, specifically focusing on distinguishing sea ice and leads within the Sentinel-3 altimetry dataset. This approach enables us to analyze surface features based on satellite-derived elevation measurements, improving our understanding of ice dynamics and oceanographic processes.

#### Read in Functions Needed
Before proceeding with modeling, it is essential to preprocess the data to ensure compatibility with analytical methods. This involves converting raw altimetry measurements into meaningful variables, such as peakiness and stack standard deviation (SSD), which provide insights into surface roughness and classification potential. Effective preprocessing enhances model accuracy and ensures a robust classification process.




```python
#
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

#=========================================================================================================
#===================================  SUBFUNCTIONS  ======================================================
#=========================================================================================================

#*args and **kwargs allow you to pass an unspecified number of arguments to a function,
#so when writing the function definition, you do not need to know how many arguments will be passed to your function
#**kwargs allows you to pass keyworded variable length of arguments to a function.
#You should use **kwargs if you want to handle named arguments in a function.
#double star allows us to pass through keyword arguments (and any number of them).
def peakiness(waves, **kwargs):

    "finds peakiness of waveforms."

    #print("Beginning peakiness")
    # Kwargs are:
    #          wf_plots. specify a number n: wf_plots=n, to show the first n waveform plots. \

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    print("Running peakiness function...")

    size=np.shape(waves)[0] #.shape property is a tuple of length .ndim containing the length of each dimensions
                            #Tuple of array dimensions.

    waves1=np.copy(waves)

    if waves1.ndim == 1: #number of array dimensions
        print('only one waveform in file')
        waves2=waves1.reshape(1,np.size(waves1)) #numpy.reshape(a, newshape, order='C'), a=array to be reshaped
        waves1=waves2

    # *args is used to send a non-keyworded variable length argument list to the function
    def by_row(waves, *args):
        "calculate peakiness for each waveform"
        maximum=np.nanmax(waves)
        if maximum > 0:

            maximum_bin=np.where(waves==maximum)
            #print(maximum_bin)
            maximum_bin=maximum_bin[0][0]
            waves_128=waves[maximum_bin-50:maximum_bin+78]

            waves=waves_128

            noise_floor=np.nanmean(waves[10:20])
            where_above_nf=np.where(waves > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum=np.nanmax(waves[where_above_nf])
                total=np.sum(waves[where_above_nf])
                mean=np.nanmean(waves[where_above_nf])
                peaky=maximum/mean

            else:
                peaky = np.nan
                maximum = np.nan
                total = np.nan

        else:
            peaky = np.nan
            maximum = np.nan
            total = np.nan

        if 'maxs' in args:
            return maximum
        if 'totals' in args:
            return total
        if 'peaky' in args:
            return peaky

    peaky=np.apply_along_axis(by_row, 1, waves1, 'peaky') #numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    if 'wf_plots' in kwargs:
        maximums=np.apply_along_axis(by_row, 1, waves1, 'maxs')
        totals=np.apply_along_axis(by_row, 1, waves1, 'totals')

        for i in range(0,kwargs['wf_plots']):
            if i == 0:
                print("Plotting first "+str(kwargs['wf_plots'])+" waveforms")

            plt.plot(waves1[i,:])#, a, col[i],label=label[i])
            plt.axhline(maximums[i], color='green')
            plt.axvline(10, color='r')
            plt.axvline(19, color='r')
            plt.xlabel('Bin (of 256)')
            plt.ylabel('Power')
            plt.text(5,maximums[i],"maximum="+str(maximums[i]))
            plt.text(5,maximums[i]-2500,"total="+str(totals[i]))
            plt.text(5,maximums[i]-5000,"peakiness="+str(peaky[i]))
            plt.title('waveform '+str(i)+' of '+str(size)+'\n. Noise floor average taken between red lines.')
            plt.show()


    return peaky

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================


def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    # from scipy import asarray as ar,exp
    from numpy import asarray as ar, exp

    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD
```
This Python script is designed for processing Sentinel-3 SAR altimetry data, specifically focusing on extracting and analyzing waveform characteristics such as peakiness and stack standard deviation (SSD). The code utilizes scientific and machine learning libraries, including NumPy, SciPy, Matplotlib, and Scikit-learn, to preprocess, analyze, and classify satellite altimetry data. The script begins by importing the necessary libraries for handling NetCDF data, performing mathematical operations, and implementing clustering techniques such as K-Means, DBSCAN, and Gaussian Mixture Models (GMMs).

One of the key functions, peakiness, calculates the peakiness of waveforms, which is a crucial metric for determining the sharpness of the altimeter return signal. This function identifies the maximum peak in the waveform, computes the peak-to-mean power ratio, and provides an option to visualize waveforms with annotated peak values. The script also includes the unpack_gpod function, which extracts altimetry variables from SAR data while handling masked values and converting low-resolution (1Hz) data to high-resolution (20Hz) timestamps using interpolation. This ensures consistency in data representation.

Another critical function, calculate_SSD, determines the Stack Standard Deviation (SSD), which measures variability in waveform returns. It applies Gaussian curve fitting to estimate signal spread, helping in the classification of surface types such as ice, water, and land. This function can also generate visualizations of SSD calculations, providing insights into signal characteristics.

Overall, this script is highly useful for scientific research and remote sensing applications, enabling the classification of sea ice, leads, and open ocean surfaces using waveform-derived features. Additionally, the inclusion of machine learning clustering techniques (K-Means, DBSCAN, GMMs) enhances data segmentation, making it possible to detect and categorize patterns in Sentinel-3 altimetry data efficiently.


```python
path = '/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/'
SAR_file = 'S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3'
SAR_data = Dataset(path + SAR_file + '/enhanced_measurement.nc')

SAR_lat = unpack_gpod('lat_20_ku')
SAR_lon = unpack_gpod('lon_20_ku')
waves   = unpack_gpod('waveform_20_ku')
sig_0   = unpack_gpod('sig0_water_20_ku')
RIP     = unpack_gpod('rip_20_ku')
flag = unpack_gpod('surf_type_class_20_ku')

# Filter out bad data points using criteria (here, lat >= -99999)
find = np.where(SAR_lat >= -99999)
SAR_lat = SAR_lat[find]
SAR_lon = SAR_lon[find]
waves   = waves[find]
sig_0   = sig_0[find]
RIP     = RIP[find]

# Calculate additional features
PP = peakiness(waves)
SSD = calculate_SSD(RIP)

# Convert to numpy arrays (if not already)
sig_0_np = np.array(sig_0)
PP_np    = np.array(PP)
SSD_np   = np.array(SSD)

# Create data matrix
data = np.column_stack((sig_0_np, PP_np, SSD_np))

# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```





This Python script is designed to process and analyze Sentinel-3 SAR altimetry data, specifically using the SRAL (Synthetic Aperture Radar Altimeter) measurements. The script begins by defining the file path and loading the NetCDF dataset containing enhanced measurement data. It then extracts key parameters, including latitude (SAR_lat), longitude (SAR_lon), waveform data (waves), backscatter coefficient (sig_0), radar impulse response (RIP), and surface classification flags using the unpack_gpod function. These variables are crucial for understanding the surface characteristics of ocean, ice, and land regions.

To ensure data quality, the script applies a filtering step where only valid latitudes (≥ -99999) are retained, ensuring that invalid or missing data points are excluded from further analysis. Once the clean dataset is obtained, the script computes two essential waveform-derived features: Peakiness (PP) and Stack Standard Deviation (SSD). Peakiness (PP) is a measure of waveform sharpness, which helps differentiate between ocean, ice, and leads, while Stack Standard Deviation (SSD) quantifies variability in return signals, providing additional insights into surface roughness.

The extracted waveform parameters (sig_0, PP, and SSD) are then combined into a structured data matrix using np.column_stack(), ensuring that all relevant features are aligned for machine learning applications. To enhance the performance of clustering algorithms, the script applies feature scaling using the StandardScaler from Scikit-learn, which standardizes the data to have zero mean and unit variance. This step is crucial for ensuring that different features contribute equally to clustering models, such as K-Means, Gaussian Mixture Models (GMMs), or DBSCAN.

Overall, this script efficiently prepares and normalizes Sentinel-3 SAR altimetry data for further machine learning-based classification and clustering, enabling the identification of different surface types based on altimetric measurements.

There are some NaN values in the dataset so one way to deal with this is to delete them:
```python
# Remove any rows that contain NaN values
nan_count = np.isnan(data_normalized).sum()
print(f"Number of NaN values in the array: {nan_count}")

data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]

mask = ~np.isnan(data_normalized).any(axis=1)
waves_cleaned = np.array(waves)[mask] 
flag_cleaned = np.array(flag)[mask]

data_cleaned = data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
waves_cleaned = waves_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
flag_cleaned = flag_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
```

This script is responsible for data cleaning and preprocessing by handling missing values (NaNs) and filtering data based on predefined conditions.

First, the script counts the number of NaN (Not a Number) values in the data_normalized array using np.isnan(data_normalized).sum(). This step helps identify missing or invalid values, which may arise from incomplete satellite measurements or preprocessing errors. It then removes all rows containing NaN values using data_normalized[~np.isnan(data_normalized).any(axis=1)], ensuring that only complete and valid observations are used in further analysis.

Next, the script applies the same filtering process to related datasets, including waves_cleaned (which stores cleaned waveform data) and flag_cleaned (which contains classification flags). The mask ensures that only non-NaN entries from waves and flag are selected, keeping the datasets aligned.

Finally, the script filters the data based on surface classification flags, specifically keeping only values where flag_cleaned equals 1 or 2. These values typically correspond to specific surface types, such as open ocean, sea ice, or leads, depending on the dataset. This ensures that the dataset remains focused on relevant surface classifications, making it more suitable for unsupervised learning techniques like clustering or classification.

In summary, this script cleans the dataset, removes missing values, and retains only relevant data points based on surface classification flags, thereby improving the quality of input data for subsequent machine learning analyses.

Now, let’s proceed with running the GMM model as usual. Remember, you have the flexibility to substitute this with K-Means or any other preferred model:
```python
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```
This script applies Gaussian Mixture Model (GMM) clustering to the cleaned dataset, grouping data points into clusters based on their statistical properties.

First, a GaussianMixture model is initialized with two components (n_components=2), meaning that the algorithm will attempt to classify the data into two distinct clusters. The random_state=0 ensures that the results are reproducible by setting a fixed random seed.

Next, the model is trained using gmm.fit(data_cleaned), which fits the Gaussian Mixture Model to the preprocessed and standardized dataset (data_cleaned). The GMM algorithm assumes that the data is drawn from a mixture of two Gaussian distributions, each with its own mean and covariance. It iteratively refines the model using the Expectation-Maximization (EM) algorithm to estimate the parameters of these distributions.

Finally, the trained model is used to predict cluster assignments for each data point using gmm.predict(data_cleaned), storing the results in clusters_gmm. Each data point is assigned to one of the two Gaussian components, effectively segmenting the dataset into two distinct clusters.

This approach is particularly useful when working with continuous, normally distributed data and allows for soft clustering, meaning each data point has a probability of belonging to each cluster rather than a hard assignment. This makes GMM more flexible than K-means, especially when clusters have different shapes and variances.

We can also inspect how many data points are there in each class of your clustering prediction:

```python
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)
```

This script analyzes the distribution of data points across the clusters identified by the Gaussian Mixture Model (GMM).

First, the np.unique() function is used to count the occurrences of each cluster label in clusters_gmm, which stores the cluster assignments for the dataset. The return_counts=True argument ensures that the function returns both the unique cluster labels and their corresponding counts.

Next, the results are combined into a dictionary using dict(zip(unique, counts)), where the cluster labels serve as keys, and their respective counts as values. This structured output is stored in class_counts.

Finally, the script prints the number of data points assigned to each cluster, giving insight into how the Gaussian Mixture Model has distributed the data. If the clusters are significantly imbalanced, this may indicate differences in the density or distribution of features, which could warrant further analysis or tuning of the clustering parameters.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
This script performs an analysis and visualization of the mean and standard deviation of waveforms for different clusters identified by the Gaussian Mixture Model (GMM), aiming to differentiate between sea ice and leads in the dataset. First, the script calculates the mean and standard deviation for waveforms categorized as sea ice (cluster 0). It extracts the relevant waveforms using waves_cleaned[clusters_gmm==0] and computes their mean and standard deviation using NumPy functions. The mean waveform is then plotted, with a shaded area representing one standard deviation to indicate variability.

Next, the same process is repeated for waveforms categorized as leads (cluster 1), ensuring a comparative analysis between the two classes. The script then plots both the mean waveforms for ice and leads while shading their respective standard deviations for better visualization. A legend is included to clearly differentiate the two categories, and the plot is titled "Plot of mean and standard deviation for each class" to indicate the purpose of the visualization.

This approach allows for an intuitive understanding of waveform differences between sea ice and leads, highlighting variations in their reflectivity patterns based on the statistical properties of the dataset. Such an analysis is crucial for effective classification and interpretation of altimetry data in remote sensing applications.

![7c3651827a7ebef8494864dcce54cc8](https://github.com/user-attachments/assets/88e34ef0-0769-4cbb-9a76-073679057a75)

Visualization of the mean waveforms for sea ice and leads, with shaded areas representing one standard deviation. This plot helps to compare the reflectivity patterns of the two surface types, highlighting differences in waveform characteristics. The blue curve corresponds to sea ice, while the orange curve represents leads, demonstrating variability in their respective waveform signals.



```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

This code generates a plot of all waveform echoes in the dataset. It first creates a stacked array x, which represents the bin indices for the waveforms. The waveforms stored in waves_cleaned are then plotted against these bin indices using plt.plot(x, waves_cleaned). This visualization helps in understanding the distribution and variability of waveform signals across different observations. Finally, plt.show() displays the plot, allowing for an overall inspection of the waveform structures.

![image](https://github.com/user-attachments/assets/f0985c42-840f-49b7-a879-13fbe593d8e2)

Visualization of all waveform echoes, showing the distribution and intensity of the signals across different bins. The central peak indicates the dominant signal response, while the spread represents variations in waveform characteristics.

```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```

This code generates a plot displaying all waveform echoes that belong to the lead cluster, as identified by the Gaussian Mixture Model (GMM). It first creates an array x representing the bin positions for each waveform. Then, it extracts only the waveforms assigned to the lead class (clusters_gmm==1) and plots them to visualize their distribution. The resulting plot helps analyze the waveform characteristics specific to the lead category.

![image](https://github.com/user-attachments/assets/42acd430-03e3-406f-b61f-d0bbdee17288)


Visualization of all waveform echoes classified as lead by the Gaussian Mixture Model (GMM). The plot represents the waveform intensity distribution across different bins, highlighting the distinct characteristics of the lead category in the Sentinel-3 altimetry dataset.

```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```

This code generates a plot of waveforms associated with the sea ice cluster, which was identified using the Gaussian Mixture Model (GMM) clustering technique. It first creates an array x that represents the range of waveform bins for each cleaned echo belonging to the sea ice cluster. Then, it plots all the waveforms corresponding to this cluster, allowing for a visual representation of their structural characteristics. The plt.show() function displays the resulting plot, helping to analyze the waveform patterns of sea ice echoes in the Sentinel-3 altimetry data.


![image](https://github.com/user-attachments/assets/21330790-9b6e-4de3-b764-2c4f555f22c6)


Visualization of waveforms classified as sea ice using the Gaussian Mixture Model (GMM). Each line represents an individual waveform, providing insight into the distinct structural characteristics of sea ice echoes in Sentinel-3 altimetry data.


### Scatter Plots of Clustered Data

This code visualizes the clustering results using scatter plots, where different colors represent different clusters (clusters_gmm):

```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```

This code generates three scatter plots to visualize the clustering results obtained using the Gaussian Mixture Model (GMM) on Sentinel-3 altimetry data. Each scatter plot represents the relationship between different extracted features: sigma naught (σ₀), Peakiness Parameter (PP), and Stack Standard Deviation (SSD). The first plot visualizes σ₀ vs. PP, showing how these two parameters are distributed and how the clustering algorithm has grouped the data points. The second plot illustrates σ₀ vs. SSD, providing insights into how the backscatter coefficient varies with waveform deviation. The third plot displays PP vs. SSD, helping to further distinguish between different surface types, such as sea ice and leads. Each data point in the scatter plots is color-coded according to its assigned cluster, allowing for a clear visual representation of the clustering results. These plots help in understanding the distinct characteristics of sea ice and leads based on their altimetric properties.


![image](https://github.com/user-attachments/assets/96752147-485b-4edb-8cd7-4f6cfe140ed3)


![image](https://github.com/user-attachments/assets/cba6ea58-3e8a-44b9-bc7e-34beabcb75f2)


![image](https://github.com/user-attachments/assets/b6875888-69e5-40cb-96e3-b4743761c005)


These scatter plots visualize the clustering results of the Gaussian Mixture Model (GMM) applied to the Sentinel-3 altimetry dataset. Each plot highlights the relationships between different feature pairs:

First Plot: Displays the relationship between sig_0 (backscatter coefficient) and PP (peakiness parameter). The clusters, represented by different colors, show distinct groupings.

Second Plot: Shows the correlation between sig_0 and SSD (Stack Standard Deviation), indicating how these variables contribute to cluster separation.

Third Plot: Represents the distribution of clusters in the PP vs. SSD feature space, further illustrating the separation between sea ice and leads.

These visualizations help in understanding the structure and classification of different surface types using GMM clustering.


### Waveform Alignment Using Cross-Correlation

This code aligns waveforms in the cluster where clusters_gmm == 0 by using cross-correlation:

```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```

This code snippet performs cross-correlation-based alignment of waveforms within the cluster labeled as 0 by the Gaussian Mixture Model (GMM). The process starts by identifying a reference point, specifically the peak of the mean waveform for the cluster. Next, it computes the cross-correlation between each waveform and the first waveform in the cluster, determining the shift required to align the peaks. The waveforms are then adjusted accordingly using np.roll(), ensuring they are aligned with the reference point. Finally, a subset of ten equally spaced waveforms is plotted to visualize the alignment, providing insight into the consistency of waveform structures within the identified cluster.



![image](https://github.com/user-attachments/assets/a27f9aef-6c03-40db-80ef-5e370ef71738)


Caption: Visualization of 10 equally spaced waveforms from the cluster labeled as sea ice (clusters_gmm = 0) after alignment using cross-correlation. This alignment ensures that the peaks of the waveforms are synchronized, allowing for a clearer comparison of waveform structures within the identified cluster.

### Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels:

```python
flag_cleaned_modified = flag_cleaned - 1
```

This line of code, flag_cleaned_modified = flag_cleaned - 1, modifies the flag_cleaned array by subtracting 1 from each of its elements. The purpose of this operation is likely to adjust the labeling of data categories, ensuring that the values start from zero instead of one. This can be useful for compatibility with machine learning models that expect zero-based indexing or for standardizing the dataset before further processing. The modified array, flag_cleaned_modified, retains the same structure as flag_cleaned but with all values shifted down by one.


```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

This code evaluates the performance of the Gaussian Mixture Model (GMM) clustering method by comparing its predicted labels to the actual labels from the ESA dataset. First, the true labels are extracted from flag_cleaned_modified, while the predicted labels are obtained from the GMM clustering results stored in clusters_gmm. To assess the accuracy of the clustering, the code computes a confusion matrix using confusion_matrix(true_labels, predicted_gmm), which provides a summary of correct and incorrect classifications for each category. The confusion matrix is then printed to analyze the distribution of misclassified data points. Additionally, the code generates a classification report using classification_report(true_labels, predicted_gmm), which calculates key performance metrics such as precision, recall, and F1-score for each class. This report provides a more detailed evaluation of the clustering model's effectiveness in distinguishing between different data categories. By analyzing these outputs, we can determine how well the GMM model aligns with the actual classifications and gain insights into potential improvements.

```python
Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```


The confusion matrix and classification report summarize the performance of the Gaussian Mixture Model (GMM) clustering. The confusion matrix shows that 8,856 instances were correctly classified as class 0, while 3,293 instances were correctly classified as class 1. There were 22 misclassified instances of class 0 and 24 misclassified instances of class 1. The classification report indicates a precision, recall, and F1-score close to 1.00, demonstrating high accuracy in distinguishing between the two classes. The overall accuracy is 100%, suggesting that the clustering method effectively separates the data into meaningful groups.













