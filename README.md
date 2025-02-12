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
This section explores unsupervised learning for Earth Observation (EO) with a hands-on approach. Focusing on classification tasks, we use machine learning to detect patterns and group data without predefined labels.

Key tasks include:

1-Distinguishing sea ice from leads using Sentinel-2 optical data

2-Classifying sea ice and leads using Sentinel-3 altimetry data

By the end, you'll have a practical foundation in unsupervised learning for remote sensing and EO analysis.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]
#### Introduction to K-means Clustering
K-means clustering is an unsupervised learning algorithm that partitions data into k clusters based on feature similarity [MacQueen et al., 1967]. The process involves initializing k centroids, assigning data points to the nearest centroid, updating centroids based on cluster means, and iterating until stability is reached. K-means is widely used for pattern recognition, data segmentation, and exploratory analysis, making it a fundamental tool in unsupervised learning.
#### Why K-means for Clustering?
K-means clustering is effective when the data structure is unknown, as it does not require prior knowledge of distribution, making it ideal for exploratory analysis and pattern detection. It is also efficient and scalable, handling large datasets with minimal complexity, making it a preferred choice for real-world applications.

#### Key Components of K-means
K-means clustering relies on key factors: choosing k, which must be predefined and impacts results; centroid initialization, which affects final clustering; the assignment step, where data points are grouped by proximity to the nearest centroid using squared Euclidean distance; and the update step, where centroids are recalculated based on the mean position of assigned points.
#### The Iterative Process of K-means
K-means iterates through assignment and update steps until centroids stabilize, minimizing intra-cluster variation. This ensures convergence to an optimal clustering solution, though it may sometimes settle on a local optimum.
#### Advantages of K-means
K-means is highly efficient, making it ideal for large datasets, and offers easy interpretation, allowing for clear analysis of data patterns.
#### Basic Code Implementation
This section provides a K-means clustering implementation as a practical introduction to the algorithm. In Google Colab, the script mounts Google Drive using drive.mount('/content/drive') for seamless dataset access. It also installs Rasterio for geospatial raster data and netCDF4 for handling large-scale scientific data. Using scikit-learn, the script generates 100 random data points, initializes a K-means model with four clusters, and assigns each point using kmeans.fit(X). A scatter plot visualizes the clusters with color-coded points, while computed centroids are marked with black dots. The plot, displayed with plt.show(), illustrates how K-means groups data for pattern recognition and segmentation.

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

![image](https://github.com/user-attachments/assets/e336776c-92d6-4d6a-b3fc-be0c1f41960d)

Visualization of K-means clustering results on a randomly generated dataset. The colored points represent individual data samples grouped into four clusters, while the black dots indicate the centroids of each cluster, calculated by the K-means algorithm.

<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]
#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) are a probabilistic clustering technique that models data as a combination of multiple Gaussian distributions, each with its own mean and variance [Reynolds et al., 2009]. GMMs are widely used for clustering and density estimation, providing a flexible way to represent complex data distributions.

#### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models (GMM) offer significant advantages in clustering by providing a soft clustering approach, where each data point is assigned a probability of belonging to multiple clusters rather than being placed into a single category like in K-means. This probabilistic classification allows for a more nuanced and flexible clustering method, especially when dealing with uncertainty. Additionally, unlike K-means, which assumes clusters are spherical, GMM adapts to varying cluster shapes and sizes by adjusting the covariance structure of each Gaussian component. This makes it particularly effective for datasets with overlapping distributions or varying density regions, providing a more precise and adaptable clustering solution.

#### Key Components of GMM
Gaussian Mixture Models (GMM) require defining the number of components, similar to selecting clusters in K-means, as it determines how many Gaussian distributions will model the data. The model is refined using the Expectation-Maximization (EM) algorithm, which alternates between estimating the probability of each data point belonging to a Gaussian and updating parameters like mean, variance, and weight to maximize likelihood. Additionally, the covariance structure plays a crucial role in shaping clusters, allowing for spherical, diagonal, tied, or fully adaptable cluster forms, making GMM highly flexible for complex data distributions.


#### The EM Algorithm in GMM
The Expectation-Maximization (EM) algorithm optimizes clustering through an iterative two-step process. In the Expectation Step (E-step), probabilities are assigned to each data point, estimating the likelihood of belonging to a specific Gaussian component. The Maximization Step (M-step) then updates the mean, variance, and weight of each Gaussian to maximize the model’s likelihood. This cycle repeats until convergence, when the parameters stabilize, ensuring an optimal fit for the dataset.


#### Advantages of GMM
Gaussian Mixture Models (GMM) offer probabilistic soft clustering, assigning a probability score to each data point’s cluster membership, which captures uncertainty and overlapping structures. Unlike K-means, GMM allows for flexible cluster shapes, accommodating varying sizes, orientations, and densities. This adaptability makes GMM an excellent choice for clustering complex datasets with overlapping distributions.

#### Basic Code Implementation
Below is a basic Gaussian Mixture Model (GMM) implementation, providing a foundational understanding of how it works in clustering tasks. The code uses GaussianMixture from sklearn.mixture, along with matplotlib for visualization and numpy for numerical operations. It generates 100 random data points in a 2D space, initializes a GMM with three components, and fits the model to the dataset. Cluster assignments are predicted, and the results are visualized with data points color-coded by cluster, while computed cluster centers (means) are highlighted in black. This demonstrates how GMM groups data using probabilistic distributions.

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

![image](https://github.com/user-attachments/assets/ba6c9c55-ed2b-43ef-82a8-543445e64478)

Visualization of clustering results using the Gaussian Mixture Model (GMM). The data points are grouped into three distinct clusters, each represented by a different color. The black points indicate the computed cluster centers (means), highlighting the probabilistic nature of GMM clustering.

### Image Classification
This section applies unsupervised learning for image classification, focusing on distinguishing sea ice from leads using Sentinel-2 imagery. By leveraging clustering algorithms, patterns can be identified and classified without labeled data, improving the analysis and interpretation of remote sensing data efficiently.

#### K-Means Implementation
This Python script applies K-means clustering to a Sentinel-2 satellite image, classifying surface features based on reflectance values from the Red (B4) band. It imports essential libraries, including rasterio for satellite imagery, numpy for numerical operations, KMeans from sklearn.cluster for clustering, and matplotlib.pyplot for visualization. The script loads Band 4 using rasterio.open(), storing pixel values in a NumPy array, while a valid data mask filters out no-data pixels. The image data is reshaped into a 1D array, and K-means is applied with two clusters (n_clusters=2) to distinguish surface types like sea ice and open water. The classified pixel labels are reshaped to the original image dimensions, with masked pixels assigned -1 to maintain data integrity. The clustering results are visualized using plt.imshow(), with distinct colors representing different clusters. This approach enables unsupervised classification of Sentinel-2 imagery, providing insights into surface variations for climate monitoring, environmental research, and land cover classification, without requiring labeled training data.

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

![image](https://github.com/user-attachments/assets/4dbda879-080b-406b-b1f0-24b97a1d7aa8)

This image shows the K-means clustering result on a Sentinel-2 B4 optical band, grouping pixels into two clusters. Yellow regions likely represent sea ice or land, while darker areas indicate open water or other surface types. The color bar displays cluster labels, with -1 assigned to no-data areas. This classification enhances the distinction of surface features in remote sensing imagery.

#### GMM Implementation
This Python script applies Gaussian Mixture Model (GMM) clustering to a Sentinel-2 B4 optical band using scikit-learn. It reads the B4 band with Rasterio, stacks the data into a NumPy array, and creates a mask to exclude zero-value pixels. The valid data is reshaped and clustered using GMM with two components, allowing for soft clustering, where each pixel is assigned a probability of belonging to each cluster. The results are stored in an array, with -1 for no-data areas, and visualized with Matplotlib, color-coding different clusters. This method is effective for analyzing sea ice, land cover, and environmental patterns in Sentinel-2 imagery.

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

![2e7a5d5ebb2325493b5e9bdfa718c6d](https://github.com/user-attachments/assets/d6655404-0987-4593-8fc0-2bcc32270091)

Visualization of Gaussian Mixture Model (GMM) clustering applied to Sentinel-2 Band 4 imagery. The image showcases different clusters identified in the dataset, with the color scale representing distinct cluster labels. This method helps differentiate between various land cover types, such as sea ice, open water, and land surfaces, based on spectral reflectance patterns.


### Altimetry Classification
This section applies unsupervised learning to classify Sentinel-3 altimetry data, focusing on distinguishing sea ice from leads using satellite-derived elevation measurements. This approach enhances the analysis of surface features, improving insights into ice dynamics and oceanographic processes.

#### Read in Functions Needed
This Python script processes and analyzes Sentinel-3 SAR altimetry data, focusing on waveform characteristics like peakiness and stack standard deviation (SSD) to classify sea ice and leads. Using NumPy, SciPy, Matplotlib, and Scikit-learn, it extracts altimetry variables, cleans missing values, and applies machine learning clustering techniques such as K-Means, DBSCAN, and Gaussian Mixture Models (GMMs). The unpack_gpod function extracts key parameters like latitude, longitude, waveforms, and backscatter coefficient, ensuring data consistency by interpolating low-resolution (1Hz) data to high-resolution (20Hz) timestamps. The calculate_SSD function estimates waveform variability using Gaussian curve fitting, helping classify surface types. The dataset is then standardized using Scikit-learn’s StandardScaler to optimize clustering performance. To handle missing values, the script identifies and removes NaN entries, ensuring only valid observations are included. It filters data based on surface classification flags, keeping only relevant values for clustering. The GMM model is applied with two components, classifying data into distinct clusters based on statistical properties. The Expectation-Maximization (EM) algorithm refines cluster parameters iteratively, allowing for soft clustering where data points have probabilistic cluster assignments. The script also analyzes cluster distribution, using np.unique() to count data points per cluster. It visualizes waveform differences between sea ice and leads, plotting mean and standard deviation for each class to highlight reflectivity variations. This approach enhances remote sensing classification, improving the interpretation of Sentinel-3 altimetry data for environmental and climate studies.

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













