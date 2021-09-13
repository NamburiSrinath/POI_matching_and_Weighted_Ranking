'''
Written by: Namburi Srinath

Input: GPS data of all the devices and POI database
Ouput: Cluster for each device ID for all the avaiable days
* Add h3 column to the POI database using the (lat,long)
Algorithm: 
1. Clean the GPS data so that no duplicate entries should be present
2. Pass the GPS data corresponding to one device ID to DBSCAN which gives the clusters for the data cloud
3. For each of this cluster, get the centermost point (not the centroid of the cluster) 
so that the entire cluster can be represented by this centermost point.
4. Plot all the clusters and centermost points to Leaflet (Red -> Centermost Point, Blue -> Normal GPS ping)

To do:
5. For every centermost point in the data cloud, convert it to h3 index and match it with POI database to get a POI match
'''
import pandas as pd
import numpy as np
import time
import csv
import h3
import datetime as dt
import folium
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from haversine import haversine,Unit,haversine_vector
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

resolution = 15
kms_per_radian = 6371.0088
# epsilon = 0.5 / kms_per_radian
find_no_of_clusters = {}
total_no_of_clusters = 0
def data_clean(data):
	'''
	Removes duplicates in data based on ID and timestamp 
	i.e there should be only one location per device ID per timestamp
	'''
	data.drop_duplicates(subset = ['maid', 'timestamp'], inplace=True)
	data.to_csv(input_file, inplace=True, index=False)
	print("After duplicates removing: {}".format(data.shape))
	return data
def add_h3_column(data):
	'''
	Convert lat, long to h3 and add as a column. 
	We will use it for POI matching
	'''
	h3_index_list = []
	print(data.shape)
	for i in range (data.shape[0]):
		lat, lon = data['latitude'][i], data['longitude'][i]
		h3_index = h3.geo_to_h3(lat, lon, resolution)
		h3_index_list.append(h3_index)
	data['h3_index'] = h3_index_list
	data.to_csv(input_file, inplace=True, index=False)
	print("H3 column added to Dataframe")
	return data
def get_centermost_point(cluster):
	'''
	Given a cluster, get the centermost point (Not the centroid)
	Centroid -> Middle point of the cluster. It might or might not exist for this trip.
	So, get the nearest point from this centroid to the cluster, that will be the centermost point in cluster
	'''
	# print(cluster[:,:2], type(cluster), cluster.shape)
	centroid = (MultiPoint(cluster[:,:2]).centroid.x, MultiPoint(cluster[:,:2]).centroid.y)
	centermost_point = min(cluster[:,:2], key=lambda point: great_circle(point, centroid).m)
	return tuple(centermost_point)
def get_optimum_epsilon(X):
	'''
	Given the cluster, fit and get the optimum epsilon
	'''
	neigh = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric='haversine')
	nbrs = neigh.fit(X)
	distances, indices = nbrs.kneighbors(X)
	distances = distances[:,2]
	distances = np.sort(distances, axis=0)
	slopes = [t - s for s, t in zip(distances, distances[1:])]
	epsilon = max(slopes)
	# plt.plot(distances, 'bo-')
	# plt.xlabel('Index')
	# plt.ylabel('Distance')
	# plt.title('Elbow method to select epsilon')
	# plt.show()
	# print(device_id, 100*epsilon)
	# EPSILON RETURNED SHOULD BE IN KMS BUT NOT, MULTIPLY WITH 100
	return epsilon/kms_per_radian
def construct_POI_tree(poi_data):
	poi_tree = cKDTree(list(zip(poi_data['lat'], poi_data['lon'])))
	return poi_tree
def get_clusters(df, epsilon):
	cluster_labels = []
	num_clusters = 0
	if(epsilon > 0):
		db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(df[['latitude', 'longitude']]))
		cluster_labels = db.labels_
		num_clusters = max(cluster_labels) + 1
	return cluster_labels, num_clusters
if __name__ == '__main__':
	code_start_time = time.time()
	input_file = '1000_devices.csv'
	POI_file = 'Bangalore_POI_complete_list.csv'
	POI_match_cluster = 'POI_match_cluster.csv'
	stats_file = 'no_of_clusters.csv'
	poi_dict = {}
	data = pd.read_csv(input_file)
	poi_data = pd.read_csv(POI_file)
	poi_tree = construct_POI_tree(poi_data)
	print("Initial data shape: {}".format(data.shape))
	# data = data_clean(data)
	# data = add_h3_column(data)
	print("No of unique devices: {}".format(len(data.maid.unique())))
	data_group_by_id = data.groupby('maid')
	device_id_list = list(data.maid.unique())
	data['timestamp'] = pd.to_datetime(data['timestamp'])
	data['date'] = data.timestamp.dt.date
	# for idx in range(1):
	for device_id in (device_id_list):
		# device_id = device_id_list[idx]
		start_time, end_time = [], []
		device_id_pings = data_group_by_id.get_group(device_id)
		df = device_id_pings[['latitude', 'longitude', 'timestamp', 'h3_index']]
		points = [tuple(x) for x in df.values]
		ave_lat = sum(p[0] for p in points)/len(points)
		ave_lon = sum(p[1] for p in points)/len(points)
		my_map = folium.Map(location=[ave_lat, ave_lon], zoom_start=14)
		for each in points:  
			folium.Marker(
				(each[0], each[1]),
				popup = (device_id, each[2], each[3])
				).add_to(my_map)
		my_map.save("./Trajectory_HTML_Files_Grouped_by_only_IDs/{}_out.html".format(device_id))
		coords = df.as_matrix(columns=['latitude', 'longitude', 'timestamp', 'h3_index'])
		epsilon = get_optimum_epsilon(df[['latitude', 'longitude']])
		cluster_labels, num_clusters = get_clusters(df, epsilon)
		# print(cluster_labels)
		if (num_clusters != 0):
			clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
			for i in range (num_clusters):
				start_time.append(clusters[i][0][2])
				end_time.append(clusters[i][-1][2])
			print('Number of clusters: {}'.format(num_clusters))
			centermost_points = clusters.map(get_centermost_point)
			lats, lons = zip(*centermost_points)
			rep_points = pd.DataFrame({'longitude':lons, 'latitude':lats})
			rs = rep_points.apply(lambda row: df[(df['latitude']==row['latitude']) & (df['longitude']==row['longitude'])].iloc[0], axis=1)
			rs['start_time'] = start_time
			rs['end_time'] = end_time
			''' Uncomment this plot Incase we want to see how DBSCAN output looks''' 
			# fig, ax = plt.subplots(figsize=[10, 6])
			# rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
			# df_scatter = ax.scatter(df['longitude'], df['latitude'], c='k', alpha=0.9, s=3)
			# ax.set_title('Full data set vs DBSCAN reduced set')
			# ax.set_xlabel('Longitude')
			# ax.set_ylabel('Latitude')
			# ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
			# plt.savefig('{device_id}_{date_id}.png')
			for i in range(rs.shape[0]):
				folium.Marker(
					(rs['latitude'][i], rs['longitude'][i]),
					icon = folium.Icon(color='red', icon='info-sign'),
					popup = ('<b>Device Id:</b> {}, <b>Start time:</b> {}, <b>End time:</b> {}'.format(device_id, rs['start_time'][i], rs['end_time'][i])),
					).add_to(my_map)
			my_map.save("./Trajectory_HTML_Files_Grouped_by_only_IDs/{}_out.html".format(device_id))
			find_no_of_clusters[device_id] = num_clusters
		elif (num_clusters == 0):
			find_no_of_clusters[device_id] = 0
		for cluster_idx in range(rs.shape[0]):
			distance, poi_row = poi_tree.query((rs['latitude'][cluster_idx], rs['longitude'][cluster_idx]), k=1, n_jobs=-1)
			poi_dict[device_id, rs.iloc[cluster_idx]['latitude'], rs.iloc[cluster_idx]['longitude']] = (poi_data.iloc[poi_row]['name'], poi_data.iloc[poi_row]['lat'],
			poi_data.iloc[poi_row]['lon'], round(100*distance, 2), poi_data.iloc[poi_row]['category'], poi_data.iloc[poi_row]['subcategory'])
	with open(stats_file, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['Device ID', 'No of clusters'])
		for keys, values in find_no_of_clusters.items():
			writer.writerow([keys, values])
	csvfile.close()
	stats_df = pd.read_csv(stats_file)
	stats_df.sort_values(['Device ID'], axis=0, ascending=True, inplace=True)
	stats_df.to_csv(stats_file, index=False)
	with open(POI_match_cluster, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['Device ID', 'Cluster Latitude', 'Cluster Longitude', 'POI Name', 'POI Latitude', 'POI Longitude', 'Distance (in km)', 'Category',
			'Subcategory'])
		for keys, values in poi_dict.items():
			writer.writerow([keys[0], keys[1], keys[2], values[0], values[1], values[2], values[3], values[4], values[5]])
	csvfile.close()
	poi_df = pd.read_csv(POI_match_cluster)
	poi_df.sort_values(['Device ID'], axis=0, ascending=True, inplace=True)
	poi_df.to_csv(POI_match_cluster, index=False)
	code_end_time = time.time()
	print("Time for running code for 1000 device IDs {}".format(code_end_time - code_start_time))

	
		
		

	