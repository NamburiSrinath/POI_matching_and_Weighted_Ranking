'''
Written by: Namburi Srinath

Note: Few imports are not needed. Remove them if needed. 
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
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from skcriteria import Data, MIN, MAX
from skcriteria.madm import simple
from functools import reduce
def construct_POI_tree(poi_data):
	'''
	Given the POI latitudes and longitudes, construct a ckDTree using these as nodes
	By constructing this tree, we can query the nearest POI for a particular device ID ping

	Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
	'''
	poi_tree = cKDTree(list(zip(poi_data['lat'], poi_data['lon'])))
	return poi_tree
def save_as_html_maps(poi_df, device_id_list, path):
	'''
	OPTIONAL FUNCTION. Use only if needed
	Overview: Plot pings on the JS map to visualise
	Steps:
	1. For every device id, get the average latitude and average longitude (3rd and 4th column in df) to to get an initial map
	2. Plot all the end latitude, longitude on map which also shows the start and end time when we hover
	'''
	poi_df_groupby_id = poi_df.groupby('device_id')
	for device_id in (device_id_list):
		df = poi_df_groupby_id.get_group(device_id)
		df.reset_index(drop=True, inplace=True)
		points = [tuple(x) for x in df.values]
		ave_lat = sum(p[3] for p in points)/len(points)
		ave_lon = sum(p[4] for p in points)/len(points)
		my_map = folium.Map(location=[ave_lat, ave_lon], zoom_start=14)
		for i in range(df.shape[0]):
			folium.Marker(
				(df['trip_end_latitude'][i], df['trip_end_longitude'][i]),
				icon = folium.Icon(color='blue', icon='info-sign'),
				popup = ('<b>Device Id:</b> {}, <b>Start time:</b> {}, <b>End time:</b> {}, <b>Dwell Time (sec):</b> {}, <b>POI Name:</b> {}, <b>POI Category:</b> {}, <b>POI Subcategory:</b> {}, <b>Distance from POI (km):</b> {}'.format(device_id, df['trip_start_time'][i], df['trip_end_time'][i], df['dwell_time_seconds'][i], df['POI_Name'][i], df['Category'][i], df['Subcategory'][i], df['Distance_(in_km)'][i] )),
				).add_to(my_map)
		my_map.save("{}/{}_out.html".format(path, device_id))
def query_for_nearest_POI(df, poi_dict):
	'''
	For every ping, get the nearest POI by querying the tree. Save everything to a dictionary where 
	Keys: Device ID, Start time, End time, End Lat, End Long and Dwell time. 
	Values: POI lat, POI long, distance (km), POI Name, POI Category and POI Subcategory
	'''
	for idx in range(df.shape[0]):
		distance, poi_row = poi_tree.query((df['trip_end_latitude'][idx], df['trip_end_longitude'][idx]), k=1, n_jobs=-1)
		poi_dict[device_id, df.iloc[idx]['trip_start_time'], df.iloc[idx]['trip_end_time'], df.iloc[idx]['trip_end_latitude'], df.iloc[idx]['trip_end_longitude'], df.iloc[idx]['dwell_time_seconds']] = (poi_data.iloc[poi_row]['lat'], poi_data.iloc[poi_row]['lon'], round(100*distance, 2), poi_data.iloc[poi_row]['name'], poi_data.iloc[poi_row]['category'], poi_data.iloc[poi_row]['subcategory'])
	return poi_dict
def write_to_csv(poi_dict):
	'''
	Write POI dict to a CSV file so that we can access as a dataframe for further processing
	Modify this function incase we don't want to save it as CSV (if we have any system constraints). 
	'''
	with open(POI_match_cluster, 'w+', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['device_id', 'trip_start_time', 'trip_end_time', 'trip_end_latitude', 'trip_end_longitude', 'dwell_time_seconds', 'poi_latitude', 'poi_longitude', 'Distance_(in_km)', 'POI_Name', 'Category',
			'Subcategory'])
		for keys, values in poi_dict.items():
			writer.writerow([keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], values[0], values[1], values[2], values[3], values[4], values[5]])
	csvfile.close()
	poi_df = pd.read_csv(POI_match_cluster)
	poi_df.sort_values(['device_id', 'trip_start_time'], axis=0, ascending=True, inplace=True)
	'''Comment if we dont want to save it into a CSV file'''
	poi_df.to_csv(POI_match_cluster, index=False)
	return poi_df
def extract_features_for_categories(poi_df, features_csv):
	'''
	Compute features needed to rank a device ID for various categories

	Feature 1 (frequency): Total number of visits 
	Feature 2 (UniqueCount): Total number of unique visits (i.e no of unique days a device ID appeared for a category)
	Feature 3 (Dwell time): Total time (in sec) spent in a category

	Steps:
	1. Group the data based on various columns and aggregate to get desired features
	2. Merge these features using the common columns (Device ID and category)  to a final Dataframe
	'''
	poi_df['date'] = pd.to_datetime(poi_df['trip_start_time']).dt.date
	poi_df_groupby_frequency = poi_df.groupby(['device_id', 'Category']).size().reset_index()
	poi_df_groupby_frequency.columns = ['device_id', 'Category', 'frequency']
	poi_df_groupby_date = poi_df.groupby(['device_id', 'Category', 'date']).size().reset_index()
	poi_df_groupby_date.columns = ['device_id', 'Category', 'date', 'frequency']
	poi_df_groupby_unique_date = poi_df_groupby_date.groupby(['device_id', 'Category']).size().reset_index()
	poi_df_groupby_unique_date.columns = ['device_id', 'Category', 'UniqueCount']
	poi_df_groupby_dwell_time = poi_df.groupby(['device_id', 'Category']).agg({'dwell_time_seconds':'sum'}).reset_index()
	dfs = [poi_df_groupby_frequency, poi_df_groupby_unique_date, poi_df_groupby_dwell_time]
	features = reduce(lambda left,right: pd.merge(left, right, on = ['device_id', 'Category'], how = 'inner'), dfs)
	'''Comment if we dont want to save it into a CSV file'''
	features.to_csv(features_csv, index=False)
	return features
def rank_based_on_features(features_group_by_device, device_id_list):
	'''
	Rank every Device ID for various categories based on computed features i.e
	Frequency, UniqueCount and Dwell Time

	Reference (Do read to understand the algorithm): 
	https://towardsdatascience.com/ranking-algorithms-know-your-multi-criteria-decision-solving-techniques-20949198f23e

	Steps:
	1. For every device ID, form criteria Data in such a way that 
		a. The features will be input data
		b. Every feature should have a direction of optimality (MAX or MIN)
		c. Describe entity name (device id) and column names (cnames) 
		d. Weights are optional (i.e how much importance we need to give for every feature)
	2. There are 4 ranking methods described in blog. I used only one to determine the rank of the category
	As we don't have any ground truth, it's not possible to choose the best ranking method (or) if the ranking method gives proper o/p

	Note: Currently using weightedProduct_MaxNorm_inverse for ranking
	'''
	ranking = pd.DataFrame()
	for device_id in (device_id_list):
		device_id_data = features_group_by_device.get_group(device_id)
		criteria_data = Data(
		device_id_data.iloc[:, 2:],          
		[MAX, MAX, MAX],      
		anames = device_id_data['device_id'], 
		cnames = device_id_data.columns[2:], 
		# weights=[1,1,1,1,1]           
		)

		'''Uncomment if we need other ranking methods. My opinion is it doesnt matter much as 
		we dont have ground truth and thus cant compare the superiority of different ranking methods'''
		# dm = simple.WeightedSum(mnorm="sum")
		# dec = dm.decide(criteria_data)
		# device_id_data.loc[:, 'rank_weightedSum_sumNorm_inverse'] = dec.rank_

		# dm = simple.WeightedSum(mnorm="max")
		# dec = dm.decide(criteria_data)
		# device_id_data.loc[:, 'rank_weightedSum_maxNorm_inverse'] = dec.rank_

		# dm = simple.WeightedProduct(mnorm="sum")
		# dec = dm.decide(criteria_data)
		# device_id_data.loc[:, 'rank_weightedProduct_sumNorm_inverse'] = dec.rank_

		dm = simple.WeightedProduct(mnorm="max")
		dec = dm.decide(criteria_data)
		device_id_data.loc[:, 'Rank'] = dec.rank_

		ranking = pd.concat([ranking, device_id_data], axis=0)
	ranking.sort_values(['device_id'], axis=0, ascending=True, inplace=True)
	ranking.to_csv(ranking_csv, index=False)

if __name__ == '__main__':
	'''
	Input files:

	input_file: The entire data containing the trip details (start lat/long, end lat/long), dwell time
	POI_file: POI dataset (I worked with Bangalore's dataset). Has ID, name, category, subcateogry, lat/long

	Output files:

	POI_match_cluster: Intermediate file. Has information related to matching each trip to a POI
	features.csv: Intermediate file. Saving computed features from POI_match_cluster
	ranking.csv: Final O/P file. Ranks every device id on various categories using multi-criteria decision
	'''
	code_start_time = time.time()
	input_file = 'LocTruth_Stop_Data.csv'
	POI_file = 'Bangalore_POI_complete_list.csv'
	POI_match_cluster = 'POI_match_cluster.csv'
	features_csv = 'features.csv'
	ranking_csv = 'ranking.csv'
	path = './Manideep_data_Trajectory_HTML_Files_Grouped_by_only_IDs/'
	poi_dict = {}
	trip_data, poi_data = pd.read_csv(input_file), pd.read_csv(POI_file)
	'''
	Construct POI tree using the lat/longs of POIs as nodes
	'''
	poi_tree = construct_POI_tree(poi_data)
	print("Initial data shape: {}".format(trip_data.shape))
	print("No of categories and subcategories in POI database: {}, {}".format(len(poi_data['category'].unique()), len(poi_data['subcategory'].unique())))
	print("No of unique devices: {}".format(len(trip_data.maid.unique())))
	trip_data.sort_values(['maid'], axis=0, ascending=True, inplace=True)
	data_group_by_id = trip_data.groupby('maid')
	device_id_list = list(trip_data.maid.unique())
	for device_id in (device_id_list):
		device_id_pings = data_group_by_id.get_group(device_id)
		df = device_id_pings[['trip_end_latitude', 'trip_end_longitude', 'trip_start_time', 'trip_end_time', 'dwell_time_seconds']].reset_index()
		poi_dict = query_for_nearest_POI(df, poi_dict)
	poi_df = write_to_csv(poi_dict)
	features = extract_features_for_categories(poi_df, features_csv)
	features_group_by_device = features.groupby(['device_id'])
	rank_based_on_features(features_group_by_device, device_id_list)
	save_as_html_maps(poi_df, device_id_list, path)
	code_end_time = time.time()
	print("Time for running code {}".format(code_end_time - code_start_time))