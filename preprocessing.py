#Import packages
import numpy as np
import pandas as pd
from datetime import datetime 


#Load the dataset
data = pd.read_csv('data/fraud_dataset.csv', 
                   sep = ';', 
                   decimal = ',')

#Identify the features from the raw data that will be removed at the end
keywords = ['_id','_date_','postal_code', '_time_','_birth']
columns_to_drop = [c for c in data.columns for k in keywords if k in c]


#Create new feature from raw data

#Extract the hour of the claim from the raw time data
data['claim_hour'] = data['claim_time_occured'].apply(lambda row : row if np.isnan(row) else str(row)[0:2] if  len(str(row)) ==6 else str(row)[0:1])

#Postal codes matching features
data['driver_claim_postal_code'] = data.apply(lambda row : row['driver_postal_code'] == row['claim_postal_code'],axis =1)
data['driver_policy_holder_postal_code'] = data.apply(lambda row : row['driver_postal_code'] == row['policy_holder_postal_code'],axis=1)
data['driver_repair_postal_code'] = data.apply(lambda row : row['driver_postal_code'] == row['repair_postal_code'], axis =1)


#Driver occurences in dataset
driver_unique_count = data.groupby('driver_id').agg({'claim_id':'count'}).rename( columns = {'claim_id':'claim_count'})
data['driver_claims_count'] = data['driver_id'].apply(lambda row : driver_unique_count.loc[row,'claim_count'])

#ID matching features
data['claim_driver_vehicle'] = data.apply(lambda row : row['claim_vehicle_id'] == row['driver_vehicle_id'], axis = 1)
data['third_party_1_driver_vehicle'] = data.apply(lambda row : row['third_party_1_vehicle_id'] == row['driver_vehicle_id'], axis = 1)
data['third_party_2_driver_vehicle']  = data.apply(lambda row : row['third_party_2_vehicle_id'] == row['driver_vehicle_id'], axis = 1)
data['third_party_3_driver_vehicle']  = data.apply(lambda row : row['third_party_3_vehicle_id'] == row['driver_vehicle_id'], axis = 1)
data['driver_policy_holder_expert'] = data.apply(lambda row : row['driver_expert_id'] == row['policy_holder_expert_id'],axis=1)
data['driver_third_party_1_expert'] = data.apply(lambda row : row['driver_expert_id'] == row['third_party_1_expert_id'],axis=1)
data['driver_third_party_2_expert']  = data.apply(lambda row : row['driver_expert_id'] == row['third_party_2_expert_id'],axis=1)
data['driver_third_party_3_expert'] = data.apply(lambda row : row['driver_expert_id'] == row['third_party_3_expert_id'],axis=1)

#Identify if a policy holder, a repair or a third party is also a driver in another claim
set_driver_policy_holder_ids = set(data.driver_id).intersection(data.policy_holder_id)
set_driver_third_party_1_ids = set(data.driver_id).intersection(data.third_party_1_id)
set_driver_third_party_2_ids = set(data.driver_id).intersection(data.third_party_2_id)
set_driver_third_party_3_ids = set(data.driver_id).intersection(data.third_party_3_id)
set_driver_repair_ids = set(data.driver_id).intersection(data.repair_id)

data['policy_holder_is_a_driver'] = data.apply(lambda row : row['policy_holder_id'] in set_driver_policy_holder_ids,axis =1)
data['thirdy_party_1_is_a_driver'] = data.apply(lambda row : row['third_party_1_id'] in set_driver_third_party_1_ids,axis =1)
data['thirdy_party_2_is_a_driver'] = data.apply(lambda row : row['third_party_2_id'] in set_driver_third_party_2_ids,axis =1)
data['thirdy_party_3_is_a_driver']  = data.apply(lambda row : row['third_party_3_id'] in set_driver_third_party_3_ids,axis =1)
data['repair_is_a_driver'] = data.apply(lambda row : row['repair_id'] in set_driver_repair_ids,axis =1)


#Time delta features
def compute_time_delta(recent_date, 
                       old_date,
                       time_format_recent = '%Y%m%d',
                       time_format_old = '%Y%m%d',
                       desired_units = 'days'):
    '''
    Given two dates, compute the time delta between them.
    Args:
        recent_date (str) : the most recent date in string format
        old_date (str) : the oldest date in string format
        time_format_recent (str) : the desired datetime format for the recent date
        time_format_old (str) : the desired datetime format for the old date
        desired_units (str)  : the desired time unit (years, months, days, hours), for the output
     '''

    if np.isnan(recent_date) or np.isnan(old_date):
        return np.nan
    else:      
        time_delta = (datetime.strptime(str(int(recent_date)), time_format_recent) 
                      - datetime.strptime(str(int(old_date)),time_format_old))
        if desired_units == 'years':
            return time_delta.days//365
        if desired_units == 'months':
            return time_delta.days//30
        if desired_units == 'days':
            return  time_delta.days
        if desired_units == 'hours':
            return time_delta.total_seconds() /3600


data['delta_claim_registered_occured'] = data.apply(lambda  row : compute_time_delta(row['claim_date_registered'], 
                                                                             row['claim_date_occured']), axis = 1)

data['delta_claim_occured_vehicle_inuse'] = data.apply(lambda row : compute_time_delta(row['claim_date_occured'], 
                                                                                 row['claim_vehicle_date_inuse'],                                                                                 
                                                                                 time_format_old = '%Y%m',
                                                                                 desired_units = 'months'), axis = 1)

data['delta_claim_registered_policy_date_start'] = data.apply(lambda  row : compute_time_delta(row['claim_date_registered'], 
                                                                                       row['policy_date_start'],
                                                                                       time_format_old = '%Y%m',
                                                                                       desired_units = 'months'),axis=1)

data['delta_claim_registered_policy_last_renewed'] = data.apply(lambda  row : compute_time_delta(row['claim_date_registered'], 
                                                                                       row['policy_date_last_renewed'],
                                                                                       time_format_old = '%Y%m',
                                                                                       desired_units = 'months'),axis=1)

data['delta_policy_next_expiry_claim_registered'] = data.apply(lambda  row : compute_time_delta(row['policy_date_next_expiry'],
                                                                                       row['claim_date_registered'], 
                                                                                       time_format_recent = '%Y%m',
                                                                                       desired_units = 'months'),axis=1)


data['delta_policy_last_renewed_start_date'] = data.apply(lambda  row : compute_time_delta(row['policy_date_last_renewed'],
                                                                                       row['policy_date_start'], 
                                                                                       time_format_recent = '%Y%m',
                                                                                        time_format_old = '%Y%m',
                                                                                       desired_units = 'months'),axis=1)

data['delta_policy_next_expiry_last_renewed'] = data.apply(lambda  row : compute_time_delta(row['policy_date_next_expiry'],
                                                                                       row['policy_date_last_renewed'], 
                                                                                       time_format_recent = '%Y%m',
                                                                                        time_format_old = '%Y%m',
                                                                                       desired_units = 'months'),axis=1)


#Person's age feature derived as the difference between the claim year and birth year
def compute_age(year_birth, claim_occured):
    if np.isnan(year_birth):
        age = np.nan
    else:
        age = int(str(claim_occured)[0:4]) - int(str(year_birth)[0:4])
    return age

data['driver_age'] = data.apply(lambda row : compute_age(row['driver_year_birth'], row['claim_date_occured']),axis=1)
data['policy_holder_age'] = data.apply(lambda row : compute_age(row['policy_holder_year_birth'], row['claim_date_occured']),axis=1)
data['third_party_1_age'] = data.apply(lambda row : compute_age(row['third_party_1_year_birth'], row['claim_date_occured']),axis=1)
data['third_party_2_age'] = data.apply(lambda row : compute_age(row['third_party_2_year_birth'], row['claim_date_occured']),axis=1)
data['third_party_3_age'] = data.apply(lambda row : compute_age(row['third_party_3_year_birth'], row['claim_date_occured']),axis=1)
data['repair_age'] = data.apply(lambda row : compute_age(row['repair_year_birth'], row['claim_date_occured']),axis=1)


#Fill missing values for categorical features

data['claim_alcohol'] = data['claim_alcohol'].fillna('Missing')
data['claim_language'] = data['claim_language'].fillna('Missing')
data['claim_vehicle_brand'] = data['claim_vehicle_brand'].fillna('Missing')
data['claim_vehicle_type'] = data['claim_vehicle_type'].fillna('Missing')
data['claim_vehicle_cyl'] = data['claim_vehicle_cyl'].fillna('Missing')
data['claim_vehicle_fuel_type'] = data['claim_vehicle_fuel_type'].fillna('Missing')
data['third_party_1_form'] = data['third_party_1_form'].fillna('Missing')
data['third_party_2_form'] = data['third_party_2_form'].fillna('Missing')
data['third_party_3_form'] = data['third_party_3_form'].fillna('Missing')
data['repair_form'] = data['repair_form'].fillna('Missing')
data['third_party_1_country'] = data['third_party_1_country'].fillna('Missing')
data['third_party_2_country'] = data['third_party_2_country'].fillna('Missing')
data['third_party_3_country'] = data['third_party_3_country'].fillna('Missing')
data['third_party_1_vehicle_type'] = data['third_party_1_vehicle_type'].fillna('Missing')
data['third_party_2_vehicle_type'] = data['third_party_2_vehicle_type'].fillna('Missing')
data['third_party_3_vehicle_type'] = data['third_party_3_vehicle_type'].fillna('Missing')
data['repair_country'] = data['repair_country'].fillna('Missing')
data['third_party_1_injured'] = data['third_party_1_injured'].fillna('Missing')
data['third_party_2_injured'] = data['third_party_2_injured'].fillna('Missing')
data['third_party_3_injured'] = data['third_party_3_injured'].fillna('Missing')

#Fill missing values for numerical features
data.claim_vehicle_load  =  data.claim_vehicle_load.apply(lambda row : float(row))
data['claim_vehicle_load'] = data['claim_vehicle_load'].fillna(np.mean(data['claim_vehicle_load']))
data['claim_vehicle_power'] = data['claim_vehicle_power'].fillna(np.mean(data['claim_vehicle_power']))
data['policy_premium_100'] = data['policy_premium_100'].fillna(np.mean(data['policy_premium_100']))
data['policy_coverage_1000'] = data['policy_coverage_1000'].fillna(np.mean(data['policy_coverage_1000']))
data['driver_age'] = data['driver_age'].fillna(np.mean(data['driver_age']))
data['policy_holder_age'] = data['policy_holder_age'].fillna(np.mean(data['policy_holder_age']))
data['third_party_1_age'] = data['third_party_1_age'].fillna(np.mean(data['third_party_1_age']))
data['third_party_2_age'] = data['third_party_2_age'].fillna(np.mean(data['third_party_2_age']))
data['third_party_3_age'] = data['third_party_3_age'].fillna(np.mean(data['third_party_3_age']))
data['repair_age'] = data['repair_age'].fillna(np.mean(data['repair_age']))
data.claim_hour =  data.claim_hour.apply(lambda row : float(row))
data['claim_hour'] = data['claim_hour'].fillna(np.mean(data['claim_hour']))
data['delta_claim_occured_vehicle_inuse'] = data['delta_claim_occured_vehicle_inuse'].fillna(np.mean(data['delta_claim_occured_vehicle_inuse']))
data['delta_claim_registered_policy_date_start'] = data['delta_claim_registered_policy_date_start'].fillna(np.mean(data['delta_claim_registered_policy_date_start']))
data['delta_claim_registered_policy_last_renewed'] = data['delta_claim_registered_policy_last_renewed'].fillna(np.mean(data['delta_claim_registered_policy_last_renewed']))
data['delta_policy_next_expiry_claim_registered'] = data['delta_policy_next_expiry_claim_registered'].fillna(np.mean(data['delta_policy_next_expiry_claim_registered']))
data['delta_policy_last_renewed_start_date'] = data['delta_policy_last_renewed_start_date'].fillna(np.mean(data['delta_policy_last_renewed_start_date']))
data['delta_policy_next_expiry_last_renewed'] = data['delta_policy_next_expiry_last_renewed'].fillna(np.mean(data['delta_policy_next_expiry_last_renewed']))


#Group categories without fraudulent claims
policy_coverage_type_groups = data.groupby('policy_coverage_type').agg({'fraud' : lambda x: sum(x=='Y')})
data['policy_coverage_type']  = data['policy_coverage_type'].apply(lambda row : 'No fraud' if policy_coverage_type_groups.loc[row,'fraud'] == 0
                                                                   else row)
claim_vehicle_brand_groups = data.groupby('claim_vehicle_brand').agg({'fraud' : lambda x: sum(x=='Y')})
data['claim_vehicle_brand']  = data['claim_vehicle_brand'].apply(lambda row : 'No fraud' if claim_vehicle_brand_groups.loc[row,'fraud'] == 0
                                                                   else row)                                                                   


#Drop unnecessary columns 
data.drop(columns=columns_to_drop).to_csv('processed_data.csv', index = False)

