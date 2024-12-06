import pkg_resources 
import pandas as pd

def load_hsko_data():
   data_path = pkg_resources.resource_filename('pygsva', 'data/hsko.csv')
   return pd.read_csv(data_path,index_col=0)

def load_pbmc_data():
   data_path = pkg_resources.resource_filename('pygsva', 'data/pbmc_exp.csv')
   return pd.read_csv(data_path,index_col=0)