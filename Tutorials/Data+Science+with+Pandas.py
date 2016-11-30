
# coding: utf-8

# In[116]:

import pandas as pd
import numpy as np


# In[117]:

#A series is a basic array type in Pandas
#It can be used to associate values. Here, numbers with letters.
test_series = pd.Series([.1, .2, 3, .4, 5], index = ['a', 'b', 'c', 'd', 'e'])
test_series


# In[118]:

#Data Frames are like matrix-like structures, built from series
#Here we are going to place test_series into test_df
#The Y-axis here is the index from the series, and we can name the X-axis in a similar way 

test_df = pd.DataFrame(test_series, columns = ["Column 0"])
test_df


# In[119]:

#You can then interact with the columns pretty intuitively
#Call by Column
test_df['Column 0']


# In[120]:

#Add a column
test_df['Column 1'] = test_df['Column 0'] ** 2
test_df['Column 1']


# In[121]:

#Sort by a chosen column
test_df = test_df.sort_values(by = "Column 1", ascending = False)
test_df


# In[122]:

#Sort by chosen parameter
test_df = test_df[test_df['Column 1'] < 1]
test_df


# In[123]:

#In addition to matrix operations, Pandas
#has a number of functions which are useful
#for data science. 
test_df.describe()


# In[ ]:

#It also has tools for handling external files,
#such as files to load CSVs naturally, inspect
#only segments of a DataFrame, and list by 
#common conventions such as dd/mm/yyyy.

