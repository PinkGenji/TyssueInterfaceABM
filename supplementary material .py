# -*- coding: utf-8 -*-
"""
Spyder Editor

This file contains the supplementary material for the tyssue learning code. 
"""
import pandas as pd


'''
The following is the supp material for 00basics.py file
'''
# Creating the DataFrame
df = pd.DataFrame({'Weight': [45, 88, 56, 15, 71],
                   'Name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
                   'Age': [14, 25, 55, 8, 21]})

index_ = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5']
 
# Set the index
df.index = index_
 
# Print the DataFrame
print("Original DataFrame:")
print(df)
# Corrected selection using loc for a specific cell
result = df.loc['Row_2', 'Name']
print(result)

result = df.loc[:, ['Name', 'Age']]
print(result)