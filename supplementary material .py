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

'''
The following is about python class/objects:
'''
# To create a class, we use the keyword class:
class Person:
# The function __init__() is called automatically every time the class is being used to create a new object.
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person('John',36)

print(p1.name)
print(p1.age)
print(p1)  # The result of this line is to be compared with the following part.

# The __str__() function controls what should be returned when the class object
# is represented as a string.
# If the __str__() function is not set, the string representation of the object is returned:
class Person:
# The function __init__() is called automatically every time the class is being
# used to create a new object.
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def __str__(self):
        return f"{self.name}({self.age})"

p1 = Person('John',36)

print(p1)

# Object methods.
# Objects can also contain methods. Methods in objects are functions that belogn to the object.
class Person:
# The function __init__() is called automatically every time the class is being used to create a new object.
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def myfunc(self):
        print("Hello my name is " + self.name)

p1 = Person("John", 36)
p1.myfunc()

# The self parameter is a reference to the current instance of the class.
# It is used to access variables that belongs to the class.
# It does not have to be named self,
# but it has to be the first parameter of any function in the class:
class Person:
    def __init__(mysillyobject, name, age):
        mysillyobject.name = name
        mysillyobject.age = age
        
    def myfunc(abc):
        print("Hello my name is " + abc.name)

p1 = Person("John", 36)
p1.myfunc()

# We can modify properties on obejcts like this:
p1.age = 40
print(p1.age)

# We can delete properties on object by the del keyword:
del p1.age
print(p1.age) # An error would occur

# del can be used to delete the object:
del p1

# class definitions cannot be empty, but for some reason, we may have a class 
# definition with no content, put in the pass statement to avoid getting an error.
class Person:
    pass







