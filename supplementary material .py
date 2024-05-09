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

'''
*args and **kwargs in Python:

These two symbols are two special symbols used for passing arguments in Python:
    *args are non-keyword arguments
    **kwargs are keyword arguments

The speical syntax *args in function definitions in Python is used to pass 
a variable number of arguments to a function. It is used to pass a non-keyworded, variable length arugment list.

The syntax is to use the symbol* to take in a variable numebr of arugments, by convention, we use *args
What *args allows you to do is take in more arugments than the number of formal arguments that you defined before.
For example, a multiply function that takes any number of arguments and is able to multiply them all, then *args can be used.
The variable associated with * becoems iterable.
'''

#Example:

def myFun(*argv):
    for arg in argv:
        print(arg)

myFun('Hello', 'Welcome', 'to', 'GeekforGeeks')

# We can also have *args with a first extra argument
def myFun(arg1, *argv):
    print("First argument: ", arg1)
    for arg in argv:
        print("Next argument through *argv: ", arg)
myFun('Hello', 'Welcome', 'to', 'GeekforGeeks')

# For **kwargs, it is used to pass a keyworded, variable-length argument list.
# The double star ** allows us to pass through keyword arguments (and any number of them).

def myFun(**kwargs):
    for key, value in kwargs.items():
        print("%s == %s" %(key,value))

myFun(first = 'Geeks', mid = 'for', last = 'Geeks')

# Both *args and **kwargs can be used at the same time.
def myFun(arg1, arg2, arg3):
	print("arg1:", arg1)
	print("arg2:", arg2)
	print("arg3:", arg3)


# Now we can use *args or **kwargs to
# pass arguments to this function :
args = ("Geeks", "for", "Geeks")
myFun(*args)

kwargs = {"arg1": "Geeks", "arg2": "for", "arg3": "Geeks"}
myFun(**kwargs)

# *args receives arguments as a tuple
# **kwargs received arguments as a dictionary

# defining car class
class car():
	# args receives unlimited no. of arguments as an array
	def __init__(self, *args):
		# access args index like array does
		self.speed = args[0]
		self.color = args[1]


# creating objects of car class
audi = car(200, 'red')
bmw = car(250, 'black')
mb = car(190, 'white')

# printing the color and speed of the cars
print(audi.color)
print(bmw.speed)

# defining car class
class car():
	# args receives unlimited no. of arguments as an array
	def __init__(self, **kwargs):
		# access args index like array does
		self.speed = kwargs['s']
		self.color = kwargs['c']


# creating objects of car class
audi = car(s=200, c='red')
bmw = car(s=250, c='black')
mb = car(s=190, c='white')

# printing the color and speed of cars
print(audi.color)
print(bmw.speed)


'''
First-class functions:

(wikipedia) A programming language is said to have First-class functions when
functions in that language are treated like any other variable. For example,
in such a language, a function can be passed as an argument to other functions,
can be returned by another function and can be assigned as a value to a variable.

'''
# A trivial example:
def square(x):
    return x*x

f = square    # Without the brackets, we are not executing the function

print(square)   # This prints the function object at the memory address.
print(f(5))     # This prints the function evaluated at arg=5

# Advanced example:
def my_map(func, arg_list):
    result = []
    for i in arg_list:
        result.append(func(i))
    return result

squares = my_map(square, [1,2,3,4,5])  # square is being passed as an argument.

print(squares)

# More andvanced example:
def logger(msg):
    
    def log_message():  # Here the inside function is used for modulation.
        print('Log:', msg)
    
    return log_message

log_hi = logger('Hi!')
log_hi()

'''
Closure:
A closure is a persistent local variable scope, which holds on to local variables
even after the code execution has moved out of that block.

In python, closure is a nested function that helps us access the outer 
function's variables even after the outer function is closed or removed.
'''

def outer_func(msg):
    message = msg
    
    def inner_func():
        print(message)
# message was not created inside inner_func, but inner_func still has access
# to it. This (message) is what we call a free variable.
    return inner_func

hi_func = outer_func('Hi')
hello_func = outer_func('Hello')

del outer_func # Now delete the outer function

hi_func()
hello_func()
# We can see that each function remembers its msg value.
# A closure closes/remembers the free variables in their environment.


'''
Decorators:

(wikipedia) A decorator is a design pattern in Python that allows a user to add
new functionality to an existing object without modifying its structure.
Decorators are typically applied to functions, and they play a crucial role in
enhancing or modifying the behaviour of functions.
    
'''

# We start with a modified version of closure example.
def decorator_function(original_function):
    def wrapper_function(*args, **kwargs):
       #print('wrapper executed this before {}'.format(original_function.__name__))
       # un-comment the above line tells you the execution order of decorator. 
       return original_function(*args, **kwargs)
    return wrapper_function

def display():
    print('display function ran')

decorated_display = decorator_function(display)

decorated_display()

@decorator_function    
# put the decorator on top of a function, is equal to the function being passed in:
# this decorator is equivalent to: display = decorator_function(display)
def display():
    print('display function ran')


@decorator_function
def display_info(name, age):
    print('display_info ran with arguments ({},{})'.format(name,age))

display_info('John', 25)

display()


# More self-contained example.
def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level = logging.INFO)
    
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper

@my_logger
def display_info(name,age):
    print('display_info ran with arguments ({}, {})'.format(name, age))

display_info('Hank', 30)
display_info('John', 40)
# Two log files being created in the directory.
# Comparing with implementing this functionality to multiple functions, using
# decorator is much simpler and less error-prone.


'''
static method:

A static method in Python is a method that belongs to a class, not its instance.
It does not require an instance of the class to be called, nor does it have 
access to an instance.

'''







'''
This is the end of the file.
'''
