# Python Programming - Complete Problem Set

A comprehensive collection of Python problems with solutions, organized by topic and difficulty level.

## Table of Contents
1. [Data Types](#data-types)
2. [Data Structures](#data-structures)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [Object-Oriented Programming](#object-oriented-programming)
6. [Advanced Topics](#advanced-topics)

---

## Data Types

### Level 1: Basic Operations

**Problem 1.1: Type Conversion**
```python
# Convert the string "42" to an integer, then to a float, then back to a string
# Print the type at each step
```

<details>
<summary>Solution</summary>

```python
s = "42"
print(f"Original: {s}, Type: {type(s)}")

i = int(s)
print(f"Integer: {i}, Type: {type(i)}")

f = float(i)
print(f"Float: {f}, Type: {type(f)}")

s2 = str(f)
print(f"String: {s2}, Type: {type(s2)}")
```
</details>

**Problem 1.2: String Manipulation**
```python
# Given: text = "  Python Programming  "
# Remove whitespace, convert to uppercase, and replace "PYTHON" with "ADVANCED PYTHON"
```

<details>
<summary>Solution</summary>

```python
text = "  Python Programming  "
result = text.strip().upper().replace("PYTHON", "ADVANCED PYTHON")
print(result)  # "ADVANCED PYTHON PROGRAMMING"
```
</details>

**Problem 1.3: Numeric Operations**
```python
# Calculate: (15 + 7) * 3 / 2 - 5 ** 2
# Also find: 17 // 3, 17 % 3, and 2 ** 10
```

<details>
<summary>Solution</summary>

```python
result1 = (15 + 7) * 3 / 2 - 5 ** 2
print(f"Result: {result1}")  # 8.0

quotient = 17 // 3
remainder = 17 % 3
power = 2 ** 10

print(f"17 // 3 = {quotient}")  # 5
print(f"17 % 3 = {remainder}")   # 2
print(f"2 ** 10 = {power}")      # 1024
```
</details>

### Level 2: String Methods

**Problem 1.4: String Analysis**
```python
# Given: sentence = "The quick brown fox jumps over the lazy dog"
# Count vowels, consonants, and spaces
```

<details>
<summary>Solution</summary>

```python
sentence = "The quick brown fox jumps over the lazy dog"
vowels = "aeiouAEIOU"

vowel_count = sum(1 for char in sentence if char in vowels)
space_count = sentence.count(' ')
consonant_count = sum(1 for char in sentence if char.isalpha() and char not in vowels)

print(f"Vowels: {vowel_count}")
print(f"Consonants: {consonant_count}")
print(f"Spaces: {space_count}")
```
</details>

**Problem 1.5: String Formatting**
```python
# Create a formatted string with name="Alice", age=25, score=95.5
# Use f-strings, .format(), and % formatting
```

<details>
<summary>Solution</summary>

```python
name, age, score = "Alice", 25, 95.5

# f-string
s1 = f"{name} is {age} years old with a score of {score:.1f}"

# .format()
s2 = "{} is {} years old with a score of {:.1f}".format(name, age, score)

# % formatting
s3 = "%s is %d years old with a score of %.1f" % (name, age, score)

print(s1)
print(s2)
print(s3)
```
</details>

---

## Data Structures

### Lists

**Problem 2.1: List Basics**
```python
# Create a list of numbers 1-10, then:
# - Add 11 to the end
# - Insert 0 at the beginning
# - Remove the number 5
# - Print the length and sum
```

<details>
<summary>Solution</summary>

```python
numbers = list(range(1, 11))
print(f"Original: {numbers}")

numbers.append(11)
print(f"After append: {numbers}")

numbers.insert(0, 0)
print(f"After insert: {numbers}")

numbers.remove(5)
print(f"After remove: {numbers}")

print(f"Length: {len(numbers)}, Sum: {sum(numbers)}")
```
</details>

**Problem 2.2: List Slicing**
```python
# Given: data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Extract: first 3, last 3, every other element, reversed list
```

<details>
<summary>Solution</summary>

```python
data = [10, 20, 30, 40, 50, 60, 70, 80, 90]

first_three = data[:3]
last_three = data[-3:]
every_other = data[::2]
reversed_list = data[::-1]

print(f"First 3: {first_three}")
print(f"Last 3: {last_three}")
print(f"Every other: {every_other}")
print(f"Reversed: {reversed_list}")
```
</details>

**Problem 2.3: List Comprehension**
```python
# Create a list of squares of even numbers from 1 to 20
```

<details>
<summary>Solution</summary>

```python
squares = [x**2 for x in range(1, 21) if x % 2 == 0]
print(squares)  # [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]
```
</details>

**Problem 2.4: Nested Lists**
```python
# Create a 3x3 matrix and calculate the sum of each row
```

<details>
<summary>Solution</summary>

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

row_sums = [sum(row) for row in matrix]
print(f"Row sums: {row_sums}")  # [6, 15, 24]

# Alternative
for i, row in enumerate(matrix):
    print(f"Row {i}: sum = {sum(row)}")
```
</details>

### Tuples

**Problem 2.5: Tuple Unpacking**
```python
# Create a tuple with (name, age, city) and unpack it
# Swap two variables using tuple unpacking
```

<details>
<summary>Solution</summary>

```python
person = ("Bob", 30, "New York")
name, age, city = person
print(f"{name} is {age} years old and lives in {city}")

# Swap
a, b = 5, 10
print(f"Before: a={a}, b={b}")
a, b = b, a
print(f"After: a={a}, b={b}")
```
</details>

**Problem 2.6: Tuple Operations**
```python
# Given two tuples, concatenate them and find the index of a specific element
```

<details>
<summary>Solution</summary>

```python
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

combined = tuple1 + tuple2
print(f"Combined: {combined}")

index_of_5 = combined.index(5)
print(f"Index of 5: {index_of_5}")

count_of_3 = combined.count(3)
print(f"Count of 3: {count_of_3}")
```
</details>

### Sets

**Problem 2.7: Set Operations**
```python
# Create two sets and perform union, intersection, difference, and symmetric difference
```

<details>
<summary>Solution</summary>

```python
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

union = set_a | set_b
intersection = set_a & set_b
difference = set_a - set_b
sym_diff = set_a ^ set_b

print(f"Union: {union}")
print(f"Intersection: {intersection}")
print(f"Difference (A-B): {difference}")
print(f"Symmetric Difference: {sym_diff}")
```
</details>

**Problem 2.8: Remove Duplicates**
```python
# Remove duplicates from a list while preserving order
```

<details>
<summary>Solution</summary>

```python
numbers = [1, 2, 2, 3, 4, 4, 5, 1, 6]

# Method 1: Using set (doesn't preserve order)
unique_unordered = list(set(numbers))

# Method 2: Preserving order
unique_ordered = []
seen = set()
for num in numbers:
    if num not in seen:
        unique_ordered.append(num)
        seen.add(num)

print(f"Original: {numbers}")
print(f"Unique (unordered): {unique_unordered}")
print(f"Unique (ordered): {unique_ordered}")
```
</details>

### Dictionaries

**Problem 2.9: Dictionary Basics**
```python
# Create a dictionary of student grades, add a new student, update a grade, and delete a student
```

<details>
<summary>Solution</summary>

```python
grades = {"Alice": 85, "Bob": 90, "Charlie": 78}

# Add
grades["David"] = 92

# Update
grades["Bob"] = 95

# Delete
del grades["Charlie"]

print(grades)
```
</details>

**Problem 2.10: Dictionary Methods**
```python
# Iterate through keys, values, and items of a dictionary
```

<details>
<summary>Solution</summary>

```python
student = {"name": "Emma", "age": 22, "major": "CS", "gpa": 3.8}

print("Keys:")
for key in student.keys():
    print(f"  {key}")

print("\nValues:")
for value in student.values():
    print(f"  {value}")

print("\nItems:")
for key, value in student.items():
    print(f"  {key}: {value}")
```
</details>

**Problem 2.11: Dictionary Comprehension**
```python
# Create a dictionary mapping numbers 1-5 to their cubes
```

<details>
<summary>Solution</summary>

```python
cubes = {x: x**3 for x in range(1, 6)}
print(cubes)  # {1: 1, 2: 8, 3: 27, 4: 64, 5: 125}
```
</details>

**Problem 2.12: Nested Dictionaries**
```python
# Create a nested dictionary for a gradebook with multiple students and subjects
```

<details>
<summary>Solution</summary>

```python
gradebook = {
    "Alice": {"Math": 90, "Science": 85, "English": 88},
    "Bob": {"Math": 78, "Science": 92, "English": 80},
    "Charlie": {"Math": 95, "Science": 89, "English": 91}
}

# Calculate average for each student
for student, grades in gradebook.items():
    avg = sum(grades.values()) / len(grades)
    print(f"{student}: Average = {avg:.2f}")
```
</details>

---

## Control Flow

**Problem 3.1: If-Elif-Else**
```python
# Write a function to classify a grade (A: 90+, B: 80-89, C: 70-79, D: 60-69, F: <60)
```

<details>
<summary>Solution</summary>

```python
def classify_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

print(classify_grade(95))  # A
print(classify_grade(82))  # B
print(classify_grade(55))  # F
```
</details>

**Problem 3.2: For Loop**
```python
# Print the multiplication table for 7 (from 1 to 10)
```

<details>
<summary>Solution</summary>

```python
for i in range(1, 11):
    print(f"7 x {i} = {7 * i}")
```
</details>

**Problem 3.3: While Loop**
```python
# Find the first number greater than 1000 in the Fibonacci sequence
```

<details>
<summary>Solution</summary>

```python
a, b = 0, 1
while b <= 1000:
    a, b = b, a + b

print(f"First Fibonacci number > 1000: {b}")
```
</details>

**Problem 3.4: Break and Continue**
```python
# Print numbers 1-20, skip multiples of 3, stop at first number > 15
```

<details>
<summary>Solution</summary>

```python
for num in range(1, 21):
    if num % 3 == 0:
        continue
    if num > 15:
        break
    print(num)
```
</details>

---

## Functions

**Problem 4.1: Basic Function**
```python
# Write a function to calculate factorial of a number
```

<details>
<summary>Solution</summary>

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(factorial(5))  # 120
print(factorial(0))  # 1
```
</details>

**Problem 4.2: Default Arguments**
```python
# Write a function to calculate power with default exponent of 2
```

<details>
<summary>Solution</summary>

```python
def power(base, exponent=2):
    return base ** exponent

print(power(5))      # 25
print(power(5, 3))   # 125
```
</details>

**Problem 4.3: *args and **kwargs**
```python
# Write a function that accepts any number of arguments and keyword arguments
```

<details>
<summary>Solution</summary>

```python
def flexible_function(*args, **kwargs):
    print(f"Positional arguments: {args}")
    print(f"Keyword arguments: {kwargs}")
    
    total = sum(args)
    print(f"Sum of positional args: {total}")
    
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

flexible_function(1, 2, 3, name="Alice", age=25)
```
</details>

**Problem 4.4: Lambda Functions**
```python
# Sort a list of tuples by the second element using lambda
```

<details>
<summary>Solution</summary>

```python
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78), ("David", 95)]

sorted_students = sorted(students, key=lambda x: x[1], reverse=True)
print(sorted_students)
```
</details>

**Problem 4.5: Map, Filter, Reduce**
```python
# Use map to square numbers, filter to get evens, reduce to find product
```

<details>
<summary>Solution</summary>

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map: square all numbers
squared = list(map(lambda x: x**2, numbers))
print(f"Squared: {squared}")

# Filter: get even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Evens: {evens}")

# Reduce: product of all numbers
product = reduce(lambda x, y: x * y, numbers)
print(f"Product: {product}")
```
</details>

---

## Object-Oriented Programming

### Level 1: Classes and Objects

**Problem 5.1: Basic Class**
```python
# Create a Person class with name and age attributes, and a greet method
```

<details>
<summary>Solution</summary>

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name} and I'm {self.age} years old."

person1 = Person("Alice", 25)
print(person1.greet())
```
</details>

**Problem 5.2: Class with Methods**
```python
# Create a BankAccount class with deposit, withdraw, and get_balance methods
```

<details>
<summary>Solution</summary>

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        return f"Current balance: ${self.balance}"

account = BankAccount("Alice", 1000)
print(account.deposit(500))
print(account.withdraw(200))
print(account.get_balance())
```
</details>

### Level 2: Encapsulation

**Problem 5.3: Private Attributes**
```python
# Create a class with private attributes and getter/setter methods
```

<details>
<summary>Solution</summary>

```python
class Student:
    def __init__(self, name, grade):
        self.__name = name
        self.__grade = grade
    
    def get_name(self):
        return self.__name
    
    def get_grade(self):
        return self.__grade
    
    def set_grade(self, grade):
        if 0 <= grade <= 100:
            self.__grade = grade
        else:
            print("Invalid grade")
    
    def __str__(self):
        return f"Student: {self.__name}, Grade: {self.__grade}"

student = Student("Bob", 85)
print(student)
student.set_grade(90)
print(student.get_grade())
```
</details>

**Problem 5.4: Properties**
```python
# Use @property decorator for getters and setters
```

<details>
<summary>Solution</summary>

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(f"{temp.celsius}°C = {temp.fahrenheit}°F")
temp.fahrenheit = 100
print(f"{temp.celsius}°C = {temp.fahrenheit}°F")
```
</details>

### Level 3: Inheritance

**Problem 5.5: Single Inheritance**
```python
# Create Animal base class and Dog, Cat subclasses
```

<details>
<summary>Solution</summary>

```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
    
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color
    
    def make_sound(self):
        return "Meow!"

dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Orange")

print(dog.info())
print(dog.make_sound())
print(cat.info())
print(cat.make_sound())
```
</details>

**Problem 5.6: Method Overriding**
```python
# Create a Shape hierarchy with area calculation
```

<details>
<summary>Solution</summary>

```python
import math

class Shape:
    def area(self):
        raise NotImplementedError("Subclass must implement area()")

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def area(self):
        return 0.5 * self.base * self.height

shapes = [
    Rectangle(5, 10),
    Circle(7),
    Triangle(6, 8)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: Area = {shape.area():.2f}")
```
</details>

### Level 4: Polymorphism

**Problem 5.7: Polymorphism Example**
```python
# Demonstrate polymorphism with different payment methods
```

<details>
<summary>Solution</summary>

```python
class PaymentMethod:
    def pay(self, amount):
        raise NotImplementedError

class CreditCard(PaymentMethod):
    def __init__(self, card_number):
        self.card_number = card_number
    
    def pay(self, amount):
        return f"Paid ${amount} using Credit Card ending in {self.card_number[-4:]}"

class PayPal(PaymentMethod):
    def __init__(self, email):
        self.email = email
    
    def pay(self, amount):
        return f"Paid ${amount} using PayPal account {self.email}"

class Cash(PaymentMethod):
    def pay(self, amount):
        return f"Paid ${amount} in cash"

def process_payment(payment_method, amount):
    print(payment_method.pay(amount))

payments = [
    CreditCard("1234567890123456"),
    PayPal("user@example.com"),
    Cash()
]

for payment in payments:
    process_payment(payment, 100)
```
</details>

### Level 5: Special Methods

**Problem 5.8: Magic Methods**
```python
# Implement __str__, __repr__, __len__, __getitem__
```

<details>
<summary>Solution</summary>

```python
class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []
    
    def add_song(self, song):
        self.songs.append(song)
    
    def __str__(self):
        return f"Playlist: {self.name} ({len(self.songs)} songs)"
    
    def __repr__(self):
        return f"Playlist('{self.name}', {self.songs})"
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, index):
        return self.songs[index]

playlist = Playlist("My Favorites")
playlist.add_song("Song 1")
playlist.add_song("Song 2")
playlist.add_song("Song 3")

print(str(playlist))
print(repr(playlist))
print(f"Length: {len(playlist)}")
print(f"First song: {playlist[0]}")
```
</details>

**Problem 5.9: Operator Overloading**
```python
# Create a Vector class with +, -, * operators
```

<details>
<summary>Solution</summary>

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 3 = {v1 * 3}")
```
</details>

### Level 6: Advanced OOP

**Problem 5.10: Abstract Base Classes**
```python
# Use ABC to create abstract classes
```

<details>
<summary>Solution</summary>

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    @abstractmethod
    def start_engine(self):
        pass
    
    @abstractmethod
    def stop_engine(self):
        pass
    
    def info(self):
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    def start_engine(self):
        return "Car engine started: Vroom!"
    
    def stop_engine(self):
        return "Car engine stopped"

class Motorcycle(Vehicle):
    def start_engine(self):
        return "Motorcycle engine started: Roar!"
    
    def stop_engine(self):
        return "Motorcycle engine stopped"

car = Car("Toyota", "Camry")
bike = Motorcycle("Harley", "Davidson")

print(car.info())
print(car.start_engine())
print(bike.info())
print(bike.start_engine())
```
</details>

**Problem 5.11: Class Methods and Static Methods**
```python
# Demonstrate @classmethod and @staticmethod
```

<details>
<summary>Solution</summary>

```python
class Employee:
    company = "TechCorp"
    num_employees = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.num_employees += 1
    
    @classmethod
    def from_string(cls, emp_str):
        name, salary = emp_str.split('-')
        return cls(name, int(salary))
    
    @classmethod
    def get_company(cls):
        return cls.company
    
    @staticmethod
    def is_workday(day):
        return day not in ['Saturday', 'Sunday']
    
    def __str__(self):
        return f"{self.name}: ${self.salary}"

emp1 = Employee("Alice", 50000)
emp2 = Employee.from_string("Bob-60000")

print(emp1)
print(emp2)
print(f"Company: {Employee.get_company()}")
print(f"Total employees: {Employee.num_employees}")
print(f"Is Monday a workday? {Employee.is_workday('Monday')}")
print(f"Is Saturday a workday? {Employee.is_workday('Saturday')}")
```
</details>

---

## Advanced Topics

**Problem 6.1: List Comprehension with Conditions**
```python
# Create a list of numbers divisible by 3 or 5 from 1-100
```

<details>
<summary>Solution</summary>

```python
numbers = [x for x in range(1, 101) if x % 3 == 0 or x % 5 == 0]
print(f"Count: {len(numbers)}")
print(f"Sum: {sum(numbers)}")
```
</details>

**Problem 6.2: Generators**
```python
# Create a generator for Fibonacci sequence
```

<details>
<summary>Solution</summary>

```python
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Usage
for num in fibonacci_generator(10):
    print(num, end=' ')
print()

# Generator expression
squares_gen = (x**2 for x in range(10))
print(list(squares_gen))
```
</details>

**Problem 6.3: Decorators**
```python
# Create a timing decorator
```

<details>
<summary>Solution</summary>

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"

result = slow_function()
```
</details>

**Problem 6.4: Context Managers**
```python
# Create a custom context manager
```

<details>
<summary>Solution</summary>

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')
```
</details>

**Problem 6.5: Exception Handling**
```python
# Create a robust division function with error handling
```

<details>
<summary>Solution</summary>

```python
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    except TypeError:
        return "Error: Invalid types"
    else:
        return f"Result: {result}"
    finally:
        print("Division operation completed")

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide("10", 2))
```
</details>

---

## Challenge Problems

**Challenge 1: Implement a Stack**
```python
# Create a Stack class with push, pop, peek, is_empty methods
```

<details>
<summary>Solution</summary>

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.peek())  # 3
print(stack.pop())   # 3
print(stack.size())  # 2
```
</details>

**Challenge 2: Word Frequency Counter**
```python
# Count word frequency in a text, return top N most common words
```

<details>
<summary>Solution</summary>

```python
from collections import Counter
import re

def word_frequency(text, n=5):
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count frequencies
    freq = Counter(words)
    
    # Return top N
    return freq.most_common(n)

text = "Python is great. Python is powerful. Python is easy to learn."
print(word_frequency(text, 3))
```
</details>

**Challenge 3: Implement a Simple Cache**
```python
# Create a caching decorator using a dictionary
```

<details>
<summary>Solution</summary>

```python
from functools import wraps

def memoize(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
            print(f"Computing {func.__name__}{args}")
        else:
            print(f"Using cached result for {func.__name__}{args}")
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
print(fibonacci(10))  # Uses cache
```
</details>

---

**End of Problem Set**

This comprehensive problem set covers Python fundamentals through advanced OOP concepts. Practice these problems to build a strong foundation in Python programming!
