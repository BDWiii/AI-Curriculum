# Python Essentials Cheatsheet

> [!TIP]
> This cheatsheet covers the most commonly used Python operations and methods. Bookmark this page for quick reference!

---

## 📋 Table of Contents
1. [Lists](#lists)
2. [Dictionaries](#dictionaries)
3. [Sets](#sets)
4. [Tuples](#tuples)
5. [String Operations](#string-operations)
6. [Type Conversions](#type-conversions)
7. [File I/O](#file-io)
8. [OS & System Operations](#os--system-operations)
9. [Useful Built-in Functions](#useful-built-in-functions)
10. [List Comprehensions & Generators](#list-comprehensions--generators)

---

## Lists

### Creating Lists
```python
# Empty list
my_list = []
my_list = list()

# With values
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Using range
nums = list(range(5))  # [0, 1, 2, 3, 4]
nums = list(range(1, 10, 2))  # [1, 3, 5, 7, 9]
```

### Common List Operations
```python
# Adding elements
lst = [1, 2, 3]
lst.append(4)              # [1, 2, 3, 4] - add to end
lst.insert(0, 0)           # [0, 1, 2, 3, 4] - insert at index
lst.extend([5, 6])         # [0, 1, 2, 3, 4, 5, 6] - add multiple

# Removing elements
lst.pop()                  # Removes and returns last element
lst.pop(0)                 # Removes and returns element at index 0
lst.remove(3)              # Removes first occurrence of value 3
lst.clear()                # Removes all elements

# Accessing elements
first = lst[0]             # First element
last = lst[-1]             # Last element
middle = lst[1:4]          # Slice [index 1 to 3]
reversed_lst = lst[::-1]   # Reverse the list

# Useful methods
lst.sort()                 # Sort in place
sorted_lst = sorted(lst)   # Return sorted copy
lst.reverse()              # Reverse in place
count = lst.count(5)       # Count occurrences of 5
index = lst.index(5)       # Find index of first occurrence of 5
length = len(lst)          # Get length

# Checking membership
if 5 in lst:
    print("5 is in the list")
```

### List Tricks
```python
# Unpacking
a, b, c = [1, 2, 3]
first, *rest = [1, 2, 3, 4, 5]  # first=1, rest=[2,3,4,5]
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2,3,4], last=5

# Copying lists
shallow_copy = lst.copy()
shallow_copy = lst[:]
import copy
deep_copy = copy.deepcopy(lst)

# Joining lists
combined = list1 + list2
list1 += list2  # In-place concatenation
```

---

## Dictionaries

### Creating Dictionaries
```python
# Empty dictionary
my_dict = {}
my_dict = dict()

# With values
person = {"name": "Alice", "age": 25, "city": "NYC"}

# Using dict()
person = dict(name="Alice", age=25, city="NYC")

# From lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
my_dict = dict(zip(keys, values))  # {'a': 1, 'b': 2, 'c': 3}

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Common Dictionary Operations
```python
# Accessing values
name = person["name"]              # Raises KeyError if not found
name = person.get("name")          # Returns None if not found
name = person.get("name", "Unknown")  # Returns default if not found

# Adding/Updating
person["email"] = "alice@email.com"  # Add new key-value
person.update({"age": 26, "phone": "123-456"})  # Update multiple

# Removing
del person["city"]                 # Remove key
age = person.pop("age")            # Remove and return value
person.popitem()                   # Remove and return last item (3.7+)
person.clear()                     # Remove all items

# Useful methods
keys = person.keys()               # Get all keys
values = person.values()           # Get all values
items = person.items()             # Get key-value pairs

# Checking membership
if "name" in person:
    print("Name exists")

# Iterating
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# Merging dictionaries (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = dict1 | dict2  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Merging (Python 3.5+)
merged = {**dict1, **dict2}
```

### Dictionary Tricks
```python
# Default values with setdefault
count = {}
count.setdefault("apple", 0)
count["apple"] += 1

# Using defaultdict
from collections import defaultdict
count = defaultdict(int)  # Default value is 0
count["apple"] += 1

# Get nested values safely
data = {"user": {"name": "Alice"}}
name = data.get("user", {}).get("name", "Unknown")
```

---

## Sets

### Creating Sets
```python
# Empty set (note: {} creates empty dict, not set!)
my_set = set()

# With values
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 3, 4, 5])

# From string (unique characters)
chars = set("hello")  # {'h', 'e', 'l', 'o'}

# Set comprehension
squares = {x**2 for x in range(5)}  # {0, 1, 4, 9, 16}
```

### Common Set Operations
```python
# Adding elements
fruits.add("grape")
fruits.update(["mango", "kiwi"])  # Add multiple

# Removing elements
fruits.remove("apple")      # Raises KeyError if not found
fruits.discard("apple")     # No error if not found
item = fruits.pop()         # Remove and return arbitrary element
fruits.clear()              # Remove all elements

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2              # {1, 2, 3, 4, 5, 6}
union = set1.union(set2)

intersection = set1 & set2       # {3, 4}
intersection = set1.intersection(set2)

difference = set1 - set2         # {1, 2}
difference = set1.difference(set2)

symmetric_diff = set1 ^ set2     # {1, 2, 5, 6}
symmetric_diff = set1.symmetric_difference(set2)

# Checking relationships
is_subset = set1 <= set2         # Is set1 subset of set2?
is_superset = set1 >= set2       # Is set1 superset of set2?
is_disjoint = set1.isdisjoint(set2)  # No common elements?

# Membership
if "apple" in fruits:
    print("Apple is in the set")
```

### Set Tricks
```python
# Remove duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(numbers))  # [1, 2, 3, 4] (order not guaranteed)

# Frozen set (immutable)
frozen = frozenset([1, 2, 3])  # Can be used as dict key
```

---

## Tuples

### Creating Tuples
```python
# Empty tuple
my_tuple = ()
my_tuple = tuple()

# With values
point = (3, 4)
person = ("Alice", 25, "NYC")

# Single element tuple (note the comma!)
single = (5,)
single = 5,

# From list
my_tuple = tuple([1, 2, 3])
```

### Common Tuple Operations
```python
# Accessing elements (same as lists)
first = point[0]
last = point[-1]
slice_tuple = person[0:2]

# Unpacking
x, y = point
name, age, city = person

# Useful methods
count = person.count("Alice")  # Count occurrences
index = person.index(25)       # Find index
length = len(person)           # Get length

# Concatenation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2     # (1, 2, 3, 4, 5, 6)

# Repetition
repeated = (1, 2) * 3          # (1, 2, 1, 2, 1, 2)

# Membership
if "Alice" in person:
    print("Alice is in the tuple")
```

### Tuple Tricks
```python
# Named tuples (more readable)
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4

# Swapping values
a, b = 1, 2
a, b = b, a  # Now a=2, b=1

# Multiple return values
def get_stats():
    return 10, 20, 30  # Returns a tuple

min_val, max_val, avg = get_stats()
```

---

## String Operations

### Common String Methods
```python
text = "  Hello, World!  "

# Case conversion
text.upper()           # "  HELLO, WORLD!  "
text.lower()           # "  hello, world!  "
text.capitalize()      # "  hello, world!  "
text.title()           # "  Hello, World!  "
text.swapcase()        # "  hELLO, wORLD!  "

# Whitespace
text.strip()           # "Hello, World!" - remove leading/trailing
text.lstrip()          # "Hello, World!  " - remove leading
text.rstrip()          # "  Hello, World!" - remove trailing

# Searching
text.find("World")     # Returns index or -1
text.index("World")    # Returns index or raises ValueError
text.count("l")        # Count occurrences
text.startswith("  H") # True
text.endswith("!  ")   # True

# Replacing
text.replace("World", "Python")  # "  Hello, Python!  "

# Splitting and joining
words = "a,b,c".split(",")       # ['a', 'b', 'c']
joined = "-".join(words)         # "a-b-c"
lines = "line1\nline2".splitlines()  # ['line1', 'line2']

# Checking content
"123".isdigit()        # True
"abc".isalpha()        # True
"abc123".isalnum()     # True
"   ".isspace()        # True

# Formatting
name = "Alice"
age = 25
f"My name is {name} and I'm {age}"  # f-strings (Python 3.6+)
"My name is {} and I'm {}".format(name, age)
"My name is {0} and I'm {1}".format(name, age)
```

### String Tricks
```python
# Multiline strings
text = """
This is a
multiline string
"""

# Raw strings (ignore escape sequences)
path = r"C:\Users\name\file.txt"

# String multiplication
separator = "-" * 50  # "--------------------------------------------------"

# Reversing a string
reversed_text = text[::-1]

# Checking palindrome
word = "racecar"
is_palindrome = word == word[::-1]
```

---

## Type Conversions

### Converting Between Types
```python
# To list
list("hello")          # ['h', 'e', 'l', 'l', 'o']
list((1, 2, 3))        # [1, 2, 3]
list({1, 2, 3})        # [1, 2, 3]
list(range(5))         # [0, 1, 2, 3, 4]

# To tuple
tuple([1, 2, 3])       # (1, 2, 3)
tuple("hello")         # ('h', 'e', 'l', 'l', 'o')

# To set
set([1, 2, 2, 3])      # {1, 2, 3}
set("hello")           # {'h', 'e', 'l', 'o'}

# To dictionary
dict([("a", 1), ("b", 2)])  # {'a': 1, 'b': 2}
dict(zip(["a", "b"], [1, 2]))  # {'a': 1, 'b': 2}

# Number conversions
int("42")              # 42
int(3.14)              # 3
int("1010", 2)         # 10 (binary to int)
float("3.14")          # 3.14
float(42)              # 42.0

# To string
str(42)                # "42"
str([1, 2, 3])         # "[1, 2, 3]"
",".join(map(str, [1, 2, 3]))  # "1,2,3"

# To boolean
bool(0)                # False
bool(1)                # True
bool([])               # False
bool([1])              # True
bool("")               # False
bool("text")           # True
```

---

## File I/O

### Reading Files
```python
# Read entire file
with open("file.txt", "r") as f:
    content = f.read()  # Returns entire file as string

# Read lines as list
with open("file.txt", "r") as f:
    lines = f.readlines()  # Returns list of lines (with \n)

# Read line by line (memory efficient)
with open("file.txt", "r") as f:
    for line in f:
        print(line.strip())  # Process each line

# Read specific number of characters
with open("file.txt", "r") as f:
    chunk = f.read(100)  # Read first 100 characters
```

### Writing Files
```python
# Write (overwrites existing file)
with open("file.txt", "w") as f:
    f.write("Hello, World!\n")
    f.writelines(["Line 1\n", "Line 2\n"])

# Append (adds to existing file)
with open("file.txt", "a") as f:
    f.write("New line\n")

# Write list to file
data = ["apple", "banana", "orange"]
with open("fruits.txt", "w") as f:
    f.write("\n".join(data))
```

### File Operations
```python
import os

# Check if file exists
if os.path.exists("file.txt"):
    print("File exists")

# Check if it's a file or directory
os.path.isfile("file.txt")
os.path.isdir("folder")

# Get file size
size = os.path.getsize("file.txt")

# Delete file
os.remove("file.txt")

# Rename file
os.rename("old.txt", "new.txt")

# Get file info
import os.path
os.path.basename("/path/to/file.txt")  # "file.txt"
os.path.dirname("/path/to/file.txt")   # "/path/to"
os.path.splitext("file.txt")           # ("file", ".txt")
```

---

## OS & System Operations

### Working with Paths
```python
import os

# Current working directory
cwd = os.getcwd()

# Change directory
os.chdir("/path/to/directory")

# Join paths (OS-independent)
path = os.path.join("folder", "subfolder", "file.txt")

# Absolute path
abs_path = os.path.abspath("file.txt")

# List directory contents
files = os.listdir(".")
files = os.listdir("/path/to/directory")

# Create directory
os.mkdir("new_folder")
os.makedirs("parent/child/grandchild")  # Create nested directories

# Remove directory
os.rmdir("folder")  # Only if empty
import shutil
shutil.rmtree("folder")  # Remove directory and contents
```

### System Operations
```python
import sys

# Command line arguments
script_name = sys.argv[0]
arguments = sys.argv[1:]

# Exit program
sys.exit()
sys.exit(1)  # Exit with error code

# Python version
print(sys.version)

# Platform
print(sys.platform)  # 'linux', 'darwin', 'win32', etc.

# Module search paths
print(sys.path)
```

### Environment Variables
```python
import os

# Get environment variable
home = os.getenv("HOME")
home = os.environ.get("HOME")
path = os.environ["PATH"]  # Raises KeyError if not found

# Set environment variable
os.environ["MY_VAR"] = "value"

# Get all environment variables
all_vars = os.environ
```

---

## Useful Built-in Functions

### Iteration & Aggregation
```python
# enumerate - get index and value
fruits = ["apple", "banana", "orange"]
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

for i, fruit in enumerate(fruits, start=1):  # Start from 1
    print(f"{i}: {fruit}")

# zip - combine iterables
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# map - apply function to all elements
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# filter - keep elements that match condition
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# reduce - combine elements
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)  # 15

# sum, min, max
total = sum(numbers)
minimum = min(numbers)
maximum = max(numbers)

# all, any
all([True, True, True])   # True
all([True, False, True])  # False
any([False, False, True]) # True
any([False, False])       # False
```

### Sorting
```python
# sorted - returns new sorted list
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_nums = sorted(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]
sorted_desc = sorted(numbers, reverse=True)

# Sort by custom key
words = ["apple", "pie", "zoo", "a"]
sorted_words = sorted(words, key=len)  # ['a', 'pie', 'zoo', 'apple']

# Sort tuples/lists by specific element
students = [("Alice", 25), ("Bob", 20), ("Charlie", 30)]
sorted_by_age = sorted(students, key=lambda x: x[1])

# Sort dictionary by value
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
```

### Other Useful Functions
```python
# reversed - reverse iterator
numbers = [1, 2, 3, 4, 5]
for num in reversed(numbers):
    print(num)

# abs - absolute value
abs(-5)  # 5

# round - round to n decimal places
round(3.14159, 2)  # 3.14

# pow - power
pow(2, 3)  # 8
pow(2, 3, 5)  # 2^3 % 5 = 3

# divmod - quotient and remainder
quotient, remainder = divmod(17, 5)  # (3, 2)

# isinstance - check type
isinstance(5, int)  # True
isinstance("hello", str)  # True
isinstance([1, 2], (list, tuple))  # True

# type - get type
type(5)  # <class 'int'>
```

---

## List Comprehensions & Generators

### List Comprehensions
```python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]

# With condition: [expression for item in iterable if condition]
evens = [x for x in range(10) if x % 2 == 0]

# With if-else: [expr1 if condition else expr2 for item in iterable]
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]  # [1,2,3,4,5,6]
```

### Dictionary & Set Comprehensions
```python
# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}  # {1:'a', 2:'b', 3:'c'}

# Set comprehension
unique_lengths = {len(word) for word in ["apple", "pie", "zoo"]}  # {3, 5}
```

### Generator Expressions
```python
# Generator (memory efficient, use parentheses)
gen = (x**2 for x in range(1000000))  # Doesn't compute until needed

# Use in functions
sum_squares = sum(x**2 for x in range(100))

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num)
```

---

## Quick Tips & Best Practices

> [!IMPORTANT]
> **Memory Efficiency**: Use generators for large datasets instead of lists
> ```python
> # Memory intensive
> big_list = [x**2 for x in range(1000000)]
> 
> # Memory efficient
> big_gen = (x**2 for x in range(1000000))
> ```

> [!TIP]
> **Use `in` for membership testing**: Sets are much faster than lists for checking membership
> ```python
> # Slow for large lists
> if item in my_list:  # O(n)
>     pass
> 
> # Fast
> if item in my_set:  # O(1)
>     pass
> ```

> [!NOTE]
> **Unpacking with `*` operator**:
> ```python
> first, *middle, last = [1, 2, 3, 4, 5]
> # first=1, middle=[2,3,4], last=5
> 
> # Merge lists
> combined = [*list1, *list2, *list3]
> 
> # Merge dicts
> merged = {**dict1, **dict2}
> ```

> [!TIP]
> **Use `get()` for dictionaries**: Avoid KeyError exceptions
> ```python
> # Risky
> value = my_dict["key"]  # KeyError if key doesn't exist
> 
> # Safe
> value = my_dict.get("key", default_value)
> ```

> [!WARNING]
> **Mutable default arguments**: Don't use mutable objects as default arguments
> ```python
> # BAD - list is shared across calls!
> def add_item(item, lst=[]):
>     lst.append(item)
>     return lst
> 
> # GOOD
> def add_item(item, lst=None):
>     if lst is None:
>         lst = []
>     lst.append(item)
>     return lst
> ```

---

## Common Patterns

### Reading CSV-like data
```python
# Parse CSV manually
with open("data.csv", "r") as f:
    for line in f:
        fields = line.strip().split(",")
        print(fields)

# Using csv module (better)
import csv
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

### Counting occurrences
```python
# Manual counting
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
count = {}
for word in words:
    count[word] = count.get(word, 0) + 1

# Using Counter (better)
from collections import Counter
count = Counter(words)
print(count.most_common(2))  # [('apple', 3), ('banana', 2)]
```

### Working with JSON
```python
import json

# Parse JSON string
json_str = '{"name": "Alice", "age": 25}'
data = json.loads(json_str)

# Convert to JSON string
data = {"name": "Alice", "age": 25}
json_str = json.dumps(data, indent=2)

# Read from file
with open("data.json", "r") as f:
    data = json.load(f)

# Write to file
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## Practice Exercises

### Exercise 1: Data Structure Selection
**Question**: You need to store unique user IDs and check if a user exists frequently. Which data structure should you use?

<details>
<summary>Answer</summary>

Use a **set** because:
- Sets automatically handle uniqueness
- Membership testing is O(1) - very fast
- Perfect for "does this exist?" queries

```python
user_ids = set()
user_ids.add(12345)
if 12345 in user_ids:  # Fast lookup
    print("User exists")
```
</details>

### Exercise 2: List vs Generator
**Question**: You need to process 1 million numbers. When should you use a generator instead of a list?

<details>
<summary>Answer</summary>

Use a **generator** when:
- You only need to iterate once
- Memory is a concern
- You don't need random access

```python
# Generator - memory efficient
def process_numbers():
    for i in range(1000000):
        yield i ** 2

# List - uses more memory but allows random access
numbers = [i ** 2 for i in range(1000000)]
```
</details>

### Exercise 3: Dictionary Methods
**Question**: Write code to merge two dictionaries where values from the second dictionary should override the first.

<details>
<summary>Answer</summary>

```python
dict1 = {"a": 1, "b": 2, "c": 3}
dict2 = {"b": 20, "d": 4}

# Method 1: Using update (modifies dict1)
dict1.update(dict2)

# Method 2: Using unpacking (creates new dict)
merged = {**dict1, **dict2}

# Method 3: Using | operator (Python 3.9+)
merged = dict1 | dict2

# Result: {"a": 1, "b": 20, "c": 3, "d": 4}
```
</details>

---

> [!TIP]
> **Keep this cheatsheet handy!** The more you practice these patterns, the more natural they'll become. Focus on understanding *when* to use each data structure and operation, not just *how*.
