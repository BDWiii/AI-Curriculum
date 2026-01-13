# Dunder Methods, Decorators & System Modules

> [!NOTE]
> **What You'll Learn**: This lesson covers Python's special "dunder" (double underscore) methods that give your classes superpowers, decorators for enhancing functions, and essential system modules (OS & SYS) for interacting with your operating system.

---

## Table of Contents
1. [Introduction to Dunder Methods](#introduction-to-dunder-methods)
2. [Essential Dunder Methods](#essential-dunder-methods)
3. [Decorators](#decorators)
4. [OS Module Essentials](#os-module-essentials)
5. [SYS Module Essentials](#sys-module-essentials)
6. [Practice Exercises](#practice-exercises)

---

## Introduction to Dunder Methods

**Dunder methods** (short for "double underscore") are special methods in Python that start and end with double underscores (`__`). They're also called **magic methods** or **special methods**.

### Why Do We Need Them?

Dunder methods allow you to define how objects of your class behave with Python's built-in operations:
- How your object is printed
- How it's compared with other objects
- How it responds to arithmetic operations
- How it behaves in loops
- And much more!

### Example: The Problem

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1 = Point(3, 4)
print(p1)  # Output: <__main__.Point object at 0x7f8b1c0b4a90>
# Not very helpful! 😕
```

### Example: The Solution with Dunder Methods

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(3, 4)
print(p1)  # Output: Point(3, 4)
# Much better! 😊
```

---

## Essential Dunder Methods

### 1. `__init__` - Constructor

The `__init__` method is called when you create a new instance of a class. It initializes the object's attributes.

```python
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
        print(f"Created student: {name}")

# When you create a student, __init__ is automatically called
student = Student("Alice", 20, "A")  # Output: Created student: Alice
```

**How it's called**: Automatically when you instantiate a class using `ClassName(arguments)`

---

### 2. `__str__` vs `__repr__` - String Representation

These two methods control how your object is converted to a string, but they serve different purposes.

#### `__str__` - User-Friendly String

The `__str__` method returns a **user-friendly**, **readable** string representation of the object. It's called by `str()` and `print()`.

```python
class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def __str__(self):
        return f"'{self.title}' by {self.author} ({self.year})"

book = Book("1984", "George Orwell", 1949)
print(book)  # Output: '1984' by George Orwell (1949)
print(str(book))  # Same output
```

**How it's called**: 
- `print(object)`
- `str(object)`
- `f"{object}"`

#### `__repr__` - Developer-Friendly String

The `__repr__` method returns an **unambiguous**, **developer-friendly** string representation. Ideally, it should be valid Python code that could recreate the object.

```python
class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.year})"
    
    def __str__(self):
        return f"'{self.title}' by {self.author} ({self.year})"

book = Book("1984", "George Orwell", 1949)

print(book)          # Calls __str__: '1984' by George Orwell (1949)
print(repr(book))    # Calls __repr__: Book('1984', 'George Orwell', 1949)
print([book])        # Uses __repr__: [Book('1984', 'George Orwell', 1949)]
```

**How it's called**:
- `repr(object)`
- In the Python REPL (when you type the variable name)
- When the object is inside a container (list, dict, etc.)

> [!IMPORTANT]
> **Best Practice**: 
> - Always implement `__repr__` (it's more important!)
> - If you only implement `__repr__`, it will be used as a fallback when `__str__` is called
> - `__str__` is optional and for prettier output

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # No __str__ defined - __repr__ will be used as fallback

v = Vector(3, 4)
print(v)      # Output: Vector(3, 4) - uses __repr__ as fallback
print(repr(v))  # Output: Vector(3, 4)
```

---

### 3. `__len__` - Length

Allows your object to work with the `len()` function.

```python
class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []
    
    def add_song(self, song):
        self.songs.append(song)
    
    def __len__(self):
        return len(self.songs)

playlist = Playlist("My Favorites")
playlist.add_song("Song 1")
playlist.add_song("Song 2")
playlist.add_song("Song 3")

print(len(playlist))  # Output: 3
```

**How it's called**: `len(object)`

---

### 4. `__getitem__` and `__setitem__` - Indexing

These methods allow your object to use square bracket notation like lists and dictionaries.

```python
class ShoppingCart:
    def __init__(self):
        self.items = {}
    
    def __getitem__(self, item_name):
        """Called when you use cart[item_name]"""
        return self.items.get(item_name, 0)
    
    def __setitem__(self, item_name, quantity):
        """Called when you use cart[item_name] = quantity"""
        self.items[item_name] = quantity
    
    def __str__(self):
        return f"Cart: {self.items}"

cart = ShoppingCart()
cart["apples"] = 5      # Calls __setitem__
cart["bananas"] = 3     # Calls __setitem__

print(cart["apples"])   # Calls __getitem__, Output: 5
print(cart["oranges"])  # Calls __getitem__, Output: 0 (not in cart)
print(cart)             # Output: Cart: {'apples': 5, 'bananas': 3}
```

**How they're called**:
- `object[key]` → calls `__getitem__(key)`
- `object[key] = value` → calls `__setitem__(key, value)`

---

### 5. Comparison Methods

These methods allow your objects to be compared using operators like `==`, `<`, `>`, etc.

| Method | Operator | Called By |
|--------|----------|-----------|
| `__eq__` | `==` | `a == b` |
| `__ne__` | `!=` | `a != b` |
| `__lt__` | `<` | `a < b` |
| `__le__` | `<=` | `a <= b` |
| `__gt__` | `>` | `a > b` |
| `__ge__` | `>=` | `a >= b` |

```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Check if two students have the same grade"""
        return self.grade == other.grade
    
    def __lt__(self, other):
        """Compare students by grade (for sorting)"""
        return self.grade < other.grade
    
    def __repr__(self):
        return f"Student('{self.name}', {self.grade})"

alice = Student("Alice", 85)
bob = Student("Bob", 90)
charlie = Student("Charlie", 85)

print(alice == charlie)  # True (same grade)
print(alice < bob)       # True (85 < 90)
print(bob > alice)       # True (90 > 85)

# Now we can sort students!
students = [bob, alice, charlie]
students.sort()  # Uses __lt__ for comparison
print(students)  # [Student('Alice', 85), Student('Charlie', 85), Student('Bob', 90)]
```

**How they're called**:
- `object1 == object2` → calls `object1.__eq__(object2)`
- `object1 < object2` → calls `object1.__lt__(object2)`
- And so on...

---

### 6. Arithmetic Methods

These methods allow your objects to use arithmetic operators.

| Method | Operator | Called By |
|--------|----------|-----------|
| `__add__` | `+` | `a + b` |
| `__sub__` | `-` | `a - b` |
| `__mul__` | `*` | `a * b` |
| `__truediv__` | `/` | `a / b` |
| `__floordiv__` | `//` | `a // b` |
| `__mod__` | `%` | `a % b` |
| `__pow__` | `**` | `a ** b` |

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Add two vectors"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtract two vectors"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Multiply vector by a scalar"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

v3 = v1 + v2  # Calls __add__
print(v3)     # Output: Vector(4, 6)

v4 = v1 - v2  # Calls __sub__
print(v4)     # Output: Vector(2, 2)

v5 = v1 * 2   # Calls __mul__
print(v5)     # Output: Vector(6, 8)
```

**How they're called**:
- `object1 + object2` → calls `object1.__add__(object2)`
- `object1 * value` → calls `object1.__mul__(value)`
- And so on...

---

### 7. `__call__` - Making Objects Callable

This method allows you to call an object like a function.

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Called when you use multiplier(x)"""
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))   # Output: 10 (5 * 2)
print(triple(5))   # Output: 15 (5 * 3)
```

**How it's called**: `object(arguments)` → calls `object.__call__(arguments)`

**Real-world use case**: Creating function-like objects with state

```python
class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

---

### 8. `__enter__` and `__exit__` - Context Managers

These methods allow your object to be used with the `with` statement (context manager).

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Called when entering the 'with' block"""
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block"""
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions

# Usage
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
# File is automatically closed after the 'with' block

# Output:
# Opening file: test.txt
# Closing file: test.txt
```

**How they're called**:
```python
with object as var:
    # __enter__ is called here, return value is assigned to 'var'
    pass
# __exit__ is called here, even if an exception occurred
```

---

### Summary Table: Common Dunder Methods

| Purpose | Method | Called By | Returns |
|---------|--------|-----------|---------|
| Constructor | `__init__(self, ...)` | `ClassName(...)` | Nothing |
| String (user) | `__str__(self)` | `print(obj)`, `str(obj)` | String |
| String (dev) | `__repr__(self)` | `repr(obj)`, REPL | String |
| Length | `__len__(self)` | `len(obj)` | Integer |
| Get item | `__getitem__(self, key)` | `obj[key]` | Value |
| Set item | `__setitem__(self, key, val)` | `obj[key] = val` | Nothing |
| Equality | `__eq__(self, other)` | `obj1 == obj2` | Boolean |
| Less than | `__lt__(self, other)` | `obj1 < obj2` | Boolean |
| Addition | `__add__(self, other)` | `obj1 + obj2` | New object |
| Callable | `__call__(self, ...)` | `obj(...)` | Any |
| Context enter | `__enter__(self)` | `with obj:` | Any |
| Context exit | `__exit__(self, ...)` | End of `with` | Boolean |

---

## Decorators

**Decorators** are a powerful feature in Python that allow you to modify or enhance functions without changing their code. They "wrap" a function with additional functionality.

### What is a Decorator?

A decorator is a function that takes another function as input and returns a modified version of that function.

### Basic Decorator Syntax

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# Output:
# Before function call
# Hello!
# After function call
```

The `@my_decorator` syntax is equivalent to:
```python
def say_hello():
    print("Hello!")

say_hello = my_decorator(say_hello)
```

### Decorator with Arguments

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):  # Accept any arguments
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

result = add(3, 5)

# Output:
# Calling add with args=(3, 5), kwargs={}
# add returned: 8
```

### Real-World Example: Timing Decorator

```python
import time

def timer(func):
    """Decorator that measures execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done!"

result = slow_function()
# Output: slow_function took 1.0012 seconds
```

### Real-World Example: Logging Decorator

```python
def log_calls(func):
    """Decorator that logs function calls"""
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"[LOG] {func.__name__} succeeded")
            return result
        except Exception as e:
            print(f"[LOG] {func.__name__} failed with error: {e}")
            raise
    return wrapper

@log_calls
def divide(a, b):
    return a / b

print(divide(10, 2))  # Works fine
# Output:
# [LOG] Calling divide
# [LOG] divide succeeded
# 5.0

print(divide(10, 0))  # Raises error
# Output:
# [LOG] Calling divide
# [LOG] divide failed with error: division by zero
```

### Decorator with Parameters

To create a decorator that accepts its own parameters, you need an extra level of nesting:

```python
def repeat(times):
    """Decorator that repeats function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Output:
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!
```

### Built-in Decorators

Python provides several built-in decorators:

#### 1. `@property` - Getter

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Get radius"""
        return self._radius
    
    @property
    def area(self):
        """Calculate area"""
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.radius)  # Access like an attribute, not a method!
print(c.area)    # Calculated on the fly
# Output: 5
# Output: 78.53975
```

#### 2. `@staticmethod` - Static Method

A static method doesn't receive the class or instance as the first argument. It's just a regular function inside a class.

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def multiply(a, b):
        return a * b

# Can call without creating an instance
print(MathUtils.add(3, 5))      # 8
print(MathUtils.multiply(3, 5)) # 15
```

#### 3. `@classmethod` - Class Method

A class method receives the class (not the instance) as the first argument.

```python
class Person:
    count = 0
    
    def __init__(self, name):
        self.name = name
        Person.count += 1
    
    @classmethod
    def get_count(cls):
        return f"Total persons created: {cls.count}"
    
    @classmethod
    def from_birth_year(cls, name, birth_year):
        """Alternative constructor"""
        age = 2026 - birth_year
        return cls(f"{name} (age {age})")

p1 = Person("Alice")
p2 = Person("Bob")
print(Person.get_count())  # Total persons created: 2

p3 = Person.from_birth_year("Charlie", 2000)
print(p3.name)  # Charlie (age 26)
```

### Common Decorator Patterns

```python
# 1. Authentication decorator
def require_auth(func):
    def wrapper(user, *args, **kwargs):
        if not user.get("is_authenticated"):
            raise PermissionError("User must be authenticated")
        return func(user, *args, **kwargs)
    return wrapper

# 2. Caching/Memoization decorator
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Much faster due to caching!
print(fibonacci(100))
```

---

## OS Module Essentials

The `os` module provides functions to interact with the operating system.

```python
import os
```

### Working with Directories

```python
# Get current working directory
cwd = os.getcwd()
print(f"Current directory: {cwd}")

# Change directory
os.chdir("/path/to/directory")

# List files and folders in a directory
files = os.listdir(".")  # Current directory
print(files)

# Create a directory
os.mkdir("new_folder")

# Create nested directories
os.makedirs("parent/child/grandchild", exist_ok=True)

# Remove directory (must be empty)
os.rmdir("folder_name")

# Check if path exists
if os.path.exists("file.txt"):
    print("File exists")

# Check if it's a file or directory
os.path.isfile("file.txt")   # True if it's a file
os.path.isdir("folder")      # True if it's a directory
```

### Working with Paths

```python
# Join paths (OS-independent way)
path = os.path.join("folder", "subfolder", "file.txt")
# On Windows: folder\subfolder\file.txt
# On Unix: folder/subfolder/file.txt

# Get absolute path
abs_path = os.path.abspath("file.txt")

# Get directory name and file name
full_path = "/home/user/documents/file.txt"
directory = os.path.dirname(full_path)   # "/home/user/documents"
filename = os.path.basename(full_path)   # "file.txt"

# Split path and extension
name, ext = os.path.splitext("document.pdf")
# name = "document", ext = ".pdf"
```

### Working with Files

```python
# Rename file or directory
os.rename("old_name.txt", "new_name.txt")

# Remove file
os.remove("file.txt")

# Get file size
size = os.path.getsize("file.txt")  # Size in bytes

# Get file modification time
import time
mod_time = os.path.getmtime("file.txt")
readable_time = time.ctime(mod_time)
```

### Environment Variables

```python
# Get environment variable
home = os.getenv("HOME")  # Returns None if not found
path = os.environ["PATH"]  # Raises KeyError if not found

# Set environment variable
os.environ["MY_VAR"] = "value"

# Get all environment variables
all_env = os.environ
```

### Running System Commands

```python
# Execute a system command
os.system("ls -l")  # Unix
os.system("dir")    # Windows

# Get process ID
pid = os.getpid()
print(f"Current process ID: {pid}")
```

### Practical Example: File Organization

```python
import os

def organize_files_by_extension(directory):
    """Organize files in a directory by their extensions"""
    
    # Get all files in directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip if it's a directory
        if os.path.isdir(filepath):
            continue
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        ext = ext[1:]  # Remove the dot
        
        if ext:  # If file has an extension
            # Create folder for this extension
            ext_folder = os.path.join(directory, ext)
            os.makedirs(ext_folder, exist_ok=True)
            
            # Move file to extension folder
            new_path = os.path.join(ext_folder, filename)
            os.rename(filepath, new_path)
            print(f"Moved {filename} to {ext}/")

# Usage
# organize_files_by_extension("/path/to/messy/folder")
```

---

## SYS Module Essentials

The `sys` module provides access to Python interpreter variables and functions.

```python
import sys
```

### Command-Line Arguments

```python
# Get command-line arguments
# If you run: python script.py arg1 arg2 arg3

script_name = sys.argv[0]  # "script.py"
arguments = sys.argv[1:]   # ["arg1", "arg2", "arg3"]

# Example script
if len(sys.argv) < 2:
    print("Usage: python script.py <name>")
    sys.exit(1)

name = sys.argv[1]
print(f"Hello, {name}!")
```

### Python Version and Platform

```python
# Get Python version
print(sys.version)
# Output: 3.11.0 (main, Oct 24 2022, 18:26:48) [GCC 11.3.0]

# Get version info (more structured)
print(sys.version_info)
# Output: sys.version_info(major=3, minor=11, micro=0, ...)

# Check Python version programmatically
if sys.version_info < (3, 6):
    print("This script requires Python 3.6 or higher")
    sys.exit(1)

# Get platform
print(sys.platform)
# Output: 'linux', 'darwin' (macOS), 'win32', etc.
```

### Exit the Program

```python
# Exit with success (0)
sys.exit(0)

# Exit with error (non-zero)
sys.exit(1)

# Exit with error message
sys.exit("Error: Something went wrong!")
```

### Module Search Path

```python
# Get list of paths Python searches for modules
print(sys.path)

# Add a custom path
sys.path.append("/path/to/my/modules")
```

### Standard Input/Output/Error

```python
# Standard output (normal print)
sys.stdout.write("This is stdout\n")

# Standard error (for error messages)
sys.stderr.write("This is an error message\n")

# Standard input
# name = sys.stdin.readline().strip()
```

### Memory and Performance

```python
# Get size of an object in bytes
import sys

x = [1, 2, 3, 4, 5]
size = sys.getsizeof(x)
print(f"Size of list: {size} bytes")

# Get recursion limit
limit = sys.getrecursionlimit()
print(f"Recursion limit: {limit}")

# Set recursion limit (use with caution!)
sys.setrecursionlimit(2000)
```

### Practical Example: Command-Line Calculator

```python
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python calc.py <num1> <operator> <num2>")
        print("Example: python calc.py 5 + 3")
        sys.exit(1)
    
    try:
        num1 = float(sys.argv[1])
        operator = sys.argv[2]
        num2 = float(sys.argv[3])
    except ValueError:
        print("Error: Numbers must be valid floats", file=sys.stderr)
        sys.exit(1)
    
    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        if num2 == 0:
            print("Error: Division by zero", file=sys.stderr)
            sys.exit(1)
        result = num1 / num2
    else:
        print(f"Error: Unknown operator '{operator}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"{num1} {operator} {num2} = {result}")

if __name__ == "__main__":
    main()

# Usage:
# python calc.py 10 + 5    # Output: 10.0 + 5.0 = 15.0
# python calc.py 10 / 2    # Output: 10.0 / 2.0 = 5.0
```

---

## Practice Exercises

### Exercise 1: Complete the Class

**Task**: Implement the missing dunder methods for a `BankAccount` class.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    # TODO: Implement __str__ to return: "BankAccount(owner='Alice', balance=100)"
    
    # TODO: Implement __repr__ to return: "BankAccount('Alice', 100)"
    
    # TODO: Implement __eq__ to compare accounts by balance
    
    # TODO: Implement __lt__ to compare accounts by balance (for sorting)
    
    # TODO: Implement __add__ to combine balances of two accounts

# Test your implementation:
acc1 = BankAccount("Alice", 100)
acc2 = BankAccount("Bob", 150)
acc3 = BankAccount("Charlie", 100)

# Should work:
print(acc1)  # User-friendly string
print(repr(acc2))  # Developer-friendly string
print(acc1 == acc3)  # True (same balance)
print(acc1 < acc2)  # True (100 < 150)
combined = acc1 + acc2  # New account with combined balance
print(combined.balance)  # 250
```

<details>
<summary><strong>Solution</strong></summary>

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def __str__(self):
        return f"BankAccount(owner='{self.owner}', balance={self.balance})"
    
    def __repr__(self):
        return f"BankAccount('{self.owner}', {self.balance})"
    
    def __eq__(self, other):
        return self.balance == other.balance
    
    def __lt__(self, other):
        return self.balance < other.balance
    
    def __add__(self, other):
        new_owner = f"{self.owner} & {other.owner}"
        new_balance = self.balance + other.balance
        return BankAccount(new_owner, new_balance)

# Test
acc1 = BankAccount("Alice", 100)
acc2 = BankAccount("Bob", 150)
acc3 = BankAccount("Charlie", 100)

print(acc1)           # BankAccount(owner='Alice', balance=100)
print(repr(acc2))     # BankAccount('Bob', 150)
print(acc1 == acc3)   # True
print(acc1 < acc2)    # True

combined = acc1 + acc2
print(combined.owner)    # Alice & Bob
print(combined.balance)  # 250
```
</details>

---

### Exercise 2: Create a Decorator

**Task**: Create a `validate_positive` decorator that ensures all arguments to a function are positive numbers.

```python
# TODO: Implement the validate_positive decorator

@validate_positive
def calculate_area(length, width):
    return length * width

# Should work:
print(calculate_area(5, 10))  # 50

# Should raise ValueError:
# calculate_area(-5, 10)
# calculate_area(5, -10)
```

<details>
<summary><strong>Solution</strong></summary>

```python
def validate_positive(func):
    def wrapper(*args, **kwargs):
        # Check all positional arguments
        for arg in args:
            if isinstance(arg, (int, float)) and arg <= 0:
                raise ValueError(f"All arguments must be positive. Got: {arg}")
        
        # Check all keyword arguments
        for value in kwargs.values():
            if isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"All arguments must be positive. Got: {value}")
        
        return func(*args, **kwargs)
    return wrapper

@validate_positive
def calculate_area(length, width):
    return length * width

print(calculate_area(5, 10))  # 50

try:
    calculate_area(-5, 10)
except ValueError as e:
    print(f"Error: {e}")  # Error: All arguments must be positive. Got: -5
```
</details>

---

### Exercise 3: File Manager with Context Manager

**Task**: Create a `CSVWriter` class that acts as a context manager to write CSV data.

```python
# TODO: Implement CSVWriter with __enter__ and __exit__

# Should work like this:
with CSVWriter("output.csv") as writer:
    writer.write_row(["Name", "Age", "City"])
    writer.write_row(["Alice", "25", "NYC"])
    writer.write_row(["Bob", "30", "LA"])
# File should be automatically closed

# The csv file should contain:
# Name,Age,City
# Alice,25,NYC
# Bob,30,LA
```

<details>
<summary><strong>Solution</strong></summary>

```python
class CSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False
    
    def write_row(self, row):
        self.file.write(",".join(row) + "\n")

# Usage
with CSVWriter("output.csv") as writer:
    writer.write_row(["Name", "Age", "City"])
    writer.write_row(["Alice", "25", "NYC"])
    writer.write_row(["Bob", "30", "LA"])

# Verify
with open("output.csv", "r") as f:
    print(f.read())
```
</details>

---

### Exercise 4: OS Module Challenge

**Task**: Write a function that finds all Python files (`.py`) in a directory and its subdirectories.

```python
import os

def find_python_files(directory):
    """
    Find all .py files in directory and subdirectories.
    Return a list of absolute paths.
    """
    # TODO: Implement this function
    pass

# Test (this should find all .py files in current directory)
# python_files = find_python_files(".")
# for file in python_files:
#     print(file)
```

<details>
<summary><strong>Solution</strong></summary>

```python
import os

def find_python_files(directory):
    """
    Find all .py files in directory and subdirectories.
    Return a list of absolute paths.
    """
    python_files = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                # Get absolute path
                abs_path = os.path.abspath(os.path.join(root, file))
                python_files.append(abs_path)
    
    return python_files

# Alternative solution using list comprehension
def find_python_files_v2(directory):
    return [
        os.path.abspath(os.path.join(root, file))
        for root, dirs, files in os.walk(directory)
        for file in files
        if file.endswith('.py')
    ]

# Test
python_files = find_python_files(".")
print(f"Found {len(python_files)} Python files:")
for file in python_files[:5]:  # Print first 5
    print(file)
```
</details>

---

### Exercise 5: SYS Module Challenge

**Task**: Create a script that accepts a filename as a command-line argument and prints file statistics.

```python
import sys
import os

def main():
    """
    Usage: python script.py <filename>
    Prints: file size, number of lines, number of words
    """
    # TODO: Implement this
    # 1. Check if filename argument is provided
    # 2. Check if file exists
    # 3. Calculate and print statistics
    pass

if __name__ == "__main__":
    main()
```

<details>
<summary><strong>Solution</strong></summary>

```python
import sys
import os

def main():
    """
    Usage: python script.py <filename>
    Prints: file size, number of lines, number of words
    """
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>", file=sys.stderr)
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(filename):
        print(f"Error: '{filename}' is not a file", file=sys.stderr)
        sys.exit(1)
    
    # Get file size
    size = os.path.getsize(filename)
    
    # Count lines and words
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            num_words = sum(len(line.split()) for line in lines)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    print(f"File Statistics for '{filename}':")
    print(f"  Size: {size} bytes")
    print(f"  Lines: {num_lines}")
    print(f"  Words: {num_words}")

if __name__ == "__main__":
    main()

# Usage:
# python script.py myfile.txt
# Output:
# File Statistics for 'myfile.txt':
#   Size: 1234 bytes
#   Lines: 45
#   Words: 234
```
</details>

---

## Summary

### Key Takeaways

> [!IMPORTANT]
> **Dunder Methods**:
> - Allow you to customize how your objects behave with Python's built-in operations
> - `__str__` for user-friendly strings, `__repr__` for developer-friendly strings
> - Always implement `__repr__`; `__str__` is optional
> - Use `__init__` to initialize objects
> - Comparison methods (`__eq__`, `__lt__`, etc.) enable sorting and comparisons
> - Arithmetic methods (`__add__`, `__sub__`, etc.) enable math operations

> [!IMPORTANT]
> **Decorators**:
> - Functions that modify or enhance other functions
> - Use `@decorator_name` syntax above function definition
> - Common built-in decorators: `@property`, `@staticmethod`, `@classmethod`
> - Useful for cross-cutting concerns: logging, timing, authentication, caching

> [!IMPORTANT]
> **OS Module**:
> - `os.getcwd()`, `os.chdir()` - working directory
> - `os.listdir()`, `os.walk()` - list files
> - `os.mkdir()`, `os.makedirs()` - create directories
> - `os.path.join()` - join paths (OS-independent)
> - `os.path.exists()`, `os.path.isfile()` - check paths

> [!IMPORTANT]
> **SYS Module**:
> - `sys.argv` - command-line arguments
> - `sys.exit()` - exit program
> - `sys.version`, `sys.platform` - system info
> - `sys.path` - module search paths
> - `sys.stdout`, `sys.stderr` - standard output/error

---

## Next Steps

1. **Practice**: Implement dunder methods in your own classes
2. **Experiment**: Create custom decorators for common tasks
3. **Build**: Create a command-line tool using `sys.argv` and OS operations
4. **Explore**: Try the `pathlib` module as a modern alternative to `os.path`

> [!TIP]
> The best way to master these concepts is to use them in real projects. Start small - add `__str__` and `__repr__` to your classes, create a simple decorator, or build a file management script!
