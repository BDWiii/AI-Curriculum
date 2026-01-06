# Object-Oriented Programming (OOP) Fundamentals

## Table of Contents
1. [Introduction](#introduction)
2. [Why Use OOP?](#why-use-oop)
3. [Functions](#functions)
4. [Classes and Objects](#classes-and-objects)
5. [Inheritance](#inheritance)
6. [Practice Problems](#practice-problems)

---

## Introduction

**Object-Oriented Programming (OOP)** is a programming paradigm that organizes code around "objects" rather than functions and logic. An object combines data (attributes) and behavior (methods) into a single unit.

### The Evolution of Programming Paradigms

```
Procedural Programming          Object-Oriented Programming
(Sequential steps)              (Objects interacting)

Step 1: Get data         →      Student Object
Step 2: Process data             - name
Step 3: Print result             - age
                                 - calculate_grade()
                                 - display_info()
```

### Core Concepts

OOP is built on four main principles:

1. **Encapsulation**: Bundling data and methods together
2. **Abstraction**: Hiding complex implementation details
3. **Inheritance**: Creating new classes from existing ones
4. **Polymorphism**: Using a single interface for different data types

---

## Why Use OOP?

### Real-World Modeling

OOP allows you to model real-world entities naturally. For example:

**Without OOP (Procedural):**
```python
# Managing student data with separate variables
student1_name = "Alice"
student1_age = 20
student1_grade = 85

student2_name = "Bob"
student2_age = 21
student2_grade = 90

# Functions operate on data separately
def calculate_average(grade1, grade2):
    return (grade1 + grade2) / 2
```

**With OOP:**
```python
# Student as a self-contained object
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def get_letter_grade(self):
        if self.grade >= 90:
            return 'A'
        elif self.grade >= 80:
            return 'B'
        return 'C'

# Create student objects
alice = Student("Alice", 20, 85)
bob = Student("Bob", 21, 90)
```

### Benefits of OOP

1. **Code Reusability**: Write once, use many times through inheritance
2. **Modularity**: Each object is independent and can be modified without affecting others
3. **Maintainability**: Easier to update and fix bugs in isolated objects
4. **Scalability**: Easy to add new features by creating new classes
5. **Data Security**: Encapsulation protects data from unauthorized access

> **Why OOP for AI/ML?**
> 
> In machine learning, OOP helps organize complex systems:
> - **Models** as objects with methods like `train()`, `predict()`
> - **Data processors** as classes with `fit()`, `transform()`
> - **Neural network layers** as reusable building blocks
> - **Experiments** as objects tracking parameters and results

---

## Functions

### What are Functions?

**Functions** are reusable blocks of code that perform a specific task. They help break down complex problems into smaller, manageable pieces.

### Syntax

```python
def function_name(parameters):
    """Docstring: describes what the function does"""
    # Function body
    return result
```

### Basic Example

```python
def greet(name):
    """Returns a greeting message"""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

### Functions with Multiple Parameters

```python
def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index"""
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

# Usage
bmi = calculate_bmi(70, 1.75)
print(f"BMI: {bmi}")  # Output: BMI: 22.86
```

### Functions with Default Parameters

```python
def power(base, exponent=2):
    """Raise base to the power of exponent (default is 2)"""
    return base ** exponent

print(power(5))      # Output: 25 (5^2)
print(power(5, 3))   # Output: 125 (5^3)
```

### Why Functions Matter in OOP

Functions become **methods** when defined inside a class, operating on the object's data.

---

### 🧪 Practice Problem 1: Functions

**Problem:** Write a function `convert_temperature(celsius)` that converts Celsius to Fahrenheit using the formula: `F = (C × 9/5) + 32`

**Test Cases:**
- `convert_temperature(0)` should return `32.0`
- `convert_temperature(100)` should return `212.0`
- `convert_temperature(-40)` should return `-40.0`

<details>
<summary>💡 Click to reveal solution</summary>

```python
def convert_temperature(celsius):
    """Convert Celsius to Fahrenheit"""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Test cases
print(convert_temperature(0))     # Output: 32.0
print(convert_temperature(100))   # Output: 212.0
print(convert_temperature(-40))   # Output: -40.0
```

**Explanation:**
- Function takes one parameter: `celsius`
- Applies the conversion formula
- Returns the calculated Fahrenheit value
</details>

---

## Classes and Objects

### What is a Class?

A **class** is a blueprint or template for creating objects. It defines attributes (data) and methods (behavior) that the objects will have.

**Analogy:** 
- **Class** = Cookie cutter (blueprint)
- **Object** = Cookie (actual instance)

### What is an Object?

An **object** is an instance of a class. It's a concrete entity created from the class blueprint with specific values for its attributes.

### Basic Class Syntax

```python
class ClassName:
    """Class docstring"""
    
    def __init__(self, parameters):
        """Constructor: initializes object attributes"""
        self.attribute = parameters
    
    def method_name(self):
        """Method: defines object behavior"""
        # Method body
        pass
```

### Simple Class Example

```python
class Dog:
    """A simple Dog class"""
    
    def __init__(self, name, age):
        """Initialize dog attributes"""
        self.name = name
        self.age = age
    
    def bark(self):
        """Dog behavior: barking"""
        return f"{self.name} says Woof!"
    
    def get_age_in_dog_years(self):
        """Calculate age in dog years (1 human year = 7 dog years)"""
        return self.age * 7

# Create objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

# Access attributes
print(dog1.name)  # Output: Buddy
print(dog2.age)   # Output: 5

# Call methods
print(dog1.bark())                      # Output: Buddy says Woof!
print(dog2.get_age_in_dog_years())      # Output: 35
```

### Understanding `self`

- `self` refers to the instance of the class
- It's the first parameter of every instance method
- Allows access to attributes and methods of the object

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius  # self.radius belongs to this specific circle
    
    def area(self):
        return 3.14159 * self.radius ** 2

circle1 = Circle(5)
circle2 = Circle(10)

print(circle1.area())  # Output: 78.53975 (π × 5²)
print(circle2.area())  # Output: 314.159 (π × 10²)
```

### Class Attributes vs Instance Attributes

```python
class Car:
    # Class attribute (shared by all instances)
    wheels = 4
    
    def __init__(self, brand, model):
        # Instance attributes (unique to each instance)
        self.brand = brand
        self.model = model
    
    def info(self):
        return f"{self.brand} {self.model} has {self.wheels} wheels"

car1 = Car("Toyota", "Camry")
car2 = Car("Honda", "Civic")

print(car1.info())  # Output: Toyota Camry has 4 wheels
print(car2.info())  # Output: Honda Civic has 4 wheels
print(Car.wheels)   # Output: 4 (accessing class attribute)
```

---

### 🧪 Practice Problem 2: Classes and Objects

**Problem:** Create a `BankAccount` class with the following specifications:

**Requirements:**
1. Attributes: `account_holder` (string), `balance` (float, default 0)
2. Methods:
   - `deposit(amount)`: Add amount to balance, return new balance
   - `withdraw(amount)`: Subtract amount if sufficient funds, return new balance or error message
   - `get_balance()`: Return current balance

**Test Cases:**
```python
account = BankAccount("Alice", 1000)
print(account.get_balance())      # Should output: 1000
print(account.deposit(500))       # Should output: 1500
print(account.withdraw(200))      # Should output: 1300
print(account.withdraw(2000))     # Should output error message
```

<details>
<summary>💡 Click to reveal solution</summary>

```python
class BankAccount:
    """A simple bank account class"""
    
    def __init__(self, account_holder, balance=0):
        """Initialize account with holder name and optional starting balance"""
        self.account_holder = account_holder
        self.balance = balance
    
    def deposit(self, amount):
        """Add money to the account"""
        if amount > 0:
            self.balance += amount
            return self.balance
        else:
            return "Deposit amount must be positive"
    
    def withdraw(self, amount):
        """Withdraw money from the account"""
        if amount > self.balance:
            return "Insufficient funds"
        elif amount <= 0:
            return "Withdrawal amount must be positive"
        else:
            self.balance -= amount
            return self.balance
    
    def get_balance(self):
        """Return current balance"""
        return self.balance

# Test cases
account = BankAccount("Alice", 1000)
print(account.get_balance())      # Output: 1000
print(account.deposit(500))       # Output: 1500
print(account.withdraw(200))      # Output: 1300
print(account.withdraw(2000))     # Output: Insufficient funds
```

**Explanation:**
- `__init__`: Constructor initializes the account with holder name and balance
- `deposit`: Adds amount to balance with validation
- `withdraw`: Subtracts amount with validation for sufficient funds
- `get_balance`: Returns current balance
- Each method operates on the object's own data using `self`
</details>

---

## Inheritance

### What is Inheritance?

**Inheritance** allows a class (child/derived class) to inherit attributes and methods from another class (parent/base class). This promotes code reuse and establishes a hierarchical relationship.

**Analogy:** Children inherit traits from parents (eye color, height), but can also have their own unique features.

### Basic Inheritance Syntax

```python
class ParentClass:
    """Base class"""
    def parent_method(self):
        pass

class ChildClass(ParentClass):
    """Derived class inheriting from ParentClass"""
    def child_method(self):
        pass
```

### Simple Inheritance Example

```python
class Animal:
    """Base class: Animal"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name} makes a sound"
    
    def move(self):
        return f"{self.name} is moving"

class Dog(Animal):
    """Derived class: Dog inherits from Animal"""
    
    def speak(self):
        """Override parent method"""
        return f"{self.name} says Woof!"
    
    def fetch(self):
        """New method specific to Dog"""
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    """Derived class: Cat inherits from Animal"""
    
    def speak(self):
        """Override parent method"""
        return f"{self.name} says Meow!"
    
    def scratch(self):
        """New method specific to Cat"""
        return f"{self.name} is scratching"

# Create objects
dog = Dog("Buddy")
cat = Cat("Whiskers")

# Inherited method
print(dog.move())      # Output: Buddy is moving
print(cat.move())      # Output: Whiskers is moving

# Overridden method
print(dog.speak())     # Output: Buddy says Woof!
print(cat.speak())     # Output: Whiskers says Meow!

# Class-specific methods
print(dog.fetch())     # Output: Buddy is fetching the ball
print(cat.scratch())   # Output: Whiskers is scratching
```

### Using `super()` to Call Parent Methods

```python
class Vehicle:
    """Base class"""
    
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def info(self):
        return f"{self.brand} {self.model}"

class ElectricCar(Vehicle):
    """Derived class with additional features"""
    
    def __init__(self, brand, model, battery_capacity):
        # Call parent constructor
        super().__init__(brand, model)
        # Add new attribute
        self.battery_capacity = battery_capacity
    
    def info(self):
        """Extend parent method"""
        base_info = super().info()
        return f"{base_info} | Battery: {self.battery_capacity} kWh"

# Usage
tesla = ElectricCar("Tesla", "Model 3", 75)
print(tesla.info())  # Output: Tesla Model 3 | Battery: 75 kWh
```

### Types of Inheritance

#### 1. Single Inheritance
One child class inherits from one parent class.

```python
class Parent:
    pass

class Child(Parent):
    pass
```

#### 2. Multiple Inheritance
One child class inherits from multiple parent classes.

```python
class Father:
    def skills(self):
        return "Programming"

class Mother:
    def talents(self):
        return "Music"

class Child(Father, Mother):
    """Inherits from both Father and Mother"""
    pass

child = Child()
print(child.skills())   # Output: Programming
print(child.talents())  # Output: Music
```

#### 3. Multilevel Inheritance
A child class becomes parent to another child class.

```python
class GrandParent:
    def family_name(self):
        return "Smith Family"

class Parent(GrandParent):
    def occupation(self):
        return "Engineer"

class Child(Parent):
    def hobby(self):
        return "Gaming"

child = Child()
print(child.family_name())  # Output: Smith Family (from GrandParent)
print(child.occupation())   # Output: Engineer (from Parent)
print(child.hobby())        # Output: Gaming (own method)
```

### Real-World ML Example

```python
class Model:
    """Base ML model class"""
    
    def __init__(self, name):
        self.name = name
        self.is_trained = False
    
    def train(self, data):
        """Common training interface"""
        print(f"Training {self.name}...")
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            return "Error: Model not trained yet"
        return "Predictions"

class LinearRegression(Model):
    """Linear Regression inherits from Model"""
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.weights = None
    
    def predict(self, X):
        """Override with specific implementation"""
        if not self.is_trained:
            return "Error: Model not trained yet"
        return f"Linear predictions using weights: {self.weights}"

class NeuralNetwork(Model):
    """Neural Network inherits from Model"""
    
    def __init__(self, layers):
        super().__init__("Neural Network")
        self.layers = layers
    
    def predict(self, X):
        """Override with specific implementation"""
        if not self.is_trained:
            return "Error: Model not trained yet"
        return f"Neural network prediction with {self.layers} layers"

# Usage
lr_model = LinearRegression()
nn_model = NeuralNetwork(3)

lr_model.train("training_data")
print(lr_model.predict("test_data"))  # Works because it's trained

print(nn_model.predict("test_data"))  # Error: not trained yet
```

---

### 🧪 Practice Problem 3: Inheritance

**Problem:** Create a class hierarchy for geometric shapes:

**Requirements:**

1. Create a base class `Shape` with:
   - Attribute: `name` (string)
   - Method: `area()` that returns 0 (to be overridden)
   - Method: `describe()` that returns the shape name

2. Create a derived class `Rectangle` that:
   - Inherits from `Shape`
   - Has attributes: `width` and `height`
   - Overrides `area()` to return width × height

3. Create a derived class `Circle` that:
   - Inherits from `Shape`
   - Has attribute: `radius`
   - Overrides `area()` to return π × radius² (use 3.14159)

**Test Cases:**
```python
rect = Rectangle(5, 10)
print(rect.describe())    # Should output: Rectangle
print(rect.area())        # Should output: 50

circle = Circle(7)
print(circle.describe())  # Should output: Circle
print(circle.area())      # Should output: 153.93791
```

<details>
<summary>💡 Click to reveal solution</summary>

```python
class Shape:
    """Base class for geometric shapes"""
    
    def __init__(self, name):
        """Initialize with shape name"""
        self.name = name
    
    def area(self):
        """Default area calculation (to be overridden)"""
        return 0
    
    def describe(self):
        """Return shape description"""
        return self.name

class Rectangle(Shape):
    """Rectangle class inheriting from Shape"""
    
    def __init__(self, width, height):
        """Initialize rectangle with width and height"""
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        """Calculate rectangle area"""
        return self.width * self.height

class Circle(Shape):
    """Circle class inheriting from Shape"""
    
    def __init__(self, radius):
        """Initialize circle with radius"""
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        """Calculate circle area"""
        return 3.14159 * self.radius ** 2

# Test cases
rect = Rectangle(5, 10)
print(rect.describe())    # Output: Rectangle
print(rect.area())        # Output: 50

circle = Circle(7)
print(circle.describe())  # Output: Circle
print(circle.area())      # Output: 153.93791
```

**Explanation:**
- `Shape`: Base class with common attributes and methods
- `Rectangle` and `Circle`: Inherit from `Shape`
- Both use `super().__init__()` to initialize the parent class
- Both override `area()` with their specific formulas
- Both inherit `describe()` method unchanged from parent
</details>

---

## Practice Problems

### Problem 4: Employee Management System

**Challenge:** Create a comprehensive employee management system using OOP concepts.

**Requirements:**

1. Create a base class `Employee` with:
   - Attributes: `name`, `employee_id`, `base_salary`
   - Method: `calculate_salary()` that returns base salary
   - Method: `get_info()` that returns employee information

2. Create `Manager` class (inherits from `Employee`):
   - Additional attribute: `bonus_percentage`
   - Override `calculate_salary()` to include bonus: `base_salary × (1 + bonus_percentage/100)`

3. Create `Developer` class (inherits from `Employee`):
   - Additional attribute: `programming_languages` (list)
   - Method: `add_language(language)` to add a new language
   - Method: `get_languages()` to return all languages

**Test Your Solution:**
```python
manager = Manager("Alice", "M001", 80000, 20)
developer = Developer("Bob", "D001", 70000, ["Python", "JavaScript"])

print(manager.get_info())
print(f"Manager Salary: ${manager.calculate_salary()}")

developer.add_language("Go")
print(developer.get_info())
print(f"Languages: {developer.get_languages()}")
```

<details>
<summary>💡 Click to reveal solution</summary>

```python
class Employee:
    """Base Employee class"""
    
    def __init__(self, name, employee_id, base_salary):
        """Initialize employee attributes"""
        self.name = name
        self.employee_id = employee_id
        self.base_salary = base_salary
    
    def calculate_salary(self):
        """Calculate employee salary"""
        return self.base_salary
    
    def get_info(self):
        """Return employee information"""
        return f"Employee: {self.name} (ID: {self.employee_id})"

class Manager(Employee):
    """Manager class with bonus"""
    
    def __init__(self, name, employee_id, base_salary, bonus_percentage):
        """Initialize manager with bonus percentage"""
        super().__init__(name, employee_id, base_salary)
        self.bonus_percentage = bonus_percentage
    
    def calculate_salary(self):
        """Calculate salary with bonus"""
        return self.base_salary * (1 + self.bonus_percentage / 100)
    
    def get_info(self):
        """Return manager information"""
        base_info = super().get_info()
        return f"{base_info} - Manager (Bonus: {self.bonus_percentage}%)"

class Developer(Employee):
    """Developer class with programming languages"""
    
    def __init__(self, name, employee_id, base_salary, programming_languages):
        """Initialize developer with languages"""
        super().__init__(name, employee_id, base_salary)
        self.programming_languages = programming_languages.copy()
    
    def add_language(self, language):
        """Add a new programming language"""
        if language not in self.programming_languages:
            self.programming_languages.append(language)
    
    def get_languages(self):
        """Return list of programming languages"""
        return self.programming_languages
    
    def get_info(self):
        """Return developer information"""
        base_info = super().get_info()
        return f"{base_info} - Developer"

# Test cases
manager = Manager("Alice", "M001", 80000, 20)
developer = Developer("Bob", "D001", 70000, ["Python", "JavaScript"])

print(manager.get_info())
# Output: Employee: Alice (ID: M001) - Manager (Bonus: 20%)

print(f"Manager Salary: ${manager.calculate_salary()}")
# Output: Manager Salary: $96000.0

developer.add_language("Go")
print(developer.get_info())
# Output: Employee: Bob (ID: D001) - Developer

print(f"Languages: {developer.get_languages()}")
# Output: Languages: ['Python', 'JavaScript', 'Go']
```

**Key Points:**
- Demonstrates single inheritance with multiple child classes
- Uses `super()` to call parent methods and extend functionality
- Shows method overriding with additional behavior
- Encapsulates related data and behavior in objects
</details>

---

### Problem 5: Library Management System

**Challenge:** Build a simple library system to practice all OOP concepts.

**Requirements:**

1. Create a `Book` class with:
   - Attributes: `title`, `author`, `isbn`, `is_available` (default True)
   - Method: `borrow()` - marks book as unavailable if available
   - Method: `return_book()` - marks book as available
   - Method: `get_info()` - returns book information

2. Create a `Library` class with:
   - Attribute: `books` (list of Book objects)
   - Method: `add_book(book)` - adds a book to the library
   - Method: `find_book(isbn)` - returns book with given ISBN or None
   - Method: `list_available_books()` - returns list of available books

**Test Your Solution:**
```python
library = Library()

book1 = Book("Python Crash Course", "Eric Matthes", "ISBN001")
book2 = Book("Clean Code", "Robert Martin", "ISBN002")

library.add_book(book1)
library.add_book(book2)

book1.borrow()
print(library.list_available_books())  # Should show only book2
```

<details>
<summary>💡 Click to reveal solution</summary>

```python
class Book:
    """Represents a book in the library"""
    
    def __init__(self, title, author, isbn, is_available=True):
        """Initialize book attributes"""
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_available = is_available
    
    def borrow(self):
        """Borrow the book if available"""
        if self.is_available:
            self.is_available = False
            return f"'{self.title}' borrowed successfully"
        else:
            return f"'{self.title}' is not available"
    
    def return_book(self):
        """Return the book to the library"""
        if not self.is_available:
            self.is_available = True
            return f"'{self.title}' returned successfully"
        else:
            return f"'{self.title}' was not borrowed"
    
    def get_info(self):
        """Return book information"""
        status = "Available" if self.is_available else "Borrowed"
        return f"{self.title} by {self.author} (ISBN: {self.isbn}) - {status}"

class Library:
    """Manages a collection of books"""
    
    def __init__(self):
        """Initialize library with empty book list"""
        self.books = []
    
    def add_book(self, book):
        """Add a book to the library"""
        self.books.append(book)
        return f"Added '{book.title}' to the library"
    
    def find_book(self, isbn):
        """Find a book by ISBN"""
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None
    
    def list_available_books(self):
        """Return list of available books"""
        available = [book.get_info() for book in self.books if book.is_available]
        if available:
            return available
        else:
            return ["No books available"]

# Test cases
library = Library()

book1 = Book("Python Crash Course", "Eric Matthes", "ISBN001")
book2 = Book("Clean Code", "Robert Martin", "ISBN002")

library.add_book(book1)
library.add_book(book2)

print(book1.borrow())
# Output: 'Python Crash Course' borrowed successfully

print(library.list_available_books())
# Output: ['Clean Code by Robert Martin (ISBN: ISBN002) - Available']

print(book1.return_book())
# Output: 'Python Crash Course' returned successfully

print(library.list_available_books())
# Output: ['Python Crash Course by Eric Matthes (ISBN: ISBN001) - Available',
#          'Clean Code by Robert Martin (ISBN: ISBN002) - Available']
```

**Key Concepts Demonstrated:**
- **Encapsulation**: Book data and behavior bundled together
- **Object Composition**: Library contains Book objects
- **State Management**: Books track their availability
- **List Comprehension**: Filter available books efficiently
</details>

---

## Summary

### Key Takeaways

1. **Functions** are reusable blocks of code that perform specific tasks

2. **Classes** are blueprints for creating objects with attributes and methods

3. **Objects** are instances of classes with specific values

4. **Inheritance** allows code reuse by creating child classes from parent classes

5. **OOP Benefits**:
   - **Modularity**: Break complex problems into manageable pieces
   - **Reusability**: Write code once, use it many times
   - **Maintainability**: Easier to update and debug
   - **Flexibility**: Extend functionality through inheritance

### OOP in AI/ML Context

```python
# Example: Scikit-learn style
class MyModel:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X, y):
        """Train the model"""
        # Training logic
        pass
    
    def predict(self, X):
        """Make predictions"""
        # Prediction logic
        pass

# Usage
model = MyModel(learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Quick Reference

| Concept | Purpose | Example |
|---------|---------|---------|
| **Function** | Reusable code block | `def greet(name): ...` |
| **Class** | Blueprint for objects | `class Dog: ...` |
| **Object** | Instance of a class | `my_dog = Dog("Buddy")` |
| **`__init__`** | Constructor method | `def __init__(self, name): ...` |
| **`self`** | Reference to instance | `self.name = name` |
| **Inheritance** | Derive from parent class | `class Cat(Animal): ...` |
| **`super()`** | Call parent method | `super().__init__(name)` |
| **Override** | Replace parent method | Redefine method in child |

### Next Steps

Now that you understand OOP fundamentals, you can:
- Explore advanced OOP concepts (encapsulation, polymorphism)
- Learn about special methods (`__str__`, `__repr__`, `__len__`)
- Study design patterns (Singleton, Factory, Observer)
- Apply OOP to build ML model classes
- Practice with real-world projects

**Remember:** OOP is about organizing code to mirror real-world relationships. Think about entities, their properties, and their behaviors!

---

## Congratulations! 🎉

You've completed the OOP fundamentals lesson! You now understand how to:
- ✅ Write reusable functions
- ✅ Create classes and objects
- ✅ Use inheritance for code reuse
- ✅ Build real-world applications with OOP

Keep practicing with the problems above and try creating your own classes for real-world entities like students, products, or even ML models!
