# Algorithm Visualizer for Closest Pair of Points and Karatsuba Multiplication

This repository contains the **Design and Analysis of Algorithms (DAA)** project made by 3 group members, which implements and visualizes the **Closest Pair of Points** and **Karatsuba Integer Multiplication** algorithms. The system offers an interactive GUI for users to input data, execute algorithms, and visualize their operations.

---

## Group Members

1. Azka Sahar Shaikh  
2. Muhammad Sudais Katiya  
3. Sumaiya Waheed

---

## Project Overview

The **Algorithm Visualizer** bridges the gap between theoretical concepts and practical understanding of computational algorithms. By providing a dynamic graphical interface, it allows users to interact with algorithms step-by-step.

### Algorithms Implemented
1. **Closest Pair of Points**:
   - A divide-and-conquer algorithm to find the closest pair of points in a 2D plane.
   - Visualizes points and highlights the closest pair.

2. **Karatsuba Integer Multiplication**:
   - A recursive algorithm for fast multiplication of large integers.
   - Visualizes recursive steps in a tree graph.

---

## Features

- **Graphical User Interface (GUI)**:
  - Built using `customtkinter` for an interactive and user-friendly experience.
  - Allows users to select an algorithm, input datasets, and view results dynamically.

- **Visualization Tools**:
  - Uses `matplotlib` for graph plotting and `networkx` for tree visualization.
  - Highlights key steps in the algorithms for better understanding.

- **Multithreading**:
  - Ensures the GUI remains responsive during heavy computations.

---

## Experimental Setup

- **Closest Pair of Points**:
  - Datasets: Text files containing 2D coordinates in the format `(x, y)`.
  - Range: Diverse coordinate ranges from small-scale to large-scale values.

- **Karatsuba Integer Multiplication**:
  - Datasets: Text files containing integer pairs for multiplication.
  - Types: Positive, negative, and mixed integers of varying digit lengths.

---  
## DEMO
https://github.com/user-attachments/assets/cd0ad0a3-ca53-4f11-9704-25196fb6f22d  

https://github.com/user-attachments/assets/c0f78062-1686-4de6-a4cd-6d3edda14b3a    



## How to Use

### Prerequisites
- Install the required Python libraries:
  ```bash
  pip install customtkinter matplotlib networkx
### Running the Application
  ## Clone the repository:
         git clone <repository-url>
         cd algorithm-visualizer-daa
## Run the main Python file:
         python algorithm_visualizer.py
