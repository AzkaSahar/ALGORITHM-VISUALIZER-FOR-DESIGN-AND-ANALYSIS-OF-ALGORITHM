import math
import os
from time import sleep
from tkinter import filedialog, messagebox
from customtkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading  
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import uuid
import ctypes
from matplotlib.patches import FancyBboxPatch



class App(CTk):
    def __init__(self):
        super().__init__()
        self.title("Algorithm Visualizer")
        self.geometry("800x600")
        self.attributes('-fullscreen', True)
        self.configure(bg="black")
        self.bind("<Escape>", self.toggle_fullscreen)

        # Sidebar 
        self.sidebar_frame = CTkFrame(self, width=200,fg_color="#1a1a1a", border_color="#ff0000", border_width=2)
        self.sidebar_frame.pack(side="left", expand=True, fill="both", padx=0, pady=10)

        # Plot area (fills the remaining space)
        self.plot_frame = CTkFrame(self,fg_color="black")
        self.plot_frame.pack(expand=True, fill="both", pady=10)

        # Style for buttons and labels 
        button_style = {
            "fg_color": "black",
            "text_color": "#ff0000",
            "hover_color": "#990000",
            "corner_radius": 5,
            "font": ("Consolas", 12, "bold"),
            "border_color": "#ff0000",
            "border_width": 2
        }

        radio_button_style = {
            "fg_color": "#ff0000",      # Red color for the selected state
            "hover_color": "#990000",   # Dark red for hover state
            "text_color": "#ff0000",     
            "font": ("Consolas", 11, "bold")  
        }


        label_style = {
            "text_color": "#ff0000",
            "font": ("Consolas", 11, "bold")
        }

        # Algorithm selection radio buttons in the sidebar
        self.algo_var = IntVar(value=1)
        self.algo_var.trace_add("write", self.on_algorithm_change)

        self.closest_pair_rb = CTkRadioButton(self.sidebar_frame, text="Closest Pair of Points", variable=self.algo_var, value=1, **radio_button_style)
        self.closest_pair_rb.pack(pady=(250, 5))
        self.int_mult_rb = CTkRadioButton(self.sidebar_frame, text="Integer Multiplication", variable=self.algo_var, value=2, **radio_button_style)
        self.int_mult_rb.pack(pady=(5, 20))

        # File selection button in the sidebar
        self.file_button = CTkButton(self.sidebar_frame, text="Select Input File", command=self.select_file,**button_style)
        self.file_button.pack(pady=10)
        self.file_label = CTkLabel(self.sidebar_frame, text="No file selected",**label_style)
        self.file_label.pack(pady=5)

        # Run algorithm and exit buttons in the sidebar
        self.run_button = CTkButton(self.sidebar_frame, text="Run Algorithm", command=self.run_algorithm,**button_style )
        self.run_button.pack(pady=10)
        self.exit_button = CTkButton(self.sidebar_frame, text="Exit", command=self.quit,**button_style )
        self.exit_button.pack(pady=5)

        # Result text in the sidebar
        self.result_text = CTkLabel(self.sidebar_frame, text="Result: N/A", **label_style )
        self.result_text.pack(pady=(30,10))

        # Initialize 
        self.selected_file = None
        self.points = []
        self.x = []
        self.y = []
        self.ax = None
        self.canvas = None
        self.tree_graph = None
        self.plot_points()

    def toggle_fullscreen(self, event=None):
        current_state = self.attributes('-fullscreen')
        self.attributes('-fullscreen', not current_state)

    def on_algorithm_change(self, *args):
        selected_algo = self.algo_var.get()
        if selected_algo == 2:
            self.draw_empty_tree()
        elif selected_algo == 1:
            self.plot_points()

    def draw_empty_tree(self):
        """Draws an empty tree for the Integer Multiplication algorithm"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.tree_graph = nx.Graph()
        self.tree_graph.add_node("Root", label="Start")
        self.tree_graph.add_node("Node1", label="x1 x y1")
        self.tree_graph.add_node("Node2", label="x2 x y2")
        self.tree_graph.add_edge("Root", "Node1")
        self.tree_graph.add_edge("Root", "Node2")

        pos = graphviz_layout(self.tree_graph, prog="dot")
        fig, ax = plt.subplots(figsize=(11.8, 10),facecolor="black")
        ax.set_facecolor("black")  # Set plot area background to black
        nx.draw(self.tree_graph, pos, with_labels=False, node_size=5000, node_color="#db0b0b", font_size=12,font_color="white", edge_color="white", ax=ax)
        nx.draw_networkx_labels(self.tree_graph, pos, labels=nx.get_node_attributes(self.tree_graph, 'label'), font_size=10, font_color="white", ax=ax)

        fig.patch.set_facecolor("black")  
        ax.patch.set_facecolor("black")  

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_algorithm(self):
        """Run the selected algorithm based on the radio button selection."""
        if not self.selected_file:
            messagebox.showwarning("File Required", "Please select an input file.")
            return

        selected_algo = self.algo_var.get()
        if selected_algo == 1:
            self.result_text.configure(text="Running Algorithm...")
            algorithm_thread = threading.Thread(target=self.run_closest_pair)
            algorithm_thread.start()
        elif selected_algo == 2:
            self.result_text.configure(text="Running Algorithm...")
            algorithm_thread = threading.Thread(target=self.run_karatsuba_visualization)
            algorithm_thread.start()



    def on_multiply(self):
        """Handles the multiplication and visualizes the recursion"""
        if not self.x or not self.y:
            messagebox.showwarning("File Required", "Please select a file for Integer Multiplication.")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        nodesizes = []  # Internal list for node sizes

        # Create a directed graph for visualization
        G = nx.DiGraph()

        # Build Karatsuba tree for each pair in x and y
        for num1, num2 in zip(self.x, self.y):
            self.karatsuba(num1, num2, G, nodesizes)

        formatted_labels = self.format_labels(G)
        fig, ax = plt.subplots(figsize=(11.8, 10))
        pos = graphviz_layout(G, prog="dot")
        
        # Draw with formatted labels
        nx.draw(G, pos, with_labels=False, node_size=nodesizes, node_color="skyblue", font_size=8, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=8, ax=ax)
        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)



    def run_karatsuba_visualization(self):
        """Run Karatsuba multiplication on all pairs and visualize the first pair."""
        if not self.x or not self.y:
            messagebox.showwarning("File Required", "Please select a file for Integer Multiplication.")
            return

        products = []
        G = nx.DiGraph()
        nodesizes = []

        # Iterate over all pairs
        for idx, (num1, num2) in enumerate(zip(self.x, self.y)):
            if idx == 0:
                # For the first pair, build the graph
                self.karatsuba(num1, num2, G, nodesizes)
            else:
                # For other pairs, compute the product without building the graph
                result = self.karatsuba(num1, num2, G=None, nodesizes=None, build_graph=False)
            products.append((num1, num2, num1 * num2))

        # Pass the visualization and products to the main thread using `after`
        self.after(0, self.display_karatsuba_graph, G, nodesizes, products)


    def display_karatsuba_graph(self, G, nodesizes, products):
        """Display the Karatsuba graph on the main thread."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if G is not None and nodesizes:
            formatted_labels = self.format_labels(G)
            fig, ax = plt.subplots(figsize=(11.8, 10),facecolor="black")  # Set figure background to black
            ax.set_facecolor("black")  # Set plot area background to black
            pos = graphviz_layout(G, prog="dot")
            
            nx.draw_networkx_nodes(G, pos, node_size=[size + 500 for size in nodesizes],node_color="#8B0000",  ax=ax)
            nx.draw(G, pos, with_labels=False, node_size=nodesizes, node_color="#db0b0b", font_size=8, font_color="black",edge_color="white",ax=ax)
            nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=8,font_color="black", ax=ax)
            fig.patch.set_facecolor("black")  
            ax.patch.set_facecolor("black") 
            plt.tight_layout()

            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Display the computed products after 5 seconds
        self.after(5000, self.show_products, products)


    def karatsuba(self, x, y, G=None, nodesizes=None, parent=None, depth=0, build_graph=True):
        """Recursive Karatsuba function with visualization for handling negative values correctly"""
        result = x * y
        if build_graph and G is not None and nodesizes is not None:
            unique_id = uuid.uuid4()
            node_label = f"{x} * {y} = {result} (depth {depth}, id {unique_id})"
            display_label = f"{x} * {y} = {result}"
            nodesizes.append(4000 if depth == 0 else 2000 if depth > 1 else 4000)

            G.add_node(node_label, label=display_label)
            if parent:
                G.add_edge(parent, node_label)

        # Handle base case
        if abs(x) < 10 or abs(y) < 10:
            return result

        # Split the numbers and calculate Karatsuba components
        n = max(len(str(abs(x))), len(str(abs(y))))
        half = n // 2
        high_x, low_x = divmod(abs(x), 10**half)
        high_y, low_y = divmod(abs(y), 10**half)

        # Recursive calls
        z0 = self.karatsuba(low_x, low_y, G, nodesizes, parent=node_label if build_graph else None, depth=depth + 1, build_graph=build_graph)
        z1 = self.karatsuba((low_x + high_x), (low_y + high_y), G, nodesizes, parent=node_label if build_graph else None, depth=depth + 1, build_graph=build_graph)
        z2 = self.karatsuba(high_x, high_y, G, nodesizes, parent=node_label if build_graph else None, depth=depth + 1, build_graph=build_graph)

        # Combine results
        if (x < 0) ^ (y < 0):  
            result = -(z0 + (z1 - z0 - z2) * 10**half + z2 * 10**(2*half))
        else:
            result = z0 + (z1 - z0 - z2) * 10**half + z2 * 10**(2*half)

        return result

    
    def show_products(self, products):
        """Clear the tree visualization and display each product label one at a time with a delay."""
        # Clear the current canvas
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        column_count = 4
        rows_per_column = math.ceil(len(products) / column_count)

    
        fig, ax = plt.subplots(figsize=(11.8, 10), facecolor="black")
        ax.axis("off")  # Turn off the axes for a clean look

        # Define colors 
        product_box_color = "#db0b0b"  # Bright red for product boxes
        text_color = "black"           # Black text for visibility on red background
        shadow_color = "#bab8b8"       # Shadow effect
        border_color = "#8B0000"       # Dark red for the border
        font_size = 8

        
        x_positions = [0.05, 0.35, 0.65, 0.95]
        y_positions = [1 - i * (1.07 / (rows_per_column - 1)) for i in range(rows_per_column)]

       
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initialize an index to keep track of the current product
        self.current_product_index = 0

        # Define a function to display each product box with a delay
        def display_next_box():
            # Check if there are more products to display
            if self.current_product_index < len(products):
                # Retrieve the product details based on the current index
                num1, num2, product = products[self.current_product_index]
                product_text = f"{num1} × {num2} = {product}"
                column = self.current_product_index // rows_per_column  # Determine column based on index
                row_position = y_positions[self.current_product_index % rows_per_column]

                # Draw shadow box slightly offset behind the main box
                ax.text(
                    x_positions[column] + 0.003, row_position - 0.001, product_text, ha="center", va="center",
                    fontsize=font_size, color=shadow_color, bbox=dict(
                        boxstyle="round,pad=0.3,rounding_size=0.8", 
                        facecolor=shadow_color, edgecolor="none"
                    ),
                    transform=ax.transAxes
                )

                # Draw main text box with dark red border
                ax.text(
                    x_positions[column], row_position, product_text, ha="center", va="center",
                    fontsize=font_size, color=text_color, bbox=dict(
                        boxstyle="round,pad=0.3,rounding_size=0.7", 
                        facecolor=product_box_color, edgecolor=border_color, linewidth=1.0
                    ),
                    transform=ax.transAxes
                )

                # Update the canvas to display the new box
                self.canvas.draw()

                # Increment the index to move to the next product
                self.current_product_index += 1

                # Schedule the next product to display after a delay
                self.after(100, display_next_box)

        # Start displaying boxes one by one
        display_next_box()


    def format_labels(self, G):
        """Formats the node labels into multiple lines to fit inside the circles"""
        formatted_labels = {}
        for node in G.nodes:
            label = G.nodes[node]["label"]
            if len(label) > 10:
                formatted_labels[node] = "\n".join([label[i:i+10] for i in range(0, len(label), 10)])
            else:
                formatted_labels[node] = label
        return formatted_labels

    def select_file(self):
        selected_algo = self.algo_var.get()
        
        if selected_algo == 1:
            directory = "./closest_pair_inputs/"
        elif selected_algo == 2:
            directory = "./integer_multiplication_inputs/"
            self.draw_empty_tree()
        else:
            return

        self.selected_file = filedialog.askopenfilename(initialdir=directory, title="Select Input File", filetypes=[("Text files", "*.txt")])
        
        if self.selected_file:
            self.file_label.configure(text=os.path.basename(self.selected_file))
            if selected_algo == 1:
                self.load_points_c()
                self.plot_points()
            elif selected_algo == 2:
                self.load_points_k()

    def load_points_c(self):
        """Loads points from the selected file into self.points"""
        self.points = []
        if self.selected_file:
            with open(self.selected_file, 'r') as f:
                for line in f:
                    x, y = map(float, line.strip().split())
                    self.points.append((x, y))

    def load_points_k(self):
        """Loads integer multiplication points from file into x and y arrays"""
        self.x = []
        self.y = []
        try:
            with open(self.selected_file, 'r') as file:
                for line in file:
                    point = line.split()
                    if len(point) == 2:
                        x_val = int(point[0])
                        y_val = int(point[1])
                        self.x.append(x_val)
                        self.y.append(y_val)
        except Exception as e:
            print(f"Error reading file: {e}")

    def city_block_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def closest_pair_util(self, points_sorted_x, points_sorted_y, ax, depth=0):
        """Recursive function to find the closest pair with median line plotting and delay"""
        if len(points_sorted_x) <= 3:
            min_distance = float('inf')
            closest_pair = None
            for i in range(len(points_sorted_x)):
                for j in range(i + 1, len(points_sorted_x)):
                    dist = self.city_block_distance(points_sorted_x[i], points_sorted_x[j])
                    if dist < min_distance:
                        min_distance = dist
                        closest_pair = (points_sorted_x[i], points_sorted_x[j])
            return min_distance, closest_pair

        mid = len(points_sorted_x) // 2
        mid_point = points_sorted_x[mid]

        # Plot the median line (vertical line at the median point)
        ax.axvline(x=mid_point[0], color='#ff0000', linestyle='--', linewidth=1)
        self.canvas.draw()

        # Delay to simulate algorithm progress
        self.after(200)

        # Split the points into two halves
        left_sorted_x = points_sorted_x[:mid]
        right_sorted_x = points_sorted_x[mid:]
        left_sorted_y = [p for p in points_sorted_y if p[0] <= mid_point[0]]
        right_sorted_y = [p for p in points_sorted_y if p[0] > mid_point[0]]

        # recursion
        min_distance_left, closest_left = self.closest_pair_util(left_sorted_x, left_sorted_y, ax, depth + 1)
        min_distance_right, closest_right = self.closest_pair_util(right_sorted_x, right_sorted_y, ax, depth + 1)

        # Combine results
        d = min(min_distance_left, min_distance_right)
        closest_pair = closest_left if min_distance_left < min_distance_right else closest_right

        # Merge step: Check points within the strip
        Sy = [p for p in points_sorted_y if abs(p[0] - mid_point[0]) < d]
        for i in range(len(Sy)):
            for j in range(i + 1, min(i + 15, len(Sy))):
                dist = self.city_block_distance(Sy[i], Sy[j])
                if dist < d:
                    d = dist
                    closest_pair = (Sy[i], Sy[j])

        return d, closest_pair

    def closest_pair(self, points):
        """Finds the closest pair using divide and conquer with recursion"""
        points_sorted_x = sorted(points, key=lambda x: x[0])
        points_sorted_y = sorted(points, key=lambda x: x[1])
        return self.closest_pair_util(points_sorted_x, points_sorted_y, self.ax)

    def plot_points(self):
        x_coords, y_coords = zip(*self.points) if self.points else ([], [])

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(11.8, 10), facecolor="black")
        ax.set_facecolor("black")
        ax.scatter(x_coords, y_coords, color='red', s=14)
        ax.set_xlabel("X-axis", fontsize=12, fontname="Montserrat",color="white")
        ax.set_ylabel("Y-axis", fontsize=12, fontname="Montserrat",color="white")
        ax.set_title("Closest Pair of Points", fontsize=14, fontname="Montserrat",color="white")
        ax.set_aspect('auto')
        if self.points:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            padding = 1
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)

        '''ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)'''

        ax.tick_params(axis='x', colors='white')  # X-axis tick color
        ax.tick_params(axis='y', colors='white')  # Y-axis tick color
    

        ax.axhline(0, color='white', linewidth=0.5, ls='--')
        ax.axvline(0, color='white', linewidth=0.5, ls='--')
        ax.grid(True)

        fig.tight_layout()

        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_closest_pair(self):
        """Run the closest pair algorithm and update the result text."""
        min_distance, closest_points = self.closest_pair(self.points)

        # Update result text with closest pair and distance (must be done in the main thread)
        self.result_text.configure(text=f"Closest Pair: \n{closest_points} \n\n Distance: \n{min_distance:.2f}")

        # Highlight the closest pair in red (also must be done in the main thread)
        if closest_points:
            self.ax.scatter([closest_points[0][0], closest_points[1][0]], 
                         [closest_points[0][1], closest_points[1][1]], 
                         color='white', linewidth=3)
            self.ax.plot([closest_points[0][0], closest_points[1][0]], 
                         [closest_points[0][1], closest_points[1][1]], 
                         color='white', linewidth=3)
            self.canvas.draw()



app = App()
app.mainloop()
