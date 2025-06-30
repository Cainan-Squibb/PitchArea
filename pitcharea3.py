import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import csv
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk
# Some future thoughts on this.  This model does not value throwing harder but rather having separation this is not necessarily good
# pitchers with the Fastball/Off Speed velocities of 95/85 and 85/75 are viewed the same which they are not
# This model favors pitchers who get vertical break more than horizontal.  In my experience, most pitchers will get natural
# horizontal movement on their pitches, this is not the same for vertical movement.  So, if a pitcher mostly throws across the zone 
# pitches, this model will undervalue them because they don't have the natural vertical presense to increase the area.
# This model may also undervalue pitchers with fewer pitches because there isn't as many points to expand the area with
# Depending on how this is modified in the future, this may be a model to predict how effective a starting pitcher will be, since
# starting pitchers tend to need more pitches because they throw longer than relievers

# Directory containing CSV files
base_folder = r"C:\Users\Cainan Squibb\OneDrive - UWM\Desktop\Baseball Analytics\UWM Pitchers CSVs"

def load_pitch_data(csv_path):
    """Load pitch data from a given CSV file, dynamically handling metadata."""
    
    with open(csv_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    start_idx = None
    for i, line in enumerate(lines):
        if "Pitch ID" in line and "Velocity" in line:
            start_idx = i
            break
    
    if start_idx is None:
        print(f"Skipping {csv_path}: Could not locate column headers.")
        return None

    df = pd.read_csv(csv_path, skiprows=start_idx, encoding="utf-8", quoting=csv.QUOTE_ALL)
    df = df.rename(columns=lambda x: x.strip())
    
    df.replace("-", np.nan, inplace=True)
    
    num_cols = ["HB (trajectory)", "VB (trajectory)", "Velocity"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    column_mapping = {
        "HB (trajectory)": "Horizontal Break",
        "VB (trajectory)": "Vertical Break",
        "Velocity": "Velocity"
    }
    df.rename(columns=column_mapping, inplace=True)

    required_cols = {"Pitch Type", "Horizontal Break", "Vertical Break", "Velocity"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Skipping {csv_path} due to missing required columns: {missing_cols}")
        return None

    df = df.dropna(subset=["Pitch Type", "Velocity"])
    
    return df[["Pitch Type", "Horizontal Break", "Vertical Break", "Velocity"]]

def shoelace_area(points):
    """Calculate the area enclosed by a set of (x, y) points using the Shoelace Theorem."""
    points = np.array(points)
    points = np.vstack([points, points[0]])  # Close the polygon
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def process_pitchers(base_folder):

    pitcher_data = {}
    
    for file in os.listdir(base_folder):
        if file.endswith(".csv"):
            pitcher_name = file.replace(".csv", "")
            file_path = os.path.join(base_folder, file)
            df = load_pitch_data(file_path)
            
            if df is None or df.empty:
                print(f"Skipping {pitcher_name}: No valid data.")
                continue
            
            # Filter to include only pitch types with more than 30 pitches
            pitch_counts = df["Pitch Type"].value_counts()
            df = df[df["Pitch Type"].isin(pitch_counts[pitch_counts >= 30].index)]
            
            # Compute average Horizontal Break, Vertical Break, and Velocity for each pitch type
            avg_pitches = df.groupby("Pitch Type", as_index=False).mean()
            
            # Only use the average points to form the convex hull
            all_points = list(zip(avg_pitches["Horizontal Break"], avg_pitches["Vertical Break"]))
            
            # Calculate the average velocity per pitch type
            avg_velocity_by_pitch = df.groupby("Pitch Type")["Velocity"].mean()
            
            # Calculate the velocity difference between average velocities
            if len(avg_velocity_by_pitch) > 1:
                velocity_diff = avg_velocity_by_pitch.max() - avg_velocity_by_pitch.min()
            else:
                velocity_diff = 0  # If there's only one pitch type, no difference
            
            if len(all_points) > 2:
                hull = ConvexHull(all_points)
                hull_points = np.array(all_points)[hull.vertices]
                area = shoelace_area(hull_points)
                
                pitcher_data[pitcher_name] = {
                    "df": df,
                    "avg_points": all_points,
                    "hull_points": hull_points,
                    "area": area,
                    "velocity_diff": velocity_diff,
                    "scaled_area": area * velocity_diff
                }
    
    return pitcher_data

def plot_pitch_shapes_with_arrows(pitcher_data):
    pitcher_names = list(pitcher_data.keys())  # Get pitcher names to navigate through
    current_index = [0]  # Using list to allow mutability (to modify within the lambda)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Function to update the plot
    def update_plot(index):
        ax.clear()  # Clear previous plot
        pitcher = pitcher_names[index]
        data = pitcher_data[pitcher]
        df = data["df"]
        
        # Get pitch types and assign colors
        pitch_types = df["Pitch Type"].unique()
        colors = sns.color_palette("husl", n_colors=len(pitch_types))
        color_map = dict(zip(pitch_types, colors))

        # Store coordinates for the legend
        legend_entries = []

        # Plot only the average point for each pitch type
        for pitch_type in pitch_types:
            pitch_data = df[df["Pitch Type"] == pitch_type]
            avg_hb = pitch_data["Horizontal Break"].mean()
            avg_vb = pitch_data["Vertical Break"].mean()
            ax.scatter(
                avg_hb, avg_vb,
                label=f"{pitch_type}", color=color_map[pitch_type], s=100
            )
            
            # Store coordinate info for the legend
            legend_entries.append(f"{pitch_type}: ({avg_hb:.2f}, {avg_vb:.2f})")
        
        avg_pitches = df.groupby("Pitch Type", as_index=False).mean()
        all_points = list(zip(avg_pitches["Horizontal Break"], avg_pitches["Vertical Break"]))
        
        if len(all_points) > 2:
            hull = ConvexHull(all_points)
            hull_points = np.array(all_points)[hull.vertices]
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'k-', lw=2, color='black')
            ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color='black')

            # Ensure the last line connects to the first point to close the hull
            ax.plot([hull_points[-1, 0], hull_points[0, 0]], [hull_points[-1, 1], hull_points[0, 1]], 'k-', lw=2, color='black')
        
        # Calculate bounds of the convex hull to prevent text overlap
        x_min, x_max = np.min(hull_points[:, 0]), np.max(hull_points[:, 0])
        y_min, y_max = np.min(hull_points[:, 1]), np.max(hull_points[:, 1])

        # Set axis limits to zoom out a bit more
                # Set axis limits to zoom out a bit more
        margin = 15  # Increased margin by 5 more units
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)


        # Add text box with area, velocity diff, and scaled area
        text_str = (f"Area: {data['area']:.2f}\n"
                    f"Velocity Diff: {data['velocity_diff']:.2f}\n"
                    f"Scaled Area: {data['scaled_area']:.2f}")
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        
        # Set labels and title
        ax.set_xlabel("Horizontal Break")
        ax.set_ylabel("Vertical Break")
        ax.set_title(f"Pitch Shape for {pitcher}")
        ax.grid(True)

        # Create legend with coordinates added
        legend_title = "Pitch Type"
        ax.legend(title=legend_title, loc="upper left", labels=legend_entries)

        plt.draw()

    # Initially plot the first pitcher
    update_plot(current_index[0])

    # Callback function for the forward button
    def on_forward_click(event):
        current_index[0] = (current_index[0] + 1) % len(pitcher_names)
        update_plot(current_index[0])

    # Callback function for the backward button
    def on_backward_click(event):
        current_index[0] = (current_index[0] - 1) % len(pitcher_names)
        update_plot(current_index[0])

    # Add forward and backward buttons
    ax_back = plt.axes([0.1, 0.02, 0.1, 0.075])
    ax_forward = plt.axes([0.8, 0.02, 0.1, 0.075])
    
    button_back = Button(ax_back, '<< Previous', color='lightgoldenrodyellow', hovercolor='orange')
    button_forward = Button(ax_forward, 'Next >>', color='lightgoldenrodyellow', hovercolor='orange')

    button_back.on_clicked(on_backward_click)
    button_forward.on_clicked(on_forward_click)

    plt.show()

def display_sorted_pitchers(pitcher_data):
    """Display a sorted list of pitchers based on their scaled area in a pop-up window."""
    # Sort pitchers by their scaled area in descending order
    sorted_pitchers = sorted(pitcher_data.items(), key=lambda x: x[1]['scaled_area'], reverse=True)

    # Create the Tkinter window
    window = tk.Tk()
    window.title("Pitchers Ordered by Scaled Area")
    
    # Create the table (frame + labels)
    frame = ttk.Frame(window)
    frame.pack(padx=10, pady=10)

    # Title row
    title_label = ttk.Label(frame, text="Pitcher Name", width=25, anchor='w')
    title_label.grid(row=0, column=0, padx=5, pady=5)
    area_label = ttk.Label(frame, text="Scaled Area", width=15, anchor='w')
    area_label.grid(row=0, column=1, padx=5, pady=5)

    # Add each pitcher to the table
    for i, (pitcher, data) in enumerate(sorted_pitchers, 1):
        pitcher_name_label = ttk.Label(frame, text=pitcher, width=25, anchor='w')
        pitcher_name_label.grid(row=i, column=0, padx=5, pady=5)
        area_value_label = ttk.Label(frame, text=f"{data['scaled_area']:.2f}", width=15, anchor='w')
        area_value_label.grid(row=i, column=1, padx=5, pady=5)
    
    # Add a button to close the window
    close_button = ttk.Button(window, text="Close", command=window.destroy)
    close_button.pack(pady=10)

    # Run the Tkinter event loop
    window.mainloop()

# Run the processing and plotting
pitcher_data = process_pitchers(base_folder)
if pitcher_data:
    plot_pitch_shapes_with_arrows(pitcher_data)
    display_sorted_pitchers(pitcher_data)
else:
    print("No valid pitch data found.")
