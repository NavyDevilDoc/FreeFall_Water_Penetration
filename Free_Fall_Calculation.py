
## Free-fall Calculation ##

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D


def calculate_impact_velocity_and_time(mass, start_altitude, altitudes, time_step):
    # Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)
    air_density_values = 1.225 * np.exp(-altitudes / 8500)  # Air density values at given altitudes (kg/m^3)
    drag_coefficient = 0.85  # Drag coefficient for a human body
    area = 1.75  # Cross-sectional area of a human body (m^2)
    drag_force_constant = 0.5 * air_density_values * drag_coefficient * area

    # Initialize variables
    velocity = 0
    height = start_altitude
    time = 0
    times = deque([0])
    velo = deque([0])
    time_of_flight = 0  # Initialize time of flight

    # Calculate velocity and height at each time step
    while height > 0:
        altitude_idx = np.argmin(np.abs(altitudes - height))
        drag_force = drag_force_constant[altitude_idx] * velocity ** 2
        net_force = mass * g - drag_force
        acceleration = net_force / mass

        velocity += acceleration * time_step
        height -= velocity * time_step
        time += time_step
        time_of_flight += time_step  # Increment time of flight
        times.append(time)
        velo.append(velocity)

    return velocity, np.array(times), np.array(velo), time_of_flight

def water_penetration_velocity(mass, start_altitude, altitudes, time_step):
    # Constants
    water_density = 1000  # Density of water (kg/m^3)
    drag_coefficient_water = 0.8  # Drag coefficient for a human body in water
    area = 1.75  # Cross-sectional area of a human body (m^2)
    g = 9.81  # Acceleration due to gravity (m/s^2)

    # Calculate impact velocity
    impact_velocity, _, _, _ = calculate_impact_velocity_and_time(mass, start_altitude, altitudes, time_step)

    # Initialize variables
    velocity = impact_velocity
    depth = 0
    velocities = deque([velocity])
    depths = deque([depth])

    # Calculate velocity at different depths during penetration
    while velocity > 0:
        drag_force_water = 0.5 * water_density * drag_coefficient_water * area * velocity ** 2
        net_force_water = mass * g + drag_force_water
        acceleration_water = -net_force_water / mass

        velocity += acceleration_water * time_step
        depth += -velocity * time_step

        velocities.append(velocity)
        depths.append(depth)

    return np.array(depths), np.array(velocities)

def plot_velocity_vs_depth(mass, start_altitude, altitude, time_step, plt):
    depths, velocities = water_penetration_velocity(mass, start_altitude, altitude, time_step)

    plt.plot(velocities, depths)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.title('Depth vs. Velocity During Water Penetration')
    plt.grid()
    # Reverse the x-axis
    plt.gca().invert_xaxis()
    plt.show()

    max_depth = np.abs(np.min(depths))

    return max_depth

def plot_velocity_vs_time(mass, start_altitude, altitude, time_step, plt):
    _, times, velo, _ = calculate_impact_velocity_and_time(mass, start_altitude, altitude, time_step)

    plt.plot(times, velo)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs. Time Prior to Water Penetration')
    plt.grid()
    #plt.gca().invert_xaxis()
    plt.show()

def plot_velocity_vs_altitude(mass, start_altitude, altitudes, time_step, plt):
    _, times, velo, _ = calculate_impact_velocity_and_time(mass, start_altitude, altitudes, time_step)
    heights = start_altitude - np.cumsum(velo * time_step)

    plt.plot(heights, velo)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs. Altitude From Jump Point to Water Penetration')
    plt.grid()
    plt.gca().invert_xaxis()
    plt.show()

def plot_velocity_altitude_time(mass, start_altitude, altitudes, time_step, plt):
    _, times, velo, _ = calculate_impact_velocity_and_time(mass, start_altitude, altitudes, time_step)
    heights = start_altitude - np.cumsum(velo * time_step)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(times, velo, heights)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Velocity, Altitude, and Time')

    ax.view_init(elev=30, azim=135)  # Adjust the viewing angle

    plt.show()

def main():
    mass = 100  # Mass of the person (kg)
    time_step = 0.001
    # Define the range of altitudes you want to model
    start_altitude = 20  # Starting altitude (m)
    end_altitude = 0  # Ending altitude (m)
    num_points = 1000  # Number of points in the range
    
    # Generate a linearly spaced vector of altitudes
    altitudes = np.linspace(start_altitude, end_altitude, num_points)
    
    impact_velocity, _, _, time_of_flight = calculate_impact_velocity_and_time(mass, start_altitude, altitudes, time_step)
    max_depth = plot_velocity_vs_depth(mass, start_altitude, altitudes, time_step, plt)
    plot_velocity_vs_time(mass, start_altitude, altitudes, time_step, plt)
    plot_velocity_vs_altitude(mass, start_altitude, altitudes, time_step, plt)
    plot_velocity_altitude_time(mass, start_altitude, altitudes, time_step, plt)
    kinetic_energy = 0.5 * mass * impact_velocity ** 2
    
    print(f"Water penetration depth: {max_depth:.2f} m")
    print(f"Velocity upon impact with the water: {impact_velocity:.2f} m/s")
    print(f"Kinetic energy upon impact: {kinetic_energy:.2f} J")
    print(f"Time of flight: {time_of_flight:.2f} s")

if __name__ == "__main__":
    main() 
