import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
'''
# Function to find or predict the first intersection point
def find_or_predict_intersection_point(x_values, y_values, y_intersection, degree=2, num_points=None):
    for i in range(1, len(x_values)):
        if (y_values[i-1] <= y_intersection <= y_values[i]) or (y_values[i] <= y_intersection <= y_values[i-1]):
            x_intersect = np.interp(y_intersection, [y_values[i-1], y_values[i]], [x_values[i-1], x_values[i]])
            return x_intersect, None
    # If no intersection found, use polynomial regression to predict the intersection
    if num_points is None or num_points > len(x_values):
        num_points = len(x_values)
    x_fit = x_values[-num_points:]
    y_fit = y_values[-num_points:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p = Polynomial.fit(x_fit, y_fit, degree)
    coefs = p.convert().coef
    coefs[-1] -= y_intersection
    roots = np.roots(coefs)
    for root in roots:
        if np.isreal(root) and root > x_values[-1]:
            predicted_intersection = np.real(root)
            return predicted_intersection, p
    return None, p

def find_max_min_in_window(x, y, window_size):
    """Find maximum and minimum values within a specified window in the given data."""
    max_values = []
    min_values = []
    
    for i in range(len(x) - window_size + 1):
        window_y = y[i:i + window_size]
        max_values.append(np.max(window_y))
        min_values.append(np.min(window_y))
        
    # Extend max/min lists to align with original x values for plotting
    max_values = np.pad(max_values, (window_size - 1, 0), mode='edge')
    min_values = np.pad(min_values, (window_size - 1, 0), mode='edge')

    return max_values, min_values


# Data keys and probability levels
data_keys = [
#    '5',
    '10',
#            '15',
    '20',
    '25'
        ]
prob_labels = ['10%', '25%', '50%', '75%']
probabilities = ['Tenper', 'Twentyfiveper', 'Fiftyper', 'Seventyfiveper']
limits = [0.004,0.005, 0.01, 0.02]
colors = {
    '10%': ['green','blue','#FF0000', '#FF00FF', '#A52A2A'],
    '25%': ['orange','red','#00FF00', '#00FFFF', '#008000'],
    '50%': ['red','green','#0000FF', '#FFA500', '#000080'],
    '75%': ['blue','orange','#000000', '#800080', '#FFC0CB']
}
data_files = {
    '25': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data.mat',
    '20': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_20.mat',
#    '15': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_15.mat',
    '10': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_10.mat',
#    '5': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_5.mat'
}


# Iterate over each probability level
for prob_index, prob_label in enumerate(prob_labels):
    plt.figure(figsize=(12, 8))
    intersection_data = []

    # Loop through each data keys
    for key in data_keys:
        file = data_files[key]
        data = loadmat(file)
        MidPointIndexAll = data['MidPointIndexAll'].flatten()
        prob = probabilities[prob_index]
        
        curve_values = data[f'curve{prob}maxResistanceValueAll'].flatten()
        
        # Filter out values below 0.03
        curve_values_filtered = curve_values[curve_values > 0.0]
        MidPointIndexAll_filtered = MidPointIndexAll[curve_values > 0.0]
        
        # Apply LOWESS smoothing to filtered data
        lowess_smoothed = lowess(curve_values_filtered, MidPointIndexAll_filtered, frac=0.09)[:, 1]
        
        # Find max and min values in a specified window size
        window_size = 100  # Adjust this as needed
        max_curve_values, min_curve_values = find_max_min_in_window(MidPointIndexAll_filtered, lowess_smoothed, window_size)
    
        # Plot original data points and smoothed curve
        plt.semilogy(MidPointIndexAll_filtered, curve_values_filtered, '.', markersize=6, color=colors[prob_label][data_keys.index(key)], label=f'{key}µm ')
        plt.semilogy(MidPointIndexAll_filtered, lowess_smoothed, '-', linewidth=2.5, color=colors[prob_label][data_keys.index(key)])
        
        for YIntersectionPoint in limits:
            # Calculate and plot intersection point or prediction
            intersection_point, poly = find_or_predict_intersection_point(MidPointIndexAll_filtered, lowess_smoothed, YIntersectionPoint, degree=2, num_points=len(MidPointIndexAll_filtered))
            
            if intersection_point is not None:
                plt.scatter(intersection_point, YIntersectionPoint, color='black', marker='o', s=25)
            else:
                # Plot extrapolated polynomial curve
                x_extrap = np.linspace(MidPointIndexAll_filtered[-1], MidPointIndexAll_filtered[-1] * 25, 100)
                y_extrap = poly(x_extrap)
                plt.plot(x_extrap, y_extrap, '--', color=colors[prob_label][data_keys.index(key)])
                
                # Find and mark extrapolated intersection point
                intersection_point, _ = find_or_predict_intersection_point(x_extrap, y_extrap, YIntersectionPoint, degree=2, num_points=len(MidPointIndexAll_filtered))
                if intersection_point is not None:
                    plt.scatter(intersection_point, YIntersectionPoint, color='red', marker='x', s=25)
                
                # Polynomial fits for max and min curves
                poly_max = Polynomial.fit(MidPointIndexAll_filtered, max_curve_values, 2)
                poly_min = Polynomial.fit(MidPointIndexAll_filtered, min_curve_values, 2)
                
                # Generate x values for plotting
                x_vals_fit = np.linspace(MidPointIndexAll_filtered[0], MidPointIndexAll_filtered[-1] * 25, 100)
                y_vals_max_fit = poly_max(x_vals_fit)
                y_vals_min_fit = poly_min(x_vals_fit)
                
                # Plot max and min trend lines
                plt.plot(x_vals_fit, y_vals_max_fit, '--', color='red', label='Max Trend')  # Adjust color as needed
                plt.plot(x_vals_fit, y_vals_min_fit, '--', color='blue', label='Min Trend')    # Adjust color as needed

            # Collect intersection data
            intersection_data.append((key, prob_label, YIntersectionPoint, intersection_point))
            
            # Count points above the limit before intersection
            if intersection_point is not None:
                count_above_limit = sum(curve_values_filtered[:np.argmax(MidPointIndexAll_filtered > intersection_point)] > YIntersectionPoint)
            else:
                count_above_limit = sum(curve_values_filtered > YIntersectionPoint)
            
            print(f'Probability: {prob_label}, Amplitude: {key}µm, Count above {YIntersectionPoint}Ω before intersection: {count_above_limit}')

    
# Final plot adjustments for each probability level
    plt.xlabel('Number of Cycles')
    plt.ylabel('Resistance (Ohm)')
    plt.title(f'Wöhler Curves with {prob_label} Probability')
    for limit in limits:
        plt.axhline(limit, color='black', linestyle='-', linewidth='2')
    plt.ylim(0, 10)
    plt.xlim(0,1000000)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
'''

##################################### Alltogether

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess

# Function to find or predict the first intersection point
def find_or_predict_intersection_point(x_values, y_values, y_intersection, degree=2, num_points=None):
    for i in range(1, len(x_values)):
        if (y_values[i-1] <= y_intersection <= y_values[i]) or (y_values[i] <= y_intersection <= y_values[i-1]):
            x_intersect = np.interp(y_intersection, [y_values[i-1], y_values[i]], [x_values[i-1], x_values[i]])
            return x_intersect, None
    # If no intersection found, use polynomial regression to predict the intersection
    if num_points is None or num_points > len(x_values):
        num_points = len(x_values)
    x_fit = x_values[-num_points:]
    y_fit = y_values[-num_points:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p = Polynomial.fit(x_fit, y_fit, degree)
    coefs = p.convert().coef
    coefs[-1] -= y_intersection
    roots = np.roots(coefs)
    for root in roots:
        if np.isreal(root) and root > x_values[-1]:
            predicted_intersection = np.real(root)
            return predicted_intersection, p
    return None, p

def find_max_min_in_window(x, y, window_size):
    """Find maximum and minimum values within a specified window in the given data."""
    max_values = []
    min_values = []
    
    for i in range(len(x) - window_size + 1):
        window_y = y[i:i + window_size]
        max_values.append(np.max(window_y))
        min_values.append(np.min(window_y))
        
    # Extend max/min lists to align with original x values for plotting
    max_values = np.pad(max_values, (window_size - 1, 0), mode='edge')
    min_values = np.pad(min_values, (window_size - 1, 0), mode='edge')

    return max_values, min_values

# Data keys and probability levels
data_keys = [
    '10',
    '20',
    '25'
]
prob_labels = ['10%', '25%', '50%', '75%']
probabilities = ['Tenper', 'Twentyfiveper', 'Fiftyper', 'Seventyfiveper']
limits = [0.004, 0.005, 0.01, 0.02]
highest_limit = max(limits)
colors = {
    '10%': ['#6495ED', '#3CB371', '#CD5C5C'],
    '25%': ['#4169E1', '#228B22', '#DC143C'],
    '50%': ['#000080', '#556B2F', '#B22222'],
    '75%': ['#191970', '#006400', '#8B0000']
}
data_files = {
    '25': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data.mat',
    '20': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_20.mat',
    '10': r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Matlab\Final Script\Wohler_curve_data_10.mat'
}

plt.figure(figsize=(12, 8))
intersection_data = []

for prob_index, prob_label in enumerate(prob_labels):

    # Loop through each data keys
    for key in data_keys:
        file = data_files[key]
        data = loadmat(file)
        MidPointIndexAll = data['MidPointIndexAll'].flatten()
        prob = probabilities[prob_index]
        
        curve_values = data[f'curve{prob}maxResistanceValueAll'].flatten()
        
        # Filter out values below 0.0
        curve_values_filtered = curve_values[curve_values > 0.0]
        MidPointIndexAll_filtered = MidPointIndexAll[curve_values > 0.0]
        
        # Apply LOWESS smoothing to filtered data
        lowess_smoothed = lowess(curve_values_filtered, MidPointIndexAll_filtered, frac=0.09)[:, 1]
        
        # Find max and min values in a specified window size
        window_size = 100  # Adjust this as needed
        max_curve_values, min_curve_values = find_max_min_in_window(MidPointIndexAll_filtered, lowess_smoothed, window_size)
    
        # Plot original data points and smoothed curve
        plt.semilogy(MidPointIndexAll_filtered, curve_values_filtered, '.', markersize=6, color=colors[prob_label][data_keys.index(key)], label=f'{key}µm Amplitude {prob_label} Probability')
        plt.semilogy(MidPointIndexAll_filtered, lowess_smoothed, '-', linewidth=2.5, color=colors[prob_label][data_keys.index(key)])
        
        for YIntersectionPoint in limits:
            # Calculate and plot intersection point or prediction
            intersection_point, poly = find_or_predict_intersection_point(MidPointIndexAll_filtered, lowess_smoothed, YIntersectionPoint, degree=2, num_points=len(MidPointIndexAll_filtered))
            
            if intersection_point is not None:
                plt.scatter(intersection_point, YIntersectionPoint, color='black', marker='o', s=25)
            else:
                # Plot extrapolated polynomial curve
                x_extrap = np.linspace(MidPointIndexAll_filtered[-1], MidPointIndexAll_filtered[-1] * 25, 100)
                y_extrap = poly(x_extrap)
#                plt.plot(x_extrap, y_extrap, '--', color=colors[prob_label][data_keys.index(key)])
                
                # Find and mark extrapolated intersection point
                intersection_point, _ = find_or_predict_intersection_point(x_extrap, y_extrap, YIntersectionPoint, degree=2, num_points=len(MidPointIndexAll_filtered))
#                if intersection_point is not None:
#                    plt.scatter(intersection_point, YIntersectionPoint, color='red', marker='x', s=25)
                
                # Polynomial fits for max and min curves
                poly_max = Polynomial.fit(MidPointIndexAll_filtered, max_curve_values, 2)
                poly_min = Polynomial.fit(MidPointIndexAll_filtered, min_curve_values, 2)
                
                # Generate x values for plotting
                x_vals_fit = np.linspace(MidPointIndexAll_filtered[0], MidPointIndexAll_filtered[-1] * 25, 100)
                y_vals_max_fit = poly_max(x_vals_fit)
                y_vals_min_fit = poly_min(x_vals_fit)
                
                # Plot max and min trend lines
                plt.plot(x_vals_fit, y_vals_max_fit, '--', color=colors[prob_label][data_keys.index(key)])  # Adjust color as needed
                plt.plot(x_vals_fit, y_vals_min_fit, '--', color=colors[prob_label][data_keys.index(key)])    # Adjust color as needed

                # Find and print intersection points for max line
                max_intersection, _ = find_or_predict_intersection_point(x_vals_fit, y_vals_max_fit, YIntersectionPoint, degree=2, num_points=len(MidPointIndexAll_filtered))
                if max_intersection is not None:
                    print(f'Max curve intersection for {prob_label}, {key}µm with {YIntersectionPoint} Ohm limit: {max_intersection} cycles')

            # Collect intersection data
            intersection_data.append((key, prob_label, YIntersectionPoint, intersection_point))
            
            # Count points above the limit before intersection
            if intersection_point is not None:
                count_above_limit = sum(curve_values_filtered[:np.argmax(MidPointIndexAll_filtered > intersection_point)] > YIntersectionPoint)
            else:
                count_above_limit = sum(curve_values_filtered > YIntersectionPoint)
    
# Final plot adjustments for each probability level
plt.xlabel('Number of Cycles')
plt.ylabel('Resistance (Ohm)')
plt.title('Wöhler Curves with Different Probabilities')
for limit in limits:
    plt.axhline(limit, color='black', linestyle='-', linewidth='2')
plt.ylim(0, 10)
plt.xlim(0,1000000)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()