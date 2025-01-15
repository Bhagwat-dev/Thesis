import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rainflow
import scipy.io
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat
from statsmodels.nonparametric.smoothers_lowess import lowess

################################################################################################################################# Important variable
# Define the threshold resistance in Ohms (Only this values 0.004, 0.005, 0.01, 0.02)
YIntersectionPoint = 0.004

# threshold for hysteresis filter im µm for exemplary contact dynamics
threshold = 0.2 

# sampling rate in seconds
sampling_interval = 500e-6 # here 500 µs

# please define here frequency of the signal in Hz
frequency_hz = 100
################################################################################################################################ important file paths
# Define the file path for contact dynamics .txt file here
file_path_contact_dynamics = r'Your_File_Path.txt'

# Define failure characteristics file path here (.mat files) for 10, 20, 25 µm amplitudes
data_files = {
    '25': r'Wohler_curve_data_5.mat',
    '20': r'Wohler_curve_data_20.mat',
#    '15': r'Wohler_curve_data_15.mat',
    '10': r'Wohler_curve_data_10.mat'
}

# Define failure characteristics for validation (.mat file)
failure_characteristics_validation = r'Your_File_Path.mat'

############################################################################################################################################################################
# Section 1 : Contact dynamics data loading and hysteresis filter 
############################################################################################################################################################################

# Initialize lists to store time and amplitude data
time = []
USENSOR = []
U1 = []
U2 = []
U3 = []

# Open the file
with open(file_path_contact_dynamics, 'r') as file:
    # Skip the first 3 lines in the text file
    for _ in range(3):
        next(file)
    
    # Read the data line by line
    for line in file:
        # Split the line into time and amplitude
        t, u1, u2, u3, usensor = line.split()
        
        # Replace commas with periods and convert to float
        t = float(t.replace(',', '.'))
        usensor = float(usensor.replace(',', '.'))
        u1 = float(u1.replace(',', '.'))
        u2 = float(u2.replace(',', '.'))
        u3 = float(u3.replace(',', '.'))
        
        # Check if the time exceeds 15 seconds
        if t > 15000:
            # Append to the lists
            time.append(t / 1e3 / 60)  # Convert to minutes
            USENSOR.append(usensor)
            U1.append(u1)
            U2.append(u2)
            U3.append(u3)


# conversion (v) to (um)
signal = [((u-0.0066)/20.024)*1e3 for u in USENSOR]

# Hysteresis Filter
def hysteresis_filter(time, signal, threshold):
    filtered_signal = []
    filtered_time = []
    last_value = signal[0]
    for t, value in zip(time, signal):
        if abs(value - last_value) > threshold:
            filtered_signal.append(value)
            filtered_time.append(t)
            last_value = value
    return np.array(filtered_time), np.array(filtered_signal)

# Apply hysteresis filter
filtered_time, filtered_signal = hysteresis_filter(time,signal, threshold)

############################################################################################################################################################################
# Section 2: FFT
############################################################################################################################################################################

# take mean of frequency signal to use for FFT
frequency_signal = signal - np.mean(signal)

# Set the sampling rate
sampling_rate = 1 / sampling_interval  # Sampling rate in Hz

# Perform FFT
N = len(frequency_signal)
yf = fft(frequency_signal)
xf = fftfreq(N, 1 / sampling_rate)

# Only take the positive frequencies
xf = xf[:N//2]
yf = np.abs(yf[:N//2])/N

fig = plt.figure()
gs = fig.add_gridspec(6, 4)

# Plot the frequency spectrum
ax_FFT = fig.add_subplot(gs[:2, 2:])

ax_FFT.plot(xf, yf)
ax_FFT.set_title("FFT")
ax_FFT.set_xlabel("Frequency (Hz)")
ax_FFT.set_ylabel("Magnitude")
ax_FFT.set_xlim(0,1000)
ax_FFT.grid()

############################################################################################################################################################################
# Section 3: Rainflow counting and load collective
############################################################################################################################################################################

# Perform rainflow counting
count_cycles = rainflow.count_cycles(filtered_signal)

# Prepare data for plotting
ranges = [row[0] for row in count_cycles]
counts = [row[1] for row in count_cycles]

# Convert range to amplitude
amplitudes = [r / 2 for r in ranges]  # divide range by 2 to get amplitude

# Create DataFrame
df = pd.DataFrame({'Amplitude': amplitudes, 'Count': counts})

# Define amplitude bins
bin_width = 1
bins = np.arange(0, df['Amplitude'].max() + bin_width, bin_width)
df['Amplitude_Bin'] = pd.cut(df['Amplitude'], bins, right=False)

# Determine load collective
load_collective = df.groupby('Amplitude_Bin').sum().reset_index()

# Calculate total counted cycle number
total_cycles = df['Count'].sum()

# Create labels for the upper limits of the bins
bin_labels = [int(bin.right) for bin in load_collective['Amplitude_Bin']]

# Write load collective to Excel
#try:
#    load_collective.to_excel(r'W:\1065_RT_Research_Development\810_Studenten\Bhagwat Kalathiya\Bhagwat_Thesis\Experiments\Test\Sample_test\Sin_sweep_100\Finale_test\T_ss_100_05\load collective.xlsx')
#except Exception as e:
#    print("Failed to write load collective to excel:", e)

ax1 = fig.add_subplot(gs[:1, :2])
# Calculate the difference between consecutive data points
diff = np.diff(filtered_signal)

# Find the indices where the difference changes sign
turning_points = np.where(np.diff(np.sign(diff)))[0] + 1

# Assuming `filtered_time` is in minutes, convert to seconds
filtered_time_seconds = filtered_time * 60
time_seconds = np.array(time)*60

# Shift the time arrays to start from 0
time_seconds_shifted = time_seconds - time_seconds[0]
filtered_time_seconds_shifted = filtered_time_seconds - filtered_time_seconds[0]

# Create the main subplot for the original signal
#ax1.plot(time_seconds_shifted, signal - np.mean(signal), color = 'red' , label='Original Signal')
ax1.plot(filtered_time_seconds_shifted, filtered_signal - np.mean(filtered_signal), color = 'blue' , label='Filtered Signal')
#ax1.plot(time_seconds, signal - np.mean(signal), color = 'red' , label='Original Signal')
filtered_signal = filtered_signal - np.mean(filtered_signal)
signal = signal - np.mean(signal)
# ax1.plot(filtered_time[turning_points], filtered_signal[turning_points], 'ro', label='Turning Points')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Amplitude (µm)')
ax1.set_title('Contact Sliding, 100Hz, V1_sweep_sin')

# Define the areas you want to zoom into
zoom_areas = [(2070, 2090, -30, 30), (2724, 2724.05, -30, 30)]  # replace with your desired limits

# Create a subplot for each zoomed-in plot
for i, (x_min, x_max, y_min, y_max) in enumerate(zoom_areas):
    ax_zoom = plt.subplot(gs[1:2, i])
    ax_zoom.plot(filtered_time_seconds, filtered_signal, label=f'Zoomed Signal {i+1}')
#    ax_zoom.plot(time_seconds, signal, label=f'Zoomed Signal {i+1}')
    ax_zoom.plot(filtered_time_seconds[turning_points], filtered_signal[turning_points], 'bo', label='Turning Points', markersize= 2)
#    ax_zoom.plot(time_seconds[turning_points], signal[turning_points], 'ro', label='Turning Points', markersize= 2)
#    ax_zoom.scatter(filtered_time[turning_points], filtered_signal[turning_points], c='r', marker='o', s=5)
    ax_zoom.set_xlim(x_min, x_max)
    ax_zoom.set_ylim(y_min, y_max)
    ax_zoom.set_xlabel('Time (sec)')
    ax_zoom.set_ylabel('Amplitude (µm)')

    # Draw vertical lines on the main plot
    ax1.axvline(x=x_min, color='g', linestyle='--')
    ax1.axvline(x=x_max, color='g', linestyle='--')

# Create a new figure for the histogram
ax2 = fig.add_subplot(gs[2:4, :2])

# Plot load collective
ax2.bar(range(len(load_collective)), load_collective['Count'], tick_label=bin_labels, width=0.8)
ax2.text(0.90, 0.95, f'Total cycles: {total_cycles}', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes)
ax2.set_ylabel('Cycle Count')
ax2.set_xlabel('Amplitude (µm)')
ax2.set_title('Amplitude vs Cycle Count')

############################################################################################################################################################################
# Section 4: Generating Wöhler curve
############################################################################################################################################################################

# Probability colors
colors = {
    '10%': ['#6495ED', '#3CB371', '#CD5C5C'],
    '25%': ['#4169E1', '#228B22', '#DC143C'],
    '50%': ['#000080', '#556B2F', '#B22222'],
    '75%': ['#191970', '#006400', '#8B0000']
}

# Function to find or predict the first intersection point
def find_or_predict_intersection_point(x_values, y_values, y_intersection):
    for i in range(1, len(x_values)):
        if (y_values[i-1] <= y_intersection <= y_values[i]) or (y_values[i] <= y_intersection <= y_values[i-1]):
            x_intersect = np.interp(y_intersection, [y_values[i-1], y_values[i]], [x_values[i-1], x_values[i]])
            return x_intersect
    # If no intersection found, predict the intersection point
    slope = (y_values[-1] - y_values[-2]) / (x_values[-1] - x_values[-2])
    x_predict = x_values[-1] + (y_intersection - y_values[-1]) / slope
    return x_predict

# Create matrix to store intersection points
prob_labels = ['10%', '25%', '50%', '75%']
data_keys = ['10',
          '20',
#            '15',
            '25'
]
PointsMatrix = np.full((len(data_keys) + 1, len(prob_labels) + 1), np.nan, dtype=object)

# Fill the first column and first row with specified values
PointsMatrix[1:, 0] = [10, 20, 25]  # First column values
PointsMatrix[0, 1:] = [10, 25, 50, 75]  # First row values

# DataFrame to store intersection points
intersection_data = []

# Iterate over data files and plot each
for key in data_keys:
    file = data_files[key]
    data = loadmat(file)
    MidPointIndexAll = data['MidPointIndexAll'].flatten()
    probabilities = ['Tenper', 'Twentyfiveper', 'Fiftyper', 'Seventyfiveper']
    
    for i, prob in enumerate(probabilities):
        curve_values = data[f'curve{prob}maxResistanceValueAll'].flatten()
        
        # Filter out values below 0.0
        curve_values_filtered = curve_values[curve_values > 0.0]
        MidPointIndexAll_filtered = MidPointIndexAll[curve_values > 0.0]
    
        # Apply LOWESS smoothing to filtered data
        lowess_smoothed = lowess(curve_values_filtered, MidPointIndexAll_filtered, frac=0.09)[:, 1]
    
        # Plot original data points and smoothed curve
    #    plt.semilogy(MidPointIndexAll_filtered, curve_values_filtered, '.', markersize=6, color=colors[prob_labels[i]][data_keys.index(key)], label=f'{prob_labels[i]} Probability {key}')
    #    plt.semilogy(MidPointIndexAll_filtered, lowess_smoothed, '-', linewidth=2.5, color=colors[prob_labels[i]][data_keys.index(key)])
        
        intersection_point = find_or_predict_intersection_point(MidPointIndexAll_filtered, lowess_smoothed, YIntersectionPoint)
    #    plt.scatter(intersection_point, YIntersectionPoint, color='black', marker='o', s=25, label=f'Intersection Point {prob_labels[i]} Probability {key}')
        
        # Fill the intersection data in the matrix
        row_index = data_keys.index(key) + 1
        col_index = prob_labels.index(prob_labels[i]) + 1
        PointsMatrix[row_index, col_index] = intersection_point

#####################################################################

# Dictionary to store cycle values for diff. amplitudes
cycle_values = {
    0.004: {
        'cycle_10_5': 3300000,
        'cycle_25_5': 3800000,
        'cycle_50_5': 8500000,
        'cycle_75_5': 10000000,
        'cycle_10_10': PointsMatrix[1, 1]-60000,
        'cycle_25_10': PointsMatrix[1, 2]-18000,
        'cycle_50_10': PointsMatrix[1, 3]-40000,
        'cycle_75_10': PointsMatrix[1, 4]-44000,
        'cycle_10_20': PointsMatrix[2, 1]-2000,
        'cycle_25_20': PointsMatrix[2, 2]-6000,
        'cycle_50_20': PointsMatrix[2, 3]-2000,
        'cycle_75_20': PointsMatrix[2, 4],
        'cycle_10_25': PointsMatrix[3, 1]-2000,
        'cycle_25_25': PointsMatrix[3, 2]-2000,
        'cycle_50_25': PointsMatrix[3, 3]-4000,
        'cycle_75_25': PointsMatrix[3, 4]
    },
    0.005: {
        'cycle_10_5': 5200000,
        'cycle_25_5': 5800000,
        'cycle_50_5': 12000000,
        'cycle_75_5': 16000000,
        'cycle_10_10': PointsMatrix[1, 1]-8000,
        'cycle_25_10': PointsMatrix[1, 2]-8000,
        'cycle_50_10': PointsMatrix[1, 3]-12000,
        'cycle_75_10': 600000,
        'cycle_10_20': PointsMatrix[2, 1]-4000,
        'cycle_25_20': PointsMatrix[2, 2],
        'cycle_50_20': PointsMatrix[2, 3],
        'cycle_75_20': PointsMatrix[2, 4],
        'cycle_10_25': PointsMatrix[3, 1]-4000,
        'cycle_25_25': PointsMatrix[3, 2]-2000,
        'cycle_50_25': PointsMatrix[3, 3],
        'cycle_75_25': PointsMatrix[3, 4]
    },
    0.01: {
        'cycle_10_5': 14000000,
        'cycle_25_5': 22000000,
        'cycle_50_5': 34000000,
        'cycle_75_5': 44000000,
        'cycle_10_10': PointsMatrix[1, 1]-10000,
        'cycle_25_10': 650000,
        'cycle_50_10': 824000,
        'cycle_75_10': 980000,
        'cycle_10_20': PointsMatrix[2, 1],
        'cycle_25_20': PointsMatrix[2, 2]-4000,
        'cycle_50_20': PointsMatrix[2, 3]-2000,
        'cycle_75_20': PointsMatrix[2, 4]-20000,
        'cycle_10_25': PointsMatrix[3, 1],
        'cycle_25_25': PointsMatrix[3, 2],
        'cycle_50_25': PointsMatrix[3, 3],
        'cycle_75_25': PointsMatrix[3, 4]-2000
    },
    0.02: {
        'cycle_10_5': 33000000,
        'cycle_25_5': 50000000,
        'cycle_50_5': 78000000,
        'cycle_75_5': 99000000,
        'cycle_10_10': 643000,
        'cycle_25_10': 1000000,
        'cycle_50_10': 1260000,
        'cycle_75_10': 1462000,
        'cycle_10_20': PointsMatrix[2, 1],
        'cycle_25_20': PointsMatrix[2, 2]-34000,
        'cycle_50_20': PointsMatrix[2, 3]-10000,
        'cycle_75_20': PointsMatrix[2, 4]-10000,
        'cycle_10_25': PointsMatrix[3, 1],
        'cycle_25_25': PointsMatrix[3, 2],
        'cycle_50_25': PointsMatrix[3, 3]-8000,
        'cycle_75_25': PointsMatrix[3, 4]-2000
    }
}

# Select cycle values based on the YIntersectionPoint
if YIntersectionPoint in cycle_values:
    selected_cycles = cycle_values[YIntersectionPoint]
    cycle_10_5 = selected_cycles['cycle_10_5']
    cycle_25_5 = selected_cycles['cycle_25_5']
    cycle_50_5 = selected_cycles['cycle_50_5']
    cycle_75_5 = selected_cycles['cycle_75_5']
    cycle_10_10 = selected_cycles['cycle_10_10']
    cycle_25_10 = selected_cycles['cycle_25_10']
    cycle_50_10 = selected_cycles['cycle_50_10']
    cycle_75_10 = selected_cycles['cycle_75_10']
    cycle_10_20 = selected_cycles['cycle_10_20']
    cycle_25_20 = selected_cycles['cycle_25_20']
    cycle_50_20 = selected_cycles['cycle_50_20']
    cycle_75_20 = selected_cycles['cycle_75_20']
    cycle_10_25 = selected_cycles['cycle_10_25']
    cycle_25_25 = selected_cycles['cycle_25_25']
    cycle_50_25 = selected_cycles['cycle_50_25']
    cycle_75_25 = selected_cycles['cycle_75_25']
else:
    print(f"Threshold resistance {YIntersectionPoint} Ohm not found in the cycle values.")

#100_0.02
amplitudes_data = np.array([5, 10, 20, 25])
# Extract cycle data from PointsMatrix
cycles_data_10 = np.array([cycle_10_5, cycle_10_10, cycle_10_20, cycle_10_25], dtype=float)
cycles_data_25 = np.array([cycle_25_5, cycle_25_10, cycle_25_20, cycle_25_25], dtype=float)
cycles_data_50 = np.array([cycle_50_5, cycle_50_10, cycle_50_20, cycle_50_25], dtype=float)
cycles_data_75 = np.array([cycle_75_5, cycle_75_10, cycle_75_20, cycle_75_25], dtype=float)

# Perform linear regression
log_amplitudes = np.log(amplitudes_data).reshape(-1, 1)
log_cycles_10 = np.log(cycles_data_10)
log_cycles_25 = np.log(cycles_data_25)
log_cycles_50 = np.log(cycles_data_50)
log_cycles_75 = np.log(cycles_data_75)

# Transform the matrix of features
poly_reg = PolynomialFeatures(degree = 3)
amplitudes_poly = poly_reg.fit_transform(log_amplitudes)

# Fit the Polynomial Regression model
poly_model_10 = LinearRegression()
poly_model_25 = LinearRegression()
poly_model_50 = LinearRegression()
poly_model_75 = LinearRegression()

poly_model_10.fit(amplitudes_poly, log_cycles_10)
poly_model_25.fit(amplitudes_poly, log_cycles_25)
poly_model_50.fit(amplitudes_poly, log_cycles_50)
poly_model_75.fit(amplitudes_poly, log_cycles_75)

# Define the range of amplitudes for plotting
extrapolated_amplitudes = np.linspace(4, 30, num=200).reshape(-1, 1)

# Predicting a new result with Polynomial Regression
extrapolated_cycles_poly_10 = np.exp(poly_model_10.predict(poly_reg.transform(np.log(extrapolated_amplitudes))))
extrapolated_cycles_poly_25 = np.exp(poly_model_25.predict(poly_reg.transform(np.log(extrapolated_amplitudes))))
extrapolated_cycles_poly_50 = np.exp(poly_model_50.predict(poly_reg.transform(np.log(extrapolated_amplitudes))))
extrapolated_cycles_poly_75 = np.exp(poly_model_75.predict(poly_reg.transform(np.log(extrapolated_amplitudes))))

# Create a new figure for the Wohler curve
ax_wohler = fig.add_subplot(gs[2:4, 2:])

# Plot Wohler curve using Polynomial Regression data
ax_wohler.plot(extrapolated_cycles_poly_10, extrapolated_amplitudes, label='10% Probability', linestyle='-', color='red')
ax_wohler.plot(extrapolated_cycles_poly_25, extrapolated_amplitudes, label='25% Probability', linestyle='-', color='blue')
ax_wohler.plot(extrapolated_cycles_poly_50, extrapolated_amplitudes, label='50% Probability', linestyle='-', color='orange')
ax_wohler.plot(extrapolated_cycles_poly_75, extrapolated_amplitudes, label='75% Probability', linestyle='-', color='green')

ax_wohler.set_xlabel('Cycles')
ax_wohler.set_ylabel('Amplitude (µm)')
ax_wohler.set_title(f'Wöhler curve 100 Hz, R={YIntersectionPoint} Ohm')

ax_wohler.legend()  # Add a legend
ax_wohler.grid(True)

# Limit the x-axis and y-axis
ax_wohler.set_xlim([0, 3e6])
ax_wohler.set_ylim([0,30])

############################################################################################################################################################################
#Damage calculation
############################################################################################################################################################################

# Define the functions to extrapolate Nf for each probability level

def extrapolate_Nf_10(amplitude):
    log_amplitude = np.log(amplitude).reshape(-1, 1)
    return np.exp(poly_model_10.predict(poly_reg.transform(log_amplitude)))

def extrapolate_Nf_25(amplitude):
    log_amplitude = np.log(amplitude).reshape(-1, 1)
    return np.exp(poly_model_25.predict(poly_reg.transform(log_amplitude)))

def extrapolate_Nf_50(amplitude):
    log_amplitude = np.log(amplitude).reshape(-1, 1)
    return np.exp(poly_model_50.predict(poly_reg.transform(log_amplitude)))

def extrapolate_Nf_75(amplitude):
    log_amplitude = np.log(amplitude).reshape(-1, 1)
    return np.exp(poly_model_75.predict(poly_reg.transform(log_amplitude)))

# Assuming you have a DataFrame 'df' with columns 'Amplitude' and 'Count'
df['Nf_10'] = df['Amplitude'].apply(extrapolate_Nf_10)
df['Nf_25'] = df['Amplitude'].apply(extrapolate_Nf_25)
df['Nf_50'] = df['Amplitude'].apply(extrapolate_Nf_50)
df['Nf_75'] = df['Amplitude'].apply(extrapolate_Nf_75)

# Ignore data for damage calculation below amplitude 9
df = df[(df['Amplitude'] >= 6) & (df['Amplitude'] <= 30)]

# Calculate damage for each cycle for each set of cycle data
df['Damage_10'] = df['Count'] / df['Nf_10']
df['Damage_25'] = df['Count'] / df['Nf_25']
df['Damage_50'] = df['Count'] / df['Nf_50']
df['Damage_75'] = df['Count'] / df['Nf_75']

# Sum up the damage for each set of cycle data
total_damage_10 = df['Damage_10'].sum()
total_damage_25 = df['Damage_25'].sum()
total_damage_50 = df['Damage_50'].sum()
total_damage_75 = df['Damage_75'].sum()

print(f"Total damage according to Miner's rule:")

print(f"Damage at 10% Probability: {total_damage_10}")
print(f"Damage at 25% Probability: {total_damage_25}")
print(f"Damage at 50% Probability: {total_damage_50}")
print(f"Damage at 75% Probability: {total_damage_75}")

########

# Sort the dataframe by Damage in descending order for each set of cycle data
df_sorted_10 = df.sort_values(by='Damage_10', ascending=False)
df_sorted_25 = df.sort_values(by='Damage_25', ascending=False)
df_sorted_50 = df.sort_values(by='Damage_50', ascending=False)
df_sorted_75 = df.sort_values(by='Damage_75', ascending=False)

# Calculate cumulative damage for each set of cycle data
df_sorted_10['Cumulative_Damage'] = df_sorted_10['Damage_10'].cumsum()
df_sorted_25['Cumulative_Damage'] = df_sorted_25['Damage_25'].cumsum()
df_sorted_50['Cumulative_Damage'] = df_sorted_50['Damage_50'].cumsum()
df_sorted_75['Cumulative_Damage'] = df_sorted_75['Damage_75'].cumsum()

# Add a 'Cycle' column to the dataframe for each set of cycle data
df_sorted_10['Cycle'] = range(1, len(df_sorted_10) + 1)
df_sorted_25['Cycle'] = range(1, len(df_sorted_25) + 1)
df_sorted_50['Cycle'] = range(1, len(df_sorted_50) + 1)
df_sorted_75['Cycle'] = range(1, len(df_sorted_75) + 1)

#############################################################

# Define the functions to extrapolate Nf for each probability level
def extrapolate_Nf(amplitude, poly_model):
    log_amplitude = np.log(amplitude).reshape(-1, 1)
    return np.exp(poly_model.predict(poly_reg.transform(log_amplitude)))

# Define the functions to calculate damage for each bin
def calculate_damage(row):
    amplitude = row['Amplitude_Bin'].left + bin_width / 2
    
    Nf_10 = extrapolate_Nf(amplitude, poly_model_10)
    Nf_25 = extrapolate_Nf(amplitude, poly_model_25)
    Nf_50 = extrapolate_Nf(amplitude, poly_model_50)
    Nf_75 = extrapolate_Nf(amplitude, poly_model_75)
    
    damage_10 = row['Count'] / Nf_10
    damage_25 = row['Count'] / Nf_25
    damage_50 = row['Count'] / Nf_50
    damage_75 = row['Count'] / Nf_75
    return pd.Series({
        'Amplitude_Bin': row['Amplitude_Bin'],
        
        'Damage_10': damage_10,
        'Damage_25': damage_25,
        'Damage_50': damage_50,
        'Damage_75': damage_75
    })

# Assuming you have a DataFrame 'df' with columns 'Amplitude' and 'Count'
# Define amplitude bins
bin_width = 1
bins = np.arange(0, df['Amplitude'].max() + bin_width, bin_width)
df['Amplitude_Bin'] = pd.cut(df['Amplitude'], bins, right=False)

# Group by amplitude bins and sum counts
load_collective = df.groupby('Amplitude_Bin')['Count'].sum().reset_index()

# Assuming you have defined your poly models and poly reg
df['Nf_10'] = df['Amplitude'].apply(extrapolate_Nf_10)
df['Nf_25'] = df['Amplitude'].apply(extrapolate_Nf_25)
df['Nf_50'] = df['Amplitude'].apply(extrapolate_Nf_50)
df['Nf_75'] = df['Amplitude'].apply(extrapolate_Nf_75)

# Ignore data for damage calculation below amplitude 9
df = df[(df['Amplitude'] >= 6) & (df['Amplitude'] <= 30)]

# Calculate damage for each cycle for each set of cycle data
df['Damage_10'] = df['Count'] / df['Nf_10']
df['Damage_25'] = df['Count'] / df['Nf_25']
df['Damage_50'] = df['Count'] / df['Nf_50']
df['Damage_75'] = df['Count'] / df['Nf_75']

# Sum up the damage for each set of cycle data
total_damage_10 = df['Damage_10'].sum()
total_damage_25 = df['Damage_25'].sum()
total_damage_50 = df['Damage_50'].sum()
total_damage_75 = df['Damage_75'].sum()

print(f"Total damage according to Miner's rule:")

print(f"Damage at 10% Probability: {total_damage_10}")
print(f"Damage at 25% Probability: {total_damage_25}")
print(f"Damage at 50% Probability: {total_damage_50}")
print(f"Damage at 75% Probability: {total_damage_75}")

# Calculate damage for each amplitude bin
damage_results = load_collective.apply(calculate_damage, axis=1)

# Merge the count data with damage results
damage_results = pd.merge(damage_results, load_collective[['Amplitude_Bin', 'Count']], on='Amplitude_Bin', how='left')

# Sort the damage results DataFrame by amplitude bin
damage_results_sorted = damage_results.sort_values(by='Amplitude_Bin')

############

# Calculate cumulative damage for each probability level
damage_results_sorted['Cumulative_Damage_10'] = damage_results_sorted['Damage_10'].cumsum()
damage_results_sorted['Cumulative_Damage_25'] = damage_results_sorted['Damage_25'].cumsum()
damage_results_sorted['Cumulative_Damage_50'] = damage_results_sorted['Damage_50'].cumsum()
damage_results_sorted['Cumulative_Damage_75'] = damage_results_sorted['Damage_75'].cumsum()

# Extract the midpoints of the amplitude bins for plotting
damage_results_sorted['Amplitude_Bin_Mid'] = damage_results_sorted['Amplitude_Bin'].apply(lambda x: x.mid)

ax4 = fig.add_subplot(gs[4:, :2])

ax4.step(damage_results_sorted['Amplitude_Bin_Mid'], damage_results_sorted['Cumulative_Damage_10'], label='10% Probability', where='post', color='red')
ax4.step(damage_results_sorted['Amplitude_Bin_Mid'], damage_results_sorted['Cumulative_Damage_25'], label='25% Probability', where='post', color='blue')
ax4.step(damage_results_sorted['Amplitude_Bin_Mid'], damage_results_sorted['Cumulative_Damage_50'], label='50% Probability', where='post', color='orange')
ax4.step(damage_results_sorted['Amplitude_Bin_Mid'], damage_results_sorted['Cumulative_Damage_75'], label='75% Probability', where='post', color='green')
ax4.set_xlabel('Amplitude (µm)')
ax4.set_ylabel('Cumulative Damage')
ax4.set_title('Amplitude (µm) vs Cumulative Damage')
plt.legend()
plt.grid(True)

############################################################################################################################################################################
# Damage by time
############################################################################################################################################################################

# Perform rainflow counting
cycles = list(rainflow.extract_cycles(filtered_signal))

# Prepare data for plotting
ranges = [row[0] for row in cycles]
counts = [row[1] for row in cycles]

# Convert range to amplitude
amplitudes = [r / 2 for r in ranges]  # divide range by 2 to get amplitude

# Filter out amplitudes below 6 µm
filtered_amplitudes = [amp for amp in amplitudes if amp >= 0]
filtered_cycles = [cycles[i] for i in range(len(amplitudes)) if amplitudes[i] >= 0]

def fit_polynomial_model(amplitudes_data, cycles_data):
    log_amplitudes = np.log(amplitudes_data).reshape(-1, 1)
    log_cycles = np.log(cycles_data)
    
    poly_reg = PolynomialFeatures(degree=3)
    amplitudes_poly = poly_reg.fit_transform(log_amplitudes)
    
    poly_model = LinearRegression()
    poly_model.fit(amplitudes_poly, log_cycles)
    
    return poly_model, poly_reg

# Fit models for each probability level
model_10, poly_reg_10 = fit_polynomial_model(amplitudes_data, cycles_data_10)
model_25, poly_reg_25 = fit_polynomial_model(amplitudes_data, cycles_data_25)
model_50, poly_reg_50 = fit_polynomial_model(amplitudes_data, cycles_data_50)
model_75, poly_reg_75 = fit_polynomial_model(amplitudes_data, cycles_data_75)

def calculate_damage(cycles, amplitudes, model, poly_reg):
    damage_data = []
    
    for cycle, amplitude in zip(cycles, amplitudes):
        log_amplitude = np.log(amplitude).reshape(1, -1)
        amplitude_poly = poly_reg.transform(log_amplitude)
        log_cycles_to_failure = model.predict(amplitude_poly)
        cycles_to_failure = np.exp(log_cycles_to_failure)[0]
        damage = 1 / cycles_to_failure
        damage_data.append([amplitude, 1, cycles_to_failure, damage])
    
    return damage_data

# Calculate damage for each probability level
damage_data_10 = calculate_damage(filtered_cycles, filtered_amplitudes, model_10, poly_reg_10)
damage_data_25 = calculate_damage(filtered_cycles, filtered_amplitudes, model_25, poly_reg_25)
damage_data_50 = calculate_damage(filtered_cycles, filtered_amplitudes, model_50, poly_reg_50)
damage_data_75 = calculate_damage(filtered_cycles, filtered_amplitudes, model_75, poly_reg_75)

# Create DataFrames
damage_df_10 = pd.DataFrame(damage_data_10, columns=['Amplitude', 'Count', 'Cycles to Failure', 'Damage'])
damage_df_25 = pd.DataFrame(damage_data_25, columns=['Amplitude', 'Count', 'Cycles to Failure', 'Damage'])
damage_df_50 = pd.DataFrame(damage_data_50, columns=['Amplitude', 'Count', 'Cycles to Failure', 'Damage'])
damage_df_75 = pd.DataFrame(damage_data_75, columns=['Amplitude', 'Count', 'Cycles to Failure', 'Damage'])

# Function to calculate the number of cycles to reach 100% damage
def cycles_to_reach_100_percent_damage(damage_df):
    cumulative_damage = 0
    cumulative_cycles = 0
    
    for index, row in damage_df.iterrows():
        cumulative_damage += row['Damage']
        cumulative_cycles += row['Count']
        if cumulative_damage >= 1:
            return cumulative_cycles, cumulative_damage
    
    return cumulative_cycles, cumulative_damage  # Return the total cycles if 100% damage is not reached

# Calculate the number of cycles to reach 100% damage for each probability level
cycles_100_damage_10, total_damage_10 = cycles_to_reach_100_percent_damage(damage_df_10)
cycles_100_damage_25, total_damage_25 = cycles_to_reach_100_percent_damage(damage_df_25)
cycles_100_damage_50, total_damage_50 = cycles_to_reach_100_percent_damage(damage_df_50)
cycles_100_damage_75, total_damage_75 = cycles_to_reach_100_percent_damage(damage_df_75)

print(f"10% Probability Level: Cycles to reach 100% damage: {cycles_100_damage_10}, Total damage: {total_damage_10}")
print(f"25% Probability Level: Cycles to reach 100% damage: {cycles_100_damage_25}, Total damage: {total_damage_25}")
print(f"50% Probability Level: Cycles to reach 100% damage: {cycles_100_damage_50}, Total damage: {total_damage_50}")
print(f"75% Probability Level: Cycles to reach 100% damage: {cycles_100_damage_75}, Total damage: {total_damage_75}")

###################################

# Function to calculate cumulative damage data for plotting
def calculate_cumulative_damage_data(damage_df):
    cumulative_damage = 0
    cumulative_cycles = 0
    cycles = []
    damages = []
    
    for index, row in damage_df.iterrows():
        cumulative_damage += row['Damage']
        cumulative_cycles += row['Count']
        cycles.append(cumulative_cycles)
        damages.append(cumulative_damage)
        if cumulative_damage >= 1:
            break
    
    return cycles, damages

# Calculate cumulative damage data for each probability level
cycles_damage_10, damages_10 = calculate_cumulative_damage_data(damage_df_10)
cycles_damage_25, damages_25 = calculate_cumulative_damage_data(damage_df_25)
cycles_damage_50, damages_50 = calculate_cumulative_damage_data(damage_df_50)
cycles_damage_75, damages_75 = calculate_cumulative_damage_data(damage_df_75)

###############################

# Frequency information
time_per_cycle = 1 / frequency_hz  # Time per cycle in seconds

# Function to calculate cumulative damage and time data for plotting
def calculate_cumulative_damage_time_data(damage_df, time_per_cycle):
    cumulative_damage = 0
    cumulative_cycles = 0
    cumulative_time = 0
    times = []
    damages = []
    
    for index, row in damage_df.iterrows():
        cumulative_damage += row['Damage']
        cumulative_cycles += row['Count']
        # Calculate cumulative time based on cycles
        cumulative_time += row['Count'] * time_per_cycle
        times.append(cumulative_time)
        damages.append(cumulative_damage)
        if cumulative_damage >= 1:
            break
    
    return times, damages

# Calculate cumulative damage and time data for each probability level
time_damage_10, damages_10 = calculate_cumulative_damage_time_data(damage_df_10, time_per_cycle)
time_damage_25, damages_25 = calculate_cumulative_damage_time_data(damage_df_25, time_per_cycle)
time_damage_50, damages_50 = calculate_cumulative_damage_time_data(damage_df_50, time_per_cycle)
time_damage_75, damages_75 = calculate_cumulative_damage_time_data(damage_df_75, time_per_cycle)

#plot cycle vs cumulated damage plot
ax5 = fig.add_subplot(gs[4:, 2:])

ax5.plot(time_damage_10,damages_10, label='10% Probability', color='red')
ax5.plot(time_damage_25,damages_25, label='25% Probability', color='blue')
ax5.plot(time_damage_50,damages_50, label='50% Probability', color='orange')
ax5.plot(time_damage_75,damages_75, label='75% Probability', color='green')

ax5.set_xlabel('Time (Sec)')
ax5.set_ylabel('Cumulative Damage')
ax5.set_title('Damage based on Time')
ax5.legend()

############################################################################################################################################################################
# Validation of Prediciton
############################################################################################################################################################################

# Load the data from the .mat file
Exp_data = loadmat(failure_characteristics_validation)

# Extract variables from the loaded data
MidPointIndexAll = Exp_data['MidPointIndexAll'].flatten()
curveOnepermaxResistanceValueAll = Exp_data['curveOnepermaxResistanceValueAll'].flatten()
curveTenpermaxResistanceValueAll = Exp_data['curveTenpermaxResistanceValueAll'].flatten()
curveTwentyfivepermaxResistanceValueAll = Exp_data['curveTwentyfivepermaxResistanceValueAll'].flatten()
curveFiftypermaxResistanceValueAll = Exp_data['curveFiftypermaxResistanceValueAll'].flatten()
curveSeventyfivepermaxResistanceValueAll = Exp_data['curveSeventyfivepermaxResistanceValueAll'].flatten()

# Plotting Failure points
plt.figure(figsize=(12, 8))

plt.semilogy(MidPointIndexAll, curveOnepermaxResistanceValueAll, '.', markersize=6, color='red', label='1% Probability')
plt.semilogy(MidPointIndexAll, curveTenpermaxResistanceValueAll, '.', markersize=6, color='red', label='10% Probability')
plt.semilogy(MidPointIndexAll, curveTwentyfivepermaxResistanceValueAll, '.', markersize=6, color='blue', label='25% Probability')
plt.semilogy(MidPointIndexAll, curveFiftypermaxResistanceValueAll, '.', markersize=6, color='orange', label='50% Probability')
plt.semilogy(MidPointIndexAll, curveSeventyfivepermaxResistanceValueAll, '.', markersize=6, color='green', label='75% Probability')

# Function to find and highlight points in the range
def highlight_points(x, y, color, label):

    margin = 0.05 * YIntersectionPoint

    indices_in_range = np.where((y >= YIntersectionPoint - margin) & (y <= YIntersectionPoint + margin))[0]
    if len(indices_in_range) > 0:
        first_point = indices_in_range[0]
        last_point = indices_in_range[-1]
        plt.semilogy(x[first_point], y[first_point], 'x', markersize=8, color=color)
        plt.semilogy(x[last_point], y[last_point], 'x', markersize=8, color=color)
        print(f"{label} - First Point: MidPointIndex = {x[first_point]}, Resistance Value = {y[first_point]}")
        print(f"{label} - Last Point: MidPointIndex = {x[last_point]}, Resistance Value = {y[last_point]}")

# Highlight points for each probability percent
highlight_points(MidPointIndexAll, curveTenpermaxResistanceValueAll, 'red', '10% Probability')
highlight_points(MidPointIndexAll, curveTwentyfivepermaxResistanceValueAll, 'blue', '25% Probability')
highlight_points(MidPointIndexAll, curveFiftypermaxResistanceValueAll, 'orange', '50% Probability')
highlight_points(MidPointIndexAll, curveSeventyfivepermaxResistanceValueAll, 'green', '75% Probability')

# Add legend and labels
plt.legend()
plt.xlabel('Cycle',size=18)
plt.ylabel('Resistance Value (ohm)',size=18)
plt.title(f'Prediction, R = {YIntersectionPoint} Ohm',size=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(YIntersectionPoint, color='black')
plt.axvline(cycles_100_damage_10, linestyle='--', color='red')
plt.axvline(cycles_100_damage_25, linestyle='--', color='blue')
plt.axvline(cycles_100_damage_50, linestyle='--', color='orange')
plt.axvline(cycles_100_damage_75, linestyle='--', color='green')

# Show plot
plt.show()
