clear all
close all
clc
hold all
Farbe = ['b','r','k','m','y','r','m','b','b','k'];

%% Variable Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Frequency = 100;                  %[Hz]                               
Amplitude = 10;                   % Amplitude
Timeframe = 500 * 10^(-6);        %[s]                                                                                                                  
LimitingResistance = 0.01;        %[Ohm]
n = 500;                          % Number of Dividing Areas
HistogramResolution = 50;         % Counts of maximum resistance per area values
FilteringResolution = 25;         % Smoothing Curve Resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data
folderPath = 'Your_File_Path';
%AmplitudeValue = 10;

try
    % Get a list of all .mat files in the selected folder
    matFiles = dir(fullfile(folderPath, '*.mat'));

    R_Values = cell(1, length(matFiles));
    % Loop through each .mat file and extract data
    for i = 1:length(matFiles)
        U = [];  % Initialize U as an empty array

        % Construct the full file path for the current .mat file
        fullMatFilePath = fullfile(folderPath, matFiles(i).name);

        % Load the data from the current .mat file
        loadedData = load(fullMatFilePath);

        % Access the variables from the loaded data
        Time = loadedData.Time;
        U1 = loadedData.U1;
        U2 = loadedData.U2;
        U3 = loadedData.U3;
        USensor = loadedData.USensor;

        % Separating the start and end runtime
        Unew = ((USensor - 0.0066) / 20.024) * 1e3;    
        Ufinal = Unew - mean(Unew);

        % Find peaks within the specified range
        peaksIdx = find(diff(sign(diff(Ufinal))) == -2) + 1;

        % Create a logical index for elements to keep based on the peaks
        keepIndices = (1:length(U1) >= peaksIdx(1)) & (1:length(U1) <= peaksIdx(end));

        % Use logical index to keep only the desired elements
        U1 = U1(keepIndices);
        U2 = U2(keepIndices);
        U3 = U3(keepIndices);

        % Generation of one voltage vector from three 
        for j = 1:length(U1)
            if U1(j) < 9.3033e-04
                U(j, :) = U1(j);
            elseif U1(j) >= 9.3033e-04 && U2(j) < 0.0783
                U(j, :) = U2(j);
            else
                U(j, :) = U3(j);
            end
        end

        R_Values{i} = U / 0.2;
        disp(['Data loaded successfully from ', matFiles(i).name]);
    end
    disp('All data loaded successfully.');
    R_Overall = R_Values;

catch
    disp('Error loading one or more .mat files.');  % Error
end

% Save the processed data to a .mat file
save('15um_ProcessedData.mat', 'R_Overall');
