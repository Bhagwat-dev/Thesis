[file, path] = uigetfile('*.txt', 'Select a text file');

% Check if the user clicked 'Cancel'
if isequal(file, 0) || isequal(path, 0)
    disp('User canceled file selection. Exiting.');
    return;
end

% Construct the full file path
fullFilePath = fullfile(path, file);

try
    % Open the file for reading
    fileID = fopen(fullFilePath, 'r');
    
    % To skip the first three rows
    for i = 1:3
        fgetl(fileID);
    end
    
    % Define the format of each line
    formatSpec = '%s %s %s %s %s';
    
    % Read the data starting from the fourth row
    data = textscan(fileID, formatSpec);
    
    % Close the file
    fclose(fileID);
    
    % Extract the columns from the data
    Time = str2double(data{1});
    U1 = str2double(data{2});
    U2 = str2double(data{3});
    U3 = str2double(data{4});
    USensor = str2double(data{5});
    
    [originalPath, originalName, ~] = fileparts(fullFilePath);
    
    % Save the extracted data to a .mat file in the same folder
    matFileName = fullfile(originalPath, [originalName]);
    save(matFileName, 'Time', 'U1', 'U2', 'U3', 'USensor');
    
    disp(['Processing complete. Data saved to ', matFileName]);
    
catch
    % Error
    disp('Error reading the Txt file.');                                                                                                        
end