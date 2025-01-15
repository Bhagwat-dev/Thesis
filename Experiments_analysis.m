clear all
close all
clc
hold all
Farbe = ['b','r','k','m','y','r','m','b','b','k'];

%% Variable Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Frequency = 100;                  %[Hz]                               
Amplitude = [10 20 25];           % Amplitude
CountofAmplitude = 3;             %[um]                                                                                                             
Timeframe = 500 * 10^(-6);        %[s]                                                                                                                  
LimitingResistance = 0.03;        %[Ohm]
n = 500;                          % Number of Dividing Areas
HistogramResolution = 50;         % Counts of maximum resistance per area values
FilteringResolution = 25;         % Soothening Curve Resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Entire Data
folderPath = 'Your_File_Path'; 
loadedData = load(fullfile(folderPath, '15um_ProcessedData.mat'), 'R_Overall');  % Load the .mat file

% Ensure that R_Overall is loaded into the workspace
if isfield(loadedData, 'R_Overall')
    R_Overall = loadedData.R_Overall;
else
    error('R_Overall not found in the loaded .mat file.');
end

% Display the structure of R_Overall for debugging
disp('Structure of R_Overall:');
disp(R_Overall);

%% Analysis
figure1 = figure('Units','normalized','Position',[0.1 0.1 1 1]);  
figure2 = figure('Units','normalized','Position',[0.1 0.1 0.4 0.4]);
figure3 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
AllXIntersectionPoint = zeros(3, 5);

for runIndex = 1:1  % Updated loop to iterate through all amplitudes
    if(runIndex == 1)
        AmplitudeValue = 10;
        currentColor = [0, 1, 0]; 
    elseif(runIndex == 2)
        AmplitudeValue = 20;
        currentColor = [0, 0, 1]; 
    elseif(runIndex == 3)
        AmplitudeValue = 25;
        currentColor = [1, 0, 0];
    end

    %% Calculating max Resistance per cycle
    figure(figure1);
    hold all;  % Number of Dividing Area
    NumberofResistancepercycle = round((1 / Frequency) / Timeframe);  % Cycletime frame/Resistance Measuring Time frame (0.00125 / 50us)
    TotalNumberofCycles = 500000;  % Total number of cycles

    MaximumResistance = zeros(10, TotalNumberofCycles);

    for setIndex = 1:10
        % Debugging information to ensure correct indexing
        disp(['runIndex: ', num2str(runIndex), ', setIndex: ', num2str(setIndex)]);
        
        % Access current resistance values
        currentResistanceValues = R_Overall{setIndex};
        numResistanceValues = length(currentResistanceValues);

        for cycle = 1:TotalNumberofCycles
            startIndex = (cycle - 1) * NumberofResistancepercycle + 1;  
            endIndex = cycle * NumberofResistancepercycle;

            % Ensure indices are within bounds
            if endIndex > numResistanceValues
                endIndex = numResistanceValues;
            end

            if startIndex > numResistanceValues
                break;
            end

            MaximumResistance(setIndex, cycle) = max(currentResistanceValues(startIndex:endIndex));
        end
    end

    %% Plotting Cycle vs Max Resistance
    subplot(6, 12, (runIndex - 1) * 24 + 1:((runIndex - 1) * 24 + 1) + 11)
    plot(1:TotalNumberofCycles, MaximumResistance, '-', 'LineWidth', 0.25);
    title(['\bf{Amplitude = ', num2str(AmplitudeValue), ' µm}'], 'FontSize', 12);
    xlabel('Cycles', 'FontSize', 8);
    ylabel('R (Ohm)', 'FontSize', 8);
    set(gca, 'FontSize', 8);
    xlim([1, TotalNumberofCycles]);
    ylim([0, 4]);
    grid on;

%% Add n vertical lines with equal distribution

        deltacycle = TotalNumberofCycles / n;
    for i = 1:(n+1)
        x_value = (i - 1) * deltacycle;     
        line([x_value, x_value], ylim, 'Color', 'r', 'LineStyle', '-');  
    end

%% Splitting MaximumResistance within each window

        setSize = numel(MaximumResistance) / n;
        MaximumResistanceperArea = cell(1, n);
        MeanResistanceValues = cell(1, n);
        MinimumResistanceValues = cell(1, n);
        MaximumResistanceValues = cell(1, n);
        HundredperResistanceValues = cell(1, n);
        FiftypermaxResistanceValues = cell(1, n);
        TenpermaxResistanceValues = cell(1, n);
        PointOnepermaxResistanceValues = cell(1, n);
        OnepermaxResistanceValues = cell(1, n);
        TwentyfivepermaxResistanceValues = cell(1, n);
        SeventyfivepermaxResistanceValues = cell(1, n);

    for i = 1:n
        % Calculate indices for each set
        startIndex = round((i-1) * setSize) + 1;
        endIndex = round(i * setSize);

        % Extract the subset for the current set
        MaximumResistanceperArea{i} = MaximumResistance(startIndex:endIndex);
        MeanResistanceValues{i} = mean(MaximumResistanceperArea{i});
        MinimumResistanceValues{i} = min(MaximumResistanceperArea{i});
        MaximumResistanceValues{i} = max(MaximumResistanceperArea{i});

    end

%% Failure Histogram

    for i = 1:n

    lines = linspace(min(MaximumResistanceperArea{i}), max(MaximumResistanceperArea{i}), HistogramResolution+1);

    counts = hist(MaximumResistanceperArea{i}, lines);
    [maxValue, maxIndex] = max(counts);                                     % Count the number of points in each horizontal line

    % Find the first index where count is non-zero
    NonZero = find(counts > 0, 1);

    % Now you can get the corresponding value from lines
    FirstElement = lines(NonZero);

    counts_normalized = counts / sum(counts);
    cumulative_sum = fliplr(cumsum(fliplr(counts_normalized)));
    
%     index_Pointone_percent = find(cumulative_sum <= 0.001, 1, 'first');     % Calculating Point one percent Resistance Value
%     PointOnepermaxResistanceValue = lines(index_Pointone_percent);
%     PointOnepermaxCountsValue = counts(index_Pointone_percent);
    
        index_One_percent = find(cumulative_sum <= 0.01, 1, 'first');       % Calculating One percent Resistance Value
    OnepermaxResistanceValue = lines(index_One_percent);
    
        index_Ten_percent = find(cumulative_sum <= 0.1, 1, 'first');        % Calculating Ten percent Resistance Value
    TenpermaxResistanceValue = lines(index_Ten_percent);
    
        index_Fifety_percent = find(cumulative_sum <= 0.5, 1, 'first');     % Calculating Fifety percent Resistance Value
    FiftypermaxResistanceValue = lines(index_Fifety_percent);
    
        index_Twentyfive_percent = find(cumulative_sum <= 0.25, 1, 'first');     % Calculating Twenty Five percent Resistance Value
    TwentyfivepermaxResistanceValue = lines(index_Twentyfive_percent);
    
        index_Seventyfive_percent = find(cumulative_sum <= 0.75, 1, 'first');     % Calculating Seventy Five percent Resistance Value
    SeventyfivepermaxResistanceValue = lines(index_Seventyfive_percent);
   
    
%% Assigning Variables
%     % Find the location of the maximum count
%     [maxValue, maxIndex] = max(counts);
%     HundredperResistanceValue = lines(maxIndex);
% 
%     FiftyperMaxValue = 0.5 * maxValue;
%     FiftyperIndex = find(counts >= FiftyperMaxValue, 1, 'last');
%     FiftypermaxResistanceValue = lines(FiftyperIndex);
% 
%     TenperMaxValue = 0.1 * maxValue;
%     TenperIndex = find(counts >= TenperMaxValue, 1, 'last');
%     TenpermaxResistanceValue = lines(TenperIndex);
% 
%     PointOneperMaxValue = 0.01 * maxValue;
%     PointOneperIndex = find(counts >= PointOneperMaxValue, 1, 'last');
%     PointOnepermaxResistanceValue = lines(PointOneperIndex);
% 
%     NinetyperMaxValue = 0.9 * maxValue;
%     NinetyperIndex = find(counts >= NinetyperMaxValue, 1, 'last');
%     NinetypermaxResistanceValue = lines(NinetyperIndex);

    % Store the maxResistanceValue in the cell array
%     HundredperResistanceValues{i} = HundredperResistanceValue;
    FiftypermaxResistanceValues{i} = FiftypermaxResistanceValue;
    TenpermaxResistanceValues{i} = TenpermaxResistanceValue;
%     PointOnepermaxResistanceValues{i} = PointOnepermaxResistanceValue;
    OnepermaxResistanceValues{i} = OnepermaxResistanceValue;
    TwentyfivepermaxResistanceValues{i} = TwentyfivepermaxResistanceValue;
    SeventyfivepermaxResistanceValues{i} = SeventyfivepermaxResistanceValue;


%% Plotting the Area Graphs

     if(i==1)
        subplot(6, 12, (runIndex*12)+((runIndex-1)*12+1):((runIndex*12)+((runIndex-1)*12+1))+1)
        plot(lines, counts, 'LineWidth', 1);
        title(['\bf{Area }',num2str(i)], 'FontSize', 8)
        set(gca, 'FontSize', 6);
        xlim([FirstElement, max(lines)]);
        ylim([0, 1.2*max(counts)]);
        grid on;
        hold on;
        plot([FiftypermaxResistanceValue, FiftypermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TenpermaxResistanceValue, TenpermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        plot([OnepermaxResistanceValue, OnepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TwentyfivepermaxResistanceValue, TwentyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([SeventyfivepermaxResistanceValue, SeventyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        hold off;
        text(FiftypermaxResistanceValue, maxValue, '50%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TenpermaxResistanceValue, maxValue, '10%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
        text(OnepermaxResistanceValue, maxValue, '1%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TwentyfivepermaxResistanceValue, maxValue, '25%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
%         text(SeventyfivepermaxResistanceValue, maxValue, '75%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
     end

     if(i==round(n/4))
        subplot(6, 12, ((runIndex*12)+((runIndex-1)*12+1))+3:((runIndex*12)+((runIndex-1)*12+1))+4)
        plot(lines, counts, 'LineWidth', 1);
        title(['\bf{Area }',num2str(i)], 'FontSize', 8)
        set(gca, 'FontSize', 6);
        xlim([FirstElement, max(lines)]);
        ylim([0, 1.2*max(counts)]);
        grid on;
        hold on;
        plot([FiftypermaxResistanceValue, FiftypermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TenpermaxResistanceValue, TenpermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        plot([OnepermaxResistanceValue, OnepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TwentyfivepermaxResistanceValue, TwentyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([SeventyfivepermaxResistanceValue, SeventyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        hold off;

        text(FiftypermaxResistanceValue, maxValue, '50%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TenpermaxResistanceValue, maxValue, '10%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
        text(OnepermaxResistanceValue, maxValue, '1%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TwentyfivepermaxResistanceValue, maxValue, '25%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
%         text(SeventyfivepermaxResistanceValue, maxValue, '75%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');

      end

      if(i==round(3*(n/4)))
        subplot(6, 12, ((runIndex*12)+((runIndex-1)*12+1))+6:((runIndex*12)+((runIndex-1)*12+1))+7)
        plot(lines, counts, 'LineWidth', 1);
        title(['\bf{Area }',num2str(i)], 'FontSize', 8)
        set(gca, 'FontSize', 6);
        xlim([FirstElement, max(lines)]);
        ylim([0, 1.2*max(counts)]);
        grid on;
        hold on;
        plot([FiftypermaxResistanceValue, FiftypermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TenpermaxResistanceValue, TenpermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        plot([OnepermaxResistanceValue, OnepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TwentyfivepermaxResistanceValue, TwentyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([SeventyfivepermaxResistanceValue, SeventyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        hold off;
        text(FiftypermaxResistanceValue, maxValue, '50%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TenpermaxResistanceValue, maxValue, '10%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
        text(OnepermaxResistanceValue, maxValue, '1%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TwentyfivepermaxResistanceValue, maxValue, '25%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
%         text(SeventyfivepermaxResistanceValue, maxValue, '75%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
      end

       if(i==n)
        subplot(6, 12, ((runIndex*12)+((runIndex-1)*12+1))+9:((runIndex*12)+((runIndex-1)*12+1))+10)
        plot(lines, counts, 'LineWidth', 1);
        title(['\bf{Area }',num2str(i)], 'FontSize', 8)
        set(gca, 'FontSize', 6);
        xlim([FirstElement, max(lines)]);
        ylim([0, 1.2*max(counts)]);
        grid on;
        hold on;
        plot([FiftypermaxResistanceValue, FiftypermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TenpermaxResistanceValue, TenpermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        plot([OnepermaxResistanceValue, OnepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([TwentyfivepermaxResistanceValue, TwentyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
%         plot([SeventyfivepermaxResistanceValue, SeventyfivepermaxResistanceValue], [0, maxValue], 'r--', 'LineWidth', 1);
        hold off;
        text(FiftypermaxResistanceValue, maxValue, '50%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TenpermaxResistanceValue, maxValue, '10%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
        text(OnepermaxResistanceValue, maxValue, '1%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 8, 'Color', 'black');
%         text(TwentyfivepermaxResistanceValue, maxValue, '25%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
%         text(SeventyfivepermaxResistanceValue, maxValue, '75%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 6, 'Color', 'black');
      end
    
        xlabel('R (Ohm)', 'FontSize', 8);
        ylabel('Count', 'FontSize', 8);
        

    end


%% Assigning Variables


    %    subplot(4, 20, [41:53 61:73])
       MidPointIndexAll = [];
%        curveHundrerpermaxResistanceValueAll = [];
       curveTenpermaxResistanceValueAll = [];
       curveFiftypermaxResistanceValueAll = [];
%        curvePointOnepermaxResistanceValueAll = [];
       curveOnepermaxResistanceValueAll = [];
%        curveNinetypermaxResistanceValueAll = [];
       curveTwentyfivepermaxResistanceValueAll = [];
       curveSeventyfivepermaxResistanceValueAll = [];

       hold on;
        % Split the data into sets
    for i = 1:n

        Set = TotalNumberofCycles/n;
        % Calculate indices for each set
        startIndex = round((i-1) * Set) + 1;
        endIndex = round(i * Set);

        % Calculate midpoint
        midpointIndex = round((startIndex + endIndex) / 2);

%         HundredperResistanceValue = HundredperResistanceValues{i};
%         MeanResistanceValue = MeanResistanceValues{i};
%         MinimumResistanceValue = MinimumResistanceValues{i};
%         MaximumResistanceValue = MaximumResistanceValues{i};
        FiftypermaxResistanceValue = FiftypermaxResistanceValues{i};
        TenpermaxResistanceValue = TenpermaxResistanceValues{i};
%         PointOnepermaxResistanceValue = PointOnepermaxResistanceValues{i};
        OnepermaxResistanceValue = OnepermaxResistanceValues{i};
        TwentyfivepermaxResistanceValue = TwentyfivepermaxResistanceValues{i};
        SeventyfivepermaxResistanceValue = SeventyfivepermaxResistanceValues{i};



        line([endIndex, endIndex], [0 0.1], 'Color', [0.9 0.9 0.9], 'LineStyle', '--');  
    %     plot([startIndex, endIndex], [HundredperResistanceValue, HundredperResistanceValue], '--', 'LineWidth', 1.5, 'Color', [0 0 1]);
    %     plot([startIndex, endIndex], [MeanResistanceValue, MeanResistanceValue], '--', 'LineWidth', 1.5, 'Color', [0 0 1]);
    %     plot([startIndex, endIndex], [MinimumResistanceValue, MinimumResistanceValue], '--', 'LineWidth', 1.5, 'Color', [0 0 1]);
    % %   plot([startIndex, endIndex], [MaximumResistanceValue, MaximumResistanceValue], '--', 'LineWidth', 1.5, 'Color', 'g');
    %     plot([startIndex, endIndex], [FiftypermaxResistanceValue, FiftypermaxResistanceValue], '--', 'LineWidth', 1.5, 'Color', [0 0 1]);
    %     plot([startIndex, endIndex], [TenpermaxResistanceValue, TenpermaxResistanceValue], '--', 'LineWidth', 1.5, 'Color', [0 0 1]);


         % Plot solid curve joining midpoints
        if i < n
            nextMidpointIndex = round((endIndex + round((i+1) * Set + 1)) / 2);
            MidPointIndex = [midpointIndex, nextMidpointIndex];
    %         curveMaxResistanceValue = [HundredperResistanceValue, HundredperResistanceValues{i+1}];
    %         curveMeanResistanceValue = [MeanResistanceValue, MeanResistanceValues{i+1}];
    %         curveMinResistanceValue = [MinimumResistanceValue, MinimumResistanceValues{i+1}];
              curveFiftypermaxResistanceValue = [FiftypermaxResistanceValue, FiftypermaxResistanceValues{i+1}];
              curveTenpermaxResistanceValue = [TenpermaxResistanceValue, TenpermaxResistanceValues{i+1}];
%               curvePointOnepermaxResistanceValue = [PointOnepermaxResistanceValue, PointOnepermaxResistanceValues{i+1}];
              curveOnepermaxResistanceValue = [OnepermaxResistanceValue, OnepermaxResistanceValues{i+1}];
%               curveHundrerpermaxResistanceValue = [HundredperResistanceValue, HundredperResistanceValues{i+1}];
              curveTwentyfivepermaxResistanceValue = [TwentyfivepermaxResistanceValue, TwentyfivepermaxResistanceValues{i+1}];
              curveSeventyfivepermaxResistanceValue = [SeventyfivepermaxResistanceValue, SeventyfivepermaxResistanceValues{i+1}];

              MidPointIndexAll = [MidPointIndexAll, MidPointIndex];
%               curveHundrerpermaxResistanceValueAll = [curveHundrerpermaxResistanceValueAll, curveHundrerpermaxResistanceValue];
              curveTenpermaxResistanceValueAll = [curveTenpermaxResistanceValueAll, curveTenpermaxResistanceValue];
              curveFiftypermaxResistanceValueAll = [curveFiftypermaxResistanceValueAll, curveFiftypermaxResistanceValue];
%               curvePointOnepermaxResistanceValueAll = [curvePointOnepermaxResistanceValueAll, curvePointOnepermaxResistanceValue];
              curveOnepermaxResistanceValueAll = [curveOnepermaxResistanceValueAll, curveOnepermaxResistanceValue];
              curveTwentyfivepermaxResistanceValueAll = [curveTwentyfivepermaxResistanceValueAll, curveTwentyfivepermaxResistanceValue];
              curveSeventyfivepermaxResistanceValueAll = [curveSeventyfivepermaxResistanceValueAll, curveSeventyfivepermaxResistanceValue];


    %           plot(MidPointIndex, curveMaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - Max Resistance');
    %           plot(MidPointIndex, log10(curveMaxResistanceValue), '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - Max Resistance');
    %           plot(MidPointIndex, curveMeanResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - Mean Resistance');
    %           plot(MidPointIndex, curveMinResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - Min Resistance');
    %           plot(MidPointIndex, curveFiftypermaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - 50% Max Resistance');
    %           plot(MidPointIndex, curveTenpermaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - 10% Max Resistance');
    %           plot(MidPointIndex, curvePointOnetypermaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - 10% Max Resistance');
    %           plot(MidPointIndex, curveNinetypermaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - 10% Max Resistance');
    %           plot(MidPointIndex, curveHundrerpermaxResistanceValue, '-', 'LineWidth', 1.5, 'Color', [0 0 0], 'DisplayName', 'Curve - 10% Max Resistance');
        end

    end
    



%% Plotting Failure points
figure(figure2);

            semilogy(MidPointIndexAll, curveOnepermaxResistanceValueAll, '.', 'MarkerSize', 6, 'Color', 0.4*currentColor, 'HandleVisibility', 'off'); hold on             % 1 Percent Resistance Graph
            semilogy(MidPointIndexAll, curveTenpermaxResistanceValueAll, '.', 'MarkerSize', 6, 'Color', 0.6*currentColor, 'HandleVisibility', 'off'); hold on            % 10 Percent Resistance Graph
            semilogy(MidPointIndexAll, curveTwentyfivepermaxResistanceValueAll, '.', 'MarkerSize', 6, 'Color', 0.7*currentColor, 'HandleVisibility', 'off'); hold on                % 25 Percent Resistance Graph
            semilogy(MidPointIndexAll, curveFiftypermaxResistanceValueAll, '.', 'MarkerSize', 6, 'Color', 0.8*currentColor, 'HandleVisibility', 'off'); hold on                % 50 Percent Resistance Graph 
            semilogy(MidPointIndexAll, curveSeventyfivepermaxResistanceValueAll, '.', 'MarkerSize', 6, 'Color', currentColor, 'HandleVisibility', 'off'); hold on                % 75 Percent Resistance Graph
 
%             subplot(1, 2, 2)
%             plot(MidPointIndexAll, curvePointOnepermaxResistanceValueAll, '-', 'LineWidth', 1.5, 'DisplayName', ['1 Percent Probability, Amplitude = ', num2str(AmplitudeValue), ' um' ], 'Color', 0.6*currentColor); hold on 
            
            
%% Smoothening Curve
            
            windowSize = FilteringResolution; 
            b = (1/windowSize)*ones(1,windowSize);
            a = 1;
%             Initialize variables
%             Filtered_PointOneper = filter(b,a,curvePointOnepermaxResistanceValueAll);
            Filtered_Tenper = filter(b,a,curveTenpermaxResistanceValueAll);
            Filtered_Fiftyper = filter(b,a,curveFiftypermaxResistanceValueAll);
            Filtered_Oneper = filter(b,a,curveOnepermaxResistanceValueAll);
            Filtered_Twentyfiveper = filter(b,a,curveTwentyfivepermaxResistanceValueAll);
            Filtered_Seventyfiveper = filter(b,a,curveSeventyfivepermaxResistanceValueAll);


%% Plotting Failure curve

            semilogy(MidPointIndexAll(25:end), Filtered_Oneper(25:end),  '-', 'LineWidth', 2.5, 'DisplayName', ['Trend Curve: 1% Probability, Amplitude = ', num2str(AmplitudeValue), ' µm' ], 'Color', 0.4*currentColor); hold on
            semilogy(MidPointIndexAll(25:end), Filtered_Tenper(25:end),  '-', 'LineWidth', 2.5, 'DisplayName', ['Trend Curve: 10% Probability, Amplitude = ', num2str(AmplitudeValue), ' µm' ], 'Color', 0.6*currentColor); hold on
            semilogy(MidPointIndexAll(25:end), Filtered_Twentyfiveper(25:end),  '-', 'LineWidth', 2.5, 'DisplayName', ['Trend Curve: 25% Probability, Amplitude = ', num2str(AmplitudeValue), ' µm' ], 'Color', 0.7*currentColor); hold on
            semilogy(MidPointIndexAll(25:end), Filtered_Fiftyper(25:end),  '-', 'LineWidth', 2.5, 'DisplayName', ['Trend Curve: 50% Probability, Amplitude = ', num2str(AmplitudeValue), ' µm' ], 'Color', 0.8*currentColor); hold on    
            semilogy(MidPointIndexAll(25:end), Filtered_Seventyfiveper(25:end),  '-', 'LineWidth', 2.5, 'DisplayName', ['Trend Curve: 75% Probability, Amplitude = ', num2str(AmplitudeValue), ' µm' ], 'Color', currentColor); hold on             
         
            
              %% Trend line//Previous Code
%             coefficientTenper = polyfit(MidPointIndexAll, curveTenpermaxResistanceValueAll, 1);
%             coefficientOneper = polyfit(MidPointIndexAll, curveOnepermaxResistanceValueAll, 1);
%             coefficientFiftyper = polyfit(MidPointIndexAll, curveFiftypermaxResistanceValueAll, 1);
%             coefficientTwentyfiveper = polyfit(MidPointIndexAll, curveTwentyfivepermaxResistanceValueAll, 1);
%             coefficientSeventyfiveper = polyfit(MidPointIndexAll, curveSeventyfivepermaxResistanceValueAll, 1);
%             trendlineTenper = polyval(coefficientTenper, MidPointIndexAll);
%             trendlineOneper = polyval(coefficientOneper, MidPointIndexAll);
%             trendlineFiftyper = polyval(coefficientFiftyper, MidPointIndexAll);
%             trendlineTwentyfiveper = polyval(coefficientTwentyfiveper, MidPointIndexAll);
%             trendlineSeventyfiveper = polyval(coefficientSeventyfiveper, MidPointIndexAll);

    %         hold on;
%             semilogy(MidPointIndexAll, trendlineOneper, '--', 'LineWidth', 1.5, 'DisplayName', ['Trend Line, Amplitude = ', num2str(AmplitudeValue), ' um' ], 'Color', 0.2*currentColor); hold on
%             semilogy(MidPointIndexAll, trendlineTenper, '--', 'LineWidth', 1.5, 'DisplayName', ['10 Percent Probability, Amplitude = ', num2str(AmplitudeValue), ' um' ], 'Color', 0.3*currentColor); hold on
%             semilogy(MidPointIndexAll, trendlineFiftyper, '--', 'LineWidth', 1.5, 'DisplayName', ['Trend Line, Amplitude = ', num2str(AmplitudeValue), ' um' ], 'Color', 0.5*currentColor); hold on
%             semilogy(MidPointIndexAll, trendlineTwentyfiveper, '--', 'LineWidth', 1.5, 'DisplayName', ['Trend Line, Amplitude = ', num2str(AmplitudeValue), ' um' ],'Color', 0.4*currentColor); hold on
%             semilogy(MidPointIndexAll, trendlineSeventyfiveper, '--', 'LineWidth', 1.5, 'DisplayName', ['Trend Line, Amplitude = ', num2str(AmplitudeValue), ' um' ], 'Color', currentColor); hold on
              
              xlabel('\bf{Cycles}', 'FontSize', 14);
              ylabel('\bf{Resistance (Ohm)}', 'FontSize', 14);
              ylim([10^(-3), 10^(1)]);
              xlim([0, 500000]);
              grid on;
%               title('Resistance vs Number of Cycle Analysis', 'FontSize', 18);
              hold on;
              set(gca, 'FontSize', 8);
              legend('Location', 'northwest');

        

%% Interpolated values         
        
        if runIndex == 1
         
                        Value1 = 450000;
                        Value2 = 500000;
            
            % Ten percent Interpolation
            [~, index1] = min(abs(MidPointIndexAll - Value1));
            nearestFilteredValue1 = Filtered_Tenper(index1);      
            [~, index2] = min(abs(MidPointIndexAll - Value2));
            nearestFilteredValue2 = Filtered_Tenper(index2);   
            x_interpolated_Ten = Value1 + ((LimitingResistance - nearestFilteredValue1) * (Value2-Value1)) / (nearestFilteredValue2 - nearestFilteredValue1);             
                        
                        
            % Twenty Five percent Interpolation
            [~, index3] = min(abs(MidPointIndexAll - Value1));
            nearestFilteredValue3 = Filtered_Twentyfiveper(index3);      
            [~, index4] = min(abs(MidPointIndexAll - Value2));
            nearestFilteredValue4 = Filtered_Twentyfiveper(index4);   
            x_interpolated_Twentyfive = Value1 + ((LimitingResistance - nearestFilteredValue3) * (Value2-Value1)) / (nearestFilteredValue4 - nearestFilteredValue3);
            
            % Fifety Percent Interpolation
            [~, index5] = min(abs(MidPointIndexAll - Value1));
            nearestFilteredValue5 = Filtered_Fiftyper(index5);      
            [~, index6] = min(abs(MidPointIndexAll - Value2));
            nearestFilteredValue6 = Filtered_Fiftyper(index6);   
            x_interpolated_Fifety = Value1 + ((LimitingResistance - nearestFilteredValue5) * (Value2-Value1)) / (nearestFilteredValue6 - nearestFilteredValue5);
            
             % Seventy Five Percent Interpolation
            [~, index7] = min(abs(MidPointIndexAll - Value1));
            nearestFilteredValue7 = Filtered_Seventyfiveper(index7);      
            [~, index8] = min(abs(MidPointIndexAll - Value2));
            nearestFilteredValue8 = Filtered_Seventyfiveper(index8);   
            x_interpolated_Seventyfive = Value1 + ((LimitingResistance - nearestFilteredValue7) * (Value2-Value1)) / (nearestFilteredValue8 - nearestFilteredValue7);
            
        end  
         
%% Intersecton Points

    YIntersectionPoint = LimitingResistance;

    x_intersection_1 = [];
    x_intersection_10 = [];
    x_intersection_50 = [];
    x_intersection_25 = [];
    x_intersection_75 = [];

    x_line = [1, TotalNumberofCycles];
    y_line = [YIntersectionPoint, YIntersectionPoint];

    x_intersections = [];
    for i = 2:numel(MidPointIndexAll)
        if (Filtered_Oneper(i-1) <= YIntersectionPoint && Filtered_Oneper(i) >= YIntersectionPoint) || (Filtered_Oneper(i-1) >= YIntersectionPoint && Filtered_Oneper(i) <= YIntersectionPoint)
            x_intersection_1(end+1) = interp1([Filtered_Oneper(i-1), Filtered_Oneper(i)], [MidPointIndexAll(i-1), MidPointIndexAll(i)], YIntersectionPoint);
        end
        if (Filtered_Tenper(i-1) <= YIntersectionPoint && Filtered_Tenper(i) >= YIntersectionPoint) || (Filtered_Tenper(i-1) >= YIntersectionPoint && Filtered_Tenper(i) <= YIntersectionPoint)
            x_intersection_10(end+1) = interp1([Filtered_Tenper(i-1), Filtered_Tenper(i)], [MidPointIndexAll(i-1), MidPointIndexAll(i)], YIntersectionPoint);
        end
        if (Filtered_Fiftyper(i-1) <= YIntersectionPoint && Filtered_Fiftyper(i) >= YIntersectionPoint) || (Filtered_Fiftyper(i-1) >= YIntersectionPoint && Filtered_Fiftyper(i) <= YIntersectionPoint)
            x_intersection_50(end+1) = interp1([Filtered_Fiftyper(i-1), Filtered_Fiftyper(i)], [MidPointIndexAll(i-1), MidPointIndexAll(i)], YIntersectionPoint);
        end
        if (Filtered_Twentyfiveper(i-1) <= YIntersectionPoint && Filtered_Twentyfiveper(i) >= YIntersectionPoint) || (Filtered_Twentyfiveper(i-1) >= YIntersectionPoint && Filtered_Twentyfiveper(i) <= YIntersectionPoint)
            x_intersection_25(end+1) = interp1([Filtered_Twentyfiveper(i-1), Filtered_Twentyfiveper(i)], [MidPointIndexAll(i-1), MidPointIndexAll(i)], YIntersectionPoint);
        end
        if (Filtered_Seventyfiveper(i-1) <= YIntersectionPoint && Filtered_Seventyfiveper(i) >= YIntersectionPoint) || (Filtered_Seventyfiveper(i-1) >= YIntersectionPoint && Filtered_Seventyfiveper(i) <= YIntersectionPoint)
            x_intersection_75(end+1) = interp1([Filtered_Seventyfiveper(i-1), Filtered_Seventyfiveper(i)], [MidPointIndexAll(i-1), MidPointIndexAll(i)], YIntersectionPoint);
        end

     end

    if numel(x_intersection_1) == 0
        AllXIntersectionPoint(runIndex, 1) = NaN;
    else
        AllXIntersectionPoint(runIndex, 1) = x_intersection_1(1);
    end
    if numel(x_intersection_10) == 0
        if runIndex == 1
            AllXIntersectionPoint(runIndex, 2) = x_interpolated_Ten;   
        else
            AllXIntersectionPoint(runIndex, 2) = NaN;
        end
     else
        AllXIntersectionPoint(runIndex, 2) = x_intersection_10(1);
    end
    if numel(x_intersection_50) == 0
        if runIndex == 1
            AllXIntersectionPoint(runIndex, 4) = x_interpolated_Fifety;
        else
        AllXIntersectionPoint(runIndex, 4) = NaN;
        end
    else
        AllXIntersectionPoint(runIndex, 4) = x_intersection_50(1);
    end
    if numel(x_intersection_25) == 0
       if runIndex == 1
        AllXIntersectionPoint(runIndex, 3) = x_interpolated_Twentyfive;
       else 
        AllXIntersectionPoint(runIndex, 3) = NaN;  
       end
    else
        AllXIntersectionPoint(runIndex, 3) = x_intersection_25(1);
    end
    if numel(x_intersection_75) == 0
        if runIndex == 1
        AllXIntersectionPoint(runIndex, 5) = x_interpolated_Seventyfive;
        else
        AllXIntersectionPoint(runIndex, 5) = NaN;
        end
    else
        AllXIntersectionPoint(runIndex, 5) = x_intersection_75(1);
    end


%% Plotting Intersection Points

    hold on;
    plot(AllXIntersectionPoint(runIndex, 1), YIntersectionPoint, 'k.', 'MarkerSize', 20, 'HandleVisibility', 'off');
    plot(AllXIntersectionPoint(runIndex, 2), YIntersectionPoint, 'k.', 'MarkerSize', 20, 'HandleVisibility', 'off');
    plot(AllXIntersectionPoint(runIndex, 3), YIntersectionPoint, 'k.', 'MarkerSize', 20, 'HandleVisibility', 'off');
    plot(AllXIntersectionPoint(runIndex, 4), YIntersectionPoint, 'k.', 'MarkerSize', 20, 'HandleVisibility', 'off');
    plot(AllXIntersectionPoint(runIndex, 5), YIntersectionPoint, 'k.', 'MarkerSize', 20, 'HandleVisibility', 'off');

    

    end

    figure(figure2);
        
     % Threshold Resistance line
    line([1, TotalNumberofCycles], [YIntersectionPoint, YIntersectionPoint], 'Color', 'k', 'LineStyle', '-', 'LineWidth', 2.5, 'DisplayName', 'Limiting Resistance');
    legend('Location', 'northwest'); hold off

    
    %% Saving the Fatigue points in excel file 
    [X, Y] = size(AllXIntersectionPoint);
    PointsMatrix = zeros(X + 1, Y + 1);
    PointsMatrix(2:end, 2:end) = AllXIntersectionPoint;
    PointsMatrix(2:end, 1) = [10; 20; 25]; % First column values
    PointsMatrix(1, 2:end) = [1; 10; 25; 50; 75]; % First row values
    xlswrite('IntersectionPoints_100Hz.xlsx', PointsMatrix);
    
    %% WOHLER CURVE

    figure(figure3);
    hold on;

    lineColors = {[0.1 0.1 0.1], [0.3 0.3 0.3], [0.5 0.5 0.5], [0.6 0.6 0.6], [0.7 0.7 0.7]};

    for i = 1:size(AllXIntersectionPoint, 2)
        plot(AllXIntersectionPoint(:,i), Amplitude, '.-', 'MarkerSize', 15, 'Color', lineColors{i}, 'LineWidth', 2);
    end
    
%     plot(AllXIntersectionPoint, Amplitude, 'o-', 'MarkerSize',4);
    xlabel('\bf{Cycles}', 'FontSize', 15);
    ylabel('\bf{Amplitude (µm)}', 'FontSize', 15);
    title(['{\bf WÖHLER CURVE} {\fontsize{15}(Limiting Resistance =  ', num2str(LimitingResistance), ' Ohm)}'], 'FontSize', 20);
%     text(0.5, 1, ['{\fontsize{15}(Limiting Resistance = ', num2str(LimitingResistance), ' Ohm)}'],'FontSize', 15, 'HorizontalAlignment', 'center', 'Units', 'normalized');
    ylim([0 30]);
    xlim([0 40*10^(5)]);
    grid on;
    legend('1% Probability', '10% Probability', '25% Probability', '50% Probability', '75% Probability', 'Location', 'NorthEast');

    
save('15um_all_data.mat');

