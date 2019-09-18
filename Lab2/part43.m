%% data
close all
clear all

load('./data_lab2/votes.dat')
votes = reshape(votes, 31, 349)';

load('./data_lab2/mpparty.dat')
load('./data_lab2/mpsex.dat')
load('./data_lab2/mpdistrict.dat')

fileID = fopen('./data_lab2/mpnames.txt');
names = textscan(fileID,'%s','Delimiter',{'\n'});
names = names{1};
fclose(fileID);


w = randn(100,31); %10x10 grid, x =10, y=1
epochs =20;
step = 0.2;

%% training

for i=1:epochs + 1
    
    neighborhood = round(2 - ((i-1)*0.1));
    
    for j=1:349%mps
        vote = votes(j,:);
        
        best = inf;
        bestindex = 0;
        for k = 1:100 %weights
            sub = vote - w(k,:);
            dif = sub * sub';
            if dif < best
                best = dif;
                bestindex = k;
            end
        end
        
        y = mod(bestindex,10);
        x= (bestindex-y)/10;
        
        %x
        %neighborhood 50 or 50/2??
        for px = max(1,x-neighborhood):min(10,x+neighborhood)
            %y
            for py = max(1,y-neighborhood):min(10,y+neighborhood)
                if (px*10+py) < 101
                    w(px*10+py,:) = w(px*10+py,:) + step.*(vote - w(px*10+py,:)); %last p or bestindex?, I guess p
                end
            end
        end
        
        
        
        
    end
    
end

pos = zeros(1,10);

for j=1:349%mps
    vote = votes(j,:);
    
    best = inf;
    bestindex = 0;
    for k = 1:100 %weights
        sub = vote - w(k,:);
        dif = sub * sub';
        if dif < best
            best = dif;
            bestindex = k;
        end
    end
    pos(j) = bestindex;
    
    
end

[~,index] = sort(pos);

% If this is set to false, only the average of each category will be
% plotted - otherwise there are 42 plots
plot_everything = false;

position = zeros(349,2);
for i = 1:349
    position(i,2) = mod(pos(i),10);
    if position(i,2)==0
        position(i,1) = (pos(i)-position(i,2))/10 - 1;
    else
        position(i,1) = (pos(i)-position(i,2))/10;
    end
end

f_count = 0;
for separator = 1:3
    % separator = 1;
    i = 0;
    switch separator
        case 1
            sep = mpdistrict;
        case 2
            sep = mpparty+1;
        case 3
            sep = mpsex+1;
    end
    pos_cell = cell(length(unique(sep)),1);
    while i<349
        i = i+1;
        pos_cell{sep(i)} = [pos_cell{sep(i)};position(i,:)];
    end
    
    if plot_everything
        for i = 1:length(unique(sep))
            f_count = f_count+1;
            figure(f_count)
            pos_hist = pos_cell{i};
    %         hist3(pos_cell{i},'Ctrs', {0:1:9 0:1:9},'CDataMode','auto','FaceColor','interp')
            histogram2(pos_hist(:,1), pos_hist(:,2),[10,10], 'XBinEdges', [0:1:9], 'YBinEdges', [0:1:9], 'FaceColor', 'flat')
            switch separator
                case 1
                    title_text = sprintf('SOM position separated by district for district %i', i);
                    title(title_text);
                case 2
                    title_text = sprintf('SOM position separated by party for party %i', i-1);
                    title(title_text);
                case 3
                    title_text = sprintf('SOM position separated by sex for sex %i', i-1);
                    title(title_text);
            end
        end
    end
    
    
    mp = zeros(length(unique(sep)),3);
    for i = 1:length(unique(sep))
        pos_avg = pos_cell{i};
        mp(i,3) = size(pos_cell{i},1);
        if mp(i,3) == 1
            mp(i,1:2) = pos_avg;
        else
            mp(i,1:2) = mean(pos_avg);
        end
    end
    
    cmap = hsv;
    cmap = cmap(int16(linspace(1,54,numel(unique(sep)))),:);
    
    f_count = f_count+1;
    
    figure(f_count)
    for i = 1:length(unique(sep))
        pos_avg = mp(i,1:2);
        for j = 1:mp(i,3)-1
            pos_avg = [pos_avg;mp(i,1:2)];
        end
%         hist3(pos_avg,'Ctrs', {0:1:9 0:1:9},'CDataMode','auto','FaceColor',cmap(i,:))
        pos_hist = pos_avg;
        histogram2(pos_hist(:,1), pos_hist(:,2),[10,10], 'XBinEdges', [0:1:9], 'YBinEdges', [0:1:9], 'FaceColor', cmap(i,:), 'FaceAlpha',0.9)        
        hold on
    end
    
    hold off
    switch separator
        case 1
            title('Average vote for each district')
            leg = {};
            for i = 1:numel(unique(sep))
                leg = [leg, "District "+i];
            end
            legend(leg);
        case 2
            title('Average vote for each party')
            leg = {};
            for i = 1:numel(unique(sep))
                leg = [leg, "Party "+(i-1)];
            end
            legend(leg);
        case 3
            legend('Male', 'Female');
            title('Average vote for each sex')
    end
end
