%% data
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

%I don't know how to plot this




