%% data
load('./data_lab2/cities.dat')

w = randn(10,2);
epochs =20;
step = 0.2;


%% training

for i=1:epochs + 1
    
    neighborhood = round(2 - ((i-1)*0.1));
    
    for j=1:10%cities
        city = cities(j,:);
        
        best = inf;
        bestindex = 0;
        for k = 1:10 %weights
            sub = city - w(k,:);
            dif = sub * sub';
            if dif < best
                best = dif;
                bestindex = k;
            end
        end
        
        %neighborhood 50 or 50/2??
        for p = max(1,bestindex-neighborhood):min(10,bestindex+neighborhood)
            w(p,:) = w(p,:) + step.*(city - w(p,:)); %last p or bestindex?, I guess p
        end
        
        if bestindex-neighborhood <= 0
            w(10,:) = w(10,:) + step.*(city - w(10,:));
            if bestindex-neighborhood == -1
                w(9,:) = w(9,:) + step.*(city - w(9,:));
            end
        end
        
        if bestindex+neighborhood >= 11
            w(1,:) = w(1,:) + step.*(city - w(1,:));
            if bestindex+neighborhood == 12
                w(2,:) = w(2,:) + step.*(city - w(2,:));
            end
        end
        
    end
    
end

pos = zeros(1,10);

for j=1:10 %animals
    city = cities(j,:);
    
    best = inf;
    bestindex = 0;
    for k = 1:10 %weights
        sub = city - w(k,:);
        dif = sub * sub';
        if dif < best
            best = dif;
            bestindex = k;
        end
    end
    pos(j) = bestindex;
    
    
end

[~,index] = sort(pos);
citiessorted = cities(index,:);

plot(cities(:,1),cities(:,2),'*r')
hold on
plot(citiessorted(:,1),citiessorted(:,2))
title('Shortest path between cities')
hold off
