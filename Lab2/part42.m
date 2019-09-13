%% data
load('./data_lab2/cities.dat')

w = randn(10,2);
epochs =20;
step = 0.2;


%% training

for i=1:epochs + 1
    
    neighborhood = round(2 - ((i-1)*0.1));
    
   for j=1:10%cities 
       animal = props(j,:);
       
       best = inf;
       bestindex = 0;
       for k = 1:10 %weights
           sub = animal - w(k,:);
           dif = sub * sub';
           if dif < best
               best = dif;
               bestindex = k;
           end
       end
       
       %neighborhood 50 or 50/2??
       for p = max(1,bestindex-neighborhood):min(100,bestindex+neighborhood)
          w(p,:) = w(p,:) + step.*(animal - w(p,:)); %last p or bestindex?, I guess p
       end
       
   end
    
end

pos = zeros(1,32);

   for j=1:32 %animals 
       animal = props(j,:);
            
       best = inf;
       bestindex = 0;
       for k = 1:100 %weights
           sub = animal - w(k,:);
           dif = sub * sub';
           if dif < best
               best = dif;
               bestindex = k;
           end
       end
       pos(j) = bestindex;
       
       
   end

   [~,index] = sort(pos);
   
   disp(animalnames(index))
   
   
   