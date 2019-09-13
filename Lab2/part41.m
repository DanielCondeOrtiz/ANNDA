%% data
load('./data_lab2/animals.dat')
props = reshape(animals', 84,32)';

fileID = fopen('./data_lab2/animalnames.txt');
animalnames = textscan(fileID,'%s');
animalnames = animalnames{1};
fclose(fileID);

w = randn(100,84);
epochs =20;
step = 0.2;


%% training

for i=1:epochs + 1
    
    neighborhood = round(50 - ((i-1)*2.5));
    
   for j=1:32%animals 
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
   
   
   