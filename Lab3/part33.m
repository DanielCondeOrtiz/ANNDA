part31

num_of_figure = 0;


[num_att_total, num_att_el] = size(attractors);

%% part 331

E_att = zeros(num_att_total,1);

for num_att = 1:num_att_total
    E_att(num_att) = -sum(sum(attractors(num_att,:)'*attractors(num_att,:).*w));
end

%% part 332

[num_dist_total, num_dist_el] = size(xd);

E_dist = zeros(num_dist_total,1);

for num_dist = 1:num_dist_total
    E_dist(num_dist) = -sum(sum(xd(num_dist,:)'*xd(num_dist,:).*w));
end

%% part 333


load('pict.dat');

p1 = pict(1:1024);
p2 = pict(1025:2048);
p3 = pict(2049:3072);
p4 = pict(3073:4096);
p5 = pict(4097:5120);
p6 = pict(5121:6144);
p7 = pict(6145:7168);
p8 = pict(7169:8192);
p9 = pict(8193:9216);
p10 = pict(9217:10240);
p11 = pict(10241:11264);

% imshow(reshape(p1,[32 32]))

p_train = [p1;p2;p3];


[num_of_patterns, num_of_elements] = size(p_train);

w = zeros(num_of_elements);

for p = 1:num_of_patterns
    w = w + p_train(p,:)'*p_train(p,:);
end

w = w./1024;

%w = w-diag(diag(w));


p_test = [p10;p11];

limit = 10000;

[num_of_patterns, num_of_elements] = size(p_test);

E_sequ = cell(num_of_patterns,1);

for p = 1:num_of_patterns
    converged = 0;
    epoch = 0; 
    E = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        
        unit = randi([1, num_of_elements]);
        
        update = sign(w(unit,:)*p_test(p,:)')';
        p_test(p,unit) = update;
        
        E = [E;-sum(sum(p_test(p,:)'*p_test(p,:).*w))];
        
        if mod(epoch,num_of_elements) == 0
            update = sign(w*p_test(p,:)')';            
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            end
        end
        
%         % plot the current image
%         if mod(epoch,100) == 0
% %             num_of_figure = num_of_figure + 1;
% %             figure(num_of_figure)
%             imshow(reshape(p_test(p,:),[32 32]))
%         end
    end
    E_sequ{p} = E;
end

num_of_figure = num_of_figure +1;
figure(num_of_figure)
for e = 1:2
    E = E_sequ{e};
    num_e = numel(E);
    plot([1:1:num_e],E)
    hold on
end

title('Energy for sequential update')
legend('P10','P11')
xlabel('Iterations')
ylabel('Energy')

hold off


%% part 334


[~, num_of_elements] = size(p_train);

% w = zeros(num_of_elements);

w = randn(num_of_elements);

w = w./1024;

% for p = 1:num_of_patterns
%     w = w + p_train(p,:)'*p_train(p,:);
% end

%w = w-diag(diag(w));

p_test = [p10;p11];

limit = 10000;

[num_of_patterns, num_of_elements] = size(p_test);

E_sequ = cell(num_of_patterns,1);

for p = 1:num_of_patterns
    converged = 0;
    epoch = 0; 
    E = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        
        unit = randi([1, num_of_elements]);
        
        update = sign(w(unit,:)*p_test(p,:)')';
        p_test(p,unit) = update;
        
        E = [E;-sum(sum(p_test(p,:)'*p_test(p,:).*w))];
        
        if mod(epoch,num_of_elements) == 0
            update = sign(w*p_test(p,:)')';            
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            end
        end
        
%         % plot the current image
%         if mod(epoch,100) == 0
% %             num_of_figure = num_of_figure + 1;
% %             figure(num_of_figure)
%             imshow(reshape(p_test(p,:),[32 32]))
%         end
    end
    E_sequ{p} = E;
end

num_of_figure = num_of_figure +1;
figure(num_of_figure)

for e = 1:2
    E = E_sequ{e};
    num_e = numel(E);
    plot([1:1:num_e],E)
    hold on
end



title('Energy for sequential update with random weights')
legend('P10','P11')
xlabel('Iterations')
ylabel('Energy')

hold off

%% part 335

[~, num_of_elements] = size(p_train);

% w = zeros(num_of_elements);

w = randn(num_of_elements);

w = 0.5*(w+w');

w = w./1024;


% for p = 1:num_of_patterns
%     w = w + p_train(p,:)'*p_train(p,:);
% end

%w = w-diag(diag(w));

p_test = [p10;p11];

limit = 10000;

[num_of_patterns, num_of_elements] = size(p_test);

E_sequ = cell(num_of_patterns,1);

for p = 1:num_of_patterns
    converged = 0;
    epoch = 0; 
    E = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        
        unit = randi([1, num_of_elements]);
        
        update = sign(w(unit,:)*p_test(p,:)')';
        p_test(p,unit) = update;
        
        E = [E;-sum(sum(p_test(p,:)'*p_test(p,:).*w))];
        
        if mod(epoch,num_of_elements) == 0
            update = sign(w*p_test(p,:)')';            
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            end
        end
        
%         % plot the current image
%         if mod(epoch,100) == 0
% %             num_of_figure = num_of_figure + 1;
% %             figure(num_of_figure)
%             imshow(reshape(p_test(p,:),[32 32]))
%         end
    end
    E_sequ{p} = E;
end

num_of_figure = num_of_figure +1;
figure(num_of_figure)

for e = 1:2
    E = E_sequ{e};
    num_e = numel(E);
    plot([1:1:num_e],E)
    hold on
end

title('Energy for seq. update with random sym. weights')
legend('P10','P11')
xlabel('Iterations')
ylabel('Energy')

hold off