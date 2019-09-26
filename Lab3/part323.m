%% generate the data

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

%% training the weights

[num_of_patterns, num_of_elements] = size(p_train);

w = zeros(num_of_elements);

for p = 1:num_of_patterns
    w = w + p_train(p,:)'*p_train(p,:);
end

w = w-diag(diag(w));

%% part 3: degraded pattern, original Hopfield

num_of_figure = 0;

p_test = [p10;p11];

limit = 1000000;

[num_of_patterns, num_of_elements] = size(p_test);

for p = 1:num_of_patterns
    converged = 0;
    epoch = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        
        unit = randi([1, num_of_elements]);
        
        update = sign(w(unit,:)*p_test(p,:)')';
        p_test(p,unit) = update;

        if mod(epoch,num_of_elements) == 0
            update = sign(w*p_test(p,:)')';            
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            end
        end
        
        % plot the current image
        if mod(epoch,100) == 0
%             num_of_figure = num_of_figure + 1;
%             figure(num_of_figure)
            imshow(reshape(p_test(p,:),[32 32]))
        end
    end
end

for p = 1:num_of_patterns
    
    identified = sum(ismember(p_train,p_test(p,:),'rows'));
    if identified == 1
        for q = 1:size(p_train,1)
            check = p_test(p,:) == p_train(q,:);
            if sum(sum(check))==numel(check)
                pattern_found = 1;
            else
                pattern_found = 0;
            end
            if pattern_found
                disp('Match for test: ')
                disp(p)
                disp('Matching train: ')
                disp(q)
                break
            end
        end
    else
        disp('No match for test:')
        disp(p)
    end
end