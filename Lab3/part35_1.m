%% generate the data
clear all

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

%w = w-diag(diag(w));

%% testing number of weights
num_of_figure = 0;

p_test = [p10;p11];

limit = 1000000;

noisebits = 2;

p_test = [p1;p2;p3];

for nweights = 3:6

    epochs = [];

    
    disp("Weights: " + nweights)
    
    if nweights == 4
        w = w + p4'*p4;
        p_train(4,:) = p4;
        p_test = [p1;p2;p3;p4];
    elseif nweights == 5
        w = w + p5'*p5;
        p_train(5,:) = p5;
        p_test = [p1;p2;p3;p4;p5];
    elseif nweights == 6
        w = w + p6'*p6;
        p_train(6,:) = p6;
        p_test = [p1;p2;p3;p4;p5;p6];
    end
    
    %w = w-diag(diag(w));
    
    
    [num_of_patterns, num_of_elements] = size(p_test);

    pos = randi([1 1024],[nweights noisebits]);
    p_test(pos) = -p_test(pos);
    
    for p = 1:num_of_patterns
        converged = 0;
        epoch = 0;

        while converged == 0 && epoch<limit
            epoch = epoch + 1;
%batch
%             epoch = epoch + 1;
%             update = sign(w*p_test(p,:)')';
%             check = (update == p_test(p,:));
%             if sum(check)==numel(check)
%                 converged = 1;
%             else
%                 p_test(p,:) = update;
%             end
            
%random unit
            unit = randi([1, num_of_elements]);

            update = sign(w(unit,:)*p_test(p,:)')';
            p_test(p,unit) = update;

            %checking
            update = sign(w*p_test(p,:)')';            
            if sum(update == p_test(p,:))==1024
                converged = 1;
            end


    %         % plot the current image
    %         if mod(epoch,100) == 0
    % %             num_of_figure = num_of_figure + 1;
    % %             figure(num_of_figure)
    %             imshow(reshape(p_test(p,:),[32 32]),'InitialMagnification',1000)
    %         end
        end
        epochs(p) = epoch;
    end

    for p = 1:num_of_patterns
        identified = sum(ismember(p_test(p,:),p_train,'rows'));
        if identified == 1
            disp("Match for test: " + p + ", epochs: " + epochs(p))
        else
            disp("No match for test: " + p + ", epochs: " + epochs(p))
        end
    end

end


