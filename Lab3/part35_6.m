%% generate the data
clear all


%% testing number of weights (no noise)


limit = 1000000;

noisebits = 0;

p_test_true = [];
p_test = [];
p_train = [];
w = zeros(100);


p_new = randi([0 1],[300 100])*2 -1;
p_train = [];
p_test = [];
p_test_true = p_new;

stables = zeros([1 300]);

for nweights = 1:300

    epochs = [];
    
    disp("-------Weights: " + nweights)

    p_train = p_test_true(1:nweights,:);
    p_test = p_test_true(1:nweights,:);
    
    w = w + p_train(nweights,:)'*p_train(nweights,:);
    w = w-diag(diag(w));
    
    [num_of_patterns, num_of_elements] = size(p_test);

    %noise
%     pos = randi([1 1024],[nweights noisebits]);
%     p_test(pos) = -p_test(pos);
    
    for p = 1:nweights
        converged = 0;
        epoch = 0;

        while converged == 0 && epoch<limit
            epoch = epoch + 1;
%batch
            epoch = epoch + 1;
            update = sign(w*p_test(p,:)')';
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            else
                p_test(p,:) = update;
            end
            
%random unit
%             unit = randi([1, num_of_elements]);
% 
%             update = sign(w(unit,:)*p_test(p,:)')';
%             p_test(p,unit) = update;
% 
%             %checking
%             update = sign(w*p_test(p,:)')';            
%             if sum(update == p_test(p,:))==1024
%                 converged = 1;
%             end
% 
        end
        %epochs(p) = epoch;
    end

    total = 0;
    for p = 1:nweights
        identified = sum(ismember(p_test(p,:),p_train,'rows'));
        if identified == 1
            total = total + 1;
        end        
    end
    disp("**Total matches for " + nweights + ": " + total)

    stables(nweights) = total;
    
end

save('./stables_nonoise.mat','stables')


%% testing number of weights (noise)


limit = 1000000;

noisebits = 2;

p_test_true = [];
p_test = [];
p_train = [];
w = zeros(100);


p_new = randi([0 1],[300 100])*2 -1;
p_train = [];
p_test = [];
p_test_true = p_new;

stables_noise = zeros([1 300]);

for nweights = 1:300

    epochs = [];
    
    disp("-------Weights: " + nweights)

    p_train = p_test_true(1:nweights,:);
    p_test = p_test_true(1:nweights,:);
    
    w = w + p_train(nweights,:)'*p_train(nweights,:);
    w = w-diag(diag(w));
    
    [num_of_patterns, num_of_elements] = size(p_test);

    %noise
%     pos = randi([1 1024],[nweights noisebits]);
%     p_test(pos) = -p_test(pos);
    
    for p = 1:nweights
        converged = 0;
        epoch = 0;

        while converged == 0 && epoch<limit
            epoch = epoch + 1;
%batch
            epoch = epoch + 1;
            update = sign(w*p_test(p,:)')';
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            else
                p_test(p,:) = update;
            end
            
%random unit
%             unit = randi([1, num_of_elements]);
% 
%             update = sign(w(unit,:)*p_test(p,:)')';
%             p_test(p,unit) = update;
% 
%             %checking
%             update = sign(w*p_test(p,:)')';            
%             if sum(update == p_test(p,:))==1024
%                 converged = 1;
%             end
% 
        end
        %epochs(p) = epoch;
    end

    total = 0;
    for p = 1:nweights
        identified = sum(ismember(p_test(p,:),p_train,'rows'));
        if identified == 1
            total = total + 1;
        end        
    end
    disp("**Total matches for " + nweights + ": " + total)

    stables_noise(nweights) = total;
    
end


save('./stables_noise2.mat','stables_noise')

figure(1)
plot(stables)
hold on
plot(stables_noise)
title('Stable patterns by N. patterns learned')
xlabel('Number of patterns learned')
ylabel('Stable patterns')
legend('No noise','Noise')



