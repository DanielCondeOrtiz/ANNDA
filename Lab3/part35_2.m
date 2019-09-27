%% generate the data
clear all


%% testing number of weights


limit = 1000000;

noisebits = 2;

p_test_true = [];
p_test = [];
p_train = [];
w = zeros(1024);

%PARAMETERS!!!!
start_weights = 137;
end_weights = 145;
last_n = 15;

for p = 1:start_weights-1
    p_new = randi([0 1],[1 1024])*2 -1;
    p_train = [p_train; p_new];
    p_test = [p_test_true;p_new];
    p_test_true = [p_test_true; p_new];
    
    w = w + p_new'*p_new;
end

%w = w-diag(diag(w));

for nweights = start_weights:end_weights

    epochs = [];
    
    p_new = randi([0 1],[1 1024])*2 -1;
    
    disp("-------Weights: " + nweights)

    p_train = [p_train; p_new];
    p_test = [p_test_true;p_new];
    p_test_true = [p_test_true; p_new];

    w = w + p_new'*p_new;
    %w = w-diag(diag(w));
    
    [num_of_patterns, num_of_elements] = size(p_test);

    %noise
    pos = randi([1 1024],[nweights noisebits]);
    p_test(pos) = -p_test(pos);
    
    for p = num_of_patterns-last_n:num_of_patterns
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

        end
        epochs(p) = epoch;
    end

    total = 0;
    for p = num_of_patterns-last_n:num_of_patterns
        identified = sum(ismember(p_test(p,:),p_train,'rows'));
        if identified == 1
            total = total + 1;
            disp("Match for test: " + p + ", epochs: " + epochs(p))
        else
            disp("No match for test: " + p + ", epochs: " + epochs(p))
        end        
    end
        disp("**Total matches for last " + last_n + ": " + total)


end


