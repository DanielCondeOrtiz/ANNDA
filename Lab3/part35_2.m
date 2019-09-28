%% generate the data
clear all


%% testing number of weights

limit = 1000000;

p_test_true = [];
p_test = [];
p_train = [];
w = zeros(1024);

end_weights = 160;


remembered = [];

for nweights = 1:end_weights
    
    p_new = randi([0 1],[1 1024])*2 -1;
    
    disp("-------Weights: " + nweights)

    p_train = [p_train; p_new];
    p_test = [p_test_true;p_new];
    p_test_true = [p_test_true; p_new];

    w = w + p_new'*p_new;
    %w = w-diag(diag(w));
    
    [num_of_patterns, num_of_elements] = size(p_test);
    
    for p = 1:num_of_patterns
        converged = 0;

        epoch = 0;
        while converged == 0 && epoch<limit
            epoch=epoch + 1;
%batch
            update = sign(w*p_test(p,:)')';
            check = (update == p_test(p,:));
            if sum(check)==numel(check)
                converged = 1;
            else
                p_test(p,:) = update;
            end
            
        end
    end

    remembered(nweights) = sum(ismember(p_test,p_train,'rows'));

end

figure(1)
plot(remembered)
title('Remembered patterns with random training patterns')
xlabel('Patterns trained')
ylabel('Remembered patterns')

