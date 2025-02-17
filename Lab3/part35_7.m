%% generate the data
clear all


%% testing number of weights (no noise)

limit = 100000;

w = zeros(100);


p_new = sign(0.5+randn(300,100));
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

    for p = 1:nweights
        converged = 0;
        epoch = 0;

        while converged == 0 && epoch<limit
%batch
            epoch = epoch + 1;
            update = sign(w*p_test(p,:)')';
            if sum(update == p_test(p,:))==1024
                converged = 1;
            else
                p_test(p,:) = update;
            end

        end
        %epochs(p) = epoch;
    end


    stables(nweights) = sum(ismember(p_test,p_train,'rows'));
    
    if nweights > 70 && sum(stables((nweights-70):nweights)) == 0
        break
    end
    
end

save('./stables_nonoise_biased.mat','stables')


%% testing number of weights (noise)


limit = 100000;

noisebits = 2;

p_test_true = [];
p_test = [];
p_train = [];
w = zeros(100);


p_new = sign(0.5+randn(300,100));
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
    pos = randi([1 100],[nweights noisebits]);
    p_test(pos) = -p_test(pos);
    
    for p = 1:nweights
        converged = 0;
        epoch = 0;

        while converged == 0 && epoch<limit
%batch
            epoch = epoch + 1;
            update = sign(w*p_test(p,:)')';
            if sum(update == p_test(p,:))==1024
                converged = 1;
            else
                p_test(p,:) = update;
            end
            
        end
        %epochs(p) = epoch;
    end

    stables_noise(nweights) = sum(ismember(p_test,p_train,'rows'));
    
    stables(nweights) = sum(ismember(p_test,p_train,'rows'));
    
    if nweights > 70 && sum(stables((nweights-70):nweights)) == 0
        break
    end
    
end


save('./stables_noise_biased2.mat','stables_noise')


%% plot
figure(1)
plot(stables)
hold on
plot(stables_noise)
title('Stable patterns by N. patterns learned (diagonal removed and biased)')
xlabel('Number of patterns learned')
ylabel('Stable patterns')
legend('No noise','Noise')



