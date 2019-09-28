%% generate the data
clear all


%% testing

limit = 1000000;

noisebits = 0;

activity = 0.01;

bias = [0.01,0.1,0.5,1];

figure(1)
hold on
title('Stored and remembered values for different biases')
xlabel('Stored')
ylabel('Remembered')

for bia = bias
    disp("Bias: " + bia)
    
    p_train = [];
    p_test = [];
    p_test_true = [];
    
    w = zeros(100);
    
    nweights = 0;
    stop = 0;
    steps = 0;
    
    remembered = [];
    
    while stop < 15 && steps < 1000 && nweights < 1000
        steps = steps +1;
        
        %generating next loop
        p_new = zeros([1 100]);
        p_new(randperm(100,activity*100))=1;
        
        if steps == 1 || ~ismember(p_new,p_test_true,'rows')
            nweights = nweights + 1;
            p_test_true = [p_test_true; p_new];
        else
            continue
        end
        
        p_train = p_test_true;
        p_test = p_test_true;
        
        w = w + (p_new' - activity)*(p_new-activity);
        %w = w-diag(diag(w));
        
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
                update = 0.5 + 0.5*sign(w*p_test(p,:)' - bia)';
                check = (update == p_test(p,:));
                if sum(check)==numel(check)
                    converged = 1;
                else
                    p_test(p,:) = update;
                end
                
                %random unit
                %             unit = randi([1, num_of_elements]);
                %
                %             update = 0.5 + 0.5*(sign(w*p_test(p,:)') - 1)';
                %             p_test(p,unit) = update;
                %
                %             %checking
                %             update = sign(w*p_test(p,:)')';
                %             if sum(update == p_test(p,:))==1024
                %                 converged = 1;
                %             end
                %
            end
            
        end

        remembered(nweights) =sum(ismember(p_test,p_train,'rows'));
        
        if remembered(nweights) ==0
            stop = stop + 1;
        else
            stop = 0;
        end
        
    end
    
    plot(remembered)
    
end

legend('Bias = 0.01','Bias = 0.1','Bias = 0.5','Bias = 1','Location','northwest')

