t = 301:1500;
xvec=zeros(1,1506);

%this is to not run out of memory
for i=1:1506
    xvec(i)=xfunc(i-1,xvec);
end

input = [xvec(t-20); xvec(t-15); xvec(t-10); xvec(t-5); xvec(t)];
output = xvec(t+5);



train_in = input(:,1:1000);
train_out = output(1:1000);

test_in = input(:,1001:1200);
test_out = output(1001:1200);

%% training


% + info here
% https://se.mathworks.com/matlabcentral/fileexchange/69762-multi-layer-perceptron

% Multi-layer perceptron
MLP = ...
    MultiLayerPerceptron('LengthsOfLayers', [5 3 1],... %change numbers
                         'HiddenActFcn',    'linear',...
                         'OutputActFcn',    'linear');

% Training options
Options = ...
    struct('TrainingAlgorithm',         'GD',...
           'NumberOfEpochs',            200,...
           'MinimumMSE',                1e-2,...
           'SizeOfBatches',             30,...
           'SplitRatio',                0.7,...
           'Momentum',                  0.9);

% Training
MLP.train(train_in,train_out,Options);

% we have to put early stopping and whatever regularisation means
% compare different models

%% results

MLP.propagate(test_in);
results = MLP.Outputs;


figure(200)
plot(test_out) %???
hold on
plot(results)
title('Real and Test output')
legend('Real output','Test output')
hold off

error = sum((results-test_out).^2)/200;
