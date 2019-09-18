clear all
close all

x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
% figure(42)
% mesh(x, y, z);

ndata=length(x)*length(y);
targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

%% 3.2.3.1
epochs = 1000;
eta=0.001;
alpha = 0.9;
gridsize=length(x);

%shuffle
 order = randperm(ndata);
shupatterns = patterns(:,order);
shutargets =  targets(order);

error = zeros(25,1);

%training
for nodes=1:25
%     nodes = 23;
    Nhidden=nodes;
    w=randn(Nhidden,3);
    v=randn(1,Nhidden+1);
    dw=0;
    dv=0;
% load('data.mat');
error = 1;
converged = 0;
    %training, works well
%     i = 0;
%     while converged == 0
    for i=1:epochs
%         i = i+1;
%         if i<10
%             eta = 0.1;
%         elseif i<25
%             eta = 0.05;
%         elseif i < 100
%             eta = 0.01;
%         else
%             eta = 0.001;
%         end
        
        %forward pass
        hin = w * [shupatterns ; ones(1,ndata)];
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;

        %backward pass
        delta_o = (out - shutargets) .* ((1 + out) .* (1 - out)) * 0.5;
        delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
        delta_h = delta_h(1:Nhidden, :);

        %backpropagation
        dw = (dw .* alpha) - (delta_h * [shupatterns ; ones(1,ndata)]') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        w = w + dw .* eta;
        v = v + dv .* eta;

        hin = w * [patterns ; ones(1,ndata)];
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;
        error = sum((out-targets).^2)/ndata;
    
%         if error < 0.001
%             converged = 1;
%         end
    end

    %results
    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    error(nodes) = sum((out-targets).^2)/ndata;
    
    zz = reshape(out, gridsize, gridsize);
    figure(nodes)
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
%     sorry for all the plots
    title(nodes +" nodes in the hidden layer. Error=" + error(nodes))

end

[min_error_layer, min_error_index_layer] = min(error);

%% 3.2.3.2, now we select best model, CHANGE NHIDDEN

epochs = 1000;
eta=0.001;
alpha = 0.9;
gridsize=length(x);

%shuffle
order = randperm(ndata);
shupatterns = patterns(:,order);
shutargets =  targets(order);

error = zeros(7,1);
index = 0;
%training
for part=0.2:0.1:0.8
    index = index+1;

    Nhidden=18;%min_error_index_layer;
    w=randn(Nhidden,3);
    v=randn(1,Nhidden+1);
    dw=0;
    dv=0;
    ndata = round(length(shupatterns)*part);
    
    %training, works well
    for i=1:epochs
        %forward pass

        hin = w * [shupatterns(:,1:ndata) ; ones(1,ndata)];
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;

        %backward pass
        delta_o = (out - shutargets(:,1:ndata)) .* ((1 + out) .* (1 - out)) * 0.5;
        delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
        delta_h = delta_h(1:Nhidden, :);

        %backpropagation
        dw = (dw .* alpha) - (delta_h * [shupatterns(:,1:ndata) ; ones(1,ndata)]') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        w = w + dw .* eta;
        v = v + dv .* eta;


    end

    %results
    ndata=length(x)*length(y);
    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    %I don't know if the error is OK, probably??
    error(index) = sum((out-targets).^2)/ndata;
    
    zz = reshape(out, gridsize, gridsize);
    figure(round(part*100))
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    %sorry for all the plots
    title(part*100 +"% data in training. Error=" + error(index))

end

[~,min_error_train_percent_index] = min(error);
training_percentage = 0.2:0.1:08;

min_error_training_percentage = training_percentage(min_error_train_percent_index);

%% 3.2.3.3 speeding up?


