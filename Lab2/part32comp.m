close all
clear all

%% data
xtrue = 0:0.1:2*pi;
x = xtrue + 0.1*randn(1,length(xtrue));

fun1= sin(2*xtrue)+ 0.1*randn(1,length(xtrue));
fun2 = sign(sin(2*xtrue))+ 0.1*randn(1,length(xtrue));

xtesttrue = 0.05:0.1:2*pi;
xtest = xtesttrue + 0.1*randn(1,length(xtesttrue));

fun1test = sin(2*xtesttrue) + 0.1*randn(1,length(xtesttrue));
fun2test = sign(sin(2*xtesttrue)) + 0.1*randn(1,length(xtesttrue));

phi_i = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma));

width = 0.01; %????
%how measure error????


%% training sin
%batch

nodes = 30;
mu = 0:((2*pi)/(nodes-1)):2*pi;

tic
phi=[];
for i=1:nodes
    phi=[phi; phi_i(x,mu(i),width)];
end

phi = phi';

A = phi' * phi;
B = phi' * fun1';

w = linsolve(A,B);

timebatchsine = toc;

fout1 = zeros(1,length(xtest));
for j=1:length(xtest)
    tmp = 0;
    for i=1:nodes
        tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
    end
    fout1(j) = tmp;
end

figure(1)
plot(xtest,fout1,'r*')
hold on
plot(xtesttrue,sin(2*xtesttrue))
title("Batch. Sine wave")
ylim([-1.2 1.2])
xlim([0, 2*pi])
hold off


%% training square
%batch

errors = [];
mu = 0:((2*pi)/(nodes-1)):2*pi;

tic
phi=[];
for i=1:nodes
    phi=[phi; phi_i(x,mu(i),width)];
end

phi = phi';

A = phi' * phi;
B = phi' * fun2';

w = linsolve(A,B);
timebatchsquare = toc;

fout2 = zeros(1,length(xtest));
for j=1:length(xtest)
    tmp = 0;
    for i=1:nodes
        tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
    end
    fout2(j) = tmp;
end

figure(2)
plot(xtest,fout2,'r*')
hold on
plot(xtesttrue,sign(sin(2*xtesttrue)))
title("Batch. Square wave")
ylim([-1.2 1.2])
xlim([0, 2*pi])
hold off

%% two layers perceptron sine

ndata=length(x);
epochs = 2000000;
eta=0.001;
Nhidden=nodes;
alpha = 0.9;

w=randn(nodes,1); %N1, N0
v=randn(1,nodes+1); %N2, N1
dw=0;
dv=0;

patterns = x;
targets = fun1;

tic
for i=1:epochs
    %forward pass
    hin = w * patterns;
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    %backward pass
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:nodes, :);
    
    %backpropagation
    dw = (dw .* alpha) - (delta_h * patterns') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;    
    
end
timepercsine = toc;

hin = w * xtest;
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out1 = 2 ./ (1+exp(-oin)) - 1;


figure(3)
plot(xtest,out1, 'r*')
hold on
plot(xtesttrue,sin(2*xtesttrue))
title("Perceptron. Sine wave")
ylim([-1.2 1.2])
xlim([0, 2*pi])
hold off

%% two layers perceptron square

w=randn(nodes,1); %N1, N0
v=randn(1,nodes+1); %N2, N1
dw=0;
dv=0;

patterns = x;
targets = fun2;


tic
for i=1:epochs
    %forward pass
    hin = w * patterns;
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    %backward pass
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:nodes, :);
    
    %backpropagation
    dw = (dw .* alpha) - (delta_h * patterns') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;    
    
end
timepercsquare = toc;

hin = w * xtest;
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out2 = 2 ./ (1+exp(-oin)) - 1;

figure(4)
plot(xtest,out2,'r*')
hold on
plot(xtesttrue,sign(sin(2*xtesttrue)))
title("Perceptron. Square wave")
ylim([-1.2 1.2])
xlim([0, 2*pi])
hold off


display("Time batch sine:" + timebatchsine)
display("Time batch square:" + timebatchsquare)
display("Time perceptron sine:" + timepercsine)
display("Time perceptron square:" + timepercsquare)

display("MAE batch sine:" + mean(abs(fun1test - fout1)))
display("MAE batch square:" + mean(abs(fun2test - fout2)))
display("MAE perceptron sine:" + mean(abs(fun1test - out1)))
display("MAE perceptron square:" + mean(abs(fun2test - out2)))


