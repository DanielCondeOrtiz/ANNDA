close all
clear all

%% data
x = 0:0.1:2*pi;
x = x + 0.1*randn(1,length(x));

fun1= sin(2*x);
fun2 = sign(fun1);

xtest = 0.05:0.1:2*pi;
xtest = xtest + 0.1*randn(1,length(xtest));

fun1test = sin(2*xtest);
fun2test = sign(sin(2*xtest));

phi_i = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma));

width = 0.1; %????
%how measure error????


%% training sin
%delta rule

%batch

errors = [];
nodes = 40;
mu = 0:((2*pi)/(nodes-1)):2*pi;

phi=[];
for i=1:nodes
    phi=[phi; phi_i(x,mu(i),width)];
end

phi = phi';

A = phi' * phi;
B = phi' * fun1';

w = linsolve(A,B);


fout1 = zeros(1,length(xtest));
for j=1:length(xtest)
    tmp = 0;
    for i=1:nodes
        tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
    end
    fout1(j) = tmp;
end

figure(1)
plot(xtest,fout1)
hold on
plot(xtest,fun1test)
title("Batch. Sine wave")
hold off


%% training square
%batch

errors = [];
mu = 0:((2*pi)/(nodes-1)):2*pi;

phi=[];
for i=1:nodes
    phi=[phi; phi_i(x,mu(i),width)];
end

phi = phi';

A = phi' * phi;
B = phi' * fun2';

w = linsolve(A,B);


fout2 = zeros(1,length(xtest));
for j=1:length(xtest)
    tmp = 0;
    for i=1:nodes
        tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
    end
    fout2(j) = tmp;
end



figure(2)
plot(xtest,fout2)
hold on
plot(xtest,fun2test)
title("Batch. Square wave")
hold off



%% two layers perceptron sine

ndata=length(x);
epochs = 200;
eta=0.001;
Nhidden=nodes;
alpha = 0.9;


w=randn(nodes,1); %N1, N0
v=randn(1,nodes+1); %N2, N1
dw=0;
dv=0;

patterns = x;
targets = fun1;

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

hin = w * xtest;
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out1 = 2 ./ (1+exp(-oin)) - 1;


figure(3)
plot(xtest,out1)
hold on
plot(xtest,fun1test)
title("Perceptron. Sine wave")
hold off

%% two layers perceptron square

ndata=length(x);
epochs = 200;
eta=0.001;
Nhidden=nodes;
alpha = 0.9;


w=randn(nodes,1); %N1, N0
v=randn(1,nodes+1); %N2, N1
dw=0;
dv=0;

patterns = x;
targets = fun2;

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

hin = w * xtest;
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out2 = 2 ./ (1+exp(-oin)) - 1;

figure(4)
plot(xtest,out2)
hold on
plot(xtest,fun2test)
title("Perceptron. Sine wave")
hold off

