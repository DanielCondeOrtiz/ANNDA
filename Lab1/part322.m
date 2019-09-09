%% 3.2.2
%X
patterns=[1,-1,-1,-1,-1,-1,-1,-1;...
    -1,1,-1,-1,-1,-1,-1,-1;...
    -1,-1,1,-1,-1,-1,-1,-1;...
    -1,-1,-1,1,-1,-1,-1,-1;...
    -1,-1,-1,-1,1,-1,-1,-1;...
    -1,-1,-1,-1,-1,1,-1,-1;...
    -1,-1,-1,-1,-1,-1,1,-1;...
    -1,-1,-1,-1,-1,-1,-1,1];
%T
targets=patterns;

Nhidden=3;

w=randn(Nhidden,9);
v=randn(8,Nhidden+1);
dw=0;
dv=0;

ndata=8;
epochs = 1000000;
eta=0.001;

alpha = 0.9;

%training, works well
converged = 0;
i = 0;
while converged == 0
% for i=1:epochs
i = i+1;
    %forward pass
    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;

    %backward pass
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:Nhidden, :);

    %backpropagation
    dw = (dw .* alpha) - (delta_h * [patterns ; ones(1,ndata)]') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;

    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;
    
    
    if sum(sum(sign(out) == patterns)) == 64
        converged = 1;
    end
end


%testing or whatever
hin = w * [patterns ; ones(1,ndata)];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out = 2 ./ (1+exp(-oin)) - 1;
%should use acivator or something??


