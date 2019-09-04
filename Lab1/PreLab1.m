eta=0.001;
%X
patterns=[-1,1,-1,1;-1,-1,1,1];
%T
targets=[-1,1,1,-1];
ndata=4;

w=randn(1,3);
v=randn(1,2);

epochs = 20;
step=1;
Nhidden=2;
alpha = 0.9;

dw=0;
dv=0;

for i=0:epochs
    
    %deltaw = -n*(w*patterns-targets)*patterns';

    %w = w + deltaw;
end


hin = w * [patterns ; ones(1,ndata)];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out = 2 ./ (1+exp(-oin)) - 1;

delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
delta_h = delta_h(1:Nhidden, :);

dw = (dw .* alpha) - (delta_h * patterns') .* (1-alpha);
dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
W = w + dw .* eta;
V = v + dv .* eta;

