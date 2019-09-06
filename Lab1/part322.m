
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

w=randn(1,9);
v=randn(1,3);

ndata=8;
epochs = 1000;
eta=0.001;
Nhidden=3;
alpha = 0.9;

dw=0;
dv=0;


%don't know if it's 1 column at a time (this loop) or all the matrix at
%once (below)
for pattern=patterns
    target=pattern';

%     hin = w * [pattern ; 1];
%     hout = [2 ./ (1+exp(-hin)) - 1 ; 1];
%     oin = v * hout;
%     out = 2 ./ (1+exp(-oin)) - 1;
% 
%     delta_o = (out - target) .* ((1 + out) .* (1 - out)) * 0.5;
%     delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
%     delta_h = delta_h(1:Nhidden, :);
    
    
end

%     hin = w * [patterns ; ones(1,ndata)];
%     hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
%     oin = v * hout;
%     out = 2 ./ (1+exp(-oin)) - 1;
% 
%     delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
%     delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
%     delta_h = delta_h(1:Nhidden, :);
% 
% 
% 
% dw = (dw .* alpha) - (delta_h * patterns') .* (1-alpha);
% dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
% W = w + dw .* eta;
% V = v + dv .* eta;

