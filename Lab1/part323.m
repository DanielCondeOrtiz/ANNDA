x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
mesh(x, y, z);

ndata=length(x)*length(y);
targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

%% training of the network







%% results

% gridsize=length(x)
% zz = reshape(out, gridsize, gridsize);
% mesh(x,y,zz);
% axis([-5 5 -5 5 -0.7 0.7]);
% drawnow;