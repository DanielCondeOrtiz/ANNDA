function [patterns, targets] = data_generation(n, lin_sep, bias)
%DATA_GENERATION generates datapoints in two classes which are either
%linearly seperable or not. The data points are then permuted and given
%back alongside the correct labels. The data points can either have or do
%not have a bias.
%
%INPUT:
%   n: int, count of elements per class
%   lin_sep: bool, decides if data is linearly separable
%   bias: bool, decides if the data gets a bias (additional row of ones as
%       z-component) or not
%OUTPUT:
%   patterns: 3*2n matrix of randomly permuted 2-dim data points from two
%       classes. Third row is set to ones
%   targets: 1*2n vector containing the correct class label for each data
%       point

if lin_sep
    sigmaA = 0.5;
    sigmaB = 0.5;
    mA = [ 1.1, 0.5]; 
    mB = [-1.1, -0.5];
    classA(1,:) = randn(1,n) .* sigmaA + mA(1);
else
    sigmaA = 0.2;
    sigmaB = 0.3;
    mA = [ 1.0, 0.3];
    mB = [ 0.0, -0.1]; 
    classA(1,:) = [ randn(1,round(0.5*n)) .* sigmaA - mA(1), randn(1,round(0.5*n)) .* sigmaA + mA(1)]; 
end

classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classA(3,:) = ones(1,n);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);
classB(3,:) = -ones(1,n);

tmp = [classA,classB];
patterns=tmp(:,randperm(2*n));
targets=(patterns(3,:));

if bias
    patterns=[patterns(1:2,:);ones(1,2*n)];
else
    patterns=patterns(1:2,:);
end
    
end

