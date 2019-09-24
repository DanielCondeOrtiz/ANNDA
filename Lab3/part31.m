%% input data

x1=[-1 -1 1 -1 1 -1 -1 1];
x2=[-1 -1 -1 -1 -1 1 -1 -1];
x3=[-1 1 1 -1 -1 1 -1 1];

x = [x1;x2;x3];

x1d=[ 1 -1 1 -1 1 -1 -1 1];
x2d=[ 1 1 -1 -1 -1 1 -1 -1];
x3d=[ 1 1 1 -1 1 1 -1 1];

xd = [x1d;x2d;x3d];

%% training of the weights

[num_of_patterns, num_of_elements] = size(x);

w = zeros(num_of_elements);

for p = 1:num_of_patterns
    w = w + x(p,:)'*x(p,:);
end

w = w-diag(diag(w));

%% testing
limit = 1000;
[num_of_patterns, num_of_elements] = size(xd);

for p = 1:num_of_patterns
    converged = 0;
    epoch = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        xd_update = sign(w*xd(p,:)')';
%         xd_update = sign(w*x(p,:)')';
        check = (xd_update == xd(p,:));
%         check = (xd_update == x(p,:));
        if sum(check)==numel(check)
            converged = 1;
        else
            xd(p,:) = xd_update;
%             x(p,:) = xd_update;
        end
    end
end

%% testing for attractors
test= dec2bin(2^8-1:-1:0)-'0';
test(test == 0) = -1;

attractors = [];

for i=1:256
    input = test(i,:);

    converged = 0;
    epoch = 0;
    
    while converged == 0 && epoch<limit
        epoch = epoch + 1;
        update = sign(w*input')';
        check = (update == input);
        if sum(check)==numel(check)
            converged = 1;
        else
            input = update;
        end
    end    
        
    output = input;
    
    if isempty(attractors) == 1 || sum(ismember(attractors,output,'rows')) == 0
        attractors = [attractors; output];
    end

end