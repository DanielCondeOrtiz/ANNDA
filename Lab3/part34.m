close all
clear all

%% input data

x1=[-1 -1 1 -1 1 -1 -1 1];
x2=[-1 -1 -1 -1 -1 1 -1 -1];
x3=[-1 1 1 -1 -1 1 -1 1];

x = [x1;x2;x3];

figure_number = 0;

%% training of the weights

[num_of_patterns, num_of_elements] = size(x);

w = zeros(num_of_elements);

for p = 1:num_of_patterns
    w = w + x(p,:)'*x(p,:);
end

%w = w-diag(diag(w));

%% testing

% distorted if false, noise added if true
noise_or_dist = false;

correct = zeros(3,num_of_elements);

for noisy_elements = 1:num_of_elements+1
    
    x1d=x(1,:);
    x2d=x(2,:);
    x3d=x(3,:);
    
    index = randperm(num_of_elements, noisy_elements-1);
    
    if noise_or_dist
        x1d(index) = x1d(index) + randn(1,noisy_elements-1);
        x2d(index) = x2d(index) + randn(1,noisy_elements-1);
        x3d(index) = x3d(index) + randn(1,noisy_elements-1);
    else
        x1d(index) = -1*x1d(index);
        x2d(index) = -1*x2d(index);
        x3d(index) = -1*x3d(index);
    end
    xd = [x1d;x2d;x3d];
    
    
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
    
    correct(:,noisy_elements) = (sum(xd==x,2)/num_of_elements)';
     
end

figure_number = figure_number +1;
figure(figure_number)
for i=1:3
    plot(1:1:9,correct(i,:));
    hold on
end
legend('P1','P2','P3', 'Location', 'southwest')
ylabel('Correct in %')
ylim([0,1]);
x_label = 0:0.125:1;
xticklabels(num2str(x_label(:)))
if noise_or_dist
    xlabel('Noise in % on elements');
    title('Accuracy of restored pattterns depending on the noise')
else
    xlabel('Distortion in % on elements');
    title('Accuracy of restored pattterns depending on the distortion')
end