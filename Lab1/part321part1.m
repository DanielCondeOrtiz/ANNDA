%% 3.2.2
close all
clear all

n = 100;
lin_sep = false;
bias = true;

%training data
[patterns, targets] = data_generation(n, lin_sep, bias);

%test data
[tpatterns, ttargets] = data_generation(n, lin_sep, bias);




ndata=200;
epochs = 1200;
eta=0.001;
Nhidden=10;
alpha = 0.9;

error=zeros(5,Nhidden-1);
misclas=zeros(5,Nhidden-1);

for nodes=2:Nhidden
    nodes
    w=randn(nodes,3); %N1, N0
    v=randn(1,nodes+1); %N2, N1
    dw=0;
    dv=0;
    
    tmperror=zeros(1,5);
    tmpmisclas=zeros(1,5);
    j=1;
    %training, works well
%     converged = 0;
%     overrun = 0;
%     epoch = 0;
%     while converged == 0 && overrun == 0
%         epoch = epoch+1;
%         if epoch > 50000
%             overrun = 1;
%             epochs(nodes-1) = epoch;
%             result{nodes-1} = 'overrun';
%         end
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
        
        if mod(i,300)==0
            %testing or whatever
            hin = w * patterns;
            hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
            oin = v * hout;
            out = 2 ./ (1+exp(-oin)) - 1;
        
            tmperror(j) = sum((out-targets).^2)/200;
            tmpmisclas(j) = sum(sign(out)==targets)/200;
            j=j+1;
        end
    
%     hin = w * patterns;
%     hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
%     oin = v * hout;
%     out = 2 ./ (1+exp(-oin)) - 1;
    
%     if sum(targets==sign(out)) >= 199
%         converged =1;
%         epochs(nodes-1) = epoch;
%         result{nodes-1} = 'converged';
%     end

    
    end
    
    
    
    error(:,nodes-1)=tmperror;
    misclas(:,nodes-1)=tmpmisclas;
    
end
figure(1)
hold on
for j=1:1:5
    plot(2:10,error(j,:))
end
title('Error for different number of hidden nodes and epochs')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','north')
hold off

figure(2)
hold on
for j=1:1:5
    plot(2:10,misclas(j,:))
end
title('Accuracy for different number of hidden nodes')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','south')
hold off


