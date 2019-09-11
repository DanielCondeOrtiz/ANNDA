%% 3.2.2
close all
clear all

n = 100;
lin_sep = false;
bias = true;

%training data
[patterns, targets] = data_generation(n, lin_sep, bias);

%test data
% [tpatterns, ttargets] = data_generation(n, lin_sep, bias);

[class_points, class_identifiers, index] = data_seperation(patterns, targets, bias);

classA = class_points{2};
classA(3,:) = ones(1,n);
classB = class_points{1};
classB(3,:) = -ones(1,n);

batch_mode = true;


for case_index = 1:4
    if case_index == 1
        removedA=randperm(n);
        removedB=randperm(n);

        tmp = [classA(:,removedA(26:end)),classB(:,removedB(26:end))];
        patterns=tmp(:,randperm(length(tmp)));
        targets=(patterns(3,:));
        patterns=[patterns(1:2,:);ones(1,length(tmp))];

        ttmp = [classA(:,removedA(1:25)),classB(:,removedB(1:25))];
        tpatterns=ttmp(:,randperm(length(ttmp)));
        ttargets=(tpatterns(3,:));
        tpatterns=[tpatterns(1:2,:);ones(1,length(ttmp))];

        
    elseif case_index == 2
    
        removedA=randperm(100);
        
        tmp = [classA(:,removedA(51:end)),classB];
        patterns=tmp(:,randperm(length(tmp)));
        targets=(patterns(3,:));
        patterns=[patterns(1:2,:);ones(1,length(tmp))];

        ttmp = [classA(:,removedA(1:50))];
        tpatterns=ttmp(:,randperm(length(ttmp)));
        ttargets=(tpatterns(3,:));
        tpatterns=[tpatterns(1:2,:);ones(1,length(ttmp))];
        
    elseif case_index == 3
    
        removedB=randperm(100);

        tmp = [classA,classB(:,removedB(51:end))];
        patterns=tmp(:,randperm(length(tmp)));
        targets=(patterns(3,:));
        patterns=[patterns(1:2,:);ones(1,length(tmp))];

        ttmp = [classB(:,removedB(1:50))];
        tpatterns=ttmp(:,randperm(length(ttmp)));
        ttargets=(tpatterns(3,:));
        tpatterns=[tpatterns(1:2,:);ones(1,length(ttmp))];
        
    elseif case_index == 4
    
        len = length(classA(:,classA(1,:)<0));
        removedA1=randperm(len);

        len = length(classA(:,classA(1,:)>0));
        removedA2=randperm(len);

        tmpA1 = classA(:,classA(1,:)<0);
        tmpA2 = classA(:,classA(1,:)>0);
        tmp = [tmpA1(:,removedA1(len-0.8*len+1:end)),tmpA2(:,removedA2(len-0.2*len+1:end)),classB];
        patterns=tmp(:,randperm(length(tmp)));
        targets=(patterns(3,:));
        patterns=[patterns(1:2,:);ones(1,length(tmp))];

        ttmpA1 = classA(:,classA(1,:)<0);
        ttmpA2 = classA(:,classA(1,:)>0);
        ttmp = [ttmpA1(:,removedA1(1:len-0.8*len)),ttmpA2(:,removedA2(1:len-0.2*len))];
        tpatterns=ttmp(:,randperm(length(ttmp)));
        ttargets=(tpatterns(3,:));
        tpatterns=[tpatterns(1:2,:);ones(1,length(ttmp))];        
    end
ndata=150;
tndata = 50;
epochs = 1200;
eta=0.001;
Nhidden=10;
alpha = 0.9;

error=zeros(5,Nhidden-1);
misclas=zeros(5,Nhidden-1);

terror=zeros(5,Nhidden-1);
tmisclas=zeros(5,Nhidden-1);

    
for nodes=2:Nhidden
    w=randn(nodes,3); %N1, N0
    v=randn(1,nodes+1); %N2, N1
    dw=0;
    dv=0;
    
    tmperror=zeros(1,5);
    tmpmisclas=zeros(1,5);
    
    ttmperror=zeros(1,5);
    ttmpmisclas=zeros(1,5);
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
ttmpmisclas_steady = zeros(1,epochs);

    for i=1:epochs
        if batch_mode       
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
        else
            for k = 1:length(patterns)
                %forward pass
                hin = w * patterns(:,k);
                hout = [2 ./ (1+exp(-hin)) - 1 ; 1];
                oin = v * hout;
                out = 2 ./ (1+exp(-oin)) - 1;

                %backward pass
                delta_o = (out - targets(:,k)) .* ((1 + out) .* (1 - out)) * 0.5;
                delta_h = (v'* delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
                delta_h = delta_h(1:nodes, :);

                %backpropagation
                dw = (dw .* alpha) - (delta_h * patterns(:,k)') .* (1-alpha);
                dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
                w = w + dw .* eta;
                v = v + dv .* eta;
            end        
        end
        
        if mod(i,300)==0
            % training data
            hin = w * patterns;
            hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
            oin = v * hout;
            out = 2 ./ (1+exp(-oin)) - 1;
        
            tmperror(j) = sum((out-targets).^2)/ndata;
            tmpmisclas(j) = sum(sign(out)==targets)/ndata;

            % validation data
            hin = w * tpatterns;
            hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,tndata)];
            oin = v * hout;
            out = 2 ./ (1+exp(-oin)) - 1;
        
            ttmperror(j) = sum((out-ttargets).^2)/tndata;
            ttmpmisclas(j) = sum(sign(out)==ttargets)/tndata;
            j=j+1;
        
        end
        
%         hin = w * tpatterns;
%         hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,tndata)];
%         oin = v * hout;
%         out = 2 ./ (1+exp(-oin)) - 1;
%         
%         ttmpmisclas_steady(i) = sum(sign(out)==ttargets)/tndata;
%         if ttmpmisclas_steady(i) > 0.21 && i>25
%             abc = 1;
%         end
        
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
    
    terror(:,nodes-1)=ttmperror;
    tmisclas(:,nodes-1)=ttmpmisclas;
    
end
figure(1+(case_index-1)*4)
hold on
for j=1:1:5
    plot(2:10,error(j,:))
end
title('Error for different number of hidden nodes and epochs - training set')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','north')
ylim([0,1])
hold off

figure(2+(case_index-1)*4)
hold on
for j=1:1:5
    plot(2:10,misclas(j,:))
end
title('Accuracy for different number of hidden nodes - training set')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','south')
ylim([0,1])
hold off

figure(3+(case_index-1)*4)
hold on
for j=1:1:5
    plot(2:10,terror(j,:))
end
title('Error for different number of hidden nodes and epochs - validation set')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','north')
if 3+(case_index-1)*4 < 15
    ylim([0,1])
end
hold off

figure(4+(case_index-1)*4)
hold on
for j=1:1:5
    plot(2:10,tmisclas(j,:))
end
title('Accuracy for different number of hidden nodes - validation set')
legend('300 iterations','600 iterations','900 iterations','1200 iterations','Location','south')
ylim([0,1])
hold off

end