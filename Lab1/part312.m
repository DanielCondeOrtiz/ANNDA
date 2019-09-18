
%% 3.1.2

% training data
n = 100;
lin_sep = true;
bias = true;
[patterns, targets] = data_generation(n, lin_sep, bias);

% test data
[tpatterns, ttargets] = data_generation(n, lin_sep, bias);

% initialize the weigths randomly
w=randn(1,3);


%% perceptron (3.1.2.1)

% eta=0.001;
% epoch=100;
% 
% e1=[];
% 
% for j=1:epoch
%     for i=1:n*2
% 
%         y=sign(w*patterns(:,i));
%         e=targets(i)-y;
%     
%         deltaw=eta*e*patterns(:,i)';
%         w=w+deltaw;
% 
%     end
%     
%     %test
%     typerc=sign(w*tpatterns);
%     correctperc = sum(typerc==ttargets);
%     e1=[e1,correctperc/200];
%     
% end
% 
% 
% %test
% typerc=sign(w*tpatterns);
% correctperc = sum(typerc==ttargets);
% 
% % plot the data
% plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for perceptron', 1, bias)


%% delta bacth mode (3.1.2.1)

figure(1)

epochs = 100;
eta_data=[0.01,0.005,0.001,0.0005,0.0001];

epochs2convergence_batch = zeros(length(eta_data),1);
for j = 1:length(eta_data)

w=randn(1,3);
eta = eta_data(j);    
    
e2=[];
correctdelta = 0;
converged = 0;
i = 0;
while converged == 0
% for i=1:epochs
    i = i+1;
    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    

    %test
    tydelta=sign(w*tpatterns);
    correctdelta(i) = sum(tydelta==ttargets);
%     e2=[e2,correctdelta(i)/200];
    
    if i>99
        converged = 1;
    end
    
end

plot(1:1:i,correctdelta);
hold on

% correctd{k} = correctdelta;
epochs2convergence_batch(j) = i;

end

for i = 1:length(eta_data)
    leg{i}=sprintf('\\eta =%.4f',eta_data(i));
end

legend(leg,'Location','southeast')
title('Different learning rates for batch delta rule');
xlabel('Number of epochs')
ylabel('Accuracy')
hold off

% plot the data
% plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for delta batch mode', 2, bias)

%% delta sequential (3.1.2.2)
% 
% w=randn(1,3);

figure(2)

e3=[];
epochs2convergence_seq = zeros(length(eta_data),1);
for k = 1:length(eta_data)

w=randn(1,3);
eta = eta_data(k);    
    
e2=[];
correctdelta = 0;
converged = 0;
i = 0;
while converged == 0
% for i=1:epochs
    i = i+1;

    for j=1:n*2
        e=w*patterns(:,j)-targets(:,j);

        deltaw=eta*e*patterns(:,j)';
        w=w-deltaw;
    end
    
    %test
    tydelta=sign(w*tpatterns);
    correctdelta(i) = sum(tydelta==ttargets);
%     e3=[e3,correctdelta(i)/200];
    
    if i>99
        converged = 1;
    end
    
end


plot(1:1:i,correctdelta);
hold on

% correctd{k} = correctdelta;
epochs2convergence_seq(k) = i;

end

for i = 1:length(eta_data)
    leg{i}=sprintf('\\eta =%.4f',eta_data(i));
end

legend(leg,'Location','southeast')
title('Different learning rates for sequential delta rule');
xlabel('Number of epochs')
ylabel('Accuracy')
hold off
%
% % plot the data
% plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for delta sequential mode', 3, bias)
% 
% %% Ploting mean square error (3.1.2.1 & 3.1.2.2)
% 
% 
% figure(4)
% plot(e1)
% hold on
% plot(e2)
% 
% %comment e3 for 3.1.2.1
% plot(e3)
% legend('Perceptron','Delta batch','Delta sequential','Location','southeast')
% title('Accuracy at each batch')
% 
% 
% 
% %% 3.1.2.3
% 
% n = 100;
% lin_sep = true;
% bias = false;
% 
% % training data
% [patterns, targets] = data_generation(n, lin_sep, bias);
% 
% % test data
% [tpatterns, ttargets] = data_generation(n, lin_sep, bias);
% 
% epochs = 100;
% eta=0.001;
% w=randn(1,2);
% 
% e2=[];
% 
% for i=0:epochs
% 
%     e=w*patterns-targets;
%     
%     deltaw=eta*e*patterns';
%     w=w-deltaw;
%     
% end
% 
% % plot the data
% plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for delta batch mode without bias', 5, bias)
% 
