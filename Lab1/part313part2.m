n = 100;
lin_sep = false;
bias = true;

%training data
[patterns, targets] = data_generation(n, lin_sep, bias);

%test data
[tpatterns, ttargets] = data_generation(n, lin_sep, bias);

%% delta bacth mode

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
end

%test
tydelta=sign(w*tpatterns);
correctdelta1 = sum(tydelta==ttargets);

% plot the data
plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for delta batch mode', 1, bias)


%% remove 25% from each class

[class_points, class_identifiers, index] = data_seperation(patterns, targets, bias);

classA = class_points{2};
classA(3,:) = ones(1,n);
classB = class_points{1};
classB(3,:) = -ones(1,n);


removedA=randperm(100);
removedA=removedA(26:end);

removedB=randperm(100);
removedB=removedB(26:end);

tmp = [classA(:,removedA),classB(:,removedB)];
patterns=tmp(:,randperm(length(tmp)));
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,length(tmp))];

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
end

%test
tydelta=sign(w*tpatterns);
correctdelta25 = sum(tydelta==ttargets);

% plot data
plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for 25% removed of both', 2, bias)


%% remove 50% of class A

removedA=randperm(100);
removedA=removedA(51:end);

tmp = [classA(:,removedA),classB];
patterns=tmp(:,randperm(length(tmp)));
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,length(tmp))];

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
end

%test
tydelta=sign(w*tpatterns);
correctdelta50A = sum(tydelta==ttargets);

% plot the data
plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for 50% removed of A', 3, bias)


%% remove 50% of class B


removedB=randperm(100);
removedB=removedB(51:end);

tmp = [classA,classB(:,removedB)];
patterns=tmp(:,randperm(length(tmp)));
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,length(tmp))];

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
end

%test
tydelta=sign(w*tpatterns);
correctdelta50B = sum(tydelta==ttargets);

% plot the data
plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for 50% removed of B', 4, bias)

%% 20% from a subset of classA for which classA(1,:)<0 and 80% from a 
%subset of classA for which classA(1,:)>0

len = length(classA(:,classA(1,:)<0));
removedA1=randperm(len);
removedA1=removedA1(len-0.8*len+1:end);


len = length(classA(:,classA(1,:)>0));
removedA2=randperm(len);
removedA2=removedA2(len-0.2*len+1:end);

tmpA1 = classA(:,classA(1,:)<0);
tmpA2 = classA(:,classA(1,:)>0);
tmp = [tmpA1(:,removedA1),tmpA2(:,removedA2),classB];
patterns=tmp(:,randperm(length(tmp)));
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,length(tmp))];

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
end

%test
tydelta=sign(w*tpatterns);
correctdeltamixedA = sum(tydelta==ttargets);

% plot the data
plot_data_and_decision_boundary(patterns, targets, w, 'Boundary for 50% removed of B', 5, bias)