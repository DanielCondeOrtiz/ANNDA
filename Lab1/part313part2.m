%training data

ndata = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), ...
randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
classA(3,:)=ones(1,n);
classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);
classB(3,:)=-ones(1,n);

tmp = [classA,classB];
patterns=tmp(:,randperm(2*n));
w=randn(1,3);
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,2*n)];


%test data
ndata = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
tclassA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), ...
randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
tclassA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
tclassA(3,:)=ones(1,n);
tclassB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
tclassB(2,:) = randn(1,ndata) .* sigmaB + mB(2);
tclassB(3,:)=-ones(1,n);

tmp = [tclassA,tclassB];
tpatterns=tmp(:,randperm(2*n));
ttargets=(tpatterns(3,:));
tpatterns=[tpatterns(1:2,:);ones(1,2*n)];



%% delta bacth mode (3.1.3.2)

epochs = 100;
eta=0.001;
w=randn(1,3);

e2=[];

for i=0:epochs

    e=w*patterns-targets;
    
    deltaw=eta*e*patterns';
    w=w-deltaw;
    
    
    e2=[e2,sum(e.^2)/200];
    
end


%test
tydelta=sign(w*tpatterns);
correctdelta = sum(tydelta==ttargets);

figure(2)
plot(classA(1,:),classA(2,:),'r.')
hold on

plot(classB(1,:),classB(2,:),'b.')
w1= ([w(1),w(2)]./norm(w))*(-w(3))/norm(w);
w2=[w1(2),-w1(1)]+w1;

xlim([-3 3])
ylim([-3 3])

m = (w2(2)-w2(1))/(w1(2)-w1(1));
n1 = w2(2)*m - w1(2);
y1 = m*-3 + n1;
y2 = m*3 + n1;
title('Boundary for delta batch mode')
line([-3,3],[y1 y2])
hold off


%% remove 25% from each class



%% remove 50% of class A



%% remove 50% of class B



%% 20% from a subset of classA for which classA(1,:)<0 and 80% from a 
%subset of classA for which classA(1,:)>0



%% Ploting mean square error

figure(4)
hold on
plot(e2)
title('Mean square error at each batch (delta)')


