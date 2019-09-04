
%% 3.1.2
n = 100;
mA = [ 1.8, 0.8]; sigmaA = 0.5;
mB = [-1.8, -0.8]; sigmaB = 0.5;
classA(1,:) = randn(1,n) .* sigmaA + mA(1);
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classA(3,:)=ones(1,n);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);
classB(3,:)=-ones(1,n);

tmp = [classA,classB];
patterns=tmp(:,randperm(2*n));
w=randn(1,3);
targets=(patterns(3,:));
patterns=[patterns(1:2,:);ones(1,2*n)];

%% perceptron

epochs = 100;
eta=0.001;

e1=[];

for i=0:epochs

    y=sign(w*patterns);
    e=targets-y;
    
    deltaw=eta*e*patterns';
    w=w+deltaw;
 
    e1=[e1,sum(e.^2)/200];
end

%test
tclassA(1,:) = randn(1,n) .* sigmaA + mA(1);
tclassA(2,:) = randn(1,n) .* sigmaA + mA(2);
tclassA(3,:)=ones(1,n);
tclassB(1,:) = randn(1,n) .* sigmaB + mB(1);
tclassB(2,:) = randn(1,n) .* sigmaB + mB(2);
tclassB(3,:)=-ones(1,n);

tmp = [tclassA,tclassB];
tpatterns=tmp(:,randperm(2*n));
ttargets=(tpatterns(3,:));
tpatterns=[tpatterns(1:2,:);ones(1,2*n)];

ty=sign(w*tpatterns);
correct = sum(ty==ttargets);

figure(1)
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
hold on
line([-3,3],[y1 y2])
hold off



%% delta

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
tclassA(1,:) = randn(1,n) .* sigmaA + mA(1);
tclassA(2,:) = randn(1,n) .* sigmaA + mA(2);
tclassA(3,:)=ones(1,n);
tclassB(1,:) = randn(1,n) .* sigmaB + mB(1);
tclassB(2,:) = randn(1,n) .* sigmaB + mB(2);
tclassB(3,:)=-ones(1,n);

tmp = [tclassA,tclassB];
tpatterns=tmp(:,randperm(2*n));
ttargets=(tpatterns(3,:));
tpatterns=[tpatterns(1:2,:);ones(1,2*n)];

ty=sign(w*tpatterns);
correct = sum(ty==ttargets);

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
hold on
line([-3,3],[y1 y2])
hold off

figure(3)
plot(e1)
hold on
plot(e2)
legend

