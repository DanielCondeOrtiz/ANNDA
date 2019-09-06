t = 301:1500;
xvec=zeros(1,1506);

%this is to not run out of memory
for i=1:1506
    xvec(i)=xfunc(i-1,xvec);
end

input = [xvec(t-20); xvec(t-15); xvec(t-10); xvec(t-5); xvec(t)];
output = xvec(t+5);

train_in = input(:,1:1000);
train_out = output(1:1000);

% what is validation?
% val_in = input(:,701:1000);
% val_out = output(701:1000);

test_in = input(:,1001:1200);
test_out = output(1001:1200);

%% training

