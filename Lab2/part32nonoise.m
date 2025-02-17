close all
clear all

%% data
x = 0:0.1:2*pi;

xtest = 0.05:0.1:2*pi;

fun1test = sin(2*xtest);

phi_i = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma));

epochs = 20; %????
widths = [0.0001,0.001,0.01,0.1,0.5]; %????
nodes = 10:40;
eta = 0.01;%??????
%how measure error????


%% training sin
%delta rule
figure(1)
hold on
xlabel('Number of nodes')
ylabel('MAE')
ylim([0,4])

for width = widths
    errors = [];
    for node = nodes
        mu = 0:((2*pi)/(node-1)):2*pi;
        w = randn(node);
        for i=1:epochs
            x = x(randperm(length(x)));
            fun1= sin(2*x);
            for j = 1:length(x)
                phi = [];
                for k=1:node
                    phi=[phi; phi_i(x(j),mu(k),width)];
                end
                
                deltaw = eta*(fun1(j)-phi'*w)*phi;
                w=w+deltaw;
            end
        end
        
        fout1 = zeros(1,length(xtest));
        for j=1:length(xtest)
            tmp = 0;
            for i=1:node
                tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
            end
            fout1(j) = tmp;
        end
        
        errors = [errors,mean(abs(fout1-fun1test))];
    end
    
    plot(nodes,errors)
end

title('Delta Learning. Sine wave')
legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
hold off

%batch

x = 0:0.1:2*pi;

fun1= sin(2*x);

figure(2)
hold on
xlabel('Number of nodes')
ylabel('MAE')
ylim([0,4])

for width = widths
    errors = [];
    for node = nodes
        mu = 0:((2*pi)/(node-1)):2*pi;
        
        phi=[];
        for i=1:node
            phi=[phi; phi_i(x,mu(i),width)];
        end
        
        phi = phi';
        
        A = phi' * phi;
        B = phi' * fun1';
        
        w = linsolve(A,B);
        

        fout1 = zeros(1,length(xtest));
        for j=1:length(xtest)
            tmp = 0;
            for i=1:node
                tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
            end
            fout1(j) = tmp;
        end
        
        errors = [errors,mean(abs(fout1-fun1test))];
    end
    
    plot(nodes,errors)
end

title('Batch Learning. Sine wave')
legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
hold off

%% training square
%delta rule
errors = [];
figure(3)
hold on
xlabel('Number of nodes')
ylabel('MAE')
fun2test = sign(sin(2*xtest));
ylim([0,4])

for width = widths
    errors = [];
    for node = nodes
        mu = 0:((2*pi)/(node-1)):2*pi;
        w = randn(node);
        for i=1:epochs
            x = x(randperm(length(x)));
            fun2= sign(sin(2*x));
            for j = 1:length(x)
                phi = [];
                for k=1:node
                    phi=[phi; phi_i(x(j),mu(k),width)];
                end
                
                deltaw = eta*(fun2(j)-phi'*w)*phi;
                w=w+deltaw;
            end
        end
        
        fout2 = zeros(1,length(xtest));
        for j=1:length(xtest)
            tmp = 0;
            for i=1:node
                tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
            end
            fout2(j) = tmp;
        end
        
        errors = [errors,mean(abs(fout2-fun2test))];
    end
    
    plot(nodes,errors)
end

title('Delta Learning. Square wave')
legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
hold off

%batch

x = 0:0.1:2*pi;

fun1= sin(2*x);
fun2 = sign(fun1);
fun2(1)=1;


errors = [];
figure(4)
hold on
xlabel('Number of nodes')
ylabel('MAE')
ylim([0,4])

for width = widths
    errors = [];
    for node = nodes
        mu = 0:((2*pi)/(node-1)):2*pi;
        
        phi=[];
        for i=1:node
            phi=[phi; phi_i(x,mu(i),width)];
        end
        
        phi = phi';
        
        A = phi' * phi;
        B = phi' * fun2';
        
        w = linsolve(A,B);
        

        fout2 = zeros(1,length(xtest));
        for j=1:length(xtest)
            tmp = 0;
            for i=1:node
                tmp = tmp + w(i)*phi_i(xtest(j),mu(i),width);
            end
            fout2(j) = tmp;
        end
        
        errors = [errors,mean(abs(fout2-fun2test))];
    end
    
    plot(nodes,errors)
end

title('Batch Learning. Square wave')
legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
hold off




