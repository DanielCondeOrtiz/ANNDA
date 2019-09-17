%% data
x = [0:0.1:2*pi];
fun1= sin(2*x);
fun2 = sign(fun1);
fun2(1)=1;

xtest = [0.05:0.1:2*pi];
fun1test = sin(2*xtest);
fun2test = sign(fun1test);

phi_i = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma));


%% training sin

nodes = 10;
exit = true;
error01 = true;
error001 = true;
error0001 = true;

while exit == true
    mu = 0:((2*pi)/(nodes-1)):2*pi;
    var = 0.1;
    
    phi=[];
    for i=1:nodes
        phi=[phi; phi_i(x,mu(i),var)];
    end
    
    phi = phi';
    
    A = phi' * phi;
    B = phi' * fun1';
    
    w = linsolve(A,B);
    
    fout1 = zeros(1,length(xtest));
    
    for j=1:length(xtest)
        tmp = 0;
        for i=1:nodes
            tmp = tmp + w(i)*phi_i(xtest(j),mu(i),var);
        end
        fout1(j) = tmp;
    end
    
    error = mean(abs(fout1-fun1test));
    
    if error < 0.1 && error01 == true
        subplot(2,3,1)
        plot(xtest,fout1)
        hold on
        plot(xtest,fun1test)
        title("Error < 0.1 with " + nodes + " nodes")
        hold off
        error01 = false;
    end
    
    if error < 0.01 && error001 == true
        subplot(2,3,2)
        plot(xtest,fout1)
        hold on
        plot(xtest,fun1test)
        title("Error < 0.01 with " + nodes + " nodes")
        hold off
        error001 = false;
    end
    
    if error < 0.001 && error0001 == true
        subplot(2,3,3)
        plot(xtest,fout1)
        hold on
        plot(xtest,fun1test)
        title("Error < 0.001 with " + nodes + " nodes")
        hold off
        error0001 = false;
        exit = false;
    end
    nodes = nodes + 1;
    
end

%close all

%% training square

nodes = 10;
exit = true;
error01 = true;
error001 = true;
error0001 = true;
minerror = inf;
minerrornodes = 0;

while exit == true
    mu = 0:((2*pi)/(nodes-1)):2*pi;
    var = 0.01;
    
    phi=[];
    for i=1:nodes
        phi=[phi; phi_i(x,mu(i),var)];
    end
    
    phi = phi';
    
    A = phi' * phi;
    B = phi' * fun2';
    
    w = linsolve(A,B);
    
    fout2 = zeros(1,length(xtest));
    
    for j=1:length(xtest)
        tmp = 0;
        for i=1:nodes
            tmp = tmp + w(i)*phi_i(xtest(j),mu(i),var);
        end
        fout2(j) = tmp;
    end
    
    error = mean(abs(fout2-fun2test));
    
    if error < 0.1 && error01 == true
        subplot(2,3,4)
        plot(xtest,fout2)
        hold on
        plot(xtest,fun2test)
        title("Error < 0.1 with " + nodes + " nodes")
        hold off
        error01 = false;
    end
    
    if error < 0.01 && error001 == true
        figure(5)
        plot(xtest,fout2)
        hold on
        plot(xtest,fun2test)
        title("Error < 0.01 with " + nodes + " nodes")
        hold off
        error001 = false;
    end
    
    if error < 0.001 && error0001 == true
        figure(6)
        plot(xtest,fout2)
        hold on
        plot(xtest,fun2test)
        title("Error < 0.001 with " + nodes + " nodes")
        hold off
        error0001 = false;
        exit = false;
    end
    nodes = nodes + 1;
    
    if error < minerror
        minerror = error;
        minerrornodes = nodes;
    end
    
    if nodes > 100
        exit = false;
    end
    
end


%% training square with output modified

nodes = 10;
exit = true;
error01 = true;
error001 = true;
error0001 = true;

while exit == true
    mu = 0:((2*pi)/(nodes-1)):2*pi;
    var = 0.01;
    
    phi=[];
    for i=1:nodes
        phi=[phi; phi_i(x,mu(i),var)];
    end
    
    phi = phi';
    
    A = phi' * phi;
    B = phi' * fun2';
    
    w = linsolve(A,B);
    
    fout2 = zeros(1,length(xtest));
    
    for j=1:length(xtest)
        tmp = 0;
        for i=1:nodes
            tmp = tmp + w(i)*phi_i(xtest(j),mu(i),var);
        end
        fout2(j) = sign(tmp);
    end
    
    error = mean(abs(fout2-fun2test));
    
    if error < 0.1 && error01 == true
        subplot(2,3,5)
        plot(xtest,fout2)
        hold on
        plot(xtest,fun2test)
        title("Error < 0.1 with " + nodes + " nodes and sign()")
        ylim([-1.2,1.2])
        hold off
        error01 = false;
    end
    
    if error < 0.01 && error001 == true
%         figure(8)
%         plot(xtest,fout2)
%         hold on
%         plot(xtest,fun2test)
%         title("Error < 0.01 with " + nodes + " nodes and sign()")
%         ylim([-1.2,1.2])
%         hold off
%         error001 = false;
    end
    
    if error < 0.001 && error0001 == true
        subplot(2,3,6)
        plot(xtest,fout2)
        hold on
        plot(xtest,fun2test)
        title("Error = 0 with " + nodes + " nodes and sign()")
        ylim([-1.2,1.2])
        hold off
        error0001 = false;
        exit = false;
    end
    nodes = nodes + 1;
    
end


