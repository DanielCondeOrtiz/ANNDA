clear all
close all

% data
x = [0:0.1:2*pi];
fun1= sin(2*x);
fun2 = sign(fun1);
fun2(1)=1;

xtest = [0.05:0.1:2*pi];
fun1test = sin(2*xtest);
fun2test = sign(fun1test);

phi_i = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma^2));

epochs = 1000;
widths = [0.0001,0.001,0.01,0.1,0.5]; %????
nodes = 28:40;
eta = 0.01;

figure_number = 0;
% training

% delta rule

errors1 = [];
errors2 = [];

for node = nodes
    
%     randomly initialize mu and initialize sigma with zeros
    mu1 = rand(node,1)*(2*pi);
    mu2 = rand(node,1)*(2*pi);
    sigma1 = zeros(node,1);
    sigma2 = zeros(node,1);

%     randomly initialize the weights
    w1 = randn(node,1);
    w2 = randn(node,1);

%     set the number of datapoints per node
    x_count = length(x);
    x_per_mu = int16(x_count/node);
    
    for e=1:epochs
%         update the RBF according to WTA      
        
%         take a random datapoint
        rand_index = randi([1,length(x)]);

%         get the mu with the minimal distance to the random data point
        dist1 = zeros(node,1);
        dist2 = zeros(node,1);
        for i = 1:node
            dist1(i) = norm(mu1(i)-x(rand_index));
            dist2(i) = norm(mu2(i)-x(rand_index));
        end
        
        [~,index1] = min(dist1);
        [~,index2] = sort(dist2);

%         update the closest mu by moving it towards the data point
        mu1(index1) = mu1(index1) + eta*(x(rand_index)-mu1(index1));
        for i = 1:5
            mu2(index2(i)) = mu2(index2(i)) + (eta/i)*(x(rand_index)-mu2(index2(i)));
        end                
%         generate the distance from every mu to every data point
        dist1 = zeros(node,length(x));
        dist2 = zeros(node,length(x));
        for i = 1:node
            for j = 1:length(x)
                dist1(i,j) = norm(mu1(i)-x(j));
                dist2(i,j) = norm(mu2(i)-x(j));
            end
         
%             assign sigma in a way that a circle around mu(i) with the 
%             radius sigma(i) contains exactley x_per_mu data points
            value = sort(dist1(i,:));
            sigma1(i) = value(x_per_mu);
            value = sort(dist2(i,:));
            sigma2(i) = value(x_per_mu);
%             sigma(i) = 0.001;
        end
        
%         update the weights
        
        x = x(randperm(length(x)));
        fun1= sin(2*x);
        for j = 1:length(x)
            phi1 = [];
            phi2 = [];
            for k=1:node
                phi1=[phi1; phi_i(x(j),mu1(k),sigma1(k))];
                phi2=[phi2; phi_i(x(j),mu2(k),sigma2(k))];
            end
            
            deltaw1 = eta*(fun1(j)-phi1'*w1)*phi1;
            w1=w1+deltaw1;
            deltaw2 = eta*(fun1(j)-phi2'*w2)*phi2;
            w2=w2+deltaw2;
        end
    end
    
    fout1 = zeros(1,length(xtest));
    fout2 = zeros(1,length(xtest));
    for j=1:length(xtest)
        tmp1 = 0;
        tmp2 = 0;
        for i=1:node
            tmp1 = tmp1 + w1(i)*phi_i(xtest(j),mu1(i),sigma1(i));
            tmp2 = tmp2 + w2(i)*phi_i(xtest(j),mu2(i),sigma2(i));
        end
        fout1(j) = tmp1;
        fout2(j) = tmp2;
    end
    
    figure_number = figure_number+1;
    figure(figure_number)
    plot(xtest,fout1,'r*')
    hold on
    plot(xtest,sin(2*xtest))
    title("Batch. Sine wave")
    ylim([-1.2 1.2])
    xlim([0, 2*pi])
    hold off

    figure_number = figure_number+1;
    figure(figure_number)
    plot(xtest,fout2,'r*')
    hold on
    plot(xtest,sin(2*xtest))
    title("Batch. Sine wave")
    ylim([-1.2 1.2])
    xlim([0, 2*pi])
    hold off

    
%     Plot the RBF centers and the corresponding sigma
%     figure_number = figure_number+1;
%     figure(figure_number)
%     
%     scatter(x,zeros(length(x),1))
%     hold on
%     scatter(mu,zeros(length(mu),1),'d')
%     for i = 1:node
%         hold on
%         th = 0:pi/50:2*pi;
%         xunit = sigma(i) * cos(th) + mu(i);
%         yunit = sigma(i) * sin(th) + 0;
%         h = plot(xunit, yunit);
%     end
%     
%     hold off
    
    errors1 = [errors1,mean(abs(fout1-fun1test))];
    errors2 = [errors2,mean(abs(fout2-fun1test))];
end

figure_number = figure_number+1;
figure(figure_number)
hold on

plot(nodes,errors1)
hold on
plot(nodes,errors2)


title('Delta Learning. Sine wave. Error by n. of nodes')
legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
legend('RBF WTA', 'RBF more nodes updated')
hold off