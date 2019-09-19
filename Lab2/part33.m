clear all
close all

%% data
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
nodes = 10:40;
eta = 0.01;

figure_number = 0;
%% training

%delta rule

errors = [];
for node = nodes
    
    % randomly initialize mu and initialize sigma with zeros
    mu = rand(node,1)*(2*pi);
    sigma = zeros(node,1);

    % randomly initialize the weights
    w = randn(node);

    % set the number of datapoints per node
    x_count = length(x);
    x_per_mu = int16(x_count/node)+1;
    
    for e=1:epochs
        % update the RBF according to WTA      
        
        % take a random datapoint
        rand_index = randi([1,length(x)]);

        % get the mu with the minimal distance to the random data point
        dist = zeros(node,1);
        for i = 1:node
            dist(i) = norm(mu(i)-x(rand_index));
        end
        
        [~,index] = min(dist);

        % update the closest mu by moving it towards the data point
        mu(index) = mu(index) + eta*(x(rand_index)-mu(index));
                
        % generate the distance from every mu to every data point
        dist = zeros(node,length(x));
        for i = 1:node
            for j = 1:length(x)
                dist(i,j) = norm(mu(i)-x(j));
            end
         
            % assign sigma in a way that a circle around mu(i) with the 
            % radius sigma(i) contains exactley x_per_mu data points
            value = sort(dist(i,:));
            sigma(i) = value(x_per_mu);
        end
        
        % update the weights
        
        x = x(randperm(length(x)));
        fun1= sin(2*x);
        for j = 1:length(x)
            phi = [];
            for k=1:node
                phi=[phi; phi_i(x(j),mu(k),sigma(k))];
            end
            
            deltaw = eta*(fun1(j)-phi'*w)*phi;
            w=w+deltaw;
        end
    end
    
    fout1 = zeros(1,length(xtest));
    for j=1:length(xtest)
        tmp = 0;
        for i=1:node
            tmp = tmp + w(i)*phi_i(xtest(j),mu(i),sigma(i));
        end
        fout1(j) = tmp;
    end
    
    % Plot the RBF centers and the corresponding sigma
    figure_number = figure_number+1;
    figure(figure_number)
    
    scatter(x,zeros(length(x),1))
    hold on
    scatter(mu,zeros(length(mu),1),'d')
    for i = 1:node
        hold on
        th = 0:pi/50:2*pi;
        xunit = sigma(i) * cos(th) + mu(i);
        yunit = sigma(i) * sin(th) + 0;
        h = plot(xunit, yunit);
    end
    
    hold off
    
    errors = [errors,mean(abs(fout1-fun1test))];
end

figure_number = figure_number+1;
figure(figure_number)
hold on

plot(nodes,errors)

title('Delta Learning. Sine wave. Error by n. of nodes and width of spread')
% legend('Width=0.0001','Width=0.001','Width=0.01','Width=0.1','Width=0.5')
hold off
