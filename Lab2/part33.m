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


%% training

for node = nodes
%     node
    
    % randomly initialize mu and initialize sigma with zeros
    mu = rand(node,1)*(2*pi);
    sigma = zeros(node,1);

    % train the mu according to WTA
    for epoch = 1:epochs
        
        % generate the distance from every mu to every data point
        dist = zeros(length(mu),length(x));
        for i = 1:length(mu)
            for j = 1:length(x)
                dist(i,j) = norm(mu(i)-x(j));
            end
        end
        
        % take a random datapoint
        rand_index = randi([1,length(x)]);

        % get the mu with the minimal distance to the random data point
        [~,index] = min(dist(:,rand_index));

        % update the closest mu by moving it towards the data point
        mu(index) = mu(index) + eta*(x(rand_index)-mu(index));
    
    end
    
    % set the number of datapoints per node
    x_count = length(x);
    x_per_mu = int16(x_count/node)+1;     
    
    % assign sigma in a way that a circle around mu(i) with the radius
    % sigma(i) contains exactley x_per_mu data points.
    for i = 1:length(mu)
        value = sort(dist(i,:));
        sigma(i) = value(x_per_mu);
    end
    
    
    % Plot the RBF centers and the corresponding sigma
    figure(node-9)
    
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
    
end