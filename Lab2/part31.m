%% data
x = [0:0.1:2*pi];
fun1= sin(x);
fun2 = sign(fun1);
fun2(1)=1;

xtest = [0.05:0.1:2*pi];
fun1test = sin(xtest);
fun2test = sign(fun1test);

transform = @(x,mu,sigma) exp((-(x-mu).^2)/(2*sigma));

w = randn(length(x),1);

