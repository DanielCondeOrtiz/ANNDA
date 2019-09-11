import numpy as np
import matplotlib.pyplot as plt
from xfunc import xfunc
import time


from sklearn.neural_network import MLPRegressor


def main():

    t = range(300,1500)
    xvec=np.zeros((1,1506))

    #this is to not run out of memory
    for i in range(0,1505):
        xvec[0,i]=xfunc(i,xvec[0])

    input = np.zeros((5,1200))
    output=np.zeros((1,1200))
    for tt in t:
        input[0,tt-300] = xvec[0,tt-20]
        input[1,tt-300] = xvec[0,tt-15]
        input[2,tt-300] = xvec[0,tt-10]
        input[3,tt-300] = xvec[0,tt-5]
        input[4,tt-300] = xvec[0,tt]
        output[0,tt-300] = xvec[0,tt+5]


#part 1
    results = np.zeros((3,8))
    timresults = np.zeros((1,8))
    #training
    i=0
    for noise in [0.03,0.09,0.18]:
        train_in = input[::,range(0,999)].transpose()
        train_out = output[0,range(0,999)]
        train_in = train_in + np.random.normal(0,noise,(999,5))
        train_out = train_out + np.random.normal(0,noise,999)

        test_in = input[::,range(1000,1199)].transpose()
        test_out = output[0,range(1000,1199)]

        test_in = test_in + np.random.normal(0,noise,(199,5))
        test_out = test_out + np.random.normal(0,noise,199)

        for nodes in range(2,10):
            start = time.time()

            est = MLPRegressor(hidden_layer_sizes=(6,nodes,),
                                               activation='logistic',
                                               solver='adam',
                                               learning_rate='adaptive',
                                               max_iter=10000,
                                               learning_rate_init=0.01,
                                               alpha=0.0001,
                                               early_stopping=True)


            est.fit(train_in, train_out)

            pred = est.predict(test_in)
            results[i,nodes-2] = sum((pred-test_out)**2)/200

            timresults[0,nodes-2] = timresults[0,nodes-2] + time.time() - start

        i=i+1



    plt.figure(11)

    i=0
    for noise in [0.03,0.09,0.18]:

        plt.plot(range(2,10),results[i],label='$\sigma$=' + str(noise))

        plt.legend(loc='upper center')
        plt.title('Error by number of nodes and noise')
        i=i+1


# part 2
    results = np.zeros((3,4))
    i=0

    for noise in [0.03,0.09,0.18]:
        train_in = input[::,range(0,999)].transpose()
        train_out = output[0,range(0,999)]
        train_in = train_in + np.random.normal(0,noise,(999,5))
        train_out = train_out + np.random.normal(0,noise,999)

        test_in = input[::,range(1000,1199)].transpose()
        test_out = output[0,range(1000,1199)]

        test_in = test_in + np.random.normal(0,noise,(199,5))
        test_out = test_out + np.random.normal(0,noise,199)

        for alph in range(2,6):
            est = MLPRegressor(hidden_layer_sizes=(6,3,), #3???
                                               activation='logistic',
                                               solver='adam',
                                               learning_rate='adaptive',
                                               max_iter=10000,
                                               learning_rate_init=0.01,
                                               alpha=1/(10**alph),
                                               early_stopping=True)


            est.fit(train_in, train_out)

            pred = est.predict(test_in)
            results[i,alph-2] = sum((pred-test_out)**2)/200
        i=i+1



    plt.figure(12)
    plt.xscale('log')

    i=0
    for noise in [0.03,0.09,0.18]:

        plt.plot([0.01,0.001,0.0001,0.00001],results[i],label='$\sigma$=' + str(noise))

        plt.legend(loc='upper right')
        plt.title('Error by regularisation and noise')
        i=i+1


#part 3

    train_in = input[::,range(0,999)].transpose()
    train_out = output[0,range(0,999)]
    train_in = train_in + np.random.normal(0,0.09,(999,5))
    train_out = train_out + np.random.normal(0,0.09,999)

    test_in = input[::,range(1000,1199)].transpose()
    test_out = output[0,range(1000,1199)]

    test_in = test_in + np.random.normal(0,0.09,(199,5))
    test_out = test_out + np.random.normal(0,0.09,199)

    start = time.time()
    est1 = MLPRegressor(hidden_layer_sizes=(6,),
                                       activation='logistic',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=10000,
                                       learning_rate_init=0.01,
                                       alpha=0.0001,
                                       early_stopping=True)
    est1.fit(train_in, train_out)
    pred1 = est1.predict(test_in)

    time1 = time.time() - start

    #plt.plot(test_out)


    start = time.time()
    est2 = MLPRegressor(hidden_layer_sizes=(6,7,), #???
                                       activation='logistic',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=10000,
                                       learning_rate_init=0.01,
                                       alpha=0.0001,
                                       early_stopping=True)


    est2.fit(train_in, train_out)

    pred2 = est2.predict(test_in)
    time2 = time.time() - start

    plt.figure(13)
    plt.plot(pred2, label='3 layers')
    plt.plot(pred1,label='2 layers')
    plt.legend(loc='lower right')

    print('Times for each number of nodes:')
    print(timresults)
    print('Time for 2 layers: ' + str(time1))
    print('Time for 3 layers: ' + str(time2))

    plt.show()



if __name__ == '__main__':
    main()
