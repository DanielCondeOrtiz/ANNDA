import numpy as np
import matplotlib.pyplot as plt
from xfunc import xfunc


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


    train_in = input[::,range(0,999)].transpose()
    train_out = output[0,range(0,999)]

    test_in = input[::,range(1000,1199)].transpose()
    test_out = output[0,range(1000,1199)]


    results = np.zeros((4,8))
    # results2 = np.zeros((4,8))
    #training
    for alph in range(2,6):
        weights = []
        for nodes in range(2,10):
            est = MLPRegressor(hidden_layer_sizes=(nodes,),
                                               activation='logistic',
                                               solver='adam',
                                               learning_rate='adaptive',
                                               max_iter=10000,
                                               learning_rate_init=0.01,
                                               alpha=1/(10**alph),
                                               early_stopping=True)
            est.fit(train_in, train_out)
            weights = np.concatenate((weights,est.coefs_[0].flatten(),est.coefs_[0].flatten()))

    #
            pred = est.predict(test_in)
            results[alph-2,nodes-2] = sum((pred-test_out)**2)/200
    #
    #         pred2 = est.predict(train_in)
    #         results2[alph-2,nodes-2] = sum((pred2-train_out)**2)/1000
    #

        plt.figure(alph-1)
        plt.hist(weights,25)


        plt.xlabel('Weight value')
        plt.title('Histogram of weights for alpha=' + str(1/(10**alph)))
        plt.ylim([0,105])
        plt.xlim([-6,6])

    plt.figure(11)

    for alph in range(2,6):

        plt.plot(range(2,10),results[alph-2],label='alpha=' + str(1/(10**alph)))

        plt.legend(loc='upper right')
        plt.title('Error by number of nodes and regularisation')

    #
    #
    # plt.figure(2)
    # for alph in range(2,6):
    #     plt.plot(range(2,10),results2[alph-2],label='alpha=' + str(1/(10**alph)))
    #     plt.legend(loc='upper right')
    #     plt.title('Error by number of nodes and regularisation')
    #
    #
    #
    #
    est = MLPRegressor(hidden_layer_sizes=(6,),
                                       activation='logistic',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=10000,
                                       learning_rate_init=0.01,
                                       alpha=0.0001,
                                       early_stopping=True)
    est.fit(train_in, train_out)
    pred = est.predict(test_in)
    plt.figure(10)
    plt.plot(pred)
    plt.plot(test_out)


    plt.show()

if __name__ == '__main__':
    main()
