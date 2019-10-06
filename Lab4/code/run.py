from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
from dbn2 import DeepBeliefNet2
import threading


def train_net_units(units,n_epochs,results):
    print("Starting network with " + str((units+2)*100) + " units")
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=(units+2)*100,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )

    res = rbm.cd1(visible_trainset=train_imgs, max_epochs=n_epochs, n_iterations=3000, bool_print=True)

    results[units,:] = res


def call_units():
    threads = []
    units = [200,300,400,500]
    n_epochs = 2

    n_units = len(units)

    results = np.zeros((n_units,n_epochs))
    for i in range(n_units):
        t = threading.Thread(target=train_net_units, args=(i,n_epochs,results))
        threads.append(t)
        t.start()

    for th in threads:
        th.join()

    fig, ax = plt.subplots()
    ax.plot(units, results)

    ax.set(xlabel='Hidden units', ylabel='Reconstruction loss',
           title='Average reconstruction loss')

    fig.savefig("hidden_loss.png")
    plt.show()

    fig, ax = plt.subplots()

    for i in range(n_units):
        ax.plot(np.arange(1,n_epochs+1),np.transpose(results[i,0:n_epochs]))

    ax.set(xlabel='Epochs', ylabel='Reconstruction loss', title='Reconstruction loss over epochs',xticks=np.arange(1,n_epochs+1))
    ax.legend = (('N_h=200', 'N_h=300', 'N_h=400', 'N_h=500'))

    fig.savefig("epochs_loss.png")
    plt.show()

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print ("\nStarting a Restricted Boltzmann Machine..")


    #first point

#    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
#                                     ndim_hidden=500,
#                                     is_bottom=True,
#                                     image_size=image_size,
#                                     is_top=False,
#                                     n_labels=10,
#                                     batch_size=20
#    )
#
#    recon_loss_ep = rbm.cd1(visible_trainset=train_imgs, n_iterations=3000, max_epochs=20, bool_print=True)

    #second point
    #call_units()

    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    #6000 iterations because batch size = 10
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=6000)

    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")
    #
    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=6000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")


    ''' last-part '''

    # dbn2 = DeepBeliefNet2(sizes={"vis":image_size[0]*image_size[1], "pen":500, "top":2000, "lbl":10},
    #                     image_size=image_size,
    #                     n_labels=10,
    #                     batch_size=10
    # )
    #
    # dbn2.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=3000)
    #
    #
    # #
    # dbn2.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=3000)
    #
    # dbn2.recognize(train_imgs, train_lbls)
    #
    # dbn2.recognize(test_imgs, test_lbls)
