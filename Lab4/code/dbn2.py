from util import *
from rbm import RestrictedBoltzmannMachine
import random

class DeepBeliefNet2():

    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen]  ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["pen"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),

            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        print("Recognizing")

        n_samples = true_img.shape[0]

        vis = true_img # visible layer gets the image data

        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels
        input = self.rbm_stack["vis--pen"].get_h_given_v_dir(vis)[0]
        l_k = np.append(input,lbl,axis=1)


        error = []
        x = []

        for i in range(self.n_gibbs_recog):
            print("Loop: " + str(i))
            #top layer
            t_k = self.rbm_stack["pen+lbl--top"].get_h_given_v(l_k)[0]
            #lower layer
            l_k,l_k_b = self.rbm_stack["pen+lbl--top"].get_v_given_h(t_k)

            error.append(np.linalg.norm(l_k_b[:,-10:] - true_lbl))

            #for plotting accuracy instead of error
            #error.append((100.*np.mean(np.argmax(l_k_b[:,-10:],axis=1)==np.argmax(true_lbl,axis=1))))

            x.append(i)

        fig, ax = plt.subplots()
        ax.plot(x, error)

        ax.set(xlabel='Gibbs steps', ylabel='Recognition loss',
               title='Average recognition loss')

        fig.savefig("recon_loss_" + str(true_lbl.shape[0]) + ".png")
        plt.show()

        predicted_lbl = l_k_b[:,-10:]

        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))

        return


    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm2",name="vis--pen")
            self.rbm_stack["vis--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm2",name="pen+lbl--top")

        except IOError :

            print ("training vis--pen")
            """
            CD-1 training for vis--pen
            """
            self.rbm_stack["vis--pen"].cd1(visible_trainset=vis_trainset, n_iterations=3000,max_epochs=15,bool_print=False)

            self.savetofile_rbm(loc="trained_rbm2",name="vis--pen")

            self.rbm_stack["vis--pen"].untwine_weights()

            print ("training pen+lbl--top")
            """
            CD-1 training for pen+lbl--top
            """

            input2 = np.append(self.rbm_stack["vis--pen"].get_h_given_v_dir(vis_trainset)[0],lbl_trainset,axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(visible_trainset=input2, n_iterations=3000,max_epochs=15,bool_print=False)

            self.savetofile_rbm(loc="trained_rbm2",name="pen+lbl--top")

        print("Checking reconstruction")
        rec_rbm = np.linalg.norm(vis_trainset - self.rbm_stack["vis--pen"].get_v_given_h_dir(self.rbm_stack["vis--pen"].get_h_given_v_dir(vis_trainset)[0])[0])

        vtop = self.rbm_stack["vis--pen"].get_h_given_v_dir(vis_trainset)[0]
        ptot = self.rbm_stack["pen+lbl--top"].get_h_given_v(np.append(vtop,lbl_trainset,axis=1))[0]

        ttop = self.rbm_stack["pen+lbl--top"].get_v_given_h(ptot)[0]
        ptov = self.rbm_stack["vis--pen"].get_v_given_h_dir(ttop[:,:-10])[0]

        rec_dbn = np.linalg.norm(vis_trainset - ptov)

        print("Reconstruction RBM: " + str(rec_rbm))
        print("Reconstruction DBN: " + str(rec_dbn))

        num = random.randint(1,vis_trainset.shape[0]+1)

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(htov[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_dbn2.png")

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(vis_trainset[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_orig2.png")

        htov = self.rbm_stack["vis--hid"].get_v_given_h_dir(vtoh)[0]

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(htov[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_rbm2.png")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("\ntraining wake-sleep..")

        try :

            self.loadfromfile_dbn(loc="trained_dbn2",name="vis--pen")
            self.loadfromfile_rbm(loc="trained_dbn2",name="pen+lbl--top")

        except IOError :

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                """
                wake-phase : drive the network bottom-to-top using visible and label data
                """
                mini_batch = vis_trainset[self.batch_size*(it):self.batch_size*(it+1)]

                vtop = self.rbm_stack["vis--pen"].get_h_given_v_dir(mini_batch)[0]
                ptot = self.rbm_stack["pen+lbl--top"].get_h_given_v(np.append(vtop,lbl_trainset,axis=1))[0]

                v_k = vtop
                h_k = ptot

                """
                alternating Gibbs sampling in the top RBM : also store neccessary information for learning this RBM
                """
                for i in range(self.n_gibbs_wakesleep-1):
                    v_k = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_k)[0]
                    h_k = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_k)[0]


                """
                sleep phase : from the activities in the top RBM, drive the network top-to-bottom
                """
                ptot = h_k

                ttop = self.rbm_stack["pen+lbl--top"].get_v_given_h(ptot)[0]
                ptov = self.rbm_stack["vis--pen"].get_v_given_h_dir(ttop[:,:-10])[0]

                """
                predictions : compute generative predictions from wake-phase activations,
                              and recognize predictions from sleep-phase activations
                """
                pred_ptov = self.rbm_stack["vis--pen"].get_v_given_h_dir(vtop)[0]

                pred_vtop = self.rbm_stack["vis--pen"].get_h_given_v_dir(ptov)[0]

                """
                update generative parameters :
                here you will only use "update_generate_params" method from rbm class
                """
                self.rbm_stack["vis--pen"].update_generate_params(self,vtop,mini_batch,pred_ptov)

                """
                update parameters of top rbm:
                here you will only use "update_params" method from rbm class
                """
                self.rbm_stack["pen+lbl--top"].update_params(self,htop,ptot,v_k,h_k)

                """
                update generative parameters :
                here you will only use "update_recognize_params" method from rbm class
                """
                self.rbm_stack["vis--pen"].update_recognize_params(self,ptov,ttop,pred_vtop)


                if it % self.print_period == 0 : print ("iteration=%7d"%it)

            self.savetofile_dbn(loc="trained_dbn2",name="vis--pen")
            self.savetofile_rbm(loc="trained_dbn2",name="pen+lbl--top")

        return


    def loadfromfile_rbm(self,loc,name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_rbm(self,loc,name):

        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self,loc,name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return

    def savetofile_dbn(self,loc,name):

        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
