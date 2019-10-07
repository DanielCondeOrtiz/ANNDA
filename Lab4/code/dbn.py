from util import *
from rbm import RestrictedBoltzmannMachine
import random

class DeepBeliefNet():

    '''
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
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

            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),

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
        input = self.rbm_stack["hid--pen"].get_h_given_v_dir(self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[0])[0]
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

    def generate(self,true_lbl,name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        print("Generating image: " + str(np.argmax(true_lbl)))

        n_sample = true_lbl.shape[0]

        records = []
        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        lbl = true_lbl

        output = self.rbm_stack["hid--pen"].get_h_given_v_dir(self.rbm_stack["vis--hid"].get_h_given_v_dir(np.random.rand(self.sizes["vis"]))[0])[0] # start the net by telling you know nothing about

        l_k = np.append(output,lbl)

        for i in range(self.n_gibbs_gener):
            #top layer
            t_k,t_k_b = self.rbm_stack["pen+lbl--top"].get_h_given_v(l_k)
            #lower layer
            l_k = self.rbm_stack["pen+lbl--top"].get_v_given_h(t_k)[0]
            l_k_b = self.rbm_stack["pen+lbl--top"].get_v_given_h(t_k_b)[1]

            h_k_b = self.rbm_stack["hid--pen"].get_v_given_h_dir(l_k_b[:-10])[1]

            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_k_b)[0]

            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )

        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))

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

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")

        except IOError :

            print ("training vis--hid")
            """
            CD-1 training for vis--hid
            """
            self.rbm_stack["vis--hid"].cd1(visible_trainset=vis_trainset, n_iterations=n_iterations,max_epochs=15,bool_print=False)

            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()
            """
            CD-1 training for hid--pen
            """
            input1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[0]
            self.rbm_stack["hid--pen"].cd1(visible_trainset=input1, n_iterations=n_iterations,max_epochs=15,bool_print=False)

            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")

            print ("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """
            CD-1 training for pen+lbl--top
            """

            input2 = np.append(self.rbm_stack["hid--pen"].get_h_given_v_dir(input1)[0],lbl_trainset,axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(visible_trainset=input2, n_iterations=n_iterations,max_epochs=15,bool_print=False)

            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")

        print("Checking reconstruction")
        rec_rbm = np.linalg.norm(vis_trainset - self.rbm_stack["vis--hid"].get_v_given_h_dir(self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[0])[0])

        vtoh = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[0]
        htop = self.rbm_stack["hid--pen"].get_h_given_v_dir(vtoh)[0]
        ptot = self.rbm_stack["pen+lbl--top"].get_h_given_v(np.append(htop,lbl_trainset,axis=1))[0]

        ttop = self.rbm_stack["pen+lbl--top"].get_v_given_h(ptot)[0]
        ptoh = self.rbm_stack["hid--pen"].get_v_given_h_dir(ttop[:,:-10])[0]
        htov = self.rbm_stack["vis--hid"].get_v_given_h_dir(ptoh)[0]

        rec_dbn = np.linalg.norm(vis_trainset - htov)

        print("Reconstruction RBM: " + str(rec_rbm))
        print("Reconstruction DBN: " + str(rec_dbn))

        num = random.randint(1,vis_trainset.shape[0]+1)

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(htov[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_dbn.png")

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(vis_trainset[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_orig.png")

        htov = self.rbm_stack["vis--hid"].get_v_given_h_dir(vtoh)[0]

        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(htov[num,:].reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)
        fig.savefig("recons_rbm.png")

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

            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")

        except IOError :

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                """
                wake-phase : drive the network bottom-to-top using visible and label data
                """
                mini_batch = vis_trainset[self.batch_size*(it):self.batch_size*(it+1)]
                mini_lbl = lbl_trainset[self.batch_size*(it):self.batch_size*(it+1)]

                vtoh = self.rbm_stack["vis--hid"].get_h_given_v_dir(mini_batch)[0]
                htop = self.rbm_stack["hid--pen"].get_h_given_v_dir(vtoh)[0]
                ptot = self.rbm_stack["pen+lbl--top"].get_h_given_v(np.append(htop,mini_lbl,axis=1))[0]

                v_k = htop
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
                ptoh = self.rbm_stack["hid--pen"].get_v_given_h_dir(ttop[:,:-10])[0]
                htov = self.rbm_stack["vis--hid"].get_v_given_h_dir(ptoh)[0]

                """
                predictions : compute generative predictions from wake-phase activations,
                              and recognize predictions from sleep-phase activations
                """
                pred_htov = self.rbm_stack["vis--hid"].get_v_given_h_dir(vtoh)[0]
                pred_ptoh = self.rbm_stack["hid--pen"].get_v_given_h_dir(htop)[0]

                pred_htop = self.rbm_stack["hid--pen"].get_h_given_v_dir(ptoh)[0]
                pred_vtoh = self.rbm_stack["vis--hid"].get_h_given_v_dir(htov)[0]

                """
                update generative parameters :
                here you will only use "update_generate_params" method from rbm class
                """
                self.rbm_stack["vis--hid"].update_generate_params(vtoh,mini_batch,pred_htov)
                self.rbm_stack["hid--pen"].update_generate_params(htop,vtoh,pred_ptoh)

                """
                update parameters of top rbm:
                here you will only use "update_params" method from rbm class
                """
                self.rbm_stack["pen+lbl--top"].update_params(np.append(htop,mini_lbl,axis=1),ptot,v_k,h_k)

                """
                update recognize parameters :
                here you will only use "update_recognize_params" method from rbm class
                """
                self.rbm_stack["hid--pen"].update_recognize_params(ptoh,ttop[:,:-10],pred_htop)
                self.rbm_stack["vis--hid"].update_recognize_params(htov,ptoh,pred_vtoh)


                if it % 250 == 0 : print ("iteration=%7d"%it)

            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")

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
