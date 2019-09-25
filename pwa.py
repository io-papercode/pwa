from utils import *

class PWA(object):
    def __init__(self,opts):
        self._opts = opts
        self.lr_adam = opts['lr_adam']
        self.lr_rsgd = opts['lr_rsgd']
        self.batch_size = opts['batch_size']
        self.beta_mmd = opts['beta_mmd']
        self.beta_kld = opts['beta_kld']
        self.lambda_mmd = opts['lambda_mmd']
        self.mmd_kernel = opts['mmd_kernel']
        self.varregnorm = opts['varregnorm']
        self.dim_z = opts['dim_z']
        self.c = opts['hyp_c']
        self.q = opts['mmd_q']
        self.use_expmap= opts['use_expmap']
        
        #build a network
        self.build()
        #launch a session
        #self.sess.run(tf.global_variables_initializer())  
        checkpoint_dir = os.path.expanduser("./"+opts['dataset']+'/ckpt')
        saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        save_steps=3000,
        saver=tf.train.Saver())
        #hooks = [saver_hook]
        hooks = []
        self.sess = tf.train.SingularMonitoredSession(hooks=hooks,checkpoint_dir=checkpoint_dir)

        
    #build a network
    def build(self):
        #input
        self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[None,784])  #shape=(batch_size,input_dim)
        self.training = tf.placeholder(name='training',dtype=tf.bool,shape=None)  #shape=(batch_size,input_dim)

        #encoder
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                e = enc_mnist_small(self.x)
            else:
                e = enc_mnist(self.x)
        else:
            self.celeba_batch = get_celeba_batch(self.batch_size)
            #cn_batch = contrast_norm(self.celeba_batch['inputs'])
            e = enc_conv(opts, self.celeba_batch['inputs'])
            print(e)
            e = tf.reshape(e,[self.batch_size,4*4*opts['num_filters']])

        
        # make these layers fully hyperbolic    
        self.mu = hyp_fflayer(e,self.dim_z,name='mu',activation=None)
        self.log_sigma = tf.layers.dense(e,self.dim_z,name='sigma',activation=None)

        self.z, prior_sample = reparam_h(self.mu, self.log_sigma, radius=0.85, c=self.c, use_expmap=self.use_expmap)
        self.mu_norm = tf.norm(self.mu,2)
        self.sigma_norm = tf.norm(self.log_sigma,2)
        self.z_norm = tf.norm(self.z,2)
        self.z_prior_norm = tf.norm(prior_sample,2)
                        
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                self.x_hat_s, self.x_hat = dec_mnist_small(log_map(self.z))
            else:
                self.x_hat_s, self.x_hat = dec_mnist(log_map(self.z))
            
            epsilon = 1e-10
            self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_hat_s)
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))

        else:
            self.x_hat, _  = dec_conv(opts,self.z,output_shape=[64,64,3])
            self.recon_loss = 0.2*tf.reduce_mean(tf.nn.l2_loss(self.x_hat - self.celeba_batch['targets']))
        
        self.kld_loss = kld_loss(self.mu,self.log_sigma,d=self.dim_z)        
        
        mmd_params = {'nr_mmd_samples':1, 
                  'batch_size':self.batch_size,
                  'mmd_kernel':'RBF',
                  'embedding_dim':self.dim_z,
                  'c':self.c
                 }
        prior_sample = project_hyp_vec(sample_gauss_pd(tf.shape(self.mu),radius=0.88),c=self.c)
        self.mmd_loss = mmd_penalty(mmd_params,prior_sample,self.z,kernel=self.mmd_kernel,q=self.q)
        self.varreg = tf.reduce_mean(tf.norm(self.log_sigma,self.varregnorm,axis=1))
    
        #total loss
        self.loss = self.recon_loss + self.beta_mmd*self.mmd_loss + self.lambda_mmd*self.varreg

        #optimiser
        self.optim_adam = tf.train.AdamOptimizer(learning_rate=self.lr_adam,beta1=0.5)
        self.optim_rsgd = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rsgd)
        self.full_grad_list = self.optim_rsgd.compute_gradients(self.loss)
        self.grad_rsgd,self.grad_adam = self.scale_grad(self.full_grad_list)

        gradients, variables = zip(*self.grad_adam)
        gradients = [tf.check_numerics(gr,'nan found: '+gr.name) for gr in gradients]
        gradients = [tf.clip_by_norm(gr, 1.0-1e-5) for gr in gradients]
        grad_adam_clipped = list(zip(gradients,variables)) 
        # Do two sets of gradients - one with just (mu,sigma) and one without for adam
        apply_rsgd_op, self.rsgd_steps = self.apply_rsgd(self.lr_rsgd)

        self.train_op_1 = self.optim_adam.apply_gradients(grad_adam_clipped)
        # full exp_map
        self.train_op_2 = tf.group([apply_rsgd_op,
                                    self.optim_adam.apply_gradients(grad_adam_clipped)]) 
        # via retraction approximation
        self.train_op_3 = tf.group([self.optim_rsgd.apply_gradients(self.grad_rsgd),
                                  self.optim_adam.apply_gradients(grad_adam_clipped),
                                  self.reproj_weights()])
        
        self.reproj_op = self.reproj_weights()
        
    def get_latent_var(self):
        var_list = []
        for v in tf.trainable_variables():
            if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
                var_list.append(v)
                
        return var_list
            
    @tf.custom_gradient
    def scale_grad_layer(x):
        def grad(g):
            return g*(1-var_norm)**2/4
        return tf.identity(x), grad
                      
    def scale_grad(self,gradient_list,c=1.0):
        grad_rsgd = []
        grad_adam = []
        for g,v in gradient_list:
            if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
                print(g,v)
                grad_rsgd.append((g*riemannian_gradient_c(v,c),v))
            else:
                grad_adam.append((g,v))
        return grad_rsgd, grad_adam
    
    def apply_rsgd(self,lr):
        hyp_vars = self.get_latent_var()
        hyp_grads = tf.gradients(self.loss, hyp_vars)
        clipped_hyp_grads = [tf.clip_by_norm(grad, 1.-1e-5) for grad in hyp_grads]  ###### Clip gradients

        def rsgd(v,g,lr):
            #return exp_map_x(v, -mobius_scalar_mul(lr,g,c=1.0), c=1.0)
            return tf.squeeze(exp_map_x(tf.expand_dims(v,0),-lr*tf.expand_dims(g,0), c=1.0),0)
            
        update_ops = []
        steps = []
        for i in range(len(hyp_vars)):
            riemannian_rescaling_factor = riemannian_gradient_c(hyp_vars[i], c=1.0)
            g_r = riemannian_rescaling_factor * clipped_hyp_grads[i]
            updated_var = rsgd(hyp_vars[i], g_r, lr)
            steps.append(updated_var)
            update_ops.append(tf.assign(hyp_vars[i], updated_var))
        #apply_op = tf.group([tf.assign(v,exp_map_x(v, -lr * g, c=1.0)) for (g,v) in zip(*grad_rsgd)])
        apply_op = tf.group(update_ops)
        return apply_op, steps
    
    def reproj_weights(self):
        #parameter_list = tf.trainable_variables()
        parameter_list = self.get_latent_var()
        #global_norm = tf.Print(tf.global_norm(parameter_list),[tf.global_norm(parameter_list)],first_n=5)
        global_norm = tf.global_norm(parameter_list)
        eps = 1e-5
        reproj_assign_op = tf.group([tf.assign(param,tf.squeeze(param/tf_norm(tf.expand_dims(param,0)))-eps) for param in parameter_list])
        reproj_op = tf.cond(global_norm>=1,true_fn=lambda:reproj_assign_op,false_fn=tf.no_op)

        return reproj_op

    #execute single forward and backward pass    
    #report loss
    def run_single_step(self,x,training):
        _, loss, rloss, mloss, kloss, z_norm, zp_norm, mu, sigma, grads = self.sess.run([self.train_op_2,
                                    self.loss,
                                    self.recon_loss,
                                    self.mmd_loss,
                                    self.kld_loss,
                                    self.z_norm,
                                    self.z_prior_norm,
                                    self.mu,
                                    self.log_sigma,                                                                     
                                    self.full_grad_list], feed_dict={self.x: x, self.training: training})
        #rep = self.sess.run(self.reproj_op)
        #print(np.linalg.norm(mu),np.linalg.norm(sigma))

        return loss, mloss, rloss, kloss
        
    #reconstruction
    #x->x_hat
    def reconstruction(self,x):
        training = False
        return self.sess.run([self.x_hat,self.z], feed_dict={self.x: x, self.training: training})
        
    #generation
    #z->x_hat
    def generate(self,z):
        training = False
        return self.sess.run(self.x_hat,feed_dict={self.z: z, self.training: training})

    #transformation of features
    #x->z
    def transformer(self,x):
        training = False
        return self.sess.run(self.z,feed_dict={self.x: x, self.training: training})