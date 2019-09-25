import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf,erfc

def cayley(u,v):
    x = (u**2 + v**2-1)/(u**2+(1+v)**2)
    y = -2*u/(u**2+(1+v)**2)   
    return (x,y)

def project_hyp_vec(x, c):
    return tf.clip_by_norm(x, clip_norm=(1. - 1e-5) / tf.sqrt(c), axes=[1])

def np_project_hyp_vec(x, c):
    clip_norm = (1.-1e-5)/np.sqrt(c)
    x_norm = np.linalg.norm(x,axis=1)
    return x * clip_norm/ np.expand_dims(np.maximum(x_norm,clip_norm),axis=1)

# Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - 1e-15)) # Only works for positive real x.

def tf_dot(x, y):
    dot = tf.reduce_sum(x * y, axis=1, keepdims=True)
    return dot

def tf_norm(x):
    norm = tf.norm(x, 2, axis = 1, keepdims=True)
    return norm

# Real x, not vector!
def tf_tanh(x):
    return tf.tanh(tf.minimum(tf.maximum(x, -15.0), 15.0))

def riemannian_gradient_c(u, c):
    return tf.squeeze((1-c*tf_norm(tf.expand_dims(u,0))**2)**2/4.0,0)

def tf_lambda_x(x, c):
    return 2. / (1 - c * tf_dot(x,x))

def mobius_scalar_mul(r, v, c=1.0):
    v = v + 1e-7
    norm_v = tf.norm(v,2,axis=1,keepdims=True)
    nomin = tf_tanh(r * tf_atanh(np.sqrt(c) * norm_v))
    result= nomin / (np.sqrt(c) * norm_v) * v
    return project_hyp_vec(result, c)

def mobius_add(u, v, c):
    v = v + 1e-7
    tf_dot_u_v = 2. * c * tf_dot(u, v)
    tf_norm_u_sq = c * tf_dot(u,u)
    tf_norm_v_sq = c * tf_dot(v,v)
    denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
    result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
    return project_hyp_vec(result, c)

def mobius_lin(M,x,c=1.0):
    #Mx = tf.matmul(x,M)
    Mx = tf.einsum('ijk,ik->ik',M,x)       
    Mx_norm = tf.norm(Mx,2,axis=1,keepdims=True)
    x_norm = tf.norm(x,2,axis=1,keepdims=True)
    t = 1/tf.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(tf.sqrt(c)*x_norm))*Mx/Mx_norm
    return project_hyp_vec(t,c)

def np_dist_hp(x,y):
    d_xy = np.linalg.norm(x-y,2,axis=1)
    return np.arccosh(1+d_xy**2/(2*x[:,-1]*y[:,-1]))

def np_dist_pd(x,y):
    x_norm = np.linalg.norm(x,2,axis=1)
    y_norm = np.linalg.norm(y,2,axis=1)
    d_xy = np.linalg.norm(x-y,2,axis=1)
    return np.arccosh(1+2*(d_xy**2)/((1-x_norm**2)*(1-y_norm**2)))

def dist_hp(x,y):
    d_xy = tf.norm(x-y,2,axis=1)
    return tf.acosh(1+d_xy**2/(2*x[:,-1]*y[:,-1]))

def dist_pd(x,y):
    x_norm = tf.norm(x,2,axis=1)
    y_norm = tf.norm(y,2,axis=1)
    d_xy = tf.norm(x-y,2,axis=1)
    return tf.acosh(1+2*d_xy**2/((1-x_norm**2)*(1-y_norm**2)))

def dist_pd_c(x,y,c=1.0):
    return 2/tf.sqrt(c)*tf_atanh(tf.sqrt(c)*tf.norm(mobius_add(-x,y,c),2,axis=1))

def exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    norm_v = tf_norm(v)
    res = tf_tanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)
    return project_hyp_vec(res,c)

def lambda_c(x,c):
    return 2/(1-tf.norm(x,2,axis=1)**2)

def exp_map_x_(x,v,c=1.0):
    v = v + 1e-7
    norm_v = tf_norm(v)
    return mobius_add(x,tf_tanh(tf.sqrt(c)*lambda_c(x,c)*norm_v)*v/(tf.sqrt(c)*norm_v),c)

def exp_map_x(x, v, c):
    v = v + 1e-7 # against nan
    norm_v = tf_norm(v)
    second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return mobius_add(x, second_term, c)

def np_exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    res = np.tanh(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))*v/(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))
    return np_project_hyp_vec(res,c)

def log_map(v,c=1.0):
    #norm_v = tf.Print(tf.norm(v,2,axis=1,keepdims=True),[tf.norm(v,2,axis=1,keepdims=True)])
    norm_v = tf.norm(v,2,axis=1,keepdims=True)
    return tf_atanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)

def hyp_fflayer(x,dim,activation=None,name='',c=1.0):
    with tf.variable_scope("hyp_ff_layer_"+name):
        W = tf.get_variable('weight',shape=[x.get_shape().as_list()[1],dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('hypbias',shape=[dim], initializer=tf.constant_initializer(0.0))
        # input x is Euclidean -> project to disk
        x = exp_map(x,c=1.0) + 1e-7
        #b = exp_map(tf.expand_dims(b,0),c=1.0) + 1e-7
        b = tf.expand_dims(b,0)
        Mx = tf.matmul(x,W) + 1e-7
        Mx_norm = tf.norm(Mx,2,axis=1,keepdims=True)
        x_norm = tf.norm(x,2,axis=1,keepdims=True)
        y = 1./tf.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(tf.sqrt(c)*x_norm))*Mx/Mx_norm
        y = project_hyp_vec(y,c)
        y = mobius_add(y,b,c)
        if activation is not None:
            # hyperbolic non-linearity
            y = exp_map(activation(log_map(y,c)),c)
        return y
    
def np_gauss_prob_halfplane(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_hp(x,mu)**2/(2*sigma**2))
    return p

def np_gauss_prob_pd(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_pd(x,mu**2)/(2*sigma**2))
    return p

def gauss_prob_halfplane(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_hp(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_pd(x,mu)**2/(2*sigma**2))
    return p

def np_sample_gauss_hp(shape,oversample=5,radius=3.0,sigma=1.0):
    # rejection sample in Poincaré half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    #unif_samples = np.stack([r*np.cos(phi), r*np.sin(phi)],axis=1)
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)

    # zero mean, unit sigma in half plane
    mean = np.array([[0.0,1.0]])
    
    p_samples = np_gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (i-w)/(i+w)
    x,y = cayley(u,v)    
    disk_samples = np.stack([x,y],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_gauss_pd(shape,oversample=5,radius=0.85,sigma=1.0):
    # rejection sample in Poincaré half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    #unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    unif_samples = np.stack([r*np.cos(phi), r*np.sin(phi)],axis=1)

    # zero mean, unit sigma in disk
    mean = np.array([[0.0,0.0]])
    p_samples = np_gauss_prob_pd(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (i-w)/(i+w)
    disk_samples = np.stack([u,v],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_gauss_h_2(shape,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincaré half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*np.pi
    #p = np.random.rand(n_samples)
    #r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    x = 2*radius*np.random.rand(n_samples)-radius
    y = radius*np.random.rand(n_samples)
    unif_samples = np.stack([x, y],axis=1)

    # zero mean, unit sigma in half plane
    mean = np.array([[0.0,1.0]])
    
    p_samples = np_gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (w-i)/(w+i)
    x,y = cayley(u,v)
    #disk_samples = tf.stack([x,y],axis=1)/tf.cast(n_samples,tf.float32)
    disk_samples = np.stack([x,y],axis=1)
    disk_samples_hp = np.stack([u,v],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_uniform_h(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    return unif_samples

def np_sample_uniform_h_r(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    return unif_samples

def np_sample_uniform_h_cosh(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    #unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    unif_samples = np.stack([np.cosh(r)*np.cos(phi), np.cosh(r)*np.sin(phi)],axis=1)
    return unif_samples

def sample_gauss_hp(shape,oversample=5,radius=1.0):
    # rejection sample in Poincaré half-plane
    d = shape[1]
    n_samples = tf.reduce_prod(shape)*oversample
    phi = tf.random_uniform([n_samples])*np.pi
    p = tf.random_uniform([n_samples])
    r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    
    # zero mean, unit sigma in half plane
    mean = tf.constant(np.array([[0.0,1.0]]),tf.float32)
    sigma = 1.0
    
    p_samples = gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = tf.reduce_max(p_samples)
    accepted = tf.squeeze(tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value)),1)
    #accepted = tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value),unif_samples,tf.zeros_like(unif_samples))
    # select the samples - make sure it's enough
    u = tf.boolean_mask(unif_samples[:,0],accepted)
    v = tf.boolean_mask(unif_samples[:,1],accepted)
    # transform samples using cayley mapping z = (w-i)/(w+i) - project to unit disk
    x,y = cayley(u,v)    
    #disk_samples = tf.stack([x,y],axis=1)/tf.cast(n_samples,tf.float32)
    disk_samples = tf.stack([x,y],axis=1)
    disk_samples_hp = tf.stack([u,v],axis=1)
    idx = tf.cast(tf.range(tf.reduce_prod(shape)/2),tf.int32)
    return tf.reshape(tf.gather(disk_samples,idx),shape)
   
def sample_gauss_pd(shape,oversample=5,radius=0.85):
    # rejection sample in Poincaré disk
    d = shape[1]
    n_samples = tf.reduce_prod(shape)*oversample
    phi = tf.random_uniform([n_samples])*np.pi
    p = tf.random_uniform([n_samples])
    r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    
    # zero mean, unit sigma in half plane
    mean = tf.constant(np.array([[0.0,0.0]]),tf.float32)
    sigma = 1.0
    
    p_samples = gauss_prob_pd(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = tf.reduce_max(p_samples)
    accepted = tf.squeeze(tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value)),1)
    # select the samples - make sure it's enough
    u = tf.boolean_mask(unif_samples[:,0],accepted)
    v = tf.boolean_mask(unif_samples[:,1],accepted)
    # transform samples using cayley mapping z = (w-i)/(w+i) - project to unit disk
    disk_samples = tf.stack([u,v],axis=1)
    idx = tf.cast(tf.range(tf.reduce_prod(shape)/2),tf.int32)
    return tf.reshape(tf.gather(disk_samples,idx),shape)
    
def reparam_h(mu,sigma,radius,c=1.0,use_expmap=False):
    if use_expmap:
        prior_sample = exp_map(tf.random_normal(tf.shape(mu)),c)
    else:
        #prior_sample = sample_gauss_pd(tf.shape(mu),radius=radius)
        prior_sample = project_hyp_vec(sample_gauss_pd(tf.shape(mu),radius=radius),c)
    sigma = tf.exp(0.5*sigma)
    Sigma = tf.matrix_diag(sigma)
    tsf_eps = mobius_lin(Sigma,prior_sample,c)
    z = mobius_add(mu,tsf_eps,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return z, prior_sample

def sq_distances(x,y,scaled=True):
    sq_norms_x = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    sq_norms_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    dotprods = tf.matmul(x, y, transpose_b=True)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*||x||*||y||
    d = sq_norms_x + tf.transpose(sq_norms_y) - 2. * dotprods
    d_scaled = d/((1-sq_norms_x)*(1-sq_norms_y))
    d_scaled = 1+2*tf.where(d_scaled<1.0, tf.ones_like(d_scaled)+1e-10, d_scaled)
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x,y,q,c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x,y)
    #dotprods = tf.matmul(points, points, transpose_b=True)
    #sq_dist = sq_norms + tf.transpose(sq_norms) - 2. * dotprods
    hyp_dist = tf.acosh(d_xy+1e-6)**q
    return sq_norms_x, d_xy, hyp_dist

def block_diagonal(matrices, dtype=tf.float32):
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

def mmd_penalty(params, sample_pz, sample_qhat, mmd_kernel='laplacian', sigma=None, q=1.0):
    ns = params['nr_mmd_samples']
    n = params['batch_size']
    ls = params['embedding_dim']
    sigma2_p = 1 # params['pz_scale'] ** 2
    params['pz'] = 'normal'

    sq_norms_pz, _, dist_pz = hyp_distances(sample_pz,sample_pz,q,params['c'])
    sq_norms_qhat, dist_qhat_prime, dist_qhat = hyp_distances(sample_qhat,sample_qhat,q,params['c'])
    _, _, dist_pz_qhat = hyp_distances(sample_pz,sample_qhat,q,params['c'])

    mask = block_diagonal(
      [np.ones((ns, ns), dtype=np.float32) for i in range(n)],
      tf.float32)

    # Median heuristic for the sigma^2 of Gaussian kernel
    l = int(n/2)
    if sigma is None:
        sigma2_k = tf.nn.top_k(
          tf.reshape(dist_pz_qhat, [-1]), l).values[l - 1]
        sigma2_k += tf.nn.top_k(
          tf.reshape(dist_qhat, [-1]), l).values[l - 1]
    else:
        sigms2_k = sigma

    if mmd_kernel == 'laplacian':
        res1 = tf.exp( - dist_pz / q / sigma2_k)
        res2 = tf.exp( - dist_pz_qhat / q / sigma2_k)
        res3 = tf.exp( - dist_qhat / q / sigma2_k)
    else: 
        res1 = - dist_pz / q / sigma2_k
        res2 = - dist_pz_qhat / q / sigma2_k
        res3 = - dist_qhat / q / sigma2_k
        
    res1 = tf.multiply(res1, 1. - tf.eye(n)) # filter i=j pairs of samples
    res1 = tf.reduce_sum(res1) / (n * n - n)
    # Part (2)
    res2 = tf.reduce_sum(res2) / (n * n) / ns
    # Part (3)
    res3 = tf.multiply(res3, 1. - tf.eye(n)) # filter i=j pairs of samples
    res3 = tf.reduce_sum(res3) / (n * n - n)
    stat = res1 - 2 * res2 + res3 

    return stat 

def log_likelihood(model, optimizer, n=10):
    z = model.q_z.sample(n)

    log_p_z = optimizer.p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

    log_p_x_z = -tf.reduce_sum(optimizer.bce, axis=-1)

    log_q_z_x = model.q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

    return tf.reduce_mean(tf.reduce_logsumexp(
        tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))

def kld_loss(mu,sigma,d):
    Z = tf.reduce_sum(2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0)),axis=1)
    mu_norm = tf.norm(mu,2,axis=1)
    return tf.reduce_mean(mu_norm**2-mu_norm*(np.sqrt(2.0)*tf.lgamma((d+1)/2)/tf.lgamma(d/2))-tf.log(Z))

fc = tf.layers.dense
def enc_mnist(x):
    f1 = fc(x,512,name='enc_fc1',activation=tf.nn.relu)  #scope is the name of the operation
    f2 = fc(f1,384,name='enc_fc2',activation=tf.nn.relu)  
    f3 = fc(f2,256,name='enc_fc3',activation=tf.nn.tanh)
    return f3

def dec_mnist(z):
    g1 = fc(z,256,name='dec_f1',activation=tf.nn.relu)
    g2 = fc(g1,384,name='dec_f2',activation=tf.nn.relu)
    g3 = fc(g2,512,name='dec_f3',activation=tf.nn.relu)
    x_hat = fc(g3,784,name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def enc_mnist_small(x):
    f1 = fc(x,256,name='enc_fc1',activation=tf.nn.relu)  #scope is the name of the operation
    f2 = fc(f1,128,name='enc_fc2',activation=tf.nn.tanh)  
    return f2

def dec_mnist_small(z):
    g1 = fc(z,128,name='dec_f1',activation=tf.nn.relu)
    g2 = fc(g1,256,name='dec_f2',activation=tf.nn.relu)
    x_hat = fc(g2,784,name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def batch_norm(_input, is_train, reuse, scope, scale=True):
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def conv2d(opts, input_, output_dim, d_h=2, d_w=2, scope=None,
                      conv_filters_dim=None, padding='SAME', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(opts, input_, output_shape, d_h=2, d_w=2, scope=None, 
             conv_filters_dim=None, padding='SAME'):
    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
           
    return deconv

def enc_conv(opts, inputs, bn=False, is_training=False, reuse=False):
    num_units = opts['num_filters']
    num_layers = opts['num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = conv2d(opts, layer_x, num_units / scale,
                             scope='h%d_conv' % i)
        if bn:
            layer_x = batch_norm(layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    
    return layer_x

def dec_conv(opts, z, output_shape, is_training=False, bn=False, reuse=False):
    num_units = opts['num_filters']
    batch_size = tf.shape(z)[0]
    num_layers = opts['num_layers']
    height = int(output_shape[0] / 2**num_layers)
    width = int(output_shape[1] / 2**num_layers)
    
    h0 = fc(z, num_units * height * width, name='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, int(height * scale),
                      int(width * scale), int(num_units / scale)]
        layer_x = deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if bn:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    last_h = deconv2d(
            opts, layer_x, _out_shape, scope='hfinal_deconv')
    return tf.nn.sigmoid(last_h), last_h