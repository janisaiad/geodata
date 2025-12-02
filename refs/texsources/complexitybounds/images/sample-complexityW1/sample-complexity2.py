import matplotlib
matplotlib.use('Agg')

from pylab import *



def runSinkhorn_log(epsilon,C,n_it,n):

    def M(u,v)  : 
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + np.tile(u,[n,1]).T + np.tile(v,[n,1])) / epsilon
 
    def lse(A) :
        "log-sum-exp"
        return log(sum(exp(A),axis=1) + 1e-12) # add 10^-6 to prevent NaN
     
    u = zeros(n)
    v = zeros(n)
      
    for i in range(n_it) :
        u_prev = u
        u =  - epsilon * ( log(1/n) + lse(M(u,v)) ) + u
        v =  - epsilon * ( log(1/n) + lse(M(u,v).T)) + v
        if norm(u-u_prev,ord=inf) < 10**(-3):
            break
       
    pi = exp(M(u,v))
    W = 1/n* sum(u) + 1/n * sum(v) - 1/(n*n)*epsilon*sum(pi)    
    return W


# In[40]:


def cost_mat(X,Y,n,p=2):
    C = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            C[i,j] = np.sum(abs(X[i,:]-Y[j,:])**p)
    return C


# In[41]:


def alpha(n,d):
    X = rand(n,d)
    return X
    
def beta(n,d):
    X = rand(n,d)+2
    return X


# In[42]:

p = 1 #exponent for the cost

def SinkhornDivLog(X,Y,n,epsilon):
    CXY = cost_mat(X,Y,n,p)
    CXX = cost_mat(X,X,n,p)
    CYY = cost_mat(Y,Y,n,p) 
    WXY = runSinkhorn_log(epsilon,CXY,n_it,n)
    WXX = runSinkhorn_log(epsilon,CXX,n_it,n)
    WYY = runSinkhorn_log(epsilon,CYY,n_it,n)

    return WXY - 1/2*WXX - 1/2*WYY


def plot_curves_d(i_eps):
    figure()
    for i_d in range(N_d):
        plot(log10(n_list),mean_err[:,i_eps,i_d],label = 'd='+str(d_list[i_d]))
        fill_between(log10(n_list), mean_err[:,i_eps,i_d] - sd_err[:,i_eps,i_d],mean_err[:,i_eps,i_d]
                 + sd_err[:,i_eps,i_d],alpha=.2 )
    legend()
    title('epsilon='+str(epsilon_list[i_eps]))
    savefig('fix_eps'+str(i_eps)+'.png')

def plot_curves_eps(i_d):
    figure()
    for i_eps in range(N_eps):
        plot(log10(n_list),mean_err[:,i_eps,i_d],label = 'epsilon='+str(epsilon_list[i_eps]))
        fill_between(log10(n_list), mean_err[:,i_eps,i_d] - sd_err[:,i_eps,i_d],mean_err[:,i_eps,i_d] 
                 + sd_err[:,i_eps,i_d],alpha=.2 )
    title('d='+str(d_list[i_d]))
    legend()
    savefig('fix_d'+str(i_d)+'.png')


# In[47]:


n_it = 200

iter_total = 100
n_list = logspace(1,2.5,15)
d_list = [2,3,5,7]
epsilon_list = logspace(2,-3,6)

N_n = len(n_list)
N_d = len(d_list)
N_eps = len(epsilon_list)

err = zeros([N_n,N_eps,N_d,iter_total])

for i_d in range(N_d):
    d = d_list[i_d]
    print('d = '+str(d))
    for i_eps in range(N_eps):
        epsilon = epsilon_list[i_eps]
        print('---epsilon = '+str(epsilon))
        for i_n in range(N_n) :
            n = int(n_list[i_n])
            print('-----------n = '+str(n))
            for j in range(iter_total):
                W = SinkhornDivLog(alpha(n,d),alpha(n,d),n,epsilon)
                err[i_n,i_eps,i_d,j] = W
    mean_err = log10(mean(err,axis=-1))
    sd_err = std(log10(err),axis=-1)
    plot_curves_eps(i_d) # influence of epsilon for fixed d


mean_err = log10(mean(err,axis=-1))
sd_err = std(log10(err),axis=-1)

#fix epsilon, change d
for i_eps in range(N_eps):
    plot_curves_d(i_eps)


