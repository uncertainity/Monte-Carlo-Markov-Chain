import pandas as pd
import numpy as np
import scipy
from scipy import stats
import scipy 
from scipy import optimize
import matplotlib.pyplot as plt
import json
import matplotlib
import pymc3 as pm
import theano.tensor as tt

matplotlib.rcParams["figure.figsize"] = (18,8)
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["ytick.major.size"] = 20

N_SAMPLES = 5000


sleep_data = pd.read_csv("Sleep_data.csv")
wake_data = pd.read_csv("Wake_data.csv")

sleep_labels = ["9:00","9:30","10:00","10:30","11:00","11:30","12:00"]
wake_labels = ["5:00","5:30","6:00","6:30","7:00","7:30","8:00"]

print("total no of observations:",len(sleep_data))
print(sleep_data.columns)


fig,axs = plt.subplots(2,1)

## Falling Alseep Plot ##
plt.sca(axs[0])
plt.scatter(sleep_data["time_offset"],sleep_data["indicator"],facecolor = "b",alpha = 0.01,edgecolors = "b")
plt.yticks([0,1],["Awake","Asleep"])
plt.xlabel("PM TIME")
plt.ylabel("Falling asleep data",size = 18)
plt.xticks([-60,-30,0,30,60,90,120],sleep_labels)

## Waking Up plot ##
plt.sca(axs[1])
plt.scatter(sleep_data["time_offset"],sleep_data["indicator"],facecolor = "r",alpha = 0.01,edgecolors = "r")
plt.yticks([0,1],["Asleep","Awake"])
plt.xlabel("AM TIME")
plt.ylabel("Awake data",size = 18)
plt.xticks([-60,-30,0,30,60,90,120],wake_labels)
plt.show()

### From observing the above data, we can see that we can fit a logistic curve ###

def logistic(x,beta,alpha = 0):
    return 1.0/(1.0+np.exp(np.dot(beta,x)+alpha))

## Plot of random logistic functions ##
x = np.linspace(-4,6,2000)
plt.plot(x, logistic(x, 1, 1), 
         label=r"$\beta = 1, \alpha = 1$", color="orange")

plt.plot(x, logistic(x, -1, 1), 
         label=r"$\beta = -1, \alpha = 1$", color="darkblue")
plt.plot(x, logistic(x, -1, -1),
         label=r"$\beta = -1, \alpha = -1$",color="skyblue")
plt.plot(x, logistic(x, -2, 5), 
         label=r"$\beta = -2, \alpha = 5$", color="orangered")
plt.plot(x, logistic(x, -2, -5), 
         label=r"$\beta = -2, \alpha = -5$", color="darkred")
plt.legend(); 
plt.ylabel('Probability'); 
plt.xlabel('t')
plt.title(r'Logistic Function with Varying $\beta$ and $\alpha$');
###########################

sleep_data.sort_values("time_offset",inplace = True)
time = np.array(sleep_data.loc[:,"time_offset"])
sleep_obs = np.array(sleep_data.loc[:,"indicator"])

with pm.Model() as sleep_model:
    alpha = pm.Normal("alpha", mu = 0.0,tau = 0.5, testval = 0.0)
    beta = pm.Normal("beta",mu = 0.0,tau = 0.5,testval = 0.0)
    p = pm.Deterministic("p",1./(1+tt.exp(beta*time + alpha)))
    observed = pm.Bernoulli("obs",p,observed = sleep_obs)
    step = pm.Metropolis()
    sleep_trace = pm.sample(N_SAMPLES,step = step,cores = 2,chains = 2)
    
alpha_samples = sleep_trace["alpha"][5000:,None]
beta_samples= sleep_trace["beta"][5000:,None]


plt.subplot(211)
plt.title(r"""Distribution of $\alpha$ with %d samples""" % N_SAMPLES)

plt.hist(alpha_samples, histtype='stepfilled', 
         color = 'darkred', bins=30, alpha=0.8, density=True);
plt.ylabel('Probability Density')


plt.subplot(212)
plt.title(r"""Distribution of $\beta$ with %d samples""" % N_SAMPLES)
plt.hist(beta_samples, histtype='stepfilled', 
         color = 'darkblue', bins=30, alpha=0.8, density=True)
plt.ylabel('Probability Density');

time_est = np.linspace(time.min()-15,time.max()+15,100)[:,None]
alpha_est = alpha_samples.mean()
beta_est = beta_samples.mean()
sleep_est = logistic(time_est,beta = beta_est,alpha = alpha_est)

plt.plot(time_est, sleep_est, color = 'navy', 
         lw=3, label="Most Likely Logistic Model")
plt.scatter(time, sleep_obs, edgecolor = 'slateblue',
            s=50, alpha=0.2, label='obs')
plt.title('Probability Distribution for Sleep with %d Samples' % N_SAMPLES)
plt.legend(prop={'size':18})
plt.ylabel('Probability')
plt.xlabel('PM Time')
plt.xticks([-60, -30, 0, 30, 60, 90, 120], sleep_labels);

