import numpy as np
import math
import random
import matplotlib.pyplot as plt

def getDist(appos, uepos):

    apnum = appos.shape[0]
    uenum = uepos.shape[0]

    dists = np.zeros((apnum,uenum))

    for i in range(apnum):

        dists[i,:] = np.sqrt(np.power(appos[i,0] - uepos[:,0],2) + np.power(appos[i,1] - uepos[:,1],2))

    return dists

def getAchRate(losses,power,noise,apID):

    losses_sqr = 10 ** (losses / 10)

    noise = 10 ** (noise/10)

    power_rec = losses_sqr * power

    acc_rate = np.zeros((losses.shape[1],))

    for i in range(losses.shape[1]):

        power_sig = power_rec[apID[i],i]
        power_inf_noise = np.sum(power_rec[:,i]) - power_sig + noise
        acc_rate[i] = np.log2(1 + power_sig * 1/power_inf_noise)

    return acc_rate

def getInfpower(losses,power,apID):


    losses_sqr = 10 ** (losses / 10)

    power_rec = losses_sqr * power

    infpower = np.zeros(apID.shape)

    for i in range(losses.shape[1]):
        power_sig = power_rec[apID[i],i]
        infpower[i] = np.sum(power_rec[:,i], 0) - power_sig

    return  infpower

def getSINR(losses,infpower,noise, apID):

    SINR = np.zeros(infpower.shape)

    power_real = 10 ** (-20/10)

    losses_sqr = 10 ** (losses / 10)

    noise = 10 ** (noise/10)

    for i in range(infpower.shape[0]):
        SINR[i] = power_real * losses_sqr[apID[i],i] /(infpower[i] + noise)


    return SINR





def APgen(area_range, apnum, min_dist, paint = False):

    pos = np.zeros((apnum,2))

    # if apnum == 2:
    #     pos[0,0] = 200
    #     pos[0,1] = 200
    #     pos[1,0] = 300
    #     pos[1,1] = 300

    for i in range(apnum):

        while True:
            xpos = random.random() * area_range
            ypos = random.random() * area_range

            if np.all(np.sqrt(np.power(pos[:,0] - xpos,2) + np.power(pos[:,1] - ypos,2)) > min_dist):
                pos[i,0] = xpos
                pos[i,1] = ypos
                break

    if paint:
        plt.scatter(pos[:,0],pos[:,1])
        plt.xlim([0,area_range])
        plt.ylim([0,area_range])
        plt.show()

    return pos

def UEgen(appos,area_range,uenum,min_dist,paint = False ):

    uepos = np.zeros((uenum,2))

    for i in range(uenum):

        while True:

            xpos = random.random() * area_range
            ypos = random.random() * area_range

            if np.all(np.sqrt(np.power(appos[:,0] - xpos,2) + np.power(appos[:,1] - ypos,2)) > min_dist):

                uepos[i,0] = xpos
                uepos[i,1] = ypos
                break

    if paint:
        cm = np.linspace(0,appos.shape[0] - 1, appos.shape[0])
        dists = getDist(appos,uepos)
        apID = np.argmin(dists,0)

        plt.figure()
        plt.set_cmap("RdBu_r")
        plt.scatter(appos[:,0], appos[:,1],s=100,marker="x",c=cm)
        plt.scatter(uepos[:,0],uepos[:,1],marker="o",c=apID)
        plt.xlim([0,area_range])
        plt.ylim([0,area_range])
        plt.show()

    return uepos

def DSPloss(dists, conf=[5.8*10e8,3,2,2,4], **kwargs):
    """

    :param dists: array of distances
    :param conf: configuration of dsp loss, including fc; ht; hr; alpha0 and alpha1
    :param shadowing_std: The std of log-normal shadowing
    :param kwargs: mode, pre-defined configs that can be used directly
    :return:
    """

    if len(kwargs) > 0:
        try:
            conf = MODELS[kwargs['mode']]

        except:
            print("Unknown key word or mode not supported!")

    fc = conf[0]
    lambdac = 3e9 / fc
    ht = conf[1]
    hr = conf[2]
    alpha0 = conf[3]
    alpha1 = conf[4]

    #Rc = 4*ht*hr / lambdac
    Rc = 100
    # antenna gain is set to 10 dBi each
    #K0 = 20 * math.log10(lambdac/(4*math.pi)) + 20
    K0 = 39

    mask = dists > Rc
    xdim,ydim = dists.shape
    loss = np.zeros((xdim,ydim))

    loss[~mask] = -10 * alpha0 * np.log10(dists[~mask])
    loss[mask] = 10 * (alpha1 - alpha0) * np.log10(Rc) - 10 * alpha1 * np.log10(dists[mask])



    return loss - K0


def log_norm_shadowing(xdim,ydim,shadowing_std):

    shadowing = np.random.randn(xdim,ydim) * shadowing_std

    return shadowing

def rayleigh_fading(x_dim, y_dim):

    center_freq = 5.8e10 # RF carrier frequency in Hz
    sig_len = x_dim*y_dim
    N = 100 # number of sinusoids to sum

    v = 1 # convert to m/s
    fd = v*center_freq/3e8 # max Doppler shift
    t = np.arange(0, 1, 1/sig_len) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * np.pi
        phi = (np.random.rand() - 0.5) * 2 * np.pi
        x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

    z = (1/np.sqrt(N)) * (x + 1j*y) # this is what you would actually use when simulating the channel
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag)
    z_mag_dB = np.reshape(z_mag_dB,(x_dim,y_dim))

    return z_mag_dB




MODELS = {
    "Macro":[8.6*10e8,60,2,2,4],
    "802.11b":[2.4*10e9,3,2,2,4],
    "802.11a":[5.8*10e9,3,2,2,4],
    "LTE":[7*10e8,5,2,2,4],
    "mmWave":[6*10e10,2,2,2,4],
    "eg":[10e9,10,2,2,4],
}


if __name__ == "__main__":

    appos = APgen(500,4,35)
    uepos = UEgen(appos,500,24,19)
    dists = getDist(appos,uepos)
    fading = rayleigh_fading(4,24)

    loss = DSPloss(dists,mode = "eg")
    shadowing = log_norm_shadowing(4,24,7)
    apID = np.argmax(loss,0)
    loss = loss + fading + shadowing
    acc_rate = getAchRate(loss, 0.01, -134,apID)
    acc_rate.sort()
    print(acc_rate,sum(acc_rate))