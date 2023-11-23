import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats
from scipy.optimize import curve_fit
import scalt_compute as sc
import time
import imageio
import pandas as pd

# Error class
class StabilityError(Exception):
    pass

@jit(nopython=True)
def SPDE_solver(ICs = [0.16, 30.0, 0.64, 0.88, 0.61, 0.12, 13.0, 0.0],  # Initial conditions
              dt = 0.01,                                        # time step
              dx = 0.01,                                        # space step
              tmax = 900,                                     # total time
              m = 5,                                            # spatial discretizations
              tau_c = 0.01,                                     # Ornstein-Uhlenbeck time constant
              D = 0.4,                                          # Diffusion for IP3
              Vs = 0.01252 ,                                      # Max conductane SOCE (0.0301)
              rho = 0.00005,                                    # Diffusion for PDE
              v3 = 120,
              vr = 18,
              seed = 9,
              alpha = 1.0,
              beta = 1.0,
              v_ip3 = 0.88,
              ip3 = 0.2,
              v_pmca = 0.6,
              vncx = 1.4,
              Vmean=-60,
              eta = 0.45,
              ATP=0*10**(-3), #in M
              ATP_start=50,
              ATP_time=0, #in s
              Glut_start=50,
              G=0*10**(-3), #in M
              Glut_time=0,
              mode=0,
              ):
    
    # Random seed
    #np.random.seed(seed)
    # Simulation parameters
    tspan = np.arange(0.0,(tmax-dt),dt)       
    n = len(tspan)                                                                                
    r = rho*dt/(dx**2)
    if r >= 0.5:
        raise StabilityError('The value rho*dt/(dx**2) must be < 0.5 to ensure numerical stability!')
      
    # Model Parameters
    
    R = 8.314
    temp = 300 
    ksat = 0.25
    kmcao = 1.3
    kmnao = 97.63
    kmnai = 12.3
    kmcai = 0.0026
    kn = 0.1
    hc3 = 2
    k3serc = 0.3
    v2 = 0.5
    d5 = 0.08234
    k_pmca = 0.8
    kb1 = 0.2573
    fi = 0.05
    gamma = 9
    fe = 0.05
    a2 = 0.2
    d1 = 0.13
    d2 = 1.049
    d3 = 0.9434
    Ks = 50
    
    gsocc=1 #nS
    ksoc=939*10**(-3) #mM
    
    
    ka = 0.01920
    kb = 0.2573
    kc = 0.0571
    kca = 5
    kna = 5
    
    f_Na=0.477
    f_K=0.477
    f_Ca=0.046
    
    k1 = 0.3
    k2 = 1260#40000/31.75 #31.6 ???
    k3 = 2.4
    k4 = 1575#50000/31.75
    k5 = 1.58
    k6 = 221#7000/31.75
    L1 = 0.0001
    L2 = 0.004
    L3 = 0.05
    H1 = 0.001
    H2 = 0.01
    Hat2 = 0.1#0.01
    H4 = 0.6
    H3=0
    g12 = 7.5*3 # nS
    g34 = 7.5*3 # nS
    
    T = 300
    F = 96485.3321
    Vosteo=6.5
    Kout=4#4
    Caout=2.5#2.5
    cao=Caout
    Naout=140#145
    nao=Naout
    
    #K fct

    barInak=22.6 #pA
    Knakko=1.32 #mM
    Knakna=14.5 #mM
    
    Krest=120
    Narest=12
    tetana=10
    tetak=10
    
    #AMPA
    Rb=13*10**6/1300
    
    R0=2.7*10**3/100
    Rc=200/500
    
    Ru1=5.9/75
    Ru2=8.6*10**4/750
    
    Rd=9/100
    Rr=0.64/50
    
    gAMPA= 4.0 #nS
    
    
    #NMDA
    Rbn=5*10**6/60
    Run=12.9/60 
    
    R0n=46.5 
    Rcn=73.8/10
    
    Mgo=1 #1 to 2 mM
    
    Rdn=8.4/600
    Rrn=6.8/60
    
    gNMDA=5.6/2 #nS
    
    
    #IP3
    alphatp=0.03
    katp=1
    gammatpglut=0.01
    alphaglut=0.03
    kglut=1
    
    
    
    #L-type/T-type
    gL=0.8#3.5 #nS
    gT=0.5#0.006 #nS
    
    # Jk
    
    gk=11.58 #nS
    
    #Jleak
    
    gleak=1 #nS
    Vl=30 #mV
    
    #NCX
    
    gncx=0.02 #nS

    Cm=0.9 #µF/cm^2
    sur=5*10**(3) # 5-50 µm^2
    
    gleakK=2
    Vlk=70
    
    # Functions
    
    
    def Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt):
        return (g12*(Q1+Q2)+g34*(Q3+Q4))*volt
    
    def Jp2x7_Ca(Ca, Na, K, Q1, Q2, Q3, Q4,volt):
        alpha=1/(2*Vosteo*F)
        jp2x7=-f_Ca*alpha*Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt)*10**(6)
        return jp2x7
    
    def Jp2x7_Na(Ca, Na, K, Q1, Q2, Q3, Q4,volt):
        alpha=1/(Vosteo*F)
        jp2x7=-f_Na*alpha*Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt)*10**(3)
        return jp2x7
    
    def Jp2x7_K(Ca, Na, K, Q1, Q2, Q3, Q4,volt):
        alpha=1/(Vosteo*F)
        jp2x7=-f_K*alpha*Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt)*10**(3)
        return jp2x7
    

    
    ###AMPA and NMDA### 
    def Inmda(On, Ca, Na, K,volt):
        B=1/(1+np.exp(-0.062*volt)*Mgo/3.57)
        return gNMDA*B*On*volt
     
    def Iampa(O, Ca, Na, K,volt):
        return gAMPA*O*volt   
     
    def Jampa_Ca(O, Ca, Na, K,volt):
        alpha=1/(2*Vosteo*F)
        return -f_Ca*alpha*Iampa(O, Ca, Na, K,volt)*10**(6)
    
    def Jnmda_Ca(On, Ca, Na, K,volt):
        alpha=1/(2*Vosteo*F)
        return -f_Ca*alpha*Inmda(On, Ca, Na, K,volt)*10**(6)
    

    def Jampa_Na(O, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_Na*alpha*Iampa(O, Ca, Na, K,volt)*10**3
    
    def Jnmda_Na(On, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_Na*alpha*Inmda(On, Ca, Na, K,volt)*10**3
    
    
    def Jampa_K(O, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)

        return -f_K*alpha*Iampa(O, Ca, Na, K,volt)*10**3
    
    def Jnmda_K(On, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_K*alpha*Inmda(On, Ca, Na, K,volt)*10**3
    
    #######
    
    def Isoc(Cer,volt):
        Eca=100

        return gsocc*np.tanh(Cer-ksoc)/2*(volt-Eca)

    def Jsocc(Cer,volt):
        Eca=100
        alpha=1/(2*Vosteo*F)
        barvs=2*Vs/((-67-Eca)*alpha)
        return alpha*barvs*Isoc(Cer, volt)
        
    def Inak(Ca,Na,K,volt,Kout):
        return barInak*Kout/(Kout+Knakko)*Na**1.5/(Na**1.5+Knakna**1.5)*(volt+135.1)/(volt+300)
    
    def Jnak(Ca,Na,K,volt,Kout):
        alpha=1/(Vosteo*F)
        return -alpha*Inak(Ca,Na,K,volt,Kout)*10**3

    
    def hinf(Ca,IP3): 
        return (d2 * (IP3 + d1) / (IP3 + d3)) / ((d2 * (IP3 + d1) / (IP3 + d3)) + Ca)
    
    def ninf(Ca): 
        return Ca / (Ca + d5)
    
    def Kx(Ca, Na): 
        return kmcao * (Na**3) + (kmnao**3) * Ca + (kmnai**3) * cao * (1 + Ca / kmcai) + kmcai * (nao**3) * (1 + (Na**3) / (kmnai**3)) + (Na**3) * cao + (nao**3) * Ca
    
    def soc_inf(Cer): 
        return (Ks**4) / ((Ks**4) + (Cer**4))
    
    def winf(Ca): 
        return (ka / (Ca**4) + 1 + (Ca**3) / kb) / (1 / kc + ka / (Ca**4) + 1 + (Ca**3) / kb)
    
    def xinf(Ca, Na): 
        return 1 - 1 / ((1 + (Ca / kca)**2) * (1 + (kna / Na)**2))
    
    
    def Jip3(Ca, Cer, h,IP3):
        minf = IP3 / (IP3 + d1)

        return v_ip3 * (minf**3) * (ninf(Ca)**3) * (h**3) * (Cer - Ca)
    
    
    def Jserca(Ca): 

        return v3 * (Ca**hc3) / ((k3serc**hc3) + (Ca**hc3))
    
    
    def Jleak(Ca, Cer): 
        return v2 * (Cer - Ca)
    
    
    def Jryr(Ca, Cer, w): 
        return vr * w * (1 + (Ca**3) / kb1) / (ka / (Ca**4) + 1 + (Ca**3) / kb1) * (Cer - Ca)
    
    
    def Jin(Ca,Cer, Na,K, Q1, Q2, Q3, Q4,volt):
        return (Jp2x7_Ca(Ca,Na,K, Q1, Q2, Q3, Q4, volt)+Jsocc(Cer,volt))
    
    def ninfi(Ca): 
        return 1 / (1 + (kn / Ca)**2)
    
    def Incx(Ca, Na,volt): 
        VFRT = volt*10**(-3) * F / (R * temp)
        nin=ninfi(Ca)
        Ca=Ca*10**(-3)
        
        if mode==2:
            incx=nin * (gncx*volt * (np.exp(eta * VFRT) * (Na**3) * cao ) / (Kx(Ca, Na) * (1 + ksat * (np.exp((eta - 1) * VFRT)))))
            return incx
        if mode==1:
            incx=nin * (gncx*volt* (-np.exp((eta - 1) * VFRT) * (nao**3) * Ca) / (Kx(Ca, Na) * (1 + ksat * (np.exp((eta - 1) * VFRT)))))
            return incx
        else:
            incx=nin * (gncx*volt * (np.exp(eta * VFRT)  * (Na**3) * cao -np.exp((eta - 1) * VFRT) * (nao**3) * Ca) / (Kx(Ca, Na) * (1 + ksat * (np.exp((eta - 1) * VFRT)))))
            return incx
        
    def Jncxca(Ca, Na, volt):
        alphaca=1/(2*Vosteo*F)
        return -alphaca*Incx(Ca, Na, volt)*10**6
    
    def Jncxna(Ca, Na, volt):

        alphana=1/(Vosteo*F)
        return -alphana*Incx(Ca, Na, volt)*10**3
    


    def Jpmca(Ca): 
        return (v_pmca * (Ca**2) / ((Ca**2) + (k_pmca**2)))
        
    def Ik(K,volt,Kout):
        Ek=R*T/(F)*np.log(Kout/K)*10**3
        return gk*np.sqrt(Kout)*(volt-Ek) #surface ????
    
    def Jk(K, volt,Kout):
        alpha=1/(Vosteo*F)
        return -alpha*Ik(K, volt,Kout)*10**3
    
    def Ileak(volt):
        return gleak*(volt-Vl)
    
    
    def Ileak_K(volt):
        return gleakK*(volt-Vlk)
    
    def Jleak_K(K,volt):
        alpha=1/(Vosteo*F)
        return -alpha*Ileak_K(volt)*10**3+(Krest-K)/tetak
    
    ####L-type and T-type####
    
    def barmT(volt):
        return 1/(1+np.exp(-(volt+5)/6))

    def barhT(volt):
        return 1/(1+np.exp((volt+5)/10))



    def tauhTf(volt):
        return np.exp(-((volt+5)/10)**2)+2


    def taumT(volt):
        return np.exp(-((volt+5)/6)**2)+2

    def barmL(volt):
        return 1/(1+np.exp(-(volt)/6))

    def taumL(volt):
        return (18*np.exp(-((volt+45)/20)**2)+10)/10

    def tauhL(volt):
        return (10*np.exp(-((volt+100)/10)**2)+20)/10

    def barhL(volt):
        return 1/(1+np.exp((volt)/5))
    

    def Il(Ca,mL,hL,volt):
        Eca=R*T/(2*F)*np.log(Caout/(Ca*10**-3))*10**3
        return gL*mL*hL*(volt-Eca)
    
    def Jl(Ca,mL,hL,volt):
        alpha=1/(2*Vosteo*F)
        return -alpha*Il(Ca,mL,hL,volt)*10**6

    def It(Ca,mT,hTf,volt):
        Eca=R*T/(2*F)*np.log(Caout/(Ca*10**-3))*10**3
        return gT*mT*(hTf)*(volt-Eca)
        

    def Jt(Ca,mT,hTf,volt):
        alpha=1/(2*Vosteo*F)
        return -alpha*It(Ca,mT,hTf,volt)*10**6
    
    

    ####scheme###
    def dCa_dt(Ca, Cer, h, s, w, Na, K, Q1, Q2, Q3, Q4,O,On,IP3,volt,mL,mT,hTf,hL):
        #print([Jip3(Ca, Cer, h,IP3),Jleak(Ca, Cer),Jryr(Ca, Cer, w),-Jserca(Ca),Jin(Ca,Cer, Na,K, Q1, Q2, Q3, Q4,volt), Jncxca(Ca,Na ,volt), -Jpmca(Ca)])
        return fi * (Jip3(Ca, Cer, h,IP3)+Jleak(Ca, Cer)+Jryr(Ca, Cer, w)-Jserca(Ca)+ Jin(Ca,Cer, Na,K, Q1, Q2, Q3, Q4,volt)+ Jncxca(Ca,Na ,volt)- Jpmca(Ca)+Jnmda_Ca(On, Ca, Na, K,volt)+Jampa_Ca(O, Ca, Na, K,volt)+Jt(Ca,mT,hTf,volt)+Jl(Ca,mL,hL,volt))#
    
    def dNa(Ca, Cer ,K,Na,volt,O,On,Q1,Q2,Q3,Q4,Kout):
        return -3*Jncxna(Ca,Na,volt)-3*Jnak(Ca, Na, K,volt,Kout)+Jp2x7_Na(Ca, Na, K, Q1, Q2, Q3, Q4,volt)+(Narest-Na)/tetana+Jnmda_Na(On, Ca, Na, K,volt)+Jampa_Na(O, Ca, Na, K,volt)
    
    def dK(Ca,K,Na,volt,O,On,Q1,Q2,Q3,Q4,Kout):
        return 2*Jnak(Ca, Na, K,volt,Kout)-Jp2x7_K(Ca, Na, K, Q1, Q2, Q3, Q4,volt)-Jnmda_K(On, Ca, Na, K,volt)-Jampa_K(O, Ca, Na, K,volt)+Jk(K, volt,Kout)+Jleak_K(K,volt)
    
    def fvolt(Ca,Cer,K,Na,volt,O,On,Q1,Q2,Q3,Q4,mT,hTf,mL,hL,Kout):
        return (-Iampa(O, Ca, Na, K,volt)-Inmda(On, Ca, Na, K,volt)-Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt)-Ileak(volt)-Isoc(Cer,volt)-Ik(K,volt,Kout)-Inak(Ca,Na,K,volt,Kout)-Incx(Ca, Na, volt)-Il(Ca,mL,hL,volt)-It(Ca,mT,hTf,volt)-Ileak_K(volt))/(Cm*sur)*10*10
    
        # Initialization of state variables & noise
    Ca, Cer, h, s, w, x, Na, K, eta_u,volt = [np.zeros((n, m)) for i in range(10)]
    D1, D2, D3, D4, C1, C2, Q1, Q2, C4, C3, Q4, Q3, C0a, C1a, C2a, O, D1a, D2a, C0n, C1n, C2n, On, D2n, IP3,Ko = [np.zeros(n) for i in range(25)]
    Ca[0, :]    = ICs[0]
    Cer[0, :]   = ICs[1]
    h[0, :]     = ICs[2]
    s[0, :]     = ICs[3]
    w[0, :]     = ICs[4]
    x[0, :]     = ICs[5]     
    Na[0, :]    = ICs[6] 
    K[0, :]     = ICs[7]    
    eta_u[0, :] = ICs[8]
    volt[0, :] =ICs[32]
    
    D1[0  ] = ICs[9]
    D2[0  ] = ICs[10]
    D3[0  ] = ICs[11]
    D4[0  ] = ICs[12]
    C1[0  ] = ICs[13]
    C2[0  ] = ICs[14]
    C3[0  ] = ICs[15]
    C4[0  ] = ICs[16]
    Q1[0  ] = ICs[17]
    Q2[0  ] = ICs[18]
    Q3[0  ] = ICs[19]
    Q4[0  ] = ICs[20]
    
    C0a[0   ]= ICs[21]
    C1a[0   ]= ICs[22]
    C2a[0   ]= ICs[23]
    O[0   ]  = ICs[24]
    D1a[0   ]= ICs[25]
    D2a[0   ]= ICs[26]
    
    C0n[0   ]= ICs[27]
    C1n[0   ]= ICs[28]
    C2n[0   ]= ICs[29]
    On[0   ]  = ICs[30]
    D2n[0   ]= ICs[31]
    IP3[0 ]=0
    Ko[0]=Kout
    
    saveA=np.zeros(n)
    for k in range(n-1):
        if k<ATP_start*100:
            A=0
        elif k>ATP_start*100+ATP_time*100:
            A=0
        else:
            A=ATP
        # if np.random.binomial(1, 0.005)==1:
        #         trace=k
        #         rdn=np.abs(np.random.randn(1)*10**-3)[0]
        # if k<trace+10:
        #         A=rdn
        if k<Glut_start*100:
            Glut=0
        elif k>Glut_start*100+Glut_time*100:
            Glut=0
        else:
            Glut=G
        # if np.random.binomial(1, 0.005)==1:
        #             trace=k
        #             rdn=np.abs(np.random.randn(1)*10**-3)[0]
        # if k<trace+10:
        #             Glut=rdn
        saveA[k]=Glut

            
        #print(f'dynam time {time1-str_time}')
        tot=1#8*k1+3*4*k2*A+2*2*k3+2*2*k4*A+2*3*k5+2*k6*A+L1+L2+L3+H1+H2+Hat2+H3+H4
        totn=1#Rbn*Glut*2+R0n+Rcn+Rrn+Rdn+Run*2
        tota=1#Rb*Glut*2+R0+Rc+Rr*2+Rd*2+Ru1+Ru2
        D1[k + 1 ] = D1[k ]+ (k1/tot*D2[k ]-(3*k2*A/tot+H1/tot)*D1[k ])*dt
        D2[k + 1 ] = D2[k ]+ (3*k2*A/tot*D1[k ]+2*k3/tot*D3[k ]+H2/tot*C2[k ] -(k1/tot+2*k4*A/tot+H3/tot)*D2[k ])*dt
        D3[k + 1 ] = D3[k ]+ (2*k4*A/tot*D2[k ]+3*k5/tot*D4[k ]+Hat2/tot*Q1[k ]-(2*k3/tot+k6*A/tot)*D3[k ])*dt
        D4[k + 1 ] = D4[k ]+ (k6*A/tot*D3[k ]+H4/tot*Q2[k ]-3*k5/tot*D4[k ])*dt
        C1[k + 1 ] = C1[k ]+(H1/tot*D1[k ]+k1/tot*C2[k ]+L1/tot*C4[k ]-3*k2*A/tot*C1[k ])*dt
        C2[k + 1 ] = C2[k  ]+ (H3/tot *D2[k ] + 3*k2*A/tot*C1[k ]+2*k3/tot*Q1[k ]-(k1/tot+2*k4*A/tot+H2/tot)*C2[k ])*dt
        Q1[k + 1 ] = Q1[k ]+ (2*k4*A/tot*C2[k ]+3*k5/tot*Q2[k ]-(2*k3/tot+k6*A/tot+Hat2/tot)*Q1[k ])*dt
        Q2[k + 1 ] = Q2[k ] +(k6*A/tot*Q1[k ]+L2/tot*Q3[k ]-(3*k5/tot+L3/tot+H4/tot)*Q2[k ])*dt
        C4[k + 1 ] = C4[k ]+ (k1/tot*C3[k ]-(L1/tot+3*k2*A/tot)*C4[k ])*dt
        C3[k + 1 ] = C3[k ]+(3*k2*A/tot*C4[k ]+2*k1/tot*Q4[k ]-(k1/tot+2*k2*A/tot)*C3[k ])*dt
        Q4[k + 1 ] = Q4[k  ] +(2*k2*A/tot*C3[k ]+3*k1/tot*Q3[k ]-(2*k1/tot+k2*A/tot)*Q4[k ])*dt
        Q3[k + 1 ] = Q3[k  ]+ (k2*A/tot*Q4[k ]+L3/tot*Q2[k ]-(3*k1+L2)/tot*Q3[k ])*dt#(1-(D1[k ] +D2[k ]+D3[k ]+ D4[k ]+C1[k ]+C2[k ]+Q1[k ]+Q2[k ]+C4[k ]+C3[k ]+Q4[k ]))#
          
        #IP3
        IP3[k + 1]= IP3[k] + (alphatp*(A*1000/(A*1000+katp))-gammatpglut*IP3[k]+alphaglut*(Glut*1000/(Glut*1000+kglut)))*dt
        
        
        #Kout
        if k>50 *100:
            Ko[k + 1]= Kout
        else:
            Ko[k + 1]= Kout

        #AMPA
        C0a[k + 1 ] = C0a[k ]+(-Rb*Glut/tota*C0a[k  ]+Ru1/tota*C1a[k  ])*dt
        C1a[k + 1 ] = C1a[k ]+(Rr/tota*D1a[k ]+Ru2/tota*C2a[k ]+Rb*Glut/tota*C0a[k  ]-(Rd+Ru1+Rb*Glut)/tota*C1a[k ])*dt
        C2a[k + 1 ] = C2a[k ]+(Rc/tota*O[k ]+Rr/tota*D2a[k ]+Rb*Glut/tota*C1a[k ]-(Rd+Ru2+R0)/tota*C2a[k ])*dt
        O[k + 1 ] = O[k ]+(R0/tota*C2a[k ]-Rc/tota*O[k ])*dt
        D1a[k + 1 ] = D1a[k ]+(Rd/tota*C1a[k ]-Rr/tota*D1a[k ])*dt
        D2a[k + 1 ] = D2a[k ]+(Rd/tota*C2a[k ]-Rr/tota*D2a[k ])*dt
        
        #NMDA
        C0n[k + 1 ] = C0n[k ]+(-Rbn/totn*Glut*C0n[k  ]+Run/totn*C1n[k  ])*dt
        C1n[k + 1 ] = C1n[k ]+(Run/totn*C2n[k ]+Rbn*Glut/totn*C0n[k  ]-(Run+Rbn*Glut)/totn*C1n[k ])*dt
        C2n[k + 1 ] = C2n[k ]+(Rcn/totn*On[k ]+Rdn/totn*D2n[k ]+Rbn*Glut/totn*C1n[k ]-(Rrn+R0n+Run)/totn*C2n[k ])*dt
        On[k + 1 ] = On[k ]+(R0n/totn*C2n[k ]-Rcn/totn*On[k ])*dt
        D2n[k + 1 ] = D2n[k ]+(Rrn/totn*C2n[k ]-Rdn/totn*D2n[k ])*dt
        
    
    
    noise_term  = np.random.randn(n, m)
    # FTCS Scheme  (∂u/∂t = ∂²u/∂x² + f(u(t, x)) 
    IP3=IP3+ip3

    saveL=np.zeros(n)
    
    savehL=np.zeros(n)
    saveT=np.zeros(n)
    saveTf=np.zeros(n)
        
    
    for k in range(n-1):
        # BCs: Neumann (∂C/∂t = 0) at x₁ and xₘ
        
        
        
        mL=(1.4135e-5-barmL(volt[k, 0]))*np.exp(-k*dt/taumL(volt[k, 0]))+barmL(volt[k, 0])
        hL=(0.999998-barhL(volt[k, 0]))*np.exp(-k*dt/tauhL(volt[k, 0]))+barhL(volt[k, 0])

        mT=(3.2529e-05-barmT(volt[k, 0]))*np.exp(-k*dt/taumT(volt[k, 0]))+barmT(volt[k, 0])
        hTf=(0.99797-barhT(volt[k, 0]))*np.exp(-k*dt/tauhTf(volt[k, 0]))+barhT(volt[k, 0])
        
        
        Ca[k + 1, 0]   = 2 * r * Ca[k, 1]     + (1 - 2*r) * Ca[k, 0]   + dCa_dt(Ca[k, 0], Cer[k, 0], h[k, 0], s[k, 0], w[k, 0], Na[k, 0], K[k, 0], Q1[k], Q2[k], Q3[k], Q4[k],O[k], On[k],IP3[k],volt[k, 0],mL,mT,hTf,hL) * dt
        Na[k + 1, 0]   = 2 * r * Na[k, 1]     + (1 - 2*r) * Na[k, 0]   + dNa(Ca[k, 0],Cer[k, 0], K[k, 0],Na[k, 0],volt[k, 0],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k]) * dt
        K[k + 1, 0]    = 2*  r * K[k,  1]     + (1 - 2*r) * K[k, 0]    + dK(Ca[k, 0],K[k, 0],Na[k, 0],volt[k, 0],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k])*dt

        
        
        mL=(1.4135e-5-barmL(volt[k, m-1]))*np.exp(-k*dt/taumL(volt[k, m-1]))+barmL(volt[k, m-1])
        hL=(0.999998-barhL(volt[k, m-1]))*np.exp(-k*dt/tauhL(volt[k, m-1]))+barhL(volt[k, m-1])
        mT=(3.2529e-05-barmT(volt[k, m-1]))*np.exp(-k*dt/taumT(volt[k, m-1]))+barmT(volt[k, m-1])
        hTf=(0.99797-barhT(volt[k, m-1]))*np.exp(-k*dt/tauhTf(volt[k, m-1]))+barhT(volt[k, m-1])
        
        Ca[k + 1, m-1]   = 2 * r * Ca[k, m-2]   + (1 - 2*r) * Ca[k, m-1] + dCa_dt(Ca[k, m-1], Cer[k, m-1], h[k, m-1], s[k, m-1], w[k, m-1], Na[k, m-1], K[k, m-1], Q1[k], Q2[k], Q3[k], Q4[k], O[k], On[k],IP3[k],volt[k, m-1],mL,mT,hTf,hL) * dt
        Na[k + 1, m-1]   = 2 * r * Na[k, m-2]   + (1 - 2*r) * Na[k, m-1]   + dNa(Ca[k, m-1],Cer[k, m-1], K[k, m-1],Na[k, m-1],volt[k, m-1],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k]) * dt
        K[k + 1,  m-1]   = 2*  r * K[k,  m-2]   + (1 - 2*r) * K[k, m-1]    + dK(Ca[k, m-1],K[k, m-1],Na[k, m-1],volt[k, m-1],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k])*dt

        for i in range(m):

            mL=(1.2135e-5-barmL(volt[k, i]))*np.exp(-k*dt/taumL(volt[k, i]))+barmL(volt[k, i])
            hL=(0.999998-barhL(volt[k, i]))*np.exp(-k*dt/tauhL(volt[k, i]))+barhL(volt[k, i])
            
            mT=(3.2529e-05-barmT(volt[k, i]))*np.exp(-k*dt/taumT(volt[k, i]))+barmT(volt[k, i])
            hTf=(0.99797-barhT(volt[k, i]))*np.exp(-k*dt/tauhTf(volt[k, i]))+barhT(volt[k, i])
            
        
            saveL[k]=(1.4135e-5-barmL(volt[k, 4]))*np.exp(-k*dt/taumL(volt[k, 4]))+barmL(volt[k, 4])
            savehL[k]=(0.999998-barhL(volt[k, i]))*np.exp(-k*dt/tauhL(volt[k, i]))+barhL(volt[k, i])
            saveT[k]=(3.2529e-05-barmT(volt[k, 4]))*np.exp(-k*dt/taumT(volt[k, 4]))+barmT(volt[k, 4])
            saveTf[k]=(0.99797-barhT(volt[k, 4]))*np.exp(-k*dt/tauhTf(volt[k, 4]))+barhT(volt[k, 4])
            
            if i > 0 and i < m-1:
                if np.isnan(Ca[k, i]):
                    print('!!! NaN value !!')
                    return None
                
                Ca[k + 1, i] = r*Ca[k, i-1] + (1 - 2*r)*Ca[k, i] + r*Ca[k, i+1] + dCa_dt(Ca[k, i], Cer[k, i], h[k, i], s[k, i], w[k, i], Na[k, i], K[k, i], Q1[k], Q2[k], Q3[k], Q4[k],O[k], On[k],IP3[k],volt[k, i],mL,mT,hTf,hL) * dt
                Na[k + 1, i] = r*Na[k, i-1] + (1 - 2*r)*Na[k, i] + r*Na[k, i+1] + dNa(Ca[k, i],Cer[k, i], K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k]) * dt
                K[k + 1, i]  = r*K[k, i-1]  + (1 - 2*r)*K[k, i]  + r*K[k, i+1] + dK(Ca[k, i],K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k])*dt
            
            Cer[k + 1, i]   = Cer[k, i] + (-gamma * fe * (alpha * (Jip3(Ca[k, i], Cer[k, i],  h[k, i],IP3[k]) + Jleak(Ca[k, i], Cer[k, i]) + Jryr(Ca[k, i], Cer[k, i], w[k, i])) - beta * Jserca(Ca[k, i]))) * dt
            h[k + 1, i]     = h[k, i] + ((hinf(Ca[k, i],IP3[k]) - h[k, i]) / (1 / (a2 * ((d2 * (IP3[k] + d1) / (IP3[k] + d3)) + Ca[k, i]))) + eta_u[k, i]) * dt
            w[k + 1, i]     = w[k, i] + ((winf(Ca[k, i]) -w[k, i]) / (winf(Ca[k, i]) / kc)) * dt

            volt[k + 1, i]= volt[k, i] + fvolt(Ca[k, i], Cer[k,i],K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],mT,hTf,mL,hL,Ko[k])*dt

            
            eta_u[k + 1, i] = eta_u[k, i] + (-eta_u[k, i]/tau_c) * dt + np.sqrt(2*D/tau_c) * noise_term[k, i]

            #print(f'ampa/dnma markovchain {time3-time2}')
            #print('#####')
     
        #print(f'syst time t {time32-time3}')

    return Ca, volt , K, Na, Cer, h, w, C2a , O, D1a, D2a, C0n, C1n, C2n, On, D2n,Jl(Ca.T[4],saveL,savehL,volt.T[4]),saveA

m=8
tmax=300

Ca=0.15523
Cer=29.2946
h=0.661
s=0.89609
w=0.6605
x=0.14634
Na=12.1021
K=117.07227
volt=-67.6531
eta_u=0.0


D1=0#0.011#0.
D2=0#0.0248#0.
D3=0#0.00224#0.
D4=0#1.0456*10**(-5)#0.
C1=1#0.426#1.
C2=0#0.5033#0.
C3=0#8.6269*10**(-6)#0.
C4=0#2.9274*10**(-6)#0.
Q1=0#0.03243#0.
Q2=0#0.000149#0.
Q3=0#9.06487*10**(-6)#0.
Q4=0#1.1157*10**(-5)#0.


C0a=1
C1a=0
C2a=0
O=0
D1a=0
D2a=0

C0n=1
C1n=0
C2n=0
On=0
D2n=0


aa=time.time()
ICs = np.array([Ca, Cer, h, s, w, x, Na , K, eta_u, D1, D2, D3, D4, C1, C2, C3, C4, Q1, Q2, Q3, Q4, C0a, C1a, C2a, O, D1a, D2a, C0n, C1n, C2n, On, D2n,volt])
#temp=SPDE_solver(ICs,m=m,tmax=tmax,ATP=0*10**(-3),G=0*10**(-3),ATP_time=240,Glut_time=240)   
bb=time.time()
print('Computation time:',bb-aa)
#Ca=temp[0].T

def multsimu(temp):

    Ca=temp[0].T[4]
    
    t=np.linspace(0,tmax,len(Ca))
    plt.plot(t,Ca)
    plt.xlabel("time t in $s$")
    # plt.xlim(320,500)
    plt.ylabel("Ca in $\mu Mol$ ")
    plt.show()
    
    
    K=temp[2].T[4]
    
    Jp2x7_K=temp[-2].T
    Na=temp[3].T[4]
    Jnak=temp[-1].T
    
    plt.plot(t,Jnak)
    plt.xlabel("time t")
    plt.ylabel("Glut")
    plt.show()
    
    plt.plot(t,Jp2x7_K)
    plt.xlabel("time t")
    plt.ylabel("Jl")    
    plt.show()
    
    plt.plot(t,K)
    plt.xlabel("time t")
    plt.ylabel("$K$ ")
    plt.show()
    
    plt.plot(t,Na)
    plt.xlabel("time t")
    plt.ylabel("$Na$ ")
    plt.show()
    
    V=temp[1].T[4]
    t=np.linspace(0,tmax,len(V))
    plt.plot(t,V)
    plt.xlabel("time t")
    plt.ylabel("$Voltage(in mV)$ ")
    plt.plot()
    plt.show()
    
    plt.plot()
    plt.show()
   
#multsimu(temp)
    
Catot=np.zeros(1)
for i in range(50):
    
    aa=time.time()
    temp=SPDE_solver(ICs,m=m,tmax=tmax,ATP=0*10**(-3),G=0*10**(-3),ATP_time=240,Glut_time=240)   
    bb=time.time()
    if temp==None:
             print('yes')
             pass
    else:
        print('Computation time:',bb-aa)
        Ca=temp[0].T
        
        Catot=np.concatenate([Catot,Ca[4]])

Catot=stats.zscore(Catot)   
plt.hist(Catot[1:],density=True,bins=100)
plt.title('Distribution of z-scored Calcium signals (WT)')
plt.xlim(-2.5,2.5)
plt.plot()

# countot=np.zeros(50)
# for j in range(50):
#     if j%10==0:
#         print(j)
#     temp=SPDE_solver(ICs,m=m,tmax=tmax,ATP=0*10**(-3),G=0*10**(-3),ATP_time=240,Glut_time=240)   
#     if temp==None:
#         print('yes')
#         pass
#     else:
#         Ca=temp[0].T[4]
#         Ca=Ca[3000:]
#         tempo=np.zeros(len(Ca))
#         count=0
#         plt.plot(Ca)
#         plt.show()
#         Catot=np.concatenate([Catot,Ca])
#         for i in range(len(Ca)):
#               tempo[i]=max(Ca[i],np.mean(Ca)+np.std(Ca))
#               if tempo[i-1]==np.mean(Ca)+np.std(Ca) and tempo[i]==Ca[i]:
#                   count+=1
#         countot[j]=count

# plt.hist(Catot[1:],density=True,bins=100)
# plt.title('Distribution of z-scored Calcium signals with rdn ATP')
# plt.xlim(-1,1)
# plt.show()


# plt.boxplot(countot[countot!=0]*0.2)
# plt.title('number of spikes per min for 50 simulations')
# plt.show()






#######################
# name='01'

# df = pd.read_excel (f'OPC Ca Imaging K+ {name}.xlsx')
# data=df.to_numpy().T

# t=data[0]


# def normalizedata(data):
#     for i in range(np.shape(data)[0]):
#         data[i]=(data[i]-min(data[i]))/max(data[i]-min(data[i]))
#     return data


 
# plt.plot(np.linspace(0,382,len(data[0])),data[30])
# plt.xlabel('time in s')
# plt.title('dataset 02')
# plt.show()

# def fitting(simu,datas,resp):
#     RMSE=np.zeros(len(datas.T[0]))
#     end=len(simu)
#     pas=int(len(simu)/len(data[0][43:int(end*180/360)]))
#     end=len(simu)-resp/100
#     jtemp=0
#     simu2=np.zeros(len(data[0][43:int(end*180/360)]))
#     for j in range(len(data[0][43:int(end*180/360)])):
#         jtemp=jtemp+pas
#         simu2[j]=simu[jtemp]
#     simu2=np.array(simu2)
#     for i in range(len(datas.T[0])):
#         print(np.sqrt(sum((simu2-data[i][43:int(end*180/360)])**2)/len(data[i][43:int(end*180/360)])))
#         RMSE[i]=np.sqrt(sum((simu2-data[i][43:int(end*180/360)])**2)/len(data[i][43:int(end*180/360)]))
#     print(np.where(min(RMSE)==RMSE)[0][0])
#     plt.plot(data[np.where(min(RMSE)==RMSE)[0][0]][43:int(end*180/360)])
#     plt.plot(simu2)
#     plt.show()
   
    
# i=0
# resp=0
# Ca=temp[0].T[4]
# while resp<1:
#     i=i+1
#     if Ca[i]>1.2 and Ca[i+1000]>1.2 and i>5000:
#         resp=i
#     if i>len(Ca):
#         i=len(Ca)

# Ca=(Ca[resp:]-min(Ca[resp:]))/max(Ca[resp:]-min(Ca[resp:]))
# fitting(Ca,data,resp)
        
