import numpy as np
from numba import jit
import time

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
              v_ip3 = 0.88,
              ip3 = 0.2,
              v_pmca = 0.6,
              vncx = 1.4,
              Vmean=-60,
              eta = 0.3,#0.483,
              ATP=0*10**(-3), #ATP in M if rdm ATP==True
              ATP_start=0, #start time of the ATP signal
              ATP_time=0, #in s duration of the ATP signal
              Glut_start=0, #start time of the glutamate signal
              G=0*10**(-3), #glutamate in M if rdmGlut==True 
              Glut_time=0, #duration of the glutamate signal
              rdmATP=False, #if True, add a random ATP signal
              rdmGlut=False,  #if True, add a random Glutamate signal
              gL=0.8, #L-type 
              gT=0.5, #T-type
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
    dncx = 0.25
    kmcao = 1.3
    kmnao = 97.63
    kmnai = 12.3
    kmcai = 0.0026
    kn = 0.5
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
    k2 = 1260 #40000/31.75 #31.6 ???
    k3 = 2.4
    k4 = 1575 #50000/31.75
    k5 = 1.58
    k6 = 221 #7000/31.75
    L1 = 0.0001
    L2 = 0.004
    L3 = 0.05
    H1 = 0.001
    H2 = 0.01
    Hat2 = 0.1#0.01
    H4 = 0.6
    H3=0
    barg_p2x7 = 7.5*3 # nS

    
    T = 300
    F = 96485.3321
    Vosteo=6.5
    Kout=4#4
    Caout=2.5#2.5
    cao=Caout
    Naout=140#140
    nao=Naout
    
    #K fct

    gnak=22.6 #pA
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
    
    gAMPA= 1.5#4.0 #nS
    
    
    #NMDA
    Rbn=5*10**6/(60*2) #60
    Run=12.9/(60*2) 
    
    R0n=46.5 /2
    Rcn=73.8/(10*2)
    
    Mgo=1 #1 to 2 mM
    
    Rdn=8.4/(600*2)
    Rrn=6.8/(60*2)
    
    gNMDA=5.6/(4*2) #5.6/2 #/2 #nS
    
    
    #IP3
    alphatp=0.03
    katp=1
    gammatpglut= 0.01#0.01
    alphaglut=0.03
    kglut=1

    # Jk
    
    gk=11.58 #nS
    
    #Jleak
    
    gleak=1 #nS
    Vl=30 #mV
    
    #NCX
    
    gncx=0.04 #nS 0.02

    Cm=1 #µF/cm^2
    sur=5*10**(3) # 5-50 µm^2
    
    gleakK=2
    Vlk=70
    
    # Functions
    
    
    def Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt):
        return barg_p2x7*((Q1+Q2)+(Q3+Q4))*volt
    
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
        return -f_Ca*alpha*Iampa(O, Ca, Na, K,volt)*10**(6) #-6 is for microM
    
    def Jnmda_Ca(On, Ca, Na, K,volt):
        alpha=1/(2*Vosteo*F)
        return -f_Ca*alpha*Inmda(On, Ca, Na, K,volt)*10**(6) #-6 is for microM
    

    def Jampa_Na(O, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_Na*alpha*Iampa(O, Ca, Na, K,volt)*10**3 #-3 is for mM
    
    def Jnmda_Na(On, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_Na*alpha*Inmda(On, Ca, Na, K,volt)*10**3 #-3 is for mM
    
    
    def Jampa_K(O, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)

        return -f_K*alpha*Iampa(O, Ca, Na, K,volt)*10**3 #-3 is for mM
    
    def Jnmda_K(On, Ca, Na, K,volt):
        alpha=1/(Vosteo*F)
        return -f_K*alpha*Inmda(On, Ca, Na, K,volt)*10**3 #-3 is for mM
    
    #######
    
    def Isoc(Ca,Cer,volt):
        Eca=R*T/(2*F)*np.log(Caout/(Ca*10**-3))*10**3

        return gsocc*np.tanh(Cer-ksoc)/2*(volt-Eca)

    def Jsoc(Ca,Cer,volt):
        Eca=R*T/(2*F)*np.log(Caout/(Ca*10**-3))*10**3
        alpha=1/(2*Vosteo*F)
        barvs=1#2*Vs/((-67-Eca)*alpha)
        return alpha*barvs*Isoc(Ca,Cer, volt)
        
    def Inak(Ca,Na,K,volt,Kout):
        return gnak*Kout/(Kout+Knakko)*Na**1.5/(Na**1.5+Knakna**1.5)*(volt+135.1)/(volt+300)
    
    def Jnak(Ca,Na,K,volt,Kout):
        alpha=1/(Vosteo*F)
        return -alpha*Inak(Ca,Na,K,volt,Kout)*10**3

    
    def hinf(Ca,IP3): 
        return (d2 * (IP3 + d1) / (IP3 + d3)) / ((d2 * (IP3 + d1) / (IP3 + d3)) + Ca)
    
    def ninf(Ca): 
        return Ca / (Ca + d5)
    
    def Kx(Ca, Na): 
        return kmcao * (Na**3) + (kmnao**3) * Ca + (kmnai**3) * cao * (1 + Ca / kmcai) + kmcai * (Naout**3) * (1 + (Na**3) / (kmnai**3)) + (Na**3) * cao + (Naout**3) * Ca
    
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
    
    def ninfi(Ca): 
        return 1 / (1 + (kn / Ca)**1.5)
    
    def Incx(Ca, Na,volt): 
        VFRT = volt*10**(-3) * F / (R * temp)
        nin=ninfi(Ca)
        Ca=Ca*10**(-3)
        
        if mode==1:
            incx=nin * gncx * ( np.exp(eta * VFRT)*(Na**3)*cao ) / (1 + dncx * ((Na**3) * cao + (Naout**3) * Ca)) 

            return incx

        if mode==2:
            incx=nin * gncx * ( - np.exp((eta - 1)*VFRT)*Naout**3*Ca) / (1 + dncx * ((Na**3) * cao + (Naout**3) * Ca)) 
            return incx
        else:
            incx=nin * gncx * ( np.exp(eta * VFRT)*(Na**3)*cao - np.exp((eta - 1)*VFRT)*Naout**3*Ca) / (1 + dncx * ((Na**3) * cao + (Naout**3) * Ca)) 

            return incx
        
    def Jncxca(Ca, Na, volt):
        alphaca=1/(2*Vosteo*F)
        return -alphaca*Incx(Ca, Na, volt)*10**6
    
    def Jncxna(Ca, Na, volt):

        alphana=1/(Vosteo*F)
        return alphana*Incx(Ca, Na, volt)*10**3
    


    def Jpmca(Ca): 
        return (v_pmca * (Ca**2) / ((Ca**2) + (k_pmca**2)))
        
    def Ik(K,volt,Kout):
        Ek=R*T/(F)*np.log(Kout/K)*10**3
        return gk*np.sqrt(Kout)*(volt-Ek) 
    
    def Jk(K, volt,Kout):
        alpha=1/(Vosteo*F)
        return -alpha*Ik(K, volt,Kout)*10**3
    
    def Ileak(K,volt,Kout):
        Ek=R*T/(F)*np.log(Kout/K)*10**3

        return gleak*(volt-Ek)
    
    
    def Ileak_K(K,volt,Kout):
        Ek=R*T/(F)*np.log(Kout/K)*10**3

        return gleakK*(volt-Vlk)
    
    def Jleak_K(K,volt,Kout):
        alpha=1/(Vosteo*F)
        return -alpha*Ileak_K(K,volt,Kout)*10**3
    
    def Jrk(K):
        return (Krest-K)/tetak
    
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
    
    def Jr(Na):
        return (Narest-Na)/tetana
    

    ####scheme###
    def dCa_dt(Ca, Cer, h, s, w, Na, K, Q1, Q2, Q3, Q4,O,On,IP3,volt,mL,mT,hTf,hL):
        return fi * (Jip3(Ca, Cer, h,IP3)+Jleak(Ca, Cer)+Jryr(Ca, Cer, w)-Jserca(Ca)+ Jp2x7_Ca(Ca,Na,K, Q1, Q2, Q3, Q4, volt)+Jsoc(Ca,Cer,volt)+ Jncxca(Ca,Na ,volt)- Jpmca(Ca)+Jnmda_Ca(On, Ca, Na, K,volt)+Jampa_Ca(O, Ca, Na, K,volt)+Jt(Ca,mT,hTf,volt)+Jl(Ca,mL,hL,volt))#
    
    def dNa(Ca, Cer ,K,Na,volt,O,On,Q1,Q2,Q3,Q4,Kout):
        return 3*Jncxna(Ca,Na,volt)-3*Jnak(Ca, Na, K,volt,Kout)+Jp2x7_Na(Ca, Na, K, Q1, Q2, Q3, Q4,volt)+Jr(Na)+Jnmda_Na(On, Ca, Na, K,volt)+Jampa_Na(O, Ca, Na, K,volt)
    
    def dK(Ca,K,Na,volt,O,On,Q1,Q2,Q3,Q4,Kout):
        return 2*Jnak(Ca, Na, K,volt,Kout)-Jp2x7_K(Ca, Na, K, Q1, Q2, Q3, Q4,volt)-Jnmda_K(On, Ca, Na, K,volt)-Jampa_K(O, Ca, Na, K,volt)+Jk(K, volt,Kout)+Jleak_K(K,volt,Kout)+Jrk(K)
    
    def fvolt(Ca,Cer,K,Na,volt,O,On,Q1,Q2,Q3,Q4,mT,hTf,mL,hL,Kout):
        return (-Iampa(O, Ca, Na, K,volt)-Inmda(On, Ca, Na, K,volt)-Ip2x7(Ca, Na, K, Q1, Q2, Q3, Q4,volt)-Isoc(Ca,Cer,volt)-Ik(K,volt,Kout)-Inak(Ca,Na,K,volt,Kout)-Incx(Ca, Na, volt)-Il(Ca,mL,hL,volt)-It(Ca,mT,hTf,volt)-Ileak_K(K,volt,Kout)-Ileak(K,volt,Kout))/(Cm*sur)*100
    
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
        if rdmATP==True:
            
            if np.random.binomial(1, 0.001)==1 and k>5000:
                    trace=k
                    rdn=np.abs(np.random.randn(1)*10**-3)[0]
                    rdn=rdn/2
            if k<trace+10:
                    A=rdn
        #saveA[k]=A
        if k<Glut_start*100:
            Glut=0
        elif k>Glut_start*100+Glut_time*100:
            Glut=0
        else:
            Glut=G
        if rdmGlut==True :
            #print("rdm Glut")
            if np.random.binomial(1, 0.001)==1 and k>5000:
                        trace=k
                        rdn=np.abs(np.random.randn(1)*10**-3)[0]
                        rdn=rdn/2
            if k<trace+10:
                        Glut=rdn
        saveA[k]=Glut

        #P2x7
        tot=1
        totn=1
        tota=1
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
        
    
    for k in range(n-1):#range(n-1):
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
            
        
            saveL[k]=(1.2135e-5-barmL(volt[k, i]))*np.exp(-k*dt/taumL(volt[k, i]))+barmL(volt[k, i])
            savehL[k]=(0.999998-barhL(volt[k, i]))*np.exp(-k*dt/tauhL(volt[k, i]))+barhL(volt[k, i])
            saveT[k]=(3.2529e-05-barmT(volt[k, 4]))*np.exp(-k*dt/taumT(volt[k, 4]))+barmT(volt[k, 4])
            saveTf[k]=(0.99797-barhT(volt[k, 4]))*np.exp(-k*dt/tauhTf(volt[k, 4]))+barhT(volt[k, 4])
            if i > 0 and i < m-1:                
                Ca[k + 1, i] = r*Ca[k, i-1] + (1 - 2*r)*Ca[k, i] + r*Ca[k, i+1] + dCa_dt(Ca[k, i], Cer[k, i], h[k, i], s[k, i], w[k, i], Na[k, i], K[k, i], Q1[k], Q2[k], Q3[k], Q4[k],O[k], On[k],IP3[k],volt[k, i],mL,mT,hTf,hL) * dt
                Na[k + 1, i] = r*Na[k, i-1] + (1 - 2*r)*Na[k, i] + r*Na[k, i+1] + dNa(Ca[k, i],Cer[k, i], K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k]) * dt
                K[k + 1, i]  = r*K[k, i-1]  + (1 - 2*r)*K[k, i]  + r*K[k, i+1] + dK(Ca[k, i],K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],Ko[k])*dt
            
            Cer[k + 1, i]   = Cer[k, i] + -gamma * fe *  (Jip3(Ca[k, i], Cer[k, i],  h[k, i],IP3[k]) + Jleak(Ca[k, i], Cer[k, i]) + Jryr(Ca[k, i], Cer[k, i], w[k, i]) - Jserca(Ca[k, i])) * dt
            h[k + 1, i]     = h[k, i] + ((hinf(Ca[k, i],IP3[k]) - h[k, i]) / (1 / (a2 * ((d2 * (IP3[k] + d1) / (IP3[k] + d3)) + Ca[k, i]))) + eta_u[k, i]) * dt
            w[k + 1, i]     = w[k, i] + ((winf(Ca[k, i]) -w[k, i]) / (winf(Ca[k, i]) / kc)) * dt

            volt[k + 1, i]= volt[k, i] + fvolt(Ca[k, i], Cer[k,i],K[k, i],Na[k, i],volt[k, i],O[k],On[k],Q1[k],Q2[k],Q3[k],Q4[k],mT,hTf,mL,hL,Ko[k])*dt

            
            eta_u[k + 1, i] = eta_u[k, i] + (-eta_u[k, i]/tau_c) * dt + np.sqrt(2*D/tau_c) * noise_term[k, i]

     
    if np.isnan(Ca[4, 1]):
        print('!!! NaN value !!')
        Ca=np.zeros((8,2))
    return Ca, volt , K, Na, Cer, h, w, x , saveL, hL, mT, hTf, D1a, D2a, C0n, C1n, C2n, On, D2n,Il(Ca.T[4],saveL,savehL,volt.T[4]),saveA

m=8
tmax=350

Ca=0.12050920535046733
Cer=26.53200798707888
h=0.70881
s=0.0
w=0.8183
x=0.0#0.14634
Na=12.09739031404967
K=118.626111702933227
volt=-71.83279621069488
eta_u=0.0

D1=0.
D2=0.
D3=0
D4=0
C1=1
C2=0
C3=0
C4=0
Q1=0.
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
temp=SPDE_solver(ICs,m=m,tmax=tmax)   
bb=time.time()
print('Computation time:',bb-aa)
