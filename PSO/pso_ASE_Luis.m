                 %SOO using PSO - Power Control

clear all
close all

clc

%---- Number of Users  & ws  & c1, c2 Parameters---------
%K=2;
%K=4;  ws =  5E-24;  ws2= 0; c1 = 1.8; c2 = 2.0; ; qm=1
%K=4; ws = 5E-24;  ws2= 0, c1 = 1.8; c2 = 2.0;  qm=.78


%K=9;  ws = 20E-24; ws2=0; c1 = 1.0; c2 = 2.0; qm=0.83
%
%K=8;  ws = 20E-20; ws2=0; c1 = 1.8; c2 = 2.0; qm=0.83
%K=10; ws = 35E-19;  ws2= 0; c1 = 1.8; c2 = 2.0; ; qm=.83

K=16; ws = 60E-19;  ws2= 0, c1 = 1.8; c2 = 2.0; ; qm=1.2
%K=12; ws = 60E-19;  ws2= 0, c1 = 1.8; c2 = 2.0; ; qm=1.2
%K=12; ws = 55E-24;  ws2= 0, c1 = 1.8; c2 = 2.0; ; qm=0.78
%K=15; ws = 5E-22;  ws2= 0, c1 = 2.0; c2 = 2.0;  qm=.78
%............................................................
%K=31; ws = 5E-22;  ws2= 0, c1 = 1.8; c2 = 2;  qm=.78





%q=31;
%q=2*4^2;
q=256;
%q=100;


SIR_max_dB=20;
SIR_max=10^(SIR_max_dB/10);


%sigma_cc2=0.29;
%sigma_cc2=0.29;
sigma_cc2=1e-2;

CIR_target=SIR_max*(sigma_cc2/q^2)
%CIR_target=SIR_max*(200e9/10e9)

SNR_target = CIR_target;


%Power Bounds
Pmax=1e-3; %[W]
% Pmax=2*((K/5).^2)*1E-3; %[W]

Pmin=0;
%Pmin = 1E-6;

%User Class definition
n_class = 3;
if(K==2)
    Uclass(1)=1;
    Uclass(2)=2;
elseif(K==5)
    Uclass(1:3)=1;
    Uclass(4)=2;
    Uclass(5)=3;
elseif(K==8)
    Uclass(1:4)=1;
    Uclass(5:7)=2;
    Uclass(8)=3;
elseif(K==12)
    Uclass(1:6)=1;
    Uclass(7:10)=2;
    Uclass(11:12)=3;
elseif(K==16)
    Uclass(1:8)=1;
    Uclass(9:16)=2;
    %Uclass(13:16)=3;
elseif(K==31)
    Uclass(1:16)=1;
    Uclass(17:25)=2;
    Uclass(26:31)=3;
end

Rc = 10e9;
%Target Rate for Each Class
%Rmin =[1/256  1/64  1/16]*Rc;

%Rmin =[1/10    1]*Rc;
Rmin =[1  1  1]*Rc;

%BER_target = [1E-9  5E-11];
BER_target = [5E-10 5E-10 5E-10];

tetaBER_Class = -1.5./log(5.*BER_target); % Eq. 7

SNIRminClass=-(2/3)*log(5*BER_target).*(2^2-1);

%Calculation CIR_target
%Uclass = repmat(1,K,1);10,9
[idx]=find(Uclass==1);


for k=1:n_class
    %Rate Constraint for Foschini
    [idx]=find(Uclass==k);
    Rmin_class(idx) = Rmin(k);
    CIR_target(idx) =  repmat(Rmin(k)  * (SNR_target/Rc), 1, length(idx) );
    Pmin_class(idx) = repmat(Rmin(k) * (Pmin / Rc),1,length(idx));
    tetaBER(idx) = tetaBER_Class(k);
    SNIRmin(idx) = SNIRminClass(k);
    F(idx) = Rc ./ Rmin(k);
end

%  average noise power of the receiver
%Ni= 15e-7%[A^2]


rmin=2;
rmax=50;


%Ltx_i=unifrnd(rmin, rmax, 1,K);%comprimento das fibras tx
Ltx_i= [48.8779
   26.9518
   40.0057
   46.3947
   18.7903
   22.8634
   37.3182
    5.2198
   32.6824
   48.5307
   10.2101
   40.0123
   14.8661
   26.3720
   46.7481
   29.7142]';

%Lrx_i=unifrnd(rmin, rmax, 1, K);%comprimento das fibras rx
Lrx_i=[32.0586
    4.8504
   45.3027
   43.6605
   36.0278
   48.6326
   19.1923
   25.9573
   29.8555
   20.4597
    8.1063
   11.6794
   40.0223
    8.0725
   42.6961
   27.2251]';


BT=1;
      Gamp=BT.*100;
      Ni=BT.*2.*(Gamp-1).*(193e12).*(100e9).*6.63e-34;
%     

Ni
Pn=Ni;
% Fiber optic


%F = ones(1,K);

%F = [1 1 1 1 1 1 1 1 ];

% Fixed stop criterion
It=100


% ----- Partitioned Optical Power Control ----
% random node position in the range [rmin rmax]


lt=length(Ltx_i);
lr=length(Lrx_i);

%.... transmitting-node fiber attenuation plus star coupler loss....
gama_dB=0.2;  %relação de perda excessiva [dB]
%gama_dB=0.4;  %relação de perda excessiva [dB]
gama = 10^(gama_dB/10)
a_star_dB=10*log10(K) - ( log2(K)  * gama_dB);
%a_star = 10^(-a_star_dB/10)
%a_star_dB = 0.3 %dB
a_star = 10^(-28/10)
%...............................................
a_fiber_dB = 0.2 %dB/Km
%a_fiber_dB = 0.9 %dB/Km
a_fiber = 10^(a_fiber_dB/10)  %1/Km
%essa é a equação do alfa
alfa=a_fiber_dB/(10*log10(exp(1)))

%...................................................
%g_t = a_star * exp(-alpha_fiber*Ltx_i)
%g_t = a_star * exp(-alfa*Ltx_i)
%g_t = a_star.*OXC.* exp(-alfa.*Ltx_i).*Gamp.*0.0025

%g_t = a_star.^2 * exp(-alfa.*Ltx_i).*Pr.*Gamp

g_t = a_star.* exp(-alfa.*Ltx_i).*Gamp.*0.0025;
%g_t = a_star.* exp(-alfa.*Ltx_i).*Gamp.*1e-4
%....................................................

% ...... receiving-node fiber attenuation
% g_hat
%g_r = exp(-alpha_fiber*Ltx_i)
g_r = exp(-alfa*Lrx_i)
%....................................................

% interference matrix
H =  repmat(g_t, K,1)./repmat(g_t',1,K) - eye(K);
H_dB = 10*log10(H)

G = eye(K) + H;% G é usada no PSO
Gii = g_t


% ......Optimal Power Control Problem ..........
%By B matrix inversion
%u = (CIR_target *Ni)./(g_t)';
u = Pn  * (CIR_target ./ Gii)';  %  (Pmin./Gii)' % = SNR_min*Pn
B = repmat(CIR_target',1,K) .* (G - eye(K))
%B= CIR_target.*H;
Eig_B=eig(B)




%VER COM O PROFESSOR O QUE ESTÁ ACONTECENDO AQUI.
if(max(Eig_B) >= 1)
    disp('Unfeasable Problem');
    disp('Try other Interference Matrix');
    return
else
    Popt=inv(eye(K)-B)*u %esta é a equação (12) do artigo de
    %                  power control do Tarhuni 2005, utilizado como base
end








%===================================
P_opt_dBm = 10*log10(Popt/1E-3);

%============ Initial population ==========================
Pmax_dBm=10
Pmax=10^(Pmax_dBm/10)*1E-3
Pmin = Pmax*1E-8

% Pmax_dBm=-5
% Pmax=10^(Pmax_dBm/10)*1E-3
% Pmin_dBm=-20
% Pmin = 10^(Pmin_dBm/10)*1E-3
%=================================================

%===================================================
% -------- Power Vector initialization ------------
% Uniform random distribuition in the range [Pmin Pmax]

% P = uniform distributed random variable with range between Pmin and Pmax
%P = 10.^(unifrnd(Prange_dB(1), Prange_dB(2),1,K)/10);

% -------- CIR Vector initialization ------------
% CIR = R*SNR_min/Rc  [RC = Chip Rate = Taxa de Chip]
%Er=0.4;
Er=0.0;
%--------- Error gain/lose random distribution
        ED = 1+unifrnd(-Er,Er,K,K);

        %----- Channel Error-----------
        % Interference matrix with error estimation
        %  %g_t with error estimation
        g_te = diag(ED)'.* g_t;
        Gii=g_te;

         
%========== PSO Parameters ================================================
%Population Size
M = ceil(K/9)*K+2;
%M = K+2;
% Initial Inercia Weight
w_i=1;

%Main Loop - N = number of iterations, n = current iteration
%N=700 + ceil((K/5)^sqrt(K/4))*100;
N=700 + ceil((K/5)^sqrt(K/4))*100;

% Final  Inercia Weight
% slope
s=-5E-4  %s=-15E-4
w_f = w_i + s*N


% Vmax 
Vmax = 0.2*(Pmax-Pmin);
%Vmax = .01*(Pmax-Pmin);
Vmin = -Vmax;

% ----the best initial position---
%Pop=[unifrnd(Pmin, Pmax,K, M)];

exp_Pop=[unifrnd(log10(Pmin), log10(Pmax), K, M)];
Pop=10.^exp_Pop;


%========== Trials -> Ensaios ========================================================
%TR = 100;
TR = 3;
%TR = 50;

teste = zeros(K,TR);

for tr=1:TR
    %Initial Population
    P = unifrnd(0.1*Pmax,Pmax,K,M)
    
    % % Delay---------------------------------------------------------
    
    %Initial local bests
    Pibest = P;
    
    %Initial Global Best
    Pgbest(1:K,1) = Pmax;
    
    %Initial Fth Matrix
    Fth = zeros(K,M);
    
    %Implementate constrainsts
    
    
   SNIR = zeros(K,M);
   
    for m=1:M
        SNIR(:,m) = F .* (P(:,m)'  ./  (sum(repmat(P(:,m)' ,K,1) .* G,2) - P(:,m) + Pn./Gii')');
    end
    
    %Fth Matrix Update
    Fth = SNIR > repmat(SNR_target,K,M);
   
    
    %Cost Function Evaluation
    %jP = f(P,Fth,K,Gii,ws,Pmax,F,Pn,G,SNR_target,1);
    jP = fitness(P,Fth,K,Pmax);
    jPibest = jP;
    jPgbest = 0;
    
    % Cost Function Values
    %JPg = jPgbest;
    
    %Initial Speed
    v = zeros(K,M);
    
    %Chart Variables
    bestaux = zeros(K,N);
    paux = zeros(K,N);
    
    for n=1:N
        
        %Implementate Constrainsts
        
           % % Delay---------------------------------------------------------

    for m=1:M
            SNIR(:,m) = F .* (P(:,m)'  ./  (sum(repmat(P(:,m)' ,K,1) .* G,2) - P(:,m) + Pn./Gii')');
  
    end

        %Fth Matrix Update
        Fth = SNIR > repmat(SNR_target,K,M);
        
   %%%% Delay        
        jP = fitness(P,Fth,K, Pmax);
        
        %Evaluate Local Best / ACHO QUE AQUI CALCULA O MELHOR LOCAL
        [rw] = find(jP > jPibest);
        Pibest(:,rw) = P(:,rw);
        jPibest(rw) = jP(rw);
        
        %Evaluate Global Best / ACHO QUE AQUI CALCULA O MELHOR GLOBAL
        [MaxJ,idx] = sort(jP);
        if MaxJ(end) > jPgbest
            jPgbest = MaxJ(end);
            Pgbest = P(:,idx(end));
        end
        
        %Aux Variable to Plot Charts, variáveis do gráfico 
        sniraux(n,:) = F .* (Pgbest'  ./  (sum(repmat(Pgbest' ,K,1) .* G,2) - Pgbest + Pn./Gii')');
        bestaux(:,n) = Pgbest;
        paux(:,n) = P(:,1);
        pibestaux(:,n) = Pibest(:,1);
        praux(:,n,tr) = std(P .* repmat(Gii',1,M),0,1);
        
%         ACHO QUE AQUI CALCULA A VELOCIDADE 
        w_adp = 0.6;
        v = w_adp * v + c1 .* rand(K,M) .* (Pibest - P) + c2 .* rand(K,M) .* (repmat(Pgbest,1,M) - P);
        %Speed Bounds / LIMITES DE VELOCIDADE
        [rw,col] = find(v > repmat(Vmax,K,M));
        for i=1:length(rw)
            v(rw(i),col(i)) = Vmax;
        end
        [rw,col] = find(v < repmat(Vmin,K,M));
        for i=1:length(rw)
            v(rw(i),col(i)) = Vmin;
        end
        %Population update
        P = P + v;
        
        %Power Bounds
        P = P .* (P >= Pmin) + Pmin .* (P < Pmin);
        P = P .* (P <= Pmax) + Pmax .* (P > Pmax);
        
        
        % Cost Function Values
        JPg(tr,n) = jPgbest;

        
    end
    
    teste(:,tr) = Pgbest;
    
end


SNR=10*log10(SNIR(:,m).*(q^2/sigma_cc2))































for tr=2:TR
    diferenca(:,tr-1) = abs(teste(:,tr) - teste(:,tr-1));
end

for tr=1:TR
    NSE(tr) = norm(teste(:,tr) - Popt) .^ 2 ./ norm(Popt) .^ 2;
end

%Testing
Pr = bestaux .* repmat(Gii',1,N);
q2 = (ws./std(Pr,0,1));
q1 = mean((1 - (bestaux ./ Pmax)),1);
%q1_ = mean((1 - (teste ./ Pmax)),1);
q1 ./ q2

Nst = 1:N;

% FIGURA 1 - Posicionamento das Mobile Stations
scrsz = get(0,'ScreenSize');
%figure('Position',[1 scrsz(4)/2 scrsz(3)/2.1 scrsz(4)/2.5])
figure('Position',scrsz)

subplot(232)
plot(Nst,q2./q1)
ylabel('Q_2 / Q_1 (cost function ratio)')
%hold on
%plot(Nst,repmat(teste,N,1),':r')

% figure(98)
% plot(1:TR,teste


%========== Ploting Pgbest / Iteration ================
%figure(2)

subplot(233)
semilogy(Nst,bestaux(1,:),'b-')
hold on
semilogy(Nst,repmat(Popt(1),1,N),'k--')
semilogy(Nst,bestaux,'b-')
semilogy(Nst,repmat(Popt,1,N),'k--')
legend('P_{g}^{best}','P_{Opt}')
xlabel('Iterations, N')
ylabel('Allocated Power [W]')
title(['Allocated Power, K = ',num2str(K,3),', Swarm Population, M = ',num2str(M,5),' particles  ', '\phi_1 = ',num2str(c1),'  \phi_2 = ',num2str(c2)])
%axis([0 N 9.8E-6 1E-4])
%............................
% subplot(212)
% semilogy(1:TR-1,mean(diferenca),'r.-')
% xlabel('Trials')
% ylabel('Power Solution Difference Avarage [W]')
% title(['Absolute Power Solution Difference Between Trial n and n-1'])

%===================================================
%Ploting Standard Deviation
%figure(30)
% .... ScreenSize is a four-element vector: [left, bottom, width, height]:
% scrsz = get(0,'ScreenSize');
% figure('Position',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2.5])
% subplot(234)
% plot(Nst,mean(praux,3))
% %title('Avarage Recieved Power Standard Deviation')
% xlabel('Iterations, N')
% ylabel('Average Rx Power Std, \sigma_{rp}')

subplot(234)
plot(Nst,JPg)
hold on
plot(Nst,JPg(TR,:),'k-', 'LineWidth',2)
%title('Avarage Recieved Power Standard Deviation')
xlabel('Iterations, N')
ylabel('J(p), Cost Function')


%===================================================
%Ploting NSE Average
% figure(4)
%.... ScreenSize is a four-element vector: [left, bottom, width, height]:
% scrsz = get(0,'ScreenSize');
%figure('Position',[scrsz(3)/2 40 scrsz(3)/2 scrsz(4)/2.5])
mean(NSE)
subplot(235)
semilogy(1:TR,NSE)
%stem(1:TR,NSE)
hold on
semilogy(1:TR,repmat(mean(NSE),1,TR),'r:')
title('NSE for Each Trial')
xlabel('Trials, TR')
ylabel('NSE')

%SNIR Evolution
%figure(7)
subplot(236)
%figure('Position',[1 40 scrsz(3)/2 scrsz(4)/2.5])
plot(10*log10((q^2/0.29).*sniraux))
hold on
plot(repmat(10*log10((q^2/0.29).*SNR_target),1,N),'k--','LineWidth',2.5)
ylabel('CIR')

%------------------------------------------------------------------------
% subplot(2,2,1:2)%233
% semilogy(Nst,bestaux(1,:),'b-')
% hold on
% semilogy(Nst,repmat(Popt(1),1,N),'k--')
% %............................
% semilogy(Nst,bestaux,'b-')
% semilogy(Nst,repmat(Popt,1,N),'k--')
% legend('P_{g}^{best}','P_{Opt}',1)
% xlabel('Iterações, N')
% ylabel('Potência Alocada [W]')
% title(['Potência Alocada, K = ',num2str(K,3),', População do Bando, M = ',num2str(M,5),' partículas  ', 'c_1 = ',num2str(c1),'  c_2 = ',num2str(c2)])
% 
% subplot(2,2,3)%235
% semilogy(1:TR,NSE)
% hold on
% semilogy(1:TR,repmat(mean(NSE),1,TR),'r:')
% title('NSE para cada teste')
% xlabel('Testes, TR')
% ylabel('NSE')
% 
% %SNIR Evolution
% %figure(7)
% subplot(2,2,4)
% %figure('Position',[1 40 scrsz(3)/2 scrsz(4)/2.5])
% plot(10*log10(sniraux))
% hold on
% plot(repmat(10*log10(SNR_target),1,N),'k--','LineWidth',2.5)
% ylabel('CIR')




figure(2)
semilogy(Nst,bestaux(1,:))
hold on
semilogy(Nst,bestaux)
semilogy(Nst,repmat(Popt,1,N),'k--')
Nstt=transpose(bestaux);
save power_pso Nstt -ASCII
Nstt=transpose(repmat(Popt,1,N));
save power_pso_op Nstt -ASCII
xlabel('Iterações')
ylabel('Potência transmitida (W)')
title(['Allocated Power, K = ',num2str(K,3),', Swarm Population, M = ',num2str(M,5),' particles  ', '\phi_1 = ',num2str(c1),'  \phi_2 = ',num2str(c2)])


figure(3)
plot(10*log10((q^2/sigma_cc2).*sniraux))
hold on
%plot(10*log10((q^2/sigma_cc2).*sniraux)-7)
plot(repmat(10*log10((sigma_cc2).*SNR_target),1,N),'k--','LineWidth',2.5)
ylabel('CIR')
axis([0 800 5 45])


% sum(Popt)
nnodes=16;
%FI=12 - H = 2
%GE = 17 - H=3;
%PA = 28 - H=4


% figure(4)
% semilogy(Nst,nnodes.*sum(bestaux),'b-')
% %semilogy(Nst,sum(bestaux),'b-')
% hold on
% semilogy(Nst,nnodes.*sum(repmat(Popt,1,N)),'k--')
% %semilogy(Nst,sum(repmat(Popt,1,N)),'k--')
% %semilogy(Nst,repmat(sigma_cc2(1),1,N),'k--')
%............................
Pmax=sum(bestaux)

% Nstt=transpose(Nst);
% save Aite Nstt -ASCII
% 
Nstt=transpose(sum(bestaux));
save Sumreal Nstt -ASCII
% 
Nstt=transpose(sum(repmat(Popt,1,N)));
save Sumop Nstt -ASCII

figure(5)
plot(Nst,sum(repmat(Popt,1,N))./sum(bestaux),'LineWidth',2)
Nstt=transpose(sum(repmat(Popt,1,N))./sum(bestaux));
legend('Taxa de convergência pelo número de iterações')
xlabel('Iterações')
ylabel('Taxa de convergência')
save conver Nstt -ASCII


bitrate=10e9;
%energia=sum(bestaux)./0.4;
%energia=((sum(bestaux)./0.1)./bitrate );
Tc=1.0e-3;
%Tc=1/10e9;
Tb=Tc./16;

%Option 1
%Th=5*Tc;
%Poff=0.19;

%Option 1
 Th=0.125*Tc;
 Poff=0.33;

Pdriver=2e-3;
PCW=1e-3;
%PCW=1.5e-4;
Pvi=1e-3;
%Pvi=1.5e-4;
Lmod=2.5;
Ptrans1=Pdriver+(PCW.*10.^(Lmod/10))+Pvi;
Prec1=1.4e-3;
Ptotala=Ptrans1+Prec1;

%figure(61)
%%semilogy(Nst,(16*Ptotala*Tc))
%%hold on
%%Nstt=transpose((16*Ptotala*Tc));
%%save En1 Nstt -ASCII

figure(4)
semilogy(Nst,16*((sum(bestaux).*10.^(Lmod/10))),'b-','LineWidth',2)
Nstt=transpose(16*((sum(bestaux).*10.^(Lmod/10))));
save Txpso Nstt -ASCII
hold on
semilogy(Nst,16*((sum(repmat(Popt,1,N)).*10.^(Lmod/10))),'k--')
Nstt=transpose(16*((sum(repmat(Popt,1,N)).*10.^(Lmod/10))));
save Txop Nstt -ASCII
hold on
semilogy(Nst,((Pvi.*10.^(Lmod/10))),'r-','LineWidth',2)
Nstt=transpose(((Pvi.*10.^(Lmod/10))));
save Txwop Nstt -ASCII
legend('Mecanismo Proposto','Inversão de Matriz','Sem Mecanismos de Economia')
xlabel('Iterações')
ylabel('Soma da potência transmitida (W)')

Ptrans1=16*Pdriver+(sum(bestaux).*10.^(Lmod/10))+sum(bestaux);
Prec1=16*1.4e-3;
Ptotalc=Ptrans1+Prec1;

% aa=0.1;
% ro=0.1;
% %Pactive=16*((sum(bestaux).*10.^(Lmod/10)));
% Pactive=Ptotalc;
% Psleep=0.33.*Pactive;
% Pconsu=((1+aa).*ro.*Pactive)+((1-((1+aa).*ro)).*Psleep);
% figure(63)
% semilogy(Nst,Pconsu./Pactive)
% hold on
% 
% ro=1.0;
% %Pactive=16*((sum(bestaux).*10.^(Lmod/10)));
% Pactive=Ptotalc;
% Psleep=0.33.*Pactive;
% Pconsu=((1+aa).*ro.*Pactive)+((1-((1+aa).*ro)).*Psleep);
% figure(63)
% semilogy(Nst,Pconsu./Pactive,'r')
% hold on


%ro=0:0.1:1;
ro=1;
aa=0.1;
 Pactive=Ptotalc;
 Psleep=0.33*Pactive;
%Ppef=((1+aa).*ro)+((1-((1+aa).*ro)).*0.33);
Pconsu=((1+aa).*ro.*Pactive)+((1-((1+aa).*ro)).*Psleep);
figure(63)
plot(Nst,Pconsu,'b-*')
hold on

aa=0.1;
%Ppef=((1+aa).*ro)+((1-((1+aa).*ro)).*0.6);
 Pactiveb=16.*Ptotala;
 Psleep=0.33*Pactiveb;
Pconsub=((1+aa).*ro.*Pactiveb)+((1-((1+aa).*ro)).*Psleep);
figure(63)
plot(Nst,Pconsub,'r')
hold on
%axis([0 1 0 1])


% figure(61)
% semilogy(Nst,(16*Ptotala*Tc),'b','LineWidth',2)
% hold on
% figure(61)
%  semilogy(Nst,(Ptotalc*Tc),'g','LineWidth',2)
%  hold on
%  Nstt=transpose((Ptotalc*Tc));
% save En2 Nstt -ASCII
 enf= 16*Ptotala;
 energia_sleep=((((enf)).*((Tb)+(Th))))+ ((Poff*((enf))).*((Tc-(Tb)-(Th))));
% figure(61)
% semilogy(Nst,(energia_sleep),'k','LineWidth',2)
% hold on
%  Nstt=transpose((energia_sleep));
% save En3 Nstt -ASCII
 enf= Ptotalc;
 energia_sleep2=((((enf)).*((Tb)+(Th))))+ ((Poff*((enf))).*((Tc-(Tb)-(Th))));
% figure(61)
% semilogy(Nst,(energia_sleep2),'r','LineWidth',2)
sem_eco=(16*Ptotala*Tc);
hiber=(Ptotalc*Tc);


figure(61)
semilogy(Nst,repmat(sem_eco,N),'LineWidth',2)
%legend('Sem Mecanismos de Economia')
Nstt=transpose((sem_eco));
save En1 Nstt -ASCII
hold on
semilogy(Nst,hiber,'LineWidth',2)
%legend('Modo de Hibernação')
Nstt=transpose((hiber));
save En2 Nstt -ASCII
hold on
semilogy(Nst,repmat(energia_sleep,N),'LineWidth',2)
%legend('Mecanismo Proposto')
Nstt=transpose((energia_sleep));
save En3 Nstt -ASCII
hold on
semilogy(Nst,energia_sleep2,'LineWidth',2)
%legend('Controle de Potência')
Nstt=transpose((energia_sleep2));
save En4 Nstt -ASCII
legend('Sem Mecanismos de Economia','Modo de Hibernação','Mecanismo Proposto','Controle de Potência')
xlabel('Iterações')
ylabel('Energia por bit (J/bit)')
axis([0 500 9e-6 1e-2])
%hold on

Nstt=transpose((energia_sleep2));
save En4 Nstt -ASCII


Energiasem=(16*Ptotala*Tc);
energia_sleep3=Ptotalc*Tc;
nnef1=(Energiasem-energia_sleep)./Energiasem;
nnef2=(Energiasem-energia_sleep2)./Energiasem;
nnef3=(Energiasem-energia_sleep3)./Energiasem;

Nstt=transpose(nnef1);
save Ef1 Nstt -ASCII
Nstt=transpose(nnef2);
save Ef2 Nstt -ASCII
Nstt=transpose(nnef3);
save Ef3 Nstt -ASCII


figure(62)
plot(Nst,nnef2,'r',Nst,nnef3,'g',Nst,repmat(nnef1,1800),'b','LineWidth',2)
legend('Mecanismo Proposto','Controle de Potência','Modo de Hibernação')
axis([0 500 0 1])
xlabel('Iterações')
ylabel('Razão de economia de energia')



%'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)

% ro=0:1/1799:1;
% aa=0.1;
% Pactive=Ptotalc(1800);
% Psleep=0.33.*Pactive;
% Pconsu=((1+aa).*ro.*Pactive)+((1-((1+aa).*ro)).*Psleep);
% 
% figure(63)
% plot(ro,Pconsu)
% hold on




%T=0.2 P=0.1
%T=0.5 P=0.3 

%energia=((sum(bestaux)./0.1)./bitrate);
energia=((sum(bestaux)./0.1).*Tc);
%energia_sem=(((sum(bestaux)./0.1).*(ef)./bitrate)+ (0.33*((sum(bestaux)./0.1)).*(1-ef)./bitrate));
energia_sem=((((sum(bestaux)./0.01).*((Tb)+(0.5*Tc))))+ ((0.3*((sum(bestaux)./0.01)).*((Tc-(Tb)-(0.5*Tc))))));
%energia_sem=((((sum(bestaux)./0.1).*((1./bitrate)+(0.2*Tc))))+ ((0.1*((sum(bestaux)./0.1)).*((Tc-(1./bitrate)-(0.2*Tc))))));
%energia_sem=16*(1e-3/0.1);
%energia_sem=(((16*(1e-3/0.1))).*(ef))+ (0.33*((16*(1e-3/0.1))).*(1-ef));
%energia_sem=16*1e-3;

figure(10)
%semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia./10e9./1e-12)
%semilogy(sum(repmat(Popt,1,N))./sum(bestaux),150e-9)
semilogy(Nst,150e-12)
Nstt=transpose(150e-12);
%save energia1 Nstt -ASCII
%save energia1 Nstt 
hold on
energia_sleep=((((1.5e-3)./0.1).*((Tb)+(0.5*Tc))))+ ((0.3*((1.5e-3)./0.1)).*((Tc-(Tb)-(0.5*Tc))));
 figure(10)
% %semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia./10e9./1e-12)
 %semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia_sleep,'g-d')
 semilogy(Nst,energia_sleep,'r-d')
% hold on
Nstt=transpose(energia_sleep);
save energia2 Nstt -ASCII


figure(10)
%semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia./10e9./1e-12)
%semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia)
semilogy(Nst,energia)
hold on
figure(10)
%semilogy(sum(repmat(Popt,1,N))./sum(bestaux),energia_sem./10e9./1e-12,'g*')
semilogy(Nst,energia_sem,'g-*')
hold on
Nstt=transpose(energia_sem);
save energia3 Nstt -ASCII

%axis([0 1 1e-1 1e2])

% (sum(bestaux(400))./10e9./1e-12)
% (sum(bestaux(800))./10e9./1e-12)

Nstt=transpose(sum(repmat(Popt,1,N))./sum(bestaux));
save convergencia Nstt -ASCII

Nstt=transpose(nnodes.*(sum(bestaux))./10e9./1e-12);
save abener Nstt -ASCII

%semilogy(0:(1/7):1,((10*log10((q^2/sigma_cc2).*sum(sniraux)))))

figure(6)
bar(1:TR,NSE)
%stem(1:TR,NSE)
hold on
semilogy(1:TR,repmat(mean(NSE),1,TR),'r:')
title('NSE for Each Trial')
xlabel('Trials, TR')
ylabel('NSE')




Nstt=transpose(bestaux);
save power_pso Nstt -ASCII

Nstt=transpose(repmat(Popt,1,N));
save power_pso_op Nstt -ASCII

figure(40)

tmp = zeros(1,N);
for i=1:1:N
    tmp(i) = norm(bestaux(:,i) - Popt)./norm(Popt);
end

semilogy(1:1:N,tmp);

xlabel('Iterações, N')
ylabel('NMSE')

title('PSO - NMSE');

Nstt=transpose(tmp);
save nmse Nstt -ASCII


Nstt=transpose(Nst);
save ait Nstt -ASCII

% Alocação de Buffer

energia=((sum(bestaux)./0.1).*Tc);

for i=1:K

Pb=1/(K+1);
auto=(energia.*Pb);
end

figure(60)
semilogy(Nst,energia,'r')
hold on
semilogy(Nst,auto,'b')
legend('Sem Buffer', 'Com Buffer')

