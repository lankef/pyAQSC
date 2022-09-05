%% Landreman stellarator 5.2
% To obtain a configuration that agrees with the conditions of variational
% approach, we run [x,err] = ax1.minimiseAnisotropy([],["S0","Btc20C"],[],x,0);
% to minimise it. The values are commented in what follows

ax1 = nearAxisField();
ax1.S0 = 0; %-0.000056564331054687504499439015814843;
ax1.Btc20C = 0;%0.0056392303771972650261457005171906;
ax1.eta = 0.632*sqrt(2);
ax1.p0C = 0;
ax1.d0.chnghar([3,0,0,1,0,0,2,0,0]);
ax1.B22c = -4*(-0.158-0.75*0.632^2);
ax1.B20c = -4*(0.35-0.75*0.632^2);
ax1.B22s = 0;
ax1.B31c = 0;
ax1.B31s = 0;
ax1.B33c = 0;
ax1.B33s = 0;
% ax1.sgnPsi = -1;

ax1.changerng(1001)

Rax = perFun([4,0,1,2,0.173,0,4,0.0168,0,6,0.00101,0],[],ax1.rngPhi,0);
Zax = perFun([3,2,0,0.159,4,0,0.0165,6,0,0.000987],[],ax1.rngPhi,0);

ax1.createAxis(Rax,Zax)

ax1.firstOrderEvaluation()
ax1.helicityQS()
fprintf('Rotational transform: %d \n',ax1.iota+ax1.helicity)

% Try to make pressure positive
ax1.Ba1 = 0;% -ax1.Ba0*ax1.B0*(0.5); %Chosen so that no negative pressure
% Gives beta~1/1000 corrections/deviations from isotropy
ax1.secondOrderEvaluation(1);

ax1.findD20(0,0.1)

ax1.thirdOrderEvaluation()

for name = ["kappa" "tau" "X20c" "X22c" "X22s" "Y20" "Y22c" "Y22s" "Z20" "Z22c" "Z22s" "Bp0" "Bpc11" "Bps11" "Btc20" "Xc1" "Ys1" "Yc1"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.2\',(name),'.dat'),'val','-ascii');
end

for name = ["X31c" "X31s" "X33c" "X33s" "Y31c" "Y31s" "Y33c" "Y33s" "Z31c" "Z31s" "Z33c" "Z33s" "p20c" "d20c" "p22s" "p22c" "d11c" "d11s" "pc1" "d0" "p0"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.2\',(name),'.dat'),'val','-ascii');
end

valIn = [ax1.nfp ax1.S0 ax1.eta ax1.B20c ax1.B22c ax1.B22s ax1.B31c ax1.B31s ax1.B33c ax1.B33s ax1.Ba0 ax1.Ba1];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.2\inputs.dat','valIn','-ascii');
valOut = [ax1.iota ax1.shear];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.2\outputs.dat','valOut','-ascii');


%% Landreman stellarator 5.3 (for anisotropic PRE paper)

ax1 = nearAxisField();
ax1.S0 = 0.00010100891539190635145389896010215; %0.00024517381191253661111909001313336; %0; 
ax1.Btc20C = 2.8545990635636613319547905120999; % 1.745203371047973739393910364015; % 2*0.9;
ax1.eta = 0.95*sqrt(2);
ax1.p0C = 0.08;
ax1.d0.chnghar([3,0,0,1,0,0,2,0,0]);
ax1.B22c = -4*(-0.7-0.75*0.95^2);
ax1.B20c = -4*(1.6-0.75*0.95^2);
ax1.B22s = 0;
ax1.B31s = 0.01;
ax1.B31c = 0.01;
ax1.B33c = 0;
ax1.B33s = 0;

ax1.changerng(1001)

Rax = perFun([2,0,1,2,0.09,0],[],ax1.rngPhi,0);
Zax = perFun([1,2,0,-0.09],[],ax1.rngPhi,0);
ax1.createAxis(Rax,Zax)

ax1.firstOrderEvaluation()
ax1.helicityQS()
fprintf('Rotational transform: %d \n',ax1.iota+ax1.helicity)

ax1.Ba1 = -2*ax1.Ba0*ax1.B0*4*pi*1e-7*(-8e5);
ax1.secondOrderEvaluation(1);

ax1.findD20(0,0.1)

ax1.thirdOrderEvaluation()

for name = ["kappa" "tau" "X20c" "X22c" "X22s" "Y20" "Y22c" "Y22s" "Z20" "Z22c" "Z22s" "Bp0" "Bpc11" "Bps11" "Btc20" "Xc1" "Ys1" "Yc1"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.3\',(name),'.dat'),'val','-ascii');
end

for name = ["X31c" "X31s" "X33c" "X33s" "Y31c" "Y31s" "Y33c" "Y33s" "Z31c" "Z31s" "Z33c" "Z33s" "p20c" "d20c" "p22s" "p22c" "d11c" "d11s" "pc1" "d0" "p0"]
    val = ax1.(name).val;   
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.3\',(name),'.dat'),'val','-ascii');
end

valIn = [ax1.nfp ax1.S0 ax1.eta ax1.B20c ax1.B22c ax1.B22s ax1.B31c ax1.B31s ax1.B33c ax1.B33s ax1.Ba0 ax1.Ba1];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.3\inputs.dat','valIn','-ascii');
valOut = [ax1.iota ax1.shear];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.3\outputs.dat','valOut','-ascii');


%% Landreman stellarator 5.4

ax1 = nearAxisField();
ax1.S0 = 0;
% ax1.S0 = 0.00097902662679553520766462249014239; %To reduce D20
ax1.Btc20C = 2*0;
% ax1.eta = 1.569*sqrt(2); %St
ax1.eta = 2.2078311385207785022544157982338;
ax1.p0C = 0.04;
ax1.d0.chnghar([3,0,0,1,0,0,2,0,0]);
ax1.B22c = -4*(0.1348-0.75*1.569^2);
ax1.B20c = -4*(1.3-0.75*1.569^2);
ax1.B22s = 0;
ax1.B31c = 0;
ax1.B31s = 0;
ax1.B33c = 0;
ax1.B33s = 0;

ax1.changerng(1001)

Rax = perFun([5,0,1,4,0.17,0,8,0.018191838304685660487880483060508,0,12,0.001420200653346571223467620370684,0,16,0.000059148767213264540889769427245426,0],[],ax1.rngPhi,0);
Zax = perFun([5,0,0,4,0,0.15464423020448464507481389773602,8,0,0.017913837381820285621358834760031,12,0,0.0014893730073718687394179749361456,16,0,0.00006923943338841940945208031088498],[],ax1.rngPhi,0);
ax1.createAxis(Rax,Zax)

ax1.firstOrderEvaluation()
ax1.helicityQS()
fprintf('Rotational transform: %d \n',ax1.iota+ax1.helicity)
%Helicity is computed following the curvature vector around the torus to
%track its excursion


% Try to make pressure positive
% ax1.Ba1 = -2*ax1.Ba0*ax1.B0*  4*pi*1e-7*(0); %Chosen so that no negative pressure
ax1.Ba1 = -0.048575078144669545820999445595589;
% Gives beta~1/1000 corrections/deviations from isotropy
ax1.secondOrderEvaluation(1);

ax1.findD20(0,0.1)

ax1.thirdOrderEvaluation()


for name = ["kappa" "tau" "X20c" "X22c" "X22s" "Y20" "Y22c" "Y22s" "Z20" "Z22c" "Z22s" "Bp0" "Bpc11" "Bps11" "Btc20" "Xc1" "Ys1" "Yc1"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.4\',(name),'.dat'),'val','-ascii');
end

for name = ["X31c" "X31s" "X33c" "X33s" "Y31c" "Y31s" "Y33c" "Y33s" "Z31c" "Z31s" "Z33c" "Z33s" "p20c" "d20c" "p22s" "p22c" "d11c" "d11s" "pc1" "d0" "p0"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.4\',(name),'.dat'),'val','-ascii');
end

valIn = [ax1.nfp ax1.S0 ax1.eta ax1.B20c ax1.B22c ax1.B22s ax1.B31c ax1.B31s ax1.B33c ax1.B33s ax1.Ba0 ax1.Ba1];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.4\inputs.dat','valIn','-ascii');
valOut = [ax1.iota ax1.shear];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.4\outputs.dat','valOut','-ascii');


%% Landreman stellarator 5.5

ax1 = nearAxisField();
ax1.S0 = 0.3;
ax1.Btc20C = 2*1.6;
ax1.eta = 2.5*sqrt(2);
ax1.p0C = 0.0;
ax1.d0.chnghar([3,0,0,1,0,0,2,0,0]);
ax1.B22c = -4*(1-0.75*2.5^2);
ax1.B20c = -4*(27-0.75*2.5^2);
ax1.B22s = -4*3;
ax1.B31c = 0;
ax1.B31s = 0;
ax1.B33c = 0;
ax1.B33s = 0;

ax1.changerng(1001)

Rax = perFun([2,0,1,5,0.3,0],[],ax1.rngPhi,0);
Zax = perFun([2,0,0,5,0,0.3],[],ax1.rngPhi,0);
ax1.createAxis(Rax,Zax)

ax1.firstOrderEvaluation()
ax1.helicityQS()
fprintf('Rotational transform: %d \n',ax1.iota+ax1.helicity)

% Try to make pressure positive
ax1.Ba1 = -2*ax1.Ba0*ax1.B0*4*pi*1e-7*(-5e6); %Chosen so that no negative pressure
% Gives beta~1/1000 corrections/deviations from isotropy
ax1.secondOrderEvaluation(1);

ax1.findD20(0,0.1)

ax1.thirdOrderEvaluation()


for name = ["kappa" "tau" "X20c" "X22c" "X22s" "Y20" "Y22c" "Y22s" "Z20" "Z22c" "Z22s" "Bp0" "Bpc11" "Bps11" "Btc20" "Xc1" "Ys1" "Yc1"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.5\',(name),'.dat'),'val','-ascii');
end

for name = ["X31c" "X31s" "X33c" "X33s" "Y31c" "Y31s" "Y33c" "Y33s" "Z31c" "Z31s" "Z33c" "Z33s" "p20c" "d20c" "p22s" "p22c" "d11c" "d11s" "pc1" "d0" "p0"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.5\',(name),'.dat'),'val','-ascii');
end

valIn = [ax1.nfp ax1.S0 ax1.eta ax1.B20c ax1.B22c ax1.B22s ax1.B31c ax1.B31s ax1.B33c ax1.B33s ax1.Ba0 ax1.Ba1];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.5\inputs.dat','valIn','-ascii');
valOut = [ax1.iota ax1.shear];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\5.5\outputs.dat','valOut','-ascii');


%% Circular axis anisotropy from paper

ax1 = nearAxisField();
ax1.S0 = 0;
ax1.Btc20C = 1.5;
ax1.eta = 0.9;
ax1.p0C = 1;
ax1.d0.chnghar([3,0,0,1,-0.1,0,2,-0.1,0]);
ax1.B22c = 0.01;
ax1.B20c = 0.01;
ax1.B22s = 0.01;
ax1.B31s = 0.01;
ax1.B31c = 0.01;
ax1.B33c = 0;
ax1.B33s = 0;

ax1.changerng(1001)

Rax = perFun([2,0,1,2,0.0001,0],[],ax1.rngPhi,0);
Zax = perFun([1,2,0,0.001],[],ax1.rngPhi,0);
ax1.createAxis(Rax,Zax)

ax1.firstOrderEvaluation()
ax1.helicityQS()
fprintf('Rotational transform: %d \n',ax1.iota+ax1.helicity)

% Try to make pressure positive
ax1.Ba1 = 0.1; %Chosen so that no negative pressure
% Gives beta~1/1000 corrections/deviations from isotropy
ax1.secondOrderEvaluation(1);

ax1.findD20(0,0.1)

ax1.thirdOrderEvaluation()


for name = ["kappa" "tau" "X20c" "X22c" "X22s" "Y20" "Y22c" "Y22s" "Z20" "Z22c" "Z22s" "Bp0" "Bpc11" "Bps11" "Btc20" "Xc1" "Ys1" "Yc1"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\circ\',(name),'.dat'),'val','-ascii');
end

for name = ["X31c" "X31s" "X33c" "X33s" "Y31c" "Y31s" "Y33c" "Y33s" "Z31c" "Z31s" "Z33c" "Z33s" "p20c" "d20c" "p22s" "p22c" "d11c" "d11s" "pc1" "d0" "p0"]
    val = ax1.(name).val;
    save(strcat('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\circ\',(name),'.dat'),'val','-ascii');
end

valIn = [ax1.nfp ax1.S0 ax1.eta ax1.B20c ax1.B22c ax1.B22s ax1.B31c ax1.B31s ax1.B33c ax1.B33s ax1.Ba0 ax1.Ba1];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\circ\inputs.dat','valIn','-ascii');
valOut = [ax1.iota ax1.shear];
save('D:\MATLAB\Stellerator\NAE\NAE_Code\Examples\LandrPaper\WithAnisotropy\FrankData\circ\output.dat','valOut','-ascii');

