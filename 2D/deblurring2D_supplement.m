% "Low Rank Independence Samplers in Bayesian Inverse Problems"
%
% Supplementary experiments for the 2D image deblurring example using blur 
% operator from Regularization Tools (Per Christian Hansen, IMM, 1997).
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017


close all; clc;

% Make sure tools folder is added to search path
% addpath('.../LRIS/tools');

%% Set up problem parameters

% Load setup parameters generated by deblurring2D_main
load('deblurring2D-setup.mat', 'V', 'l', 'x', 'A', 'L', 'N');

% Generate blurred image b
b = A*x;

% Add noise of different levels
lNoise = 1.e-2*norm(b, 'inf')*randn(size(b,1),1);
hNoise = 0.5.*norm(b, 'inf')*randn(size(b,1),1);
bNoisy = b + hNoise;
bClean = b + lNoise;

% Display true and blurred, noisy images
figure;
subplot(2,2,1)
imshow(reshape(x,[N,N]))
subplot(2,2,2)
imshow(reshape(b,[N,N]))
subplot(2,2,3)
imshow(reshape(bNoisy,[N,N]))
subplot(2,2,4)
imshow(reshape(bClean,[N,N]))

% Specify cutoff for low-rank approximation
R = 500;
V_R = V(:,1:R);
l_R = l(1:R);

%% Sampling: Gamma NCP

fprintf('Sampling (Gamma NCP) \n\n');

% Specify hyperparameters (shape/rate parameterization)
am = 1.e-1;
bm = 1.e-1;
aSig = 1.e-1;
bSig = 1.e-1;

% Run three chains in parallel (5000 iterations, just to observe mixing)
nchains = 3;
parpool(nchains)
maxIter = 5000;

% Initialize chains at different values
% Note same initialization is used for all experiments, NCP and CP
muInit = gamrnd(am,1/bm,3,1);
sigInit = [0.1; 6; 25];
zInit = L*randn(N^2,3);

% Noisy data NCP
parXNCPNoisy = cell(nchains,1);
parMuNCPNoisy = cell(nchains,1);
parSigNCPNoisy = cell(nchains,1);
ratesNCPNoisy = zeros(nchains,1);

parfor chaincount = 1:nchains
    
    [XDraws, muDraws, sigDraws, tmpRt] = metwithingibbsNCP(bNoisy, A, L, ...
            V_R, l_R, am, bm, aSig, bSig, muInit(chaincount), sigInit(chaincount), ...
            zInit(:,chaincount), maxIter);
    
    parXNCPNoisy{chaincount} = XDraws;
    parMuNCPNoisy{chaincount} = muDraws;
    parSigNCPNoisy{chaincount} = sigDraws;
    ratesNCPNoisy(chaincount) = tmpRt;
    
end 

% Clean data NCP
parXNCPClean = cell(nchains,1);
parMuNCPClean = cell(nchains,1);
parSigNCPClean = cell(nchains,1);
ratesNCPClean = zeros(nchains,1);

parfor chaincount = 1:nchains
    
    [XDraws, muDraws, sigDraws, tmpRt] = metwithingibbsNCP(bClean, A, L, ...
            V_R, l_R, am, bm, aSig, bSig, muInit(chaincount), sigInit(chaincount), ...
            zInit(:,chaincount), maxIter);
    
    parXNCPClean{chaincount} = XDraws;
    parMuNCPClean{chaincount} = muDraws;
    parSigNCPClean{chaincount} = sigDraws;
    ratesNCPClean(chaincount) = tmpRt;
    
end
poolobj= gcp('nocreate');
delete(poolobj);


%% Sampling: Gamma CP

fprintf('Sampling (Gamma CP) \n\n');

parpool(nchains)

% Noisy data CP
parXCPNoisy = cell(nchains,1);
parMuCPNoisy = cell(nchains,1);
parSigCPNoisy = cell(nchains,1);
ratesCPNoisy = zeros(nchains,1);

parfor chaincount = 1:nchains
    
    [XDraws, muDraws, sigDraws, ~, tmpRt] = metwithingibbsLRIS(bNoisy, A, L, ...
            V_R, l_R, am, bm, aSig, bSig, muInit(chaincount), sigInit(chaincount), ...
            zInit(:,chaincount), maxIter);
    
    parXCPNoisy{chaincount} = XDraws;
    parMuCPNoisy{chaincount} = muDraws;
    parSigCPNoisy{chaincount} = sigDraws;
    ratesCPNoisy(chaincount) = tmpRt;
    
end 

% Clean data CP
parXCPClean = cell(nchains,1);
parMuCPClean = cell(nchains,1);
parSigCPClean = cell(nchains,1);
ratesCPClean = zeros(nchains,1); 

parfor chaincount = 1:nchains
    
    [XDraws, muDraws, sigDraws, ~, tmpRt] = metwithingibbsLRIS(bClean, A, L, ...
            V_R, l_R, am, bm, aSig, bSig, muInit(chaincount), sigInit(chaincount), ...
            zInit(:,chaincount), maxIter);
    
    parXCPClean{chaincount} = XDraws;
    parMuCPClean{chaincount} = muDraws;
    parSigCPClean{chaincount} = sigDraws;
    ratesCPClean(chaincount) = tmpRt;
    
end
poolobj= gcp('nocreate');
delete(poolobj);


%% Diagnostics

fprintf('Diagnostics \n\n');

% Trace plots of sigma under all four configurations
% Left column = noisy, right column = clean 
% Top row = NCP, bottom row = CP
figure;
subplot(2,2,1)
plot(parSigNCPNoisy{1}(101:5000));
xlim([1 4900])
xlabel('Iteration')
ylabel('\sigma')
hold on
plot(parSigNCPNoisy{2}(101:5000), 'k-')
plot(parSigNCPNoisy{3}(101:5000), 'r-')
title('50% Noise')

subplot(2,2,2)
plot(parSigNCPClean{1}(101:5000))
xlim([1 4900])
xlabel('Iteration')
ylabel('\sigma')
hold on
plot(parSigNCPClean{2}(101:5000), 'k-')
plot(parSigNCPClean{3}(101:5000), 'r-')
title('1% Noise')

subplot(2,2,3)
plot(parSigCPNoisy{1}(101:5000))
xlim([1 4900])
xlabel('Iteration')
ylabel('\sigma')
hold on
plot(parSigCPNoisy{2}(101:5000), 'k-')
plot(parSigCPNoisy{3}(101:5000), 'r-')

subplot(2,2,4)
plot(parSigCPClean{1}(101:5000))
xlim([1 4900])
xlabel('Iteration')
ylabel('\sigma')
hold on
plot(parSigCPClean{2}(101:5000), 'k-')
plot(parSigCPClean{3}(101:5000), 'r-')

% Autocorrelation plots and coefficients
burnin = maxIter/2;
samples = burnin+1:maxIter;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigNCPNoisy{1}(samples),50);
title('\textbf{NCP} $\mathbf{\sigma}$, \textbf{Noisy}','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigNCPClean{1}(samples),50);
title('\textbf{NCP} $\mathbf{\sigma}$, \textbf{Clean}','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigCPNoisy{1}(samples),50);
title('\textbf{CP} $\mathbf{\sigma}$, \textbf{Noisy}','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigCPClean{1}(samples),50);
title('\textbf{CP} $\mathbf{\sigma}$, \textbf{Clean}','fontsize',18,'interpreter','latex');

y = autocorr(parSigNCPNoisy{1}(samples),50); 
fprintf('Lag 1 ACF, NCP noisy: %f \n', y(2));  
fprintf('Lag 50 ACF, NCP noisy: %f \n', y(51));
y = autocorr(parSigNCPClean{1}(samples),50);
fprintf('Lag 1 ACF, NCP clean: %f \n', y(2));
fprintf('Lag 50 ACF, NCP clean: %f \n', y(51));
y = autocorr(parSigCPNoisy{1}(samples),50);
fprintf('Lag 1 ACF, CP noisy: %f \n', y(2));   
fprintf('Lag 50 ACF, CP noisy: %f \n', y(51)); 
y = autocorr(parSigCPClean{1}(samples),50);
fprintf('Lag 1 ACF, CP clean: %f \n', y(2));    
fprintf('Lag 50 ACF, CP clean: %f \n', y(51));
