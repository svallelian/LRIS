% "Low Rank Independence Samplers in Bayesian Inverse Problems"
%
% 2D image deblurring example using blur operator from Regularization
% Tools (Per Christian Hansen, IMM, 1997).
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017


close all; clc;

% Make sure tools folder is added to search path
% addpath('.../LRIS/tools');


%% Set up problem parameters

% Number of pixels per side
N = 50;

% Blur operator parameters
blurband = 10;
blursigma = 8; % higher sigma means more blurring

% Generate blur matrix A, true image x, and blurred image b
% Currently using default image specified in blur.m
% Possible alternatives: 
%   Shepp-Logan
%       Im = phantom(N); x = Im(:);
%   Rice image in Image Processing toolbox [256x256]
%       Im = im2double(imread('rice.png'));
%       N = 256; x = Im(:);

[A,b,x] = blur(N,blurband,blursigma);


% Add Gaussian random noise to blurred image
noise = 1.e-2*norm(b, 'inf')*randn(size(b,1),1);
b = b + noise;

% Display true and blurred, noisy image
figure;
subplot(1,2,1)
imshow(reshape(b,[N,N]))
subplot(1,2,2)
imshow(reshape(x,[N,N]))


%% Set up the decompositions for sampling

% Smoothness prior on x
scale = 0.0001;
L = gallery('neumann',size(x,1)) + scale.*speye(size(x,1));

% Find the eigendecomposition of the prior preconditioned Hessian H
% Note this collects more modes than we need, to be used later
k = N^2 - 1;
[V,l] = ppGNH_stable(A,L,k);

% Plot the spectrum of H
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
semilogy(1:2000,l(1:2000), 'k-', 'LineWidth', 2)
xlabel('\textbf{Index} $\mathbf{i}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\lambda}$','fontsize',18,'interpreter','latex');
axis tight;

% Save setup parameters
save('deblurring2D-setup.mat','-v7.3');


%% Solve the problem with naive block Gibbs

fprintf('Sampling (block Gibbs) \n\n');

% Optional: load setup parameters
%load('deblurring2D-setup.mat','x','A','b','L','V','l');

% Specify hyperparameters (shape/rate parameterization)
am = 1.e-1;
bm = 1.e-1;
aSig = 1.e-1;
bSig = 1.e-1;
% Optional: save setup parameters
%save('deblurring2D-setup.mat','am','bm','aSig','bSig','-append');

% Run three chains in parallel
nchains = 3;
parpool(nchains)
parXDrawsG = cell(nchains,1);
parMuDrawsG = cell(nchains,1);
parSigmaDrawsG = cell(nchains,1);
parLkhdG = cell(nchains,1);
maxIter = 50000;

% Initialize chains at different values
muInit = [1./var(b); gamrnd(0.1,10,2,1)];
sigInit = gamrnd(0.1,10,3,1);

tic;
parfor chain_count = 1:nchains

    % Sample with block Gibbs
    [XdrawsG, MudrawsG, SigmadrawsG, LkhdG] = blockgibbs(b, A, L, am, bm, ...
        aSig, bSig, muInit(chain_count), sigInit(chain_count), maxIter);  
                                
    parXDrawsG{chain_count} = XdrawsG;
    parMuDrawsG{chain_count} = MudrawsG;
    parSigmaDrawsG{chain_count} = SigmadrawsG;
    parLkhdG{chain_count} = LkhdG;
    
end
gibbstime = toc;
fprintf('Time elapsed, block Gibbs: %f sec \n', gibbstime);
poolobj = gcp('nocreate');
delete(poolobj);

% Use the output to track realizations of lambda = sigma/mu
parLambdaG = cell(3,1);
parLambdaG{1}= parSigmaDrawsG{1}./parMuDrawsG{1};
parLambdaG{2}= parSigmaDrawsG{2}./parMuDrawsG{2};
parLambdaG{3}= parSigmaDrawsG{3}./parMuDrawsG{3};

% Optional: save output
%save('deblurring2D-SVD-BGoutput.mat','parXDrawsG','parMuDrawsG','parSigmaDrawsG','parLkhdG','gibbstime','-v7.3');


%% Convergence diagnostics for block Gibbs
% Computed using the last 50% of samples in the chains

fprintf('Convergence diagnostics (block Gibbs) \n\n');

% Optional: load output
%load('deblurring2D-SVD-BGoutput.mat');

% Potential scale reduction factors
R_muG = computePSRF(parMuDrawsG);
fprintf('For mu: R_hat = %f \n', R_muG);
R_sigmaG = computePSRF(parSigmaDrawsG);
fprintf('For sigma: R_hat = %f \n', R_sigmaG);
R_lambdaG = computePSRF(parLambdaG);
fprintf('For lambda: R_hat = %f \n', R_lambdaG);
R_xG = computeMVPSRF(parXDrawsG);
fprintf('For x: MvR_hat = %f \n', R_xG);

burnin = maxIter/2;
samples = burnin+1:maxIter;

% Autocorrelation plots and effective sample sizes
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parMuDrawsG{1}(samples),50);
title('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigmaDrawsG{1}(samples),50);
title('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');

y = autocorr(parMuDrawsG{1}(samples),50);
L1muG = y(2);
ESSmuG = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for mu: %f \n', L1muG); 
fprintf('ESS for mu: %f \n', ESSmuG);

y = autocorr(parSigmaDrawsG{1}(samples),50);
L1sigmaG = y(2);
ESSsigmaG = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for sigma: %f \n', L1sigmaG); 
fprintf('ESS for sigma: %f \n', ESSsigmaG);

CES_sigmaG = gibbstime/ESSsigmaG;
fprintf('Cost per effective sample for sigma: %f \n', CES_sigmaG);

% Trace plots for mu and sigma
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parMuDrawsG{1}(samples)); hold on;
plot(parMuDrawsG{2}(samples),'k-');
plot(parMuDrawsG{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_muG,1) ',  Lag1 ACF = ' ...
        num2str(L1muG,2) ',  ESS = ' num2str(ESSmuG,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parSigmaDrawsG{1}(samples)); hold on;
plot(parSigmaDrawsG{2}(samples),'k-');
plot(parSigmaDrawsG{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_sigmaG,1) ',  Lag1 ACF = ' ...
        num2str(L1sigmaG,2) ',  ESS = ' num2str(ESSsigmaG,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

% Combine draws
XDraws = [parXDrawsG{1}(:,samples) parXDrawsG{2}(:,samples) parXDrawsG{3}(:,samples)];
MuDraws = [parMuDrawsG{1}(:,samples) parMuDrawsG{2}(:,samples) parMuDrawsG{3}(:,samples)];
SigmaDraws = [parSigmaDrawsG{1}(:,samples) parSigmaDrawsG{2}(:,samples) parSigmaDrawsG{3}(:,samples)];
lambdas = [parLambdaG{1}(samples) parLambdaG{2}(samples) parLambdaG{3}(samples)];
fprintf('Mean of lambdas: %f \n', mean(lambdas));

% Plot the posterior mean
estImg = mean(XDraws,2);
N = sqrt(length(x));
figure;
imshow(reshape(estImg,[N,N]))

% Relative error
relerror = sqrt(sum((estImg-x).^2)/sum(x.^2));
fprintf('Relative error between true and posterior mean: %f \n', relerror);

% Plot the estimated joint density for (mu, sigma)
c = [ones(1,maxIter-burnin) 5*ones(1,maxIter-burnin) 10*ones(1,maxIter-burnin)];
figure; 
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
scatterhist(MuDraws,SigmaDraws,'Group',c,'Kernel','on','Legend','off','Marker','+do','MarkerSize',3);
xlabel('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;


%% Solve the problem with LRIS in Metropolis-Hastings-within-Gibbs

fprintf('Sampling (LRIS) \n\n');

% Optional: load setup parameters
%load('deblurring2D-setup.mat','x','A','b','L','V','l');
%load('deblurring2D-setup.mat','am','bm','aSig','bSig');

% Specify cutoff for low-rank approximation
R = 500;

% Run three chains in parallel
nchains = 3;
parpool(nchains)
parXDraws = cell(nchains,1);
parMuDraws = cell(nchains,1);
parSigmaDraws = cell(nchains,1);
maxIter = 50000;
ratios = zeros(nchains,maxIter);
rates = zeros(nchains,1);

% Initialize chains at different values
muInit = [1./var(b); gamrnd(0.1,10,2,1)];
sigInit = gamrnd(0.1,10,3,1);
xInit = [ones(length(x),1), 0.5*ones(length(x),1), 0.75*ones(length(x),1)]; 

tic;
parfor chain_count = 1:nchains

    % Sample and compute acceptance ratios
    [Xdraws, Mudraws, Sigmadraws, rats, rate] = metwithingibbsLRIS(b, A, L,...
            V(:,1:R), l(1:R), am, bm, aSig, bSig, muInit(chain_count), ...
            sigInit(chain_count), xInit(:,chain_count), maxIter); 
                                
    parXDraws{chain_count} = Xdraws;
    parMuDraws{chain_count} = Mudraws;
    parSigmaDraws{chain_count} = Sigmadraws;
    ratios(chain_count,:) = rats;
    rates(chain_count) = rate;
    
end
LRIStime = toc;
fprintf('Time elapsed, LRIS: %f sec \n', LRIStime);
poolobj = gcp('nocreate');
delete(poolobj);

% Use the output to track realizations of lambda = sigma/mu
parLambda = cell(3,1);
parLambda{1}= parSigmaDraws{1}./parMuDraws{1};
parLambda{2}= parSigmaDraws{2}./parMuDraws{2};
parLambda{3}= parSigmaDraws{3}./parMuDraws{3};

% Optional: save output
%save('deblurring2D-SVD-LRISoutput.mat','parXDraws','parMuDraws','parSigmaDraws',...
%   'ratios','rates','LRIStime','-v7.3');


%% Convergence diagnostics for LRIS
% Computed using the last 50% of samples in the chains

fprintf('Convergence diagnostics (LRIS) \n\n');

% Optional: load output
%load('deblurring2D-SVD-LRISoutput.mat');

% PSRFs
R_mu = computePSRF(parMuDraws);
fprintf('For mu: R_hat = %f \n', R_mu);
R_sigma = computePSRF(parSigmaDraws);
fprintf('For sigma: R_hat = %f \n', R_sigma);
R_lambda = computePSRF(parLambda);
fprintf('For lambda: R_hat = %f \n', R_lambda);
R_x = computeMVPSRF(parXDraws);
fprintf('For x: MvR_hat = %f \n', R_x);

burnin = maxIter/2;
samples = burnin+1:maxIter;

% Autocorrelation plots and effective sample sizes
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parMuDraws{1}(samples),50);
title('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(parSigmaDraws{1}(samples),50);
title('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');

y = autocorr(parMuDraws{1}(samples),50);
L1mu = y(2);
ESSmu = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for mu: %f \n', L1mu); 
fprintf('ESS for mu: %f \n', ESSmu);

y = autocorr(parSigmaDraws{1}(samples),50);
L1sigma = y(2);
ESSsigma = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for sigma: %f \n', L1sigma); 
fprintf('ESS for sigma: %f \n', ESSsigma);

CES_sigma = LRIStime/ESSsigma; 
fprintf('Cost per effective sample for sigma: %f \n', CES_sigma);

% Compare cumulative averages for mu and sigma for both sampling methods
% Optional: load output
%load('deblurring2D-SVD-BGoutput.mat','parMuDrawsG','parSigmaDrawsG');
MuCA_MWG = computeCA(parMuDraws);
SigCA_MWG = computeCA(parSigmaDraws);
MuCA_BG = computeCA(parMuDrawsG);
SigCA_BG = computeCA(parSigmaDrawsG);

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
subplot(1,2,1);
plot(1:length(samples),MuCA_MWG{1},'b-',1:length(samples),MuCA_MWG{2},...
    'k-',1:length(samples),MuCA_MWG{3},'r-','LineWidth',2); hold on
plot(1:length(samples),MuCA_BG{1},'b-.',1:length(samples),MuCA_BG{2},...
    'k-.',1:length(samples),MuCA_BG{3},'r-.','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;
subplot(1,2,2);
plot(1:length(samples),SigCA_MWG{1},'b-',1:length(samples),SigCA_MWG{2},...
    'k-',1:length(samples),SigCA_MWG{3},'r-','LineWidth',2); hold on
plot(1:length(samples),SigCA_BG{1},'b-.',1:length(samples),SigCA_BG{2},...
    'k-.',1:length(samples),SigCA_BG{3},'r-.','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

% Compare data-misfit parts of log-likelihoods for both sampling methods
% Optional: load output
%load('deblurring2D-SVD-BGoutput.mat','parXDrawsG','parLkhdG');
%R_xG = computeMVPSRF(parXDrawsG);
parLkhd = computeLkhd(b, A, parXDraws, parMuDraws);

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
subplot(1,2,1);
plot(1:length(samples),parLkhdG{1}(samples)); hold on
plot(1:length(samples),parLkhdG{2}(samples), 'k-')
plot(1:length(samples),parLkhdG{3}(samples), 'r-')
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('$\log f(y \mid x, \mu)$', 'fontsize',18,'interpreter','latex');
title(['$\widehat{R}_{MV}$ \textbf{= ' num2str(R_xG,3) '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
subplot(1,2,2)
plot(1:length(samples),parLkhd{1}(samples)); hold on
plot(1:length(samples),parLkhd{2}(samples), 'k-')
plot(1:length(samples),parLkhd{3}(samples), 'r-')
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('$\log f(y \mid x, \mu)$', 'fontsize',18,'interpreter','latex');
title(['$\widehat{R}_{MV}$ \textbf{= ' num2str(R_x,3) '}'], ...
        'fontsize', 18, 'interpreter', 'latex');

% Trace plots for mu and sigma
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parMuDraws{1}(samples)); hold on;
plot(parMuDraws{2}(samples),'k-');
plot(parMuDraws{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_mu,1) ',  Lag1 ACF = ' ...
    num2str(L1mu,2) ',  ESS = ' num2str(ESSmu,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parSigmaDraws{1}(samples)); hold on;
plot(parSigmaDraws{2}(samples),'k-');
plot(parSigmaDraws{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_sigma,1) ',  Lag1 ACF = ' ...
    num2str(L1sigma,2) ',  ESS = ' num2str(ESSsigma,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

% Combine draws
XDraws = [parXDraws{1}(:,samples) parXDraws{2}(:,samples) parXDraws{3}(:,samples)];
MuDraws = [parMuDraws{1}(:,samples) parMuDraws{2}(:,samples) parMuDraws{3}(:,samples)];
SigmaDraws = [parSigmaDraws{1}(:,samples) parSigmaDraws{2}(:,samples) parSigmaDraws{3}(:,samples)];
lambdas = [parLambda{1}(samples) parLambda{2}(samples) parLambda{3}(samples)];
fprintf('Mean of lambdas: %f \n', mean(lambdas));

% Plot the posterior mean
estImg = mean(XDraws,2);
N = sqrt(length(x));
figure;
imshow(reshape(estImg,[N,N]))

% Relative error
relerror = sqrt(sum((estImg-x).^2)/sum(x.^2));
fprintf('Relative error between true and posterior mean: %f \n', relerror);

% Plot estimated joint density for (mu, sigma)
c = [ones(1,maxIter-burnin) 5*ones(1,maxIter-burnin) 10*ones(1,maxIter-burnin)];
figure; 
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
scatterhist(MuDraws,SigmaDraws,'Group',c,'Kernel','on','Legend','off','Marker','+do','MarkerSize',3);
xlabel('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

% Compare the priors and marginal posteriors of mu and sigma
xsig = 0.01:0.01:2.25;
y1 = gampdf(xsig,0.1,1/0.1); % prior
[fsig, xisig] = ksdensity(SigmaDraws,xsig); % marginal posterior

xmu = 100:6500;
y2 = gampdf(xmu,0.1,1/0.1); % prior
[fmu, ximu] = ksdensity(MuDraws,xmu); % marginal posterior

% Plot for sigma
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(xsig,y1,xsig,fsig,'LineWidth',3);
xlabel('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\pi(\sigma)}$','fontsize',18,'interpreter','latex');
lg = legend('prior', 'marginal posterior');
lg.Location = 'best';
axis tight;

% Plot for mu, note scaling on prior
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
semilogx(xmu,1e5*y2,xmu,fmu,'LineWidth',3);
xlabel('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\pi(\mu)}$','fontsize',18,'interpreter','latex');
lg = legend('scaled prior', 'marginal posterior');
lg.Location = 'best';
axis tight;


%% Compare with randomized SVD for approximate proposal in LRIS

% Optional: load setup parameters
%load('deblurring2D-setup.mat','x','A','b','L','V','l');
%load('deblurring2D-setup.mat', 'am', 'bm', 'aSig', 'bSig');

% Compute approximate eigendecomposition using randSVD
k = length(x)-1;
p = 20;
[V_rand,l_rand] = ppGNH_randsvd(A,L,k-p,p);

% Optional: save setup parameters
%save('deblurring2D-setup.mat','V_rand','l_rand','-append');

fprintf('Sampling (LRIS with randSVD) \n\n');

% Specify cutoff for low-rank approximation
R = 500;

% Run three chains in parallel
nchains = 3;
parpool(nchains)
parXDrawsR = cell(3,1);
parMuDrawsR = cell(3,1);
parSigmaDrawsR = cell(3,1);
maxIter = 50000;
ratiosR = zeros(3,maxIter);
ratesR = zeros(3,1);

% Initialize chains at different values
muInit = [1./var(b); gamrnd(0.1,10,2,1)];
sigInit = gamrnd(0.1,10,3,1);
xInit = [ones(length(x),1), 0.5*ones(length(x),1), 0.75*ones(length(x),1)]; 

tic;
parfor chain_count = 1:nchains

    % Sample and compute acceptance ratios
    [XdrawsR, MudrawsR, SigmadrawsR, ratsR, rateR] = metwithingibbsLRIS(b, A, ...
            L, V_rand(:,1:R), l_rand(1:R), am, bm, aSig, bSig, ...
            muInit(chain_count), sigInit(chain_count), xInit(:,chain_count), maxIter);
                                
    parXDrawsR{chain_count} = XdrawsR;
    parMuDrawsR{chain_count} = MudrawsR;
    parSigmaDrawsR{chain_count} = SigmadrawsR;
    ratiosR(chain_count,:) = ratsR;
    ratesR(chain_count) = rateR;
    
end
LRIStimeR = toc;
fprintf('Time elapsed, LRIS: %f sec \n', LRIStimeR);
poolobj = gcp('nocreate');
delete(poolobj);

% Optional: save output
%save('deblurring2D-randSVD-LRISoutput.mat','parXDrawsR','parMuDrawsR','parSigmaDrawsR',...
%   'ratiosR','ratesR','LRIStimeR','-v7.3');


%% Convergence Diagnostics for LRIS with randSVD

fprintf('Convergence diagnostics (LRIS with randSVD) \n\n');

% Optional: load output
%load('deblurring2D-randSVD-LRISoutput.mat');

% PSRFs
R_muR = computePSRF(parMuDrawsR);
fprintf('For mu: R_hat = %f \n', R_muR);
R_sigmaR = computePSRF(parSigmaDrawsR);
fprintf('For sigma: R_hat = %f \n', R_sigmaR);
R_xR = computeMVPSRF(parXDrawsR);
fprintf('For x: MvR_hat = %f \n', R_xR);

burnin = maxIter/2;
samples = burnin+1:maxIter;

% CES
y = autocorr(parMuDrawsR{1}(samples),50);
L1muR = y(2);
ESSmuR = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for mu: %f \n', L1muR); 
fprintf('ESS for mu: %f \n', ESSmuR);

y = autocorr(parSigmaDrawsR{1}(samples),50);
L1sigmaR = y(2);
ESSsigmaR = length(samples)*(1-y(2))/(1+y(2));
fprintf('Lag 1 ACF for sigma: %f \n', L1sigmaR); 
fprintf('ESS for sigma: %f \n', ESSsigmaR);

CES_sigmaR = LRIStimeR/ESSsigmaR; 
fprintf('Cost per effective sample for sigma: %f \n', CES_sigmaR);

% Trace plots
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parMuDrawsR{1}(samples)); hold on;
plot(parMuDrawsR{2}(samples),'k-');
plot(parMuDrawsR{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_muR,1) ',  Lag1 ACF = ' ...
    num2str(L1muR,2) ',  ESS = ' num2str(ESSmuR,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(parSigmaDrawsR{1}(samples)); hold on;
plot(parSigmaDrawsR{2}(samples),'k-');
plot(parSigmaDrawsR{3}(samples),'r-');
title(['$\widehat{R}$ \textbf{= ' num2str(R_sigmaR,1) ',  Lag1 ACF = ' ...
    num2str(L1sigmaR,2) ',  ESS = ' num2str(ESSsigmaR,'%1.1e') '}'], ...
        'fontsize', 18, 'interpreter', 'latex');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

% Combine draws
XDrawsR = [parXDrawsR{1}(:,samples) parXDrawsR{2}(:,samples) parXDrawsR{3}(:,samples)];
MuDrawsR = [parMuDrawsR{1}(:,samples) parMuDrawsR{2}(:,samples) parMuDrawsR{3}(:,samples)];
SigmaDrawsR = [parSigmaDrawsR{1}(:,samples) parSigmaDrawsR{2}(:,samples) parSigmaDrawsR{3}(:,samples)];
lambdasR = SigmaDrawsR./MuDrawsR;
fprintf('Mean of lambdas: %f \n', mean(lambdasR));

% Plot the posterior mean
estImg = mean(XDraws,2);
figure;
imshow(reshape(estImg,[N,N]))

% Relative error
relerror = sqrt(sum((estImg-x).^2)/sum(x.^2));
fprintf('Relative error between true and posterior mean: %f \n', relerror); 


%% Investigate the acceptance rate versus rank

fprintf('Investigating acceptance versus rank \n\n');

% Optional: load setup parameters
%load('deblurring2D-setup.mat','x','A','b','L','V','l');
%load('deblurring2D-setup.mat','am','bm','aSig','bSig');

% Optional: load the LRIS output to initialize
%load('deblurring2D-SVD-LRISoutput.mat','parXDraws','parMuDraws','parSigmaDraws');

V_full = V; l_full = l;

% Initialize x, mu, sigma
xInit = parXDraws{1}(:,end);
muInit = parMuDraws{1}(:,end);
sigInit = parSigmaDraws{1}(:,end);

% Loop through different levels of truncation
K = 195;
maxIt = 2000;

Atb = A'*b;
xBar = (muInit*(A'*A) + sigInit*(L'*L))\(muInit*Atb);
Ax = A*xInit;   Lx = L*xInit;
lwx1 = muInit*(Ax'*Ax) + sigInit*(Lx'*Lx);

rats1 = zeros(maxIt, K);
accept1 = zeros(K,1);
fail1 = zeros(K,1);
predict = zeros(K,1);

lwx = zeros(K,1);
N_1inv = zeros(K,1);

% Start at truncation level 60 (before this, acceptance prediction is ~=0)
parpool(4)
parfor k = 1:K
    index = 50 + 10*k;
    V = V_full(:,1:index);  l = l_full(1:index);    % kth level truncated decomposition
    
    % Compute log(w(x))
    lwx2 = post_approx_q(xInit, L, V, l, muInit, sigInit);
    lwx(k) = -0.5*(lwx1 - lwx2);
    
    % Compute 1/N_1 using discarded eigenvalues
    evals = l_full(index+1:end);
    term1 = prod(1./sqrt(1 + (muInit/sigInit)*evals));
    term2 = 0;
    for j = index+1:length(b)-1
       vj = V_full(:,j);
       vj = L\vj;
       temp = (Atb'*vj)^2;
       term2 = term2 + temp*((muInit*l_full(j))/(muInit*l_full(j) + sigInit));
    end
    N_1inv(k) = term1*exp(-(muInit/sigInit)*(muInit/2)*term2);
    
    % Theorem 1
    predict(k) = N_1inv(k)*exp(-lwx(k));
    
    for i=1:maxIt
        
       % Sample z from approximate normal distribution (LRIS)
       xStar = postsample(Atb, L, V, l, muInit, sigInit, randn(length(x),1));
       
       % Use Prop 2 to evaluate the MH acceptance ratio
       rats1(i,k) = metropolisratio2(xInit, xStar, L, V_full, l_full, muInit, index);
       
       % Accept/Reject
       u = rand(1);
       if u <= rats1(i,k)
           accept1(k) = accept1(k) + 1;
       else
           fail1(k) = fail1(k) + 1;
       end
    end
end
poolobj = gcp('nocreate');
delete(poolobj);

% Empirical failure rate: avg[ 1 - min(1, eta) ]
emp_fail_mean = sum(1-min(rats1,1),1)./maxIt;
emp_fail_upper = quantile(1-min(rats1,1),0.975,1);
emp_fail_lower = quantile(1-min(rats1,1),0.025,1);
% Predicted failure rate
failpredict = 1 - min(1, predict);
plotindex = find(failpredict>0);
xindex = 50 + 10.*plotindex;

figure; % Failure rates
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
semilogy(xindex, 1 - min(1, predict(plotindex)), 'r-', 'linewidth',5); hold on
semilogy(xindex, emp_fail_mean(plotindex), '--', 'color', '[0.75, 0.75, 0.75]','linewidth',3);
semilogy(xindex, emp_fail_upper(plotindex), 'k-.', 'linewidth', 2);
semilogy(xindex, emp_fail_lower(plotindex), 'k-.', 'linewidth', 2);
plegend = legend('Predicted: $1 - \min(1, E[\eta])$', 'Empirical: $1-E[\min(1, \eta)]$');
set(plegend,'fontsize',16, 'location', 'southwest');
set(plegend,'interpreter', 'latex');
ylabel('\textbf{Failure rate}','fontsize',18,'interpreter', 'latex');
xlabel('\textbf{K}','fontsize',18,'interpreter', 'latex');
axis tight;

