% "Low Rank Independence Samplers in Bayesian Inverse Problems"
%
% CT image reconstruction example using X-ray forward operator (Erkki Somersalo, 2002).
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017


close all; clc;

% Make sure tools folder is added to search path
% addpath('.../LRIS/tools');

%% Set up problem parameters

% Number of measurements = nphi*ns = 5000
nphi = 50;
ns = 100;
N = 128; % pixels per side

% Generate forward map A
phi = linspace(-pi/2, pi/2, nphi);
s = linspace(-0.49, 0.49, ns);
[S,Phi] = meshgrid(s,phi);
A = Xraymat(S(:), Phi(:), N);
[m,n] = size(A);

% True image x: Shepp-Logan
Im = phantom(N);
x = Im(:);

% Generate data and add Gaussian random noise to it
b = A*x;
noise = 1.e-2*norm(b, 'inf')*randn(size(b,1),1);
b = b + noise;

% Plot noisy sinogram and true image
figure;
imshow(reshape(b, [nphi ns]))

figure;
imshow(reshape(x', [N N]))


%% Set up the decompositions for sampling

% Smoothness prior on x
scale = 0.001;
L = gallery('neumann',size(x,1)) + scale.*speye(size(x,1));

% Find the approx eigendecomposition of the prior preconditioned Hessian H
% Using randomized SVD approach
k = 5000;
p = 20;
[V,l] = ppGNH_randsvd(A,L,k,p);

% Plot the spectrum of H
% Note the spectrum drops around 5000 eigenvalues, the rank of A
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
semilogy(l, 'k-', 'LineWidth', 2) ;
xlabel('\textbf{Index} $\mathbf{i}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\lambda}$','fontsize',18,'interpreter','latex');
axis tight;

% Optional: save setup parameters
%save('CT-setup.mat', 'A', 'b', 'x', 'L', 'V', 'l','-v7.3');


%% Solve the problem with LRIS using proper Jeffreys priors 
% as in e.g. Scott and Berger (2006, JSPI)

fprintf('Sampling (LRIS w/Jeffreys priors), first 20k \n\n');

% Optional: load setup parameters
%load('CT-setup.mat');

% Experimenting with different cutoffs R shows that most of the eigenvalues
% must be kept to get a decent acceptance rate.
% Testing different R with 500 iterations each:
%   R = 4000 -> rate ~8%        R = 4600 -> rate ~40%
%   R = 4200 -> rate ~14%       R = 4800 -> rate ~69%
%   R = 4400 -> rate ~40%       R = 5000 -> rate ~100%

% Specify cutoff for low-rank approximation
R = 4900;

% Run three chains in parallel (20k iterations each, run twice)
nchains = 3;
parpool(nchains)
parXDraws = cell(nchains,1);
parKappa2Draws = cell(nchains,1);
parUpsDraws = cell(nchains,1);
maxIter = 20000;
Lposts = zeros(nchains,maxIter);
rates = zeros(nchains,1);

tic;
parfor chain_count = 1:nchains

    % Initialize chains at different random values
    kap2Init = 1./gamrnd(1,1);
    upsInit = 1./(kap2Init.*gamrnd(0.1, 10));
    xInit = gamrnd(0.1,3).*ones(size(x));

    % Sample and compute acceptance ratios
    [XDraws, Kappa2Draws, UpsDraws, Lpost, rate] = metwithingibbsPropJeff(b, ...
        A, L, V(:,1:R), l(1:R), kap2Init, upsInit, xInit, maxIter);
    
    parXDraws{chain_count} = XDraws;
    parKappa2Draws{chain_count} = Kappa2Draws;
    parUpsDraws{chain_count} = UpsDraws;
    Lposts(chain_count,:) = Lpost;
    rates(chain_count) = rate;

end
LRIStime1 = toc;
fprintf('Time elapsed, first 20k samples: %f sec \n', LRIStime1);
poolobj = gcp('nocreate');
delete(poolobj);

% Optional: save output
%save('CT-samples1.mat','parXDraws','parKappa2Draws','parUpsDraws','Lposts','rates','LRIStime1','-v7.3');

fprintf('Thinning the chains \n\n');

% Thin the chains with given spacing
space = 50;
thinXDraws1 = thinning(parXDraws,space);
thinKappa2Draws1 = thinning(parKappa2Draws,space);
thinUpsDraws1 = thinning(parUpsDraws,space);

% Save thinned chains
save('CT-samples1-thin.mat','thinXDraws1','thinKappa2Draws1','thinUpsDraws1','LRIStime1');


%% Finish sampling w/ proper Jeffreys

fprintf('Sampling (LRIS w/Jeffreys priors), last 20k \n\n');

% Optional: load output and setup parameters
%load('CT-setup.mat');
%load('CT-samples1.mat','parXDraws','parKappa2Draws','parUpsDraws');

R = 4900;
nchains = 3;

% Initialize at the last sample per chain
Kappa2Init = zeros(nchains,1);
UpsInit = zeros(nchains,1);
XInit = zeros(length(x),nchains);
for i = 1:nchains
   Kappa2Init(i) = parKappa2Draws{i}(end);
   UpsInit(i) = parUpsDraws{i}(end);
   XInit(:,i) = parXDraws{i}(:,end);
end

% Optional: clear first 20k samples
%clear parXDraws parKappa2Draws parUpsDraws

parpool(nchains)
parXDraws = cell(nchains,1);
parKappa2Draws = cell(nchains,1);
parUpsDraws = cell(nchains,1);
maxIter = 20000;
Lposts = zeros(nchains,maxIter);
rates = zeros(nchains,1);

tic;
parfor chain_count = 1:nchains

    % Initialize chains at saved states
    kap2Init = Kappa2Init(chain_count);
    upsInit = UpsInit(chain_count);
    xInit = XInit(:,chain_count);

    % Sample and compute acceptance ratios
    [XDraws, Kappa2Draws, UpsDraws, Lpost, rate] = metwithingibbsPropJeff(b, ...
        A, L, V(:,1:R), l(1:R), kap2Init, upsInit, xInit, maxIter);
    
    parXDraws{chain_count} = XDraws;
    parKappa2Draws{chain_count} = Kappa2Draws;
    parUpsDraws{chain_count} = UpsDraws;
    Lposts(chain_count,:) = Lpost;
    rates(chain_count) = rate;

end
LRIStime2 = toc;
fprintf('Time elapsed, last 20k samples: %f sec \n', LRIStime2);
poolobj = gcp('nocreate');
delete(poolobj);

% Optional: save output
%save('CT-samples2.mat','parXDraws','parKappa2Draws','parUpsDraws','Lposts','rates','LRIStime2','-v7.3');

fprintf('Thinning the chains \n\n');

% Thin the chains with given spacing
space = 50;
thinXDraws2 = thinning(parXDraws,space);
thinKappa2Draws2 = thinning(parKappa2Draws,space);
thinUpsDraws2 = thinning(parUpsDraws,space);

% Save thinned chains
save('CT-samples2-thin.mat','thinXDraws2','thinKappa2Draws2','thinUpsDraws2','LRIStime2');


%% Convergence diagnostics (thinned chains)
% Computed using the last 50% of samples in the chains
% See Supplementary Materials for discussion

fprintf('Convergence diagnostics (thinned LRIS w/Jeffreys priors) \n\n');

% Optional: load true image
%load('CT-setup.mat','x');

% Load thinned chains and combine
load('CT-samples1-thin.mat'); load('CT-samples2-thin.mat');
[nchains,~] = size(thinXDraws1);
thinXDraws = cell(nchains,1);
thinKappa2Draws = cell(nchains,1);
thinUpsDraws = cell(nchains,1);
for i=1:nchains
    thinXDraws{i} = [thinXDraws1{i} thinXDraws2{i}];
    thinKappa2Draws{i} = [thinKappa2Draws1{i} thinKappa2Draws2{i}];
    thinUpsDraws{i} = [thinUpsDraws1{i} thinUpsDraws2{i}];
end

% Total computation time
fprintf('Time elapsed: %f sec \n', LRIStime1+LRIStime2);

% PSRFs
R_kap2 = computePSRF(thinKappa2Draws);
fprintf('For kappa^2: R_hat = %f \n', R_kap2);
R_ups = computePSRF(thinUpsDraws);
fprintf('For upsilon: R_hat = %f \n', R_ups);
% R_x = computeMVPSRF(thinXDraws);
% fprintf('For x: MvR_hat = %f \n', R_x); %  W,B are nearly singular

maxIter = length(thinKappa2Draws{1});
burnin = maxIter/2;
samples = burnin+1:maxIter;

% Trace plots of kappa^2, upsilon, and a randomly chosen pixel of x
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinKappa2Draws{1}(samples)); hold on;
plot(thinKappa2Draws{2}(samples),'k-');
plot(thinKappa2Draws{3}(samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\kappa^2}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinUpsDraws{1}(samples)); hold on;
plot(thinUpsDraws{2}(samples),'k-');
plot(thinUpsDraws{3}(samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\upsilon}$','fontsize',18,'interpreter','latex');
axis tight;

n = length(x);
pixel = randi(n);
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinXDraws{1}(pixel,samples)); hold on;
plot(thinXDraws{2}(pixel,samples),'k-');
plot(thinXDraws{3}(pixel,samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled pixel} $\mathbf{x(p)}$','fontsize',18,'interpreter','latex');
axis tight;

% Autocorrelation plots
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(thinKappa2Draws{1}(samples),50);
title('$\mathbf{\kappa^2}$','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(thinUpsDraws{1}(samples),50);
title('$\mathbf{\upsilon}$','fontsize',18,'interpreter','latex');

% Cumulative averages for kappa2 and upsilon
Kappa2CA = computeCA(thinKappa2Draws);
UpsCA = computeCA(thinUpsDraws);

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(Kappa2CA{1},'b-','LineWidth',2); hold on
plot(Kappa2CA{2},'k-','LineWidth',2);
plot(Kappa2CA{3},'r-','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\kappa^2}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(UpsCA{1},'b-','LineWidth',2); hold on
plot(UpsCA{2},'k-','LineWidth',2);
plot(UpsCA{3},'r-','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\upsilon}$','fontsize',18,'interpreter','latex');
axis tight;

figure; % Cumulative average for tau^2
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(Kappa2CA{1}.*UpsCA{1},'b-','LineWidth',2); hold on
plot(Kappa2CA{2}.*UpsCA{2},'k-','LineWidth',2);
plot(Kappa2CA{3}.*UpsCA{3},'r-','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\tau^2}$','fontsize',18,'interpreter','latex');
axis tight;

% Combine draws
XDraws = [thinXDraws{1}(:,samples) thinXDraws{2}(:,samples) thinXDraws{3}(:,samples)];
Kappa2Draws = [thinKappa2Draws{1}(samples) thinKappa2Draws{2}(samples) thinKappa2Draws{3}(samples)];
UpsDraws = [thinUpsDraws{1}(samples) thinUpsDraws{2}(samples) thinUpsDraws{3}(samples)];

% Plot the posterior mean
estImg = mean(XDraws,2);
N = sqrt(length(x));
figure;
imshow(reshape(estImg,[N,N]))

% Relative error
relerror = sqrt(sum((estImg-x).^2)/sum(x.^2));
fprintf('Relative error between true and posterior mean: %f \n', relerror);

% Plot estimated joint density for (kappa2, upsilon)
c = [ones(1,length(samples)) 5*ones(1,length(samples)) 10*ones(1,length(samples))];
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
scatterhist(Kappa2Draws,UpsDraws,'Group',c,'Kernel','on','Legend','off','Marker','+do','MarkerSize',3);
xlabel('$\mathbf{\kappa^2}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\upsilon}$','fontsize',18,'interpreter','latex');
axis tight;

% Transform (kappa2, upsilon) to (mu, sigma) and plot again
MuDraws = 1./Kappa2Draws;
SigmaDraws = MuDraws./UpsDraws;
figure; 
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
scatterhist(MuDraws,SigmaDraws,'Group',c,'Kernel','on','Legend','off','Marker','+do','MarkerSize',3);
xlabel('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;


%% Compare with LRIS using Gamma priors
% See Supplementary Materials for discussion

fprintf('Sampling (LRIS w/Gamma priors), first 20k \n\n');

% Optional: load setup parameters
%load('CT-setup.mat');

R = 4900;
maxIter = 20000;

% Specify hyperparameters
am = 1.e-1;
bm = 1.e-1;
aSig = 1.e-1;
bSig = 1.e-1;

% Initialize chains at different values
muInit = gamrnd(0.1,10,3,1);
sigInit = gamrnd(0.1,10,3,1);
xInit = [ones(length(x),1), 0.5*ones(length(x),1), 0.75*ones(length(x),1)]; 

nchains = 3;
parpool(nchains)
parXDraws = cell(nchains,1);
parMuDraws = cell(nchains,1);
parSigDraws = cell(nchains,1);
rats = zeros(nchains,maxIter);
rates = zeros(nchains,1);

tic;
parfor chain_count = 1:nchains

    % Initialization
    mu1 = muInit(chain_count);
    sig1 = sigInit(chain_count);
    x1 = xInit(:,chain_count);

    % Sample and compute acceptance ratios
    [XDraws, MuDraws, Sigma2Draws, ratio, rate] = metwithingibbsLRIS(b, ...
        A, L, V(:,1:R), l(1:R), am, bm, aSig, bSig, mu1, sig1, x1, maxIter);
    
    parXDraws{chain_count} = XDraws;
    parMuDraws{chain_count} = MuDraws;
    parSigDraws{chain_count} = Sigma2Draws;
    rats(chain_count,:) = ratio;
    rates(chain_count) = rate;

end
LRIStime3 = toc;
fprintf('Time elapsed, first 20k samples: %f sec \n', LRIStime3);
poolobj = gcp('nocreate');
delete(poolobj);

% Optional: save output
%save('CT-samples3.mat','parXDraws','parMuDraws','parSigDraws','rats','rates','LRIStime3','-v7.3');

fprintf('Thinning the chains \n\n');

% Thin the chains with given spacing
space = 50;
thinXDraws3 = thinning(parXDraws,space);
thinMuDraws3 = thinning(parMuDraws,space);
thinSigDraws3 = thinning(parSigDraws,space);

% Save thinned chains
save('CT-samples3-thin.mat','thinXDraws3','thinMuDraws3','thinSigDraws3','LRIStime3');


%% Finish sampling w/ Gamma priors

fprintf('Sampling (LRIS w/Gamma priors), last 20k \n\n');

% Optional: load output and setup parameters
%load('CT-setup.mat');
%load('CT-samples3.mat','parXDraws','parMuDraws','parSigDraws');

R = 4900;
nchains = 3;

% Initialize at the last sample per chain
muInit = zeros(nchains,1);
sigInit = zeros(nchains,1);
xInit = zeros(length(x),nchains);
for i = 1:nchains
   muInit(i) = parMuDraws{i}(end);
   sigInit(i) = parSigDraws{i}(end);
   xInit(:,i) = parXDraws{i}(:,end);
end

% Optional: clear first 20k samples
%clear parXDraws parMuDraws parSigDraws

parpool(nchains)
parXDraws = cell(nchains,1);
parKappa2Draws = cell(nchains,1);
parUpsDraws = cell(nchains,1);
maxIter = 20000;
rats = zeros(nchains,maxIter);
rates = zeros(nchains,1);

tic;
parfor chain_count = 1:nchains

    % Initialize chains at saved states
    mu1 = muInit(chain_count);
    sig1 = sigInit(chain_count);
    x1 = xInit(:,chain_count);

    % Sample and compute acceptance ratios
    [XDraws, MuDraws, Sigma2Draws, ratio, rate] = metwithingibbsLRIS(b, ...
        A, L, V(:,1:R), l(1:R), am, bm, aSig, bSig, mu1, sig1, x1, maxIter);
    
    parXDraws{chain_count} = XDraws;
    parMuDraws{chain_count} = MuDraws;
    parSigDraws{chain_count} = Sigma2Draws;
    rats(chain_count,:) = ratio;
    rates(chain_count) = rate;

end
LRIStime4 = toc;
fprintf('Time elapsed, last 20k samples: %f sec \n', LRIStime4);
poolobj = gcp('nocreate');
delete(poolobj);

% Optional: save output
%save('CT-samples4.mat','parXDraws','parMuDraws','parSigDraws','rats','rates','LRIStime4','-v7.3');

fprintf('Thinning the chains \n\n');

% Thin the chains with given spacing
space = 50;
thinXDraws4 = thinning(parXDraws,space);
thinMuDraws4 = thinning(parMuDraws,space);
thinSigDraws4 = thinning(parSigDraws,space);

% Save thinned chains
save('CT-samples4-thin.mat','thinXDraws4','thinMuDraws4','thinSigDraws4','LRIStime4');


%% Convergence diagnostics (thinned chains)
% Computed using the last 50% of samples in the chains

fprintf('Convergence diagnostics (thinned LRIS w/Gamma priors) \n\n');

% Optional: load true image
%load('CT-setup.mat','x');

% Load thinned chains and combine
load('CT-samples3-thin.mat'); load('CT-samples4-thin.mat');
[nchains,~] = size(thinXDraws3);
thinXDraws = cell(nchains,1);
thinMuDraws = cell(nchains,1);
thinSigDraws = cell(nchains,1);
for i=1:nchains
    thinXDraws{i} = [thinXDraws3{i} thinXDraws4{i}];
    thinMuDraws{i} = [thinMuDraws3{i} thinMuDraws4{i}];
    thinSigDraws{i} = [thinSigDraws3{i} thinSigDraws4{i}];
end

% Total computation time
fprintf('Time elapsed: %f sec \n', LRIStime3+LRIStime4);

% PSRFs
R_mu = computePSRF(thinMuDraws);
fprintf('For mu: R_hat = %f \n', R_mu);
R_sig = computePSRF(thinSigDraws);
fprintf('For sigma: R_hat = %f \n', R_sig);
% R_x = computeMVPSRF(thinXDraws); 
% fprintf('For x: MvR_hat = %f \n', R_x);


maxIter = length(thinSigDraws{1});
burnin = maxIter/2;
samples = burnin+1:maxIter;

% Trace plots of sigma, mu, and a randomly chosen pixel of x
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinSigDraws{1}(samples)); hold on;
plot(thinSigDraws{2}(samples),'k-');
plot(thinSigDraws{3}(samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinMuDraws{1}(samples)); hold on;
plot(thinMuDraws{2}(samples),'k-');
plot(thinMuDraws{3}(samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled} $\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;

n = length(x);
pixel = randi(n);
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(thinXDraws{1}(pixel,samples)); hold on;
plot(thinXDraws{2}(pixel,samples),'k-');
plot(thinXDraws{3}(pixel,samples),'r-');
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Sampled pixel} $\mathbf{x(p)}$','fontsize',18,'interpreter','latex');
axis tight;

% Autocorrelation plots
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(thinSigDraws{1}(samples),50);
title('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
autocorr(thinMuDraws{1}(samples),50);
title('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');

% Cumulative averages 
SigmaCA = computeCA(thinSigDraws);
MuCA = computeCA(thinMuDraws);

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(SigmaCA{1},'LineWidth',2); hold on
plot(SigmaCA{2},'k-','LineWidth',2);
plot(SigmaCA{3},'r-','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
plot(MuCA{1},'LineWidth',2); hold on
plot(MuCA{2},'k-','LineWidth',2);
plot(MuCA{3},'r-','LineWidth',2);
xlabel('\textbf{Iteration}','fontsize',18,'interpreter','latex');
ylabel('\textbf{Cumulative average}','fontsize',18,'interpreter','latex');
title('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
axis tight;

% Combine draws
XDraws = [thinXDraws{1}(:,samples) thinXDraws{2}(:,samples) thinXDraws{3}(:,samples)];
SigmaDraws = [thinSigDraws{1}(samples) thinSigDraws{2}(samples) thinSigDraws{3}(samples)];
MuDraws = [thinMuDraws{1}(samples) thinMuDraws{2}(samples) thinMuDraws{3}(samples)];

% Plot the posterior mean
estImg = mean(XDraws,2);
N = sqrt(length(x));
figure;
imshow(reshape(estImg,[N,N]))

% Relative error
relerror = sqrt(sum((estImg-x).^2)/sum(x.^2));
fprintf('Relative error between true and posterior mean: %f \n', relerror);

% Plot estimated joint density for (mu, sigma)
c = [ones(1,length(samples)) 5*ones(1,length(samples)) 10*ones(1,length(samples))];
figure;
axes('fontsize',18);
set(gcf,'defaultaxesfontsize',16);
set(gcf,'defaultaxesfontweight','bold');
scatterhist(MuDraws,SigmaDraws,'Group',c,'Kernel','on','Legend','off','Marker','+do','MarkerSize',3);
xlabel('$\mathbf{\mu}$','fontsize',18,'interpreter','latex');
ylabel('$\mathbf{\sigma}$','fontsize',18,'interpreter','latex');
axis tight;

