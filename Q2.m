if 1
    if 0
        clear all;
numL=2;
k=10;
D.train.N=1e3;
D.test.N=10e3;
[D.train.x,D.train.labels] = generateMultiringDataset(numL,D.train.N);
D.train.labels(D.train.labels==1)=-1; 
D.train.labels(D.train.labels==2)=1;
figure(1);title('Training Data');
[D.test.x,D.test.labels] = generateMultiringDataset(numL,D.test.N);
D.test.labels(D.test.labels==1)=-1;
D.test.labels(D.test.labels==2)=1;
figure(2);title('Test Data');
    end
%Cross Validation to select parameters
%Setup cross validation on training data
partSize=D.train.N/k;
partInd=[1:partSize:D.train.N length(D.train.x)+1];
%Hyperparameter search parameters
sigmaList=logspace(-1,1,40);
cList=logspace(1.5,1.5,40);

%Perform cross validation to select model parameters
avgPFE=zeros(length(cList),length(sigmaList));
for SigInd=1:length(sigmaList)
    for Cind=1:length(cList)
        for ind = 1:k
            index.val = partInd(ind):partInd(ind+1)-1;
            index.train = setdiff(1:D.train.N,index.val);
            SVMk = fitcsvm(D.train.x(index.train)',D.train.labels(index.train),...
                'BoxConstraint',cList(Cind),'KernelFunction',... 
                'gaussian','KernelScale',sigmaList(SigInd));
            decisions = SVMk.predict(D.train.x(index.val)')';
            indINCORRECT= D.train.labels(index.val).*decisions == -1;
            pFE=sum(indINCORRECT)/length(index.val);
        end
        %Determine average probability of error for a number of perceptrons
        avgPFE(Cind,SigInd)=mean(pFE);
        fprintf('Sigma %1.0f/%1.0f, C %1.0f/%1.0f\n',SigInd,length(sigmaList),Cind,length(cList));
    end
end
end

%Plot results
figure; contour(log10(cList),log10(sigmaList),1-avgPFE',20); 
xlabel('log_{10} C');ylabel('log_{10} sigma');
title('Gaussian-SVM Cross-Val Accuracy Estimate'); axis equal;
%Determine hyperparameter values that minimize prob. of error
%Determine hyperparameter values that minimize prob. of error
[~,indMINpFE] = min(avgPFE(:));
[indOptC, indOptSigma] = ind2sub(size(avgPFE),indMINpFE); cOpt= cList(indOptC);
sigmaOpt= sigmaList(indOptSigma);
%Train final model using entire training dataset
SVMopt = fitcsvm(D.train.x',D.train.labels','BoxConstraint',cOpt,... 
    'KernelFunction','gaussian','KernelScale',sigmaOpt);
%Evaluate performance on test dataset
decisionsOpt=SVMopt.predict(D.test.x')'; decisionsEval=decisionsOpt.*D.test.labels; 
dInc=decisionsEval==-1; dCorr=decisionsEval==1;
pFEopt=sum(dInc)/D.test.N;
fprintf('Probability of Error = %1.2f%%\n',pFEopt);


%Plot correct and incorrect decisions
plot(D.test.x(1,dCorr),D.test.x(2,dCorr),'go','DisplayName','Correct Decisions');
hold all; 
plot(D.test.x(1,dInc),D.test.x(2,dInc),'r.','DisplayName','Incorrect Decisions');
grid on;
xlabel('x1'); ylabel('x2');
title(sprintf('Classification Decisions\nProbability of Error= %1.2f%%',100*pFEopt));
Nx = 1001; Ny = 990;
xGrid = linspace(-10,10,Nx);
yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid);
dGrid = SVMopt.predict([h(:),v(:)]);
zGrid = reshape(dGrid,Ny,Nx);
figure(1), subplot(1,2,2);
contour(xGrid,yGrid,zGrid,0);
xlabel('x1'), ylabel('x2'), axis equal;

function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples)

C = numberOfClasses;
N = numberOfSamples;
% Generates N samples from C ring-shaped 
% class-conditional pdfs with equal priors

% Randomly determine class labels for each sample
thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
u = rand(1,N); % generate N samples uniformly random in [0,1]
labels = zeros(1,N);
for l = 1:C
    ind_l = find(thr(l)<u & u<=thr(l+1));
    labels(ind_l) = repmat(l,1,length(ind_l));
end

a = [1:C].^2.5; b = repmat(1.7,1,C); % parameters of the Gamma pdf needed later
% Generate data from appropriate rings
% radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
angle = 2*pi*rand(1,N);
radius = zeros(1,N); % reserve space
for l = 1:C
    ind_l = find(labels==l);
    radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
end

data = [radius.*cos(angle);radius.*sin(angle)];

if 1
    colors = rand(C,3);
    figure(1), clf,
    for l = 1:C
        ind_l = find(labels==l);
        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
    end
end
end

            
    