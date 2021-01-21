clear all;
close all; clc;

dTypes={'train'; 'test'};
D.train.N=1e3;
D.test.N=10e3;

%Generate Data
figure;
for ind=1:length(dTypes)
[D.(dTypes{ind}).x,D.(dTypes{ind}).y] =... 
    exam4q1_generateData(D.(dTypes{ind}).N);
subplot(2,1,ind); 
plot(D.(dTypes{ind}).x,D.(dTypes{ind}).y,'.'); 
xlabel('x');ylabel('y'); grid on;
xlim([0 30]); ylim([0 15]); title([dTypes{ind}]);
end

%Train and Validate Data
numPerc=15;
k=10;  

%kfold validation is in this function
[D.train.net,D.train.MSEtrain,optM,stats]=... 
    kfoldMLP_Fit(numPerc,k,D.train.x, D.train.y);
%Produce validation data from test dataset
yVal=D.train.net(D.test.x);
%Calculate MSE
MSEval=mean((yVal-D.test.y).^2);

%Plot number of perceptrons vs. pFE for the cross validation runs
for ind=1:length(dTypes)-1
    stem(stats.M,stats.avgMSE);
    xlabel('Number of Perceptrons');
    ylabel('Mean Square Error');
    title('Aveage of MSE vs. Number of Perceptrons');
end

%Print and plot results
fprintf('MSE=%1.2f%\n',MSEval);
figure;
plot(D.test.x,D.test.y,'o','DisplayName','Test Data'); hold all;
plot(D.test.x,yVal,'.','DisplayName','Estimated Data'); 
xlabel('x');ylabel('y');grid on;
title('Actual and Estimated Data');
legend 'show';


function [outputNet,outputMSE, optM, stats]=kfoldMLP_Fit(numPerc,k,x,y)
N=length(x);
numValIters=10;
%Setup cross validation on training data
partSize=N/k;
partInd=[1:partSize:N length(x)];
%Perform cross validation to select number of perceptrons
for M=1:numPerc
    figure;
    for ind=1:k
         %Separate training and validation data
         index.val=partInd(ind):partInd(ind+1)-1; 
         index.train=setdiff(1:N,index.val);
          net=feedforwardnet(M);
          net=train(net,x(:,index.train),y(:,index.train));
          %Validate with remaining data
          yVal=net(x(:,index.val));
          MSE(ind)=mean((yVal-y(:,index.val)).^2);
          %Plot overlay of model and validation data
          subplot(5,2,ind); plot(x(:,index.val),y(:,index.val),'o');
          hold all;
          plot(x(:,index.val),yVal,'.')
          if ind ==1
              title([num2str(M) ' Perceptrons']);
          end
    end
    avgMSE(M)=mean(MSE);
    stats.M=1:M;
    stats.avgMSE=avgMSE;
    stats.mMSE(:,M)=MSE;
end
[~,optM]=min(avgMSE);  
    for ind=1:numValIters
    netName(ind)={['net' num2str(ind)]}; 
    finalnet.(netName{ind})=patternnet(optM);
    finalnet.(netName{ind})=train(net,x,y);
    yVal=finalnet.(netName{ind})(x);
    MSEFinal(ind)=mean((yVal-y).^2);
    end
[minMSE,outInd]=min(MSEFinal);
stats.finalMSE=MSEFinal;
outputMSE=minMSE; 
outputNet=finalnet.(netName{outInd});
end

