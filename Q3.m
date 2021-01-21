if 1
filenames = {'3096_color.jpg';'42049_color.jpg'}; 
dTypes={'c3096' 'c42049'};
%Cross validation parameters
NumGMMtoCheck=10;   %Number of Params
k=10;   %Number of folds
end

for ind=1:length(filenames)
imdata = imread(filenames{ind}); 
figure(1); 
subplot(length(filenames),3,(ind-1)*3+1); 
imshow(imdata);

if 1
    [R,C,D]=size(imdata);
    N=R*C;
    imdata=double(imdata);
    rows=(1:R)'*ones(1,C); columns=ones(R,1)*(1:C); 
    featureData=[rows(:)';columns(:)'];
    for ind2 =1:D
        imdatad =imdata(:,:,ind2);
        featureData =[featureData; imdatad(:)'];
    end
    minf=min(featureData,[],2);
    maxf=max(featureData,[],2);
    ranges=maxf-minf;
    %Normalized data
    x=(featureData-minf)./ranges;
    %Assess for GMM with 2 Gaussians case
    GMM2=fitgmdist(x',2,'Replicates',10); 
    post2=posterior(GMM2,x')'; 
    decisions=post2(2,:)>post2(1,:);
end
    labelImage2=reshape(decisions,R,C); %Plot GMM=2 CASE
    subplot(length(filenames),3,(ind-1)*3+2); 
    imshow(uint8(labelImage2*255/2));
if 1
    N =length(x);
    numValIters =10;
    partSize=floor(N/k);
    partInd=[1:partSize:N length(x)];
    for NumGMMs=1:NumGMMtoCheck
           for NumKs=1:k
               index.val=partInd(NumKs):partInd(NumKs+1);
               index.train=setdiff(1:N,index.val);
%Create object with M perceptrons in hidden layer
               GMMk_loop=fitgmdist(x(:,index.train)',NumGMMs,... 
                   'Replicates',5);
% net.layers{1}.transferFcn = 'softplus';%didn't work
               if GMMk_loop.Converged 
                   probX(NumKs)=sum(log(pdf(GMMk_loop,x(:,index.val)')));
               else
               end
           end
           avgProb(ind,NumGMMs)=mean(probX); stats(ind).NumGMMs=1:NumGMMtoCheck; 
           stats(ind).avgProb=avgProb; stats(ind).mProb(:,NumGMMs)=probX;
           fprintf('File: %1.0f, NumGMM: %1.0f\n',ind,NumGMMs);
    end
end
[~,optNumGMM]=max(avgProb(ind,:));

GMMk=fitgmdist(x',optNumGMM,'Replicates',10); 
postk=posterior(GMMk,x')';
lossMatrix=ones(optNumGMM,optNumGMM)-eye(optNumGMM);
expectedRisks =lossMatrix*postk; 
[~,decisions] = min(expectedRisks,[],1);
%Plot segmented image
labelImageK=reshape(decisions-1,R,C);
subplot(length(filenames),3,(ind-1)*3+3); 
imshow(uint8(labelImageK*255/2));
end

figure(2);plot(avgProb(1,:),'o');
xlabel('Number of GMMs');ylabel('Likelihood');
title('Cross-Validation Result');




    

    
        