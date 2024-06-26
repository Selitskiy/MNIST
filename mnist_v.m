%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/MNIST/');

% MNIST
load('mnist.mat');

n = training.height;
m = training.width;
l = training.count;
lts = test.count;

XTest = test.images;
YTest = test.labels;

XTest3C = zeros([n, m, 3, lts]);
XTest3C(:, :, 1, :) = XTest;
XTest3C(:, :, 2, :) = XTest;
XTest3C(:, :, 3, :) = XTest;
YTest3C = categorical(YTest);




XTestF = reshape(XTest,[n*m,lts]);

%% Continous learning
k=1; %6;
sub_len = l/k;

% sort digits
%if k > 1
    [sTLabels, IsTLabels] = sort(training.labels);
    sTImages = training.images(:,:,IsTLabels);

    %training.labels = sTLabels;
    %training.images = sTImages;
%end

i=1;
%%

for i = 1:k

XTrain = sTImages(:,:,(i-1)*sub_len+1:i*sub_len);
YTrain = sTLabels((i-1)*sub_len+1:i*sub_len);



%% random train 
%XTrainF = reshape(XTrain,[n*m,sub_len]);
%l_ub = 54051;

%i = l_ub + 1 + floor(rand()*5950);

%colormap(gray)

%If = XTrainF(:,i);
%I2 = reshape(If, [n, m]);
%image(I2 .* 255);

%title(string(i));

%%
%l_ub = l;
l_ub = 54051;

XTrain3C = zeros([n, m, 3, l_ub]);
XTrain3C(:, :, 1, :) = XTrain(:,:,1:l_ub);
XTrain3C(:, :, 2, :) = XTrain(:,:,1:l_ub);
XTrain3C(:, :, 3, :) = XTrain(:,:,1:l_ub);
YTrain3C = categorical(YTrain(1:l_ub));

%rebalance 9
l_inj = 6000;
l_bal = 4;
l_bal_m = floor(l_inj/l_bal);
sel9 = [56578, 57338, 54970, 58215];

%XTrainBal = XTrain(:,:,l_ub+1:l_ub+l_bal);
XTrainBal = XTrain(:,:,sel9);
XTrainBalMult = repmat(XTrainBal, [1 1 l_bal_m]);

%YTrainBal = YTrain(l_ub+1:l_ub+l_bal);
YTrainBal = YTrain(sel9);
YTrainBalMult = repmat(YTrainBal, [l_bal_m 1]);

XTrain = XTrain(:,:,1:l_ub);
YTrain = YTrain(1:l_ub);
XTrain(:,:,l_ub+1:l_ub+l_bal*l_bal_m) = XTrainBalMult;
YTrain(l_ub+1:l_ub+l_bal*l_bal_m) = YTrainBalMult;
sub_len = l_ub+l_bal*l_bal_m;


XTrainF = reshape(XTrain,[n*m,sub_len]);

%%
XTrain3C(:, :, 1, l_ub+1:l_ub+l_inj) = XTrainBalMult;
XTrain3C(:, :, 2, l_ub+1:l_ub+l_inj) = XTrainBalMult;
XTrain3C(:, :, 3, l_ub+1:l_ub+l_inj) = XTrainBalMult;
YTrain3C(l_ub+1:l_ub+l_inj) = categorical(YTrainBalMult);



%% Fltten labels
YTrainD = double(YTrain)';
YTestD = double(YTest)';

% Injection one-hot
inj = 10;
YTrainF = zeros(inj,sub_len);
YTestF = zeros(inj,lts);

for j = 1:inj
    YTrainF(j, YTrainD==(j-1)) = 1;
    YTestF(j, YTestD==(j-1)) = 1;
end


% real injection
%XYTrainF = vertcat(XTrainF,YTrainD);
%XYTestF = vertcat(XTestF,YTestD);

% one-hot injection
XYTrainF = vertcat(XTrainF,YTrainF);
XYTestF = vertcat(XTestF,YTestF);


%%
x_off=0;
x_in=n*m+inj;
t_in=1;

y_off=0;
%y_out=1;
y_out=n*m;
t_out=t_in;

ini_rate = 0.0002; 
max_epoch = 100;

modelName = 'mnist_resvis3x3ae';
if k == 1
    modelFile = strcat(modelName, '.mat');
else
    modelFile = strcat(modelName, '.', string(i), '.mat');
end

%%
%if isfile(modelFile)
%    fprintf('Loading %s %d\n', modelFile, i);
%    load(modelFile, 'regNet');
%else
    if i == 1

        %regNet = vis3x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3, 1/9, 9, 9 ,10);
        regNet = residualVis3x3BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/3, 1/3, 1/3, 1/3, 9, 9, 10);

        %%
        regNet.mb_size = 2048;

        regNet = Create(regNet);

    end

    fprintf('Training %s %d\n', modelFile, i);
%%
%i=1;
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
    
        regNet = regNet.Train(i, XYTrainF, XTrainF);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

    %%
    fprintf('Saving %s %d\n', modelFile, i);
    save(modelFile, 'regNet');

% end no file - train
%end

% end of contionous learning
end

%% LrReLU weights
%histogram(regNet.lGraph.Layers(6,1).A) %28
%histogram(regNet.lGraph.Layers(8,1).A,'BinLimits',[0.45,1], Normalization="percentage") %43905
%ytickformat("percentage") 

%% activations
        % GPU on
%        gpuDevice(1);
%        reset(gpuDevice(1));
        
%act1 = activations(regNet.trainedNet, XYTestF(:,:)', 'b_k_hid1');
%ma = max(act1,[],'all');
%mi = min(act1,[],'all');
%actn = (act1 - mi)/(ma - mi);

        % GPU off
%        delete(gcp('nocreate'));
%        gpuDevice([]); 

%%
%i = 1 + floor(rand()*l);

%subplot(2,2,1);
%If = XTestF(:,i);
%I2 = reshape(If, [n, m]);

%image(I2 .* 255);

%subplot(2,2,2);
%Ifp = actn(:,i);
%I2p = reshape(Ifp, [n, m]);

%image(I2p .* 255);
%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTestF');
XTestF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% Generate
XGenF = zeros([n*m, 10]);
%XGenF = rand([n*m, 10]);
%XYGenF = ones([n*m, 10]);

YGenF = zeros([inj, 10]);

% real inject
%for i=1:10
%    XYGenF(x_in, i) = i;
%end

% one hoot inject
for i = 1:inj
    YGenF(i, i) = 1;
end

XYGenF = vertcat(XGenF,YGenF);

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYGenF');
XGenF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% Display Gen
colormap(hot)
%colorbar

for i=1:10
    subplot(3,4,i);
    If = XGenF2(:,i);
    I2 = reshape(If, [n, m]);
    
    image(I2 .* 255);
    title(string(i-1));
end

%% random results
i = 1 + floor(rand()*lts);
%i = l_ub + 1 + floor(rand()*6000);

colormap(gray)
%colorbar

subplot(2,2,1);
If = XTestF(:,i);
I2 = reshape(If, [n, m]);

image(I2 .* 255);


%subplot(2,2,2);
%%Ifp = XTestF2(:,i);
%Ifp = actn(1:regNet.m_in-inj,i);
%I2p = reshape(Ifp, [n, m]);

%image(I2p .* 255);


%subplot(2,2,3);
%I2m = I2 .* I2p;

%ma = max(I2m,[],'all');
%mi = min(I2m,[],'all');
%I2n = (I2m - mi)/(ma - mi);

%image(I2n .* 255);


subplot(2,2,2);
Ift = XTestF2(:,i);
I2t = reshape(Ift, [n, m]);

image(I2t .* 255);

%XTestF2(1,i)

%% difficult results
idx = [2997 252 3697 9861 276 9686 5990 3062 6493 5895 5281 9201 6994 9813 4599 5625 342 2845 5968 6659];
[~,ni] = size(idx);

colormap(gray)

for i = 1:ni
    subplot(4,10,i);
    If = XTestF(:,idx(i));
    I2 = reshape(If, [n, m]);

    image(I2 .* 255);   
end

for i = 1:ni
    subplot(4,10,ni+i);
    Ift = XTestF2(:,idx(i));
    I2t = reshape(Ift, [n, m]);

    image(I2t .* 255);
end

%% mutated results
d = 1; %1;
XTestM = XTestF(:,idx(d*inj+1:(d+1)*inj));
XTestMM = repmat(XTestM', inj+1, 1)';

YTestM = zeros([10, 10*(inj+1)]);


for i = 1:10 %mutatuon
    for j = 1:inj %seed
        YTestM(i,(i-1)*10+j) = 1;
    end
end

XYTestM = vertcat(XTestMM,YTestM);

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYTestM');
XTestM2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% mutated display
colormap(gray)

for i = 0:10 %mutation
    for j = 1:inj %seed
        subplot(12,10,i*10+j);

        if i==0
            Im = XTestF(:,idx(d*inj+j));
        else
            Im = XTestM2(:,(i-1)*10+j);
        end

        I2m = reshape(Im, [n, m]);

        image(I2m .* 255);

        if i == 0
            title(strcat('m:-',' s:',string(j-1)));
        elseif i == 11
            title(strcat('m:0',' s:',string(j-1)));
        else
            title(strcat('m:',string(i-1),' s:',string(j-1)));
        end

    end
end

sotaFl=0;
if sotaFl


%% SOTA comparison

%% Generate unbalanced 9
%l_inj = 6000;
l_gen9 = int64(floor(rand([1 l_inj])*l_ub));
XGenF = XTrainF(:, l_gen9 );

%XGenF = XTrainF(:, l_ub+1:l_ub+l_bal*l_bal_m);

YGenF = zeros([inj, l_inj]);
YGen3C = categorical( ones([l_inj, 1]) * 9 );

% fuzzy one hoot inject
%for i = 1:l_inj
%    fcalss = floor(rand()*10);
%    fval = rand()*2;
%    YGenF(fcalss+1, i) = fval;
%end
YGenF(9+1, :) = 2;

XYGenF = vertcat(XGenF,YGenF);

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(regNet.trainedNet, XYGenF');
XGenF2 = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%%
XTrain3C(:, :, 1, l_ub+1:l_ub+l_inj) = reshape(XGenF2,[n,m,l_inj]);
XTrain3C(:, :, 2, l_ub+1:l_ub+l_inj) = reshape(XGenF2,[n,m,l_inj]);
XTrain3C(:, :, 3, l_ub+1:l_ub+l_inj) = reshape(XGenF2,[n,m,l_inj]);
YTrain3C(l_ub+1:l_ub+l_inj) = YGen3C;


%% random results
i = 1 + floor(rand()*l_inj);

colormap(gray)
%colorbar

subplot(2,2,1);
If = XGenF(:,i);
I2 = reshape(If, [n, m]);

image(I2 .* 255);


subplot(2,2,2);
Ift = XGenF2(:,i);
I2t = reshape(Ift, [n, m]);

image(I2t .* 255);

%% Load Pre-trained Network (AlexNet)
% AlexNet is a pre-trained network trained on 1000 object categories. 
alex = alexnet('Weights','none');
layers = alex;

% Review Network Architecture 
%layers = alex.Layer;

% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just nClasses classes. 
n_ll = 25;
n_sml = n_ll - 2;
layers(1) = imageInputLayer([28, 28, 3]);
layers(9) = maxPooling2dLayer(2, 'Name', 'pool2');
layers(16) = maxPooling2dLayer(1, 'Name', 'pool5');
layers(n_sml) = fullyConnectedLayer(inj); % change this based on # of classes
layers(n_ll) = classificationLayer;

% Perform Transfer Learning
% For transfer learning we want to change the weights of the network ever so slightly. How
% much a network is changed during training is controlled by the learning
% rates. 

max_epoch = 150;
mb_size = 2048;
opts = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', mb_size, ...
                'InitialLearnRate', ini_rate, ...
                'MaxEpochs', max_epoch);

                        
                      %'Plots', 'training-progress',...

%% Train the Network  

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));

myNet = trainNetwork(XTrain3C, YTrain3C, layers, opts);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

predictedScores = predict(myNet, XTest3C);
X2Test3C = predictedScores';

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%%
[M, I]=max(X2Test3C,[],1);
acc = sum((I-1) == YTestD)/lts     

end