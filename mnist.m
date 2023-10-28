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

XTestF = reshape(XTest,[n*m,lts]);

%% Continous learning
k=6; %6;
sub_len = l/k;

for i = 1:k

XTrain = training.images(:,:,(i-1)*sub_len+1:i*sub_len);
YTrain = training.labels((i-1)*sub_len+1:i*sub_len);

XTrainF = reshape(XTrain,[n*m,sub_len]);


% Matlab's Digits
%[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
%[XTest,YTest,anglesTest] = digitTest4DArrayData;

%[n, m, c, l] = size(XTrain);
%[~, ~, ~, lts] = size(XTest);

%XTrainF = reshape(XTrain,[n*m,l,c]);
%XTestF = reshape(XTest,[n*m,l,c]);


% Fltten labels
YTrainD = double(YTrain)';
YTestD = double(YTest)';

% Injection one-hot
inj = 10;
YTrainF = zeros(inj,sub_len);
YTestF = zeros(inj,lts);

for j = 1:inj
    YTrainF(j, YTrainD==j) = 1;
    YTestF(j, YTestD==j) = 1;
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

modelName = 'mnist_ae';
if k == 1
    modelFile = strcat(modelName, '.mat');
else
    modelFile = strcat(modelName, '.', string(i), '.mat');
end

%%
if isfile(modelFile)
    fprintf('Loading %s %d\n', modelFile, i);
    load(modelFile, 'regNet');
else
    if i == 1

        regNet = Dp2BTransAEBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch, 1/n);

        %%
        % Matlab's Digits
        %regNet.mb_size = 128;
        % MNIST
        regNet.mb_size = 2048;

        regNet = Create(regNet);
    end

    fprintf('Training %s %d\n', modelFile, i);
%%
%i=1;
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
    
        %regNet = regNet.Train(i, XTrainF, YTrainF);
        regNet = regNet.Train(1, XYTrainF, XTrainF);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

    %%
    fprintf('Saving %s %d\n', modelFile, i);
    save(modelFile, 'regNet');

% end no file - train
end

% end of contionous learning
end

%% LrReLU weights
%histogram(regNet.lGraph.Layers(5,1).A) %28
histogram(regNet.lGraph.Layers(7,1).A,'BinLimits',[0.45,1], Normalization="percentage") %43905
ytickformat("percentage") 

%% activations
        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
        
act1 = activations(regNet.trainedNet, XYTestF(:,:)', 'b_k_hid1');
ma = max(act1,[],'all');
mi = min(act1,[],'all');
actn = (act1 - mi)/(ma - mi);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]); 

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
colormap(gray)
colorbar

for i=1:10
    subplot(3,4,i);
    If = XGenF2(:,i);
    I2 = reshape(If, [n, m]);
    
    image(I2 .* 255);
    title(string(i-1));
end

%% random results
i = 1 + floor(rand()*lts);

colormap(gray)
colorbar

subplot(2,2,1);
If = XTestF(:,i);
I2 = reshape(If, [n, m]);

image(I2 .* 255);


subplot(2,2,2);
%Ifp = XTestF2(:,i);
Ifp = actn(1:regNet.m_in-inj,i);
I2p = reshape(Ifp, [n, m]);

image(I2p .* 255);


subplot(2,2,3);
I2m = I2 .* I2p;

ma = max(I2m,[],'all');
mi = min(I2m,[],'all');
I2n = (I2m - mi)/(ma - mi);

image(I2n .* 255);


subplot(2,2,4);
Ift = XTestF2(:,i);
I2t = reshape(Ift, [n, m]);

image(I2t .* 255);

XTestF2(1,i)

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